import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

try:
    from flash_attn.modules.mha import FlashCrossAttention
except ModuleNotFoundError:
    FlashCrossAttention = None

if FlashCrossAttention or hasattr(F, "scaled_dot_product_attention"):
    FLASH_AVAILABLE = True
else:
    FLASH_AVAILABLE = False

torch.backends.cudnn.deterministic = True


from copy import deepcopy
import logging



def MLP(channels, do_bn=True):
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n - 1):
            if do_bn:
                layers.append(nn.BatchNorm1d(channels[i]))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


@torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
def normalize_keypoints(kpts, size=None, shape=None):
    if size is None:
        assert shape is not None
        _, _, h, w = shape
        one = kpts.new_tensor(1)
        size = torch.stack([one * w, one * h])[None]

    shift = size.float().to(kpts) / 2
    scale = size.max(1).values.float().to(kpts) * 0.7
    kpts = (kpts - shift[:, None]) / scale[:, None, None]
    return kpts


class KeypointEncoder(nn.Module):
    def __init__(self, feature_dim, layers, use_scores=True):        
        super().__init__()
        self.use_scores = use_scores
        c = 3 if use_scores else 2
        self.encoder = MLP([c] + list(layers) + [feature_dim])
        nn.init.constant_(self.encoder[-1].bias, 0.0)

    def forward(self, kpts, scores):
        if self.use_scores:
            inputs = [kpts.transpose(1, 2), scores.unsqueeze(1)]
        else:
            inputs = [kpts.transpose(1, 2)]
        return self.encoder(torch.cat(inputs, dim=1))


def attention(query, key, value):
    dim = query.shape[1]
    scores = torch.einsum("bdhn,bdhm->bhnm", query, key) / dim**0.5
    prob = torch.nn.functional.softmax(scores, dim=-1)
    return torch.einsum("bhnm,bdhm->bdhn", prob, value), prob


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model):
        super().__init__()
        assert d_model % h == 0
        self.dim = d_model // h
        self.h = h
        self.merge = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])

    def forward(self, query, key, value):
        b = query.size(0)
        query, key, value = [
            layer(x).view(b, self.dim, self.h, -1)
            for layer, x in zip(self.proj, (query, key, value))
        ]
        x, _ = attention(query, key, value)
        return self.merge(x.contiguous().view(b, self.dim * self.h, -1))


class AttentionalPropagation(nn.Module):
    def __init__(self, num_dim, num_heads):
        super().__init__()
        self.attn = MultiHeadedAttention(num_heads, num_dim)
        self.mlp = MLP([num_dim * 2, num_dim * 2, num_dim])
        nn.init.constant_(self.mlp[-1].bias, 0.0)

    def forward(self, x, source):
        message = self.attn(x, source, source)
        return self.mlp(torch.cat([x, message], dim=1))


class AttentionalGNN(nn.Module):
    def __init__(self, feature_dim, layer_names):
        super().__init__()
        self.layers = nn.ModuleList(
            [AttentionalPropagation(feature_dim, 4) for _ in range(len(layer_names))]
        )
        self.names = layer_names

    def forward(self, desc0, desc1):
        for i, (layer, name) in enumerate(zip(self.layers, self.names)):
            layer.attn.prob = []
            if self.training:
                delta0, delta1 = checkpoint(
                    self._forward, layer, desc0, desc1, name, preserve_rng_state=False
                )
            else:
                delta0, delta1 = self._forward(layer, desc0, desc1, name)
            desc0, desc1 = (desc0 + delta0), (desc1 + delta1)
            del delta0, delta1
        return desc0, desc1

    def _forward(self, layer, desc0, desc1, name):
        if name == "self":
            return layer(desc0, desc0), layer(desc1, desc1)
        elif name == "cross":
            return layer(desc0, desc1), layer(desc1, desc0)
        else:
            raise ValueError(name)


def log_sinkhorn_iterations(Z, log_mu, log_nu, iters):
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
    return Z + u.unsqueeze(2) + v.unsqueeze(1)


def log_optimal_transport(scores, alpha, iters):
    b, m, n = scores.shape
    one = scores.new_tensor(1)
    ms, ns = (m * one).to(scores), (n * one).to(scores)

    bins0 = alpha.expand(b, m, 1)
    bins1 = alpha.expand(b, 1, n)
    alpha = alpha.expand(b, 1, 1)

    couplings = torch.cat(
        [torch.cat([scores, bins0], -1), torch.cat([bins1, alpha], -1)], 1
    )

    norm = -(ms + ns).log()
    log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
    log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])
    log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)

    Z = log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
    Z = Z - norm  # multiply probabilities by M+N
    return Z


def arange_like(x, dim):
    return x.new_ones(x.shape[dim]).cumsum(0) - 1  # traceable in 1.1


class SuperGlue(nn.Module):
    default_conf = {
        "descriptor_dim": 256,
        "weights": "outdoor",
        "keypoint_encoder": [32, 64, 128, 256],
        "GNN_layers": ["self", "cross"] * 9,
        "num_sinkhorn_iterations": 50,
        "filter_threshold": 0.2,
        "mp": False,  # enable mixed precision
        "use_scores": True,
        "loss": {
            "nll_balancing": 0.5,
        },
    }
    required_data_keys = ["image0", "image1"]

    url = "https://github.com/magicleap/SuperGluePretrainedNetwork/raw/master/models/weights/superglue_{}.pth"  # noqa: E501

    features = {
        "superpoint": {
            "weights": "outdoor",
            "input_dim": 256,
        },
        # "disk": {
        #     "weights": "outdoor",
        #     "input_dim": 128,
        # },
        # "aliked": {
        #     "weights": "outdoor",
        #     "input_dim": 128,
        # },
        # "sift": {
        #     "weights": "outdoor",
        #     "input_dim": 128,
        #     "add_scale_ori": True,
        # },
    }

    def __init__(self, features="superpoint", **conf):
        super().__init__()
        self.conf = conf = SimpleNamespace(**{**self.default_conf, **conf})
        if features is not None:
            if features not in self.features:
                raise ValueError(
                    f"Unsupported features: {features} not in "
                    f"{{{','.join(self.features)}}}"
                )
            for k, v in self.features[features].items():
                setattr(conf, k, v)

        self.kenc = KeypointEncoder(
            conf.descriptor_dim, conf.keypoint_encoder, conf.use_scores
        )

        self.gnn = AttentionalGNN(conf.descriptor_dim, conf.GNN_layers)

        self.final_proj = nn.Conv1d(
            conf.descriptor_dim, conf.descriptor_dim, kernel_size=1, bias=True
        )
        
        bin_score = torch.nn.Parameter(torch.tensor(1.0))
        self.register_parameter("bin_score", bin_score)

        if conf.weights:
            assert conf.weights in ["indoor", "outdoor"]
            url = self.url.format(conf.weights)
            self.load_state_dict(torch.hub.load_state_dict_from_url(url))
            logging.info(f"Loading SuperGlue trained for {conf.weights}.")

        # # TBD
        # if conf.input_dim != conf.descriptor_dim:
        #     self.input_proj = nn.Linear(conf.input_dim, conf.descriptor_dim, bias=True)               
        # else:
        #     self.input_proj = nn.Identity()



    def forward(self, data: dict) -> dict:
        """
        Match keypoints and descriptors between two images

        Input (dict):
            image0: dict
                keypoints: [B x M x 2]
                descriptors: [B x M x D]
                image: [B x C x H x W] or image_size: [B x 2]
            image1: dict
                keypoints: [B x N x 2]
                descriptors: [B x N x D]
                image: [B x C x H x W] or image_size: [B x 2]
        Output (dict):
            log_assignment: [B x M+1 x N+1]
            matches0: [B x M]
            matching_scores0: [B x M]
            matches1: [B x N]
            matching_scores1: [B x N]
            matches: List[[Si x 2]], scores: List[[Si]]
        """
        for key in self.required_data_keys:
            assert key in data, f"Missing key {key} in data"

        with torch.autocast(enabled=self.conf.mp, device_type="cuda"):
            return self._forward(data)
            
            

    def _forward(self, data: dict) -> dict:
        # # Original SuperGlue
        # desc0 = data["descriptors0"].transpose(-1, -2)
        # desc1 = data["descriptors1"].transpose(-1, -2)
        # kpts0, kpts1 = data["keypoints0"], data["keypoints1"]
        
        # Modified from LightGlue repo
        data0, data1 = data["image0"], data["image1"]
        kpts0, kpts1 = data0["keypoints"], data1["keypoints"]
        desc0 = data0["descriptors"].detach().contiguous().transpose(-1, -2)
        desc1 = data1["descriptors"].detach().contiguous().transpose(-1, -2)

        if kpts0.shape[1] == 0 or kpts1.shape[1] == 0:  # no keypoints
            shape0, shape1 = kpts0.shape[:-1], kpts1.shape[:-1]
            return {
                "matches0": kpts0.new_full(shape0, -1, dtype=torch.int),
                "matches1": kpts1.new_full(shape1, -1, dtype=torch.int),
                "matching_scores0": kpts0.new_zeros(shape0),
                "matching_scores1": kpts1.new_zeros(shape1),
            }

        ## Original SuperGlue
        # view0, view1 = data["image0"], data["image1"]
        # kpts0 = normalize_keypoints(
        #     kpts0, size=view0.get("image_size"), shape=view0["image"].shape
        # )
        # kpts1 = normalize_keypoints(
        #     kpts1, size=view1.get("image_size"), shape=view1["image"].shape
        # )

        # Modified from SuperGlue
        kpts0 = normalize_keypoints(
            kpts0, size=data0["image_size"], shape=None
        )
        kpts1 = normalize_keypoints(
            kpts1, size=data1["image_size"], shape=None
        )

        assert torch.all(kpts0 >= -1) and torch.all(kpts0 <= 1)
        assert torch.all(kpts1 >= -1) and torch.all(kpts1 <= 1)
        
        # # Original SuperGlue
        # desc0 = desc0 + self.kenc(kpts0, data["keypoint_scores0"])
        # desc1 = desc1 + self.kenc(kpts1, data["keypoint_scores1"])

        # Modified from SuperGlue
        desc0 = desc0 + self.kenc(kpts0, data0["keypoint_scores"])
        desc1 = desc1 + self.kenc(kpts1, data1["keypoint_scores"])        
        
        desc0, desc1 = self.gnn(desc0, desc1)

        mdesc0, mdesc1 = self.final_proj(desc0), self.final_proj(desc1)

        scores = torch.einsum("bdn,bdm->bnm", mdesc0, mdesc1)
        cost = scores / self.conf.descriptor_dim**0.5

        scores = log_optimal_transport(
            cost, self.bin_score, iters=self.conf.num_sinkhorn_iterations
        )

        max0, max1 = scores[:, :-1, :-1].max(2), scores[:, :-1, :-1].max(1)
        m0, m1 = max0.indices, max1.indices
        mutual0 = arange_like(m0, 1)[None] == m1.gather(1, m0)
        mutual1 = arange_like(m1, 1)[None] == m0.gather(1, m1)
        zero = scores.new_tensor(0)
        mscores0 = torch.where(mutual0, max0.values.exp(), zero)
        mscores1 = torch.where(mutual1, mscores0.gather(1, m1), zero)
        valid0 = mutual0 & (mscores0 > self.conf.filter_threshold)
        valid1 = mutual1 & valid0.gather(1, m1)
        m0 = torch.where(valid0, m0, m0.new_tensor(-1))
        m1 = torch.where(valid1, m1, m1.new_tensor(-1))


        b, n, _ = kpts1.shape
        matches, mscores = [], []
        for k in range(b):
            valid = m0[k] > -1
            m_indices_0 = torch.where(valid)[0]
            m_indices_1 = m0[k][valid]
            # if do_point_pruning:
            #     m_indices_0 = ind0[k, m_indices_0]
            #     m_indices_1 = ind1[k, m_indices_1]
            matches.append(torch.stack([m_indices_0, m_indices_1], -1))
            mscores.append(mscores0[k][valid])

        ## Original SuperGlue
        # return {
        #     "sinkhorn_cost": cost,
        #     "log_assignment": scores,
        #     "matches0": m0,
        #     "matches1": m1,
        #     "matching_scores0": mscores0,
        #     "matching_scores1": mscores1,
        # }
    
        # Modified from LightGlue
        pred = {
            "sinkhorn_cost": cost,
            "log_assignment": scores,
            "matches0": m0,
            "matches1": m1,
            "matches": matches,
            "matching_scores0": mscores0,
            "matching_scores1": mscores1,
        }

        return pred