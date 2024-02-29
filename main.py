# Import necessary dependecies
from pathlib import Path
from lightglue import LightGlue, SuperPoint, SuperGlue, LightGlue_custom
from lightglue.utils import load_image, rbd
from lightglue import viz2d
import torch
import os
import pandas as pd
import time
import numpy as np
import cv2 as cv

def check_fundamental_matrix(ransacThresh, ransacConf, kpts0, kpts1, m_kpts0, m_kpts1):
    if len(m_kpts0) < 8 or len(m_kpts1) < 8:        
        num_f_inlier = -1
        num_f_outlier = -1
    else:
        pts0_fmat = m_kpts0.int()                
        pts0_fmat = pts0_fmat.numpy(force=True)
        pts1_fmat = m_kpts1.int()
        pts1_fmat = pts1_fmat.numpy(force=True)
                        
        F, mask = cv.findFundamentalMat(pts0_fmat,pts1_fmat,cv.FM_RANSAC, ransacReprojThreshold=ransacThresh, confidence=ransacConf)
        
        if mask is None:
            kpts0_fmat = kpts0.int()
            kpts0_fmat = kpts0_fmat.numpy(force=True)
            kpts1_fmat = kpts1.int()
            kpts1_fmat = kpts1_fmat.numpy(force=True)
            pts0_outlier_fmat = kpts0_fmat
            pts1_outlier_fmat = kpts1_fmat
            pts0_fmat = []
            pts1_fmat = []

        else:
            pts0_outlier_fmat = np.int32(pts0_fmat[mask.ravel()==0])
            pts1_outlier_fmat = np.int32(pts1_fmat[mask.ravel()==0])
            pts0_fmat = pts0_fmat[mask.ravel()==1]
            pts1_fmat = pts1_fmat[mask.ravel()==1]

        num_f_inlier = len(pts0_fmat)
        num_f_outlier = len(pts0_outlier_fmat)
        
        return pts0_fmat, pts1_fmat, num_f_inlier, num_f_outlier        

def eval_matcher(setup, dir_output, image0, image1, fname0, fname1):
    t_start = time.time()
    
    # Load
    extractor = setup['extractor']
    matcher = setup['matcher_model']
    matcher_name = setup['matcher_name']
    device = setup['device']

    # Extract features
    feats0 = extractor.extract(image0.to(device))
    feats1 = extractor.extract(image1.to(device))

    # Match features with matcher
    matches01 = matcher({"image0": feats0, "image1": feats1})
    feats0, feats1, matches01 = [
        rbd(x) for x in [feats0, feats1, matches01]
    ]  # remove batch dimension

    # Identify matched keypoints
    kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
    m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]
    conf_threshold = matcher.default_conf["filter_threshold"] # Min confidence threshold of matcher
    valid0 = (matches01['matching_scores0'] >  conf_threshold)
    valid1 = (matches01['matching_scores1'] >  conf_threshold)
    matching_num0 = sum(valid0.long())
    matching_num1 = sum(valid1.long())
    mconf0 = matches01['matching_scores0'][valid0]
    mconf1 = matches01['matching_scores1'][valid1]

    # Ensure consistency of matches
    sum_mconf0 = sum(mconf0)
    sum_mconf1 = sum(mconf1)
    try:
        assert torch.round(sum_mconf0, decimals=3) == torch.round(sum_mconf1, decimals=3)
    except:
        print("0 points met confidence threshold!")
    assert matching_num0 == matching_num1

    # Calculate norm-score and match-prop
    num_kpts0 = len(kpts0)
    num_kpts1 = len(kpts1)
    matching_score = sum_mconf0 / matching_num0
    match_prop = matching_num0 / min(num_kpts0, num_kpts1)

    if setup["f_removal"] == True:
        ransacThresh = setup["ransacThresh"]
        ransacConf = setup["ransacConf"] 
        
        pts0_fmat, pts1_fmat, num_f_inlier, num_f_outlier = check_fundamental_matrix(ransacThresh, ransacConf, kpts0, kpts1, m_kpts0, m_kpts1)
        image2, image3 = image0, image1

        axes = viz2d.plot_images([image2, image3])
        viz2d.plot_matches(m_kpts0, m_kpts1, color="red", lw=0.3)
        
        if num_f_inlier > 0:
            viz2d.plot_matches(pts0_fmat, pts1_fmat, color="lime", lw=0.3)

        label_text_EG = [
                        matcher_name + " with " + list(matcher.features.keys())[0] + " and fundamental matrix outlier removal",
                        'Keypoints: {}:{}'.format(num_kpts0, num_kpts1),
                        'Matches: {}'.format(matching_num0),
                        'norm-score: {:.4f}'.format(matching_score),
                        'match-prop: {:.4f}'.format(match_prop), 
                        'matching-num: {:4f}'.format(matching_num0),
                        'conf-thresh: {:4f}'.format(conf_threshold),
                        'ransacThresh: {}'.format(setup["ransacThresh"]),
                        'ransacConf: {:4f}'.format(setup["ransacConf"]),               
                        'num_f_inlier: {:4f}'.format(num_f_inlier),
                        'num_f_outlier: {:4f}'.format(num_f_outlier)
                    ]

        text_pos = [0.01, 0.99]
        for labels in label_text_EG:
            viz2d.add_text(0, text=labels, pos=text_pos, fs=12)
            text_pos[1] = text_pos[1] - 0.05

        # Save image matching results
        if not os.path.exists(dir_output):
            os.makedirs(dir_output)
        fname1_eg = fname1 + "_EG"
        viz2d.save_plot(path = Path( dir_output / fname1_eg))       
        viz2d.close_all()

    else:
        num_f_inlier = "N.A"
        num_f_outlier = "N.A"        
        ransacThresh = "N.A"
        ransacConf = "N.A"                
        pass
    
    t_end = time.time()

    # Plot primary
    axes = viz2d.plot_images([image0, image1])
    viz2d.plot_matches(m_kpts0, m_kpts1, color="lime", lw=0.2)
    label_text = [
                    matcher_name + " with " + list(matcher.features.keys())[0],
                    'Keypoints: {}:{}'.format(num_kpts0, num_kpts1),
                    'Matches: {}'.format(matching_num0),
                    'norm-score: {:.4f}'.format(matching_score),
                    'match-prop: {:.4f}'.format(match_prop), 
                    'matching-num: {:4f}'.format(matching_num0),
                    'conf-thresh: {:4f}'.format(conf_threshold)
                ]

    text_pos = [0.01, 0.99]
    for labels in label_text:
        viz2d.add_text(0, text=labels, pos=text_pos, fs=12)
        text_pos[1] = text_pos[1] - 0.05

    # Save image matching results
    if not os.path.exists(dir_output):
        os.makedirs(dir_output)
    viz2d.save_plot(path = Path( dir_output / fname1))
    viz2d.close_all()

    t_elasped = t_end - t_start
    dct = {
            'model': matcher_name,
            'features': list(matcher.features.keys())[0],
            'conf_thresh': conf_threshold,
            'ransacThresh': ransacThresh,
            'ransacConf': ransacConf,            
            'image0': fname0,
            'image1': fname1,
            'num_kp_image0': num_kpts0,
            'num_kp_image1': num_kpts1,
            'num_matched': matching_num0.item(),
            'norm_score': matching_score.item(),
            'match_prop': match_prop.item(),
            'num_f_inlier': num_f_inlier,
            'num_f_outlier': num_f_outlier,            
            't_elasped': t_elasped
            }
    data = {k:[v] for k,v in dct.items()}
    df = pd.DataFrame(data)
    print(fname0 + " (Query) and " + fname1 + " (Selected) successfully processed!")
    return df

def eval_matcher_on_dataset(setup, dir_query, dir_selected, dir_output):
    # Loop through all query images
    df_list_all  = []
    for query in dir_query.iterdir():
        query_fname, _ = os.path.splitext(query.name)
        try:
            query_image = load_image(query)

            # Make directory if output to save queried image does not exist
            query_results_dir = Path(dir_output / query_fname)
            if not os.path.exists(query_results_dir):
                os.makedirs(query_results_dir)

            df_list_query  = []
            for selected in dir_selected.iterdir():
                selected_fname, _ = os.path.splitext(selected.name)
                selected_image = load_image(selected)
                try:
                    df = eval_matcher(setup, query_results_dir, query_image, selected_image, query_fname, selected_fname)
                    df_list_query.append(df)
                except:                
                    matcher = setup['matcher_model']
                    matcher_name = setup['matcher_name']
                    dct = {
                            'model': matcher_name,
                            'features': list(matcher.features.keys())[0],
                            'conf_thresh': matcher.default_conf["filter_threshold"],
                            'image0': query_fname,
                            'image1': selected_fname,
                            'num_kp_image0': "N.A",
                            'num_kp_image1': "N.A",
                            'num_matched': "N.A",
                            'norm_score': "N.A",
                            'match_prop': "N.A",
                            't_elasped': "N.A"
                            }
                    data = {k:[v] for k,v in dct.items()}
                    df = pd.DataFrame(data)
                df_list_all.append(df)
                df_list_query.append(df)
            df_query = pd.concat(df_list_query)
            df_query.to_csv(query_results_dir.with_suffix('.csv'))
            print("Finished processing " + query_fname)
        except:
            print("Failed to process " + query_fname)
                                        
    final_df = pd.concat(df_list_all)
    final_df.to_csv(dir_output.with_suffix('.csv'))


if __name__ == '__main__':
    # Configs from original code
    torch.set_grad_enabled(False)    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 'mps', 'cpu'

    LG_setup = {
            'extractor': SuperPoint(max_num_keypoints=2048).eval().to(device),
            'matcher_model': LightGlue(features="superpoint").eval().to(device),
            'matcher_name': "LightGlue",
            'device': device,
            'f_removal': True,
            "ransacThresh": 7,
            "ransacConf": 0.99
    }
    LG_dir_query = Path("assets/Querytest")
    LG_dir_selected = Path("assets/selectedtest")
    LG_dir_output = Path("output/LightGlue")
    eval_matcher_on_dataset(LG_setup, LG_dir_query, LG_dir_selected, LG_dir_output)

    # SG_setup = {
    #         'extractor': SuperPoint(max_num_keypoints=2048).eval().to(device),
    #         'matcher_model': SuperGlue(features="superpoint").eval().to(device),
    #         'matcher_name': "SuperGlue",
    #         'device': device,
    #          'f_removal': True,
    #          "ransacThresh": 7,
    #          "ransacConf": 0.99
    # }

    # SG_dir_query = Path("assets/QueryImages")
    # SG_dir_selected = Path("assets/selectedFrames")
    # SG_dir_output = Path("output/SuperGlue")
    # eval_matcher_on_dataset(SG_setup, SG_dir_query, SG_dir_selected, SG_dir_output)


    # # Single Evaluation
    # images = Path("assets")
    # image0 = load_image(images / "DSC_0411.JPG")
    # image1 = load_image(images / "DSC_0410.JPG")
    # fname0 = "DSC_0411"
    # fname1 = "DSC_0410"
    # dir_output = Path("output")
    # eval_matcher(setup, dir_output, image0, image1, fname0, fname1)
