import numpy as np
from pycocotools import mask as mask_utils
from scipy.optimize import linear_sum_assignment


def compute_av_loc(data):
    alphas = np.arange(0.05, 0.99, 0.05)
    res = {}
    res['FA'] = np.zeros((len(alphas)), dtype=float)
    res['FAn'] = np.array([None] * len(alphas))   # frame accuracy in no sound source
    res['FAs'] = np.array([None] * len(alphas))   # frame accuracy in single sound source
    res['FAm'] = np.array([None] * len(alphas))   # frame accuracy in multi sound source

    res['frame_num_n_all'] = np.zeros((len(alphas)), dtype=int)
    res['frame_num_n_tp'] = np.zeros((len(alphas)), dtype=int)
    res['frame_num_s_all'] = np.zeros((len(alphas)), dtype=int)
    res['frame_num_s_tp'] = np.zeros((len(alphas)), dtype=int)
    res['frame_num_m_all'] = np.zeros((len(alphas)), dtype=int)
    res['frame_num_m_tp'] = np.zeros((len(alphas)), dtype=int)

    frame_num_all = data['num_timesteps']
    gt_classes = data['gt_classes']
    gt_dets = data['gt_dets']
    raw_classes = data['raw_classes']
    raw_dets = data['raw_dets']
    pred_classes = data['tracker_classes']
    pred_dets = data['tracker_dets']

    # 1. Find the best trajectory between gt and pred
    unique_gt_ids = []
    unique_tracker_ids = []
    for t in range(data['num_timesteps']):
        unique_gt_ids += list(np.unique(data['gt_ids'][t]))
        unique_tracker_ids += list(np.unique(data['tracker_ids'][t]))
    # Re-label IDs such that there are no empty IDs
    if len(unique_gt_ids) > 0:
        unique_gt_ids = np.unique(unique_gt_ids)
        gt_id_map = np.nan * np.ones((np.max(unique_gt_ids) + 1))
        gt_id_map[unique_gt_ids] = np.arange(len(unique_gt_ids))
        for t in range(data['num_timesteps']):
            if len(data['gt_ids'][t]) > 0:
                data['gt_ids'][t] = gt_id_map[data['gt_ids'][t]].astype(int)
    if len(unique_tracker_ids) > 0:
        unique_tracker_ids = np.unique(unique_tracker_ids)
        tracker_id_map = np.nan * np.ones((np.max(unique_tracker_ids) + 1))
        tracker_id_map[unique_tracker_ids] = np.arange(len(unique_tracker_ids))
        for t in range(data['num_timesteps']):
            if len(data['tracker_ids'][t]) > 0:
                data['tracker_ids'][t] = tracker_id_map[data['tracker_ids'][t]].astype(int)
    data['num_tracker_ids'] = len(unique_tracker_ids)
    data['num_gt_ids'] = len(unique_gt_ids)
    # Variables counting global association
    potential_matches_count = np.zeros((data['num_gt_ids'], data['num_tracker_ids']))
    gt_id_count = np.zeros((data['num_gt_ids'], 1))
    tracker_id_count = np.zeros((1, data['num_tracker_ids']))
    # First loop through each timestep and accumulate global track information.
    for t, (gt_ids_t, tracker_ids_t) in enumerate(zip(data['gt_ids'], data['tracker_ids'])):
        # Count the potential matches between ids in each timestep
        # These are normalised, weighted by the match similarity.
        similarity = data['similarity_scores'][t]
        sim_iou_denom = similarity.sum(0)[np.newaxis, :] + similarity.sum(1)[:, np.newaxis] - similarity
        sim_iou = np.zeros_like(similarity)
        sim_iou_mask = sim_iou_denom > 0 + np.finfo('float').eps
        sim_iou[sim_iou_mask] = similarity[sim_iou_mask] / sim_iou_denom[sim_iou_mask]
        potential_matches_count[gt_ids_t[:, np.newaxis], tracker_ids_t[np.newaxis, :]] += sim_iou
        # Calculate the total number of dets for each gt_id and tracker_id.
        gt_id_count[gt_ids_t] += 1
        tracker_id_count[0, tracker_ids_t] += 1
    # Calculate overall jaccard alignment score (before unique matching) between IDs
    global_alignment_score = potential_matches_count / (gt_id_count + tracker_id_count - potential_matches_count)
    # Hungarian algorithm to find best matches
    match_rows, match_cols = linear_sum_assignment(-global_alignment_score)

    # 2. Compute FSLA
    for a, alpha in enumerate(alphas):
        frame_num_n_all = 0   # total frames in no sound source
        frame_num_s_all = 0   # total frames in single sound source
        frame_num_m_all = 0   # total frames in multi sound source
        frame_num_n_tp = 0    # true positive frames in no sound source
        frame_num_s_tp = 0    # true positive frames in single sound source
        frame_num_m_tp = 0    # true positive frames in multi sound source

        for frame_id in range(frame_num_all):
            # classes
            gt_classes_per_frame = gt_classes[frame_id]
            raw_classes_per_frame = raw_classes[frame_id]
            pred_classes_per_frame = pred_classes[frame_id]
            # masks
            gt_dets_per_frame = gt_dets[frame_id]
            raw_dets_per_frame = raw_dets[frame_id]
            pred_dets_per_frame = pred_dets[frame_id]

            if len(pred_dets_per_frame) > 0:
                pred_dets_per_frame_f = [di for di in pred_dets_per_frame if di['counts'] != 'PPTl0']
            else:
                pred_dets_per_frame_f = pred_dets_per_frame

            # Masks must have the same class and number
            if (set(gt_classes_per_frame) == set(pred_classes_per_frame)) and (len(gt_dets_per_frame) == len(pred_dets_per_frame_f)):
                # 1) no sound source
                if len(gt_dets_per_frame) == 0:
                    frame_num_n_all += 1
                    frame_num_n_tp += 1
                # 2) single sound source
                elif len(gt_dets_per_frame) == 1:
                    frame_num_s_all += 1
                    index_gt = [index for index, value in enumerate(raw_dets_per_frame) if value is not None][0]
                    index_pred = [index for index, element in enumerate(match_cols) if element == index_gt]
                    if index_pred != []:
                        ious = mask_utils.iou(gt_dets_per_frame, [pred_dets_per_frame[index_pred[0]]], [False])
                        if np.all(ious > alpha):
                            frame_num_s_tp += 1
                # 3) multi sound source
                else:
                    frame_num_m_all += 1
                    flags = [0] * len(match_rows)
                    for tr in range(len(match_rows)):
                        if (raw_classes_per_frame[match_rows[tr]] == pred_classes_per_frame[match_cols[tr]]):
                            if raw_dets_per_frame[match_rows[tr]] == None:
                                if pred_dets_per_frame[match_cols[tr]]['counts'] == 'PPTl0':
                                    flags[tr] = 1
                            else:
                                iou = mask_utils.iou([raw_dets_per_frame[match_rows[tr]]],
                                                     [pred_dets_per_frame[match_cols[tr]]], [False])
                                if np.all(iou > alpha):
                                    flags[tr] = 1
                    if all(ff == 1 for ff in flags):
                        frame_num_m_tp += 1
            else:
                if len(gt_dets_per_frame) == 0:
                    frame_num_n_all += 1
                elif len(gt_dets_per_frame) == 1:
                    frame_num_s_all += 1
                else:
                    frame_num_m_all += 1

        assert frame_num_all == (frame_num_n_all + frame_num_s_all + frame_num_m_all)

        if frame_num_n_all > 0:
            res['FAn'][a] = frame_num_n_tp / frame_num_n_all
            res['frame_num_n_all'][a] = frame_num_n_all
            res['frame_num_n_tp'][a] = frame_num_n_tp
        else:
            res['FAn'][a] = None
            res['frame_num_n_all'][a] = 0
            res['frame_num_n_tp'][a] = 0

        if frame_num_s_all > 0:
            res['FAs'][a] = frame_num_s_tp / frame_num_s_all
            res['frame_num_s_all'][a] = frame_num_s_all
            res['frame_num_s_tp'][a] = frame_num_s_tp
        else:
            res['FAs'][a] = None
            res['frame_num_s_all'][a] = 0
            res['frame_num_s_tp'][a] = 0

        if frame_num_m_all > 0:
            res['FAm'][a] = frame_num_m_tp / frame_num_m_all
            res['frame_num_m_all'][a] = frame_num_m_all
            res['frame_num_m_tp'][a] = frame_num_m_tp
        else:
            res['FAm'][a] = None
            res['frame_num_m_all'][a] = 0
            res['frame_num_m_tp'][a] = 0

        res['FA'][a] = (frame_num_n_tp + frame_num_s_tp + frame_num_m_tp) / frame_num_all

    return res


def combine_av_loc_sequences(all_res):
    """Combines metrics across all sequences"""
    res = {}
    fields_num = ['frame_num_n_all', 'frame_num_s_all', 'frame_num_m_all', 'frame_num_n_tp', 'frame_num_s_tp', 'frame_num_m_tp']
    for field in fields_num:
        res[field] = sum([all_res[k][field] for k in all_res.keys()])

    res_final = {}

    res_final['FAn'] = res['frame_num_n_tp'] / res['frame_num_n_all']
    res_final['FAn_count'] = res['frame_num_n_tp']
    res_final['FAn_all'] = res['frame_num_n_all']
    res_final['FAs'] = res['frame_num_s_tp'] / res['frame_num_s_all']
    res_final['FAs_count'] = res['frame_num_s_tp']
    res_final['FAs_all'] = res['frame_num_s_all']
    res_final['FAm'] = res['frame_num_m_tp'] / res['frame_num_m_all']
    res_final['FAm_count'] = res['frame_num_m_tp']
    res_final['FAm_all'] = res['frame_num_m_all']
    res_final['FA'] = (res['frame_num_n_tp'] + res['frame_num_s_tp'] + res['frame_num_m_tp']) / (res['frame_num_n_all'] + res['frame_num_s_all'] + res['frame_num_m_all'])

    return res_final