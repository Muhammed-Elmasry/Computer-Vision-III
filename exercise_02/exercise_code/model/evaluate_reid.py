import torch
from exercise_code.model.distance_metrics import euclidean_squared_distance, cosine_distance
from exercise_code.model.distance_metrics import compute_distance_matrix
import numpy as np

def extract_features(model, data_loader):
    device = torch.device("cuda:0" if next(model.parameters()).is_cuda else "cpu")
    f_, pids_, camids_ = [], [], []
    for data in data_loader:
        imgs, pids, camids = data['img'], data['pid'], data['camid']
        imgs = imgs.to(device)
        features = model(imgs)
        features = features.cpu().clone()
        f_.append(features)
        pids_.extend(pids)
        camids_.extend(camids)
    f_ = torch.cat(f_, 0)
    pids_ = np.asarray(pids_)
    camids_ = np.asarray(camids_)
    return f_, pids_, camids_

def eval_market1501(distmat, q_pids, g_pids, q_camids, g_camids, max_rank):
    """Evaluation with market1501 metric
    For each query identity, its gallery images from the same camera view are discarded.

    This code needs to be simplified.
    """
    num_q, num_g = distmat.shape

    if num_g < max_rank:
        max_rank = num_g
        print(
            'Note: number of gallery samples is quite small, got {}'.
            format(num_g)
        )

    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0. # number of valid query

    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        raw_cmc = matches[q_idx][
            keep] # binary vector, positions with value 1 are correct matches
        if not np.any(raw_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = raw_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = raw_cmc.sum()
        tmp_cmc = raw_cmc.cumsum()
        tmp_cmc = [x / (i+1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * raw_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, 'Error: all query identities do not appear in gallery'

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP


def evaluate_reid(model, test_loader, ranks=[1, 5, 10, 20]):
    with torch.no_grad():
        model.eval()
        print('Extracting features from query set...')
        q_feat, q_pids, q_camids = extract_features(model, test_loader['query'])
        # print('Done, obtained {}-by-{} matrix'.format(q_feat.size(0), q_feat.size(1)))
        print(f"Done, obtained {q_feat.size(0)}-by-{q_feat.size(1)} matrix")

        print('Extracting features from gallery set ...')
        g_feat, g_pids, g_camids = extract_features(model, test_loader['gallery'])
        # print('Done, obtained {}-by-{} matrix'.format(g_feat.size(0), g_feat.size(1)))
        print(f"Done, obtained {g_feat.size(0)}-by-{g_feat.size(1)} matrix")

        ############### Ajust the metic function here ###############
        metric_fn = euclidean_squared_distance
        ############################################################

        distmat = compute_distance_matrix(q_feat, g_feat, metric_fn=metric_fn)
        distmat = distmat.numpy()

        print('Computing CMC and mAP ...')
        cmc, mAP = eval_market1501(
            distmat,
            q_pids,
            g_pids,
            q_camids,
            g_camids,
            max_rank=50
        )

        print('** Results **')
        print(f'mAP: {mAP:.1%}')
        print('CMC curve')
        for r in ranks:
            print(f'Rank-{r:<3}: {cmc[r - 1]:.1%}')
        return cmc[0], mAP
