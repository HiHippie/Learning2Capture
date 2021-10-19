import numpy as np
import torch


use_gpu = torch.cuda.is_available()


def get_neighberd(start_idx, G ,mark_list):

    depth_neighber_idx = torch.sort(G[start_idx, :], descending=False)[1]
    for dni in depth_neighber_idx:
        if dni not in mark_list:
            return dni
        else:
            continue

def DS(args, support_data, support_label, query_data):

    sampled_data, sampled_label = [], []

    if args.shots > 1:
        base_mean = []
        for i in range(args.ways):
            start_base = torch.mean(torch.from_numpy(support_data)[torch.where(torch.from_numpy(support_label) == i)[0]], dim=0, keepdim=True)
            base_mean.append(start_base)
        base_mean = torch.cat(base_mean)
    else:
        base_mean = torch.from_numpy(support_data)[:args.shots*args.ways]

    X = torch.from_numpy(np.concatenate([base_mean, query_data], axis=0))
    X1 = X.unsqueeze(1)
    X2 = X.unsqueeze(0)
    distance = torch.norm(X1 - X2, p=2, dim=2)

    idx = 0
    rdy2sample_idx, latent_label = [], []
    cls_std_list = []

    for i in range(args.ways):
        depth_idx_query_list = []
        start_idx = idx
        for _ in range(args.topk):
            next_point = get_neighberd(start_idx=start_idx, G=distance[:, args.ways:], mark_list=(depth_idx_query_list))
            depth_idx_query_list.append(next_point)
            start_idx = args.ways + next_point
        idx += 1
        latent_label.extend([support_label[i]]*len(depth_idx_query_list))
        rdy2sample_idx.extend(depth_idx_query_list)

        cls_std = np.std(query_data[depth_idx_query_list], axis=0)
        cls_std_list.append(cls_std)

    tmp_cls_idx_ = np.array(list(range(args.ways))*(args.topk*args.num_latent)).reshape((args.topk*args.num_latent,args.ways)).T.reshape(-1)
    for i, idx in enumerate(rdy2sample_idx):
        label = latent_label[i]
        tmp_cls_idx = tmp_cls_idx_[i]
        mean = query_data[idx]
        std = cls_std_list[tmp_cls_idx]
        sampled_data.extend(np.array(
            [np.random.normal(mean[j], std[j], args.num_latent) for j in range(len(mean))]).T)
        sampled_label.extend([label] * args.num_latent)

    return sampled_data, sampled_label