import numpy as np
import torch



def WS(args, support_data, support_label, query_data):

    mean_list = []
    query_based_sample, query_based_sample_label = [], []

    topQs = []
    topQs_dis = []
    stds = []

    for j in range(args.ways):

        novel_task = torch.from_numpy(support_data).view(args.shots, args.ways, -1).transpose(1, 0)[j, :, :]
        if args.shots > 1:
            mean = torch.mean(novel_task, dim=0, keepdim=False).detach().cpu().numpy()

        else:
            mean = novel_task.squeeze(0).numpy()

        mean_list.append(mean)

        distance = np.linalg.norm(query_data - mean, axis=1)
        topQ = np.argsort(distance)[:args.topk]
        topQs.append(topQ)
        topQs_dis.append([distance[topQ]])
        std = np.std(np.concatenate([query_data[topQ], novel_task.numpy()]), axis=0, keepdims=False)
        stds.append(std)

    for c in range(args.ways):
        topQ = topQs[c]
        for top in topQ:
            std = stds[c]
            query_based_sample.extend(np.array(
                [np.random.normal(query_data[top][i], std[i], int(args.num_latent)) for i in
                 range(len(mean))]).T)
            query_based_sample_label.extend([support_label[c]] * (int(args.num_latent)))

    sampled_data = np.array(query_based_sample)
    sampled_label = np.array(query_based_sample_label)

    return sampled_data, sampled_label