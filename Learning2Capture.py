import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from tqdm import tqdm
import os
from sample_method.WideSearch import WS
from sample_method.DeepSearch import DS
import FSLTask
import random
import pickle
import configargparse
from Proto_train import proto_train
from sklearn.neural_network import MLPClassifier

use_gpu = torch.cuda.is_available()

os.environ['KMP_DUPLICATE_LIB_OK']='True'

def sample_case(ld_dict, shot, way=5, num_qry=15):
    # Sample meta task
    sample_class = random.sample(list(ld_dict.keys()), way)
    train_input = []
    test_input = []
    test_label = []
    train_label = []
    for each_class in sample_class:
        total_samples = shot + num_qry
        if len(ld_dict[each_class]) < total_samples:
            total_samples = len(ld_dict[each_class])

        samples = random.sample(ld_dict[each_class], total_samples)
        train_label += [each_class] * len(samples[:shot])
        test_label += [each_class] * len(samples[shot:])
        train_input += samples[:shot]
        test_input += samples[shot:]
    train_input = np.array(train_input).astype(np.float32)
    test_input = np.array(test_input).astype(np.float32)
    return train_input, test_input, train_label, test_label


def main(args):
    # ---- data loading

    beta = 0.5
    if_tukey_transform = True
    if_sample = True


    _datasetFeaturesFiles = "../checkpoints/{}_{}/features.plk".format(args.dataset, args.backbone)


    with open(_datasetFeaturesFiles, 'rb') as f:
        myfeatures = pickle.load(f)

    novel_feature = myfeatures[2]
    # base_feature = myfeatures[4]


    # ---- classification for each task
    lr_acc_list, svm_acc_list, nn_acc_list, proto_acc_list = [], [], [], []

    for i in tqdm(range(args.n_runs)):  # ndatas: (n_runs, n_samples, dimension)

        support_data, query_data, support_label, query_label = \
            sample_case(ld_dict=novel_feature, shot=args.shots,way=args.ways, num_qry=args.n_queries)

        support_label = np.array(support_label).reshape((args.ways, -1)).T.reshape(-1)
        support_data = np.array(support_data).reshape((args.ways, args.shots, -1)).transpose(1, 0, 2).reshape(args.ways * args.shots, -1)

        query_label = np.array(query_label).reshape((args.ways, -1)).T.reshape(-1)
        query_data = np.array(query_data).reshape((args.ways, args.n_queries, -1)).transpose(1, 0, 2).reshape(args.ways * args.n_queries, -1)


        # # ---- Tukey's transform
        if if_tukey_transform:
            support_data = np.power(support_data[:, ], beta)
            query_data = np.power(query_data[:, ], beta)

        # ---- feature sampling
        if if_sample:
            if args.method == 'WS' or 'Prototype':
                # train data
                sampled_data, sampled_label = WS(args, support_data, support_label, query_data)
            if args.method == 'DS':
                # train data
                sampled_data, sampled_label = DS(args, support_data, support_label, query_data)

            X_aug = np.concatenate([support_data, sampled_data])
            Y_aug = np.concatenate([support_label, sampled_label])

        else:
            X_aug, Y_aug = support_data, support_label

        if args.method == 'Prototype':

            proto_test_acc = proto_train(
                [X_aug, Y_aug, query_data, query_label],
                args.ways, args.shots, args.num_latent, args.topk)
            proto_acc_list.append(proto_test_acc.item())
            # print('【Prototype】【%d/%d】%s %d way %d shot  ACC : %f' % (
            # i, n_runs, dataset, n_ways, n_shot, float(np.mean(proto_acc_list))))

        else:

            # ---- LR train classifier
            LRclassifier = LogisticRegression(max_iter=1000).fit(X=X_aug, y=Y_aug)
            predicts = LRclassifier.predict(query_data)
            acc = np.mean(predicts == query_label)
            # print('【LR】【%d/%d】%s %d way %d shot  ACC : %f' % (
            #     i, n_runs, dataset, n_ways, n_shot, float(np.mean(lr_acc_list))))
            lr_acc_list.append(acc)

            # ---- SVM train classifier
            SVMclassifier = SVC(max_iter=1000).fit(X=X_aug, y=Y_aug)
            predicts = SVMclassifier.predict(query_data)
            acc = np.mean(predicts == query_label)
            # print('【SVM】【%d/%d】%s %d way %d shot  ACC : %f' % (
            #     i, n_runs, dataset, n_ways, n_shot, float(np.mean(svm_acc_list))))
            svm_acc_list.append(acc)

            # ---- NN train classifer
            NNclassifier = MLPClassifier(random_state=123, max_iter=500, hidden_layer_sizes=(128, 64)).fit(X=X_aug, y=Y_aug)
            predicts = NNclassifier.predict(query_data)
            acc = np.mean(predicts == query_label)
            nn_acc_list.append(acc)
    if args.method == 'Prototype':
        return float(np.mean(proto_acc_list)), 1.96*np.std(proto_acc_list)/np.sqrt(args.n_runs)
    else:
        return float(np.mean(lr_acc_list)), float(np.mean(svm_acc_list)), float(np.mean(nn_acc_list)), \
               1.96*np.std(lr_acc_list)/np.sqrt(args.n_runs), 1.96*np.std(svm_acc_list)/np.sqrt(args.n_runs), 1.96*np.std(nn_acc_list)/np.sqrt(args.n_runs)


if __name__ == '__main__':

    parser = configargparse.ArgParser(description='Learning2Capture')
    parser.add_argument('--dataset', type=str, default='tiered', help='mini/tiered/cub')
    parser.add_argument('--method', type=str, default="WS", help='DS/WS/Prototype')
    parser.add_argument('--backbone', type=str, default='res18', help='res18/wrn')
    parser.add_argument('--ways', type=int, default=5, help='N-way K-shot task setup')
    parser.add_argument('--shots', type=int, default=1, help='N-way K-shot task setup {1/5}')
    parser.add_argument('--topk', type=int, default=10, help='topk selection in DS=3/WS=10/prototype=10')
    parser.add_argument('--num_latent', type=int, default=1, help='number of generated samples default=1')
    parser.add_argument('--n_queries', type=int, default=15, help='number of query samples')
    parser.add_argument('--n_runs', type=int, default=10000, help='number of query samples')
    args = parser.parse_args()

    print("----------{}-{}-{}W{}S-{}----------".format
          (args.dataset, args.backbone, args.ways, args.shots, args.method))
    best_acc = 0.

    if args.method == 'Prototype':
        proto_acc, proto_ci95 = main(args)
        print('Prototype-based Classifier: {:.4f}({:.4f})'.format(proto_acc*100, proto_ci95*100))

    else:
        lr_acc, svm_acc, nn_acc, lr_ci95, svm_ci95, nn_ci95 = main(args)
        print('LR-based Classifier: {:.4f}({:.4f})'.format(lr_acc*100, lr_ci95*100))
        print('SVM-based Classifier: {:.4f}({:.4f})'.format(svm_acc*100, svm_ci95*100))
        print('NN-based Classifier: {:.4f}({:.4f})'.format(nn_acc*100, nn_ci95*100))
