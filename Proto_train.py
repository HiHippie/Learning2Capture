from torch.utils.data import DataLoader, TensorDataset
import torch
from learn2capture.prototype_loss import PrototypicalLoss


def proto_train(data, n_ways, n_shot, KL, topK):

    [X_aug, Y_aug, query_data, query_label] = data
    X_aug, Y_aug, query_data, query_label = \
        torch.from_numpy(X_aug), torch.from_numpy(Y_aug), torch.from_numpy(query_data), torch.from_numpy(query_label)

    X_test, Y_test = torch.cat((X_aug, query_data), dim=0), torch.cat((Y_aug, query_label), dim=0)
    test_loader = DataLoader(TensorDataset(X_test, Y_test.long()), batch_size=X_test.size(0), shuffle=False)

    proto_loss = PrototypicalLoss(n_support=n_ways*n_shot+n_ways*topK*KL)

    # test on query set
    for i, (test_x, test_y) in enumerate(test_loader):
        test_x, test_y = test_x.cuda().float(), test_y.cuda().float()

        loss, query_acc = proto_loss(test_x, test_y.long())

    return query_acc