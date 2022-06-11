import torch
import utils
import time
import numpy as np
import torch.nn.functional as F

from model import FAGCN


if __name__ == '__main__':
    dataset_name = 'chameleon'
    # dataset_name = 'Pubmed'

    heter_dataset = ['chameleon', 'cornell', 'film', 'squirrel', 'texas', 'wisconsin']
    homo_dataset = ['Cora', 'Citeseer', 'Pubmed']
    num_hidden = 32
    dropout = 0.5
    eps = 0.3
    layer_num = 2
    lr = 0.01
    weight_decay = 5e-5
    max_epoch = 500
    patience = 200

    re_generate_train_val_test = True

    if dataset_name in heter_dataset:
        data, num_features, num_classes = utils.load_heter_data(dataset_name)
    elif dataset_name in homo_dataset:
        dataset = utils.load_homo_data(dataset_name)
        data = dataset[0]
        num_features = dataset.num_features
        num_classes = dataset.num_classes
    else:
        print("We do not have {} dataset right now.".format(dataset_name))

    utils.set_seed(15)

    if re_generate_train_val_test:
        idx_train, idx_val, idx_test = utils.split_nodes(data.y, 0.1, 0.1, 0.8, 15)
    else:
        if dataset_name in heter_dataset:
            idx_train, idx_val, idx_test = utils.split_nodes(data.y, 0.1, 0.1, 0.8, 15)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


    data = data.to(device)
    idx_train = idx_train.to(device)
    idx_val = idx_val.to(device)
    idx_test = idx_test.to(device)



    net = FAGCN(data, num_features, num_hidden, num_classes, dropout, eps, layer_num)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

    dur = []
    los = []
    loc = []
    counter = 0
    min_loss = 100.0
    max_acc = 0.0



    for epoch in range(max_epoch):
        if epoch >= 3:
            t0 = time.time()
        net.train()
        logp = net(data.x)


        cla_loss = F.nll_loss(logp[idx_train], data.y[idx_train])
        loss = cla_loss
        train_acc = utils.accuracy(logp[idx_train], data.y[idx_train])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        net.eval()
        logp = net(data.x)
        test_acc = utils.accuracy(logp[idx_test], data.y[idx_test])
        loss_val = F.nll_loss(logp[idx_val], data.y[idx_val]).item()
        val_acc = utils.accuracy(logp[idx_val], data.y[idx_val])
        los.append([epoch, loss_val, val_acc, test_acc])

        if loss_val < min_loss and max_acc < val_acc:
            min_loss = loss_val
            max_acc = val_acc
            counter = 0
        else:
            counter += 1

        if counter >= patience and dataset_name in homo_dataset:
            print('early stop')
            break

        if epoch >= 3:
            dur.append(time.time() - t0)

        print("Epoch {:05d} | Loss {:.4f} | Train {:.4f} | Val {:.4f} | Test {:.4f} | Time(s) {:.4f}".format(
            epoch, loss_val, train_acc, val_acc, test_acc, np.mean(dur)))

    if dataset_name in homo_dataset or 'syn' in dataset_name:
        los.sort(key=lambda x: x[1])
        acc = los[0][-1]
        print(acc)
    else:
        los.sort(key=lambda x: -x[2])
        acc = los[0][-1]
        print(acc)
