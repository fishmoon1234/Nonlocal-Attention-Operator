import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.fft
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import savemat

import operator
from functools import reduce
from functools import partial
import os
from timeit import default_timer
from utilities_darcy import *
import sys

seed = 0  # int(sys.argv[1])

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)

class MS_Loss(nn.Module):


    def __init__(self, tokenN, head, dk, nlayer, gridN, feature_dim, out_dim=1):
        super(MS_Loss, self).__init__()

        self.tokenN = tokenN
        self.gridN = gridN
        self.head = head
        self.dk = dk
        self.nlayer = nlayer
        self.feature_dim = feature_dim
        self.out_dim = out_dim
        self.kernel = DenseNet([4, 32, 64, 1], torch.nn.LeakyReLU)
        self.kernelf = DenseNet([4, 32, 64, 1], torch.nn.LeakyReLU)

        grid = self.get_grid(gridN, gridN)
        gridT = grid.permute(1, 0, 2)
        self.dx = (grid[1, 0, 0] - grid[0, 0, 0])
        self.dy = (grid[0, 1, 1] - grid[0, 0, 1])
        grid = grid.reshape(self.tokenN, 2)
        gridT = gridT.reshape(self.tokenN, 2)
        edge = torch.zeros((self.tokenN, self.tokenN, 4))
        for i in range(self.tokenN):
            for j in range(self.tokenN):
                edge[i, j, :2] = grid[i, :]
                edge[i, j, 2:] = gridT[j, :]

        self.edge = edge.cpu()
        self.grid = grid.cpu()

        for i in range(self.head):
            for j in range(self.nlayer):
                self.add_module('fcq_%d_%d' % (j, i), nn.Linear(feature_dim, self.dk))
                self.add_module('fck_%d_%d' % (j, i), nn.Linear(feature_dim, self.dk))

        for j in range(self.nlayer):
            self.add_module('fcn%d' % j, nn.LayerNorm([self.tokenN * 2, feature_dim]))

    def forward(self, xy):
        # m=nn.Softmax(dim=2)
        # m=nn.Tanh()
        m = nn.Identity()
        batchsize = xy.shape[0]

        # to generate continuous W^P
        All_edgeu = self.kernel(self.edge).view(1, self.tokenN, self.tokenN)
        All_edgef = self.kernelf(self.edge).view(1, self.tokenN, self.tokenN)
        weight = All_edgeu.repeat([batchsize, 1, 1])
        weightf = All_edgef.repeat([batchsize, 1, 1])

        Vinput = self._modules['fcn%d' % 0](xy) #+ torch.kron(torch.ones((batchsize, 1, 1), device=xy.device), P[:, :xy.shape[1], :]), no positional encoding
        out_ft = torch.zeros((batchsize, self.feature_dim, self.out_dim), device=xy.device)
        for j in range(self.nlayer - 1):
            mid_ft = torch.zeros((batchsize, self.tokenN * 2, self.feature_dim), device=xy.device)
            for i in range(self.head):
                Q = self._modules['fcq_%d_%d' % (j, i)](Vinput)
                K = self._modules['fck_%d_%d' % (j, i)](Vinput)
                Attn = m(torch.matmul(Q, torch.transpose(K, 1, 2)) / torch.sqrt(torch.tensor(self.dk / 1.)))
                V = self.dx * self.dy * (torch.matmul(Attn[:, :, :], Vinput[:, :, :]))
                mid_ft = mid_ft + V  # V is directly added together bc Wp=I following simplifying transformer blocks
            Vinput = self._modules['fcn%d' % (j+1)](mid_ft) + Vinput  #skip connection is of critical here

        for i in range(self.head):
            Q = self._modules['fcq_%d_%d' % (self.nlayer - 1, i)](Vinput)
            K = self._modules['fck_%d_%d' % (self.nlayer - 1, i)](Vinput)
            Attn = m(torch.matmul(Q, torch.transpose(K, 1, 2)) / torch.sqrt(torch.tensor(self.dk / 1.)))
            Vu = self.dx * self.dy * (torch.matmul(Attn[:, :self.tokenN, :self.tokenN], xy[:, :self.tokenN, :]))  # Vu shape: batch size X tokenN X feature_dim
            Vu = Vu.permute(0, 2, 1)  # Vu shape: batch size X feature_dim X tokenN
            Vf = self.dx * self.dy * (torch.matmul(Attn[:, self.tokenN:, :self.tokenN], xy[:, :self.tokenN, :]))  # Vf shape: batch size X tokenN X feature_dim
            Vf = Vf.permute(0, 2, 1)
            out_ft += self.dx * self.dy * torch.matmul(Vu, weight)
            out_ft += self.dx * self.dy * torch.matmul(Vf, weightf)

        return out_ft.permute(0, 2, 1) #shape: batch size X tokenN X out_dim


    def get_grid(self, size_x, size_y):
        gridx = torch.tensor(np.linspace(-1, 1, size_x))
        gridx = gridx.reshape(size_x, 1, 1).repeat([1, size_y, 1])
        gridy = torch.tensor(np.linspace(-1, 1, size_y))
        gridy = gridy.reshape(1, size_y, 1).repeat([size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=2).float()

def scheduler(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer


def LR_schedule(learning_rate, steps, scheduler_step, scheduler_gamma):
    # print(steps//scheduler_step)
    return learning_rate * np.power(scheduler_gamma, (steps // scheduler_step))


def main():
    # model setup
    head = 1
    dk = 40  # size of WQ and WK
    nlayer = 2  # number of layers
    learning_rate = 0.01
    gamma = 0.9
    wd = 1e-6
    step_size = 100
    epochs = 10000
    batch_size = 50
    gridN = 21

    X_TRAIN_PATH = './darcy_data/DarcyStatic_A100F100_yu_inv.mat'

    ntrain = 90
    ntest = 10

    sample_per_task = 100
    rands = 100

    train_indexes = np.arange(0, 10000)  # np.loadtxt("train_indexes_l1.txt", dtype=int)
    train_list = list(train_indexes)[:ntrain*sample_per_task]
    test_list = list(train_indexes)[-ntest*sample_per_task:]

    reader = MatReader(X_TRAIN_PATH)
    sol_train = reader.read_field('sol')[train_list, :].view(ntrain*sample_per_task, gridN, gridN)
    f_train = reader.read_field('chi')[train_list, :].view(ntrain*sample_per_task, gridN, gridN)

    sol_test = reader.read_field('sol')[test_list, :].view(ntest*sample_per_task, gridN, gridN)
    f_test = reader.read_field('chi')[test_list, :].view(ntest*sample_per_task, gridN, gridN)


    x_list = []
    f_list = []
    feature_dim = 50
    out_dim = gridN ** 2


    #data augmentation, by permuting samples
    for t in range(ntrain):
        for yyrand in range(rands):
            x_train_20 = []
            f_train_20 = []
            crand = torch.randperm(100) #torch.arange(yyrand,yyrand+sall, dtype=torch.int64) #torch.randperm(100)
            x_train_20.append((sol_train[t * sample_per_task+crand[:feature_dim], ...]))  # [100, 21, 21]
            f_train_20.append((f_train[t * sample_per_task+crand[:feature_dim], ...]))
            x_sample = torch.cat(x_train_20, axis=0)  # [100, 21, 21]
            f_sample = torch.cat(f_train_20, axis=0)
            x_list.append(x_sample.unsqueeze(0))  # [1, 100, 21, 21]
            f_list.append(f_sample.unsqueeze(0))

    x_train = torch.cat(x_list, axis=0).permute(0, 2, 3, 1).reshape(ntrain*rands, -1, feature_dim)
    f_train = torch.cat(f_list, axis=0).permute(0, 2, 3, 1).reshape(ntrain*rands, -1, feature_dim)

    x_list = []
    f_list = []

    for t in range(ntest):
        for yyrand in range(rands):
            x_test_20 = []
            f_test_20 = []
            crand = torch.randperm(100) #torch.arange(yyrand,yyrand+sall, dtype=torch.int64) #torch.randperm(100)
            x_test_20.append((sol_test[t * sample_per_task + crand[:feature_dim], ...]))  # [100, 21, 21]
            f_test_20.append((f_test[t * sample_per_task + crand[:feature_dim], ...]))

            x_sample = torch.cat(x_test_20, axis=0)  # [100, 21, 21]
            f_sample = torch.cat(f_test_20, axis=0)
            x_list.append(x_sample.unsqueeze(0))  # [1, 100, 21, 21]
            f_list.append(f_sample.unsqueeze(0))

    x_test = torch.cat(x_list, axis=0).permute(0, 2, 3, 1).reshape(ntest*rands, -1, feature_dim)
    f_test = torch.cat(f_list, axis=0).permute(0, 2, 3, 1).reshape(ntest*rands, -1, feature_dim)

    x_list = []
    f_list = []

    print(f'>> Train and test data shape: {f_train.shape} and {f_test.shape}')

    x_normalizer = GaussianNormalizer(x_train)
    f_normalizer = GaussianNormalizer(f_train)
    x_train_n = x_normalizer.encode(x_train)
    f_train = f_normalizer.encode(f_train)
    x_test_n = x_normalizer.encode(x_test)
    f_test = f_normalizer.encode(f_test)

    xy_train = torch.cat((f_train, x_train_n), dim=1)
    xy_test = torch.cat((f_test, x_test_n), dim=1)

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(f_train, x_train_n, xy_train),batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(f_test, x_test_n, xy_test),batch_size=batch_size, shuffle=False)

    alltrain = x_train.shape[0]
    alltest = x_test.shape[0]


    base_dir = './log_darcy/attention_sine_NAO/NAO_torch_seed%d' % seed
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    myloss = LpLoss(size_average=False)
    #f_normalizer.cuda()
    total_step = 0
    test_l2 = 0.0
    tokenN = x_train.shape[1]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not torch.cuda.is_available():
        print(f'>> Device being used: {device}')
    else:
        print(f'>> Device being used: {device} ({torch.cuda.get_device_name(0)})')

    model_u = MS_Loss(tokenN, head, dk, nlayer, gridN, feature_dim, out_dim).to(device)
    print(f'>> Total number of model params: {count_params(model_u)}')

    optimizer = torch.optim.Adam(model_u.parameters(), lr=learning_rate, weight_decay=wd)
    model_u_filename = '%s/NAO_u_depth%d_dk%d_r%d_lr%f_gamma%f_wd%f_nothing_int_gtou7.ckpt' % (base_dir, nlayer, dk, head, learning_rate, gamma, wd)
#    model_u.load_state_dict(torch.load('./NAO_u_depth2_dk40_r1_lr0.010000_gamma0.700000_wd0.000001_nothing_int_gtou7.ckpt'))

    # positional encoding (uncomment the commented 3 lines below for positional encoding as a function of sin and cos)
    # P = torch.zeros((1, max_len, feature_dim)).to(device)
    # yyX = torch.arange(max_len).reshape(-1, 1) / torch.pow(1e4, torch.arange(0, feature_dim, 2) / feature_dim)
    # P[:, :, 0::2] = torch.sin(yyX)
    # P[:, :, 1::2] = torch.cos(yyX)

    # positional encoding for 2D
    # p_enc_2d = PositionalEncoding2D(feature_dim)
    # P = torch.zeros((1, gridN, gridN, feature_dim))
    # P = p_enc_2d(P).reshape(1, -1, feature_dim).permute(0, 2, 1).to(device)

    P = torch.zeros((1, out_dim, feature_dim)).to(device)

    train_loss_min = 100
    test_loss_min = 100

    for ep in range(epochs):
        optimizer = scheduler(optimizer, LR_schedule(learning_rate, ep, step_size, gamma))
        model_u.train()
        t1 = default_timer()
        train_l2 = 0
        for x, y, xy in train_loader:
            x, y, xy = x.to(device), y.to(device), xy.to(device)
            this_batch_size = xy.shape[0]
            optimizer.zero_grad()
            out_u = model_u(xy)
            out = x_normalizer.decode(out_u)
            y = x_normalizer.decode(y)

            loss = myloss(out.reshape(this_batch_size, -1), y.reshape(this_batch_size, -1))
            loss.backward()

            optimizer.step()
            train_l2 += loss.item()
        train_l2 /= alltrain

        model_u.eval()
        if train_loss_min > train_l2:
            train_loss_min = train_l2
            test_l2 = 0.0
            with torch.no_grad():
                for x, y, xy in test_loader:
                    x, y, xy = x.to(device), y.to(device), xy.to(device)
                    this_batch_size = xy.shape[0]
                    optimizer.zero_grad()
                    out_u = model_u(xy)
                    out = x_normalizer.decode(out_u)
                    y = x_normalizer.decode(y)

                    test_l2 += myloss(out.reshape(this_batch_size, -1), y.reshape(this_batch_size, -1))

                test_l2 /= alltest
                if test_loss_min > test_l2:
                    test_loss_min = test_l2
                    torch.save(model_u.state_dict(), model_u_filename)

        t2 = default_timer()
        print('depth: %d epochs: [%d/%d]  running time:%.3f  current training error: %f current test error: %f best training error: %f best test error: %f' % (
                    nlayer, ep, epochs, t2 - t1, train_l2, test_l2, train_loss_min, test_loss_min))
        total_step += 1


if __name__ == '__main__':
    main()
