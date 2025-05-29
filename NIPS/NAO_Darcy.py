import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import os
from timeit import default_timer
from utilities import *
import sys


class NAO(nn.Module):
    def __init__(self, ntoken, head, dk, nlayer, ngrid, feature_dim, out_dim, device):
        super().__init__()

        self.ntoken = ntoken
        self.ngrid = ngrid
        self.head = head
        self.dk = dk
        self.nlayer = nlayer
        self.feature_dim = feature_dim
        self.out_dim = out_dim
        self.kernel = DenseNet([4, 32, 64, 1], torch.nn.LeakyReLU)
        self.kernelf = DenseNet([4, 32, 64, 1], torch.nn.LeakyReLU)

        grid = self.get_grid(ngrid, ngrid)
        self.dx = (grid[1, 0, 0] - grid[0, 0, 0])
        self.dy = (grid[0, 1, 1] - grid[0, 0, 1])
        grid = grid.reshape(self.ntoken, 2)

        self.edge = self.compute_edges(grid, device)

        for i in range(self.head):
            for j in range(self.nlayer):
                self.add_module('fcq_%d_%d' % (j, i), nn.Linear(feature_dim, self.dk))
                self.add_module('fck_%d_%d' % (j, i), nn.Linear(feature_dim, self.dk))

        for j in range(self.nlayer):
            if j == 0:
                self.add_module('fcn%d' % j, nn.LayerNorm([self.ntoken * 2, feature_dim]))
            else:
                self.add_module('fcn%d' % j, nn.LayerNorm(feature_dim))

    def forward(self, xy):
        batchsize = xy.shape[0]

        # generate continuous W^P
        weight = self.kernel(self.edge).view(1, self.ntoken, self.ntoken)
        weightf = self.kernelf(self.edge).view(1, self.ntoken, self.ntoken)

        Vinput = self._modules['fcn%d' % 0](xy)  #+ torch.kron(torch.ones((batchsize, 1, 1), device=xy.device), P[:, :xy.shape[1], :]), no positional encoding
        out_ft = torch.zeros((batchsize, self.feature_dim, self.out_dim), device=xy.device)

        # intermediate layers
        for j in range(self.nlayer - 1):
            mid_ft = torch.zeros((batchsize, self.ntoken * 2, self.feature_dim), device=xy.device)
            for i in range(self.head):
                Q = self._modules['fcq_%d_%d' % (j, i)](Vinput)
                K = self._modules['fck_%d_%d' % (j, i)](Vinput)
                Attn = torch.matmul(Q, K.transpose(1, 2)) / torch.sqrt(torch.tensor(self.dk, device=xy.device))
                V = self.dx * self.dy * torch.matmul(Attn, Vinput)
                mid_ft = mid_ft + V  # V is directly added together assuming Wp=I, following simplifying transformer blocks
            Vinput = self._modules['fcn%d' % (j + 1)](mid_ft) + Vinput  # skip connection

        # final layer
        for i in range(self.head):
            Q = self._modules['fcq_%d_%d' % (self.nlayer - 1, i)](Vinput)
            K = self._modules['fck_%d_%d' % (self.nlayer - 1, i)](Vinput)
            Attn = torch.matmul(Q, K.transpose(1, 2)) / torch.sqrt(torch.tensor(self.dk, device=xy.device))
            Vu = self.dx * self.dy * torch.matmul(Attn[:, :self.ntoken, :self.ntoken], xy[:, :self.ntoken, :])  # Vu shape: batch size x ntoken x feature_dim
            Vf = self.dx * self.dy * torch.matmul(Attn[:, self.ntoken:, :self.ntoken], xy[:, :self.ntoken, :])  # Vf shape: batch size x ntoken x feature_dim

            out_ft += self.dx * self.dy * torch.matmul(Vu.permute(0, 2, 1), weight)
            out_ft += self.dx * self.dy * torch.matmul(Vf.permute(0, 2, 1), weightf)

        return out_ft.permute(0, 2, 1)  # shape: batch size x ntoken x out_dim

    def compute_edges(self, grid, device):
        grid_i = grid.unsqueeze(1).expand(-1, self.ntoken, -1)
        grid_j = grid.unsqueeze(0).expand(self.ntoken, -1, -1)
        edges = torch.cat((grid_i, grid_j), dim=-1).to(device)
        return edges

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


def main(i_seed, learning_rate, gamma, wd):
    torch.manual_seed(i_seed)
    torch.cuda.manual_seed(i_seed)
    np.random.seed(i_seed)

    # model setup
    head = 1
    nlayer = 4  # number of layers
    dk = 40  # size of WQ and WK
    feature_dim = 50

    ngrid = 21
    out_dim = ngrid ** 2

    step_size = 100
    epochs = 5000
    batch_size = 100

    ntrain = 90
    ntest = 10

    sample_per_task = 100
    rands = 100

    ntrain_total = ntrain * rands
    ntest_total = ntest * rands

    data_path = './data_darcy/DarcyStatic_A100F100_10000x%dx%d_chi_sol_source.mat' % (ngrid, ngrid)
    reader = MatReader(data_path)
    sol_train = reader.read_field('sol')[:ntrain*sample_per_task, :].view(ntrain*sample_per_task, ngrid, ngrid)
    ff_train = reader.read_field('source')[:ntrain*sample_per_task, :].view(ntrain*sample_per_task, ngrid, ngrid)

    sol_test = reader.read_field('sol')[-ntest*sample_per_task:, :].view(ntest*sample_per_task, ngrid, ngrid)
    ff_test = reader.read_field('source')[-ntest*sample_per_task:, :].view(ntest*sample_per_task, ngrid, ngrid)

    # data augmentation by permuting samples
    x_train = []
    f_train = []
    for t in range(ntrain):
        for _ in range(rands):
            crand = torch.randperm(sample_per_task)
            x_train.append(sol_train[t * sample_per_task+crand[:feature_dim], ...].unsqueeze(0))  # [1, feature_dim, 21, 21]
            f_train.append(ff_train[t * sample_per_task+crand[:feature_dim], ...].unsqueeze(0))

    x_train = torch.cat(x_train, dim=0).permute(0, 2, 3, 1).reshape(ntrain*rands, -1, feature_dim)
    f_train = torch.cat(f_train, dim=0).permute(0, 2, 3, 1).reshape(ntrain*rands, -1, feature_dim)

    x_test = []
    f_test = []
    for t in range(ntest):
        for _ in range(rands):
            crand = torch.randperm(sample_per_task)
            x_test.append(sol_test[t * sample_per_task + crand[:feature_dim], ...].unsqueeze(0))  # [1, feature_dim, 21, 21]
            f_test.append(ff_test[t * sample_per_task + crand[:feature_dim], ...].unsqueeze(0))

    x_test = torch.cat(x_test, dim=0).permute(0, 2, 3, 1).reshape(ntest*rands, -1, feature_dim)
    f_test = torch.cat(f_test, dim=0).permute(0, 2, 3, 1).reshape(ntest*rands, -1, feature_dim)

    print(f'>> Train and test data shape: {f_train.shape} and {f_test.shape}')

    # normalize data
    x_normalizer = GaussianNormalizer(x_train)
    x_train_n = x_normalizer.encode(x_train)
    x_test_n = x_normalizer.encode(x_test)

    f_normalizer = GaussianNormalizer(f_train)
    f_train = f_normalizer.encode(f_train)
    f_test = f_normalizer.encode(f_test)

    xy_train = torch.cat((f_train, x_train_n), dim=1)
    xy_test = torch.cat((f_test, x_test_n), dim=1)

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train,
                                                                              xy_train),
                                               batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test,
                                                                             xy_test),
                                              batch_size=batch_size, shuffle=False)

    base_dir = './checkpoints_Darcy/NAO_n%d_seed%d' % (ntrain_total, seed)
    os.makedirs(base_dir, exist_ok=True)

    res_dir = './res_NAO_Darcy'
    os.makedirs(res_dir, exist_ok=True)
    res_file = "%s/NAO_Darcy_n%d_dk%d.txt" % (res_dir, ntrain_total, dk)
    if not os.path.isfile(res_file):
        f = open(res_file, "w")
        f.write(f'ntrain_total, seed, lr, gamma, wd, train_lowest, train_best, test, best_epoch, time (hrs)\n')
        f.close()

    myloss = LpLoss(size_average=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not torch.cuda.is_available():
        print(f'>> Device being used: {device}')
    else:
        print(f'>> Device being used: {device} ({torch.cuda.get_device_name(0)})')

    model = NAO(out_dim, head, dk, nlayer, ngrid, feature_dim, out_dim, device).to(device)
    print(f'>> Total number of model params: {count_params(model)}')

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=wd)
    model_filename = '%s/NAO_g2u_depth%d_dk%d_lr%.1e_gamma%.2f_wd%.1e.ckpt' % (base_dir,
                                                                               nlayer,
                                                                               dk,
                                                                               learning_rate,
                                                                               gamma,
                                                                               wd)

    train_loss_lowest = train_loss_best = test_loss_best = 1e8
    best_epoch = 0
    t0 = default_timer()
    for ep in range(epochs):
        optimizer = scheduler(optimizer, LR_schedule(learning_rate, ep, step_size, gamma))
        model.train()
        t1 = default_timer()
        train_l2 = 0
        for y, xy in train_loader:
            y, xy = y.to(device), xy.to(device)
            this_batch_size = xy.shape[0]
            optimizer.zero_grad()
            out = model(xy)
            loss = myloss(out.reshape(this_batch_size, -1), y.reshape(this_batch_size, -1))
            loss.backward()

            optimizer.step()
            train_l2 += loss.item()
        train_l2 /= ntrain_total

        if train_l2 < train_loss_best:
            train_loss_lowest = train_l2
            test_l2 = 0.0
            model.eval()
            with torch.no_grad():
                for y, xy in test_loader:
                    y, xy = y.to(device), xy.to(device)
                    this_batch_size = xy.shape[0]
                    optimizer.zero_grad()
                    out = model(xy)
                    test_l2 += myloss(out.reshape(this_batch_size, -1), y.reshape(this_batch_size, -1))

            test_l2 /= ntest_total
            if test_l2 < test_loss_best:
                best_epoch = ep
                train_loss_best = train_l2
                test_loss_best = test_l2
                torch.save(model.state_dict(), model_filename)

                t2 = default_timer()
                print(f'>> ep [{(ep + 1): >{len(str(epochs))}d}/{epochs}], runtime: {(t2 - t1):.1f}s, '
                      f'train err: {train_l2:.4f}, test err: {test_l2:.4f}', flush=True)
            else:
                t2 = default_timer()
                print(f'>> ep [{(ep + 1): >{len(str(epochs))}d}/{epochs}], runtime: {(t2 - t1):.1f}s, '
                      f'train err: {train_l2:.4f} (best: [{best_epoch + 1}], '
                      f'{train_loss_best:.4f}/{test_loss_best:.4f})', flush=True)
        else:
            t2 = default_timer()
            print(f'>> ep [{(ep + 1): >{len(str(epochs))}d}/{epochs}], runtime: {(t2 - t1):.1f}s, '
                  f'train err: {train_l2:.4f} (best: [{best_epoch + 1}], '
                  f'{train_loss_best:.4f}/{test_loss_best:.4f})', flush=True)

    t3 = default_timer()

    f = open(res_file, "a")
    f.write(f'{ntrain_total}, {i_seed}, {learning_rate}, {gamma}, {wd}, {train_loss_lowest}, {train_loss_best}, '
            f'{test_loss_best}, {best_epoch}, {(t3 - t0) / 3600:.2f}\n')
    f.close()

if __name__ == '__main__':
    seed = 0

    lrs = [3e-2]
    gammas = [0.9]
    wds = [1e-5]

    icount = 0
    case_total = len(lrs) * len(gammas) * len(wds)
    for lr in lrs:
        for gamma in gammas:
            for wd in wds:
                icount += 1
                print("-" * 100)
                print(f'>> running case {icount}/{case_total}: lr={lr}, gamma={gamma}, wd={wd}')
                print("-" * 100)
                main(seed, lr, gamma, wd)
    print('\n')
    print(f'********** Training completed! **********')
