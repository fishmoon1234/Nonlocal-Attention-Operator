import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import os
from timeit import default_timer
from utilities import *
import sys
from collections import OrderedDict


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
        self.device = device
        self.sqrt_dk_inv = 1 / torch.sqrt(torch.tensor(dk, device=device))

        self.kernel = DenseNet([4, 32, 64, 1], torch.nn.LeakyReLU)
        self.kernelf = DenseNet([4, 32, 64, 1], torch.nn.LeakyReLU)

        grid = self.get_grid(ngrid, ngrid)
        self.dx = (grid[1, 0, 0] - grid[0, 0, 0])
        self.dy = (grid[0, 1, 1] - grid[0, 0, 1])
        grid = grid.reshape(self.ntoken, 2)

        self.edge = self.compute_edges(grid)

        self.fcq = nn.ModuleList([nn.ModuleList([nn.Linear(feature_dim, dk) for _ in range(head)]) for _ in range(nlayer)])
        self.fck = nn.ModuleList([nn.ModuleList([nn.Linear(feature_dim, dk) for _ in range(head)]) for _ in range(nlayer)])
        # self.fcn = nn.ModuleList([nn.LayerNorm([ntoken * 2, feature_dim]) for _ in range(nlayer)])
        self.fcn = nn.ModuleList([nn.LayerNorm([ntoken * 2, feature_dim]) if i == 0 else nn.LayerNorm(feature_dim) for i in range(nlayer)])

    def forward(self, xy):
        batchsize = xy.shape[0]

        # generate continuous W^P
        weight = self.kernel(self.edge).view(1, self.ntoken, self.ntoken)
        weightf = self.kernelf(self.edge).view(1, self.ntoken, self.ntoken)

        Vinput = self.fcn[0](xy)  #+ torch.kron(torch.ones((batchsize, 1, 1), device=xy.device), P[:, :xy.shape[1], :]), no positional encoding
        out_ft = torch.zeros((batchsize, self.out_dim, self.feature_dim), device=self.device)

        # intermediate layers
        for j in range(self.nlayer - 1):
            mid_ft = torch.zeros_like(Vinput)
            for i in range(self.head):
                Q = self.fcq[j][i](Vinput)
                K = self.fck[j][i](Vinput)
                KV = self.dx * self.dy * torch.matmul(K.transpose(1, 2), Vinput) * self.sqrt_dk_inv
                mid_ft += torch.matmul(Q, KV)  # V is directly added together assuming Wp=I, following simplifying transformer blocks
            Vinput = self.fcn[j + 1](mid_ft) + Vinput  # skip connection

        # final layer
        for i in range(self.head):
            Q = self.fcq[self.nlayer - 1][i](Vinput)
            K = self.fck[self.nlayer - 1][i](Vinput)
            KV = self.dx * self.dy * torch.matmul(K.transpose(1, 2)[..., :self.ntoken], xy[:, :self.ntoken, :]) * self.sqrt_dk_inv
            out_ft += self.dx * self.dy * torch.matmul(torch.matmul(torch.cat((weight, weightf), dim=-1), Q), KV)

        return out_ft  # shape: batch size x ntoken x out_dim

    def compute_edges(self, grid):
        grid_i = grid.unsqueeze(1).expand(-1, self.ntoken, -1)
        grid_j = grid.unsqueeze(0).expand(self.ntoken, -1, -1)
        edges = torch.cat((grid_i, grid_j), dim=-1).to(self.device)
        return edges

    def get_grid(self, size_x, size_y):
        gridx = torch.linspace(0, 1, size_x).reshape(size_x, 1, 1).repeat([1, size_y, 1])
        gridy = torch.linspace(0, 1, size_y).reshape(1, size_y, 1).repeat([size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=2)


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

    base_dir = './checkpoints_Darcy_g2u_ngrid%d/LinearNAO_n%d_seed%d' % (ngrid, ntrain_total, seed)
    os.makedirs(base_dir, exist_ok=True)

    res_dir = './res_NIPS_Darcy_g2u_ngrid%d' % ngrid
    os.makedirs(res_dir, exist_ok=True)
    res_file = "%s/LinearNAO_l%d_Darcy_n%d_dk%d_ngrid%d.txt" % (res_dir, nlayer, ntrain_total, dk, ngrid)
    if not os.path.isfile(res_file):
        f = open(res_file, "w")
        f.write(f'ntrain_total, seed, lr, gamma, wd, train_lowest, train_best, test1000, test2000, test3000, test4000, '
                f'test, best_epoch, time (hrs)\n')
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
    model_filename = '%s/NAO_g2u_l%d_dk%d_lr%.1e_gamma%.2f_wd%.1e.ckpt' % (base_dir,
                                                                               nlayer,
                                                                               dk,
                                                                               learning_rate,
                                                                               gamma,
                                                                               wd)

    train_loss_lowest = train_loss_best = test_loss_best = 1e8
    test_loss_best_ep = OrderedDict()
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
        if ep + 1 > 999 and (ep + 1) % 1000 == 0:
            test_loss_best_ep[ep + 1] = test_loss_best.item()

    t3 = default_timer()

    with open(res_file, 'a') as f:
        test_loss_best_str = ', '.join(map(str, test_loss_best_ep.values()))
        f.write(f'{ntrain_total}, {i_seed}, {learning_rate}, {gamma}, {wd}, {train_loss_lowest}, {train_loss_best}, '
                f'{test_loss_best_str}, {best_epoch}, {(t3 - t0) / 3600:.2f}\n')

if __name__ == '__main__':
    seed = 0

    lrs = [3e-2]
    gammas = [0.9]
    wds = [1e-5]

    # if len(sys.argv) > 1:
    #     lrs = [lrs[int(sys.argv[1])]]

    icount = 0
    case_total = len(lrs) * len(gammas) * len(wds)
    for lr in lrs:
        for gamma in gammas:
            for wd in wds:
                icount += 1
                print("-" * 100)
                print(f'>> running case {icount}/{case_total}: lr={lr}, gamma={gamma}, wd={wd}')
                print("-" * 100, flush=True)
                main(seed, lr, gamma, wd)
    print('\n')
    print(f'********** Training completed! **********')
