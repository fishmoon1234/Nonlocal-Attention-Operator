import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import os
from timeit import default_timer

from utilities import *
import sys
from collections import OrderedDict


################################################################
# fourier layer
################################################################
class SpectralConv2d(nn.Module):
    def __init__(self, modes1, modes2):
        super().__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """
        self.modes1 = modes1  # number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.weights1 = nn.Parameter(torch.rand(self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(torch.rand(self.modes1, self.modes2, dtype=torch.cfloat))

    # complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x, y), (in_channel, out_channel, x, y) -> (batch, out_channel, x, y)
        return torch.einsum("bixy,xy->bixy", input, weights)

    def forward(self, x):
        batch_size, out_channels = x.shape[0], x.shape[1]
        # compute Fourier coefficients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)
        # multiply relevant Fourier modes
        out_ft = torch.zeros(batch_size, out_channels, x.size(-2), x.size(-1) // 2 + 1, dtype=torch.cfloat,
                             device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = torch.mul(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = torch.mul(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class NIPS(nn.Module):
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
        self.nvars = (ntoken + out_dim) // ngrid // ngrid

        modes = ngrid // 2 + 1

        self.convx = nn.ModuleList([nn.ModuleList([SpectralConv2d(modes, modes) for _ in range(head)]) for _ in range(nlayer)])
        self.convy = nn.ModuleList([nn.ModuleList([SpectralConv2d(modes, modes) for _ in range(head)]) for _ in range(nlayer)])
        self.convfx = nn.ModuleList([nn.ModuleList([SpectralConv2d(modes, modes) for _ in range(head)]) for _ in range(nlayer)])
        self.convfy = nn.ModuleList([nn.ModuleList([SpectralConv2d(modes, modes) for _ in range(head)]) for _ in range(nlayer)])

        grid = self.get_grid(ngrid, ngrid)
        self.dx = (grid[1, 0, 0] - grid[0, 0, 0])
        self.dy = (grid[0, 1, 1] - grid[0, 0, 1])

        self.fcq = nn.ModuleList([nn.ModuleList([nn.Linear(feature_dim, dk) for _ in range(head)]) for _ in range(nlayer)])
        self.fck = nn.ModuleList([nn.ModuleList([nn.Linear(feature_dim, dk) for _ in range(head)]) for _ in range(nlayer)])
        self.fcn = nn.ModuleList([nn.LayerNorm([ntoken + out_dim, feature_dim]) if i == 0 else nn.LayerNorm(feature_dim) for i in range(nlayer)])

    def forward(self, xy):
        batchsize = xy.shape[0]
        Vinput = self.fcn[0](xy)
        out_ft = torch.zeros((batchsize, self.out_dim, self.feature_dim), device=self.device)

        # intermediate layers
        for j in range(self.nlayer - 1):
            mid_ft = torch.zeros_like(Vinput)
            for i in range(self.head):
                v = Vinput.reshape(batchsize, self.nvars, self.ngrid, self.ngrid, self.feature_dim).permute(1, 0, 4, 2, 3)
                v = torch.cat((self.convx[j][i](v[0]).unsqueeze(2),
                               self.convy[j][i](v[1]).unsqueeze(2),
                               self.convfx[j][i](v[2]).unsqueeze(2),
                               self.convfy[j][i](v[3]).unsqueeze(2)
                               ), dim=2).reshape(batchsize, self.feature_dim, -1).permute(0, 2, 1)

                Q = self.fcq[j][i](v)
                K = self.fck[j][i](Vinput)
                KV = self.dx * self.dy * torch.matmul(K.transpose(1, 2), Vinput) * self.sqrt_dk_inv

                mid_ft += torch.matmul(Q, KV)
            Vinput = self.fcn[j + 1](mid_ft) + Vinput  # skip connection

        # final layer
        for i in range(self.head):
            v = Vinput.reshape(batchsize, self.nvars, self.ngrid, self.ngrid, self.feature_dim).permute(1, 0, 4, 2, 3)
            v = torch.cat((self.convx[-1][i](v[0]).unsqueeze(2) + self.convfx[-1][i](v[2]).unsqueeze(2),
                            self.convy[-1][i](v[1]).unsqueeze(2) + self.convfy[-1][i](v[3]).unsqueeze(2)
                           ), dim=2).reshape(batchsize, self.feature_dim, -1).permute(0, 2, 1)

            Q = self.fcq[-1][i](v)
            K = self.fck[-1][i](Vinput[:, :self.ntoken, :])
            KV = self.dx * self.dy * torch.matmul(K.transpose(1, 2), xy[:, :self.ntoken, :]) * self.sqrt_dk_inv

            out_ft += torch.matmul(Q, KV)

        return out_ft  # shape: batch size x out_dim x feature_dim

    def get_grid(self, size_x, size_y):
        gridx = torch.linspace(0, 1, size_x).reshape(size_x, 1, 1).repeat([1, size_y, 1])
        gridy = torch.linspace(0, 1, size_y).reshape(1, size_y, 1).repeat([size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=2)

    def compute_edges(self, grid):
        grid_i = grid.unsqueeze(1).expand(-1, self.ntoken, -1)
        grid_j = grid.unsqueeze(0).expand(self.ntoken, -1, -1)
        edges = torch.cat((grid_i, grid_j), dim=-1).to(self.device)
        return edges


def scheduler(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer


def LR_schedule(learning_rate, steps, scheduler_step, scheduler_gamma):
    # print(steps//scheduler_step)
    return learning_rate * np.power(scheduler_gamma, (steps // scheduler_step))


def data_augmentation_via_randperm(prefix, nsample, rands, sample_per_task, feature_dim, out_dim, sol_total, f_total):
    x_train = []
    f_train = []

    for t in range(nsample):
        for _ in range(rands):
            crand = torch.randperm(sample_per_task)
            x_sample = sol_total[prefix + t * sample_per_task + crand[:feature_dim]]  # [100, 21, 21]
            f_sample = torch.cat((f_total[prefix + t * sample_per_task + crand[:feature_dim], :, :, 0],
                                  f_total[prefix + t * sample_per_task + crand[:feature_dim], :, :, 1]),
                                 dim=1)
            x_train.append(x_sample.unsqueeze(0))  # [1, 100, 21, 21]
            f_train.append(f_sample.unsqueeze(0))

    x_train = torch.cat(x_train, dim=0).permute(0, 4, 2, 3, 1).reshape(nsample * rands, 2, out_dim, feature_dim)
    f_train = torch.cat(f_train, dim=0).permute(0, 2, 3, 1).reshape(nsample * rands, 2 * out_dim, feature_dim)
    return x_train, f_train


def main(i_seed, learning_rate, gamma, wd):

    torch.manual_seed(i_seed)
    torch.cuda.manual_seed(i_seed)
    np.random.seed(i_seed)

    # model setup
    head = 1
    nlayer = 4  # number of layers
    dk = 100  # size of WQ and WK
    # feature_dim = 100
    feature_dim = dk

    ngrid = 29
    out_dim = 2 * ngrid ** 2

    step_size = 100
    epochs = 500
    batch_size = 100

    task_num = 500
    ntrain = 450
    ntest = task_num - ntrain

    sample_per_task = 200
    rands = 100

    ntrain_total = ntrain * rands
    ntest_total = ntest * rands

    data_path = './data_mmnist/DATA_u_b_chi_100000x%dx%dxd.mat' % (ngrid, ngrid)
    reader = MatReader(data_path)
    # shuffle_index = torch.randperm(task_num)
    indis_index = []
    test_index = []
    img_per_digit = 50
    n_digits = 10
    for i in range(n_digits):
        indis_index.extend(list(range(i * img_per_digit, i * img_per_digit + img_per_digit - 5)))
        test_index.extend(list(range(i * img_per_digit + img_per_digit - 5, (i + 1) * img_per_digit)))
    indis_index.extend(test_index)

    sol_total = reader.read_field('u').reshape(task_num, sample_per_task, ngrid, ngrid, 2)[indis_index].reshape(task_num * sample_per_task, ngrid, ngrid, 2)
    f_total = reader.read_field('b').reshape(task_num, sample_per_task, ngrid, ngrid, 2)[indis_index].reshape(task_num * sample_per_task, ngrid, ngrid, 2)

    print(f'>> Randomly permuting data {rands} times..', flush=True)

    print(f'>> Generating training data..', flush=True)
    x_train, f_train = data_augmentation_via_randperm(0, ntrain, rands, sample_per_task, feature_dim, out_dim // 2, sol_total, f_total)
    x_train = x_train.reshape(ntrain_total, -1, feature_dim)
    xy_train = torch.cat((f_train, x_train), dim=1)

    print(f'>> Generating test data..', flush=True)
    x_test, f_test = data_augmentation_via_randperm(ntrain * sample_per_task, ntest, rands, sample_per_task, feature_dim, out_dim // 2, sol_total, f_total)
    x_test = x_test.reshape(ntest_total, -1, feature_dim)
    xy_test = torch.cat((f_test, x_test), dim=1)

    print(f'>> x Train and test data shape: {x_train.shape} and {x_test.shape}', flush=True)
    print(f'>> f Train and test data shape: {f_train.shape} and {f_test.shape}', flush=True)

    # x_normalizer = GaussianNormalizer(x_train)
    #    y_normalizer = GaussianNormalizer(y_train)
    # f_normalizer = GaussianNormalizer(f_train)
    # xy1_normalizer = GaussianNormalizer(xy1_train)
    # xy2_normalizer = GaussianNormalizer(xy2_train)
    #    x_train = x_normalizer.encode(x_train)
    #    y_train = y_normalizer.encode(y_train)
    #    f_train = f_normalizer.encode(f_train)
    #    xy1_train = xy1_normalizer.encode(xy1_train)
    #    xy2_train = xy2_normalizer.encode(xy2_train)
    #    x_test = x_normalizer.encode(x_test)
    #    y_test = y_normalizer.encode(y_test)
    #    f_test = f_normalizer.encode(f_test)
    #    xy1_test = xy1_normalizer.encode(xy1_test)
    #    xy2_test = xy2_normalizer.encode(xy2_test)

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, xy_train),
                                               batch_size=batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, xy_test),
                                              batch_size=batch_size, shuffle=False)


    base_dir = './checkpoints_mmnist_dk%d/NIPS_n%d_seed%d' % (dk, ntrain_total, seed)
    os.makedirs(base_dir, exist_ok=True)

    res_dir = './res_NIPS_mmnist_dk%d' % dk
    os.makedirs(res_dir, exist_ok=True)
    res_file = "%s/NIPS_l%d_mmnist_n%d_dk%d_ngrid%d.txt" % (res_dir, nlayer, ntrain_total, dk, ngrid)
    if not os.path.isfile(res_file):
        f = open(res_file, "w")
        f.write(f'ntrain_total, seed, lr, gamma, wd, train_lowest, train_best, test100, test200, test300, test400, '
                f'test, best_epoch, time (hrs)\n')
        f.close()

    myloss = LpLoss(size_average=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not torch.cuda.is_available():
        print(f'>> Device being used: {device}', flush=True)
    else:
        print(f'>> Device being used: {device} ({torch.cuda.get_device_name(0)})', flush=True)

    d = f_train.shape[1]
    model = NIPS(d, head, dk, nlayer, ngrid, feature_dim, out_dim, device).to(device)
    print(f'>> Total number of model params: {count_params(model)}', flush=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=wd)
    model_filename = '%s/NIPS_l%d_dk%d_lr%.1e_gamma%.2f_wd%.1e.ckpt' % (
        base_dir, nlayer, dk, learning_rate, gamma, wd)

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
        if ep + 1 > 99 and (ep + 1) % 100 == 0:
            test_loss_best_ep[ep + 1] = test_loss_best.item()

    t3 = default_timer()

    with open(res_file, 'a') as f:
        test_loss_best_str = ', '.join(map(str, test_loss_best_ep.values()))
        f.write(f'{ntrain_total}, {i_seed}, {learning_rate}, {gamma}, {wd}, {train_loss_lowest}, {train_loss_best}, '
                f'{test_loss_best_str}, {best_epoch}, {(t3 - t0) / 3600:.2f}\n')

if __name__ == '__main__':
    seed = 0

    lrs = [1e-2]
    gammas = [0.7]
    wds = [1e-8]

    # if len(sys.argv) > 1:
    #     lrs = [lrs[int(sys.argv[1])]]
    #     wds = [wds[int(sys.argv[2])]]

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
