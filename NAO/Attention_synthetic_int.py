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
from utilities_synthetic import *
import sys

from d2l import torch as d2l

seed = 0 #int(sys.argv[1])

torch.manual_seed(seed)
np.random.seed(seed)


def scheduler(optimizer,lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer
def LR_schedule(learning_rate,steps,scheduler_step,scheduler_gamma):
    #print(steps//scheduler_step)
    return learning_rate*np.power(scheduler_gamma,(steps//scheduler_step))

base_dir = './log/attention_sine_fo/meta_FO_torch_seed%d' %(seed)
if not os.path.exists(base_dir):
    os.makedirs(base_dir)

myloss = LpLoss(size_average=False)

#implement the model
class MS_Loss(nn.Module):

    def __init__(self, tokenN, head, dk, nlayer, feature_dim):
        super(MS_Loss, self).__init__()

        self.feature_dim = feature_dim #number of data pairs (denoted as d in paper) in each sample
        self.tokenN = tokenN #number of tokens (denoted as N in paper)
        self.head = head #number of total heads
        self.dk = dk #dimension (maximum rank) for Q and K
        self.nlayer = nlayer #number of iterative layers

        # to generate continuous W^P
        self.kernel = DenseNet([1, 32, 64, 1], torch.nn.LeakyReLU)
        self.kernelf = nn.Parameter(torch.randn(1).double().cpu(), requires_grad=True)
        # to generate normalization, to enable cross-resolution generalization
        self.fnw = DenseNet([1, 32, 64, 1], torch.nn.LeakyReLU)
        self.fnb = DenseNet([1, 32, 64, 1], torch.nn.LeakyReLU)
        self.gamma = torch.nn.Parameter(torch.randn(1).double().cpu(), requires_grad=True)
        self.beta = torch.nn.Parameter(torch.randn(1).double().cpu(), requires_grad=True)

        for i in range(self.head):
            for j in range(self.nlayer):
                self.add_module('fcq_%d_%d' % (j,i), nn.Linear(feature_dim, self.dk).double())
                self.add_module('fck_%d_%d' % (j,i), nn.Linear(feature_dim, self.dk).double())

    def forward(self, xy, P, delta):

        feature_dim = self.feature_dim
        m=nn.Identity()
        batchsize = xy.shape[0]
        grid = self.get_grid(batchsize, self.tokenN, xy.device)
        # for W^P,u in the paper
        weight = self.kernel(grid).view(batchsize, self.tokenN, 1)
        dx = (grid[0,1,0]-grid[0,0,0])

        # for a continuous normalizer
        fn_weight = torch.cat((self.fnw(grid).view(batchsize, self.tokenN, 1),self.gamma*torch.ones(batchsize,1,1).double()),1)
        fn_bias = torch.cat((self.fnb(grid).view(batchsize, self.tokenN, 1),self.beta*torch.ones(batchsize,1,1).double()),1)

        Vinput = xy + torch.kron(torch.ones(batchsize,1,1),P[:, :xy.shape[1], :]) #can use positional encoding. Here P=0 (positional encoding disabled)
        out_ft = torch.zeros((batchsize, feature_dim), dtype=torch.double, device=x.device)
        for j in range(self.nlayer-1):
            mid_ft = torch.zeros((batchsize, self.tokenN+1, feature_dim), dtype=torch.double, device=x.device)
            for i in range(self.head):
                Q = self._modules['fcq_%d_%d' % (j,i)](Vinput)
                K = self._modules['fck_%d_%d' % (j,i)](Vinput)
                Attn = m(torch.matmul(Q,torch.transpose(K,1,2))/torch.sqrt(torch.tensor(self.dk/1., dtype=torch.double)))
                V = dx*(torch.matmul(Attn[:,:,:],Vinput[:,:,:]))#.squeeze(2)
                mid_ft = mid_ft + V
            mid_ft_mean = torch.mean(mid_ft,1).reshape(batchsize, 1, feature_dim).repeat([1, self.tokenN+1, 1])
            Vinput = torch.addcmul(fn_bias, (mid_ft - mid_ft_mean),fn_weight) + Vinput

        for i in range(self.head):
            Q = self._modules['fcq_%d_%d' % (self.nlayer-1, i)](Vinput)
            K = self._modules['fck_%d_%d' % (self.nlayer-1, i)](Vinput)
            Attn = m(torch.matmul(Q, torch.transpose(K, 1, 2)) / torch.sqrt(torch.tensor(self.dk / 1., dtype=torch.double)))
            Vu = dx*(torch.matmul(Attn[:, :self.tokenN, :self.tokenN], xy[:, :self.tokenN, :]))  # Vu shape: batch size X tokenN X feature_dim
            Vu = Vu.permute(0,2,1)  # Vu shape: batch size X feature_dim X tokenN
            Vf = dx*(torch.matmul(Attn[:, self.tokenN:, :self.tokenN], xy[:, :self.tokenN, :])) # Vf shape: batch size X 1 X feature_dim
            out_ft += dx*torch.matmul(Vu, weight).squeeze(2) #contribution from self attention
            out_ft += (Vf.squeeze(1))*self.kernelf #contribution from cross attention


        return out_ft

    #generate uniform mesh on [-1,1], for normalizer
    def get_grid(self, batchsize, size_x, device):
        gridx = torch.tensor(np.linspace(-1, 1, size_x), dtype=torch.double)
        gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
        return gridx.to(device)



head = 1 # number of heads (in multihead attention model)
dk = 20 # size of WQ and WK
delta = 11.005 # boudary width
nlayer=2 # number of layers

dx = 0.0125 # resolution 0.0125 0.025 0.05 0.1
bdry_width =np.floor(delta/dx)
kernel = 'sine' # kernel function type :sine,gaussian
task = "task_exp_amplitude_"+str(dx)

batch_size = 10
learning_rate = 0.003
epochs = 20000
step_size = 100
gamma = 0.9
max_len = 1000
skip = 302
feature_dim = skip

#def get_data_all_sine_k(delta,kernel,dx,task):
E_2 = ['1','2','3','4','6','7','8']
E_cos = ['0', '1','2','3','4', '5' ,'6']
E_poly = ['1','2','3','4', '5' ,'6','7']
train_data = []
x_train_data_normalized = []
y_train_data_normalized = []
x_test_data = []
y_test_data = []
X_list = []
Y_list = []

# read all meta train tasks and generate the normalizer
norm_type = 'sample'
for k in E_2:
    X = torch.from_numpy(np.load('./synthetic_data/sample_Xdata_sine_meta_dx_con_'+ k + '.npy'))
    Y = torch.from_numpy(np.load('./synthetic_data/sample_Ydata_sine_meta_dx_con_'+ k + '.npy'))
    tts = int(X.shape[0]/skip)
    for i in range(tts):
        X_sample = X[i:i+tts*skip:tts,:]
        X_sample = X_sample.permute(1, 0)
        Y_sample = Y[i:i+tts*skip:tts,:]
        Y_sample = Y_sample.permute(1, 0)
        X_list.append(X_sample.unsqueeze(0))
        Y_list.append(Y_sample.unsqueeze(0))

##############################
# uncomment to load diverse data
##############################
# for k in E_cos:
#     X = torch.from_numpy(np.load('./synthetic_data/sample_Xdata_cos_meta_dx_con_' + k + '_decay.npy'))
#     Y = torch.from_numpy(np.load('./synthetic_data/sample_Ydata_cos_meta_dx_con_' + k + '_decay.npy'))
#     tts = int(X.shape[0]/skip)
#     for i in range(tts):
#         X_sample = X[i:i+tts*skip:tts,:]
#         X_sample = X_sample.permute(1, 0)
#         Y_sample = Y[i:i+tts*skip:tts,:]
#         Y_sample = Y_sample.permute(1, 0)
#         X_list.append(X_sample.unsqueeze(0))
#         Y_list.append(Y_sample.unsqueeze(0))
#
# for k in E_poly:
#     X = torch.from_numpy(np.load('./synthetic_data/sample_Xdata_poly_meta_dx_con_' + k + '_decay.npy'))
#     Y = torch.from_numpy(np.load('./synthetic_data/sample_Ydata_poly_meta_dx_con_' + k + '_decay.npy'))
#     tts = int(X.shape[0]/skip)
#     for i in range(tts):
#         X_sample = X[i:i+tts*skip:tts,:]
#         X_sample = X_sample.permute(1, 0)
#         Y_sample = Y[i:i+tts*skip:tts,:]
#         Y_sample = Y_sample.permute(1, 0)
#         X_list.append(X_sample.unsqueeze(0))
#         Y_list.append(Y_sample.unsqueeze(0))


all_X = torch.cat(X_list, axis=0)
all_Y = torch.cat(Y_list, axis=0)

x_normalizer = GaussianNormalizer(all_X)
y_normalizer = GaussianNormalizer(all_Y)

# read all meta train tasks and with unit gaussian normalizer to normalized the data
for k in E_2:
    X_train = torch.from_numpy(np.load('./synthetic_data/sample_Xdata_sine_meta_dx_con_'+ k + '.npy'))
    Y_train = torch.from_numpy(np.load('./synthetic_data/sample_Ydata_sine_meta_dx_con_'+ k + '.npy'))
    tts = int(X_train.shape[0]/skip)
    for i in range(tts):
        X_sample = X_train[i:i+tts*skip:tts,:]
        X_sample = X_sample.permute(1, 0)
        Y_sample = Y_train[i:i+tts*skip:tts,:]
        Y_sample = Y_sample.permute(1, 0)
        X = x_normalizer.encode(X_sample)
        Y = y_normalizer.encode(Y_sample)
        x_train_data_normalized.append(X.unsqueeze(0))
        y_train_data_normalized.append(Y.unsqueeze(0))

##############################
# uncomment to load diverse data
##############################
# for k in E_cos:
#     X = torch.from_numpy(np.load('./synthetic_data/sample_Xdata_cos_meta_dx_con_' + k + '_decay.npy'))
#     Y = torch.from_numpy(np.load('./synthetic_data/sample_Ydata_cos_meta_dx_con_' + k + '_decay.npy'))
#     tts = int(X_train.shape[0]/skip)
#     for i in range(tts):
#         X_sample = X_train[i:i+tts*skip:tts,:]
#         X_sample = X_sample.permute(1, 0)
#         Y_sample = Y_train[i:i+tts*skip:tts,:]
#         Y_sample = Y_sample.permute(1, 0)
#         X = x_normalizer.encode(X_sample)
#         Y = y_normalizer.encode(Y_sample)
#         x_train_data_normalized.append(X.unsqueeze(0))
#         y_train_data_normalized.append(Y.unsqueeze(0))
#
# for k in E_poly:
#     X = torch.from_numpy(np.load('./synthetic_data/sample_Xdata_poly_meta_dx_con_' + k + '_decay.npy'))
#     Y = torch.from_numpy(np.load('./synthetic_data/sample_Ydata_poly_meta_dx_con_' + k + '_decay.npy'))
#     tts = int(X_train.shape[0]/skip)
#     for i in range(tts):
#         X_sample = X_train[i:i+tts*skip:tts,:]
#         X_sample = X_sample.permute(1, 0)
#         Y_sample = Y_train[i:i+tts*skip:tts,:]
#         Y_sample = Y_sample.permute(1, 0)
#         X = x_normalizer.encode(X_sample)
#         Y = y_normalizer.encode(Y_sample)
#         x_train_data_normalized.append(X.unsqueeze(0))
#         y_train_data_normalized.append(Y.unsqueeze(0))

X_train = torch.cat(x_train_data_normalized,axis=0)
Y_train = torch.cat(y_train_data_normalized,axis=0)
XY_train = torch.cat([X_train,Y_train],axis=1)


k = '5'
X_test= torch.from_numpy(np.load('./synthetic_data/sample_Xdata_sine_meta_dx_con_'+ k + '.npy'))
Y_test = torch.from_numpy(np.load('./synthetic_data/sample_Ydata_sine_meta_dx_con_'+ k + '.npy'))
tts = int(X_test.shape[0] / skip)
for i in range(tts):
    X_sample = X_test[i:i+tts*skip:tts, :]
    X_sample = X_sample.permute(1, 0)
    Y_sample = Y_test[i:i+tts*skip:tts, :]
    Y_sample = Y_sample.permute(1, 0)
    X = x_normalizer.encode(X_sample)
    Y = y_normalizer.encode(Y_sample)
    x_test_data.append(X.unsqueeze(0))
    y_test_data.append(Y.unsqueeze(0))

X_test = torch.cat(x_test_data,axis=0)
Y_test = torch.cat(y_test_data,axis=0)
XY_test = torch.cat([X_test,Y_test],axis=1)


train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_train, Y_train, XY_train), batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_test, Y_test, XY_test), batch_size=batch_size, shuffle=False)


#y_normalizer.cuda()
total_step = 0
test_l2 = 0.0
ntrain = X_train.shape[0]
tokenN = X_train.shape[1]
ntest = X_test.shape[0]

model = MS_Loss(tokenN, head, dk, nlayer, feature_dim)#.cuda()
print(count_params(model))

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
model_filename = '%s/FO_depth%d_dk%d_r%d_softmax.ckpt' % (base_dir, nlayer, dk, head)
#model.load_state_dict(torch.load(model_filename))

P = torch.zeros((1, max_len, feature_dim))
#yyX = torch.arange(max_len, dtype=torch.double).reshape(-1, 1) / torch.pow(10000, torch.arange(0, feature_dim, 2, dtype=torch.double) / feature_dim)
#P[:, :, 0::2] = torch.sin(yyX)
#P[:, :, 1::2] = torch.cos(yyX)

train_loss_min = 100
for ep in range(epochs):
    optimizer = scheduler(optimizer, LR_schedule(learning_rate, ep, step_size, gamma))
    model.train()
    t1 = default_timer()
    train_l2 = 0
    for x, y, xy in train_loader:
#        x, y = x.cuda(), y.cuda()
        this_batch_size = xy.shape[0]
        optimizer.zero_grad()
        out = model(xy,P,delta).reshape(this_batch_size, 1, feature_dim)
        out = y_normalizer.decode(out)
        y = y_normalizer.decode(y)

        loss = myloss(out.view(this_batch_size, -1), y.view(this_batch_size, -1))
        loss.backward()

        optimizer.step()
        train_l2 += loss.item()

    model.eval()
    if train_loss_min > train_l2 / ntrain:
        train_loss_min = train_l2 / ntrain
        torch.save(model.state_dict(), model_filename)
        test_l2 = 0.0
        with torch.no_grad():
            for x, y, xy in test_loader:
#                x, y = x.cuda(), y.cuda()
                this_batch_size = xy.shape[0]
                out = model(xy,P,delta).reshape(this_batch_size, 1, feature_dim)
                out = y_normalizer.decode(out)
                y = y_normalizer.decode(y)

                test_l2 += myloss(out.view(this_batch_size, -1), y.view(this_batch_size, -1)).item()
            test_l2 /= ntest

    train_l2 /= ntrain

    t2 = default_timer()
    print(
        'depth: %d epochs: [%d/%d]  running time:%.3f  current training error: %f best training error: %f best test error: %f' % (
        nlayer, ep, epochs, t2 - t1, train_l2, train_loss_min, test_l2))
    total_step += 1

