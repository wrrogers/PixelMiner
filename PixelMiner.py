"""
PixelCNN++ implementation following https://github.com/openai/pixel-cnn/

References:
    1. Salimans, PixelCNN++ 2017
    2. van den Oord, Pixel Recurrent Neural Networks 2016a
    3. van den Oord, Conditional Image Generation with PixelCNN Decoders, 2016c
    4. Reed 2016 http://www.scottreed.info/files/iclr2017.pdf
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from tqdm import tqdm

from math import log10, sqrt

from parameters import Parameters

args = Parameters()

print('USING PIXELMINER VERSION FINAL v3!')

def PSNR(original, compressed, max=255):
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    #max_pixel = 255.0
    psnr = 20 * log10(max / sqrt(mse))
    return psnr

# --------------------
# Helper functions
# --------------------

def down_shift(x):
    out = F.pad(x, (0,0,1,0,0,0))[:,:,:,:-1,:]
    return out

def right_shift(x):
    return F.pad(x, (1,0,0,0))[:,:,:,:,:-1]

def concat_elu(x):
    return F.elu(torch.cat([x, -x], dim=1))

# --------------------
# Model components
# --------------------

class Conv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        nn.utils.weight_norm(self)

class Conv3d(nn.Conv3d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        nn.utils.weight_norm(self)

class ConvTranspose3d(nn.ConvTranspose3d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        nn.utils.weight_norm(self)

class DownShiftedConv3d(nn.Module):
    def __init__(self, n_channels, out_channels, kernel_size, stride=(1,1,1)):
        super().__init__()
        stride2d = stride[1:]
        self.conv2d = Conv2d(n_channels, out_channels, (3, 3), padding=(1,1), stride=stride2d)
        self.conv3d1 = DownShiftedConv3dSqueeze(n_channels, out_channels, kernel_size)
        self.conv3d2 = DownShiftedConv3dSqueeze(out_channels, out_channels, kernel_size, stride=stride)

    def forward(self, x):
        xt  = x[:, :, 0, :, :]
        xm1 = x[:, :, :2, :, :]
        xm2 = x[:, :, 1:, :, :]
        xb  = x[:, :, -1, :, :]

        xt  = self.conv2d(xt)
        xm1 = self.conv3d1(xm1)
        xm2 = self.conv3d1(xm2)
        xm  = torch.cat([xm1, xm2], 2)
        xm  = self.conv3d2(xm)
        xb  = self.conv2d(xb)
        xt = xt.unsqueeze(2)
        xb = xb.unsqueeze(2)
        x = torch.cat([xt, xm, xb], 2)
        return x

class DownRightShiftedConv3d(nn.Module):
    def __init__(self, n_channels, out_channels, kernel_size, stride=(1,1,1)):
        super().__init__()
        stride2d = stride[1:]
        self.conv2d = Conv2d(n_channels, out_channels, (3, 3), padding=(1,1), stride=stride2d)
        self.conv3d1 = DownRightShiftedConv3dSqueeze(n_channels, out_channels, kernel_size)
        self.conv3d2 = DownRightShiftedConv3dSqueeze(out_channels, out_channels, kernel_size, stride=stride)

    def forward(self, x):
        xt  = x[:, :, 0, :, :]
        xm1 = x[:, :, :2, :, :]
        xm2 = x[:, :, 1:, :, :]
        xb  = x[:, :, -1, :, :]

        xt  = self.conv2d(xt)
        xm1 = self.conv3d1(xm1)
        xm2 = self.conv3d1(xm2)
        xm  = torch.cat([xm1, xm2], 2)
        xm  = self.conv3d2(xm)
        xb  = self.conv2d(xb)

        xt = xt.unsqueeze(2)
        xb = xb.unsqueeze(2)
        x = torch.cat([xt, xm, xb], 2)
        return x

class DownShiftedConv3dSqueeze(Conv3d):
    def forward(self, x):
        # pad H above and W on each side
        Dk, Hk, Wk = self.kernel_size
        #print(x.size())
        x = F.pad(x, ((Wk-1)//2, (Wk-1)//2, Hk-1, 0))
        #print('Squeeze shape:', x.size())
        return super().forward(x)

class DownRightShiftedConv3dSqueeze(Conv3d):
    def forward(self, x):
        # pad above and on left (ie shift input down and right)
        Dk, Hk, Wk = self.kernel_size
        #print(x.size())
        x = F.pad(x, (Wk-1, 0, Hk-1, 0))
        #print('Squeeze shape:', x.size())
        return super().forward(x)

class DownShiftedConv3dold(nn.Module):
    def __init__(self, n_channels, out_channels, kernel_size, stride=(1,1,1)):
        super().__init__()
        self.conv = DownShiftedConv3dSqueeze(n_channels, out_channels, kernel_size, stride=stride)

    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.conv(x)
        x3 = self.conv(x)
        x = torch.cat([x1, x2, x3], 2)
        return x

class DownRightShiftedConv3dold(nn.Module):
    def __init__(self, n_channels, out_channels, kernel_size, stride=(1,1,1)):
        super().__init__()
        self.conv = DownRightShiftedConv3dSqueeze(n_channels, out_channels, kernel_size, stride=stride)

    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.conv(x)
        x3 = self.conv(x)
        x = torch.cat([x1, x2, x3], 2)
        return x

class DownShiftedConvTranspose3d(ConvTranspose3d):
    def forward(self, x):
        x = super().forward(x)
        _, _, Dout, Hout, Wout = x.shape
        Dk, Hk, Wk = self.kernel_size
        Ds, Hs, Ws = self.stride
        return x[:, :, :, :Hout-Hk+Hs, (Wk)//2: Wout]  # see pytorch doc for ConvTranspose output

class DownRightShiftedConvTranspose3d(ConvTranspose3d):
    def forward(self, x):
        x = super().forward(x)
        _, _, Dout, Hout, Wout = x.shape
        Dk, Hk, Wk = self.kernel_size
        Ds, Hs, Ws = self.stride
        return x[:, :, :, :Hout-Hk+Hs, :Wout-Wk+Ws]  # see pytorch doc for ConvTranspose output

class GatedResidualLayer(nn.Module):
    def __init__(self, conv, n_channels, kernel_size, drop_rate=0, shortcut_channels=None, n_cond_classes=None, relu_fn=concat_elu):
        super().__init__()
        self.relu_fn = relu_fn

        self.c1 = conv(2*n_channels, n_channels, kernel_size)
        if shortcut_channels:
            self.c1c = Conv3d(2*shortcut_channels, n_channels, kernel_size=1)

        if drop_rate > 0:
            self.dropout = nn.Dropout(drop_rate)
        self.c2 = conv(2*n_channels, 2*n_channels, kernel_size)
        if n_cond_classes:
            self.proj_h = nn.Linear(n_cond_classes, 2*n_channels)

    def forward(self, x, a=None, h=None):

        c1 = self.c1(self.relu_fn(x))

        if a is not None:  # shortcut connection if auxiliary input 'a' is given

            c1 = c1 + self.c1c(self.relu_fn(a))

        c1 = self.relu_fn(c1)
        if hasattr(self, 'dropout'):
            c1 = self.dropout(c1)
        c2 = self.c2(c1)
        if h is not None:
            c2 += self.proj_h(h)[:,:,None,None]
        a, b = c2.chunk(2,1)

        out = x + a * torch.sigmoid(b)
        return out

# --------------------
# PixelCNN
# --------------------

class PixelCNNpp(nn.Module):
    def __init__(self, image_dims=(1,3,64,64), n_channels=64, n_res_layers=1, n_logistic_mix=10, n_cond_classes=None, drop_rate=0.5):
        super().__init__()

        # input layers for `up` and `up and to the left` pixels
        self.u_input  = DownShiftedConv3d(image_dims[0]+1, n_channels, kernel_size=(2,2,3))

        self.ul_input_d = DownShiftedConv3d(image_dims[0]+1, n_channels, kernel_size=(2,1,3))
        self.ul_input_dr = DownRightShiftedConv3d(image_dims[0]+1, n_channels, kernel_size=(2,2,1))

        # up pass
        self.up_u_modules = nn.ModuleList([
            *[GatedResidualLayer(DownShiftedConv3d, n_channels, (2,2,3), drop_rate, None, n_cond_classes) for _ in range(n_res_layers)],
            DownShiftedConv3d(n_channels, n_channels, kernel_size=(2,2,3), stride=(1,2,2)),
            *[GatedResidualLayer(DownShiftedConv3d, n_channels, (2,2,3), drop_rate, None, n_cond_classes) for _ in range(n_res_layers)],
            DownShiftedConv3d(n_channels, n_channels, kernel_size=(2,2,3), stride=(1,2,2)),
            *[GatedResidualLayer(DownShiftedConv3d, n_channels, (2,2,3), drop_rate, None, n_cond_classes) for _ in range(n_res_layers)]])

        self.up_ul_modules = nn.ModuleList([
            *[GatedResidualLayer(DownRightShiftedConv3d, n_channels, (2,2,2), drop_rate, n_channels, n_cond_classes) for _ in range(n_res_layers)],
            DownRightShiftedConv3d(n_channels, n_channels, kernel_size=(2,2,2), stride=(1,2,2)),
            *[GatedResidualLayer(DownRightShiftedConv3d, n_channels, (2,2,2), drop_rate, n_channels, n_cond_classes) for _ in range(n_res_layers)],
            DownRightShiftedConv3d(n_channels, n_channels, kernel_size=(2,2,2), stride=(1,2,2)),
            *[GatedResidualLayer(DownRightShiftedConv3d, n_channels, (2,2,2), drop_rate, n_channels, n_cond_classes) for _ in range(n_res_layers)]])

        # down pass
        self.down_u_modules = nn.ModuleList([
            *[GatedResidualLayer(DownShiftedConv3d, n_channels, (2,2,3), drop_rate, n_channels, n_cond_classes) for _ in range(n_res_layers)],
            DownShiftedConvTranspose3d(n_channels, n_channels, kernel_size=(1,2,3), stride=(1,2,2)),
            *[GatedResidualLayer(DownShiftedConv3d, n_channels, (2,2,3), drop_rate, n_channels, n_cond_classes) for _ in range(n_res_layers+1)],
            DownShiftedConvTranspose3d(n_channels, n_channels, kernel_size=(1,2,3), stride=(1,2,2)),
            *[GatedResidualLayer(DownShiftedConv3d, n_channels, (2,2,3), drop_rate, n_channels, n_cond_classes) for _ in range(n_res_layers+1)]])

        self.down_ul_modules = nn.ModuleList([
            *[GatedResidualLayer(DownRightShiftedConv3d, n_channels, (2,2,2), drop_rate, 2*n_channels, n_cond_classes) for _ in range(n_res_layers)],
            DownRightShiftedConvTranspose3d(n_channels, n_channels, kernel_size=(1,2,2), stride=(1,2,2)),
            *[GatedResidualLayer(DownRightShiftedConv3d, n_channels, (2,2,2), drop_rate, 2*n_channels, n_cond_classes) for _ in range(n_res_layers+1)],
            DownRightShiftedConvTranspose3d(n_channels, n_channels, kernel_size=(1,2,2), stride=(1,2,2)),
            *[GatedResidualLayer(DownRightShiftedConv3d, n_channels, (2,2,2), drop_rate, 2*n_channels, n_cond_classes) for _ in range(n_res_layers+1)]])

        # output logistic mix params
        self.output_conv1 = Conv3d(n_channels, 3 * n_logistic_mix, kernel_size=1) # means, coefficients, logscales

    def forward(self, x, h=None):
        batch = x.size(0)
        height, width = x.size(-2), x.size(-1)
        # add channel of ones to distinguish image from padding later on

        x = F.pad(x, (0,0,0,0,0,0,0,1), value=1)

        # input layer
        u_list  = [down_shift(self.u_input(x))]

        ul_list = [down_shift(self.ul_input_d(x)) + right_shift(self.ul_input_dr(x))]

        # up pass
        for u_module, ul_module in zip(self.up_u_modules, self.up_ul_modules):
            #print('--------------------- ', 'Gated:', isinstance(u_module, GatedResidualLayer))
            u_list  += [u_module(u_list[-1], h=h) if isinstance(u_module, GatedResidualLayer) else u_module(u_list[-1])]
            #print('--------------------- ', 'Gated:', isinstance(ul_module, GatedResidualLayer))
            ul_list += [ul_module(ul_list[-1], u_list[-1], h)] if isinstance(ul_module, GatedResidualLayer) else [ul_module(ul_list[-1])]

        # down pass
        u = u_list.pop()
        ul = ul_list.pop()


        for n, (u_module, ul_module) in enumerate(zip(self.down_u_modules, self.down_ul_modules)):
            #print('--------------------- ', n, 'Gated:', isinstance(u_module, GatedResidualLayer))
            u  = u_module(u, u_list.pop(), h) if isinstance(u_module, GatedResidualLayer) else u_module(u)
            #print('--------------------- ', n, 'Gated:', isinstance(ul_module, GatedResidualLayer))
            ul = ul_module(u, torch.cat([u, ul_list.pop()],1), h) if isinstance(ul_module, GatedResidualLayer) else ul_module(ul)

        x = self.output_conv1(F.elu(ul))
        x = x.view(batch, -1, height, width)

        return x

# --------------------
# Loss functions
# --------------------

def log_sum_exp(x):
    """ numerically stable log_sum_exp implementation that prevents overflow """
    # TF ordering
    axis  = len(x.size()) - 1
    m, _  = torch.max(x, dim=axis)
    m2, _ = torch.max(x, dim=axis, keepdim=True)
    return m + torch.log(torch.sum(torch.exp(x - m2), dim=axis))


def log_prob_from_logits(x):
    """ numerically stable log_softmax implementation that prevents overflow """
    # TF ordering
    axis = len(x.size()) - 1
    m, _ = torch.max(x, dim=axis, keepdim=True)
    return x - m - torch.log(torch.sum(torch.exp(x - m), dim=axis, keepdim=True))

def to_one_hot(tensor, n, fill_with=1.):
    # we perform one hot encore with respect to the last axis
    one_hot = torch.FloatTensor(tensor.size() + (n,)).zero_()
    if tensor.is_cuda : one_hot = one_hot.cuda()
    one_hot.scatter_(len(tensor.size()), tensor.unsqueeze(-1), fill_with)
    return Variable(one_hot)

def discretized_mix_logistic_loss_1d(x, l):
    """ log-likelihood for mixture of discretized logistics, assumes the data has been rescaled to [-1,1] interval """
    x = x[:, :, 1].squeeze().unsqueeze(1)
    l = l.squeeze()

    # Pytorch ordering
    x = x.permute(0, 2, 3, 1)
    l = l.permute(0, 2, 3, 1)
    xs = [int(y) for y in x.size()]
    ls = [int(y) for y in l.size()]

    # here and below: unpacking the params of the mixture of logistics
    nr_mix = int(ls[-1] / 3)
    #nr_mix = args.n_logistic_mix

    logit_probs = l[:, :, :, :nr_mix]
    l = l[:, :, :, nr_mix:].contiguous().view(xs + [nr_mix * 2]) # 2 for mean, scale
    means = l[:, :, :, :, :nr_mix]
    log_scales = torch.clamp(l[:, :, :, :, nr_mix:2 * nr_mix], min=-7.)
    # here and below: getting the means and adjusting them based on preceding
    # sub-pixels
    x = x.contiguous()
    x = x.unsqueeze(-1) + Variable(torch.zeros(xs + [nr_mix]).cuda(), requires_grad=False)

    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1. / 255.)
    cdf_plus = torch.sigmoid(plus_in)
    min_in = inv_stdv * (centered_x - 1. / 255.)
    cdf_min = torch.sigmoid(min_in)
    # log probability for edge case of 0 (before scaling)
    log_cdf_plus = plus_in - F.softplus(plus_in)
    # log probability for edge case of 255 (before scaling)
    log_one_minus_cdf_min = -F.softplus(min_in)
    cdf_delta = cdf_plus - cdf_min  # probability for all other cases
    mid_in = inv_stdv * centered_x
    # log probability in the center of the bin, to be used in extreme cases
    # (not actually used in our code)
    log_pdf_mid = mid_in - log_scales - 2. * F.softplus(mid_in)

    inner_inner_cond = (cdf_delta > 1e-5).float()
    inner_inner_out  = inner_inner_cond * torch.log(torch.clamp(cdf_delta, min=1e-12)) + (1. - inner_inner_cond) * (log_pdf_mid - np.log(127.5))
    inner_cond       = (x > 0.999).float()
    inner_out        = inner_cond * log_one_minus_cdf_min + (1. - inner_cond) * inner_inner_out
    cond             = (x < -0.999).float()
    log_probs        = cond * log_cdf_plus + (1. - cond) * inner_out
    log_probs        = torch.sum(log_probs, dim=3) + log_prob_from_logits(logit_probs)

    out = -torch.sum(log_sum_exp(log_probs))

    return out


loss_fn = discretized_mix_logistic_loss_1d

# --------------------
# Sampling and generation functions
# --------------------

def sample_from_discretized_mix_logistic_1d(l):
    # Pytorch ordering
    l = l.permute(0, 2, 3, 1)
    ls = [int(y) for y in l.size()]
    xs = ls[:-1] + [1] #[3]

    nr_mix = int(ls[-1] / 3)

    # unpack parameters
    logit_probs = l[:, :, :, :nr_mix]

    l = l[:, :, :, nr_mix:].contiguous().view(xs + [nr_mix * 2]) # for mean, scale

    # sample mixture indicator from softmax
    temp = torch.FloatTensor(logit_probs.size())
    if l.is_cuda : temp = temp.cuda()
    temp.uniform_(1e-5, 1. - 1e-5)
    temp = logit_probs.data - torch.log(- torch.log(temp))
    _, argmax = temp.max(dim=3)

    one_hot = to_one_hot(argmax, nr_mix)
    sel = one_hot.view(xs[:-1] + [1, nr_mix])
    # select logistic parameters
    means = torch.sum(l[:, :, :, :, :nr_mix] * sel, dim=4)
    log_scales = torch.clamp(torch.sum(
        l[:, :, :, :, nr_mix:2 * nr_mix] * sel, dim=4), min=-7.)
    u = torch.FloatTensor(means.size())
    if l.is_cuda : u = u.cuda()
    u.uniform_(1e-5, 1. - 1e-5)
    u = Variable(u)
    x = means + torch.exp(log_scales) * (torch.log(u) - torch.log(1. - u))
    x0 = torch.clamp(torch.clamp(x[:, :, :, 0], min=-1.), max=1.)
    out = x0.unsqueeze(1)
    return out

def generate_fn(model, data_loader, n_samples, image_dims, device, h=None):
    out = next(iter(data_loader))
    print("The generate size is:", out.size())
    out = out[:args.n_samples].to(device)
    out[:, 2, :, :] = torch.zeros(128,128)
    with tqdm(total=(image_dims[1]*image_dims[2]), desc='Generating {} images'.format(out.size(0))) as pbar:
        for yi in range(image_dims[1]):
            if yi < args.ymin:
                continue
            for xi in range(image_dims[2]):
                logits = model(out, h)
                sample = sample_from_discretized_mix_logistic_1d(logits, image_dims)[:,:,yi,xi]
                out[:,2,yi,xi] = sample[:,2]

                pbar.update()
    return out


if __name__ == '__main__':
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    print('Visible Device:', os.environ['CUDA_VISIBLE_DEVICES'])
    x = torch.zeros((2,1,3,64,64)).cuda()
    model = PixelCNNpp().cuda()
    l = model(x, None)
    print(l.size())
    loss = discretized_mix_logistic_loss_1d(x, l)
    print(loss)
