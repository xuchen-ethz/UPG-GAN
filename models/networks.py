import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.autograd import Variable
from torch.optim import lr_scheduler
import torch.nn.functional as F # for cross entropy loss
import numpy as np
from torch.nn.modules.utils import _pair, _quadruple
###############################################################################
# Functions
###############################################################################

def weights_init_normal(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.xavier_normal(m.weight.data, gain=0.02)
    elif classname.find('Linear') != -1:
        init.xavier_normal(m.weight.data, gain=0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    print(classname)
    if classname.find('Conv') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def init_weights(net, init_type='normal'):
    print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=True)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

def get_non_linearity(layer_type='relu'):
    if layer_type == 'relu':
        nl_layer = functools.partial(nn.ReLU, inplace=True)
    elif layer_type == 'lrelu':
        nl_layer = functools.partial(nn.LeakyReLU, negative_slope=0.2, inplace=True)
    elif layer_type == 'elu':
        nl_layer = functools.partial(nn.ELU, inplace=True)
    else:
        raise NotImplementedError('nonlinearity activitation [%s] is not found' % layer_type)
    return nl_layer

def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def define_G(input_nc, output_nc, ngf, which_model_netG, norm='batch', use_dropout=False, init_type='normal', gpu_ids=[], nz=0, upsample='basic', tanh=True):
    netG = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert(torch.cuda.is_available())

    if which_model_netG == 'resnet_9blocks':
        netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9, gpu_ids=gpu_ids)
    elif which_model_netG == 'resnet_6blocks':
        netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6, gpu_ids=gpu_ids, tanh=tanh)
    elif which_model_netG == 'unet_128':
        netG = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout, gpu_ids=gpu_ids, upsample=upsample)
    elif which_model_netG == 'unet_256':
        netG = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout, gpu_ids=gpu_ids, upsample=upsample)
    elif which_model_netG == 'resnet_6blocks_all':
        netG = G_Resnet_add_all(input_nc, output_nc, nz, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6, gpu_ids=gpu_ids, upsample=upsample)
    elif which_model_netG == 'resnet_9blocks_all':
        netG = G_Resnet_add_all(input_nc, output_nc, nz, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9, gpu_ids=gpu_ids, upsample=upsample)
    elif which_model_netG == 'unet_128_all':
        netG = G_Unet_add_all(input_nc, output_nc, nz, 7, ngf, norm_layer=norm_layer, nl_layer=get_non_linearity(layer_type='relu'),
                              use_dropout=use_dropout, gpu_ids=gpu_ids, upsample=upsample)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % which_model_netG)
    if len(gpu_ids) > 0:
        netG.cuda(gpu_ids[0])
    init_weights(netG, init_type=init_type)
    return netG


def define_D(input_nc, ndf, which_model_netD,
             n_layers_D=3, norm='batch', use_sigmoid=False, init_type='normal', gpu_ids=[]):
    netD = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert(torch.cuda.is_available())
    if which_model_netD == 'basic':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)
    elif which_model_netD == 'n_layers':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)
    elif which_model_netD == 'pixel':
        netD = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)
    elif which_model_netD == 'wgan_critic':
        netD = WassersteinGANCritic(input_nc, conv_output_dim=2, gpu_ids=gpu_ids)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' %
                                  which_model_netD)
    if use_gpu:
        netD.cuda(gpu_ids[0])
    init_weights(netD, init_type=init_type)
    return netD

def define_AE(input_nc, nz, ngf, norm='batch', use_dropout=False, init_type='normal', gpu_ids=[]):
    netAE = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)
    nl = 'lrelu'  # use leaky relu for E
    nl_layer = get_non_linearity(layer_type=nl)
    if use_gpu:
        assert(torch.cuda.is_available())

    # netAE = AutoEncoder(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, gpu_ids=gpu_ids)
    netAE = AE_NLayers(input_nc, nz, ngf, nl_layer=nl_layer, norm_layer=norm_layer, gpu_ids=gpu_ids)

    if len(gpu_ids) > 0:
        netAE.cuda(gpu_ids[0])
    init_weights(netAE, init_type=init_type)
    return netAE

def define_E(input_nc, output_nc, ndf, which_model_netE,
             norm='batch', nl='lrelu',
             init_type='xavier', gpu_ids=[], vaeLike=False, pooling='max'):
    netE = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)
    nl = 'lrelu'  # use leaky relu for E
    nl_layer = get_non_linearity(layer_type=nl)
    if use_gpu:
        assert(torch.cuda.is_available())
    if which_model_netE == 'resnet_128':
        netE = E_ResNet(input_nc, output_nc, ndf, n_blocks=4, norm_layer=norm_layer, nl_layer=nl_layer, gpu_ids=gpu_ids, vaeLike=vaeLike, pooling=pooling)
    elif which_model_netE == 'resnet_256':
        netE = E_ResNet(input_nc, output_nc, ndf, n_blocks=5, norm_layer=norm_layer, nl_layer=nl_layer, gpu_ids=gpu_ids, vaeLike=vaeLike, pooling=pooling)
    elif which_model_netE == 'conv_128':
        netE = E_NLayers(input_nc, output_nc, ndf, n_layers=4, norm_layer=norm_layer, nl_layer=nl_layer, gpu_ids=gpu_ids, vaeLike=vaeLike, pooling=pooling)
    elif which_model_netE == 'conv_256':
        netE = E_NLayers(input_nc, output_nc, ndf, n_layers=5, norm_layer=norm_layer, nl_layer=nl_layer, gpu_ids=gpu_ids, vaeLike=vaeLike, pooling=pooling)
    else:
        raise NotImplementedError('Encoder model name [%s] is not recognized' % which_model_netE)
    if use_gpu:
        netE.cuda(gpu_ids[0])
    init_weights(netE, init_type=init_type)
    return netE

def define_E(input_nc, output_nc, ndf, which_model_netE,
             norm='batch', nl='lrelu',
             init_type='xavier', gpu_ids=[], vaeLike=False, pooling='max'):
    netE = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)
    nl = 'lrelu'  # use leaky relu for E
    nl_layer = get_non_linearity(layer_type=nl)
    if use_gpu:
        assert(torch.cuda.is_available())
    if which_model_netE == 'resnet_128':
        netE = E_ResNet(input_nc, output_nc, ndf, n_blocks=4, norm_layer=norm_layer, nl_layer=nl_layer, gpu_ids=gpu_ids, vaeLike=vaeLike, pooling=pooling)
    elif which_model_netE == 'resnet_256':
        netE = E_ResNet(input_nc, output_nc, ndf, n_blocks=5, norm_layer=norm_layer, nl_layer=nl_layer, gpu_ids=gpu_ids, vaeLike=vaeLike, pooling=pooling)
    elif which_model_netE == 'resnet_full':
        netE = E_ResNet(input_nc, output_nc, ndf, n_blocks=7, norm_layer=norm_layer, nl_layer=nl_layer, gpu_ids=gpu_ids, vaeLike=vaeLike, pooling=pooling)
    elif which_model_netE == 'conv_128':
        netE = E_NLayers(input_nc, output_nc, ndf, n_layers=4, norm_layer=norm_layer, nl_layer=nl_layer, gpu_ids=gpu_ids, vaeLike=vaeLike, pooling=pooling)
    elif which_model_netE == 'conv_256':
        netE = E_NLayers(input_nc, output_nc, ndf, n_layers=5, norm_layer=norm_layer, nl_layer=nl_layer, gpu_ids=gpu_ids, vaeLike=vaeLike, pooling=pooling)
    else:
        raise NotImplementedError('Encoder model name [%s] is not recognized' % which_model_netE)
    if use_gpu:
        netE.cuda(gpu_ids[0])
    init_weights(netE, init_type=init_type)
    return netE

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


##############################################################################
# Classes
##############################################################################


# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)
# Defines the total variation loss
class TVLoss(nn.Module):
    def __init__(self,  tensor=torch.FloatTensor):
        super(TVLoss, self).__init__()

    def forward(self, input):

        w_variance = torch.mean((input[:, :, :, 1:] - input[:, :, :, :-1]) ** 2)
        h_variance = torch.mean((input[:, :, 1:, :] - input[:, :, :-1, :]) ** 2)
        loss = (w_variance + h_variance)
        return loss
# Defines the 2d cross entropy loss
# Copied from
# https://github.com/ycszen/pytorch-seg/blob/master/loss.py
class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True, softmax=True):
        super(CrossEntropyLoss2d, self).__init__()

        self.nll_loss = nn.NLLLoss2d(weight, size_average)
        self.softmax = softmax

    def forward(self, inputs, targets):
        if self.softmax:
            return self.nll_loss(F.log_softmax(inputs), targets)
        else:
            return self.nll_loss(F.logsigmoid(inputs), targets)
# Defines the generator that consists of Resnet blocks between a few
# downsampling/upsampling operations.
# Code and idea originally from Justin Johnson's architecture.
# https://github.com/jcjohnson/fast-neural-style/
class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, gpu_ids=[], padding_type='reflect', tanh=True):
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                           bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        if tanh:
            model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)

# Define the total variation loss
# copy from https://zhuanlan.zhihu.com/p/31421408

# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout2d(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[], upsample='basic'):
        super(UnetGenerator, self).__init__()
        self.gpu_ids = gpu_ids

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True, upsample=upsample)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout, upsample=upsample)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, upsample=upsample)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer, upsample=upsample)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer, upsample=upsample)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer, upsample=upsample)

        self.model = unet_block

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)


# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False, upsample='basic'):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = upsampleLayer(inner_nc*2, outer_nc, upsample=upsample)
            down = [downconv]
            up = [uprelu]  + upconv + [nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = upsampleLayer(inner_nc, outer_nc, upsample=upsample)
            down = [downrelu, downconv]
            up = [uprelu] + upconv + [upnorm]
            model = down + up
        else:
            upconv = upsampleLayer(inner_nc*2, outer_nc, upsample=upsample)
            down = [downrelu, downconv, downnorm]
            up = [uprelu] + upconv + [upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, gpu_ids=[]):
        super(NLayerDiscriminator, self).__init__()
        self.gpu_ids = gpu_ids
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        if len(self.gpu_ids) and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)

class PixelDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d, use_sigmoid=False, gpu_ids=[]):
        super(PixelDiscriminator, self).__init__()
        self.gpu_ids = gpu_ids
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
            
        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        if use_sigmoid:
            self.net.append(nn.Sigmoid())

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        if len(self.gpu_ids) and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.net, input, self.gpu_ids)
        else:
            return self.net(input)


class AutoEncoder(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[], padding_type='reflect'):
        # assert(n_blocks >= 0)
        super(AutoEncoder, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        encoder = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                           bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 6
        for i in range(n_downsampling-1):
            mult = 2**i
            encoder += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        encoder += [nn.Conv2d(ngf * mult*2, 2, kernel_size=3,
                              stride=2, padding=1, bias=use_bias),
                    norm_layer(2),
                    nn.ReLU(True)]

        decoder = [nn.ConvTranspose2d(2, ngf * mult * 2,
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        for i in range(1, n_downsampling):
            mult = 2**(n_downsampling - i)
            decoder += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        decoder += [nn.ReflectionPad2d(3)]
        decoder += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        decoder += [nn.Tanh()]

        self.encoder = nn.Sequential(*encoder)
        self.decoder = nn.Sequential(*decoder)

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            code = nn.parallel.data_parallel(self.encoder, input, self.gpu_ids)
            output = nn.parallel.data_parallel(self.decoder, code, self.gpu_ids)

        else:
            code = self.encoder(input)
            output = self.decoder(code)

        return code, output

class AE_NLayers(nn.Module):
    def __init__(self, input_nc, output_nc=1, ndf=64, n_layers=7,
                 norm_layer=None, nl_layer=None, gpu_ids=[], vaeLike=False):
        super(AE_NLayers, self).__init__()
        self.gpu_ids = gpu_ids

        kw, padw = 4, 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nl_layer()]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 4)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw)]
            if norm_layer is not None  and n < 6:
                sequence += [norm_layer(ndf * nf_mult)]
            sequence += [nl_layer()]
        # sequence += [nn.AvgPool2d(8)]
        self.conv = nn.Sequential(*sequence)
        self.fc = nn.Sequential(*[nn.Linear(ndf * nf_mult, output_nc), nn.LeakyReLU(0.2, True)])
        self.fc2 = nn.Sequential(*[nn.Linear(output_nc, ndf * nf_mult)])


        # deconv = [nn.Upsample(scale_factor=8, mode='nearest')]
        deconv = []
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**(n_layers - n - 1), 4)
            deconv += [
                nn.ConvTranspose2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw)]
            if norm_layer is not None and (n_layers - n + 1) < 7:
                deconv += [norm_layer(ndf * nf_mult)]
            deconv += [nl_layer()]
        deconv += [nn.ConvTranspose2d(ndf, input_nc, kernel_size=kw, stride=2, padding=padw), nn.Tanh()]
        self.deconv = nn.Sequential(*deconv)


    def forward(self, x):
        x_conv = self.conv(x)
        # return self.deconv(x_conv), x_conv
        # print x_conv
        conv_flat = x_conv.view(x.size(0), -1)
        code = self.fc(conv_flat)
        return self.deconv( self.fc2(code).view(x_conv.size(0),x_conv.size(1),x_conv.size(2),x_conv.size(3))), code

# Defines the encoder
# and related util functions
# copied from BicyleGAN: networks.py
def conv3x3(in_planes, out_planes):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                     padding=1, bias=True)

# two usage cases, depend on kw and padw
def upsampleConv(inplanes, outplanes, kw, padw):
    sequence = []
    sequence += [nn.Upsample(scale_factor=2, mode='nearest')]
    sequence += [nn.Conv2d(inplanes, outplanes, kernel_size=kw, stride=1, padding=padw, bias=True)]
    return nn.Sequential(*sequence)


def meanpoolConv(inplanes, outplanes):
    sequence = []
    sequence += [nn.AvgPool2d(kernel_size=2, stride=2)]
    sequence += [nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, padding=0, bias=True)]
    return nn.Sequential(*sequence)

def maxpoolConv(inplanes, outplanes):
    sequence = []
    sequence += [nn.MaxPool2d(kernel_size=2, stride=2)]
    sequence += [nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, padding=0, bias=True)]
    return nn.Sequential(*sequence)

def convMeanpool(inplanes, outplanes):
    sequence = []
    sequence += [conv3x3(inplanes, outplanes)]
    sequence += [nn.AvgPool2d(kernel_size=2, stride=2)]
    return nn.Sequential(*sequence)

def convMaxpool(inplanes, outplanes):
    sequence = []
    sequence += [conv3x3(inplanes, outplanes)]
    sequence += [nn.MaxPool2d(kernel_size=2, stride=2)]
    return nn.Sequential(*sequence)

def upsampleLayer(inplanes, outplanes, upsample='basic', padding_type='zero'):
    # padding_type = 'zero'
    if upsample == 'basic':
        upconv = [nn.ConvTranspose2d(inplanes, outplanes, kernel_size=4, stride=2, padding=1)]
    elif upsample == 'bilinear':
        upconv = [nn.Upsample(scale_factor=2, mode='bilinear'),
                  nn.ReflectionPad2d(1),
                  nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=1, padding=0)]
    else:
        raise NotImplementedError('upsample layer [%s] not implemented' % upsample)
    return upconv

class BasicBlock(nn.Module):
    def __init__(self, inplanes, outplanes, norm_layer=None, nl_layer=None, pooling='max'):
        super(BasicBlock, self).__init__()
        layers = []
        if norm_layer is not None:
            layers += [norm_layer(inplanes)]
        layers += [nl_layer()]
        layers += [conv3x3(inplanes, inplanes)]
        if norm_layer is not None:
            layers += [norm_layer(inplanes)]
        layers += [nl_layer()]
        if pooling == 'mean':
            layers += [convMeanpool(inplanes, outplanes)]
            self.shortcut = meanpoolConv(inplanes, outplanes)
        elif pooling == 'max':
            layers += [convMaxpool(inplanes, outplanes)]
            self.shortcut = maxpoolConv(inplanes, outplanes)
        else:
            raise NotImplementedError("Pooling mode only max or mean.")

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x) + self.shortcut(x)
        return out


class E_ResNet(nn.Module):
    def __init__(self, input_nc=3, output_nc=1, ndf=64, n_blocks=4,
                 norm_layer=None, nl_layer=None, gpu_ids=[], vaeLike=False, pooling='max'):
        super(E_ResNet, self).__init__()
        self.gpu_ids = gpu_ids
        self.vaeLike = vaeLike
        max_ndf = 4
        conv_layers = [nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1, bias=True)]
        for n in range(1, n_blocks):
            input_ndf = ndf * min(max_ndf, n)  # 2**(n-1)
            output_ndf = ndf * min(max_ndf, n+1)  # 2**n
            if n < 6:
                conv_layers += [BasicBlock(input_ndf, output_ndf, norm_layer, nl_layer, pooling)]
            else:
                conv_layers += [BasicBlock(input_ndf, output_ndf, None, nl_layer, pooling)]
        if n_blocks <= 5:
            if pooling == 'max':
                conv_layers += [nl_layer(), nn.MaxPool2d(8)]
            elif pooling == 'mean':
                conv_layers += [nl_layer(), nn.AvgPool2d(8)]
            else:
                raise NotImplementedError("Pooling mode only max or mean.")

        if vaeLike:
            self.fc = nn.Sequential(*[nn.Linear(output_ndf, output_nc)])
            self.fcVar = nn.Sequential(*[nn.Linear(output_ndf, output_nc)])
        else:
            self.fc = nn.Sequential(*[nn.Linear(output_ndf, output_nc)])
        self.conv = nn.Sequential(*conv_layers)

    def forward(self, x):
        x_conv = self.conv(x)
        conv_flat = x_conv.view(x.size(0), -1)
        output = self.fc(conv_flat)
        if self.vaeLike:
            outputVar = self.fcVar(conv_flat)
            return output, outputVar
        else:
            return output
        return output

class E_NLayers(nn.Module):
    def __init__(self, input_nc, output_nc=1, ndf=64, n_layers=3,
                 norm_layer=None, nl_layer=None, gpu_ids=[], vaeLike=False):
        super(E_NLayers, self).__init__()
        self.gpu_ids = gpu_ids
        self.vaeLike = vaeLike

        kw, padw = 4, 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nl_layer()]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 4)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw)]
            if norm_layer is not None:
                sequence += [norm_layer(ndf * nf_mult)]
            sequence += [nl_layer()]
        sequence += [nn.AvgPool2d(8)]
        self.conv = nn.Sequential(*sequence)
        self.fc = nn.Sequential(*[nn.Linear(ndf * nf_mult, output_nc)])
        if vaeLike:
            self.fcVar = nn.Sequential(*[nn.Linear(ndf * nf_mult, output_nc)])

    def forward(self, x):
        x_conv = self.conv(x)
        conv_flat = x_conv.view(x.size(0), -1)
        output = self.fc(conv_flat)
        if self.vaeLike:
            outputVar = self.fcVar(conv_flat)
            return output, outputVar
        return output

# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
class G_Unet_add_all(nn.Module):
    def __init__(self, input_nc, output_nc, nz, num_downs, ngf=64,
                 norm_layer=None, nl_layer=None, use_dropout=False, gpu_ids=[], upsample='basic'):
        super(G_Unet_add_all, self).__init__()
        self.gpu_ids = gpu_ids
        self.nz = nz
        # construct unet structure
        unet_block = UnetBlock_with_z(ngf*8, ngf * 8, ngf * 8, nz, None, innermost=True,
                                      norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample)
        unet_block = UnetBlock_with_z(ngf*8, ngf * 8, ngf * 8, nz, unet_block,
                                      norm_layer=norm_layer, nl_layer=nl_layer, use_dropout=use_dropout, upsample=upsample)
        for i in range(num_downs - 6):
            unet_block = UnetBlock_with_z(ngf*8, ngf * 8, ngf * 8, nz, unet_block,
                                          norm_layer=norm_layer, nl_layer=nl_layer, use_dropout=use_dropout, upsample=upsample)
        unet_block = UnetBlock_with_z(ngf*4, ngf * 4, ngf * 8, nz, unet_block, norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample)
        unet_block = UnetBlock_with_z(ngf*2, ngf * 2, ngf * 4, nz, unet_block, norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample)
        unet_block = UnetBlock_with_z(ngf, ngf, ngf * 2, nz, unet_block, norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample)
        unet_block = UnetBlock_with_z(input_nc, output_nc, ngf, nz, unet_block, outermost=True, norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample)
        self.model = unet_block

    def forward(self, x, z):
        return self.model(x, z)


class UnetBlock_with_z(nn.Module):
    def __init__(self, input_nc, outer_nc, inner_nc, nz=0,
                 submodule=None, outermost=False, innermost=False,
                 norm_layer=None, nl_layer=None, use_dropout=False, upsample='basic', padding_type='zero'):
        super(UnetBlock_with_z, self).__init__()
        p = 0
        downconv = []
        if padding_type == 'reflect':
            downconv += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            downconv += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        self.outermost = outermost
        self.innermost = innermost
        self.nz = nz
        input_nc = input_nc + nz
        downconv += [nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=p)]
        downrelu = nn.LeakyReLU(0.2, True)  # downsample is different from upsample
        uprelu = nl_layer()

        if outermost:
            upconv = upsampleLayer(inner_nc*2, outer_nc, upsample=upsample, padding_type=padding_type)
            down = downconv
            up = [uprelu] + upconv + [nn.Tanh()]
        elif innermost:
            upconv = upsampleLayer(inner_nc, outer_nc, upsample=upsample, padding_type=padding_type)
            down = [downrelu] + downconv
            up = [uprelu] + upconv
            if norm_layer is not None:
                up += [norm_layer(outer_nc)]
        else:
            upconv = upsampleLayer(inner_nc*2, outer_nc, upsample=upsample, padding_type=padding_type)
            down = [downrelu] + downconv
            if norm_layer is not None:
                down += [norm_layer(inner_nc)]
            up = [uprelu] + upconv

            if norm_layer is not None:
                up += [norm_layer(outer_nc)]

            if use_dropout:
                up += [nn.Dropout(0.5)]
        self.down = nn.Sequential(*down)
        self.submodule = submodule
        self.up = nn.Sequential(*up)

    def forward(self, x, z):
        # print(x.size())
        if self.nz > 0:
            z_img = z.view(z.size(0), z.size(1), 1, 1).expand(z.size(0), z.size(1), x.size(2), x.size(3))
            x_and_z = torch.cat([x, z_img], 1)
        else:
            x_and_z = x

        if self.outermost:
            x1 = self.down(x_and_z)
            x2 = self.submodule(x1, z)
            return self.up(x2)
        elif self.innermost:
            x1 = self.up(self.down(x_and_z))
            return torch.cat([x1, x], 1)
        else:
            x1 = self.down(x_and_z)
            x2 = self.submodule(x1, z)
            return torch.cat([self.up(x2), x], 1)

# Defines the generator that consists of Resnet blocks between a few
# downsampling/upsampling operations.
# Code and idea originally from Justin Johnson's architecture.
# https://github.com/jcjohnson/fast-neural-style/
class G_Resnet_add_all(nn.Module):
    def __init__(self, input_nc, output_nc, nz, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, gpu_ids=[], padding_type='reflect', upsample='basic'):
        assert(n_blocks >= 0)
        super(G_Resnet_add_all, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        resnet_chain = []
        down = DownsampleBlock_with_z(input_nc, ngf, norm_layer=norm_layer, nz=nz, first=True, prev=None)

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            down = DownsampleBlock_with_z(ngf * mult, ngf * mult * 2, nz=nz, norm_layer=norm_layer, prev=down)

        mult = 2**n_downsampling
        for i in range(n_blocks):
            resnet_chain += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer,
                                        use_dropout=use_dropout, use_bias=use_bias)]

        up = None
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            up = UpsampleBlock_with_z(ngf * mult, int(ngf * mult / 2), nz=nz, norm_layer=norm_layer, prev=up, use_dropout=use_dropout, upsample=upsample)

        up = UpsampleBlock_with_z(ngf, output_nc, nz=nz, norm_layer=norm_layer, last=True, prev=up, use_dropout=use_dropout, upsample=upsample)

        self.up = up
        self.down = down
        self.resnet_chain = nn.Sequential(*resnet_chain)



    def forward(self, x, z):
        x = self.down(x,z)
        x = self.resnet_chain(x)
        x = self.up(x,z)
        return x

class DownsampleBlock_with_z(nn.Module):
    def __init__(self, input_nc, output_nc, nz=0, norm_layer=nn.BatchNorm2d, prev = None, first=False):

        super(DownsampleBlock_with_z, self).__init__()


        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if first:
            model = [nn.ReflectionPad2d(3),
                     nn.Conv2d(input_nc+nz, output_nc, kernel_size=7, padding=0,
                               bias=use_bias),
                     norm_layer(output_nc),
                     nn.ReLU(True)]
        else:
            model = [nn.Conv2d(input_nc+nz, output_nc, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(output_nc),
                      nn.ReLU(True)]

        self.nz = nz
        self.model = nn.Sequential(*model)
        self.prev = prev

    def forward(self, x, z):
        if self.prev is not None:
            x = self.prev(x, z)
        if self.nz > 0:
            z_img = z.view(z.size(0), z.size(1), 1, 1).expand(z.size(0), z.size(1), x.size(2), x.size(3))
            x_and_z = torch.cat([x, z_img], 1)
        else:
            x_and_z = x
        return self.model(x_and_z)

class UpsampleBlock_with_z(nn.Module):
    def __init__(self, input_nc, output_nc, nz=0, norm_layer=nn.BatchNorm2d, prev = None, last=False, use_dropout=False,upsample='basic'):

        super(UpsampleBlock_with_z, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if last:
            model = [nn.ReflectionPad2d(3),
                    nn.Conv2d(input_nc+nz, output_nc, kernel_size=7, padding=0),
                    nn.Tanh()]

        else:
            if upsample == 'basic':
                model = [nn.ConvTranspose2d(input_nc+nz, output_nc,
                                                 kernel_size=3, stride=2,
                                                 padding=1, output_padding=1,
                                                 bias=use_bias),
                              norm_layer(output_nc),
                              nn.ReLU(True)]
            elif upsample == 'bilinear':
                model = upsampleLayer(input_nc+nz, output_nc, upsample=upsample) + \
                             [norm_layer(output_nc),
                             nn.ReLU(True)]
            else:
                raise NotImplementedError("Upsampling only basic and bilinear.")

        if use_dropout and not last:
            model += [nn.Dropout2d(0.5)]

        self.nz = nz
        self.model = nn.Sequential(*model)
        self.prev = prev

    def forward(self, x, z):
        if self.prev is not None:
            x = self.prev(x, z)
        if self.nz > 0:
            z_img = z.view(z.size(0), z.size(1), 1, 1).expand(z.size(0), z.size(1), x.size(2), x.size(3))
            x_and_z = torch.cat([x, z_img], 1)
        else:
            x_and_z = x

        return self.model(x_and_z)


## WGAN related
class WassersteinGANLoss(nn.Module):
    """WassersteinGANLoss
    ref: Wasserstein GAN (https://arxiv.org/abs/1701.07875)
    """

    def __init__(self):
        super(WassersteinGANLoss, self).__init__()

    def __call__(self, fake, real=None, generator_loss=True):
        if generator_loss:
            wloss = fake.mean()
        else:
            wloss = real.mean() - fake.mean()
        return -wloss

class WassersteinGANCritic(nn.Module):
    def __init__(self, in_channels, conv_output_dim, gpu_ids=[]):
        super(WassersteinGANCritic, self).__init__()
        self.gpu_ids = gpu_ids
        self.conv_net = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2),
            # nn.InstanceNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            # nn.InstanceNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2),
            # nn.InstanceNorm2d(256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(256, 256, kernel_size=4, stride=2),
            # nn.InstanceNorm2d(256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(256, 256, kernel_size=4, stride=2),
            # nn.InstanceNorm2d(256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        self.mlp_net = nn.Sequential(
            # nn.Linear(512 * conv_output_dim * conv_output_dim, 512),
            nn.Linear(256 * conv_output_dim * conv_output_dim, 512),
            nn.LeakyReLU(negative_slope=0.2, inplace=False),
            nn.Linear(512, 512),
            nn.LeakyReLU(negative_slope=0.2, inplace=False),
            nn.Linear(512, 1),
        )

    def forward(self, input):
        h = self.conv_net(input)
        h = h.view(h.size(0), h.size(1) * h.size(2) * h.size(3))

        return self.mlp_net(h)


class MedianPool2d(nn.Module):
    """ Median pool (usable as median filter when stride=1) module.

    Args:
         kernel_size: size of pooling kernel, int or 2-tuple
         stride: pool stride, int or 2-tuple
         padding: pool padding, int or 4-tuple (l, r, t, b) as in pytorch F.pad
         same: override padding and enforce same padding, boolean
    """

    def __init__(self, kernel_size=3, stride=1, padding=0, same=False):
        super(MedianPool2d, self).__init__()
        self.k = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _quadruple(padding)  # convert to l, r, t, b
        self.same = same

    def _padding(self, x):
        if self.same:
            ih, iw = x.size()[2:]
            if ih % self.stride[0] == 0:
                ph = max(self.k[0] - self.stride[0], 0)
            else:
                ph = max(self.k[0] - (ih % self.stride[0]), 0)
            if iw % self.stride[1] == 0:
                pw = max(self.k[1] - self.stride[1], 0)
            else:
                pw = max(self.k[1] - (iw % self.stride[1]), 0)
            pl = pw // 2
            pr = pw - pl
            pt = ph // 2
            pb = ph - pt
            padding = (pl, pr, pt, pb)
        else:
            padding = self.padding
        return padding

    def forward(self, x):
        # using existing pytorch functions and tensor ops so that we get autograd,
        # would likely be more efficient to implement from scratch at C/Cuda level
        x = F.pad(x, self._padding(x), mode='reflect')
        x = x.unfold(2, self.k[0], self.stride[0]).unfold(3, self.k[1], self.stride[1])
        x = x.contiguous().view(x.size()[:4] + (-1,)).median(dim=-1)[0]
        return x

