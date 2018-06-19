import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
import itertools
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks



class VAECycleGANModel(BaseModel):
    def name(self):
        return 'VAECycleGANModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        self.netE = networks.define_E(opt.output_nc, opt.nz + opt.cond_nc, opt.nef,
                                      which_model_netE=opt.which_model_netE,
                                      norm=opt.norm, nl=opt.nl,
                                      init_type=opt.init_type, gpu_ids=self.gpu_ids,
                                      vaeLike=True, pooling='max')

        which_G_A  = 'resnet_6blocks_all' if opt.fineSize < 256 else 'resnet_9blocks_all'
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc,
                                        opt.ngf, which_G_A, opt.norm, opt.dropout, opt.init_type, self.gpu_ids, nz= opt.nz+opt.cond_nc)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc,
                                        opt.ngf, opt.which_model_netG, opt.norm, opt.dropout, opt.init_type, self.gpu_ids)

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            if opt.cond_D:
                self.netD_A = networks.define_D(opt.output_nc + opt.cond_nc, opt.ndf,
                                                opt.which_model_netD,
                                                opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)
            else:
                self.netD_A = networks.define_D(opt.output_nc, opt.ndf,
                                                opt.which_model_netD,
                                                opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)

            self.netD_B = networks.define_D(opt.input_nc, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)
        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
            self.load_network(self.netE  , 'E'  , which_epoch)
            self.load_network(self.netG_A, 'G_A', which_epoch)
            self.load_network(self.netG_B, 'G_B', which_epoch)
            if self.isTrain:
                self.load_network(self.netD_A, 'D_A', which_epoch)
                self.load_network(self.netD_B, 'D_B', which_epoch)

        if self.isTrain:
            self.old_lr = opt.lr
            self.fake_A_pool = ImagePool(opt.pool_size)
            self.fake_B_pool = ImagePool(opt.pool_size)
            # define loss functions
            # KL loss not presented, it's done straightforward
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            self.criterionTV = networks.TVLoss()
            self.criterionClass = torch.nn.L1Loss()

            # initialize optimizers

            self.optimizer_E = torch.optim.Adam(self.netE.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_A = torch.optim.Adam(self.netD_A.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_B = torch.optim.Adam(self.netD_B.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers = []
            self.schedulers = []
            self.optimizers.append(self.optimizer_E)
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D_A)
            self.optimizers.append(self.optimizer_D_B)
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

        print('---------- Networks initialized -------------')
        networks.print_network(self.netE)
        networks.print_network(self.netG_A)
        networks.print_network(self.netG_B)
        if self.isTrain:
            networks.print_network(self.netD_A)
            networks.print_network(self.netD_B)
        print('-----------------------------------------------')

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        input_A = input['A' if AtoB else 'B']
        input_B = input['B' if AtoB else 'A']

        if len(self.gpu_ids) > 0:
            input_A = input_A.cuda(self.gpu_ids[0], async=True)
            input_B = input_B.cuda(self.gpu_ids[0], async=True)

        self.input_A = input_A
        self.input_B = input_B
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

        if self.opt.cond_nc > 0 and self.opt.isTrain:
            input_C = input['C']
            if len(self.gpu_ids) > 0:
                input_C = input_C.cuda(self.gpu_ids[0], async=True)
            self.input_C = input_C

    def forward(self):
        self.real_A = Variable(self.input_A)
        self.real_B = Variable(self.input_B)
        if self.opt.cond_nc > 0:
            self.real_C = Variable(self.input_C)

    def test(self, z, encode=False):

        real_B = Variable(self.input_B, volatile=True)
        real_A = Variable(self.input_A, volatile=True)

        if encode:
            mu, logvar = self.netE.forward(real_B)
            if self.opt.cond_nc > 0.0:
                class_label = mu[:, :self.opt.cond_nc]
                mu = mu[:, self.opt.cond_nc:]
                logvar = logvar[:, self.opt.cond_nc:]
            std = logvar.mul(0.5).exp_()
            eps = self.get_z_random(std.size(0), std.size(1), 'gauss')
            z = mu
            # z = eps.mul(std).add_(mu)
            if self.opt.cond_nc > 0.0:
                z = torch.cat([z, class_label], 1)
        else:
            z = z.astype(np.float32)
            z = np.reshape(z, (1, len(z)))
            z = Variable(torch.from_numpy(z).cuda(self.gpu_ids[0], async=True))

        fake_B = self.netG_A(real_A, z)
        self.rec_A = self.netG_B(fake_B).data
        self.fake_B = fake_B.data

        fake_A = self.netG_B(real_B)

        self.rec_B = self.netG_A(fake_A, z).data
        self.fake_A = fake_A.data

        self.real_A = real_A.data
        self.real_B = real_B.data

        return z

    # get image paths, only used during testing phase
    def get_image_paths(self):
        return self.image_paths

    def backward_D_basic(self, netD, real, fake):
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        # backward
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        if self.opt.cond_D:
            b, _, h, w = self.fake_B.size()
            c = self.opt.cond_nc
            c_img = self.real_C.view(b, c, 1, 1).expand(b, c, h, w)
            fake_B = torch.cat([self.fake_B, c_img], dim=1)
            real_B = torch.cat([self.real_B, c_img], dim=1)
        else:
            fake_B = self.fake_B
            real_B = self.real_B

        fake_B = self.fake_B_pool.query(fake_B)
        loss_D_A = self.backward_D_basic(self.netD_A, real_B, fake_B)
        self.loss_D_A = loss_D_A.data[0]

    def backward_D_B(self):
        fake_A = self.fake_A_pool.query(self.fake_A)
        loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)
        self.loss_D_B = loss_D_B.data[0]

    def backward_EG(self):
        lambda_tv = self.opt.tv
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        lambda_class = self.opt.lambda_class

        # Forwarding networks
        # get encoded z for B-A-B
        mu, logvar = self.netE.forward(self.real_B)
        self.mu = mu
        if lambda_class > 0.0:
            pred_class = mu[:,:self.opt.cond_nc]
            mu = mu[:,self.opt.cond_nc:]
            logvar = logvar[:,self.opt.cond_nc:]

        std = logvar.mul(0.5).exp_()
        eps = self.get_z_random(std.size(0), std.size(1), 'gauss')
        z_encoded = eps.mul(std).add_(mu)
        if lambda_class > 0.0:
            z_encoded = torch.cat([z_encoded, self.real_C],1)
        self.z_encoded_real = z_encoded

        # get random z for A-B-A

        if self.opt.use_random_z:
            z_random = self.get_z_random(self.real_A.size(0), self.opt.nz, 'gauss')
        else:
            z_random = z_encoded
        self.z_random = z_random

        # fake_B = self.netG_A(real_A_with_random_z)
        fake_B = self.netG_A(self.real_A, z_random)
        fake_A = self.netG_B(self.real_B)

        # rec_B = self.netG_A(fake_A_with_encoded_z)
        rec_B = self.netG_A(fake_A, z_encoded)
        rec_A = self.netG_B(fake_B)


        # Classification Loss
        if lambda_class > 0 and self.opt.cond_nc > 0:
            loss_class = self.criterionClass(pred_class, self.real_C) * lambda_class
            self.loss_class = loss_class.data[0]
        else:
            loss_class = 0

        # Total variation loss
        if lambda_tv > 0:
            loss_tv_A = self.criterionTV(fake_B) * lambda_tv
            loss_tv_B = self.criterionTV(fake_A) * lambda_tv
        else:
            loss_tv_A = 0
            loss_tv_B = 0

        if self.opt.lambda_z > 0.0:
            self.mu2, logvar2 = self.netE.forward(fake_B)
            self.z_encoded_fake = self.mu2


        # GAN loss D_A(G_A(A))
        # New: concatenate the guidance
        pred_fake = self.netD_A(fake_B)
        loss_G_A = self.criterionGAN(pred_fake, True)

        # GAN loss D_B(G_B(B))
        pred_fake = self.netD_B(fake_A)
        loss_G_B = self.criterionGAN(pred_fake, True)

        # Forward cycle loss
        loss_cycle_A = self.criterionCycle(rec_A, self.real_A) * lambda_A

        # Backward cycle loss
        loss_cycle_B = self.criterionCycle(rec_B, self.real_B) * lambda_B

        # KL loss
        kl_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
        loss_kl = torch.sum(kl_element).mul_(-0.5) * self.opt.lambda_kl

        # combined loss
        loss_G = loss_G_A + loss_G_B + loss_cycle_A + loss_cycle_B \
                 + loss_tv_B + loss_tv_A + loss_kl + loss_class
        loss_G.backward(retain_graph=True)

        self.fake_B = fake_B.data
        self.fake_A = fake_A.data
        self.rec_A = rec_A.data
        self.rec_B = rec_B.data

        self.loss_G_A = loss_G_A.data[0]
        self.loss_G_B = loss_G_B.data[0]
        self.loss_cycle_A = loss_cycle_A.data[0]
        self.loss_cycle_B = loss_cycle_B.data[0]
        self.loss_kl = loss_kl.data[0]

        if lambda_tv > 0:
            self.loss_tv_A = loss_tv_A.data[0]
            self.loss_tv_B = loss_tv_B.data[0]

    def backward_G_alone(self):
        # 3, reconstruction |z_predit-z_random|
        if self.opt.lambda_z > 0.0:
            self.loss_z_L1 = torch.mean(torch.abs(self.mu2 - self.mu)) * self.opt.lambda_z
            self.loss_z_L1.backward()
        else:
            self.loss_z_L1 = 0.0

    def optimize_parameters(self):
        # forward
        self.forward()
        # G_A and G_B and E
        self.optimizer_E.zero_grad()
        self.optimizer_G.zero_grad()
        self.backward_EG()
        self.optimizer_G.step()
        self.optimizer_E.step()
        if self.opt.lambda_z > 0.0:
            self.optimizer_G.zero_grad()
            self.optimizer_E.zero_grad()
            self.backward_G_alone()
            self.optimizer_G.step()
        # D_A
        self.optimizer_D_A.zero_grad()
        self.backward_D_A()
        self.optimizer_D_A.step()
        # D_B
        self.optimizer_D_B.zero_grad()
        self.backward_D_B()
        self.optimizer_D_B.step()

    def get_current_errors(self):
        if self.opt.lambda_z == 0.0:
            loss_z_L1 = 0
        else:
            loss_z_L1 = self.loss_z_L1.data[0]
        ret_errors = OrderedDict([('D_A', self.loss_D_A), ('G_A', self.loss_G_A), ('Cyc_A', self.loss_cycle_A),
                                 ('D_B', self.loss_D_B), ('G_B', self.loss_G_B), ('Cyc_B',  self.loss_cycle_B),
                                  ('KL', self.loss_kl), ('z_L1', loss_z_L1)])

        if self.opt.lambda_class > 0:
            ret_errors['Class'] = self.loss_class
        if self.opt.tv > 0.0:
            ret_errors['tv_A'] = self.loss_tv_A
            ret_errors['tv_B'] = self.loss_tv_B

        return ret_errors

    def get_current_visuals(self):
        # New: add visuals for guidance

        convert_A = util.tensor2im if self.opt.input_nc <= 3 else util.class_tensor2im_A
        convert_B = util.tensor2im if self.opt.output_nc <= 3 else util.class_tensor2im_B

        real_A = convert_A(self.input_A)
        fake_B = convert_B(self.fake_B)
        rec_A = convert_A(self.rec_A)
        real_B = convert_B(self.input_B)
        fake_A = convert_A(self.fake_A)
        rec_B = convert_B(self.rec_B)
        to_display = [('real_A', real_A), ('fake_B', fake_B), ('rec_A', rec_A),
                                   ('real_B', real_B), ('fake_A', fake_A), ('rec_B', rec_B)];
        if self.opt.cond_nc <= 3 and self.opt.cond_nc > 0:
            real_C = util.tensor2im(self.input_C)
            to_display.append(('real_C', real_C))

        ret_visuals = OrderedDict(to_display)
        if self.opt.isTrain and self.opt.identity > 0.0:
            ret_visuals['idt_A'] = util.tensor2im(self.idt_A)
            ret_visuals['idt_B'] = util.tensor2im(self.idt_B)
        return ret_visuals

    def save(self, label):
        self.save_network(self.netE  , 'E'  , label, self.gpu_ids)
        self.save_network(self.netG_A, 'G_A', label, self.gpu_ids)
        self.save_network(self.netD_A, 'D_A', label, self.gpu_ids)
        self.save_network(self.netG_B, 'G_B', label, self.gpu_ids)
        self.save_network(self.netD_B, 'D_B', label, self.gpu_ids)

    def get_z_random(self, batchSize, nz, random_type='gauss'):
        z = self.Tensor(batchSize, nz)
        if random_type == 'uni':
            z.copy_(torch.rand(batchSize, nz) * 2.0 - 1.0)
        elif random_type == 'gauss':
            z.copy_(torch.randn(batchSize, nz))
        z = Variable(z)
        return z

    def test_simple(self, z_sample, input=None, encode_real_B=False):
        if input is not None:
            self.set_input(input)

        z = self.test(z_sample,encode_real_B)

        real_A = util.tensor2im(self.real_A)
        fake_B = util.tensor2im(self.fake_B)
        real_B = util.tensor2im(self.real_B)
        return self.image_paths, real_A, fake_B, real_B, z