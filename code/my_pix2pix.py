# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 19:14:07 2022

@author: javig
"""

import torch
import my_networks
import torch.nn as nn
from collections import OrderedDict
import functools


class Pix2PixModel():
    def __init__(self, 
                 isTrain = True,
                 gan_mode = 'lsgan',
                 learning_rate = 0.0002,
                 lambda_L1 = 100,
                 input_nc = 3,
                 output_nc = 3,
                 ngf = 64,
                 ndf = 64,
                 norm_layer= functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True), #functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
                 dropout_G = False,
                 dropout_D = False,
                 label_flipping = False,
                 label_smoothing = False,
                 n_layers_D = 3):
        """Initialize the pix2pix class.

        Parameters:
            gan_mode: 'the type of GAN objective. vanilla| lsgan
            ngf: # of gen filters in the last conv layer'
            ndf # of discrim filters in the first conv layer
            learning_rate: 'initial learning rate for adam'
            input_nc_ # of input image channels: 3 for RGB and 1 for grayscale'
            output_nc: # of output image channels: 3 for RGB and 1 for grayscale'
            norm_layer normalization later for both discriminator and generator
            use_dropout use dropout in generator
        """        
        # define networks (both generator and discriminator)
        self.isTrain = isTrain
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gan_mode = gan_mode
        self.learning_rate = learning_rate
        self.optimizers = []
        self.lambda_L1 = lambda_L1
        self.label_flipping = label_flipping
        self.label_smoothing = label_smoothing
        
        self.netG = my_networks.UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=dropout_G)
        self.netG = self.netG.to(self.device)
        
        if self.isTrain: 
            self.netD = my_networks.NLayerDiscriminator(input_nc=output_nc+1, ndf=64, n_layers=n_layers_D, norm_layer=norm_layer, use_dropout=dropout_D)
            self.netD = self.netD.to(self.device)
        
        
        if self.isTrain:
            self.criterionGAN = my_networks.GANLoss(self.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), self.learning_rate)
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), self.learning_rate)
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
    
        
    def forward(self, real_A):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        fake_B = self.netG(real_A)  # G(A)
        
        return fake_B
        
    def backward_D(self, real_A, fake_B, real_B):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((real_A, fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake.to(self.device), False, self.label_flipping, self.label_smoothing)
        # Real
        real_AB = torch.cat((real_A, real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True, self.label_flipping, self.label_smoothing)
        if self.gan_mode == 'wgangp':
            gradient_penalty, gradients = my_networks.cal_gradient_penalty(self.netD,real_AB,fake_AB.detach(),self.device)
            gradient_penalty.backward(retain_graph=True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self, real_A, fake_B, real_B, epoch, starting_epoch_D):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((real_A, fake_B), 1)
        pred_fake = self.netD(fake_AB)
        
        if epoch+1 > starting_epoch_D:
            self.loss_G_GAN = self.criterionGAN(pred_fake, True, False, False)
        else:
            self.loss_G_GAN = 0
            
        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(fake_B, real_B) * self.lambda_L1
        # combine loss and calculate gradients
        if epoch+1 > starting_epoch_D:
            self.loss_G = self.loss_G_GAN + self.loss_G_L1
        else:
            self.loss_G = self.loss_G_L1 # IF WARM START
        self.loss_G.backward()

    def optimize_parameters(self, real_A, real_B, epoch, starting_epoch_D):
        fake_B = self.forward(real_A)                   # compute fake images: G(A)
        if epoch+1 > starting_epoch_D: #IF WARM START
            # update D
            self.set_requires_grad(self.netD, True)  # enable backprop for D
            self.optimizer_D.zero_grad()     # set D's gradients to zero
            self.backward_D(real_A, fake_B, real_B)                # calculate gradients for D
            self.optimizer_D.step()          # update D's weights
        else:
            self.loss_D_fake = 0
            self.loss_D_real = 0
            self.loss_D = 0
        
        # # update D
        # self.set_requires_grad(self.netD, True)  # enable backprop for D
        # self.optimizer_D.zero_grad()     # set D's gradients to zero
        # self.backward_D(real_A, fake_B, real_B)                # calculate gradients for D
        # self.optimizer_D.step()          # update D's weights        
        
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G(real_A, fake_B, real_B, epoch, starting_epoch_D)                  # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights
        
    def get_current_losses(self, loss_names = ['D_fake','D_real','D','G_GAN','G_L1','G']):
        """Return traning losses / errors. train.py will print out these errors on console, and save them to a file"""
        errors_ret = OrderedDict()
        for name in loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(getattr(self, 'loss_' + name))  # float(...) works for both scalar tensor and float number
        return errors_ret
    
    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

        