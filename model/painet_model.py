import torch
import torch.nn as nn
from model.base_model import BaseModel
from model.networks import base_function, external_function
import model.networks as network
from util import task, util
import itertools
import data as Dataset
import numpy as np
from itertools import islice
import random
import os
import clip

import torch.nn.functional as F


class Painet(BaseModel): # which is only for parsing models
    def name(self):
        return "parsing network"

    @staticmethod
    def modify_options(parser, is_train=True):
        parser.add_argument('--netG', type=str, default='pose', help='The name of net Generator')
        parser.add_argument('--init_type', type=str, default='orthogonal', help='Initial type')

        # if is_train:
        parser.add_argument('--use_spect_g', action='store_false', help="whether use spectral normalization in generator")
        # parser.add_argument('--use_spect_d', action='store_false', help="whether use spectral normalization in discriminator")
        parser.add_argument('--save_input', action='store_false', help="whether save the input images when testing")
        parser.add_argument('--use_reduc_layer', type= util.str2bool, default= True, help="whether to use reduction layer")

        parser.set_defaults(use_spect_g=False)
        # parser.set_defaults(use_spect_d=True)
        parser.set_defaults(save_input=False)

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.loss_names = ['par', 'par1'] # only for parsing models

        if opt.use_masked_SPL1:
            self.visual_names = ['input_BP1', 'show_SPL1', 'input_BP2', 'parsav', 'label_P2'] # [:3 [input], 'Generated segmentation map', 'True segmentation map']
        else:
            self.visual_names = ['input_BP1', 'input_BP2', 'parsav', 'label_P2']
        self.model_names = ['G'] # 

        self.FloatTensor = torch.cuda.FloatTensor if len(self.gpu_ids)>0 \
            else torch.FloatTensor

        # define the generator
        use_reduc_layer = opt.use_reduc_layer # 
        self.net_G = network.define_g(opt, image_nc=opt.image_nc, structure_nc=opt.structure_nc, ngf=64,
                                 use_spect=opt.use_spect_g, norm='instance', activation='LeakyReLU', use_reduc_layer= use_reduc_layer, use_text= opt.use_text, use_masked_SPL1= opt.use_masked_SPL1) # only for the segmentation model

        # define the CLIP
        if opt.use_text:
            self.model_clip, _ = clip.load("ViT-B/32", device= 'cuda')
        
        trained_list = ['parnet']
        for k,v in self.net_G.named_parameters():
            flag = False
            for i in trained_list:
                if i in k:
                    flag = True
            if flag:
                #v.requires_grad = False
                print(k)
        
        self.L1loss = torch.nn.L1Loss()
        self.parLoss = CrossEntropyLoss2d()#torch.nn.BCELoss()
             
        if self.isTrain:
            # define the loss functions
            

            # define the optimizer
            self.optimizer_G = torch.optim.Adam(itertools.chain(
                                               filter(lambda p: p.requires_grad, self.net_G.parameters())),
                                               lr=opt.lr, betas=(0.9, 0.999))
            self.optimizers.append(self.optimizer_G)


        # load the pre-trained model and schedulers
        self.setup(opt)


    def set_input(self, input):
        # move to GPU and change data types
        # self.input = input
        if self.opt.use_masked_SPL1:
            input_P1, input_BP1, input_SPL1, self.show_SPL1, self.show_TXT = input['P1'], input['BP1'], input['SPL1_masked'], input['SPL1_img'], input['TEXT']
            input_P2, input_BP2, input_SPL2, label_P2 = input['P2'], input['BP2'], input['SPL2'], input['label_P2']
        else:
            input_P1, input_BP1, self.show_TXT = input['P1'], input['BP1'], input['TEXT']
            input_P2, input_BP2, input_SPL2, label_P2 = input['P2'], input['BP2'], input['SPL2'], input['label_P2']

        if len(self.gpu_ids) > 0:
            self.input_P1 = input_P1.cuda(self.gpu_ids[0])
            self.input_BP1 = input_BP1.cuda(self.gpu_ids[0])
            self.input_SPL1 = input_SPL1.cuda(self.gpu_ids[0]) if self.opt.use_masked_SPL1 else None
            self.input_P2 = input_P2.cuda(self.gpu_ids[0])
            self.input_BP2 = input_BP2.cuda(self.gpu_ids[0])  
            self.input_SPL2 = input_SPL2.cuda(self.gpu_ids[0])  
            self.label_P2 = label_P2.cuda(self.gpu_ids[0])
            self.input_TXT1 = self.model_clip.encode_text(clip.tokenize(self.show_TXT).cuda()) if hasattr(self, 'model_clip') else None
            

        self.image_paths=[]
        for i in range(self.input_P1.size(0)):
            self.image_paths.append(os.path.splitext(input['P1_path'][i])[0] + '_' + input['P2_path'][i])

    def bilinear_warp(self, source, flow):
        [b, c, h, w] = source.shape
        x = torch.arange(w).view(1, -1).expand(h, -1).type_as(source).float() / (w-1)
        y = torch.arange(h).view(-1, 1).expand(-1, w).type_as(source).float() / (h-1)
        grid = torch.stack([x,y], dim=0)
        grid = grid.unsqueeze(0).expand(b, -1, -1, -1)
        grid = 2*grid - 1
        flow = 2*flow/torch.tensor([w, h]).view(1, 2, 1, 1).expand(b, -1, h, w).type_as(flow)
        grid = (grid+flow).permute(0, 2, 3, 1)
        input_sample = F.grid_sample(source, grid).view(b, c, -1)
        return input_sample

    def test(self):
        """Forward function used in test time"""
        self.parsav  = self.net_G(self.input_BP1, self.input_BP2, self.input_SPL1, self.input_TXT1)
        
        # parsing loss
        label_P2 = self.label_P2.squeeze(1).long()
        #print(self.input_SPL2.min(), self.input_SPL2.max(), self.parsav.min(), self.parsav.max())
        self.loss_par = self.parLoss(self.parsav,label_P2)# * 20. 
        self.loss_par1 = self.L1loss(self.parsav, self.input_SPL2)  * 100
        ## test flow ##
        
        """
        self.save_results(img_gen, data_name='vis')
        if self.opt.save_input or self.opt.phase == 'val':
            self.save_results(self.input_P1, data_name='ref')
            self.save_results(self.input_P2, data_name='gt')
            result = torch.cat([self.input_P1, img_gen, self.input_P2], 3)
            self.save_results(result, data_name='all')
        """      

    def forward(self):
        """Run forward processing to get the inputs"""
        self.parsav = self.net_G(self.input_BP1, self.input_BP2, self.input_SPL1, self.input_TXT1)
      

    def backward_G(self):
        """Calculate parsing loss for the generator"""
        
        # parsing loss
        label_P2 = self.label_P2.squeeze(1).long()
        #print(self.input_SPL2.min(), self.input_SPL2.max(), self.parsav.min(), self.parsav.max())
        self.loss_par = self.parLoss(self.parsav,label_P2)# * 20. 
        self.loss_par1 = self.L1loss(self.parsav, self.input_SPL2)  * 100
        
        # If need GAN loss to distinguish between fake segmentation maps or true segmentation maps

        total_loss = 0

        for name in self.loss_names:
            # if name != 'dis_img_gen':
                #print(getattr(self, "loss_" + name))
            total_loss += getattr(self, "loss_" + name)
        total_loss.backward()


    def optimize_parameters(self):
        """update network weights"""
        self.forward()
        
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        
class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True, ignore_index=255):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss(weight, size_average, ignore_index)
        self.softmax = nn.LogSoftmax(dim=1)
    def forward(self, inputs, targets):
        return self.nll_loss(self.softmax(inputs), targets)
