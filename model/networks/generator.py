import sys
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.networks.base_network import BaseNetwork
from model.networks.base_function import *
from torch.nn.utils.spectral_norm import spectral_norm as SpectralNorm

from util.util import feature_normalize
from typing import Union

class ParsingNet(nn.Module):
    """
    define a parsing net to generate target parsing
    """
    def __init__(self, input_nc, output_nc, ngf=32, norm_layer=nn.InstanceNorm2d, act=nn.LeakyReLU(0.2), use_spect=False):
        super(ParsingNet, self).__init__()

        self.conv1 = BlockEncoder(input_nc, ngf*2, ngf, norm_layer, act, use_spect)
        self.conv2 = BlockEncoder(ngf*2, ngf*4, ngf*4, norm_layer, act, use_spect)

        self.conv3 = BlockEncoder(ngf*4, ngf*8, ngf*8, norm_layer, act, use_spect)
        self.conv4 = BlockEncoder(ngf*8, ngf*16, ngf*16, norm_layer, act, use_spect)
        self.deform3 = Gated_conv(ngf*16, ngf*16, norm_layer=norm_layer)
        self.deform4 = Gated_conv(ngf*16, ngf*16, norm_layer=norm_layer)

        self.up1 = ResBlockDecoder(ngf*16, ngf*8, ngf*8, norm_layer, act, use_spect)
        self.up2 = ResBlockDecoder(ngf*8, ngf*4, ngf*4, norm_layer, act, use_spect)


        self.up3 = ResBlockDecoder(ngf*4, ngf*2, ngf*2, norm_layer, act, use_spect)
        self.up4 = ResBlockDecoder(ngf*2, ngf, ngf, norm_layer, act, use_spect)

        self.parout = Output(ngf, 8, 3, norm_layer ,act, None)
        self.makout = Output(ngf, 1, 3, norm_layer, act, None)

    def forward(self, input):
        #print(input.shape)
        x = self.conv2(self.conv1(input))
        x = self.conv4(self.conv3(x))
        x = self.deform4(self.deform3(x))

        x = self.up2(self.up1(x))
        x = self.up4(self.up3(x))

        #print(x.shape)
        par = self.parout(x)
        mak = self.makout(x)
        
        par = (par+1.)/2.
        

        return par, mak


class PoseGenerator(BaseNetwork):
    def __init__(self, image_nc=3, structure_nc=18, output_nc=3, ngf=64, norm='instance', 
                activation='LeakyReLU', use_spect=True, use_coord=False, use_reduc_layer= False, use_text= False):
        super(PoseGenerator, self).__init__()

        self.use_coordconv = True
        self.match_kernel = 3
        self.use_reduc_layer = use_reduc_layer
        
        if use_reduc_layer:
            self.linear = nn.Linear(512, 8)
        
        input_feature_num = 8+18*2+8 if use_reduc_layer else 8+18*2+512
        input_feature_num = input_feature_num if use_text else 8+18*2
        
        self.parnet = ParsingNet(input_feature_num, 8)
        
        

    def forward(self, pose1, pose2, par1, text1: Union[torch.Tensor, None]):
        """_summary_

        Args:
            pose1 (_type_): torch.Size([4, 18, 256, 256])
            pose2 (_type_): torch.Size([4, 18, 256, 256])
            par1 (_type_): torch.Size([4, 8, 256, 256])
            text1 (_type_, optional): torch.Size([4, 512])
        """

        ######### my par   for image editing.
        '''
        parcode,mask = self.parnet(torch.cat((par1, pose1, pose2),1))
        parsav = parcode
        par = torch.argmax(parcode, dim=1, keepdim=True)
        bs, _, h, w = par.shape
       # print(SPL2_img.shape,SPL1_img.shape)
        num_class = 8
        tmp = par.view( -1).long()
        ones = torch.sparse.torch.eye(num_class).cuda() 
        ones = ones.index_select(0, tmp)
        SPL2_onehot = ones.view([bs, h,w, num_class])
        #print(SPL2_onehot.shape)
        SPL2_onehot = SPL2_onehot.permute(0, 3, 1, 2)
        par2 = SPL2_onehot
        '''
        if text1 is None:
            
            h, w = par1.shape[-2:]
            parcode, _ = self.parnet(torch.cat((par1, pose1, pose2),1))
            
        else:
            if self.use_reduc_layer:
                text1 = self.linear(text1.float())
            b, embed_dim = text1.shape
            h, w = par1.shape[-2:]
            parcode, _ = self.parnet(torch.cat((par1, pose1, pose2, text1.view(b, embed_dim, 1, 1).expand(b, embed_dim, h, w)),1))

        return parcode