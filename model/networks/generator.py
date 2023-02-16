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
                activation='LeakyReLU', use_spect=True, use_coord=False, use_reduc_layer= False, 
                use_text= False, use_masked_SPL1= True, use_BP1 = True, parsing_net_choice= 'ParsingNet', 
                stage_choice= 'stage1'):
        super(PoseGenerator, self).__init__()

        self.use_coordconv = True
        self.match_kernel = 3
        self.use_reduc_layer = use_reduc_layer
        self.stage_choice = stage_choice 
        
        
        
        if use_reduc_layer:
            assert parsing_net_choice == 'ParsingNet' and use_text
            self.linear = nn.Linear(512, 8)
        
        input_feature_num = 8+18*2 if use_BP1 else 8+18
        if use_text:
            input_feature_num = input_feature_num + 8 if use_reduc_layer else input_feature_num+512
        input_feature_num = input_feature_num if use_masked_SPL1 else input_feature_num - 8
        
        if parsing_net_choice == 'ParsingNet':
            self.parnet = ParsingNet(input_feature_num, 8)
        elif parsing_net_choice == 'ShapeUNet_FCNHead':
            # self.parnet = ShapeUNet_FCNHead_model
            # param == 
            pass
        
        if stage_choice == 'stage_12':
            self.define_networks(norm, activation, ngf)
        

    def define_networks(self, norm, activation, ngf):
        
        norm_layer = get_norm_layer(norm_type=norm)
        nonlinearity = get_nonlinearity_layer(activation_type=activation)
        
        self.Zencoder = Zencoder(3, ngf)
        
        self.imgenc = VggEncoder()
        self.getMatrix = GetMatrix(ngf*4, 1)
        
        self.phi = nn.Conv2d(in_channels=ngf*4+3, out_channels=ngf*4, kernel_size=1, stride=1, padding=0)
        self.theta = nn.Conv2d(in_channels=ngf*4+3, out_channels=ngf*4, kernel_size=1, stride=1, padding=0)

        self.parenc = HardEncoder(8+18+8+3, ngf)

        self.dec = BasicDecoder(3)

        self.efb = EFB(ngf*4, 256)
        self.res = ResBlock(ngf*4, output_nc=ngf*4, hidden_nc=ngf*4, norm_layer=norm_layer, nonlinearity= nonlinearity,
                learnable_shortcut=False, use_spect=False, use_coord=False)
                
        self.res1 = ResBlock(ngf*4, output_nc=ngf*4, hidden_nc=ngf*4, norm_layer=norm_layer, nonlinearity= nonlinearity,
                learnable_shortcut=False, use_spect=False, use_coord=False)

        # self.loss_fn = torch.nn.MSELoss()
        
        
    def generate_parsing_map(self, pose1: Union[torch.Tensor, None], pose2, par1: Union[torch.Tensor, None], text1: Union[torch.Tensor, None]):
        
        if text1 is None:
            
            h, w = pose2.shape[-2:]
            if par1 is not None:
                if pose1 is not None:
                    parcode, _ = self.parnet(torch.cat((par1, pose1, pose2),1))
                else:
                    parcode, _ = self.parnet(torch.cat((par1, pose2),1))
            else:
                if pose1 is not None:
                    parcode, _ = self.parnet(torch.cat((pose1, pose2),1))
                else:
                    parcode, _ = self.parnet(pose2)
        else:
            if self.use_reduc_layer:
                text1 = self.linear(text1.float())
            b, embed_dim = text1.shape
            if par1 is not None:
                h, w = par1.shape[-2:]
                if pose1 is not None:
                    parcode, _ = self.parnet(torch.cat((par1, pose1, pose2, text1.view(b, embed_dim, 1, 1).expand(b, embed_dim, h, w)),1))
                else:
                    parcode, _ = self.parnet(torch.cat((par1, pose2, text1.view(b, embed_dim, 1, 1).expand(b, embed_dim, h, w)),1))
            else:
                h, w = pose2.shape[-2:]
                if pose1 is not None:
                    parcode, _ = self.parnet(torch.cat((pose1, pose2, text1.view(b, embed_dim, 1, 1).expand(b, embed_dim, h, w)),1))
                else:
                    parcode, _ = self.parnet(torch.cat((pose2, text1.view(b, embed_dim, 1, 1).expand(b, embed_dim, h, w)),1))
            
        return parcode

    def forward(self, img1: Union[torch.Tensor, None], img2: Union[torch.Tensor, None],
                pose1: Union[torch.Tensor, None], pose2: torch.Tensor, par1: Union[torch.Tensor, None], text1: Union[torch.Tensor, None]):
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
        parse_gen2 = self.generate_parsing_map(pose1, pose2, par1, text1)
        
        if self.stage_choice == 'stage_12':
            # bs: batch size; s_size: the feature number of segmentation map 8; cs: the feature number of codes 256
            codes_vector, exist_vector, img1code = self.Zencoder(img1, par1) # shape of these three vectors (bs, s_size+1, cs)[global and local]; (bs, s_size)[state for each batch size and each seg feature]; (bs, cs, hs, ws)[feature before self.get_code]
            parcode = self.parenc(torch.cat((par1, parse_gen2, pose2, img1), 1))
        
            return None # which needs to be modified.
        
        elif self.stage_choice == 'stage1':
            return parse_gen2