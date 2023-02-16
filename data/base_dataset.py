import os
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import random
import torchvision.transforms.functional as F
from util import pose_utils
from PIL import Image
import pandas as pd
import torch
import math
import numbers
import clip
from util import util

class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--angle', type=float, default=False)
        parser.add_argument('--shift', type=float, default=False)
        parser.add_argument('--scale', type=float, default=False)
        parser.add_argument('--use_text', type= util.str2bool, default= True, help= "whether to use CLIP text embeddings")
        parser.add_argument('--mask_choice', type= str, default= 'both', choices=['upper_clothes', 'both', 'wo'], help= "choose the masked part of the segmentation map. upper_clothes means masking the upper clothing; both means masking the upper clothing and arms; wo means without masking")
        parser.add_argument('--use_masked_SPL1', type= util.str2bool, default= True, help= "whether to use masked segmentation map of pose 1")
        parser.add_argument('--seg_map_visual_choice', type= str, default= 'horizontal', choices=['horizontal', 'vertical'], help= "choose the way for visualizing seg map in test set")
        parser.add_argument('--use_pose1', type= util.str2bool, default= True, help= "whether to use conditional pose")
        return parser

    def initialize(self, opt):
        self.opt = opt
        self.image_dir, self.bone_file, self.name_pairs, self.par_dir, self.fname_shape_pair = self.get_paths(opt)
        size = len(self.name_pairs)
        self.dataset_size = size
        self.class_num = 8
        self.shape_text = ["sleeveless", "short-sleeve", "medium-sleeve", "long-sleeve", "not long-sleeve", "not visible"]

        if isinstance(opt.load_size, int):
            self.load_size = (opt.load_size, opt.load_size)
        else:
            self.load_size = opt.load_size


        transform_list=[]
        # transform_list.append(transforms.Resize(size=self.load_size))
        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5)))
        self.trans = transforms.Compose(transform_list) 

        self.annotation_file = pd.read_csv(self.bone_file, sep=':')
        self.annotation_file = self.annotation_file.set_index('name')
        
        self.return_dict = {'P1': 'P1', 'BP1': 'BP1', 'P2': 'P2', 'BP2': 'BP2', 'SPL1_masked': 'SPL1_onehot_masked', 'SPL1_img': 'SPL1_tensor_mask', 'TEXT': 'txt', 'SPL2':'SPL2_onehot', 'label_P2': 'SPL2',
                'P1_path': 'P1_name', 'P2_path': 'P2_name'}
        
        if not opt.use_pose1:
            del self.return_dict['BP1']
        if not opt.use_masked_SPL1:
            del self.return_dict['SPL1_masked']
            del self.return_dict['SPL1_img'] # Choose 'BP1', 'SPL1_onehot_masked', 'SPL1_tensor_mask' based on self.opt.use_pose1, self.opt.use_masked_SPL1
            
        # For the masked choice.
    
    def masked_choice(self, choice: str, temp: torch.Tensor) -> None:
        """Masking choice and modify the value of temp in-place.

        Args:
            choice (str): ['upper_clothes', 'both', 'wo']
            temp (torch.Tensor): 
        """
        if choice == 'upper_clothes':
            temp[temp == 3] = 0
            pass
        elif choice == 'both':
            temp[temp == 3] = 0
            temp[temp == 6] = 0
            pass
        elif choice == 'wo':
            pass
        else:
            raise ValueError('The choice must be in the expression [\'upper_clothes\', \'both\', \'wo\']')
    
    def preserve_part_SPL2(self, temp: torch.Tensor) -> None:
        
        for i in range(8):
            if i != 3:
                temp[temp == i] = 0
        pass
        
    def get_paths(self, opt):
        label_paths = []
        image_paths = []
        instance_paths = []
        par_paths = []
        assert False, "A subclass of MarkovAttnDataset must override self.get_paths(self, opt)" # Which means the subclass should have .get_paths method to override get_paths method in baseclass
        return label_paths, image_paths, instance_paths, par_paths, fname_shape_pair
    
    def process_SPL(self, Person_name: str):
        """Contain same codes for SPL1 and SPL2

        Args:
            Person_name (str): P1_name or P2_name
        """
        SPL_path = os.path.join(self.par_dir, Person_name[:-4]+'.png')
        SPL_img = Image.open(SPL_path)#.crop(regions)
        snp = np.expand_dims(np.array(SPL_img),-1)
        snp = np.concatenate([snp,snp,snp], -1)
        SPL_img = Image.fromarray(np.uint8(snp))
        
        return SPL_img
    
    def get_masked_SPL1(self, P1_name: str):
        
        SPL1_img  = self.process_SPL(P1_name)
        
        angle, shift, scale = self.getRandomAffineParam()
        SPL1_img = F.affine(SPL1_img, angle=angle, translate=shift, scale=scale, shear=0, fillcolor=(128, 128, 128))
        
        SPL1_img = np.expand_dims(np.array(SPL1_img)[:,:,0],0)#[:,:,40:-40] # 1*256*176
        
        _, h, w = SPL1_img.shape
        num_class = self.class_num
        
        tmp = torch.from_numpy(SPL1_img).view( -1).long()
        self.masked_choice(self.opt.mask_choice, tmp)
        # tmp[tmp == 3] = 0 # masked the upper clothes
        # tmp[tmp == 6] = 0 # masked arms
        
        ones = torch.sparse.torch.eye(num_class)
        ones = ones.index_select(0, tmp)
        SPL1_onehot_masked = ones.view([h,w, num_class])
        #print(SPL2_onehot.shape)
        SPL1_onehot_masked = SPL1_onehot_masked.permute(2,0,1)
        
        # SPL1_img
        SPL1_img_temp = torch.from_numpy(SPL1_img).long()
        SPL1_tensor_mask = torch.zeros_like(SPL1_img_temp ,dtype= SPL1_img_temp.dtype) # Create mask
        SPL1_tensor_mask.copy_(SPL1_img_temp.data)
        
        self.masked_choice(self.opt.mask_choice, SPL1_tensor_mask)
        
        return SPL1_onehot_masked, SPL1_tensor_mask
    
    def get_SPL2(self, P2_name: str):
        
        SPL2_img = self.process_SPL(P2_name)
        
        angle, shift, scale = self.getRandomAffineParam()
        angle, shift, scale = angle*0.2, (shift[0]*0.5,shift[1]*0.5), 1
        
        SPL2_img = F.affine(SPL2_img, angle=angle, translate=shift, scale=scale, shear=0, fillcolor=(128, 128, 128))
        SPL2_img = np.expand_dims(np.array(SPL2_img)[:,:,0],0)#[:,:,40:-40]
        
        _, h, w = SPL2_img.shape
       # print(SPL2_img.shape,SPL1_img.shape)
        num_class = self.class_num
        
        tmp = torch.from_numpy(SPL2_img).view( -1).long()
        if not self.opt.use_masked_SPL1:
            self.preserve_part_SPL2(tmp)
        
        ones = torch.sparse.torch.eye(num_class)
        ones = ones.index_select(0, tmp)
        SPL2_onehot = ones.view([h,w, num_class])
        #print(SPL2_onehot.shape)
        SPL2_onehot = SPL2_onehot.permute(2,0,1)
        
        SPL2 = torch.from_numpy(SPL2_img).long()
        if not self.opt.use_masked_SPL1:
            self.preserve_part_SPL2(SPL2)
        
        return SPL2_onehot, SPL2

    def __getitem__(self, index):
        P1_name, P2_name = self.name_pairs[index]
        P1_path = os.path.join(self.image_dir, P1_name) # person 1
        P2_path = os.path.join(self.image_dir, P2_name) # person 2


        regions = (40,0,216,256)
        P1_img = Image.open(P1_path).convert('RGB')#.crop(regions)
        P2_img = Image.open(P2_path).convert('RGB')#.crop(regions)
        
        if np.array(P1_img).shape[1]==176:
            tmp = np.ones([256, 40, 3])*255
            P1_img = Image.fromarray(np.uint8(np.concatenate([tmp, np.array(P1_img), tmp],1)))
            P2_img = Image.fromarray(np.uint8(np.concatenate([tmp, np.array(P2_img), tmp],1)))
        
        #P1_img = F.resize(P1_img, self.load_size)
        #P2_img = F.resize(P2_img, self.load_size)
        #SPL1_img = F.resize(SPL1_img, (256,256),interpolation=0)
        #SPL2_img = F.resize(SPL2_img, (256,256),interpolation=0)

        angle, shift, scale = self.getRandomAffineParam()
        P1_img = F.affine(P1_img, angle=angle, translate=shift, scale=scale, shear=0, fillcolor=(128, 128, 128))
        center = (P1_img.size[0] * 0.5 + 0.5, P1_img.size[1] * 0.5 + 0.5)
        
        if self.opt.use_pose1:
            affine_matrix = self.get_affine_matrix(center=center, angle=angle, translate=shift, scale=scale, shear=0)
            BP1 = self.obtain_bone(P1_name, affine_matrix)
            
        P1 = self.trans(P1_img)




        angle, shift, scale = self.getRandomAffineParam()
        angle, shift, scale = angle*0.2, (shift[0]*0.5,shift[1]*0.5), 1 # Reduce the deform parameters of the generated image
        P2_img = F.affine(P2_img, angle=angle, translate=shift, scale=scale, shear=0, fillcolor=(128, 128, 128))
        center = (P1_img.size[0] * 0.5 + 0.5, P1_img.size[1] * 0.5 + 0.5)
        affine_matrix = self.get_affine_matrix(center=center, angle=angle, translate=shift, scale=scale, shear=0)
        BP2 = self.obtain_bone(P2_name, affine_matrix)
        P2 = self.trans(P2_img)

        #print(SPL1_img.shape)
       # SPL1_img = SPL1_img.transpose(2,0)
       # SPL2_img = SPL2_img.transpose(2,0)

        # tmp[tmp == 3] = 0 # masked the upper clothes
        # tmp[tmp == 6] = 0 # masked arms
        
        # SPL1_tensor_mask[SPL1_tensor_mask == 3] = 0 # For upper clothing
        # SPL1_tensor_mask[SPL1_tensor_mask == 6] = 0 # For arms
        # SPL1_tensor_mask[SPL1_tensor_mask == 5] = 0 # For lower clothing

        #print(SPL1.shape)
        
        # Read the text
        txt = self.shape_text[self.fname_shape_pair[P1_name]] if self.opt.use_text else 'without text'# TXT1_complement
        # token_txt = clip.tokenize(txt).cuda()
        # TXT1_complement = self.model_clip.encode_text(token_txt)
        
        SPL2_onehot, SPL2 = self.get_SPL2(P2_name)
        
        if self.opt.use_masked_SPL1:
            SPL1_onehot_masked, SPL1_tensor_mask = self.get_masked_SPL1(P1_name)
        
        params_dict = dict()
        
        for key, value in self.return_dict.items():
            params_dict[key] = locals()[value]
        
        return params_dict

    def obtain_bone(self, name, affine_matrix):
        string = self.annotation_file.loc[name]
        array = pose_utils.load_pose_cords_from_strings(string['keypoints_y'], string['keypoints_x'])
        pose  = pose_utils.cords_to_map(array, self.load_size, self.opt.old_size, affine_matrix)
        pose = np.transpose(pose,(2, 0, 1))
        pose = torch.Tensor(pose)
        return pose  

   

    def __len__(self):
        return self.dataset_size

    def name(self):
        assert False, "A subclass of BaseDataset must override self.name"

    def getRandomAffineParam(self):
        if self.opt.angle is not False:
            angle = np.random.uniform(low=self.opt.angle[0], high=self.opt.angle[1])
        else:
            angle = 0
        if self.opt.scale is not False:
            scale   = np.random.uniform(low=self.opt.scale[0], high=self.opt.scale[1])
        else:
            scale=1
        if self.opt.shift is not False:
            shift_x = np.random.uniform(low=self.opt.shift[0], high=self.opt.shift[1])
            shift_y = np.random.uniform(low=self.opt.shift[0], high=self.opt.shift[1])
        else:
            shift_x=0
            shift_y=0
        return angle, (shift_x,shift_y), scale

    def get_inverse_affine_matrix(self, center, angle, translate, scale, shear):
        # code from https://pytorch.org/docs/stable/_modules/torchvision/transforms/functional.html#affine
        # Helper method to compute inverse matrix for affine transformation

        # As it is explained in PIL.Image.rotate
        # We need compute INVERSE of affine transformation matrix: M = T * C * RSS * C^-1
        # where T is translation matrix: [1, 0, tx | 0, 1, ty | 0, 0, 1]
        #       C is translation matrix to keep center: [1, 0, cx | 0, 1, cy | 0, 0, 1]
        #       RSS is rotation with scale and shear matrix
        #       RSS(a, scale, shear) = [ cos(a + shear_y)*scale    -sin(a + shear_x)*scale     0]
        #                              [ sin(a + shear_y)*scale    cos(a + shear_x)*scale     0]
        #                              [     0                  0          1]
        # Thus, the inverse is M^-1 = C * RSS^-1 * C^-1 * T^-1


        angle = math.radians(angle)
        if isinstance(shear, (tuple, list)) and len(shear) == 2:
            shear = [math.radians(s) for s in shear]
        elif isinstance(shear, numbers.Number):
            shear = math.radians(shear)
            shear = [shear, 0]
        else:
            raise ValueError(
                "Shear should be a single value or a tuple/list containing " +
                "two values. Got {}".format(shear))
        scale = 1.0 / scale

        # Inverted rotation matrix with scale and shear
        d = math.cos(angle + shear[0]) * math.cos(angle + shear[1]) + \
            math.sin(angle + shear[0]) * math.sin(angle + shear[1])
        matrix = [
            math.cos(angle + shear[0]), math.sin(angle + shear[0]), 0,
            -math.sin(angle + shear[1]), math.cos(angle + shear[1]), 0
        ]
        matrix = [scale / d * m for m in matrix]

        # Apply inverse of translation and of center translation: RSS^-1 * C^-1 * T^-1
        matrix[2] += matrix[0] * (-center[0] - translate[0]) + matrix[1] * (-center[1] - translate[1])
        matrix[5] += matrix[3] * (-center[0] - translate[0]) + matrix[4] * (-center[1] - translate[1])

        # Apply center translation: C * RSS^-1 * C^-1 * T^-1
        matrix[2] += center[0]
        matrix[5] += center[1]
        return matrix

    def get_affine_matrix(self, center, angle, translate, scale, shear):
        matrix_inv = self.get_inverse_affine_matrix(center, angle, translate, scale, shear)

        matrix_inv = np.matrix(matrix_inv).reshape(2,3)
        pad = np.matrix([0,0,1])
        matrix_inv = np.concatenate((matrix_inv, pad), 0)
        matrix = np.linalg.inv(matrix_inv)
        return matrix
 
