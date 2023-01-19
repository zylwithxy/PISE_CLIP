import os.path
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
import pandas as pd
from util import pose_utils
import numpy as np
import torch
from typing import Dict
import clip

class FashionDataset(BaseDataset):

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser = BaseDataset.modify_commandline_options(parser, is_train)
        if is_train:
            parser.set_defaults(load_size=256)
        else:
            parser.set_defaults(load_size=256)
        parser.set_defaults(old_size=(256, 256))
        parser.set_defaults(structure_nc=18)
        parser.set_defaults(image_nc=3)
        parser.set_defaults(display_winsize=256)
        return parser



    def get_paths(self, opt):
        root = opt.dataroot
        phase = opt.phase
        # For adapting the name of ADGAN.
        
        pairLst = os.path.join(root, 'deepmultimodal-resize-pairs-%s.csv' %phase)#'fasion-pairs-%s.csv' % phase)
        # pairLst = os.path.join(root, 'fasion-pairs-%s.csv' %phase)#'fasion-pairs-%s.csv' % phase)
#        pairLst = os.path.join(root, 'arbf_pres.csv')
        name_pairs = self.init_categories(pairLst)
        
        image_dir = os.path.join(root, 'fashion_resize','%s' % phase)
        # image_dir = os.path.join(root, '%s' % phase)
        
        # deepmultimodal-fasion-resize-annotation-train.csv
        bonesLst = os.path.join(root, 'deepmultimodal-fasion-resize-annotation-%s.csv' %phase)
        # bonesLst = os.path.join(root, 'fasion-annotation-%s.csv' %phase)#'fasion-annotation-%s.csv' % phase)
        
        par_dir = os.path.join(root, '%sSPL8' %phase)
        
        shape_file_path = os.path.join(root, f'shape_{phase}.txt')
        
        self.check_file_path(image_dir, bonesLst, par_dir, shape_file_path)
        
        fname_shape_pair = self.read_shape0_file(shape_file_path)
        
        # model_clip, preprocess = clip.load("ViT-B/32", device= 'cuda') # do not need preprocess temporarily.
        
        return image_dir, bonesLst, name_pairs, par_dir, fname_shape_pair


    def check_file_path(self, *args):
        """Check if the file exists
        args: file names
        """
        for args in args:
            assert os.path.exists(args)

    def init_categories(self, pairLst):
        assert os.path.exists(pairLst)
        pairs_file_train = pd.read_csv(pairLst)
        size = len(pairs_file_train)
        pairs = []
        print('Loading data pairs ...')
        for i in range(size):
            pair = [pairs_file_train.iloc[i]['from'], pairs_file_train.iloc[i]['to']]
            pairs.append(pair)

        print('Loading data pairs finished ...')  
        return pairs
    
    def read_shape0_file(self, shape_file_path: str) -> Dict[str, int]:
        """Read the file which contains the filename[str] and file shape[int]

        Args:
            shape_file_path (str): shape file path.
        """
        fname_shape_pair = dict()
        with open(shape_file_path, 'rt') as f:
            for row in f:
                row_list = row.split()
                fname_shape_pair[row_list[0]] = int(row_list[1])
                
        return fname_shape_pair
        

    def name(self):
        return "FashionDataset"

                
