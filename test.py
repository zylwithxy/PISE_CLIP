from options.test_options import TestOptions
import data as Dataset
from model import create_model
from util.visualizer import Visualizer
from itertools import islice
import numpy as np
import torch
import time
from tqdm import tqdm

if __name__=='__main__':
    # get testing options
    opt = TestOptions().parse()
    # creat a dataset
    dataset = Dataset.create_dataloader(opt)

    dataset_size = len(dataset) * opt.batchSize
    print('testing images = %d' % dataset_size)
    # create a model
    model = create_model(opt)
    
    visualizer = Visualizer(opt)
    
    pbar = tqdm(total= dataset_size)

    with torch.no_grad():
        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            model.set_input(data)
            model.test()
            
            if i % opt.display_freq == 0:
                full_vis_dict, shape_txt  = model.get_current_visuals()
                visualizer.display_current_results(full_vis_dict, shape_txt, i)
                
            if i % opt.print_freq == 0:
                t = (time.time() - iter_start_time)
                losses = model.get_current_errors()
                visualizer.print_current_eval(i, losses, t)
            
            pbar.update(opt.batchSize)
                
                
        print('\nEnd testing')