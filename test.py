from options.test_options import TestOptions
import data as Dataset
from model import create_model
from util.visualizer import Visualizer
from itertools import islice
import numpy as np
import torch
import time
from tqdm import tqdm
from util.segmentation_score import SegmentationMetric

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
    
    seg_metric = SegmentationMetric(8, opt.seg_metric_choice, opt.filter_bg) # Evaluation metrics. mIOU

    with torch.no_grad():
        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            model.set_input(data)
            model.test()
            
            seg_metric.update(model.parsav, model.label_P2.squeeze(1).long())
            
            if i % opt.display_freq == 0:
                full_vis_dict, shape_txt  = model.get_current_visuals()
                visualizer.display_current_results(full_vis_dict, shape_txt, i)
                
            if i % opt.print_freq == 0:
                t = (time.time() - iter_start_time)
                losses = model.get_current_errors()
                losses.update((name, value) for name, value in zip(('pixAcc', 'mIoU'), seg_metric.get_current()))
                visualizer.print_current_eval(i, losses, t)
            
            pbar.update(opt.batchSize)
        
        metric = dict(zip(('pixAcc', 'mIoU'), seg_metric.get()))
        visualizer.print_eval_metric(metric)
        # print(f'The pixAcc is {seg_metric.get()[0]}, mIoU is {seg_metric.get()[1]}')
        print('\nEnd testing')