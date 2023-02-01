import time
from options.train_options import TrainOptions
import data as Dataset
from model import create_model
from util.visualizer import Visualizer
from tqdm import tqdm


if __name__ == '__main__':
    # get training options
    opt = TrainOptions().parse()
    # create a dataset
    dataset = Dataset.create_dataloader(opt)
    dataset_size = len(dataset) * opt.batchSize # total_iteration += 1 which is equal to datasize += opt.batchSize
    print('training images = %d' % dataset_size)
    # create a model
    model = create_model(opt)
    # model = model.to()  
    # create a visualizer
    visualizer = Visualizer(opt)
    # training flag
    keep_training = True
    max_iteration = opt.niter+opt.niter_decay
    epoch = 0
    total_epoch = max_iteration // (len(dataset) // opt.batchSize + 1) + 1
    
    total_iteration = opt.iter_count
    
    pbar = tqdm(total= max_iteration)

    # training process
    while(keep_training):
        epoch_start_time = time.time()
        epoch+=1
        print('\n Training epoch: %d' % epoch)

        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            total_iteration += 1
            model.set_input(data)
            model.optimize_parameters()

            # display images on visdom and save images
            if total_iteration % opt.display_freq == 0:
                full_vis_dict, shape_txt  = model.get_current_visuals()
                visualizer.display_current_results(full_vis_dict, shape_txt, epoch)
                if hasattr(model, 'distribution'):
                    visualizer.plot_current_distribution(model.get_current_dis()) 

            # print training loss and save logging information to the disk
            if total_iteration % opt.print_freq == 0:
                losses = model.get_current_errors()
                t = (time.time() - iter_start_time) / opt.batchSize
                visualizer.print_current_errors(epoch, total_epoch, total_iteration, losses, t)
                if opt.display_id > 0:
                    visualizer.plot_current_errors(total_iteration, losses)

            if total_iteration % opt.eval_iters_freq == 0:
                model.eval() 
                if hasattr(model, 'eval_metric_name'):
                    eval_results = model.get_current_eval_results()  
                    visualizer.print_current_eval(epoch, total_iteration, eval_results)
                    if opt.display_id > 0:
                        visualizer.plot_current_score(total_iteration, eval_results)
                    
            # save the latest model every <save_latest_freq> iterations to the disk
            if total_iteration % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_iteration))
                model.save_networks('latest')

            # save the model every <save_iter_freq> iterations to the disk
            if total_iteration % opt.save_iters_freq == 0:
                print('saving the model of iterations %d' % total_iteration)
                model.save_networks(total_iteration)

            if total_iteration > max_iteration:
                keep_training = False
                break
            pbar.update(1)

        model.update_learning_rate()

        print('\nEnd training')
