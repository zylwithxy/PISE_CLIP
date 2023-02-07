#!/bin/bash
python test.py --name=parsing_net_reduc_wo_text \
               --model=painet \
               --gpu_ids=0 \
               --dataroot /media/beast/WD2T/XUEYu/dataset_pose_transfer/Pose_transfer_codes/deepfashion \
               --batchSize 1 \
               --display_freq 50 \
               --use_reduc_layer True \
               --use_text False \
               --mask_choice both \
               --nThreads 2 \
               --use_masked_SPL1 True \
               --print_freq 10