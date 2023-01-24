python train.py --name=parsing_net_reduc \
                --model=painet \
                --gpu_ids=1 \
                --dataroot /media/beast/WD2T/XUEYu/dataset_pose_transfer/Pose_transfer_codes/deepfashion \
                --batchSize 8 \
                --niter 750000 \
                --display_freq 200