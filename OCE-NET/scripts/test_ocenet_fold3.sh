set -ex
python3 /home/xinyi/OCE-NET/test.py \
--dataroot /home/data/HCP \
--checkpoints_dir ./checkpoints3 \
--name ocenet_fodnorm_zscore_f3 \
--model lesion_inpaint_5loss \
--input_nc 45 \
--output_nc 45 \
--init_type kaiming \
--dataset_mode fod \
--batch_size 4 \
--gpu_ids 1 \
--conv_type ocenet \
--test_fold 3 \
