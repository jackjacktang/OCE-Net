"""
Training script for LG-Net.

See more details at https://github.com/jackjacktang/LG-Net/
"""
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from utils.utils import *
import torch
import os
import numpy as np

from data_loading.interfaces import HCP_FOD_norm_NIFTI_interface
from data_loading import HCP_FOD_Data_IO
from processing import Preprocessor_HCP_FOD, Data_Augmentation
from processing.subfunctions import Normalization_FOD

import matplotlib.pyplot as plt
import random
import subprocess

os.environ['KMP_DUPLICATE_LIB_OK']='True'
# device = torch.device('cpu')

torch.manual_seed(1)
np.random.seed(1)
random.seed(1)
torch.cuda.manual_seed_all(1)
torch.cuda.manual_seed(1)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

if __name__ == '__main__':

    opt = TrainOptions().parse()   # get training options

    # Initialize Data IO Interface for NIfTI data
    # We are using 4 classes due to [background, brain, lesion]
    interface = HCP_FOD_norm_NIFTI_interface(channels=opt.input_nc, classes=opt.output_nc)
    # Create Data IO object to load and write samples in the file structure
    data_io = HCP_FOD_Data_IO(interface, input_path=opt.dataroot, output_path="predictions_"+str(opt.name),
                          batch_path="fod_norm_tensor", delete_batchDir=False)

    # Create and configure the Data Augmentation class
    # data_aug = Data_Augmentation(cycles=10, scaling=False, rotations=False,
    #                              elastic_deform=False, mirror=False,
    #                              brightness=False, contrast=False, gamma=False,
    #                              gaussian_noise=False)
    data_aug = None

    # Create a pixel value normalization Subfunction for z-score scaling
    sf_zscore = Normalization_FOD(mode="z-score")
    # Assemble Subfunction classes into a list
    sf = [sf_zscore]
    # Create and configure the Preprocessor class
    pp = Preprocessor_HCP_FOD(data_io, data_aug=data_aug, batch_size=opt.batch_size, subfunctions=sf,
                      prepare_subfunctions=False, prepare_batches=False, analysis="patchwise-grid",
                      patch_shape=(64, 64, 64), save_coords=True)
    # Adjust the patch overlap for predictions
    pp.patchwise_overlap = (32, 32, 32)
    # pp = PreprocessorHcp(data_io, data_aug=data_aug, batch_size=opt.batch_size, subfunctions=sf,
    #                   prepare_subfunctions=True, prepare_batches=False,
    #                   analysis="patchwise-crop", patch_shape=(64, 64, 64))
    # # Adjust the patch overlap for predictions
    # pp.patchwise_overlap = (32, 32, 32)

    # Access all available samples in our file structure
    sample_list = data_io.get_indiceslist()
    sample_list.sort()

    if opt.phase == 'preprocess':
        # back up samples as pickles
        pp.run_backup(sample_list, training=True)
    elif opt.phase == 'splitfolds':
        pp.split_folds(sample_list, k_fold=5, evaluation_path="folds")
    elif opt.phase == 'train':
        fold_subdir = os.path.join("OCE-NET", "folds", "fold_" + str(opt.test_fold))
        # Obtain training and validation data set
        training, validation = pp.load_csv2fold(os.path.join(fold_subdir, "sample_list.csv"))

        torch.cuda.empty_cache()
        opt.pp = pp
        opt.data_io = data_io
        # opt.sample_list = sample_list
        opt.sample_list = training
        # opt.serial_batches = True

        dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
        dataset_size = len(dataset)    # get the number of images in the dataset.
        print('The number of training images = %d' % dataset_size)

        model = create_model(opt)      # create a model given opt.model and other options
        model.setup(opt)               # regular setup: load and print networks; create schedulers
        recorder = Recorder(opt)
        total_iters = 0                # the total number of training iterations

        loss_list = []
        for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
            tot_loss = 0.
            epoch_start_time = time.time()  # timer for entire epoch
            iter_data_time = time.time()    # timer for data loading per iteration
            epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch

            # traning
            for i, data in enumerate(dataset):  # inner loop within one epoch
                iter_start_time = time.time()  # timer for computation per iteration
                total_iters += opt.batch_size
                epoch_iter += opt.batch_size
                model.set_input(data)
                model.optimize_parameters()

                losses = model.get_current_losses()
                tot_loss += losses['R']
                # print(losses['R'])

                if total_iters % opt.display_freq == 0:
                    save_result = total_iters == 0
                    recorder.plot_current_losses(total_iters, losses)

                if total_iters % opt.print_freq == 0:
                    t_comp = (time.time() - iter_start_time) / opt.batch_size
                    t_data = iter_start_time - iter_data_time
                    recorder.print_current_losses(epoch, total_iters, losses, t_comp, t_data)

                if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                    print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                    save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                    model.save_networks(save_suffix)

                iter_data_time = time.time()
                break

            model.update_learning_rate()  # update learning rates at the end of every epoch.
            loss_list.append(tot_loss / (i+1))
            print('[TRAIN] loss:', tot_loss / (i+1))

            if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
                print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
                model.save_networks('latest')
                model.save_networks(epoch)

            print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
            if epoch >= 0:
                break