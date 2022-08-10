import os
import torch
from data.base_dataset import BaseDataset
import numpy as np


class FODDataset(BaseDataset):
    """A dataset class for brain image dataset.

    It assumes that the directory '/path/to/data/train' contains brain image slices
    in torch.tensor format to speed up the I/O. Otherwise, you can load MRI brain images
    using nibabel package and preprocess the slices during loading period.
    """

    def __init__(self, opt):
        """
        Initialize this dataset class.
        """
        BaseDataset.__init__(self, opt)
        self.sub_list = os.listdir(opt.dataroot)
        self.opt = opt

        self.input_nc = self.opt.input_nc
        self.output_nc = self.opt.output_nc

        # set up training/validation
        pp = opt.pp
        shuffle_batches = opt.shuffle_batches
        if self.opt.phase == 'train':
            training = True
            validation = False
            self.sample_list_tr = opt.sample_list
            self.set_up(self.sample_list_tr, pp, training, validation, shuffle_batches)
        elif self.opt.phase == 'test':
            training = False
            validation = False
            self.sample_list_ts = opt.sample_list
            self.set_up([self.sample_list_ts[opt.eval_sample]], pp, training, validation, shuffle_batches)
        else:
            training = False
            validation = True
            self.sample_list_ts = opt.sample_list
            self.set_up([self.sample_list_ts[opt.eval_sample]], pp, training, validation, shuffle_batches)

    def set_up(self, sample_list, preprocessor, training=False, validation=False, shuffle=False):
        # Parse sample list
        if isinstance(sample_list, list) : self.sample_list = sample_list.copy()
        elif type(sample_list).__module__ == np.__name__ :
            self.sample_list = sample_list.tolist()
        else : raise ValueError("Sample list have to be a list or numpy array!")

        # Create a working environment from the handed over variables
        self.preprocessor = preprocessor
        self.training = training
        self.validation = validation
        self.shuffle = shuffle

        # Crop patches from samples
        if self.preprocessor.save_coords:
            self.coords_batches = preprocessor.run(sample_list, training, validation)
            # Load samples
            self.samples = {}
            for index in sample_list:
                # Load sample and process provided subfunctions on image data
                if not self.preprocessor.prepare_batches:
                    sample = self.opt.data_io.sample_loader(index, backup=False)
                else:
                    sample = self.opt.data_io.sample_loader(index, backup=True)
                if not self.preprocessor.prepare_subfunctions:
                    for sf in self.preprocessor.subfunctions:
                        sf.preprocessing(sample, training=training)
                    self.samples[index] = sample
                break
        else:
            self.batches = preprocessor.run(sample_list, training, validation)

    # Return the next batch for associated index
    def __getitem__(self, idx):

        if self.preprocessor.save_coords:
            self.now_sample = self.samples[self.coords_batches[0][idx]['index']]

            x_start = self.coords_batches[0][idx]['x_start']
            x_end = self.coords_batches[0][idx]['x_end']
            y_start = self.coords_batches[0][idx]['y_start']
            y_end = self.coords_batches[0][idx]['y_end']
            z_start = self.coords_batches[0][idx]['z_start']
            z_end = self.coords_batches[0][idx]['z_end']

            x = torch.from_numpy(self.now_sample.img_data[x_start:x_end, y_start:y_end, z_start:z_end])
            y = torch.from_numpy(self.now_sample.seg_data[x_start:x_end, y_start:y_end, z_start:z_end])
        else:
            # w h d c
            x, y = torch.from_numpy(self.batches[0][idx]), torch.from_numpy(self.batches[1][idx])

        # w h d c
        brain = x.clone()
        lesion = y.clone()

        gt = brain.clone()
        brain[lesion[:,:,:,0] == 1] = 0
        # w h d c -> c w h d
        brain = brain.permute(3,0,1,2)
        lesion = lesion.permute(3,0,1,2)
        gt = gt.permute(3,0,1,2)

        if brain.shape[1:-1] != lesion.shape[1:-1]:
            print('not the same shape', brain.shape, lesion.shape)

        ret = {}
        ret['brain'] = brain
        ret['lesion'] = lesion
        ret['gt'] = gt
        return ret

    # Return the number of batches for one epoch
    def __len__(self):
        return len(self.coords_batches[0])

    # At every epoch end: Shuffle batchPointer list and reset sample_list
    def on_epoch_end(self):
        if self.shuffle and self.training:
            if self.preprocessor.prepare_batches:
                np.random.shuffle(self.batchpointers)
            else:
                np.random.shuffle(self.sample_list)

    def modify_commandline_options(parser, is_train):
        """
        Add any new dataset-specific options, and rewrite default values for existing options.
        """
        parser.add_argument('--test_fold', default='1', type=int, help='test fold number')
        parser.add_argument('--shuffle_batches', default=True, type=bool, help='whether batch order should be shuffled or not ?')
        parser.add_argument('--iterations', default=None, type=int, help='Number of iterations (batches) in a single epoch.')
        return parser
