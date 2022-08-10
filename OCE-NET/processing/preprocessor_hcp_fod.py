#==============================================================================#
#  Author:       Dominik Müller                                                #
#  Copyright:    2020 IT-Infrastructure for Translational Medical Research,    #
#                University of Augsburg                                        #
#                                                                              #
#  This program is free software: you can redistribute it and/or modify        #
#  it under the terms of the GNU General Public License as published by        #
#  the Free Software Foundation, either version 3 of the License, or           #
#  (at your option) any later version.                                         #
#                                                                              #
#  This program is distributed in the hope that it will be useful,             #
#  but WITHOUT ANY WARRANTY; without even the implied warranty of              #
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the               #
#  GNU General Public License for more details.                                #
#                                                                              #
#  You should have received a copy of the GNU General Public License           #
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.       #
#==============================================================================#
#-----------------------------------------------------#
#                   Library imports                   #
#-----------------------------------------------------#
# External libraries
import numpy as np
import threading
# Internal libraries/scripts
from .data_augmentation import Data_Augmentation
from utils.patch_operations import slice_matrix, concat_matrices, pad_patch, crop_patch
import nibabel as nib
import os
import csv


#-----------------------------------------------------#
#                 Preprocessor class                  #
#-----------------------------------------------------#
# Class to handle all preprocessing functionalities
class Preprocessor_HCP_FOD:
    #---------------------------------------------#
    #                Initialization               #
    #---------------------------------------------#
    """ Initialization function for creating a Preprocessor object.
    This class provides functionality for handling all preprocessing methods. This includes diverse
    optional processing subfunctions like resampling, clipping, normalization or custom subfcuntions.
    This class processes the data into batches which are ready to be used for training, prediction and validation.

    The user is only required to create an instance of the Preprocessor class with the desired specifications
    and Data IO instance (optional also Data Augmentation instance).

    Args:
        data_io (Data_IO):                      Data IO class instance which handles all I/O operations according to the user
                                                defined interface.
        batch_size (integer):                   Number of samples inside a single batch.
        subfunctions (list of Subfunctions):    List of Subfunctions class instances which will be SEQUENTIALLY executed on the data set.
                                                (clipping, normalization, resampling, ...)
        data_aug (Data_Augmentation):           Data Augmentation class instance which performs diverse data augmentation techniques.
                                                If no Data Augmentation is provided, an instance with default settings will be created.
                                                Use data_aug=None, if you want no data augmentation at all.
        prepare_subfunctions (boolean):         Should all subfunctions be prepared and backup to disk before starting the batch generation
                                                (True), or should the subfunctions preprocessing be performed during runtime? (False).
        prepare_batches (boolean):              Should all batches be prepared and backup to disk before starting the training (True),
                                                or should the batches be created during runtime? (False).
        analysis (string):                      Modus selection of analysis type. Options:
                                                - "fullimage":      Analysis of complete image data
                                                - "patchwise-crop": Analysis of random cropped patches from the image
                                                - "patchwise-grid": Analysis of patches by splitting the image into a grid
        patch_shape (integer tuple):            Size and shape of a patch. The variable has to be defined as a tuple.
                                                For Example: (64,128,128) for 64x128x128 patch cubes.
                                                Be aware that the x-axis represents the number of slices in 3D volumes.
                                                This parameter will be redundant if fullimage or patchwise-crop analysis is selected!!
    """
    def __init__(self, data_io, batch_size, subfunctions=[],
                 data_aug=Data_Augmentation(), prepare_subfunctions=False,
                 prepare_batches=False, analysis="patchwise-crop",
                 patch_shape=None, save_coords=False):
        print('Preprocessor_HCP_FOD is created.')
        # Parse Data Augmentation
        if isinstance(data_aug, Data_Augmentation):
            self.data_augmentation = data_aug
        else:
            self.data_augmentation = None
        # Exception: Analysis parameter check
        analysis_types = ["patchwise-crop", "patchwise-grid", "fullimage"]
        if not isinstance(analysis, str) or analysis not in analysis_types:
            raise ValueError('Non existent analysis type in preprocessing.')
        # Exception: Patch-shape parameter check
        if (analysis == "patchwise-crop" or analysis == "patchwise-grid") and \
            not isinstance(patch_shape, tuple):
            raise ValueError("Missing or wrong patch shape parameter for " + \
                             "patchwise analysis.")
        # Parse parameter
        self.data_io = data_io
        self.batch_size = batch_size
        self.subfunctions = subfunctions
        self.prepare_subfunctions = prepare_subfunctions
        self.prepare_batches = prepare_batches
        self.analysis = analysis
        self.patch_shape = patch_shape
        self.save_coords = save_coords

    #---------------------------------------------#
    #               Class variables               #
    #---------------------------------------------#
    patchwise_overlap = (0,0,0)             # In patchwise_analysis, an overlap can be defined between adjuncted patches.
    patchwise_skip_blanks = True           # In patchwise_analysis, patches, containing only the background annotation,
                                            # can be skipped with this option. This result into only
                                            # training on relevant patches and ignore patches without any information.
    patchwise_skip_class = 0                # Class, which will be skipped if patchwise_skip_blanks is True
    img_queue = []                          # Intern queue of already processed and data augmentated images or segmentations.
                                            # The function create_batches will use this queue to create batches
    cache = dict()                          # Cache additional information and data for patch assembling after patchwise prediction
    thread_lock = threading.Lock()          # Create a threading lock for multiprocessing

    #---------------------------------------------#
    #               Prepare Batches               #
    #---------------------------------------------#
    # Preprocess data and prepare the batches for a given list of indices
    def run(self, indices_list, training=True, validation=False):
        # Iterate over all samples
        self.img_queue = []
        self.coord_queue = []
        img_list = []
        seg_list = []

        print('---------------preprocessing-----------------')
        i = 0
        for index in indices_list:
            # Load sample
            if not self.prepare_batches:
                sample = self.data_io.sample_loader(index, backup=False)
            # Load sample from backup
            else:
                sample = self.data_io.sample_loader(index, backup=True)

            # Decide if data augmentation should be performed
            if training and not validation and self.data_augmentation is not None:
                data_aug = True
            else:
                data_aug = False

            # Run Fullimage analysis
            if self.analysis == "fullimage":
                ready_data, img_data, seg_data = self.analysis_fullimage(sample, training,
                                                     data_aug)
            # Run patchwise cropping analysis
            elif self.analysis == "patchwise-crop" and training:
                ready_data, img_data, seg_data = self.analysis_patchwise_crop(sample, data_aug)
            # Run patchwise grid analysis
            else:
                if not training:
                    self.cache["shape_" + str(index)] = sample.img_data.shape
                ready_data, img_data, seg_data, coords_img_data, coords_seg_data = self.analysis_patchwise_grid(sample, training, data_aug, index)

            # Create threading lock to avoid parallel access
            with self.thread_lock:
                # Put the preprocessed data at the image queue end
                if self.save_coords:
                    ready_coord = list(zip(coords_img_data, coords_seg_data))
                    self.coord_queue.extend(ready_coord)
                else:
                    self.img_queue.extend(ready_data)

            print('crop patches from: ', i, index, sample.img_data.shape, len(ready_data), len(ready_coord))
            i+=1
            break
        print('-----------end preprocessing-----------------')

        if not self.save_coords:
            batches = self.collect_batch(self.img_queue)
            return batches
        else:
            coords_batches = self.collect_batch(self.coord_queue)
            return coords_batches


    # Gather patches and combine them into a batch
    def collect_batch(self, img_queue):
        # Iterate over the images which will be relocated in a batch
        img_list = []
        seg_list = []
        for j in range(len(img_queue)):
            # Access these images
            img = img_queue[j][0]
            if len(img_queue[j]) == 2:
                seg = img_queue[j][1]
            else:
                seg = None
            # Add images to associated list
            img_list.append(img)
            seg_list.append(seg)

        # Combine images into a batch
        batch_img = np.stack(img_list, axis=0)
        if any(elem is None for elem in seg_list):
            batch_seg = None
        else:
            batch_seg = np.stack(seg_list, axis=0)
        # Combine batch_img and batch_seg into a tuple
        batch = (batch_img, batch_seg)
        # Return finished batch
        return batch

    #---------------------------------------------#
    #          Prediction Postprocessing          #
    #---------------------------------------------#
    # Postprocess prediction data
    def postprocessing(self, sample, prediction):
        # Reassemble patches into original shape for patchwise analysis
        if self.analysis == "patchwise-crop" or \
            self.analysis == "patchwise-grid":
            # Check if patch was padded
            slice_key = "slicer_" + str(sample)
            if slice_key in self.cache:
                prediction = crop_patch(prediction, self.cache[slice_key])
            # Load cached shape & Concatenate patches into original shape
            seg_shape = self.cache.pop("shape_" + str(sample))
            prediction = concat_matrices(patches=prediction,
                                    image_size=seg_shape,
                                    window=self.patch_shape,
                                    overlap=self.patchwise_overlap,
                                    three_dim=self.data_io.interface.three_dim)
        # For fullimages remove the batch axis
        else : prediction = np.squeeze(prediction, axis=0)
        # Run Subfunction postprocessing on the prediction
        for sf in reversed(self.subfunctions):
            prediction = sf.postprocessing(prediction)
        # Return postprocessed prediction
        return prediction


    # backup samples as pickles
    def run_backup(self, indices_list, training=True):
        # Iterate over all samples
        for index in indices_list:
            if index != '.DS_Store':
                # Load sample
                sample = self.data_io.sample_loader(index, load_seg=True, load_pred=False, load_gt=False)
                # Backup sample as pickle to disk
                self.data_io.backup_sample(sample)
                print('backup case: ', index)
                # break
        print('-----------------------------------------------------------')

    # -----------------------------------------------------#
    #           Splitted k-fold Cross-Validation          #
    # -----------------------------------------------------#
    """ Function for splitting a data set into k-folds. The splitting will be saved
        in files, which can be used for running a single fold run.
        In contrast to the normal cross_validation() function, this allows running
        folds parallelized on multiple GPUs.

    Args:
        sample_list (list of indices):          A list of sample indicies which will be used for validation.
        k_fold (integer):                       The number of k-folds for the Cross-Validation. By default, a
                                                3-fold Cross-Validation is performed.
    """
    def split_folds(self, sample_list, k_fold=5, evaluation_path="evaluation"):
        # Randomly permute the sample list
        # samples_permuted = np.random.permutation(sample_list)
        samples_permuted = sample_list
        # Split sample list into folds
        folds = np.array_split(samples_permuted, k_fold)
        fold_indices = list(range(len(folds)))
        # Iterate over each fold
        for i in fold_indices:
            # Subset training and validation data set
            training = np.concatenate([folds[x] for x in fold_indices if x != i],
                                      axis=0)
            validation = folds[i]
            # Initialize evaluation subdirectory for current fold
            subdir = self.create_directories(evaluation_path, "fold_" + str(i))
            fold_cache = os.path.join(subdir, "sample_list.csv")
            # Write sampling to disk
            self.write_fold2csv(fold_cache, training, validation)

    # Create an evaluation subdirectory and change path
    def create_directories(self, eval_path, subeval_path=None):
        # Create evaluation directory if necessary
        if not os.path.exists(eval_path):
            os.mkdir(eval_path)
        # Create evaluation subdirectory if necessary
        if subeval_path is not None:
            # Concatenate evaluation subdirectory path if present
            subdir = os.path.join(eval_path, subeval_path)
            # Set up the evaluation subdirectory
            if not os.path.exists(subdir):
                os.mkdir(subdir)
            # Return path to evaluation subdirectory
            return subdir

    # -----------------------------------------------------#
    #                   CSV Management                    #
    # -----------------------------------------------------#
    # Subfunction for writing a fold sampling to disk
    def write_fold2csv(self, file_path, training, validation):
        with open(file_path, "w") as csvfile:
            writer = csv.writer(csvfile, delimiter=" ")
            writer.writerow(["TRAINING:"] + list(training))
            writer.writerow(["VALIDATION:"] + list(validation))

    # Subfunction for loading a fold sampling from disk
    def load_csv2fold(self, file_path):
        training = None
        validation = None
        with open(file_path, "r") as csvfile:
            reader = csv.reader(csvfile, delimiter=" ")
            for row in reader:
                if not training:
                    training = row[1:]
                else:
                    validation = row[1:]
        return training, validation

    #---------------------------------------------#
    #           Patch-wise grid Analysis          #
    #---------------------------------------------#
    def analysis_patchwise_grid(self, sample, training, data_aug, index=None):
        # Slice image into patches
        patches_img, coords_img = slice_matrix(sample.img_data, self.patch_shape,
                                   self.patchwise_overlap,
                                   self.data_io.interface.three_dim, index,
                                   save_coords=self.save_coords)

        patches_seg, coords_seg = slice_matrix(sample.seg_data, self.patch_shape,
                                   self.patchwise_overlap,
                                   self.data_io.interface.three_dim, index,
                                   save_coords=self.save_coords)

        # Skip blank patches (only background)
        if training and self.patchwise_skip_blanks:
            # Iterate over each patch
            for i in reversed(range(0, len(patches_seg))):
                # IF patch DON'T contain any non background class -> remove it
                if np.sum(patches_seg[i]) == 0:
                    del patches_img[i]
                    del patches_seg[i]
                    if self.save_coords:
                        del coords_img[i]
                        del coords_seg[i]
        # Concatenate a list of patches into a single numpy array
        img_data = np.stack(patches_img, axis=0)
        seg_data = np.stack(patches_seg, axis=0)
        # coordinates
        if self.save_coords:
            coords_img_data = np.stack(coords_img, axis=0)
            coords_seg_data = np.stack(coords_seg, axis=0)

        # Pad patches if necessary
        if img_data.shape[1:-1] != self.patch_shape and training:
            img_data = pad_patch(img_data, self.patch_shape, return_slicer=False)
            seg_data = pad_patch(seg_data, self.patch_shape, return_slicer=False)
        elif img_data.shape[1:-1] != self.patch_shape and not training:
            img_data, slicer = pad_patch(img_data, self.patch_shape,
                                         return_slicer=True)
            seg_data = pad_patch(seg_data, self.patch_shape, return_slicer=False)
            self.cache["slicer_" + str(sample.index)] = slicer
        # Run data augmentation
        if data_aug:
            img_data, seg_data = self.data_augmentation.run(img_data, seg_data)
        else:
            img_data, seg_data = img_data, seg_data

        # Create tuple of preprocessed data
        ready_data = list(zip(img_data, seg_data))

        # Return preprocessed data tuple
        if self.save_coords:
            return ready_data, img_data, seg_data, coords_img_data, coords_seg_data
        else:
            return ready_data, img_data, seg_data

    #---------------------------------------------#
    #           Patch-wise crop Analysis          #
    #---------------------------------------------#
    def analysis_patchwise_crop(self, sample, data_aug):
        # If skipping blank patches is active
        if self.patchwise_skip_blanks:
            # Slice image and segmentation into patches
            patches_img = slice_matrix(sample.img_data, self.patch_shape,
                                       self.patchwise_overlap,
                                       self.data_io.interface.three_dim)
            patches_seg = slice_matrix(sample.seg_data, self.patch_shape,
                                       self.patchwise_overlap,
                                       self.data_io.interface.three_dim)
            # Skip blank patches (only background)
            for i in reversed(range(0, len(patches_seg))):
                # IF patch DON'T contain any non background class -> remove it
                if not np.any(patches_seg[i][...,self.patchwise_skip_class] != 1):
                    del patches_img[i]
                    del patches_seg[i]
            # Select a random patch
            pointer = np.random.randint(0, len(patches_img))
            img = patches_img[pointer]
            seg = patches_seg[pointer]
            # Expand image dimension to simulate a batch with one image
            img_data = np.expand_dims(img, axis=0)
            seg_data = np.expand_dims(seg, axis=0)
            # Pad patches if necessary
            if img_data.shape[1:-1] != self.patch_shape:
                img_data = pad_patch(img_data, self.patch_shape,
                                     return_slicer=False)
                seg_data = pad_patch(seg_data, self.patch_shape,
                                     return_slicer=False)
            # Run data augmentation
            if data_aug:
                img_data, seg_data = self.data_augmentation.run(img_data,
                                                                seg_data)
            else:
                # print('data aug', img_data.shape, seg.shape)
                img_data, seg_data = img_data.permute(0, 4, 1, 2, 3), seg_data.permute(0, 4, 1, 2, 3)
                # If skipping blank is not active -> random crop
        else:
            # Access image and segmentation data
            img = sample.img_data
            seg = sample.seg_data

            # If no data augmentation should be performed
            # -> create Data Augmentation instance without augmentation methods
            if not data_aug or self.data_augmentation is None:
                cropping_data_aug = Data_Augmentation(cycles=1 ,
                                            scaling=False, rotations=False,
                                            elastic_deform=False, mirror=False,
                                            brightness=False, contrast=False,
                                            gamma=False, gaussian_noise=False)
            else : cropping_data_aug = self.data_augmentation

            # Configure the Data Augmentation instance to cropping
            cropping_data_aug.cropping = True
            cropping_data_aug.cropping_patch_shape = self.patch_shape

            # # Expand image dimension to simulate a batch with one image
            img_data = np.expand_dims(img, axis=0)
            seg_data = np.expand_dims(seg, axis=0)
            # Run data augmentation and cropping
            img_data, seg_data = cropping_data_aug.run(img_data, seg_data)
        # Create tuple of preprocessed data
        ready_data = list(zip(img_data, seg_data))
        # Return preprocessed data tuple
        return ready_data, img_data, seg_data

    #---------------------------------------------#
    #             Full-Image Analysis             #
    #---------------------------------------------#
    def analysis_fullimage(self, sample, training, data_aug):
        # Access image and segmentation data
        img = sample.img_data
        seg = sample.seg_data
        # Expand image dimension to simulate a batch with one image
        img_data = np.expand_dims(img, axis=0)
        seg_data = np.expand_dims(seg, axis=0)
        # Run data augmentation
        if data_aug:
            img_data, seg_data = self.data_augmentation.run(img_data, seg_data)
        # Create tuple of preprocessed data
        ready_data = list(zip(img_data, seg_data))
        # Return preprocessed data tuple
        return ready_data, img_data, seg_data
