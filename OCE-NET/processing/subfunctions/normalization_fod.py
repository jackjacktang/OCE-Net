#==============================================================================#
#  Author:       Dominik Müller, Xinyi Wang                                               #
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
# Internal libraries/scripts
from processing.subfunctions.abstract_subfunction import Abstract_Subfunction
import nibabel as nib
#-----------------------------------------------------#
#          Subfunction class: Normalization           #
#-----------------------------------------------------#
""" A Normalization Subfunction class which normalizes the intensity pixel values of an image using
    the Z-Score technique (default setting), through scaling to [0,1] or to grayscale [0,255].

Args:
    mode (string):          Mode which normalization approach should be performed.
                            Possible modi: "z-score", "minmax" or "grayscale"

Methods:
    __init__                Object creation function
    preprocessing:          Pixel intensity value normalization the imaging data
    postprocessing:         Do nothing
"""
class Normalization_FOD(Abstract_Subfunction):
    #---------------------------------------------#
    #                Initialization               #
    #---------------------------------------------#
    def __init__(self, mode="z-score"):
        self.mode = mode
        self.mean = None
        self.std = None
        self.min = None
        self.max = None

        self.brain_img = None
        self.brain_mask = None
        self.ref_img = None
        self.ref_img_2 = None
        self.ref_mask = None
        self.lesion_mask = None

    #---------------------------------------------#
    #                Preprocessing                #
    #---------------------------------------------#
    def preprocessing(self, sample, training=True):

        # Access image, ground truth and prediction data
        brain_img = sample.img_data.squeeze().copy()
        lesion_mask = sample.seg_data.squeeze()

        # ref mask: gt, healthy brain mask
        ref_img = brain_img.copy()
        ref_img_2 = brain_img.copy()
        ref_mask = ref_img.copy()
        ref_mask[ref_mask != 0] = 1

        # brain_img: unhealthy brain with holes
        # brain mask: unhealthy brain mask
        brain_img[lesion_mask == 1] = 0
        brain_mask = brain_img.copy()
        brain_mask[brain_mask != 0] = 1

        # Perform z-score normalization
        if self.mode == "z-score":
            # Compute mean and standard deviation
            mean = np.mean(brain_img[brain_mask == 1])
            std = np.std(brain_img[brain_mask == 1])
            self.mean = mean
            self.std = std
            # Scaling
            brain_img[brain_mask == 1] = (brain_img[brain_mask == 1] - mean) / std
            brain_img[brain_mask == 0] = 0
            # print(self.mode, sample.index, 'brain_img done.')
            ref_img[ref_mask != 0] = (ref_img[ref_mask == 1] - mean) / std
            ref_img[ref_mask == 0] = 0
            # print(self.mode, sample.index, 'ref_img done.')

            image_normalized = ref_img

        # Perform MinMax normalization between [0,1]
        elif self.mode == "minmax":
            # Identify minimum and maximum
            min_val = np.min(brain_img[brain_mask == 1])
            max_val = np.max(brain_img[brain_mask == 1])
            val_range = max_val - min_val
            self.min = min_val
            self.max = max_val
            # Scaling
            brain_img[brain_mask == 1] = (brain_img[brain_mask == 1] - min_val) / val_range
            brain_img[brain_mask == 0] = 0
            ref_img[ref_mask != 0] = (ref_img[ref_mask == 1] - min_val) / val_range
            ref_img[ref_mask == 0] = 0

            # image_normalized = brain_img
            image_normalized = ref_img

        elif self.mode == "grayscale":
            # Identify minimum and maximum
            max_value = np.max(brain_img[brain_mask == 1])
            min_value = np.min(brain_img[brain_mask == 1])
            # Scaling
            ref_img[ref_mask != 0] = (ref_img[ref_mask == 1] - min_value) / (max_value - min_value)
            ref_img[ref_mask == 0] = 0
            image_scaled = ref_img
            image_normalized = np.around(image_scaled * 255, decimals=0)

        else : raise NameError("Subfunction - Normalization: Unknown modus")
        # Update the sample with the normalized image
        sample.img_data = image_normalized
        self.brain_img = brain_img
        self.brain_mask = brain_mask
        self.ref_img = ref_img
        self.ref_img_2 = ref_img_2
        self.ref_mask = ref_mask
        self.lesion_mask = lesion_mask
        self.sample = sample

    #---------------------------------------------#
    #               Postprocessing                #
    #---------------------------------------------#
    def postprocessing(self, prediction):
        return prediction
