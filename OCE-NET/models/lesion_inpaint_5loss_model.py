from .loss import *
from utils import *
from .networks import define_net
from .base_model import BaseModel
import torch
import nibabel as nib
import os

class LesionInpaint5LossModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """
        Add new LGC-specific options.
        """
        parser.add_argument('--lambda_lesion', type=float, default=10, help='weight for lesion area L1 loss')
        parser.add_argument('--lambda_tissue', type=float, default=1, help='weight for valid tissue L1 loss')
        parser.add_argument('--conv_type', type=str, default='unet')

        opt, _ = parser.parse_known_args()
        return parser

    def __init__(self, opt):
        """
        Initialize this Lesion Inpaint class.
        """
        BaseModel.__init__(self, opt)
        self.loss_names = ['R']
        self.model_names = ['inpaint']
        self.opt = opt

        # define the inpainting network
        self.net_inpaint = define_net(self.opt.input_nc, opt.output_nc, opt.conv_type, opt.norm,
                                              self.opt.init_type, self.opt.init_gain, gpu_ids=self.opt.gpu_ids)

        if self.opt.isTrain:
            # define the loss functions
            self.lesion_loss = L1Loss(weight=opt.lambda_lesion)
            # define the optimizer
            self.optimizer_inpaint = torch.optim.Adam(self.net_inpaint.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_inpaint)
        else:
            # define the loss functions
            self.lesion_loss = L1Loss(weight=opt.lambda_lesion)



    def set_input(self, input):
        """
        Read the data of input from dataloader then
        """
        if self.isTrain:
            self.brain = input['brain'].to(self.device)  # get masked brain, i.e. lesion areas have value of 0
            self.lesion = input['lesion'].to(self.device)
            self.gt = input['gt'].to(self.device)  # get original brain
        else:
            self.brain = input['brain'].to(self.device)  # get masked brain, i.e. lesion areas have value of 0
            self.lesion = input['lesion'].to(self.device)
            self.gt = input['gt'].to(self.device)  # get original brain


    def forward(self):
        """
        Run forward pass
        """
        # combine pseudo slice and corresponding mask as input
        self.inpainted = self.net_inpaint(self.brain)
        self.out1 = self.inpainted[:,0:1,:,:,:]
        self.out2 = self.inpainted[:,1:6,:,:,:]
        self.out3 = self.inpainted[:,6:15, :, :, :]
        self.out4 = self.inpainted[:,15:28, :, :, :]
        self.out5 = self.inpainted[:,28:45, :, :, :]

        if not self.opt.isTrain:
            self.loss_lesion_1 = self.lesion_loss(self.lesion * self.out1, self.lesion * self.gt[:, 0:1, :, :, :])
            self.loss_lesion_2 = self.lesion_loss(self.lesion * self.out2, self.lesion * self.gt[:, 1:6, :, :, :])
            self.loss_lesion_3 = self.lesion_loss(self.lesion * self.out3, self.lesion * self.gt[:, 6:15, :, :, :])
            self.loss_lesion_4 = self.lesion_loss(self.lesion * self.out4, self.lesion * self.gt[:, 15:28, :, :, :])
            self.loss_lesion_5 = self.lesion_loss(self.lesion * self.out5, self.lesion * self.gt[:, 28:45, :, :, :])
            self.loss_R = self.loss_lesion_1 + self.loss_lesion_2 + self.loss_lesion_3 + self.loss_lesion_4 + self.loss_lesion_5
        return self.inpainted

    def backward_inpaint(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # calculate loss given the input and intermediate results
        # calculate reconstruction loss
        self.loss_lesion_1 = self.lesion_loss(self.lesion * self.out1, self.lesion * self.gt[:,0:1,:,:,:])
        self.loss_lesion_2 = self.lesion_loss(self.lesion * self.out2, self.lesion * self.gt[:,1:6,:,:,:])
        self.loss_lesion_3 = self.lesion_loss(self.lesion * self.out3, self.lesion * self.gt[:,6:15,:,:,:])
        self.loss_lesion_4 = self.lesion_loss(self.lesion * self.out4, self.lesion * self.gt[:,15:28,:,:,:])
        self.loss_lesion_5 = self.lesion_loss(self.lesion * self.out5, self.lesion * self.gt[:,28:45,:,:,:])
        self.loss_R = self.loss_lesion_1 + self.loss_lesion_2 + self.loss_lesion_3 + self.loss_lesion_4 + self.loss_lesion_5
        self.loss_R.backward()

    def optimize_parameters(self):
        """Update network weights; it will be called in every training iteration."""
        self.forward()

        # update optimizer of the inpainting network
        self.optimizer_inpaint.zero_grad()
        self.backward_inpaint()
        self.optimizer_inpaint.step()


