import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

from omegaconf import OmegaConf

from typing import Dict, List, Tuple, Union


from COIGAN.training.losses.masked_losses import SmoothMaskedL1, ResNetPLSmoothMasked
from COIGAN.training.losses.perceptual import ResNetPL
from COIGAN.training.losses.common_losses import (
    d_logistic_loss,
    d_r1_loss,
    g_nonsaturating_loss,
    g_path_regularize
)
from COIGAN.utils.debug_utils import (
    check_nan
)

LOGGER = logging.getLogger(__name__)


class CoiganLossManager:

    def __init__(self,
        generator_losses,
        generator_reg,
        discriminator_losses,
        discriminator_reg,
        generator,
        discriminator,
        g_optimizer,
        d_optimizer,
        device,
        use_ref_disc = False,
        ref_discriminator = None,
        ref_d_optimizer = None,
        ref_discriminator_losses = None,
        ref_discriminator_reg = None
    ):
        """
        Init method of the CoiganLossManager class.

        Args:
            generator_losses (Dict): the generator's losses configuration
            generator_reg (Dict): the generator's regularization configuration
            discriminator_losses (Dict): the discriminator's losses configuration
            discriminator_reg (Dict): the discriminator's regularization configuration
            generator (nn.Module): the generator
            discriminator (nn.Module): the discriminator
            g_optimizer (torch.optim.Optimizer): the generator's optimizer
            d_optimizer (torch.optim.Optimizer): the discriminator's optimizer
            device (torch.device): the device to use

            ref_discriminator (nn.Module): the reference discriminator
            ref_d_optimizer (torch.optim.Optimizer): the reference discriminator's optimizer
            ref_discriminator_losses (Dict): the reference discriminator's losses configuration
            ref_discriminator_reg (Dict): the reference discriminator's regularization configuration

        
        TODO: add the regularization for the generator
        """

        self.device = device
        self.use_ref_disc = use_ref_disc
        self.metrics = {}

        ##################################################################
        ### Generator losses
        ##################################################################
        self.init_generator_losses(**generator_losses)
        self.init_generator_regularizations(**generator_reg)
        self.generator = generator
        self.g_opt = g_optimizer
        
        ##################################################################
        ### Discriminator losses
        ##################################################################
        self.init_discriminator_losses(**discriminator_losses)
        self.init_discriminator_regularizations(**discriminator_reg)
        self.discriminator = discriminator
        self.d_opt = d_optimizer

        ##################################################################
        ### Ref discriminator losses
        ##################################################################
        if self.use_ref_disc:
            self.init_ref_discriminator_losses(**ref_discriminator_losses)
            self.init_ref_discriminator_regularizations(**ref_discriminator_reg)
            self.ref_discriminator = ref_discriminator
            self.ref_d_opt = ref_d_optimizer


    def init_discriminator_losses(
        self,
        logistic: Dict = None
    ):
        """
        Init the discriminator losses.

        Args:
            lsgan (Dict): the lsgan loss configuration
            hinge (Dict): the hinge loss configuration

        """

        self.loss_logistic = None
        if logistic is not None and logistic['weight'] > 0:
            self.loss_logistic = d_logistic_loss
            self.loss_logistic_weight = logistic['weight']


    def init_discriminator_regularizations(
        self,
        d_reg_every: int = 16,
        r1: Dict = None
    ):
        """
        Discriminator regularization.

        Args:
            d_reg_every (int): the number of steps between each regularization
            r1 (Dict): the r1 regularization configurations
        """
        self.d_reg_every = d_reg_every
        self.d_reg_steps_count = 0

        self.r1_reg = None
        if r1 is not None and r1['weight'] > 0:
            self.r1_reg = d_r1_loss
            self.r1_reg_weight = r1['weight']
        

    def discriminator_regularization(
        self,
        real_input
    ):  # sourcery skip: extract-method, merge-nested-ifs, swap-nested-ifs
        """
        Compute the discriminator regularization.
        NOTE: the real input must have the gradients enabled.

        Args:
            real_pred (Tensor): the real prediction
            real_input (Tensor): the real input
        
        """

        d_regs = {}

        if self.d_reg_steps_count % self.d_reg_every == 0:
            if self.r1_reg is not None:
                
                real_input.requires_grad = True
                real_pred = self.discriminator(real_input)
                real_pred = real_pred[0] if isinstance(real_pred, tuple) else real_pred

                # NOTE: the real_pred[0] multiplied by 0 is used to include the gradient of the discriminator in the graph, without uue its value
                # without it ddp will trow an error considering that some gradients in the graph are not used!!!
                d_regs["d_r1_loss"] = (self.r1_reg(real_pred, real_input) * self.r1_reg_weight * self.d_reg_every + (0 * real_pred[0]))[0]
                #check_nan(d_regs['d_r1_loss'])

                # apply the regularization
                self.discriminator.zero_grad()
                d_regs["d_r1_loss"].backward()
                self.d_opt.step()

        self.d_reg_steps_count += 1
        self.metrics.update(d_regs)


    def discriminator_loss(self, fake_score, real_score):
        """
        Compute the discriminator loss.

        Args:
            fake_score (Tensor): the fake score
            real_score (Tensor): the real score

        Returns:
            Tensor: the discriminator loss

        NOTE: there is only one discriminator loss at this time
            if more losses will be added the method need some changes.

        """
        discriminator_losses = {}
        if self.loss_logistic is not None:
            discriminator_losses["d_logistic_loss"] = self.loss_logistic(real_score, fake_score) * self.loss_logistic_weight
            #check_nan(discriminator_losses['d_logistic_loss'])

            # update the discriminator
            self.discriminator.zero_grad()
            discriminator_losses["d_logistic_loss"].backward()
            self.d_opt.step()
        
        self.metrics.update(discriminator_losses)
    

    def init_ref_discriminator_losses(
        self,
        logistic: Dict = None
    ):
        """
        Init the ref discriminator losses.

        Args:
            logistic (Dict): the logistic loss configuration

        """

        self.loss_ref_logistic = None
        if logistic is not None and logistic['weight'] > 0:
            self.loss_ref_logistic = d_logistic_loss
            self.loss_ref_logistic_weight = logistic['weight']


    def init_ref_discriminator_regularizations(
        self,
        d_reg_every: int = 16,
        r1: Dict = None
    ):
        """
        Ref Discriminator regularization.

        Args:
            d_reg_every (int): the number of steps between each regularization
            r1 (Dict): the r1 regularization configurations
        """
        self.ref_d_reg_every = d_reg_every
        self.ref_d_reg_steps_count = 0

        self.ref_r1_reg = None
        if r1 is not None and r1['weight'] > 0:
            self.ref_r1_reg = d_r1_loss
            self.ref_r1_reg_weight = r1['weight']
        

    def ref_discriminator_regularization(
        self,
        real_input
    ):  # sourcery skip: extract-method, merge-nested-ifs, swap-nested-ifs
        """
        Compute the ref discriminator regularization.
        NOTE: the real input must have the gradients enabled.

        Args:
            real_input (Tensor): the real input
        
        """

        ref_d_regs = {}

        if self.ref_d_reg_steps_count % self.ref_d_reg_every == 0:
            if self.ref_r1_reg is not None:
                
                real_input.requires_grad = True
                real_pred = self.ref_discriminator(real_input)
                real_pred = real_pred[0] if isinstance(real_pred, tuple) else real_pred

                # NOTE: the real_pred[0] multiplied by 0 is used to include the gradient of the discriminator in the graph, without uue its value
                # without it ddp will trow an error considering that some gradients in the graph are not used!!!
                ref_d_regs["ref_d_r1_loss"] = (self.ref_r1_reg(real_pred, real_input) * self.ref_r1_reg_weight * self.ref_d_reg_every + (0 * real_pred[0]))[0]
                #check_nan(ref_d_regs['ref_d_r1_loss'])

                # apply the regularization
                self.ref_discriminator.zero_grad()
                ref_d_regs["ref_d_r1_loss"].backward()
                self.ref_d_opt.step()

        self.ref_d_reg_steps_count += 1
        self.metrics.update(ref_d_regs)


    def ref_discriminator_loss(self, fake_score, real_score):
        """
        Compute the ref discriminator loss.

        Args:
            fake_score (Tensor): the fake score
            real_score (Tensor): the real score

        Returns:
            Tensor: the discriminator loss

        NOTE: there is only one discriminator loss at this time
            if more losses will be added the method need some changes.

        """
        ref_d_loss = {}
        if self.loss_logistic is not None:
            ref_d_loss["ref_d_logistic_loss"] = self.loss_ref_logistic(real_score, fake_score) * self.loss_ref_logistic_weight
            #check_nan(ref_d_loss['ref_d_logistic_loss'])

            # update the discriminator
            self.ref_discriminator.zero_grad()
            ref_d_loss["ref_d_logistic_loss"].backward()
            self.ref_d_opt.step()
        
        self.metrics.update(ref_d_loss)
    

    def init_generator_losses(
        self,
        reduction: str = 'mean',
        l1: Dict = None,
        l1_smooth_masked = None,
        mse: Dict = None,
        feature_matching: Dict = None,
        resnet_pl: Dict = None,
        resnet_pl_smooth_masked = None,
        adversarial: Dict = None,
        ref_adversarial: Dict = None
    ):
        """
        Init the generator losses.

        Args:
            reduction (str): the reduction method
            l1 (Dict): the l1 loss configuration
            mse (Dict): the mse loss configuration
            feature_matching (Dict): the feature matching loss configuration
            resnet_pl (Dict): the perceptual loss configuration
            adversarial (Dict): the adversarial loss configuration

        """
        # check there are at least one loss
        if  l1 is None and mse is None and feature_matching is None and resnet_pl is None:
            raise ValueError('At least one generator loss must be specified')

        self.gen_loss_reduction = reduction

        self.loss_l1 = None
        if l1 is not None and l1['weight'] > 0:
            self.loss_l1 = nn.L1Loss(reduction='mean')
            self.loss_l1_weight = l1['weight']
        
        self.loss_l1_smooth_masked = None
        if l1_smooth_masked is not None and l1_smooth_masked['weight'] > 0:
            self.loss_l1_smooth_masked_weight = l1_smooth_masked['weight']
            self.loss_l1_smooth_masked = SmoothMaskedL1(
                device=self.device,
                **l1_smooth_masked.kwargs
                )

        self.loss_mse = None
        if mse is not None and mse['weight'] > 0:
            self.loss_mse = nn.MSELoss(reduction='mean')
            self.loss_mse_weight = mse['weight']
        
        self.loss_resnet_pl = None
        if resnet_pl is not None and resnet_pl['weight'] > 0:
            self.loss_resnet_pl = ResNetPL(**resnet_pl.kwargs).to(self.device)
            self.loss_resnet_pl_weight = resnet_pl['weight']
        
        self.loss_resnet_pl_smooth_masked = None
        if resnet_pl_smooth_masked is not None and resnet_pl_smooth_masked['weight'] > 0:
            self.loss_resnet_pl_smooth_masked = ResNetPLSmoothMasked(
                device=self.device,
                **resnet_pl_smooth_masked.kwargs
                )
            self.loss_resnet_pl_smooth_masked_weight = resnet_pl_smooth_masked['weight']

        self.loss_ref_adversarial = None
        if ref_adversarial is not None and ref_adversarial['weight'] > 0 and self.use_ref_disc:
            self.loss_ref_adversarial = g_nonsaturating_loss
            self.loss_ref_adversarial_weight = ref_adversarial['weight']

        self.loss_adversarial = None
        if adversarial is not None and adversarial['weight'] > 0:
            self.loss_adversarial = g_nonsaturating_loss
            self.loss_adversarial_weight = adversarial['weight']


    def init_generator_regularizations(
        self,
        g_reg_every: int = 4,
        path_lenght: Dict = None
    ):
        """
        Generator regularization.

        Args:
            g_reg_every (int): the number of steps between each regularization
        """
        self.g_reg_every = g_reg_every
        self.g_reg_steps_count = 0

        self.path_reg = None
        if path_lenght is not None and path_lenght['weight'] > 0:
            self.path_reg = g_path_regularize
            self.path_reg_weight = path_lenght['weight']
            self.path_reg_decay = path_lenght['decay']
            self.mean_path_lenght = 0


    def generator_regularization(
        self,
        gen_in
    ):  # sourcery skip: extract-method, merge-nested-ifs
        """
        Compute the generator regularization.


        Args:
            gen_in (torch.Tensor): the generator input tensor.
        """

        g_regs = {}

        if self.g_reg_steps_count % self.g_reg_every == 0:
            if self.path_reg:
                
                gen_in.requires_grad = True
                gen_out = self.generator(gen_in)
                
                path_loss, self.mean_path_lenght, path_lenghts = self.path_reg(gen_out, gen_in, self.mean_path_lenght, self.path_reg_decay)
                g_regs["g_path_loss"] = path_loss * self.path_reg_weight * self.g_reg_every + (0 * gen_out[0, 0, 0, 0])
                g_regs["g_path_length"] = path_lenghts.mean()
                g_regs["g_mean_path_length"] = self.mean_path_lenght
                #check_nan(g_regs)

                # apply the regularization
                self.generator.zero_grad()
                g_regs["g_path_loss"].backward()
                self.g_opt.step()

        self.g_reg_steps_count += 1
        self.metrics.update(g_regs)


    def generator_loss(
        self, 
        fake, 
        real, 
        disc_fake_out, 
        ref_disc_out_fake = None, 
        input_masks = None
    ):
        """
        Compute the generator loss.

        Args:
            fake(Tensor): the fake image in output of the generator
            real (Tensor): the real image in input of the generator
            disc_fake_out (Tensor): the discriminator output for the fake image
            ref_disc_out_fake (Tensor): the reference discriminator output for the fake image
            input_masks (Tensor): the input masks passed to the generator

        Returns:
            Tensor: the generator loss

        """
        generator_losses = {}

        # compute the generator loss
        if self.loss_l1 is not None:
            generator_losses["g_loss_l1"] = self.loss_l1(fake, real) * self.loss_l1_weight
        if self.loss_l1_smooth_masked is not None:
            generator_losses["g_loss_l1_smasked"] = self.loss_l1_smooth_masked(fake, real, input_masks) * self.loss_l1_smooth_masked_weight

        if self.loss_mse is not None:
            generator_losses["g_loss_mse"] = self.loss_mse(fake, real) * self.loss_mse_weight 
        
        if self.loss_resnet_pl is not None:
            generator_losses["loss_resnet_pl"] = self.loss_resnet_pl(fake, real) * self.loss_resnet_pl_weight
        if self.loss_resnet_pl_smooth_masked is not None:
            generator_losses["loss_resnet_pl_smasked"] = self.loss_resnet_pl_smooth_masked(fake, real, input_masks) * self.loss_resnet_pl_smooth_masked_weight        


        if self.loss_ref_adversarial is not None and self.use_ref_disc:
            generator_losses["g_loss_ref_adv"] = self.loss_ref_adversarial(ref_disc_out_fake) * self.loss_ref_adversarial_weight
        if self.loss_adversarial is not None:
            generator_losses["g_loss_adv"] = self.loss_adversarial(disc_fake_out) * self.loss_adversarial_weight

        if self.gen_loss_reduction == 'mean':
            generator_losses["g_loss"] = torch.mean(torch.stack(list(generator_losses.values())))
        elif self.gen_loss_reduction == 'sum':
            generator_losses["g_loss"] = torch.sum(torch.stack(list(generator_losses.values())))

        #check_nan(generator_losses)

        # update the generator
        self.generator.zero_grad()
        generator_losses["g_loss"].backward()
        self.g_opt.step()

        self.metrics.update(generator_losses)