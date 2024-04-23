###########################
# Latent ODEs for Irregularly-Sampled Time Series
# Author: Yulia Rubanova
###########################

import numpy as np
import sklearn as sk
import numpy as np
# import gc
import torch
import torch.nn as nn
from torch.nn.functional import relu

import lib.utils as utils
from lib.utils import get_device
from lib.encoder_decoder import *
from lib.likelihood_eval import *

from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal
from torch.distributions import kl_divergence, Independent
from lib.base_models import VAE_Baseline


class LatentODE(VAE_Baseline):
    def __init__(self, input_dim, latent_dim, encoder_z0, decoder, diffeq_solver,
                 z0_prior, device, obsrv_std=None,
                 use_binary_classif=False, use_poisson_proc=False,
                 linear_classifier=False,
                 classif_per_tp=False,
                 n_labels=1,
                 train_classif_w_reconstr=False,
                 reconstruct_from_latent=False):

        super(LatentODE, self).__init__(
            input_dim=input_dim, latent_dim=latent_dim,
            z0_prior=z0_prior,
            device=device, obsrv_std=obsrv_std,
            use_binary_classif=use_binary_classif,
            classif_per_tp=classif_per_tp,
            linear_classifier=linear_classifier,
            use_poisson_proc=use_poisson_proc,
            n_labels=n_labels,
            train_classif_w_reconstr=train_classif_w_reconstr,
            reconstruct_from_latent=reconstruct_from_latent)

        self.encoder_z0 = encoder_z0
        self.diffeq_solver = diffeq_solver
        self.decoder = decoder
        self.use_poisson_proc = use_poisson_proc

    def get_reconstruction(self, time_steps_to_predict, truth, truth_time_steps, truth_to_predict = None,
                           mask=None, n_traj_samples=1, run_backwards=True, mode=None, re_encode=0, \
                           running_mode: str = "training"):

        if isinstance(self.encoder_z0, Encoder_z0_ODE_RNN) or \
                isinstance(self.encoder_z0, Encoder_z0_RNN):

            truth_w_mask = truth
            if mask is not None:
                truth_w_mask = torch.cat((truth, mask), -1)
            first_point_mu, first_point_logvar, z_mu, z_logvar, latent_ys = self.encoder_z0(
                truth_w_mask, truth_time_steps, run_backwards=run_backwards)
            # Make clone of latest memory state for re-encode step
            last_yi, last_yi_logvar = self.encoder_z0.last_yi.clone(), self.encoder_z0.last_yi_logvar.clone()
            # If extrapolation mode, we forward the "to-predict" time points to the encoder
            if mode == "extrap":
                # Pick actual last sample
                first_point_mu = z_mu[:, -1, :]
                first_point_logvar = z_logvar[:, -1, :]
                if running_mode == "training" and self.reconstruct_from_latent:
                    truth_w_mask2predict = torch.cat((truth_to_predict, torch.ones_like(truth_to_predict)), -1)
                    # z_mu, z_std and latent_ys is replaced by the extrap states
                    _, _, z_mu, z_logvar, latent_ys = self.encoder_z0(
                        truth_w_mask2predict, time_steps_to_predict, run_backwards=False, use_last_state=True)

            means_z0 = first_point_mu.repeat(n_traj_samples, 1, 1)
            logvar_z0 = first_point_logvar.repeat(n_traj_samples, 1, 1)
            first_point_enc = utils.sample_standard_gaussian(means_z0, logvar_z0)

            latents_zs = utils.sample_standard_gaussian(z_mu.repeat(n_traj_samples, 1, 1, 1),
                                                        z_logvar.repeat(n_traj_samples, 1, 1, 1))

        else:
            raise Exception("Unknown encoder type {}".format(type(self.encoder_z0).__name__))

        first_point_stdv = (first_point_logvar * .5).exp()
        assert (torch.sum(first_point_stdv < 0) == 0.)

        if self.use_poisson_proc:
            n_traj_samples, n_traj, n_dims = first_point_enc.size()
            # append a vector of zeros to compute the integral of lambda
            zeros = torch.zeros([n_traj_samples, n_traj, self.input_dim]).to(get_device(truth))
            first_point_enc_aug = torch.cat((first_point_enc, zeros), -1)
            means_z0_aug = torch.cat((means_z0, zeros), -1)
        else:
            first_point_enc_aug = first_point_enc
            means_z0_aug = means_z0

        assert (not torch.isnan(time_steps_to_predict).any())
        assert (not torch.isnan(first_point_enc).any())
        assert (not torch.isnan(first_point_enc_aug).any())

        # Shape of sol_y [n_traj_samples, n_samples, n_timepoints, n_latents]
        # Run this iff re_encode > 0
        if re_encode > 0:
            # Revert memory state right after feeding last "interp" sample
            self.encoder_z0.last_yi, self.encoder_z0.last_yi_logvar = last_yi, last_yi_logvar
            sol_y, pred_x = self.get_reconstruction_with_reencode(first_point_enc_aug, time_steps_to_predict, re_encode)
        else:
            # Otherwise run standard pipeline
            sol_y = self.diffeq_solver(first_point_enc_aug, time_steps_to_predict)

            if self.use_poisson_proc:
                sol_y, log_lambda_y, int_lambda, _ = self.diffeq_solver.ode_func.extract_poisson_rate(sol_y)

                assert (torch.sum(int_lambda[:, :, 0, :]) == 0.)
                assert (torch.sum(int_lambda[0, 0, -1, :] <= 0) == 0.)

            pred_x = self.decoder(sol_y)

        # Reconstructions from encoder's latent space
        reconstructed_x = self.decoder(latents_zs.permute(0,2,1,3))

        all_extra_info = {
            "first_point": (first_point_mu, first_point_stdv, first_point_enc),
            "z_mu": z_mu,
            "z_logvar": z_logvar,
            "latent_traj": sol_y,
            "latent_traj_encoder": latents_zs,
            "reconstructed_traj_encoder": reconstructed_x,
            "memory_state": latent_ys
        }

        if self.use_poisson_proc:
            # intergral of lambda from the last step of ODE Solver
            all_extra_info["int_lambda"] = int_lambda[:, :, -1, :]
            all_extra_info["log_lambda_y"] = log_lambda_y

        if self.use_binary_classif:
            if self.classif_per_tp:
                all_extra_info["label_predictions"] = self.classifier(sol_y)
            else:
                all_extra_info["label_predictions"] = self.classifier(first_point_enc).squeeze(-1)

        return pred_x, all_extra_info

    def sample_traj_from_prior(self, time_steps_to_predict, n_traj_samples=1, re_encode=0):
        # input_dim = starting_point.size()[-1]
        # starting_point = starting_point.view(1,1,input_dim)

        # Sample z0 from prior
        starting_point_enc = self.z0_prior.sample([n_traj_samples, 1, self.latent_dim]).squeeze(-1)

        starting_point_enc_aug = starting_point_enc
        if self.use_poisson_proc:
            n_traj_samples, n_traj, n_dims = starting_point_enc.size()
            # append a vector of zeros to compute the integral of lambda
            zeros = torch.zeros(n_traj_samples, n_traj, self.input_dim).to(self.device)
            starting_point_enc_aug = torch.cat((starting_point_enc, zeros), -1)

        if re_encode > 0:
            sol_y, pred_x = self.get_reconstruction_with_reencode(starting_point_enc_aug, time_steps_to_predict,
                                                                  re_encode, sample_prior=True)
        else:
            sol_y = self.diffeq_solver.sample_traj_from_prior(starting_point_enc_aug, time_steps_to_predict,
                                                              n_traj_samples=3)
            if self.use_poisson_proc:
                sol_y, log_lambda_y, int_lambda, _ = self.diffeq_solver.ode_func.extract_poisson_rate(sol_y)

            pred_x = self.decoder(sol_y)

        return pred_x

    def get_reconstruction_with_reencode(self, first_point, time_steps_to_predict, re_encode, sample_prior = False):
        # Save latent state in memory in case we need to re-encode from scratch again
        if sample_prior:
            n_trajs, n_samples, n_dims = first_point.size()
            self.encoder_z0.last_yi = torch.zeros(n_trajs, n_samples, n_dims * 2).to(self.device) # * 1e-2
            self.encoder_z0.last_yi_logvar = torch.zeros(n_trajs, n_samples, n_dims * 2).to(self.device)
        n_timestamps = time_steps_to_predict.size(0)
        n_chunks = int(np.ceil(n_timestamps / re_encode))
        # Skip first timestamp as this is added in for loop
        time_blocks = time_steps_to_predict[1:].chunk(n_chunks, dim=0)
        prev_z = first_point
        prev_timestamp = time_steps_to_predict[0]
        sol_y = []
        pred_x = []
        for block in time_blocks:
            t_block = torch.cat((prev_timestamp.view(1), block), dim=0)
            # Solve latent ODE for each block
            z = self.diffeq_solver(prev_z, t_block)
            x_hat = self.decoder(z)
            # Augment x with mask of ones and forward through encoder to do re-encoding
            mask = torch.ones_like(x_hat)
            len_x = x_hat.shape[-1]
            x_hat = torch.cat((x_hat, mask), dim=-1)
            _, _, z_mu, z_logvar, _ = self.encoder_z0(x_hat, t_block, run_backwards=False, use_last_state=True)
            mean_z = z_mu[:, -1, :]
            z_hat = mean_z
            # Concatenate the solutions (do not add last element as it will be in next block)
            sol_y.append(z[:, :, :-1, :])
            pred_x.append(x_hat[:, :, :-1, 0:len_x])
            # Update the previous z
            prev_z = z_hat
            prev_timestamp = block[-1]
        # Transform list to tensor and add last element to the solution
        sol_y = torch.cat(sol_y, dim=2)
        sol_y = torch.cat((sol_y, prev_z.unsqueeze(dim=2)), dim=2)
        pred_x = torch.cat(pred_x, dim=2)
        pred_x = torch.cat((pred_x, x_hat[:, :, -1:, 0:len_x]), dim=2)
        return sol_y, pred_x
