import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from lie_algebra_full import so3_log, so3_exp
from utils import compute_normalization, compute_error_angles, nll_mat

def validate(model, loader, loss_fn, config):
    model.eval()
    with torch.no_grad():
        loss = torch.tensor([0.])
        angular_error = 0.
        nll = 0.
        total_samples = 0
        total_batches = len(loader)

        for batch_idx, (y_obs, C_gt) in enumerate(loader):
            #Identity matrix as the initialization
            y_obs = y_obs.to(config['device'])
            C_gt = C_gt.to(config['device'])

            batch_size = C_gt.shape[0]
            total_samples += batch_size

            C_est, Rinv = model(y_obs)

            angular_error += compute_error_angles(C_est, C_gt).sum()
            nll += nll_mat(C_est, C_gt, Rinv).sum()
            loss += loss_fn(C_est, C_gt, Rinv).mean()

    return loss.item()/total_batches, angular_error/total_samples*(180./3.1415), nll/total_samples


def train(model, loader, loss_fn, optimizer, config):

    #Train!
    model.train()
    total_batches = len(loader)
    total_loss = 0.
    angular_error = 0.

    for batch_idx, (y_obs, C_gt) in enumerate(loader):
        
        y_obs = y_obs.to(config['device'])
        C_gt = C_gt.to(config['device'])

        C_est, Rinv = model(y_obs)

        if model.num_hydra_heads > 1:
            batch_size = C_gt.shape[0]
            C_gt = C_gt.repeat((model.num_hydra_heads, 1, 1))

        loss = loss_fn(C_est, C_gt, Rinv).mean()


        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    return total_loss/total_batches
    
    
