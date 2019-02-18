import torch
import numpy as np
from liegroups.torch import SO3
from lie_algebra_full import so3_log, so3_exp
from utils import quat_norms, nll_quat



def validate(model, loader, loss_fn, config):
    model.eval()
    with torch.no_grad():
        loss = torch.tensor([0.])
        angular_error = 0.
        nll = 0.        
        total_samples = 0.

        for batch_idx, (y_obs, q_gt) in enumerate(loader):
            #Identity matrix as the initialization
            y_obs = y_obs.to(config['device'])
            q_gt = q_gt.to(config['device'])
            batch_size = q_gt.shape[0]

            q_est, R = model(y_obs)

            loss_b = loss_fn(q_est, q_gt)
            loss = loss + loss_b
            angular_error += 4*torch.asin(0.5*quat_norms(q_est, q_gt)).sum()
            nll += nll_quat(q_est, q_gt, R).sum()
            total_samples += batch_size

    return loss.item()/len(loader), angular_error/total_samples*180./3.1415, nll/total_samples


def train(model, loader, loss_fn, optimizer, config):

    #Train!
    model.train()
    total_batches = len(loader)
    total_loss = 0.
    angular_error = 0.

    for batch_idx, (y_obs, q_gt) in enumerate(loader):
        #Identity matrix as the initialization
        y_obs = y_obs.to(config['device'])
        q_gt = q_gt.to(config['device'])

        if model.num_hydra_heads > 1:
            q_gt = q_gt.repeat((model.num_hydra_heads, 1))

        q_est = model(y_obs)
        loss = loss_fn(q_est, q_gt)

        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    return total_loss/total_batches