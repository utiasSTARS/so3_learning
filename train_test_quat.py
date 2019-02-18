import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from liegroups.torch import SO3
from lie_algebra_full import so3_log, so3_exp
from utils import quat_norm_diff, nll_quat, quat_ang_error



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

            q_est, Rinv = model(y_obs)

            loss_b = loss_fn(q_est, q_gt, Rinv).mean()
            loss = loss + loss_b
            angular_error += quat_ang_error(q_est, q_gt).sum()
            # print('Batch: {}. Loss:{}'.format(batch_idx, loss_b.item()))
            # if batch_idx==1:
            #     print(q_est)
            #     print(q_gt)
            #     print(loss_b)
            
            #print(nll_quat(q_est, q_gt, R))
            nll += nll_quat(q_est, q_gt, Rinv).sum()
            total_samples += batch_size


        # if model.num_hydra_heads > 1:
        #     wd = model.heads[0].fc0.weight.data - model.heads[1].fc0.weight.data
        #     print(wd.norm())


    return loss.item()/len(loader), (angular_error/total_samples)*(180./3.1415), nll/total_samples


def train(model, loader, loss_fn, optimizer, config):

    #Train!
    model.train()
    total_batches = len(loader)
    total_loss = 0.
    for batch_idx, (y_obs, q_gt) in enumerate(loader):
        #Identity matrix as the initialization
        y_obs = y_obs.to(config['device'])
        q_gt = q_gt.to(config['device'])
        q_est, Rinv = model(y_obs)

        if model.num_hydra_heads == 1:
            loss = loss_fn(q_est, q_gt, Rinv).mean()

        else:
            batch_size = q_gt.shape[0]
            q_gt = q_gt.repeat((model.num_hydra_heads, 1))
            loss = loss_fn(q_est, q_gt, Rinv).mean()

            # #Only select the heads that give the minimum loss
            # all_loss = loss_fn(q_est, q_gt, Rinv)
            # all_loss = all_loss.view(model.num_hydra_heads, batch_size)
            # #(loss, _) = torch.min(all_loss, dim=0)'
            # Or randomize heads
            # i = torch.randint(model.num_hydra_heads, (batch_size,))
            # j = torch.arange(batch_size)
            # loss = all_loss[i,j]
            # loss = loss.mean()

        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    return total_loss/total_batches