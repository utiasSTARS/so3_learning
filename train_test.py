import torch, time
import numpy as np
from torch.utils.data import Dataset, DataLoader
from liegroups.torch import SO3
from lie_algebra import so3_log, so3_exp
from utils import quat_norm_diff, nll_quat, quat_ang_error, perturb_quat_for_hydranet
from vis import plot_errors_with_sigmas
import torchvision


def validate(model, loader, loss_fn, config, output_history=False, output_grid=False):
    model.eval()

    with torch.no_grad():
        loss = torch.tensor([0.])
        angular_error = 0.
        nll = 0.        
        total_samples = 0.

        if output_history:
            q_gt_hist = []
            q_est_hist = []
            R_est_hist = []
            R_direct_hist = []

        for batch_idx, (y_obs, q_gt) in enumerate(loader):

            start = time.time()

            if isinstance(y_obs, list):
                y_obs[0] = y_obs[0].to(config['device'])
                y_obs[1] = y_obs[1].to(config['device'])
            else:
                y_obs = y_obs.to(config['device'])

            # if batch_idx == int(len(loader)/2) + 1 and output_grid:
            #     print('SAVING IMAGE GRID')
            #     torchvision.utils.save_image(torchvision.utils.make_grid(y_obs), '7scenes/jittered_image.png')

            q_gt = q_gt.to(config['device'])
            batch_size = q_gt.shape[0]

            q_est, Rinv, Rinv_direct = model(y_obs)

            loss_b = loss_fn(q_est, q_gt, Rinv).mean()
            loss = loss + loss_b
            angular_error += quat_ang_error(q_est, q_gt).sum()
            nll += nll_quat(q_est, q_gt, Rinv).sum()

            if output_history:
                q_gt_hist.append(q_gt)
                q_est_hist.append(q_est)
                R_est_hist.append(Rinv.inverse())
                R_direct_hist.append(Rinv_direct.inverse())

            total_samples += batch_size

            print('Single mini batch in: {:3.3f}'.format(time.time() - start))

    avg_loss = loss.item() / len(loader)
    avg_err = (angular_error/total_samples)*(180./3.1415)
    avg_nll = nll/total_samples

    if output_history:
        q_gt_hist = torch.cat(q_gt_hist, dim=0).cpu()
        q_est_hist = torch.cat(q_est_hist, dim=0).cpu()
        R_est_hist = torch.cat(R_est_hist, dim=0).cpu()
        R_direct_hist = torch.cat(R_direct_hist, dim=0).cpu()

        return (avg_loss, avg_err, avg_nll, (q_gt_hist, q_est_hist, R_est_hist, R_direct_hist))

    else:
        return (avg_loss, avg_err, avg_nll)


def train(model, loader, loss_fn, optimizer, config, q_target_sigma=0.):

    #Train!
    model.train()
    total_batches = len(loader)
    total_loss = 0.
    #Necessary to have the same noise at every epoch
    torch.manual_seed(42)

    for batch_idx, (y_obs, q_gt) in enumerate(loader):
        #Identity matrix as the initialization
        if isinstance(y_obs, list):
            y_obs[0] = y_obs[0].to(config['device'])
            y_obs[1] = y_obs[1].to(config['device'])
        else:
            y_obs = y_obs.to(config['device'])

        q_gt = q_gt.to(config['device'])
        q_est, Rinv = model(y_obs)

        if model.num_hydra_heads == 1:
            loss = loss_fn(q_est, q_gt, Rinv).mean()

        else:
            batch_size = q_gt.shape[0]
            q_gt = perturb_quat_for_hydranet(q_gt, model.num_hydra_heads, q_target_sigma)
            loss = loss_fn(q_est, q_gt, Rinv).mean()

            # #Only select the heads that give the minimum loss
            # all_loss = loss_fn(q_est, q_gt, Rinv)
            # all_loss = all_loss.view(model.num_hydra_heads, batch_size)
            # losses = all_loss.mean(dim=1)
            #
            # for loss in losses:
            #     loss.backward(retain_graph=True)

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