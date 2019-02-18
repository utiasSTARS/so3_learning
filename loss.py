import torch 
from lie_algebra import so3_log
from utils import normalize_vecs, quat_log_diff, batch_logdet3


class SO3NLLLoss(torch.nn.Module):
    def __init__(self):
        super(SO3NLLLoss, self).__init__()

    def forward(self, C_est, C_target, Rinv):
        if C_est.dim() < 3:
            C_est = C_est.unsqueeze(0)
            C_target = C_target.unsqueeze(0)

        residual = so3_log(C_est.bmm(C_target.transpose(1,2))).unsqueeze(2)

        weighted_term = 0.5 * residual.transpose(1, 2).bmm(Rinv).bmm(residual)
        nll = weighted_term.squeeze() - 0.5 * batch_logdet3(Rinv)

        return nll

class SO3FrobNorm(torch.nn.Module):
    def __init__(self, average=True):
        super(SO3FrobNorm, self).__init__()
        self.average = average
    
    def forward(self, C_est, C_gt):
        if C_est.dim() < 3:
            C_est = C_est.unsqueeze(0)
            C_gt = C_gt.unsqueeze(0)

        loss = ((C_est - C_gt).norm(dim=(1,2))**2)

        if self.average:    
            return loss.mean()
        else:
            return loss


class QuatLoss(torch.nn.Module):
    def __init__(self, reduce=True):
        super(QuatLoss, self).__init__()
        self.reduce = reduce
    
    def forward(self, q_est, q_gt, Rinv):
        if q_est.dim() < 2:
            q_est = q_est.unsqueeze(0)
            q_gt = q_gt.unsqueeze(0)

        loss = torch.min((q_est - q_gt).pow(2).sum(dim=1), (q_est + q_gt).pow(2).sum(dim=1))
    
        if self.reduce:
            return loss.mean()
        else:
            return loss

class QuatNLLLoss(torch.nn.Module):
    def __init__(self, reduce=False):
        super(QuatNLLLoss, self).__init__()
        self.reduce = reduce
    
    def forward(self, q_est, q_gt, Rinv):
        if q_est.dim() < 2:
            q_est = q_est.unsqueeze(0)
            q_gt = q_gt.unsqueeze(0)

        residual = quat_log_diff(q_est, q_gt).unsqueeze(2)
        weighted_term = 0.5*residual.transpose(1,2).bmm(Rinv).bmm(residual)
        nll = weighted_term.squeeze() - 0.5*batch_logdet3(Rinv)

        #nll = torch.min((q_est - q_gt).pow(2).sum(dim=1), (q_est + q_gt).pow(2).sum(dim=1))
        if self.reduce:
            return nll.mean()
        else:
            return nll