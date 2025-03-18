from utils import *


def l2_loss(
        loss_funct: torch.nn,
        pred: torch.Tensor,
        targ: torch.Tensor
):
    loss = loss_funct(pred, targ)
    return loss


def average_displacement_error(
        pred: torch.Tensor,
        targ: torch.Tensor,
):
    displacement = torch.sqrt(torch.sum((pred - targ) ** 2, dim=-1))
    ade_per_agent = displacement.mean(dim=(1, 2))
    ADE = ade_per_agent.mean().item()
    return ADE


def final_displacement_error(
        pred: torch.Tensor,
        targ: torch.Tensor,
):
    displacement = torch.sqrt(torch.sum((pred[:, :, -1, :] - targ[:, :, -1, :]) ** 2, dim=-1))
    fde_per_agent = displacement.mean(dim=1)
    FDE = fde_per_agent.mean().item()
    return FDE


def average(values):
    AVG = sum(values) / len(values)
    return AVG
