import torch
import statistics
import pickle
import os
import json
import gdown
import random
from torch import nn

state_to_code = {
    ('G', 'G', 'G', 'R', 'R', 'G', 'R', 'R', 'G', 'R'): 0,
    ('G', 'R', 'R', 'R', 'R', 'G', 'R', 'R', 'R', 'R'): 1,
    ('R', 'R', 'G', 'R', 'R', 'G', 'R', 'R', 'G', 'R'): 2,
    ('R', 'R', 'R', 'G', 'G', 'R', 'G', 'G', 'R', 'G'): 3,
    ('R', 'R', 'R', 'R', 'G', 'R', 'R', 'G', 'R', 'R'): 4,
    ('R', 'R', 'R', 'R', 'R', 'R', 'G', 'G', 'R', 'R'): 5,
    ('R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R'): 6,
    ('R', 'R', 'R', 'Y', 'Y', 'R', 'R', 'R', 'R', 'Y'): 7,
    ('R', 'R', 'R', 'Y', 'Y', 'R', 'Y', 'Y', 'R', 'Y'): 8,
    ('Y', 'Y', 'Y', 'R', 'R', 'Y', 'R', 'R', 'Y', 'R'): 9,
    ('G', 'G', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R'): 10,
    ('R', 'R', 'R', 'G', 'R', 'R', 'G', 'R', 'R', 'R'): 11,
    ('R', 'R', 'R', 'R', 'R', 'R', 'Y', 'Y', 'R', 'R'): 12,
    ('Y', 'R', 'R', 'R', 'R', 'Y', 'R', 'R', 'R', 'R'): 13,
    ('G', 'G', 'G', 'G', 'G', 'R', 'R', 'R', 'R', 'R'): 14,
    ('Y', 'Y', 'Y', 'Y', 'Y', 'R', 'R', 'R', 'R', 'R'): 15,
    ('Y', 'Y', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R'): 16,
    ('R', 'R', 'R', 'R', 'R', 'G', 'G', 'R', 'R', 'R'): 17,
    ('R', 'R', 'R', 'R', 'R', 'G', 'G', 'G', 'G', 'G'): 18,
    ('R', 'R', 'R', 'R', 'R', 'Y', 'Y', 'Y', 'Y', 'Y'): 19,
    ('R', 'R', 'R', 'R', 'R', 'Y', 'Y', 'R', 'R', 'R'): 20,
    ('R', 'G', 'R', 'G', 'G', 'R', 'R', 'R', 'R', 'R'): 21,
    ('R', 'R', 'R', 'R', 'R', 'R', 'G', 'R', 'G', 'R'): 22,
    ('G', 'R', 'G', 'R', 'R', 'R', 'R', 'R', 'R', 'R'): 23,
    ('R', 'R', 'R', 'R', 'R', 'G', 'R', 'G', 'R', 'G'): 24,
    ('G', 'R', 'G', 'R', 'G', 'R', 'R', 'R', 'R', 'R'): 25,
}


def make_multilayer(
        block=None,
        in_out_dims=None,
        in_channels: int = 64,
        out_channels: int = 64,
        activation_function: nn.Module = nn.LeakyReLU(negative_slope=0.35),
        dropout: float = 0.0,
        batch_norm_features=None,
        num_blocks: int = 1,
        change: int = 0,
        disc: bool = False,
        device=torch.device("cpu")
):
    layers = []
    if in_out_dims is None:
        for _ in range(num_blocks):
            if block is None:
                layers.append(nn.Sequential(
                    nn.Linear(in_features=in_channels, out_features=out_channels, device=device),
                    nn.BatchNorm1d(num_features=batch_norm_features) if disc else nn.Identity(),
                    activation_function,
                    nn.Dropout(p=dropout) if dropout > 0 else nn.Identity(),
                )
                )
            else:
                layers.append(block(in_channels, out_channels, activation_function, device=device).to(device))
                if change:
                    in_channels = out_channels
    else:
        for (in_dim, out_dim) in zip(in_out_dims[:-1], in_out_dims[1:]):
            if block is None:
                layers.append(nn.Sequential(
                    nn.Linear(in_features=in_dim, out_features=out_dim, device=device),
                    nn.BatchNorm1d(num_features=out_dim) if batch_norm_features is None else nn.BatchNorm1d(num_features=batch_norm_features) if disc else nn.Identity(),
                    activation_function,
                    nn.Dropout(p=dropout) if dropout > 0 else nn.Identity(),
                )
                )
            else:
                layers.append(block(in_channels, out_channels, activation_function, device=device).to(device))
    return nn.Sequential(*layers).to(device)


def map_tl_to_code(seq):
    state_tuple = tuple(seq)
    digit_code = state_to_code.get(state_tuple)
    return digit_code


def printProgressAction(
        action: str,
        iteration: int
):
    print(f'\r{action} {iteration}', end=" ")


def agents_selection(
        data_batch,
        num_agents: int = 8,
        device=torch.device("cpu")
):
    past_ids = data_batch['past_and_curr'][0][0]
    future_ids = data_batch['future'][0][0]
    past_pos = data_batch['past_and_curr'][3][0]
    past_state = data_batch['past_and_curr'][4][0]
    future_pos = data_batch['future'][1][0]

    curr_ids = past_ids[-1, :, :]
    mask = (past_ids != 0) & (future_ids != 0)

    valid_indices = mask.nonzero(as_tuple=True)
    batch_indices = valid_indices[0]
    time_indices = valid_indices[1]

    filtered_past_ids = torch.zeros((past_ids.size(0), num_agents, 1), dtype=past_ids.dtype, device=device)
    filtered_past_pos = torch.zeros((past_pos.size(0), num_agents, 2), dtype=past_pos.dtype, device=device)
    filtered_past_state = torch.zeros((past_state.size(0), num_agents, 4), dtype=past_state.dtype, device=device)
    filtered_future_pos = torch.zeros((future_pos.size(0), num_agents, 2), dtype=future_pos.dtype, device=device)

    k = 0
    for j, val_curr_ids in enumerate(curr_ids):
        flag = 0
        for i in range(past_ids.size(0)):
            valid_idx = torch.unique(time_indices[batch_indices == i])
            m = torch.BoolTensor(past_ids[i, valid_idx, :] == val_curr_ids)
            if len(valid_idx) >= num_agents:
                if torch.any(m) and k < num_agents:
                    indices = torch.nonzero(m).squeeze(1)
                    filtered_past_ids[i, k, 0] = past_ids[i, valid_idx, :][indices[0, 0], indices[0, 1]]
                    filtered_past_pos[i] = past_pos[i, valid_idx, :][indices[0, 0], indices[0, 1]]
                    filtered_past_state[i] = past_state[i, valid_idx, :][indices[0, 0], indices[0, 1]]
                    filtered_future_pos[i] = future_pos[i, valid_idx, :][indices[0, 0], indices[0, 1]]
                    flag += 1
            else:
                return [], []
        if k < num_agents and flag == num_agents:
            k += 1

    past_scenes = data_batch['past_and_curr'][1][0]
    past_tl = data_batch['past_and_curr'][2][0]

    past = [
        filtered_past_ids,
        past_scenes,
        past_tl,
        filtered_past_pos,
        filtered_past_state
    ]
    target = filtered_future_pos

    return past, target


def EarlyStopping(
        curr_monitor: float,
        old_monitor: float,
        count: int,
        patience: int = 5,
        min_delta: float = 0.0
):
    stop = False
    if (curr_monitor - old_monitor) <= min_delta:
        count += 1
        if count > patience:
            stop = True
            return count, stop
    count = 0
    return count, stop


def max_values(
        t_a: torch.Tensor,
        t_b: torch.Tensor
):
    if t_a is None:
        return t_b
    new_t = t_a.clone()
    mask = (new_t < t_b)
    new_t[mask] = t_b[mask]
    return new_t


def min_values(
        t_a: torch.Tensor,
        t_b: torch.Tensor
):
    if t_a is None:
        return t_b
    new_t = t_a.clone()
    mask = (new_t > t_b)
    new_t[mask] = t_b[mask]
    return new_t


def min_max_normalize(
        data: torch.Tensor,
        min_val: torch.Tensor,
        max_vals: torch.Tensor,
        eps: float = 1e-10
):
    range_vals = max_vals - min_val + eps
    normalized_tensor = (data - min_val) / range_vals
    return normalized_tensor


def min_max_normalize_images(
        images: torch.Tensor,
        eps: float = 1e-10,
):
    min_vals = 0
    max_vals = 255
    range_vals = max_vals - min_vals + eps
    normalized_images = (images - min_vals) / range_vals
    return normalized_images


def init_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 1)
        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    nn.init.kaiming_normal_(param.data)
                elif 'weight_hh' in name:
                    nn.init.kaiming_normal_(param.data)
                elif 'bias' in name:
                    nn.init.constant_(param.data, 1)


class PeriodicPadding2D(nn.Module):
    def __init__(
            self,
            pad_width: int,
            device=torch.device("cpu")
    ):
        super().__init__()
        self.pad_width = pad_width
        self.dev = device

    def forward(
            self,
            inputs: torch.Tensor
    ):
        if self.pad_width == 0:
            return inputs.to(self.dev)
        inputs = inputs.to(self.dev)
        inputs_padded = torch.cat((inputs[:, :, :, -self.pad_width:],
                                   inputs,
                                   inputs[:, :, :, :self.pad_width],), dim=-1, ).to(self.dev)
        inputs_padded = nn.functional.pad(inputs_padded, (0, 0, self.pad_width, self.pad_width), ).to(self.dev)
        return inputs_padded


class ResidualBlock(nn.Module):
    def __init__(
            self,
            in_channels: int = 64,
            out_channels: int = 64,
            activation_function: nn.Module = nn.ReLU,
            device=torch.device("cpu")
    ):
        super(ResidualBlock, self).__init__()
        self.in_channels = in_channels
        self.periodic_zeros_padding = PeriodicPadding2D(1, device)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=0).to(device)
        self.activation_function = activation_function.to(device)
        self.bn1 = nn.BatchNorm2d(out_channels).to(device)
        self.dropout = nn.Dropout(0.1).to(device)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1).to(device)
        self.bn2 = nn.BatchNorm2d(out_channels).to(device)
        self.shortcut = nn.Identity().to(device)
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1)).to(device),
                nn.BatchNorm2d(out_channels).to(device)
            ).to(device)

    def forward(
            self,
            x: torch.Tensor
    ):
        residual = self.shortcut(x)
        x = self.periodic_zeros_padding(x)
        x = self.conv1(x)
        x = self.activation_function(x)
        x = self.bn1(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.activation_function(x)
        x = self.bn2(x)
        x = self.dropout(x)
        x += residual
        return x
