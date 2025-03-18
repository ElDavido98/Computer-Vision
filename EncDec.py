from utils import *
from metrics import l2_loss


class Encoder(nn.Module):
    def __init__(
            self,
            in_features: int = 2,
            embedding_dim: int = 64,
            batch_size: int = 8,
            hidden_dim: int = 64,
            num_layers: int = 1,
            dropout: float = 0.0,
            discriminator_flag: bool = False,
            device=torch.device("cpu")
    ):
        super(Encoder, self).__init__()

        self.discriminator_flag = discriminator_flag
        self.device = device
        self.state_tuple = (
            torch.zeros(num_layers, batch_size, hidden_dim).to(self.device),
            torch.zeros(num_layers, batch_size, hidden_dim).to(self.device)
        )

        dropout_lstm = dropout
        if num_layers == 1:
            dropout_lstm = 0.0
        # Spatial Embedding
        self.spatial_embedding = make_multilayer(
            in_channels=in_features,
            out_channels=embedding_dim,
            batch_norm_features=batch_size,
            dropout=dropout,
            disc=discriminator_flag,
            device=device
        )
        # LSTM
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout_lstm,
            device=self.device
        )
        batch_norm_features = None
        if self.discriminator_flag:
            if num_layers != 1:
                batch_norm_features = batch_size
            self.output = make_multilayer(
                in_out_dims=[hidden_dim, batch_size * hidden_dim, batch_size, 1],
                batch_norm_features=batch_norm_features,
                dropout=dropout,
            )
            del self.output[-1][-2]
            self.output.append(nn.Sigmoid())
        else:
            self.output = make_multilayer(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                batch_norm_features=batch_size,
                dropout=dropout,
                num_blocks=int(num_layers // 2)
            )

    def __call__(
            self,
            x: torch.Tensor
    ):
        y = self.forward(x)
        return y

    def forward(
            self,
            x: torch.Tensor
    ):
        if x.dim() == 1:
            x = x.unsqueeze(1)
        if x.dim() == 2:
            x = x.unsqueeze(2)

        x_embedding = self.spatial_embedding(x)

        output, (hidden, cell) = self.lstm(x_embedding, self.state_tuple)

        if self.discriminator_flag:
            out = self.output(hidden.squeeze()).squeeze()
        else:
            out = self.output(hidden)

        return out


class Decoder(nn.Module):
    def __init__(
            self,
            embedding_dim: int = 64,
            hidden_dim: int = 64,
            output_dim: int = 2,
            bottleneck_dim: int = 1024,
            num_layers: int = 2,
            batch_size: int = 8,
            dropout: float = 0.0,
            num_steps: int = 8,
            num_paths: int = 4,
            mean: float = 0.0,
            std: float = 1.0,
            device=torch.device("cpu")
    ):
        super(Decoder, self).__init__()
        self.L2 = torch.nn.MSELoss(reduction='sum')

        dropout_lstm = dropout
        if num_layers == 1:
            dropout_lstm = 0.0

        self.device = device
        self.num_layers = num_layers
        self.num_paths = num_paths
        self.num_steps = num_steps
        self.mean = mean
        self.std = std
        self.batch_size = batch_size
        self.bottleneck_dim = bottleneck_dim

        self.state_tuple = (
            torch.zeros(num_layers, batch_size, hidden_dim).to(self.device),
            torch.zeros(num_layers, batch_size, hidden_dim).to(self.device)
        )

        # Spatial Embedding
        self.spatial_embedding = make_multilayer(
            in_channels=embedding_dim,
            out_channels=embedding_dim,
            batch_norm_features=num_layers,
            dropout=dropout,
        )
        # LSTM
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_lstm,
            device=self.device
        )
        # Spatial Embedding during Prediction
        self.spatial_embedding_pred = make_multilayer(
            in_channels=hidden_dim,
            out_channels=embedding_dim,
            batch_norm_features=num_layers,
            dropout=dropout,
        )
        # Output Layer
        self.output_layers = make_multilayer(
            in_out_dims=[hidden_dim, output_dim],
            dropout=dropout,
        )
        del self.output_layers[-1][-2]
        self.output_layers.append(nn.ReLU())

        self.pool_h = PoolHiddenNet(
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            batch_size=batch_size,
            hidden_dim=hidden_dim,
            bottleneck_dim=bottleneck_dim,
            dropout=dropout,
            device=device
        )
        self.h = make_multilayer(
            in_out_dims=[(hidden_dim + bottleneck_dim), bottleneck_dim, hidden_dim],
            batch_norm_features=batch_size,
            dropout=dropout,
        )

    def __call__(
            self,
            data: torch.Tensor,
            targ: torch.Tensor,
            train: bool = True
    ):
        y = self.forward(data, targ, train)
        return y

    def forward(
            self,
            features: torch.Tensor,
            target: torch.Tensor,
            train: bool = True
    ):
        if train:
            pred_trajectories = torch.empty(self.num_paths, features.size(0), self.num_steps, 2, device=self.device)
            L2_losses_mean = torch.empty(self.num_paths, device=self.device)
            for path_idx in range(self.num_paths):
                pred_traj_per_step = torch.empty(features.size(0), self.num_steps, 2, device=self.device)
                noise = self.mean + (self.std * torch.randn(features.size(), device=self.device))
                noised_features = features + noise
                noised_embedding = self.spatial_embedding(noised_features)
                state_tuple = self.state_tuple
                for step_idx in range(self.num_steps):
                    output, (hidden, cell) = self.lstm(noised_embedding, state_tuple)
                    pred_pos = self.output_layers(output[:, -1, :].squeeze(1))
                    noise = self.mean + (self.std * torch.randn(output.size(), device=self.device))
                    noised_output = output + noise
                    noised_embedding = self.spatial_embedding_pred(noised_output)
                    pool_h = self.pool_h(hidden, pred_pos)
                    pool_h = pool_h.view([self.num_layers, self.batch_size, self.bottleneck_dim])
                    new_hidden = self.h(torch.cat([hidden, pool_h], dim=-1))
                    state_tuple = (new_hidden, cell)
                    pred_traj_per_step[:, step_idx, :] = pred_pos
                    L2_losses_mean[path_idx] = l2_loss(self.L2, pred_pos, target[:, step_idx, :]).mean()
                pred_trajectories[path_idx] = pred_traj_per_step
            # Choose the trajectory with the minimum L2 loss
            min_loss_index = torch.argmin(L2_losses_mean)
            return pred_trajectories[min_loss_index]
        else:
            pred_traj_per_step = torch.empty(features.size(0), self.num_steps, 2, device=self.device)
            noise = self.mean + (self.std * torch.randn(features.size(), device=self.device))
            noised_features = features + noise
            noised_embedding = self.spatial_embedding(noised_features)
            state_tuple = self.state_tuple
            for step_idx in range(self.num_steps):
                output, (hidden, cell) = self.lstm(noised_embedding, state_tuple)
                pred_pos = self.output_layers(output[:, -1, :].squeeze(1))
                noise = self.mean + (self.std * torch.randn(output.size(), device=self.device))
                noised_output = output + noise
                noised_embedding = self.spatial_embedding_pred(noised_output)
                pool_h = self.pool_h(hidden, pred_pos)
                pool_h = pool_h.view([self.num_layers, self.batch_size, self.bottleneck_dim])
                new_hidden = self.h(torch.cat([hidden, pool_h], dim=-1))
                state_tuple = (new_hidden, cell)
                pred_traj_per_step[:, step_idx, :] = pred_pos
            return pred_traj_per_step


class PoolHiddenNet(nn.Module):
    def __init__(
            self,
            embedding_dim: int = 64,
            num_layers: int = 1,
            batch_size: int = 8,
            hidden_dim: int = 64,
            bottleneck_dim: int = 1024,
            dropout: float = 0.0,
            device=torch.device("cpu")
    ):
        super(PoolHiddenNet, self).__init__()

        self.device = device
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim
        self.embedding_dim = embedding_dim
        self.num_layers = None
        self.batch_size = None

        seq_start_end = torch.zeros(num_layers, 2)
        start_idx, end_idx = 0, 0
        for i in range(num_layers):
            start_idx = end_idx
            end_idx = start_idx + (batch_size * hidden_dim)
            seq_start_end[i, 0] = start_idx
            seq_start_end[i, 1] = end_idx

        mlp_pre_dim = embedding_dim + hidden_dim
        mlp_pre_pool_dims = [mlp_pre_dim, 512, bottleneck_dim]

        self.spatial_embedding = make_multilayer(
            in_out_dims=[2, embedding_dim],
            dropout=dropout,
            device=device
        )
        self.mlp_pre_pool = make_multilayer(
            in_out_dims=mlp_pre_pool_dims,
            dropout=dropout,
            device=device
        )

    def __call__(
            self,
            x: torch.Tensor,
            end_pos: torch.Tensor
    ):
        if x.dim() == 2:
            self.num_layers = 1
            self.batch_size = x.size(0)
        elif x.dim() == 3:
            self.num_layers = x.size(0)
            self.batch_size = x.size(1)

        seq_start_end = ((i * self.batch_size, (i + 1) * self.batch_size) for i in range(self.num_layers))
        y = self.forward(x, seq_start_end, end_pos)
        return y

    @staticmethod
    def repeat(
            tensor: torch.Tensor,
            num_reps: int
    ):
        col_len = tensor.size(1)
        tensor = tensor.unsqueeze(dim=1).repeat(1, num_reps, 1)
        tensor = tensor.view(-1, col_len)
        return tensor

    def forward(
            self,
            h_states: torch.Tensor,
            seq_start_end,
            end_pos: torch.Tensor
    ):
        pool_h = []
        h_states_new_shape = h_states.view(-1, self.hidden_dim)
        for _, (start, end) in enumerate(seq_start_end):
            num_agents = end - start
            curr_hidden = h_states_new_shape[start:end]
            curr_end_pos = end_pos
            curr_hidden_1 = curr_hidden.repeat(num_agents, 1)
            curr_end_pos_1 = curr_end_pos.repeat(num_agents, 1)
            curr_end_pos_2 = self.repeat(curr_end_pos, num_agents)
            curr_rel_pos = curr_end_pos_1 - curr_end_pos_2
            curr_rel_embedding = self.spatial_embedding(curr_rel_pos.to(self.device))
            mlp_h_input = torch.cat([curr_rel_embedding, curr_hidden_1], dim=1)
            curr_pool_h = self.mlp_pre_pool(mlp_h_input)
            curr_pool_h = curr_pool_h.view(num_agents, num_agents, -1).max(1)[0]
            pool_h.append(curr_pool_h)
        pool_h = torch.cat(pool_h, dim=0)
        return pool_h[None, :, :]
