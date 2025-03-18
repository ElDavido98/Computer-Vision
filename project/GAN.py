import torch

from utils import *
from data_processing import Processor
from EncDec import *


class GAN(nn.Module):
    def __init__(
            self,
            embedding_dim: int = 64,
            enc_hidden_dim: int = 64,
            dec_hidden_dim: int = 128,
            num_layers_G: int = 1,
            num_layers_D: int = 1,
            dropout_G: float = 0.0,
            dropout_D: float = 0.35,
            output_dim: int = 2,
            num_steps: int = 8,
            num_paths: int = 4,
            bottleneck_dim: int = 1024,
            concat_dim: int = 1280,
            in_channels: int = 64,
            in_height: int = 90,
            in_width: int = 160,
            out_channels: int = 64,
            hidden_channels: int = 64,
            num_blocks: int = 18,
            seq_length: int = 8,
            batch_size: int = 8,
            flag: list = None,
            G_learning_rate: float = 5e-4,
            D_learning_rate: float = 5e-4,
            G_weight_decay: float = 1e-2,
            D_weight_decay: float = 1e-2,
            device=torch.device("cpu")
    ):
        super(GAN, self).__init__()

        if flag is None:
            flag = [0, 0, 0]

        self.seq_length = seq_length
        self.batch_size = batch_size
        self.min_max_pos, self.min_max_state = None, None
        self.count_G = 0
        self.done_G = False
        self.count_D = 0
        self.done_D = False
        self.done = False
        self.validation_losses_G, self.validation_losses_D = [], []
        self.previous_validation_losses_G, self.previous_validation_losses_D = self.validation_losses_G, self.validation_losses_D
        self.eval_pred, self.eval_targ = None, None
        self.device = device

        self.loss = nn.BCELoss()

        self.Generator = Generator(
            in_channels=in_channels,
            in_height=in_height,
            in_width=in_width,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            num_blocks=num_blocks,
            embedding_dim=embedding_dim,
            enc_hidden_dim=enc_hidden_dim,
            dec_hidden_dim=dec_hidden_dim,
            num_layers=num_layers_G,
            dropout=dropout_G,
            bottleneck_dim=bottleneck_dim,
            concat_dim=concat_dim,
            output_dim=output_dim,
            num_steps=num_steps,
            num_paths=num_paths,
            flag=flag,
            device=device
        )
        self.Discriminator = Encoder(
            embedding_dim=embedding_dim,
            batch_size=batch_size,
            hidden_dim=enc_hidden_dim,
            num_layers=num_layers_D,
            dropout=dropout_D,
            discriminator_flag=True,
            device=device
        )

        self.Generator.apply(init_weights)
        self.Discriminator.apply(init_weights)

        self.generator_optimizer = torch.optim.AdamW(
            self.Generator.parameters(),
            lr=G_learning_rate,
            weight_decay=G_weight_decay
        )
        self.discriminator_optimizer = torch.optim.AdamW(
            self.Discriminator.parameters(),
            lr=D_learning_rate,
            weight_decay=D_weight_decay
        )

    def pre_train_step(self):
        self.count_G = 0
        self.count_D = 0
        self.validation_losses_G, self.validation_losses_D = [], []
        self.previous_validation_losses_G, self.previous_validation_losses_D = self.validation_losses_G, self.validation_losses_D

    def pre_test_step(self, dim):
        self.eval_pred = torch.zeros((dim, self.seq_length, self.batch_size, 2))
        self.eval_targ = torch.zeros((dim, self.seq_length, self.batch_size, 2))

    def train_step(
            self,
            data: torch.Tensor,
            targ: torch.Tensor,
    ):
        generated_trajectory = self.Generator(data, targ, self.min_max_pos, self.min_max_state)

        D_gen = self.Discriminator(generated_trajectory)

        D_targ = self.Discriminator(targ)
        self.update_step(D_gen, D_targ, d_g='D')
        D_gen = self.Discriminator(generated_trajectory)
        self.update_step(D_gen, d_g='G')

    def validation_step(
            self,
            data: torch.Tensor,
            targ: torch.Tensor,
    ):
        generated_trajectory = self.Generator(data, targ, self.min_max_pos, self.min_max_state)

        D_gen = self.Discriminator(generated_trajectory)
        D_targ = self.Discriminator(targ)

        return D_gen, D_targ

    def test_step(
            self,
            curr_num: int,
            data: torch.Tensor,
    ):
        empty = torch.zeros(data[3].size(), device=self.device)
        generated_trajectory = self.Generator(data, empty, self.min_max_pos, self.min_max_state, train=False)
        self.eval_pred[curr_num, :, :, :] = generated_trajectory
        self.eval_targ[curr_num, :, :, :] = target_trajectory

    def update_step(
            self,
            fake_data: torch.Tensor,
            real_data: torch.Tensor = None,
            d_g: str = 'D',
    ):
        if d_g == 'D':
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z))) #
            self.discriminator_optimizer.zero_grad()
            # Loss with all-real batch
            real_label = torch.ones(real_data.size(), device=self.device) * random.uniform(0.9, 1.)
            loss_D_real = self.loss(real_data, real_label)
            # Loss with all-fake batch
            fake_label = torch.ones(fake_data.size(), device=self.device) * random.uniform(0, 0.1)
            loss_D_fake = self.loss(fake_data, fake_label)
            # Loss computation + Update
            loss_D = loss_D_real + loss_D_fake
            loss_D.backward(retain_graph=True)
            self.discriminator_optimizer.step()

        if d_g == 'G':
            # (2) Update G network: maximize log(D(G(z))) #
            self.generator_optimizer.zero_grad()
            label = torch.ones(fake_data.size(), device=self.device) * random.uniform(0.9, 1.)
            loss_G = self.loss(fake_data, label)
            loss_G.backward(retain_graph=True)
            self.generator_optimizer.step()

    def post_validation_step(
            self,
            fake_data,
            real_data
    ):
        # D loss #
        # Loss with all-real batch
        real_label = torch.ones(real_data.size(), device=self.device) * random.uniform(0.8, 1.)
        loss_D_real = self.loss(real_data, real_label)
        # Loss with all-fake batch
        fake_label = torch.ones(fake_data.size(), device=self.device) * random.uniform(0, 0.2)
        loss_D_fake = self.loss(fake_data, fake_label)
        # Loss computation + Update
        loss_D = loss_D_real + loss_D_fake

        # G loss #
        label = torch.ones(fake_data.size(), device=self.device) * random.uniform(0.8, 1.)
        loss_G = self.loss(fake_data.detach(), label)

        self.validation_losses_G.append(loss_G)
        self.validation_losses_D.append(loss_D)

    def stopping(self):
        if self.previous_validation_losses_G is None or self.previous_validation_losses_D is None:
            self.count_G = 0
            self.count_D = 0
        else:
            self.count_G, self.done_G = EarlyStopping(sum(self.validation_losses_G)/len(self.validation_losses_G),
                                                      sum(self.previous_validation_losses_G)/len(self.previous_validation_losses_G),
                                                      self.count_G)
            self.count_D, self.done_D = EarlyStopping(sum(self.validation_losses_D)/len(self.validation_losses_D),
                                                      sum(self.previous_validation_losses_D)/len(self.previous_validation_losses_D),
                                                      self.count_D)
            if self.done_G and self.done_D:
                self.done = True


class Generator(nn.Module):
    def __init__(
            self,
            in_channels: int = 64,
            in_height: int = 360,
            in_width: int = 640,
            out_channels: int = 64,
            hidden_channels: int = 64,
            num_blocks: int = 18,
            embedding_dim: int = 64,
            enc_hidden_dim: int = 16,
            dec_hidden_dim: int = 32,
            num_layers: int = 1,
            dropout: float = 0.75,
            bottleneck_dim: int = 1024,
            concat_dim: int = 320,
            output_dim: int = 2,
            num_steps: int = 8,
            num_paths: int = 4,
            flag: list = None,
            device=torch.device("cpu")
    ):
        super(Generator, self).__init__()

        if flag is None:
            flag = [0, 0, 0]
        self.processor = Processor(
            in_channels=in_channels,
            in_height=in_height,
            in_width=in_width,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            num_blocks=num_blocks,
            embedding_dim=embedding_dim,
            hidden_dim=enc_hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            bottleneck_dim=bottleneck_dim,
            concat_dim=concat_dim,
            flag=flag,
            device=device
        )
        self.decoder = Decoder(
            embedding_dim=concat_dim,
            hidden_dim=dec_hidden_dim,
            output_dim=output_dim,
            bottleneck_dim=bottleneck_dim,
            num_layers=num_layers,
            dropout=dropout,
            num_steps=num_steps,
            num_paths=num_paths,
            device=device
        )

    def __call__(
            self,
            data: torch.Tensor,
            targ: torch.Tensor,
            min_max_pos: torch.Tensor,
            min_max_state: torch.Tensor,
            train: bool = True
    ):
        y = self.forward(data, targ, min_max_pos, min_max_state, train)
        return y

    def forward(
            self,
            data: torch.Tensor,
            targ: torch.Tensor,
            min_max_pos: torch.Tensor,
            min_max_state: torch.Tensor,
            train: bool
    ):
        processed_data = self.processor(data, min_max_pos, min_max_state)
        generated_trajectory = self.decoder(processed_data, targ, train)

        return generated_trajectory
