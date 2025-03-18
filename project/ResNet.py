from utils import *


class ResNet(nn.Module):
    def __init__(
            self,
            in_channels: int = 64,
            out_channels: int = 64,
            hidden_channels: int = 64,
            num_blocks: int = 18,
            kernel_size: int = 7,
            stride: int = 1,
            padding: int = 3,
            dropout: float = 0.75,
            device=torch.device("cpu")
    ):
        super(ResNet, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.periodic_zeros_padding = PeriodicPadding2D(3, device=device)
        self.image_projection = nn.Conv2d(in_channels, hidden_channels, kernel_size=kernel_size, stride=stride, padding=padding).to(device)
        self.res_net_blocks = make_multilayer(
            block=ResidualBlock,
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            num_blocks=num_blocks,
            change=1,
            device=device
        )
        self.norm = nn.BatchNorm2d(hidden_channels).to(device)
        self.out = nn.Conv2d(hidden_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=(padding+3)).to(device)
        self.relu = nn.ReLU().to(device)

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
        x = x.permute(0, 3, 1, 2)
        x = self.dropout(x)
        x = self.periodic_zeros_padding(x)
        x = self.image_projection(x)
        x = self.res_net_blocks(x)
        x = self.norm(x)
        x = self.relu(x)
        x = self.out(x)
        return x
