import torch
from torch import nn


class DummyRawStateEncoder(nn.Module):
    """
    code minigrid raw states as agent position and direction via one hot encoder
    """
    def __init__(self, grid_size):
        super().__init__()
        self.output_size = (grid_size - 2) ** 2 + 4

    def forward(self, x: torch.Tensor):
        x = x[:, 1:-1, 1:-1]
        agent_pos = (x[:, :, :, 0] == 10)
        directions = x[agent_pos][:, -1:]

        agent_pos = agent_pos.flatten(start_dim=1).float()
        directions = torch.eq(directions, torch.arange(4, device=x.device)).float()

        return torch.cat([agent_pos, directions], dim=1)


class DummyImageStateEncoder(nn.Module):
    """
    code minigrid image states as agent position and direction via one hot encoder
    """
    def __init__(self, grid_size, conf):
        super().__init__()
        self.tile_size = conf['tile_size']
        self.output_size = (grid_size // self.tile_size - 2) ** 2 + 4

    def forward(self, x: torch.Tensor):
        x = x.permute(0, 3, 1, 2)
        x = x[:, :1, self.tile_size:-self.tile_size, self.tile_size:-self.tile_size]
        a = nn.MaxPool2d(self.tile_size, self.tile_size)(x) > 0

        b = x[a.repeat_interleave(self.tile_size, dim=-2).repeat_interleave(self.tile_size, dim=-1)]
        b = b.reshape(-1, self.tile_size, self.tile_size)
        b = [b[:, :, -1].sum(dim=1) == 0, b[:, -1].sum(dim=1) == 0, b[:, :, 0].sum(dim=1) == 0, b[:, 0].sum(dim=1) == 0]

        directions = torch.cat([x.unsqueeze(-1) for x in b], dim=1).float()
        agent_pos = a.flatten(start_dim=1).float()

        return torch.cat([agent_pos, directions], dim=1)
