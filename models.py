import torch
from torch import nn


class WorkerNetwork(nn.Module):
    def __init__(self, state_encoder, emb_size, action_size, config):
        super().__init__()
        self.state_encoder = state_encoder
        hidden_size = config['head.hidden_size']
        fc_layers = [
            nn.Linear(state_encoder.output_size + emb_size, hidden_size),
            nn.ReLU(inplace=False),
            nn.Linear(hidden_size, action_size)
        ]
        self.fc = nn.Sequential(*fc_layers)
        self.output_size = action_size

    def forward(self, states, goal_states_emb):
        states_emb = self.state_encoder(states)
        x = torch.cat((goal_states_emb, states_emb), 1)
        return self.fc(x)


class MasterWorkerNetwork(nn.Module):
    """
    Master-Worker model
    """
    def __init__(self, master, worker):
        super(MasterWorkerNetwork, self).__init__()
        self.master = master
        self.worker = worker

    def forward(self, states, goal_states, goal_states_emb=None):
        goal_states_emb = self.master(goal_states) if goal_states_emb is None else goal_states_emb
        return self.worker(states, goal_states_emb)


class ExpectedStepsAmountRegressor(nn.Module):
    """
    Expected Steps Amount Regressor
    """
    def __init__(self, conf):
        super().__init__()
        cnn_output_size = conf['input_grid_size']
        cur_channels = 6
        conv_layers = []
        for n_channels, kernel_size, max_pool in zip(conf['n_channels'], conf['kernel_sizes'], conf['max_pools']):
            conv_layers.append(nn.Conv2d(cur_channels, n_channels, kernel_size))
            cnn_output_size -= kernel_size - 1
            cur_channels = n_channels
            if max_pool > 1:
                conv_layers.append(nn.MaxPool2d(max_pool, max_pool))
                cnn_output_size //= max_pool

        self.conv_net = nn.Sequential(*conv_layers)
        conv_output_size = cur_channels * cnn_output_size ** 2

        hidden_size = conf['head.hidden_size']
        fc_layers = [
            # nn.BatchNorm1d(conv_output_size),
            nn.Linear(conv_output_size, hidden_size),
            nn.ReLU(inplace=False),
            nn.Linear(hidden_size, 1)
        ]
        self.fc = nn.Sequential(*fc_layers)
        self.output_size = 1

    def forward(self,  states, goal_states):
        x = torch.cat((states, goal_states), dim=3).permute(0, 3, 1, 2)
        x = self.conv_net(x).reshape(x.shape[0], -1)
        return self.fc(x).squeeze()


def get_master_worker_net(state_encoder, goal_state_encoder, action_size, config):
    worker = WorkerNetwork(
        state_encoder=state_encoder,
        emb_size=goal_state_encoder.output_size,
        action_size=action_size,
        config=config['worker']
    )
    return MasterWorkerNetwork(master=goal_state_encoder, worker=worker)
