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

    def forward(self, states, goal_states):
        goal_states_emb = self.master(goal_states)
        return self.worker(states, goal_states_emb)


def get_master_worker_net(state_encoder, goal_state_encoder, action_size, config):
    worker = WorkerNetwork(
        state_encoder=state_encoder,
        emb_size=goal_state_encoder.output_size,
        action_size=action_size,
        config=config['worker']
    )
    return MasterWorkerNetwork(master=goal_state_encoder, worker=worker)
