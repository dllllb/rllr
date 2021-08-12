import copy
import logging
import matplotlib.pyplot as plt
import numpy
import torch
import torch.nn.functional as F
import tqdm

from math import log, exp
from functools import partial
from torch.nn.utils import clip_grad_norm_

from rllr.utils import get_conf, switch_reproducibility_on
from rllr.utils.logger import init_logger
from rlplay.engine import BaseActorModule
from rlplay.engine.utils.plyr import apply, suply, xgetitem
from rlplay.engine.rollout import multi, single
from rlplay.engine.rollout.evaluate import evaluate

from train_worker import gen_navigation_env, get_encoders, get_master_worker_net

logger = logging.getLogger(__name__)


class EncoderActor(BaseActorModule):
    def __init__(self, state_encoder, epsilon=0.1):
        super().__init__()

        self.encoder = state_encoder

        # for updating the exploration epsilon in the clones
        self.register_buffer('epsilon', torch.tensor(epsilon))

    def forward(self, obs, act, rew, fin, *, hx=None, stepno=None, virtual=False):
        qv, hx = self.encoder(obs), ()
        val, actions = qv.max(dim=-1)

        if self.training:
            *head, n_actions = qv.shape
            actions = actions.where(
                torch.rand(head, device=self.epsilon.device).gt(self.epsilon),
                torch.randint(n_actions, size=head, device=self.epsilon.device))

        return actions, hx, dict(q=qv, value=val)


def timeshift(state, *, shift=1):
    """Get current and shfited slices of nested objects."""
    # use xgetitem to lett None through
    # XXX `curr[t]` = (x_t, a_{t-1}, r_t, d_t), t=0..T-H
    curr = suply(xgetitem, state, index=slice(None, -shift))

    # XXX `next[t]` = (x_{t+H}, a_{t+H-1}, r_{t+H}, d_{t+H}), t=0..T-H
    next = suply(xgetitem, state, index=slice(shift, None))

    return curr, next


# @torch.enable_grad()
def ddq_learn(fragment, module, *, gamma=0.95, target=None, double=False):
    r"""Compute the Double-DQN loss over a _contiguous_ fragment of a trajectory.

    Details
    -------
    In Q-learning the action value function minimizes the TD-error

    $$
        r_{t+1}
            + \gamma 1_{\neg d_{t+1}} v^*(z_{t+1})
            - q(z_t, a_t; \theta)
        \,, $$

    w.r.t. Q-network parameters $\theta$ where $z_t$ is the actionable state,
    $r_{t+1}$ is the reward for $s_t \to s_{t+1}$ transition. The value of
    $z_t$ include the current observation $x_t$ and the recurrent state $h_t$,
    the last action $a_{t-1}$, the last reward $r_t$, and termination flag
    $d_t$.

    In the classic Q-learning there is no target network and the next state
    optimal state value function is bootstrapped using the current Q-network
    (`module`):

    $$
        v^*(z_{t+1})
            \approx \max_a q(z_{t+1}, a; \theta)
        \,. $$

    The DQN method, proposed by

        [Minh et al. (2013)](https://arxiv.org/abs/1312.5602),

    uses a secondary Q-network (`target`) to estimate the value of the next
    state:

    $$
        v^*(z_{t+1})
            \approx \max_a q(z_{t+1}, a; \theta^-)
        \,, $$

    where $\theta^-$ are the frozen parameters of the target Q-network. The
    Double DQN algorithm of

        [van Hasselt et al. (2015)](https://arxiv.org/abs/1509.06461)

    unravels the $\max$ operator as
    $
        \max_k u_k \equiv u_{\arg \max_k u_k}
    $
    and replaces the outer $u$ with the Q-values of the target Q-network, while
    computing the inner $u$ (inside the $\arg\max$) with the current Q-network.
    Specifically, the Double DQN value estimate is

    $$
        v^*(z_{t+1})
            \approx q(z_{t+1}, \hat{a}_{t+1}; \theta^-)
            \,,
            \hat{a}_{t+1}
                = \arg \max_a q(z_{t+1}, a; \theta)
        \,, $$

    for $
        \hat{a}_{t+1}
            = \arg \max_a q(s_{t+1}, a; \theta)
    $ being the action taken by the current Q-network $\theta$ at $z_{t+1}$.

    Recurrent DQN
    -------------
    The key problem with the recurrent state $h_t$ in $z_t$ is its representaion
    drift: the endogenous states used for collecting trajectory data during the
    rollout are produced by an actor with stale perameters $\theta_{\text{old}}$,
    and thus might have high discrepancy with the recurrent state produced by
    the current Q-network $\theta$ or the target $\theta-_$. To mitigate this

        [Kapturowski et al. (2018)](https://openreview.net/forum?id=r1lyTjAqYX)

    proposed to spend a slice `burnin` of the recorded trajectory on
    aligning the recurrent representation. Specifically, starting with $h_0$
    (contained in `fragment.hx`) they propose to launch two sequences $h_t$
    and $h^-_t$ from the same $h^-_0 = h_0$ using $q(\cdot; \theta)$ and
    $q(\cdot; \theta^-)$, respectively.
    """

    trajectory, hx = fragment.state, fragment.hx
    obs, act, rew, fin = trajectory.obs, trajectory.act, trajectory.rew, trajectory.fin

    # get $Q(z_t, h_t, \cdot; \theta)$ for all t=0..T
    _, _, info_module = module(
        obs, act, rew, fin, hx=hx, stepno=trajectory.stepno)

    # get the next state `state[t+1]` $z_{t+1}$ to access $a_t$
    state_next = suply(xgetitem, trajectory, index=slice(1, None))

    # $\hat{A}_t$, the module's response to current and next state,
    #  contains the q-values. `curr` is $q(z_t, h_{t+1}, \cdot; \theta)$
    #  and `next` is $q(z_{t+1}, h_{t+1}, \cdot; \theta)$ is `next`.
    info_module_curr, info_module_next = timeshift(info_module)

    # get $q(z_t, h_t, a_t; \theta)$ for all t=0..T-1
    q_replay = info_module_curr['q'].gather(-1, state_next.act.unsqueeze(-1))

    # get $\hat{v}_{t+1}(z_{t+1}) = ...$
    with torch.no_grad():
        if target is None:
            # get $... = \max_a Q(z_{t+1}, h_{t+1}, a; \theta)$
            q_value = info_module_next['q'].max(dim=-1, keepdim=True).values

        else:
            _, _, info_target = target(
                obs, act, rew, fin, hx=hx, stepno=trajectory.stepno)

            info_target_next = suply(xgetitem, info_target, index=slice(1, None))
            if not double:
                # get $... = \max_a Q(z_{t+1}, h^-_{t+1}, a; \theta^-)$
                q_value = info_target_next['q'].max(dim=-1, keepdim=True).values

            else:
                # get $\hat{a}_{t+1} = \arg \max_a Q(z_{t+1}, h_{t+1}, a; \theta)$
                hat_act = info_module_next['q'].max(dim=-1).indices.unsqueeze(-1)

                # get $... = Q(z_{t+1}, h^-_{t+1}, \hat{a}_{t+1}; \theta^-)$
                q_value = info_target_next['q'].gather(-1, hat_act)

        # get $r_{t+1} + \gamma 1_{d_{t+1}} \hat{v}_{t+1}(z_{t+1})$ using inplace ops
        q_value.masked_fill_(state_next.fin.unsqueeze(-1), 0.)
        q_value.mul_(gamma).add_(state_next.rew.unsqueeze(-1))

    # td-error ell-2 loss
    return F.mse_loss(q_replay, q_value, reduction='sum')


def collate(records):
    """collate identically keyed dicts"""
    out, n_records = {}, 0
    for record in records:
        for k, v in record.items():
            out.setdefault(k, []).append(v)

    return out


def main(args=None):
    config = get_conf(args)
    switch_reproducibility_on(config['seed'])

    state_encoder, goal_state_encoder = get_encoders(config)
    net = get_master_worker_net(state_encoder, goal_state_encoder, config)

    gamma = 0.6
    use_target = False
    use_double = False

    factory_eval = partial(gen_navigation_env, conf=config['env'])
    factory = partial(gen_navigation_env, conf=config['env'])

    learner = EncoderActor(net)

    learner.train()
    device_ = torch.device('cuda:0')  # torch.device('cpu')
    learner.to(device=device_)

    optim = torch.optim.SGD(learner.parameters(), lr=1e-1)
    T, B = 25, 8

    batchit = multi.rollout(
        factory,
        learner,
        n_steps=T,
        n_actors=3,  # the number of parallel actors
        n_per_actor=B,  # the number of independent environments run in each actor
        n_buffers=16,  # the size of the pool of buffers, into which rollout fragments are collected. Should not be less than `n_actors`.
        n_per_batch=2,  # the number of fragments collated into a batch
        sticky=False,
        pinned=False,
        clone=True,
        close=False,
        device=device_,
        start_method='spawn',  # fork in notebook for macos, spawn in linux
    )

    test_it = evaluate(factory_eval, learner, n_envs=4, n_steps=500,
                   clone=False, device=device_, start_method='spawn')

    torch.set_num_threads(1)

    # the training loop
    losses, rewards = [], []
    decay = -log(2) / 50  # exploration epsilon halflife
    for epoch in tqdm.tqdm(range(100)):
        # freeze the target for q
        target = copy.deepcopy(learner) if use_target else None

        for j, batch in zip(range(20), batchit):
            loss = ddq_learn(batch, learner, target=target,
                             gamma=gamma, double=use_double)


            optim.zero_grad()
            loss.backward()
            grad = clip_grad_norm_(learner.parameters(), max_norm=1e3)
            optim.step()

            losses.append(dict(
                loss=float(loss), grad=float(grad),
            ))

        learner.epsilon.mul_(exp(decay)).clip_(0.1, 1.0)

        # fetch the evaluation results lagged by one inner loop!
        rewards.append(next(test_it))

    # close the generators
    batchit.close()
    test_it.close()

    data = {k: numpy.array(v) for k, v in collate(losses).items()}
    plt.semilogy(data['loss'])
    plt.semilogy(data['grad'])


if __name__ == '__main__':
    init_logger(__name__)
    init_logger('rllr.env.wrappers')
    main()
