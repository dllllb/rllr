import math
import torch

from torch import nn
from torch.nn import functional as F
from typing import Tuple

class VQEmbedding(nn.Embedding):
    r"""Vector-quantized mebedding layer.

    Note
    ----
    My own implementation taken from
        ~/Github/general-scribbles/coding/vq-VAE.ipynb

    Details
    -------
    The key idea of the [vq-VAE](https://arxiv.org/abs/1711.00937) is how
    to train the nearest-neighbour-based quantization embeddings and how
    the gradients are to be backpropped through them:

    $$
    \operatorname{vq}(z; e)
        = \sum_k e_k 1_{R_k}(z)
        \,,
        \partial_z \operatorname{vq}(z; e) = \operatorname{id}
        \,. $$

    This corresponds to a degenerate conditional categorical rv
    $k^\ast_z$ with distribution $
        p(k^\ast_z = j\mid z)
            = 1_{R_j}(z)
    $ where

    $$
    R_j = \bigl\{
        z\colon
            \|z - e_j\|_2 < \min_{k\neq j} \|z - e_k\|_2
        \bigr\}
    \,, $$

    are the cluster affinity regions w.r.t. $\|\cdot \|_2$ norm. Note that we
    can compute

    $$
    k^\ast_z
        := \arg \min_k \frac12 \bigl\| z - e_k \bigr\|_2^2
        = \arg \min_k
            \frac12 \| e_k \|_2^2
            - \langle z, e_k \rangle
        \,. $$

    The authors propose STE for grads and mutual consistency losses for
    the embeddings:
    * $\| \operatorname{sg}(z) - e_{k^\ast_z} \|_2^2$ -- forces the embeddings
    to match the latent cluster's centroid (recall the $k$-means algo)
      * **NB** in the paper they use just $e$, but in the latest code they use
      the selected embeddings
      * maybe we should compute the cluster sizes and update to the proper
      centroid $
          e_j = \frac1{
              \lvert {i: k^\ast_{z_i} = j} \rvert
          } \sum_{i: k^\ast_{z_i} = j} z_i
      $.
    * $\| z - \operatorname{sg}(e_{k^\ast_z}) \|_2^2$ -- forces the encoder
    to produce the latents, which are consistent with the cluster they are
    assigned to.
    """
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        alpha: float = 0.,
        eps: float = 1e-5,
    ) -> None:
        super().__init__(
            num_embeddings,
            embedding_dim,
            max_norm=None,
            padding_idx=None,
            scale_grad_by_freq=False,
            sparse=False,
        )

        self.alpha, self.eps = alpha, eps

        # if `alpha` is zero then `.weight` is updated by other means
        self.register_buffer('ema_vecs', None)
        self.register_buffer('ema_size', None)
        if self.alpha <= 0:
            return

        # demote `.weight` to a buffer and disable backprop for it
        # XXX can promote buffer to parameter, but not back, so we `delattr`.
        #  Also non-inplace `.detach` creates a copy not reflected in referrers.
        weight = self.weight
        delattr(self, 'weight')
        self.register_buffer('weight', weight.detach_())

        # allocate buffer for tracking k-means cluster centroid updates
        self.register_buffer(
            'ema_vecs', self.weight.clone(),
        )
        self.register_buffer(
            'ema_size', torch.zeros_like(self.ema_vecs[:, 0]),
        )

    @torch.no_grad()
    def _update(
        self,
        input: torch.Tensor,
        indices: torch.Tensor,
    ) -> None:
        """Update the embedding vectors by Exponential Moving Average.
        """

        # `input` is `... x F`, `indices` are `...`
        affinity = F.one_hot(indices, self.num_embeddings).to(input)
        # XXX 'affinity' is `... x C`

        # sum the F-dim input vectors into bins by affinity
        #  S_j = \sum_i 1_{k_i = j} x_i
        #  n_j = \lvert i: k_i=j \rvert
        upd_vecs = torch.einsum('...f, ...k -> kf', input, affinity)
        upd_size = torch.einsum('...k -> k', affinity)

        # track cluster size and unnormalized vecs with EMA
        self.ema_vecs.lerp_(upd_vecs, self.alpha)
        self.ema_size.lerp_(upd_size, self.alpha)

        # Apply \epsilon-Laplace correction
        n = self.ema_size.sum()
        coef = n / (n + self.num_embeddings * self.eps)
        size = coef * (self.ema_size + self.eps).unsqueeze(1)
        self.weight.data.copy_(self.ema_vecs / size)

    @torch.no_grad()
    def lookup(
        self,
        input: torch.Tensor,
    ) -> torch.Tensor:
        """Lookup the index of the nearest embedding.
        """
        emb = self.weight
        # k(z) = \arg \min_k \|E_k - z\|^2
        #      = \arg \min_k \|E_k\|^2 - 2 E_k^\top z + \|z\|^2
        # XXX no need to compute the norm fully since we do not
        #  backprop through the input when clustering.

        sqr = (emb * emb).sum(dim=1)
        cov = torch.einsum('...j, kj -> ...k', input, emb)
        return torch.argmin(sqr - 2 * cov, dim=-1)

    def fetch(
        self,
        indices: torch.LongTensor,
        at: int = -1,
    ) -> torch.Tensor:
        """fetch embeddings and put their dim at position `at`"""
        vectors = super().forward(indices)  # call Embedding.forward

        # indices.shape is batch x *spatial
        dims = list(range(indices.ndim))
        at = (vectors.ndim + at) if at < 0 else at
        # vectors.permute(0, input.ndim-1, *range(1, input.ndim-1))
        return vectors.permute(*dims[:at], indices.ndim, *dims[at:])

    def forward(
        self,
        input: torch.Tensor,
        reduction: str = 'sum',
    ) -> Tuple[torch.Tensor]:
        """vq-VAE clustering with straight-through estimator and commitment
        losses.

        Details
        -------
        Implements

            [van den Oord et al. (2017)](https://arxiv.org/abs/1711.00937).

        See further details in the class docstring.
        """
        # lookup the index of the nearest embedding and fetch it
        indices = self.lookup(input)
        vectors = self.fetch(indices)

        # commitment and embedding losses, p. 4 eq. 3.
        # loss = - \log p(x \mid q(x))
        #      + \|[z(x)] - q(x)\|^2     % embedding loss (dictionary update)
        #      + \|z(x) - [q(x)]\|^2   % encoder's commitment loss
        # where z(x) is output of the encoder network
        #       q(x) = e_{k(x)}, for k(x) = \arg\min_k \|z(x) - e_k\|^2
        # XXX p.4 `To make sure the encoder commits to an embedding and
        #          its output does not grow, since the volume of the embedding
        #          space is dimensionless`
        # XXX the embeddings receive no gradients from the reconstruction loss
        embedding = F.mse_loss(vectors, input.detach(), reduction=reduction)
        commitment = F.mse_loss(input, vectors.detach(), reduction=reduction)

        # the straight-through grad estimator: copy grad from q(x) to z(x)
        output = input + (vectors - input).detach()

        # update the weights only if we are in training mode
        if self.training and self.alpha > 0:
            self._update(input, indices)
            # XXX `embedding` loss is non-diffable if we use ewm updates

        return output, indices, embedding, commitment


class SendRecv(nn.Module):
    def __init__(
        self,
        send: nn.Module,
        recv: nn.Module,
        num_embeddings: int,
        embedding_dim: int,
        alpha: float=0.9
    ) -> None:
        super().__init__()
        self.send = send
        self.recv = recv
        self.vq = VQEmbedding(num_embeddings, embedding_dim, alpha=alpha)

    def forward(self, input: torch.Tensor, reduction: str = 'sum') -> Tuple[torch.Tensor]:
        # input is `B C`
        # sender `send : X -> R^{M F}`

        z = self.send(input) # B C -> B M F

        # vq-vae `vq : R^F -> [K] -> R^F`
        emb, codes, embedding, commitment = self.vq(z, reduction)

        # receiver `recv : R^{M F} -> X`
        x_hat = self.recv(emb)

        # compute the enropy of the quantization layer
        prob = torch.bincount(codes.flatten(), minlength=self.vq.num_embeddings) / codes.numel()
        ent = - F.kl_div(prob.new_zeros(()), prob, reduction='sum')

        return x_hat, codes, embedding, commitment, float(ent) / math.log(2)
