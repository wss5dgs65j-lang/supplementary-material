# evaluation/yan/YANModel.py

# ----------------------------------------------------------------------------------------- #
# This script copies the model architecture of YAN from code/trainer and allows generation  #
# ----------------------------------------------------------------------------------------- #


import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Tuple





# ------- Probability path class ------- #
class LinearConditionalProbabilityPath:
    """Linear path from x0 to x1.
    Args:
        x0: Starting state of the probability path, (B, L, d)
        x1: Ending state of the probability path, (B, L, d)
    """
    def __init__(self, x0: torch.Tensor, x1: torch.Tensor):
        assert x0.shape == x1.shape, "In the linear path, x0 and x1 must have the same shape."
        self.x0 = x0
        self.x1 = x1
        self.dim = x1.dim()
        self.B = x1.shape[0]

    def sample_conditional_path(self, t: torch.Tensor) -> torch.Tensor:
        """Generate samples of the conditional distribution p_t(xt|x1): xt = (1-t) * x0 + t * x1.
        Args:
            t: (B,)
        Returns:
            xt: (B, L, d)
        """
        assert t.dim() == 1 and t.shape[0] == self.B, "The shape of t should be (B,)"
        t = t.view((-1,) + (1,) * (self.dim-1))
        return (1 - t) * self.x0 + t * self.x1
    
    def conditional_vector_field(self) -> torch.Tensor:
        """Evaluate the conditional vector field at state xt and time t: u_t(xt|x1) = x1 - x0"""
        return self.x1 - self.x0



# ------- ODE class ------- #
class YANODE:
    """The ODE of YAN
    Args:
        model: A pytorch model that inputs (xt, t, ...), and output the vector field at (xt, t).
    """
    def __init__(self, model: nn.Module):
        self.model = model

    def drift_coefficient(self, xt: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
        """Return the drift coefficient of YAN at state xt and time t
        Args:
            xt: current flow status, (B, L, d).
            t: current time, (B,).
            **kwargs: h_enc, att_mask, enc_att_mask, expert_idx
        Returns:
            vector field: (B, L, d)
        """
        h_enc = kwargs['h_enc']
        att_mask = kwargs.get('att_mask')
        enc_att_mask = kwargs.get('enc_att_mask')
        expert_idx = kwargs.get('expert_idx')

        if expert_idx is None:
            raise ValueError("expert_idx must be provided for MoE Flow ODE (fixed-expert integration).")

        return self.model.forward_expert(xt, t, h_enc, expert_idx, att_mask, enc_att_mask) # type: ignore


# ------- Solver class ------- #
class Solver(ABC):
    @abstractmethod
    def step(self, xt: torch.Tensor, t: torch.Tensor, dt: torch.Tensor, **kwargs) -> torch.Tensor:
        """Take one simulation step
        Args:
            xt: current state at t, (B, *dim)
            t: current time, (B,)
            dt: change in time, (B,)
        Returns:
            next state at time t+dt, (B, *dim)
        """
        pass

    def solve(self, x: torch.Tensor, ts: torch.Tensor, trajectory: bool = False, **kwargs) -> torch.Tensor:
        """Solve for the state x after ts timesteps
        Args:
            x: Initial state at time ts[0], (B, *dim)
            ts: Time steps, (n_timesteps, B)
            trajectory: Whether to return the entire trajectory
            **kwargs: h_enc, att_mask, enc_att_mask, expert_idx
        Returns:
            If trajectory=False, return the final state at time ts[-1], (B, *dim). 
            Else, return the trajectory, (n_timesteps, B, *dim)
        """
        xs = [x.clone()] if trajectory else []
        n_timesteps = ts.shape[0]
        for t_idx in range(n_timesteps-1):
            t = ts[t_idx,:]                     # (B,)
            dt = ts[t_idx+1,:] - t              # (B,)
            x = self.step(x, t, dt, **kwargs)   # (B, *dim)
            if trajectory:
                xs.append(x.clone())
        return torch.stack(xs, dim=0) if trajectory else x


class EulerSolver(Solver):
    def __init__(self, ode: YANODE):
        self.ode = ode
    
    def step(self, xt: torch.Tensor, t: torch.Tensor, dt: torch.Tensor, **kwargs) -> torch.Tensor:
        """One Euler step.
        Args:
            xt: current state at t, (B, L, d)
            t: current time, (B,)
            dt: change in time, (B,)
        Returns:
            next state at time t+dt, (B, L, d)
        """
        dt_new_shape = (-1,) + (1,) * (xt.dim() - 1)
        vec_field = self.ode.drift_coefficient(xt, t, **kwargs)
        return xt + vec_field * (dt.view(dt_new_shape))






## ------- Sinusoidal Position Embeddings ------- ##
class SinusoidalPositionEmbeddings(nn.Module):
    """Time embedding. The forward pass takes time t: (B,) and returns time embedding: (B, emb_dim).
    Args:
        emb_dim: embedding dimension
    """
    def __init__(self, emb_dim:int):
        super().__init__()
        self.emb_dim = emb_dim

    def forward(self, t):
        half_dim = self.emb_dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=t.device, dtype=torch.float32) * -embeddings)
        embeddings = t.unsqueeze(1) * embeddings.unsqueeze(0) # (B, half_dim)
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1) # (B, emb_dim)
        return embeddings
    

## ------- Multi-Head Attention ------- ##
class MultiHeadAttention(nn.Module):
    """Multi-head attention module that supports both self-attention and cross-attention.

    For self-attention, the forward pass takes one input tensor `x`.
    For cross-attention, it takes `x` (for Query) and `kv_source` (for Key/Value).

    Args:
        d_in: dimension of the input embedding vectors
        d_out: dimension of the output context vectors
        dropout: dropout rate
        n_heads: number of heads
        qkv_bias: whether to include bias for the Linear layers for query, key, value computations
    """
    def __init__(self, d_in: int, d_out: int, dropout: float, n_heads: int, qkv_bias: bool = False):
        super().__init__()

        assert d_out % n_heads==0, 'd_out must be divisible by n_heads'
        self.d_out = d_out
        self.n_heads = n_heads
        self.head_dim = d_out//n_heads

        self.W_q = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_k = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_v = nn.Linear(d_in, d_out, bias=qkv_bias)

        self.out_proj = nn.Linear(d_out, d_out)

        self.dropout_p = dropout
    
    def forward(self, x: torch.Tensor, 
                kv_source: torch.Tensor | None = None,
                att_mask: torch.Tensor | None = None,
                precomputed_kv: tuple[torch.Tensor, torch.Tensor] | None = None) -> torch.Tensor:
        """A forward pass of the multi-head attention module.
        Args:
            x: input embedding vectors, (B, q_len, d_in)
            kv_source (optional): 
                If None, performs self-attention, then kv_len=q_len.
                If provided, performs cross-attention, (B, kv_len, d_in).
            att_mask (optional):
                If provided, avoid performing attention on padding tokens, (B, q_len, kv_len).
                - 1 for valid tokens that are not paddings
                - 0 for invalid tokens that are paddings
            precomputed_kv (optional): (k_cached, v_cached). If provided, use cache kv and avoid 
                re-computation. Both k_cached and v_cached are (B, n_heads, kv_len, head_dim).
        Returns:
            context_vec: context vectors, (B, q_len, d_out)
        """
        kv_source = x if kv_source is None else kv_source

        bs, q_len, _ = x.shape
        _, kv_len, _ = kv_source.shape

        # Compute Q
        q = self.W_q(x) # (bs, q_len, d_out)
        q = q.view(bs, q_len, self.n_heads, self.head_dim).transpose(1, 2) # (bs, n_heads, q_len, head_dim)
        
        # Compute K/V or use cached K/V
        if precomputed_kv is None:
            k = self.W_k(kv_source) # (bs, kv_len, d_out)
            v = self.W_v(kv_source) # (bs, kv_len, d_out)
            k = k.view(bs, kv_len, self.n_heads, self.head_dim).transpose(1, 2) # (bs, n_heads, kv_len, head_dim)
            v = v.view(bs, kv_len, self.n_heads, self.head_dim).transpose(1, 2) # (bs, n_heads, kv_len, head_dim)
        else:
            k, v = precomputed_kv # (bs, n_heads, kv_len, head_dim)
            assert k.shape == v.shape == (bs, self.n_heads, kv_len, self.head_dim)
        
        # Scaled dot product attention
        att_scores = q @ k.transpose(2, 3) # (bs, n_heads, q_len, kv_len)
        att_scores = att_scores / k.shape[-1]**0.5 # scaled by square root of embedding dimension
        if att_mask is not None:
            att_mask = att_mask.to(dtype=torch.bool)
            att_mask = att_mask.view(bs, 1, q_len, kv_len) # (bs, 1, q_len, kv_len)
            att_scores.masked_fill_(~att_mask, -torch.inf) # (bs, n_heads, q_len, kv_len)

        # Find completely masked rows [-inf, -inf, -inf, -inf] and set the first 
        # element to 0, i.e., [0., -inf, -inf, -inf], resulting in [1, 0, 0, 0] 
        # after softmax. This prevents nan in `att_weights`.
        all_masked_rows = (att_scores == -torch.inf).all(dim=-1, keepdim=True) # (bs, n_heads, q_len, 1)
        other_parts = torch.zeros_like(att_scores[:,:,:,1:], dtype=torch.bool) # (bs, n_heads, q_len, kv_len-1)
        # In `recover_mask`, only the first element of completely masked rows is 
        # True, all others elements are False. 
        recover_mask = torch.cat([all_masked_rows, other_parts], dim=-1) # (bs, n_heads, q_len, kv_len)
        att_scores = att_scores.masked_fill(recover_mask, 0.)

        # Attention weights matrix
        att_weights = torch.softmax(att_scores, dim=-1)
        att_weights = F.dropout(att_weights, p=self.dropout_p, training=self.training) # (bs, n_heads, q_len, kv_len)

        context_vec = att_weights @ v # (bs, n_heads, q_len, head_dim)
        context_vec = context_vec.transpose(1, 2) # (bs, q_len, n_heads, head_dim)
        context_vec = context_vec.contiguous().view(bs, q_len, self.d_out)
        context_vec = self.out_proj(context_vec)

        return context_vec    


## ------- Adaptive LayerNorm ------- ##
class LayerNorm(nn.Module):
    """Perform (adaptive) layer normalization
    Args:
        emb_dim: embedding dimension
        adaptive: whether take adaptive layer normalization
        eps: stability number to prevent division by zero
    """
    def __init__(self, emb_dim: int, adaptive: bool = False, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.adaptive = adaptive

        if not adaptive:
            # In adaptive LayerNorm, scale and shift are provided
            # In standard LayerNorm, scale and shift are learnable parameters
            self.scale = nn.Parameter(torch.ones(emb_dim))
            self.shift = nn.Parameter(torch.zeros(emb_dim))
    
    def forward(self, x: torch.Tensor, 
                scale: torch.Tensor | None = None, 
                shift: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x: (B, len, emb_dim)
            scale: (B, emb_dim), need .unsqueeze(1) to become (B, 1, emb_dim)
            shift: (B, emb_dim), need .unsqueeze(1) to become (B, 1, emb_dim)
        """
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x-mean) / torch.sqrt(var+self.eps)

        if self.adaptive:
            if scale is None or shift is None:
                raise ValueError("Scale and shift must be provided for adaptive LayerNorm.")
            return scale.unsqueeze(1) * norm_x + shift.unsqueeze(1)
        else:
            return self.scale * norm_x + self.shift




## ------- Feed-Forward ------- ##
class FeedForward(nn.Module):
    """A simple feed-forward module
    Args:
        emb_dim: embedding dimension
    """
    def __init__(self, emb_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(emb_dim, 4 * emb_dim),
            nn.GELU(),
            nn.Linear(4 * emb_dim, emb_dim)
        )
    def forward(self, x) -> torch.Tensor:
        return self.layers(x)



## ------- Transformer Block ------- ## 
class TransformerBlock(nn.Module):
    """A transformer block. 
    Args:
        cfg: configuration dictionary
        include_cross_att: True if this transformer block includes a cross-attention module
        include_time_modulate_LN: True if the LayerNorm is adaptive and modulated by time t.
    """
    def __init__(self, cfg: dict, include_cross_att: bool, include_time_modulate_LN: bool):
        super().__init__()
        
        d = cfg['d_model']
        dropout = cfg['dropout']
        n_heads = cfg['attention_heads']
        qkv_bias = cfg['qkv_bias']
        self.include_cross_att = include_cross_att
        self.include_time_modulate_LN = include_time_modulate_LN

        self.self_att = MultiHeadAttention(
            d_in=d, d_out=d, dropout=dropout, n_heads=n_heads, qkv_bias=qkv_bias
        )
        self.ff = FeedForward(d)
        self.dropout = nn.Dropout(p=dropout)

    
        self.norm_self_att = LayerNorm(d, adaptive=include_time_modulate_LN)        # for self-attention
        self.norm_ff = LayerNorm(d, adaptive=include_time_modulate_LN)              # for feed-forward

        if include_cross_att:
            self.cross_att = MultiHeadAttention(
                d_in=d, d_out=d, dropout=dropout, n_heads=n_heads, qkv_bias=qkv_bias
            )
            self.norm_cross_att = LayerNorm(d, adaptive=include_time_modulate_LN)   # for cross-attention

        if include_time_modulate_LN:
            self.adaLN = nn.Sequential(
                nn.GELU(),
                nn.Linear(d, 4 * d + 2 * include_cross_att * d)                     # Each 2d for each LN
            )

    def _2d_att_mask(self, q_mask: torch.Tensor, kv_mask: torch.Tensor) -> torch.Tensor:
        """Create 2d attention mask.
        Note: For any masked token in `q_mask`, it results in a completely masked "row" in the 
        2d mask `qkv_att_mask`, since the "rows" are determined by q_mask. Similarly, for any 
        masked token in `kv_mask`, it results in a completely masked "column" in `qkv_att_mask`.
        Args:
            q_mask: (B, q_len)
            kv_mask: (B, kv_len)
        Returns:
            qkv_att_mask: (B, q_len, kv_len)
        """
        mask_for_q = q_mask.unsqueeze(2)            # (B, q_len, 1)
        mask_for_kv = kv_mask.unsqueeze(1)          # (B, 1, kv_len)
        qkv_att_mask = mask_for_q * mask_for_kv     # (B, q_len, kv_len)
        return qkv_att_mask
    
    def forward(self, 
                q: torch.Tensor,
                q_mask: torch.Tensor | None = None,
                t_embedding: torch.Tensor | None = None,
                kv: torch.Tensor | None = None,
                kv_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            q:                      Query, (B, q_len, d)
            q_mask      (optional): Query attention mask, (B, q_len), 1 for non-paddings and 0 for paddings.
            t_embedding (optional): Time embedding, (B, d). Must be provided if include_time_modulate_LN = True.
            kv          (optional): Key and value, (B, kv_len, d). Must be provided if include_cross_att = True.
            kv_mask     (optional): Key-value attention mask, (B, kv_len).
        Returns:
            New query, (B, q_len, d)
        """
        # ----- Sanity check -----
        if q_mask is None:
            q_mask = torch.ones(q.shape[:2], dtype=torch.long, device=q.device)                 # (B, q_len)
        else:
            assert q_mask.shape == q.shape[:2], 'Shape mismatch between q_mask and q'

        if self.include_cross_att:
            assert kv is not None, 'TransformerBlock missing input: kv'
            if kv_mask is None:
                kv_mask = torch.ones(kv.shape[:2], dtype=torch.long, device=q.device)           # (B, kv_len)
            else:
                assert kv_mask.shape == kv.shape[:2], 'Shape mismatch between kv_mask and kv'


        scale1, shift1, scale2, shift2, scale3, shift3 = None, None, None, None, None, None
        if self.include_time_modulate_LN:
            assert t_embedding is not None, 'TransformerBlock missing input: t_embedding'
            modulation = self.adaLN(t_embedding)                                                # (B, 4d) or (B, 6d)
            if self.include_cross_att:
                scale1, shift1, scale2, shift2, scale3, shift3 = modulation.chunk(6, dim=1)     # Each (B, d)
            else:
                scale1, shift1, scale2, shift2 = modulation.chunk(4, dim=1)                     # Each (B, d)
        
        # ----- Self-Attention Module -----
        qkv_att_mask = self._2d_att_mask(q_mask, q_mask)                                        # (B, q_len, q_len)
        shortcut = q
        q_norm = self.norm_self_att(q, scale=scale1, shift=shift1)
        self_att_output = self.self_att(q_norm, att_mask=qkv_att_mask)
        q = self.dropout(self_att_output) + shortcut


        # ----- Cross-Attention Module -----
        if self.include_cross_att:
            qkv_att_mask = self._2d_att_mask(q_mask, kv_mask)                                   # (B, q_len, kv_len) # type: ignore
            shortcut = q
            q_norm = self.norm_cross_att(q, scale=scale3, shift=shift3)
            cross_att_output = self.cross_att(q_norm, kv_source=kv, att_mask=qkv_att_mask)
            q = self.dropout(cross_att_output) + shortcut


        # ----- Feed-Forward Module -----
        shortcut = q
        q_norm = self.norm_ff(q, scale=scale2, shift=shift2)
        ff_output = self.ff(q_norm)
        q = self.dropout(ff_output) + shortcut
        
        return q



## ------- Residual Block ------- ## 
class ResidualBlock(nn.Module):
    def __init__(self, emb_dim: int, dropout: float):
        super().__init__()
        self.norm = LayerNorm(emb_dim, False)
        self.ff = FeedForward(emb_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x_norm = self.norm(x)
        x_ff = self.ff(x_norm)
        return self.dropout(x_ff) + shortcut


def count_parameters(model:nn.Module) -> None:
    """Count the total and trainable parameters in the model"""
    total = 0
    trainable = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        total += num_params
        if param.requires_grad:
            trainable += num_params

    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    print(f"Frozen parameters: {total - trainable:,}")
    print(f"Proportion of trainable parameters {100 * trainable / total:.2f}%")








class YANEncoder(nn.Module):
    def __init__(self, cfg: dict, shared_tok_emb, shared_pos_emb):
        super().__init__()

        self.shared_tok_emb = shared_tok_emb
        self.shared_pos_emb = shared_pos_emb

        d = cfg['d_model']

        self.dropout = nn.Dropout(cfg['dropout'])
        self.trf_blocks = nn.ModuleList([
            TransformerBlock(cfg, False, False) for _ in range(cfg['encoder_n_trf_blocks'])
        ])
        self.final_norm = LayerNorm(d, adaptive=False)
        self.latent_noise = cfg['latent_noise']

    def forward(self, enc_ids: torch.Tensor, 
                enc_att_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            enc_ids: Encoder ids, (B, Lx). Note: It can be any token ids that need to be encoded.
            enc_att_mask: Encoder attention mask, (B, Lx).
        Returns:
            h_enc: Encoder hidden states, (B, Lx, d).
        """
        B, Lx = enc_ids.shape
        tok_embedding = self.shared_tok_emb(enc_ids)                        # (B, Lx, d)
        position = torch.arange(Lx, device=enc_ids.device).expand(B, -1)    # (B, Lx)
        pos_embedding = self.shared_pos_emb(position)                       # (B, Lx, d)

        x = self.dropout(tok_embedding + pos_embedding)                     # (B, Lx, d)
        for block in self.trf_blocks:
            x = block(x, q_mask = enc_att_mask)                             # (B, Lx, d)
        h_enc = self.final_norm(x)                                          # (B, Lx, d)

        noise_mult = torch.empty(B, 1, 1, device=enc_ids.device).uniform_(0, self.latent_noise) # (B,)
        h_enc = h_enc + noise_mult * torch.randn_like(h_enc)                # (B, Lx, d)

        return h_enc


class YANDecoder(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg

        self.dropout = nn.Dropout(cfg['dropout'])
        self.resi_blocks = nn.Sequential(*[
            ResidualBlock(cfg['d_model'], cfg['dropout']) 
            for _ in range(cfg['decoder_n_resi_blocks'])
        ])
        self.lm_head = nn.Linear(cfg['d_model'], cfg['vocab_size'])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, Ly, d)
        Returns:
            logits: (B, Ly, V)
        """
        x = self.dropout(x)
        x = self.resi_blocks(x)
        logits = self.lm_head(x)
        return logits


class YANAutoEncoder(nn.Module):
    def __init__(self, cfg: dict): 
        super().__init__()
        self.cfg = cfg
        d = cfg['d_model']

        self.shared_tok_emb = nn.Embedding(cfg['vocab_size'], d)
        self.shared_pos_emb = nn.Embedding(cfg['max_length']+1, d)

        self.encoder = YANEncoder(cfg, self.shared_tok_emb, self.shared_pos_emb)
        self.decoder = YANDecoder(cfg)

    def forward(self, enc_ids: torch.Tensor, 
                enc_att_mask: torch.Tensor | None = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            enc_ids: Encoder ids, (B, Lx). (Any token ids to be encoded)
            enc_att_mask: Encoder attention mask, (B, Lx).
        Returns:
            z: (B, Lx, d)
            logits: (B, Lx, V)
        """
        z = self.encoder(enc_ids, enc_att_mask)     # (B, Lx, d)
        logits = self.decoder(z)                    # (B, Lx, V)
        return z, logits


class YANFlow(nn.Module):
    def __init__(self, cfg: dict, shared_pos_emb):
        super().__init__()
        self.cfg = cfg
        d = cfg['d_model']
        self.K = cfg['moe_n_experts']
        self.gate_temperature = cfg.get('moe_gate_temperature', 1.0)

        # ----- Embed position: (B, Ly) -> (B, Ly, d) ----- #
        self.shared_pos_emb = shared_pos_emb

        # ----- Embed t: (B,) -> (B, d) -> (B, d) ----- #
        self.time_emb = nn.Sequential(
            SinusoidalPositionEmbeddings(d),
            nn.Linear(d, d),
            nn.GELU()
        )

        # ----- Main transformer trunk: (B, Ly, d) -> h_trunk (B, Ly, d) ----- #
        self.dropout = nn.Dropout(cfg['dropout'])
        self.trf_blocks = nn.ModuleList([
            TransformerBlock(cfg, i in cfg['cross_att_idx'], True) for i in range(cfg['n_trf_blocks'])
        ])
        self.final_norm = LayerNorm(d, adaptive=False)

        # ----- Learn K expert heads: h_trunk (B, Ly, d) -> K vector fields, each (B, Ly, d) ----- #
        self.expert_out_proj = nn.ModuleList([FeedForward(d) for _ in range(self.K)])

        # ----- Learn token-wise gates: [h_trunk_token, h_enc_pool, t_emb] -> (B, Ly, K) ----- #
        gate_dim = cfg.get('moe_gate_dim_multiplier', 1) * d
        self.gate = nn.Sequential(
            nn.Linear(3 * d, gate_dim),
            nn.GELU(),
            nn.Linear(gate_dim, self.K)
        )

        # ----- Create ODE ----- #
        self.ode = YANODE(self)
        self.solver = EulerSolver(self.ode)


    def _masked_mean_pool(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Mean pooling over L tokens / sentence-level pooling: (B, L, d) -> (B, d).
        Args:
            x: (B, L, d)
            mask: (B, L), 1 for non-paddings and 0 for paddings.
        Returns:
            Mean pooling over L tokens: (B, d)
        """
        if mask is None:
            return x.mean(1)
        m = mask.unsqueeze(-1).to(x.dtype)          # (B, L, 1)
        n_valid = m.sum(1).clamp(min=1.0)           # (B, 1)
        return (x * m).sum(1) / n_valid             # (B, d)
    
    def trunk(
        self, xt: torch.Tensor, t: torch.Tensor, h_enc: torch.Tensor, 
        att_mask: torch.Tensor | None = None, enc_att_mask: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Main transformer trunk. 
        Args:
            xt: Current flow state, (B, Ly, d).
            t: Current time, (B,).
            h_enc: Encoder hidden states, (B, Lx, d).
            att_mask (optional): Attention mask, (B, Ly), 1 for non-paddings and 0 for paddings.
            enc_att_mask (optional): Encoder attention mask, (B, Lx), 1 for non-paddings and 0 for paddings.
        Returns:
            h_trunk: Trunk hidden state, (B, Ly, d).
            t_embedding: Time embedding, (B, d).
        """
        B, Ly, _ = xt.shape
        position = torch.arange(Ly, device=xt.device).expand(B, -1)     # (B, Ly)
        pos_embedding = self.shared_pos_emb(position)                   # (B, Ly, d)
        t_embedding = self.time_emb(t)                                  # (B, d)

        x = xt + pos_embedding                                          # (B, Ly, d)
        x = self.dropout(x)                                             # (B, Ly, d)    
        for block in self.trf_blocks:
            x = block(x, att_mask, t_embedding, h_enc, enc_att_mask)    # (B, Ly, d)
        
        h_trunk = self.final_norm(x)                                    # (B, Ly, d)
        return h_trunk, t_embedding
    
    def compute_gate_probs(
        self, h_trunk: torch.Tensor, t_embedding: torch.Tensor, h_enc: torch.Tensor, 
        att_mask: torch.Tensor | None = None, enc_att_mask: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute gate probabilities
        Args:
            h_trunk: Trunk hidden state, (B, Ly, d).
            t_embedding: Time embedding, (B, d).
            h_enc: Encoder hidden states, (B, Lx, d).
            att_mask (optional): Attention mask, (B, Ly), 1 for non-paddings and 0 for paddings.
            enc_att_mask (optional): Encoder attention mask, (B, Lx), 1 for non-paddings and 0 for paddings.
        Returns:
            gate_pi: Gate probability (after softmax and temperature scaling), (B, Ly, K).
            gate_logits: Gate logits (before softmax and after temperature scaling), (B, Ly, K).
        """
        B, Ly, d = h_trunk.shape 
        h_enc_pool = self._masked_mean_pool(h_enc, enc_att_mask)                # (B, d)
        h_enc_pool = h_enc_pool.unsqueeze(1).expand(B, Ly, d)                   # (B, Ly, d)
        t_tok = t_embedding.unsqueeze(1).expand(B, Ly, d)                       # (B, Ly, d)

        gate_in = torch.cat([h_trunk, h_enc_pool, t_tok], dim=-1)               # (B, Ly, 3d)
        gate_logits = self.gate(gate_in) / self.gate_temperature                # (B, Ly, K)
        if att_mask is not None:
            m = att_mask.unsqueeze(-1).to(gate_logits.dtype)                    # (B, Ly, 1)
            gate_logits = gate_logits * m                                       # (B, Ly, K)

        gate_pi = torch.softmax(gate_logits, dim=-1)                            # (B, Ly, K)
        if att_mask is not None:
            # For pad tokens, set probs to uniform exactly
            m_bool = att_mask.unsqueeze(-1).bool()                              # (B, Ly, 1)
            uniform = torch.full_like(gate_pi, 1.0 / self.K)                    # (B, Ly, K)
            gate_pi = torch.where(m_bool, gate_pi, uniform)

        return gate_pi, gate_logits


    def forward(
        self, xt: torch.Tensor, t: torch.Tensor, h_enc: torch.Tensor, 
        att_mask: torch.Tensor | None = None, enc_att_mask: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            xt: Current flow state, (B, Ly, d).
            t: Current time, (B,).
            h_enc: Encoder hidden states, (B, Lx, d).
            att_mask (optional): Attention mask, (B, Ly), 1 for non-paddings and 0 for paddings.
            enc_att_mask (optional): Encoder attention mask, (B, Lx), 1 for non-paddings and 0 for paddings.
        Returns:
            vec_fields: K vector fields at state xt and time t, (B, K, Ly, d).
            gate_pi: Gate probability (after softmax and temperature scaling), (B, Ly, K).
            gate_logits: Gate logits (before softmax and after temperature scaling), (B, Ly, K).
        """
        h_trunk, t_embedding = self.trunk(xt, t, h_enc, att_mask, enc_att_mask)
        gate_pi, gate_logits = self.compute_gate_probs(
            h_trunk, t_embedding, h_enc, att_mask, enc_att_mask
        )
        vec_fields = torch.stack([
            expert_head(h_trunk) for expert_head in self.expert_out_proj
        ], dim=1)       # (B, K, Ly, d)
        return vec_fields, gate_pi, gate_logits
        

    def sample_expert(
        self, x0: torch.Tensor, h_enc: torch.Tensor,
        att_mask: torch.Tensor | None = None, enc_att_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Assign expert based on initial state x0 at t=0.
        Args:
            x0: Initial state at time t=0, (B, Ly, d).
            h_enc: Encoder hidden states, (B, Lx, d).
            att_mask (optional): Attention mask, (B, Ly), 1 for non-paddings and 0 for paddings.
            enc_att_mask (optional): Encoder attention mask, (B, Lx), 1 for non-paddings and 0 for paddings.
        Returns:
            expert_idx: Expert assignment index, (B, Ly), torch.long.
        """
        B = x0.shape[0]
        t0 = torch.zeros(B, device=x0.device)                                   # (B,)
        _, gate_pi0, _ = self(x0, t0, h_enc, att_mask, enc_att_mask)            # (B, Ly, K)
        expert_idx = torch.distributions.Categorical(probs=gate_pi0).sample()   # (B, Ly)
        if att_mask is not None:
            expert_idx = expert_idx.masked_fill(att_mask == 0, 0)
        return expert_idx

    def forward_expert(
        self, xt: torch.Tensor, t: torch.Tensor, h_enc: torch.Tensor, expert_idx: torch.Tensor,
        att_mask: torch.Tensor | None = None, enc_att_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Compute the expert vector field at (xt, t, h_enc), using expert assignment expert_idx.
        Args:
            xt: Current flow state, (B, Ly, d).
            t: Current time, (B,).
            h_enc: Encoder hidden states, (B, Lx, d).
            expert_idx: Expert assignment index, (B, Ly), torch.long.
        Returns:
            vec_field: (B, Ly, d)
        """
        B, Ly, d = xt.shape
        assert expert_idx.dtype == torch.long
        assert expert_idx.min() >= 0 and expert_idx.max() < self.K
        assert expert_idx.shape == (B, Ly), f"expert_idx must be (B,Ly), got {expert_idx.shape}"

        vec_fields, _, _ = self(xt, t, h_enc, att_mask, enc_att_mask)               # (B, K, Ly, d)
        vec_fields = vec_fields.permute(0, 2, 1, 3)                                 # (B, Ly, K, d)
        idx = expert_idx.unsqueeze(-1).unsqueeze(-1)                                # (B, Ly, 1, 1)
        idx = idx.expand(B, Ly, 1, d)                                               # (B, Ly, 1, d)
        vec_field = vec_fields.gather(dim=2, index=idx).squeeze(2)                  # (B, Ly, d)
        return vec_field



    def solve_ode(
        self, x0: torch.Tensor, h_enc: torch.Tensor,
        att_mask: torch.Tensor | None = None, enc_att_mask: torch.Tensor | None = None, 
        expert_idx: torch.Tensor | None = None,
        n_timesteps: int | None = None, trajectory: bool = False
    ) -> torch.Tensor:
        """Solve for the final state x1 with each token following the corresponding expert vector field.
        Args:
            x0: Initial state at time ts[0], (B, Ly, d).
            h_enc: Encoder hidden states, (B, Lx, d).
            att_mask (optional): Attention mask, (B, Ly), 1 for non-paddings and 0 for paddings.
            enc_att_mask (optional): Encoder attention mask, (B, Lx), 1 for non-paddings and 0 for paddings.
            expert_idx (optional): Fixed expert id, (B, Ly). If None, then sample it at t=0 using gate_pi(x0, 0).
            n_timesteps (optional): The number of Euler steps. If provided, override cfg['ode_steps'].
            trajectory (optional): Whether to return the entire trajectory.
        Returns:
            If trajectory=False, return the final state x1, (B, Ly, d). 
            Else, return the trajectory, (n_timesteps, B, Ly, d).
        """
        if expert_idx is not None:
            assert expert_idx.dim() == 2 and expert_idx.shape[0] == x0.shape[0] \
                and expert_idx.shape[1] == x0.shape[1]

        if n_timesteps is None:
            n_timesteps = self.cfg['ode_steps']
        ts = torch.linspace(0, 1, steps=n_timesteps, device=x0.device)              # (n_timesteps,)
        ts = ts.unsqueeze(-1).expand(-1, x0.shape[0])                               # (n_timesteps, B)

        if expert_idx is None:
            expert_idx = self.sample_expert(x0, h_enc, att_mask, enc_att_mask)      # (B, Ly)

        x1 = self.solver.solve(x0, ts, trajectory, h_enc=h_enc, att_mask=att_mask, 
                               enc_att_mask=enc_att_mask, expert_idx=expert_idx)    # (B, Ly, d)
        return x1





class YAN(nn.Module):
    def __init__(self, cfg: dict, yan_ae: YANAutoEncoder | None = None):
        super().__init__()
        self.cfg = cfg
        self.beta_a = cfg.get('beta_a', 1.0)
        self.beta_b = cfg.get('beta_b', 1.0)

        if yan_ae is None:
            yan_ae = YANAutoEncoder(cfg)
        
        self.shared_tok_emb = yan_ae.shared_tok_emb
        self.shared_pos_emb = yan_ae.shared_pos_emb

        self.encoder = yan_ae.encoder
        self.flow = YANFlow(cfg, self.shared_pos_emb)
        self.decoder = yan_ae.decoder

        freeze_enc = cfg.get('freeze_enc', True)
        for param in self.encoder.parameters():
            param.requires_grad = not freeze_enc
        self.freeze_enc = freeze_enc

        freeze_flow = cfg.get('freeze_flow', True)
        for param in self.flow.parameters():
            param.requires_grad = not freeze_flow

        freeze_dec = cfg.get('freeze_dec', True)
        for param in self.decoder.parameters():
            param.requires_grad = not freeze_dec

        freeze_tok_emb = cfg.get('freeze_tok_emb', True)
        for param in self.shared_tok_emb.parameters():
            param.requires_grad = not freeze_tok_emb

        freeze_pos_emb = cfg.get('freeze_pos_emb', True)
        for param in self.shared_pos_emb.parameters():
            param.requires_grad = not freeze_pos_emb

        self.init_noise = cfg['init_noise']

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(mu)
        return mu + std * eps


    def align_enc_to_tgt_length(
        self, enc_ids: torch.Tensor, tgt_length: int,
        enc_att_mask: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor | None]:
        """Adjust the sequence length of enc_ids to the target length.
        Args:
            enc_ids: Encoder ids, (B, Lx).
            tgt_length: Target length.
            enc_att_mask: Encoder attention mask, (B, Lx), 1 for non-paddings and 0 for paddings.
        Returns:
            adjusted_enc_ids: (B, Ly)
            adjusted_enc_att_mask: (B, Ly)
        """
        Lx = enc_ids.shape[1]
        Ly = tgt_length
        pad_token_id = self.cfg['pad_token_id']
        eos_token_id = self.cfg['eos_token_id']

        if Lx == Ly:
            return enc_ids, enc_att_mask
        
        adjusted_enc_att_mask = None
        if Lx < Ly:
            diff = Ly - Lx 
            adjusted_enc_ids = F.pad(enc_ids, (0, diff), value=pad_token_id)
            if enc_att_mask is not None:
                adjusted_enc_att_mask = F.pad(enc_att_mask, (0, diff), value=0)
        
        else:
            adjusted_enc_ids = enc_ids[:, :Ly].clone()
            if enc_att_mask is not None:
                adjusted_enc_att_mask = enc_att_mask[:, :Ly].clone()

            last_token = adjusted_enc_ids[:, -1]  # (B,)
            mask_to_replace = (last_token != pad_token_id) & (last_token != eos_token_id)
            if mask_to_replace.any():
                adjusted_enc_ids[mask_to_replace, -1] = eos_token_id
                if adjusted_enc_att_mask is not None:
                    adjusted_enc_att_mask[mask_to_replace, -1] = 1
        return adjusted_enc_ids, adjusted_enc_att_mask


    def forward(self, 
                enc_ids: torch.Tensor,
                tgt_ids: torch.Tensor, 
                att_mask: torch.Tensor | None = None,
                enc_att_mask: torch.Tensor | None = None) -> Tuple[torch.Tensor, ...]:
        """
        Args:
            enc_ids: Encoder ids, (B, Lx).
            tgt_ids: Target ids, (B, Ly).
            att_mask: Target attention mask, (B, Ly), 1 for non-paddings and 0 for paddings.
            enc_att_mask: Encoder attention mask, (B, Lx), 1 for non-paddings and 0 for paddings.
        Returns:
            vec_field_target: (B, Ly, d)
            vec_field_experts: K vector fields at state xt and time t, (B, K, Ly, d).
            gate_pi: Gate probability (after softmax and temperature scaling), (B, Ly, K).
            gate_logits: Gate logits (before softmax and after temperature scaling), (B, Ly, K).
            x1: (B, Ly, d)
            x1_hat: (B, Ly, d)
            logits: (B, Ly, V)
            expert_idx: Expert assignment index, (B, Ly), torch.long.
        """
        B, Ly = tgt_ids.shape
        device = enc_ids.device

        # ----- Encode the source ----- #
        if self.freeze_enc:
            with torch.no_grad():
                h_enc = self.encoder(enc_ids, enc_att_mask)                     # (B, Lx, d)
        else:
            h_enc = self.encoder(enc_ids, enc_att_mask)                         # (B, Lx, d)

        # ----- Generate x0 ----- #
        x0 = self.init_noise * torch.randn(B, Ly, h_enc.shape[-1], device=device)

        # ----- Generate target x1 ----- #
        if self.freeze_enc:
            with torch.no_grad():
                x1 = self.encoder(tgt_ids, att_mask)                            # (B, Ly, d)
        else:
            x1 = self.encoder(tgt_ids, att_mask)


        # ----- Learn the vector field ----- #
        path = LinearConditionalProbabilityPath(x0, x1)
        vec_field_target = path.conditional_vector_field()                      # (B, Ly, d)
        beta_a = torch.tensor(self.beta_a, device=device)
        beta_b = torch.tensor(self.beta_b, device=device)
        t = torch.distributions.Beta(beta_a, beta_b).sample((B,))               # (B,)
        xt = path.sample_conditional_path(t)                                    # (B, Ly, d)

        vec_field_experts, gate_pi, gate_logits = self.flow(xt, t, h_enc, att_mask, enc_att_mask)


        # ----- Solve for x1 ----- #
        expert_idx = self.flow.sample_expert(x0, h_enc, att_mask, enc_att_mask)
        x1_hat = self.flow.solve_ode(x0, h_enc, att_mask, enc_att_mask, expert_idx)

        # ----- Get logits ----- #
        logits = self.decoder(x1_hat)                                       # (B, Ly, V)

        return vec_field_target, vec_field_experts, gate_pi, gate_logits, x1, x1_hat, logits, expert_idx



    @torch.no_grad()
    def generate(self, enc_ids: torch.Tensor, max_tgt_len: int, 
                 enc_att_mask: torch.Tensor | None = None, 
                 n_timesteps: int | None = None, 
                 trajectory: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            enc_ids: Encoder ids, (B, Lx).
            max_tgt_len: Maximum target length to generate. The effective tokens will end with <EOS>.
            enc_att_mask (optional): Encoder attention mask, (B, Lx), 1=non-paddings and 0=paddings.
            n_timesteps (optional): The number of Euler steps. If provided, override cfg['ode_steps'].
            trajectory (optional): Whether to return the entire trajectory.
        Returns:
            x1_hat: Final state at t = 1.
                - (B, max_tgt_len, d) if trajectory = False.
                - (n_timesteps, B, max_tgt_len, d) if trajectory = True.
            logits: Decoded logits. 
                - (B, max_tgt_len, V) if trajectory = False.
                - (n_timesteps, B, max_tgt_len, V) if trajectory = True.
        """
        B = enc_ids.shape[0]
        device = enc_ids.device
        Ly = max_tgt_len
        if n_timesteps is None:
            n_timesteps = self.cfg['ode_steps']
        att_mask = torch.ones((B, Ly), device=device)

        # ----- Encode the source ----- #
        h_enc = self.encoder(enc_ids, enc_att_mask)     # (B, Lx, d)

        # ----- Generate x0 ----- #
        x0 = self.init_noise * torch.randn(B, Ly, h_enc.shape[-1], device=device)

        # ----- Solve for x1 ----- #
        expert_idx = self.flow.sample_expert(x0, h_enc, att_mask, enc_att_mask)
        x1_hat = self.flow.solve_ode(x0, h_enc, enc_att_mask=enc_att_mask, expert_idx=expert_idx, 
                                     n_timesteps=n_timesteps, trajectory=trajectory)

        # ----- Get logits ----- #
        logits = self.decoder(x1_hat)
        return x1_hat, logits


    @torch.no_grad()
    def _generate_from_x0(self, enc_ids: torch.Tensor, x0: torch.Tensor, max_tgt_len: int,
                          enc_att_mask: torch.Tensor | None = None, n_timesteps: int | None = None, 
                          trajectory: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """Deterministic generation given x0.
        Args:
            enc_ids: (B, Lx).
            x0: (B, Ly, d).
        Returns:
            x1_hat, logits: same format as generate().
        """
        B = enc_ids.shape[0]
        device = enc_ids.device
        Ly = max_tgt_len
        if n_timesteps is None:
            n_timesteps = self.cfg['ode_steps']
        att_mask = torch.ones((B, Ly), device=device)
        
        # ----- Encode the source ----- #
        h_enc = self.encoder(enc_ids, enc_att_mask)
        d = h_enc.shape[-1]
        assert x0.shape == (B, Ly, d), f"x0 shape {x0.shape} != {(B, Ly, d)}"
        x0 = x0.to(device)

        # ----- Solve for x1 ----- #
        expert_idx = self.flow.sample_expert(x0, h_enc, att_mask, enc_att_mask)
        x1_hat = self.flow.solve_ode(x0, h_enc, enc_att_mask=enc_att_mask, expert_idx=expert_idx, 
                                     n_timesteps=n_timesteps, trajectory=trajectory)
        
        # ----- Get logits ----- #
        logits = self.decoder(x1_hat)
        return x1_hat, logits

    @torch.no_grad()
    def generate_k(self, enc_ids: torch.Tensor, max_tgt_len: int, K: int, 
                   enc_att_mask: torch.Tensor | None = None, n_timesteps: int | None = None, 
                   trajectory: bool = False, 
                   base_seed: int = 1234) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate K candidates by sampling different x0 with a local torch.Generator.
        Returns:
            x1_hat_k:
                - (K, B, Ly, d) if trajectory=False
                - (K, n_timesteps, B, Ly, d) if trajectory=True
            logits_k:
                - (K, B, Ly, V) if trajectory=False
                - (K, n_timesteps, B, Ly, V) if trajectory=True
            x0_k: (K, B, Ly, d)
        """
        self.eval()

        B = enc_ids.shape[0]
        device = enc_ids.device
        Ly = max_tgt_len

        h_enc = self.encoder(enc_ids, enc_att_mask)
        d = h_enc.shape[-1]

        g = torch.Generator(device=device)
        g.manual_seed(base_seed)

        x0_k = self.init_noise * torch.randn((K, B, Ly, d), device=device, generator=g)
        x1_list, logit_list = [], []
        for k in range(K):
            x1_hat, logits = self._generate_from_x0(
                enc_ids=enc_ids, x0=x0_k[k], max_tgt_len=max_tgt_len, enc_att_mask=enc_att_mask,
                n_timesteps=n_timesteps, trajectory=trajectory)
            x1_list.append(x1_hat)
            logit_list.append(logits)

        x1_hat_k = torch.stack(x1_list, dim=0)
        logits_k = torch.stack(logit_list, dim=0)
        return x1_hat_k, logits_k, x0_k