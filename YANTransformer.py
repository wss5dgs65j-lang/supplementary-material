# trainer/YANTransformer.py

import torch
import torch.nn as nn
import math
import torch.nn.functional as F

try:
    from mamba_ssm.modules.mamba_simple import Mamba
except Exception:
    Mamba = None


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


## ------- Mamba Block ------- ##
class BiMamba(nn.Module):
    """
    Bidirectional Mamba token mixer.
    Input/Output: (B, L, D)
    """
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        if Mamba is None:
            raise ImportError("mamba-ssm is not installed. Please `pip install mamba-ssm`.")

        self.fwd = Mamba(d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.bwd = Mamba(d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.proj = nn.Linear(2 * d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)
        y_fwd = self.fwd(x)                     # (B, L, D)

        # backward pass: reverse sequence, apply Mamba, reverse back
        x_rev = torch.flip(x, dims=[1])
        y_bwd = self.bwd(x_rev)
        y_bwd = torch.flip(y_bwd, dims=[1])     # (B, L, D)

        y = torch.cat([y_fwd, y_bwd], dim=-1)   # (B, L, 2D)
        return self.proj(y)                     # (B, L, D)


class MambaFeedForward(nn.Module):
    """
    Feed-forward module used only in Mamba blocks.
    Supports configurable width multiplier to control parameter count.
    """
    def __init__(self, cfg: dict):
        super().__init__()
        d = cfg["d_model"]
        mult = cfg.get("mamba_ff_mult", 4)
        ff_dropout = cfg['dropout']
        self.layers = nn.Sequential(
            nn.Linear(d, int(mult * d)),
            nn.GELU(),
            nn.Dropout(ff_dropout),
            nn.Linear(int(mult * d), d)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class MambaTransformerBlock(nn.Module):
    """
    Drop-in-ish replacement for TransformerBlock:
    - self-attention replaced by BiMamba (bidirectional token mixer)
    - cross-attention stays as MultiHeadAttention (optional)
    - FFN uses MambaFeedForward
    - supports adaptive LayerNorm modulated by time embedding (adaLN), same as your TransformerBlock
    """
    def __init__(self, cfg: dict, include_cross_att: bool, include_time_modulate_LN: bool):
        super().__init__()

        d = cfg['d_model']
        dropout = cfg['dropout']
        n_heads = cfg['attention_heads']
        qkv_bias = cfg['qkv_bias']

        self.include_cross_att = include_cross_att
        self.include_time_modulate_LN = include_time_modulate_LN

        # --- Self mixer (Mamba) ---
        self.self_mixer = BiMamba(
            d_model=d,
            d_state=cfg.get("mamba_d_state", 16),
            d_conv=cfg.get("mamba_d_conv", 4),
            expand=cfg.get("mamba_expand", 2),
        )

        # --- FFN ---
        self.ff = MambaFeedForward(cfg)
        self.dropout = nn.Dropout(p=dropout)

        # --- Norms ---
        self.norm_self = LayerNorm(d, adaptive=include_time_modulate_LN)
        self.norm_ff = LayerNorm(d, adaptive=include_time_modulate_LN)

        # --- Optional cross-attention (keep your original attention for conditioning) ---
        if include_cross_att:
            self.cross_att = MultiHeadAttention(
                d_in=d, d_out=d, dropout=dropout, n_heads=n_heads, qkv_bias=qkv_bias
            )
            self.norm_cross = LayerNorm(d, adaptive=include_time_modulate_LN)

        # --- Optional time-modulated LN parameters ---
        if include_time_modulate_LN:
            self.adaLN = nn.Sequential(
                nn.GELU(),
                nn.Linear(d, 4 * d + 2 * include_cross_att * d)
            )

    @staticmethod
    def _2d_att_mask(q_mask: torch.Tensor, kv_mask: torch.Tensor) -> torch.Tensor:
        # same as your TransformerBlock
        return q_mask.unsqueeze(2) * kv_mask.unsqueeze(1)

    @staticmethod
    def _apply_token_mask(x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        # x: (B, L, D), mask: (B, L) with 1 valid, 0 pad
        if mask is None:
            return x
        m = mask.unsqueeze(-1).to(dtype=x.dtype)
        return x * m

    def forward(
        self,
        q: torch.Tensor,
        q_mask: torch.Tensor | None = None,
        t_embedding: torch.Tensor | None = None,
        kv: torch.Tensor | None = None,
        kv_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # ----- Sanity checks (keep same behavior) -----
        if q_mask is None:
            q_mask = torch.ones(q.shape[:2], dtype=torch.long, device=q.device)
        else:
            assert q_mask.shape == q.shape[:2], 'Shape mismatch between q_mask and q'

        if self.include_cross_att:
            assert kv is not None, 'MambaTransformerBlock missing input: kv'
            if kv_mask is None:
                kv_mask = torch.ones(kv.shape[:2], dtype=torch.long, device=q.device)
            else:
                assert kv_mask.shape == kv.shape[:2], 'Shape mismatch between kv_mask and kv'

        # ----- Time modulation params -----
        scale1 = shift1 = scale2 = shift2 = scale3 = shift3 = None
        if self.include_time_modulate_LN:
            assert t_embedding is not None, 'MambaTransformerBlock missing input: t_embedding'
            modulation = self.adaLN(t_embedding)
            if self.include_cross_att:
                scale1, shift1, scale2, shift2, scale3, shift3 = modulation.chunk(6, dim=1)
            else:
                scale1, shift1, scale2, shift2 = modulation.chunk(4, dim=1)

        # ===== Self "attention" (Mamba token mixer) =====
        shortcut = q
        q_norm = self.norm_self(q, scale=scale1, shift=shift1)
        y = self.self_mixer(q_norm)                       # (B, L, D)
        y = self._apply_token_mask(y, q_mask)             # keep pads stable
        q = self.dropout(y) + shortcut

        # ===== Cross-attention (unchanged) =====
        if self.include_cross_att:
            att2d = self._2d_att_mask(q_mask, kv_mask)    # (B, q_len, kv_len)
            shortcut = q
            q_norm = self.norm_cross(q, scale=scale3, shift=shift3)
            y = self.cross_att(q_norm, kv_source=kv, att_mask=att2d)
            y = self._apply_token_mask(y, q_mask)
            q = self.dropout(y) + shortcut

        # ===== FFN (unchanged) =====
        shortcut = q
        q_norm = self.norm_ff(q, scale=scale2, shift=shift2)
        y = self.ff(q_norm)
        y = self._apply_token_mask(y, q_mask)
        q = self.dropout(y) + shortcut

        return q


