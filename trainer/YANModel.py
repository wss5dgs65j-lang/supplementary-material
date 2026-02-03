# trainer/YANModel.py

import torch
import torch.nn as nn
from typing import Tuple
import torch.nn.functional as F
from trainer.YANTransformer import *
from trainer.YANMath import *

class YANEncoder(nn.Module):
    def __init__(self, cfg: dict, shared_tok_emb, shared_pos_emb):
        super().__init__()

        self.shared_tok_emb = shared_tok_emb
        self.shared_pos_emb = shared_pos_emb

        d = cfg['d_model']
        self.dropout = nn.Dropout(cfg['dropout'])

        self.encoder_use_mamba = cfg.get("encoder_use_mamba", False)
        if self.encoder_use_mamba:
            self.mamba_blocks = nn.ModuleList([
                MambaTransformerBlock(cfg, False, False) for _ in range(cfg['encoder_n_trf_blocks'])
            ]) 
            self.trf_blocks = None
        else:  
            self.trf_blocks = nn.ModuleList([
                TransformerBlock(cfg, False, False) for _ in range(cfg['encoder_n_trf_blocks'])
            ])
            self.mamba_blocks = None

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
        blocks = self.mamba_blocks if self.encoder_use_mamba else self.trf_blocks
        for block in blocks:
            x = block(x, q_mask=enc_att_mask)                               # (B, Lx, d)
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

        # ----- Main transformer / mamba trunk: (B, Ly, d) -> h_trunk (B, Ly, d) ----- #
        self.dropout = nn.Dropout(cfg['dropout'])
        self.flow_use_mamba = cfg.get('flow_use_mamba', False)
        if self.flow_use_mamba:
            self.mamba_blocks = nn.ModuleList([
                MambaTransformerBlock(cfg, i in cfg['cross_att_idx'], True) for i in range(cfg['n_trf_blocks'])
            ])
            self.trf_blocks = None 
        else:
            self.trf_blocks = nn.ModuleList([
                TransformerBlock(cfg, i in cfg['cross_att_idx'], True) for i in range(cfg['n_trf_blocks'])
            ])
            self.mamba_blocks = None
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
        blocks = self.mamba_blocks if self.flow_use_mamba else self.trf_blocks
        for block in blocks:
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



