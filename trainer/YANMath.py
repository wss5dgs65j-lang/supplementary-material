# trainer/YANMath.py

import torch
import torch.nn as nn
from abc import ABC, abstractmethod


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

