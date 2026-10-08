#!/usr/bin/env python3
"""Minimal inference helper for microtest v4 viewers."""
from __future__ import annotations

import torch
from tensordict import TensorDict
from rsl_rl.modules import ActorCritic


def build_policy(ckpt_path: str) -> ActorCritic:
    sample = TensorDict(
        {"policy": torch.zeros(1, 51), "critic": torch.zeros(1, 55)}, batch_size=[1]
    )
    policy = ActorCritic(
        sample,
        {"policy": ["policy"], "critic": ["critic"]},
        num_actions=14,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        init_noise_std=0.5,
        noise_std_type="scalar",
    )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    policy.load_state_dict(ckpt["model_state_dict"])
    policy.eval()
    print(f"loaded {ckpt_path} (iter {ckpt.get('iter', '?')})")
    return policy
