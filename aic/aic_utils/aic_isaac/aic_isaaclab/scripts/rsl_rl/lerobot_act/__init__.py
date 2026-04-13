"""
Standalone ACT policy extracted from huggingface/lerobot.
For inference only — no training pipeline, no environment dependencies.

Usage:
    from lerobot_act import ACTPolicy, ACTConfig

    policy = ACTPolicy.from_pretrained("path/to/checkpoint")
    action = policy.select_action(observation_batch)
"""
from .configuration_act import ACTConfig
from .modeling_act import ACTPolicy

__all__ = ["ACTPolicy", "ACTConfig"]
