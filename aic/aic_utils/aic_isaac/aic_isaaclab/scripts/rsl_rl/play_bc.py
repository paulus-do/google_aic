import os

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

import time
import json
import torch
import numpy as np
import cv2
import draccus
from pathlib import Path
from typing import Callable, Dict, Any, List
# LeRobot & Safetensors
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.act.configuration_act import ACTConfig
from safetensors.torch import load_file
from huggingface_hub import snapshot_download

# -------------------------------------------------------------------------
    # 1. Configuration & Weights Loading
    # -------------------------------------------------------------------------
repo_id = "grkw/aic_act_policy"
# Path to your checkpoint folder
policy_path = Path(
    snapshot_download(
        repo_id=repo_id,
        allow_patterns=["config.json", "model.safetensors", "*.safetensors"],
    )
)

# Load Config Manually (Fixes 'Draccus' error by removing unknown 'type' field)
with open(policy_path / "config.json", "r") as f:
    config_dict = json.load(f)
    if "type" in config_dict:
        del config_dict["type"]

config = draccus.decode(ACTConfig, config_dict)

# Load Policy Architecture & Weights
self.policy = ACTPolicy(config)
model_weights_path = policy_path / "model.safetensors"
self.policy.load_state_dict(load_file(model_weights_path))
self.policy.eval()
self.policy.to(self.device)

self.get_logger().info(f"ACT Policy loaded on {self.device} from {policy_path}")

# -------------------------------------------------------------------------
# 2. Normalization Stats Loading
# -------------------------------------------------------------------------
stats_path = (
    policy_path / "policy_preprocessor_step_3_normalizer_processor.safetensors"
)
stats = load_file(stats_path)

# Helper to extract and shape stats for broadcasting
def get_stat(key, shape):
    return stats[key].to(self.device).view(*shape)

# Image Stats (1, 3, 1, 1) for broadcasting against (Batch, Channel, Height, Width)
self.img_stats = {
    "left": {
        "mean": get_stat("observation.images.left_camera.mean", (1, 3, 1, 1)),
        "std": get_stat("observation.images.left_camera.std", (1, 3, 1, 1)),
    },
    "center": {
        "mean": get_stat("observation.images.center_camera.mean", (1, 3, 1, 1)),
        "std": get_stat("observation.images.center_camera.std", (1, 3, 1, 1)),
    },
    "right": {
        "mean": get_stat("observation.images.right_camera.mean", (1, 3, 1, 1)),
        "std": get_stat("observation.images.right_camera.std", (1, 3, 1, 1)),
    },
}
print(f"Image stats: {self.img_stats}")

# Robot State Stats (1, 26)
self.state_mean = get_stat("observation.state.mean", (1, -1))
self.state_std = get_stat("observation.state.std", (1, -1))
print(f"Robot state mean: {self.state_mean}")
print(f"Robot state std: {self.state_std}")

# Action Stats (1, 7) - Used for Un-normalization
self.action_mean = get_stat("action.mean", (1, -1))
self.action_std = get_stat("action.std", (1, -1))
print(f"Action mean: {self.action_mean}")
print(f"Action std: {self.action_std}")

# Config
self.image_scaling = 0.25  # Must match AICRobotAICControllerConfig

    self.get_logger().info("Normalization statistics loaded successfully.")