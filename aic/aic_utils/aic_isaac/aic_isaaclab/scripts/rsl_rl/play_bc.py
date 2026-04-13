# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a BC (ACT) policy checkpoint inside Isaac Lab."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# ---------------------------------------------------------------------------
# Argument parsing — we keep play.py's args but drop RL-specific ones
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Play a BC (ACT) policy in Isaac Lab.")
parser.add_argument(
    "--video", action="store_true", default=False, help="Record videos during play."
)
parser.add_argument(
    "--video_length", type=int, default=200, help="Length of recorded video (steps)."
)
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
parser.add_argument(
    "--num_envs", type=int, default=1, help="Number of environments to simulate."
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--seed", type=int, default=None, help="Seed used for the environment."
)
parser.add_argument(
    "--real-time",
    action="store_true",
    default=False,
    help="Run in real-time, if possible.",
)
parser.add_argument(
    "--policy_freq",
    type=float,
    default=4.0,
    help="ACT policy inference frequency in Hz (default: 4 Hz, matching RunACT).",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras — we NEED them for the BC policy's image observations
args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows — imports that depend on Isaac Sim being launched."""

import os
import time
import json

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

import gymnasium as gym
import torch
import numpy as np
import cv2
import draccus
from pathlib import Path

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

import aic_task.tasks  # noqa: F401

# ---------------------------------------------------------------------------
# ACT Policy imports — local lerobot to avoid package-dependency conflicts
# ---------------------------------------------------------------------------
from safetensors.torch import load_file
from huggingface_hub import snapshot_download

sys.path.insert(0, str(Path(__file__).parent))
from lerobot_act import ACTConfig, ACTPolicy


# ===========================================================================
# Helper: load ACT policy + normalization stats (mirrors RunACT.__init__)
# ===========================================================================
def load_act_policy(device: torch.device):
    """Download and instantiate the ACT policy with normalization statistics.

    Returns:
        policy:      ACTPolicy in eval mode on `device`
        img_stats:   dict of {camera_name: {mean, std}} tensors shaped (1,3,1,1)
        state_mean:  (1, 26) tensor
        state_std:   (1, 26) tensor
        action_mean: (1, 7) tensor
        action_std:  (1, 7) tensor
    """
    repo_id = "grkw/aic_act_policy"

    policy_path = Path(
        snapshot_download(
            repo_id=repo_id,
            allow_patterns=["config.json", "model.safetensors", "*.safetensors"],
        )
    )

    # --- Config ---
    with open(policy_path / "config.json", "r") as f:
        config_dict = json.load(f)
    # Strip fields that draccus / ACTConfig don't recognise
    invalid_fields = {"type", "use_peft", "push_to_hub", "repo_id", "private", "tags", "license"}
    for key in invalid_fields:
        config_dict.pop(key, None)

    config = draccus.decode(ACTConfig, config_dict)

    # --- Weights ---
    policy = ACTPolicy(config)
    policy.load_state_dict(load_file(policy_path / "model.safetensors"))
    policy.eval()
    policy.to(device)

    # --- Normalization stats ---
    stats_path = policy_path / "policy_preprocessor_step_3_normalizer_processor.safetensors"
    stats = load_file(stats_path)

    def get_stat(key, shape):
        return stats[key].to(device).view(*shape)

    img_stats = {}
    for cam in ("left", "center", "right"):
        img_stats[cam] = {
            "mean": get_stat(f"observation.images.{cam}_camera.mean", (1, 3, 1, 1)),
            "std":  get_stat(f"observation.images.{cam}_camera.std",  (1, 3, 1, 1)),
        }

    state_mean  = get_stat("observation.state.mean", (1, -1))
    state_std   = get_stat("observation.state.std",  (1, -1))
    action_mean = get_stat("action.mean", (1, -1))
    action_std  = get_stat("action.std",  (1, -1))

    print(f"[ACT] Policy loaded on {device} from {policy_path}")
    print(f"[ACT] State dim: {state_mean.shape[-1]}, Action dim: {action_mean.shape[-1]}")

    return policy, img_stats, state_mean, state_std, action_mean, action_std


# ===========================================================================
# Helper: convert a raw image tensor/array to a normalised ACT input
# ===========================================================================
def img_to_tensor(
    img_np: np.ndarray,
    device: torch.device,
    scale: float,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    """Resize, permute, normalise a (H, W, 3) uint8 numpy image for ACT.

    This mirrors RunACT._img_to_tensor but takes a numpy array directly
    instead of a ROS Image message, since Isaac Lab gives us arrays/tensors.
    """
    # 1. Resize
    if scale != 1.0:
        img_np = cv2.resize(img_np, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    # 2. HWC -> CHW, float, /255, add batch dim, normalise
    tensor = (
        torch.from_numpy(img_np)
        .permute(2, 0, 1)
        .float()
        .div(255.0)
        .unsqueeze(0)
        .to(device)
    )
    return (tensor - mean) / std


# ===========================================================================
# Helper: extract ACT-compatible observations from the Isaac Lab environment
# ===========================================================================
def extract_act_observations(
    env,
    device: torch.device,
    img_stats: dict,
    state_mean: torch.Tensor,
    state_std: torch.Tensor,
    image_scaling: float = 0.25,
    env_idx: int = 0,
) -> dict:
    """Pull raw sensor data from the Isaac Lab env and package it for ACT.

    -----------------------------------------------------------------------
    IMPORTANT — you MUST adapt the sensor / articulation accessor names
    below to match your specific Isaac Lab task configuration.  The names
    used here (e.g. "left_camera", "center_camera", "right_camera" for
    tiled cameras, "robot" for the articulation) are *placeholders* that
    follow common AIC conventions.  Check your task's scene configuration
    to find the correct attribute names.
    -----------------------------------------------------------------------

    The function accesses the *unwrapped* environment to reach Isaac Lab's
    scene-level sensor objects, because the standard `env.get_observations()`
    returns a flat RL tensor that does not contain raw camera images.

    Returns:
        obs_dict  — dict with keys matching ACTPolicy.select_action() input:
            "observation.images.left_camera"   : (1, 3, H', W') float tensor
            "observation.images.center_camera"  : (1, 3, H', W') float tensor
            "observation.images.right_camera"   : (1, 3, H', W') float tensor
            "observation.state"                 : (1, 26) float tensor
    """
    scene = env.unwrapped.scene

    # ------------------------------------------------------------------
    # A) CAMERA IMAGES
    # ------------------------------------------------------------------
    # Isaac Lab cameras (TiledCamera / Camera) store their latest render
    # in  camera_sensor.data.output["rgb"]  →  shape (num_envs, H, W, 4)
    # The 4th channel is alpha; we take only the first 3 (RGB).
    #
    # >>> Adjust the attribute names to match YOUR scene config <<<
    # ------------------------------------------------------------------
    camera_map = {
        "left":   scene["left_camera"],    # or e.g. scene.sensors["left_camera"]
        "center": scene["center_camera"],
        "right":  scene["right_camera"],
    }

    obs_dict = {}
    for cam_name, cam_sensor in camera_map.items():
        # (num_envs, H, W, 4) on GPU → take env_idx, drop alpha, move to CPU numpy
        rgba = cam_sensor.data.output["rgb"]          # (N, H, W, 4) uint8 or float
        img_gpu = rgba[env_idx, :, :, :3]             # (H, W, 3)

        # If the simulator returns float [0,1], convert to uint8
        if img_gpu.dtype == torch.float32:
            img_np = (img_gpu.cpu().numpy() * 255).astype(np.uint8)
        else:
            img_np = img_gpu.cpu().numpy().astype(np.uint8)

        obs_dict[f"observation.images.{cam_name}_camera"] = img_to_tensor(
            img_np, device, image_scaling,
            img_stats[cam_name]["mean"],
            img_stats[cam_name]["std"],
        )

    # ------------------------------------------------------------------
    # B) ROBOT STATE  (26-dim, must match RunACT training order exactly)
    # ------------------------------------------------------------------
    # The 26 dimensions in RunACT.prepare_observations are:
    #   TCP position          (3)  — from controller_state.tcp_pose.position
    #   TCP orientation quat  (4)  — from controller_state.tcp_pose.orientation
    #   TCP linear velocity   (3)  — from controller_state.tcp_velocity.linear
    #   TCP angular velocity  (3)  — from controller_state.tcp_velocity.angular
    #   TCP error             (6)  — from controller_state.tcp_error
    #   Joint positions       (7)  — from joint_states.position[:7]
    #
    # How to get these from Isaac Lab depends on your env/scene setup.
    # Below is a TEMPLATE using common Isaac Lab APIs.  You will almost
    # certainly need to adjust attribute names.
    # ------------------------------------------------------------------
    robot = scene["robot"]  # Articulation asset

    # --- One-time discovery: print all body names so you can verify the EE index ---
    if not hasattr(extract_act_observations, "_ee_idx_resolved"):
        body_names = robot.data.body_names
        print(f"[ACT DEBUG] Robot body names: {body_names}")
        print(f"[ACT DEBUG] Robot joint names: {robot.data.joint_names}")
        print(f"[ACT DEBUG] num_bodies={robot.data.body_pos_w.shape[1]}, "
              f"num_joints={robot.data.joint_pos.shape[1]}")

        # Try to find the EE body by common name patterns.
        ee_candidates = [
            "gripper_tcp", "tool0", "flange",
            "ee_link", "tool_link", "flange_link", "panda_hand",
            "end_effector", "tcp_link", "link_ee", "gripper_link",
        ]
        ee_idx = None
        for candidate in ee_candidates:
            try:
                result = robot.find_bodies(candidate)
                if result is not None and len(result[0]) > 0:
                    ee_idx = result[0][0]
                    print(f"[ACT DEBUG] Found EE body '{candidate}' at index {ee_idx}")
                    break
            except (ValueError, RuntimeError):
                continue

        if ee_idx is None:
            # Fallback: use the last body (often the EE in serial manipulators)
            ee_idx = robot.data.body_pos_w.shape[1] - 1
            print(f"[ACT DEBUG] No known EE name found, using last body index {ee_idx} "
                  f"('{body_names[ee_idx]}'). Adjust if wrong!")

        extract_act_observations._ee_idx = ee_idx
        extract_act_observations._ee_idx_resolved = True

    ee_idx = extract_act_observations._ee_idx

    # Joint positions — (num_envs, num_joints) → first 7 for the arm
    joint_pos = robot.data.joint_pos[env_idx, :7].cpu().numpy()  # (7,)

    # TCP pose — from the articulation's rigid-body state for the EE body
    # body_pos_w: (num_envs, num_bodies, 3),  body_quat_w: (num_envs, num_bodies, 4)
    tcp_pos  = robot.data.body_pos_w[env_idx, ee_idx].cpu().numpy()    # (3,)
    tcp_quat = robot.data.body_quat_w[env_idx, ee_idx].cpu().numpy()   # (4,) wxyz in Isaac Lab!

    # IMPORTANT: Isaac Lab uses (w, x, y, z) quaternion convention,
    # but RunACT training data uses (x, y, z, w) from ROS.
    # Re-order: wxyz → xyzw
    tcp_quat = np.array([tcp_quat[1], tcp_quat[2], tcp_quat[3], tcp_quat[0]], dtype=np.float32)

    # TCP velocity — from rigid body state
    tcp_lin_vel = robot.data.body_lin_vel_w[env_idx, ee_idx].cpu().numpy()  # (3,)
    tcp_ang_vel = robot.data.body_ang_vel_w[env_idx, ee_idx].cpu().numpy()  # (3,)

    # TCP error — this is controller-specific.  If your Isaac Lab env
    # exposes a Cartesian impedance controller with error tracking, read
    # it from there.  Otherwise, you can compute it as (desired - actual)
    # for position (3) and orientation (3), or set to zeros if your
    # controller doesn't produce this.
    tcp_error = np.zeros(6, dtype=np.float32)  # PLACEHOLDER — see note above

    # Assemble the 26-dim state vector in EXACTLY the same order as training
    state_np = np.concatenate([
        tcp_pos,        # 3
        tcp_quat,       # 4  (x, y, z, w)
        tcp_lin_vel,    # 3
        tcp_ang_vel,    # 3
        tcp_error,      # 6
        joint_pos,      # 7
    ]).astype(np.float32)

    assert state_np.shape == (26,), f"State vector has {state_np.shape[0]} dims, expected 26"

    # Normalise
    raw = torch.from_numpy(state_np).float().unsqueeze(0).to(device)
    obs_dict["observation.state"] = (raw - state_mean) / state_std

    return obs_dict


# ===========================================================================
# Main play loop
# ===========================================================================
@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    agent_cfg=None,  # we don't use an RL agent config, but hydra provides it
):
    """Play with the ACT BC policy inside Isaac Lab."""

    # --- Environment configuration ---
    env_cfg.scene.num_envs = (
        args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    )
    if args_cli.seed is not None:
        env_cfg.seed = args_cli.seed
    env_cfg.sim.device = (
        args_cli.device if args_cli.device is not None else env_cfg.sim.device
    )

    # --- Create the environment ---
    env = gym.make(
        args_cli.task,
        cfg=env_cfg,
        render_mode="rgb_array" if args_cli.video else None,
    )
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # --- Video recording ---
    if args_cli.video:
        log_dir = os.path.join("logs", "act_bc_play")
        os.makedirs(log_dir, exist_ok=True)
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording video.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # --- Device ---
    device = torch.device(env_cfg.sim.device)

    # --- Load ACT policy + stats ---
    policy, img_stats, state_mean, state_std, action_mean, action_std = (
        load_act_policy(device)
    )

    # --- Timing ---
    sim_dt = env.unwrapped.step_dt          # simulation step duration
    policy_dt = 1.0 / args_cli.policy_freq  # how often the ACT policy runs (0.25s at 4Hz)
    # How many sim steps to repeat each ACT action for
    steps_per_action = max(1, int(round(policy_dt / sim_dt)))
    print(f"[INFO] sim_dt={sim_dt:.4f}s, policy_dt={policy_dt:.4f}s → "
          f"repeating each action for {steps_per_action} sim steps")

    image_scaling = 0.25  # must match training (RunACT.image_scaling)

    # --- Reset ---
    obs_rl, _ = env.reset()  # the flat RL observation (we won't use it for ACT)
    policy.reset()
    timestep = 0

    print("[INFO] Starting ACT BC play loop …")

    # ------------------------------------------------------------------
    # Main simulation loop
    # ------------------------------------------------------------------
    while simulation_app.is_running():
        loop_start = time.time()

        # 1. Extract structured observations for ACT
        obs_dict = extract_act_observations(
            env, device, img_stats, state_mean, state_std,
            image_scaling=image_scaling, env_idx=0,
        )

        # 2. ACT inference
        with torch.inference_mode():
            normalized_action = policy.select_action(obs_dict)  # (1, 7)

        # 3. Un-normalise
        raw_action = (normalized_action * action_std) + action_mean  # (1, 7)
        action_np = raw_action[0].cpu().numpy()  # (7,)

        print(f"[ACT] step {timestep} | twist: {action_np[:6]} | gripper: {action_np[6]:.4f}")

        # 4. Convert to the shape the Isaac Lab env expects
        #    ACT outputs 7 dims: 6D Cartesian twist (lin xyz + ang xyz) + gripper.
        #    The Isaac Lab env action space is 6D (twist only), so drop the gripper dim.
        action_6d = action_np[:6]
        action_tensor = torch.from_numpy(action_6d).float().unsqueeze(0).to(device)

        # 5. Step the simulation (repeat the same action for steps_per_action)
        for _ in range(steps_per_action):
            obs_rl, _, dones, _, _ = env.step(action_tensor)

            # If the episode resets, re-initialise ACT's internal state
            if dones.any():
                policy.reset()

        timestep += 1

        # --- Video exit ---
        if args_cli.video and timestep >= args_cli.video_length:
            break

        # --- Real-time pacing ---
        if args_cli.real_time:
            elapsed = time.time() - loop_start
            sleep_time = policy_dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    env.close()
    print("[INFO] ACT BC play loop finished.")


if __name__ == "__main__":
    main()
    simulation_app.close()
