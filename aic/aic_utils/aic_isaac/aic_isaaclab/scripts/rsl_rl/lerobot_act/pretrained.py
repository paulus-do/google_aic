"""
Minimal base classes for inference-only usage of ACT policy.
Stripped of: draccus, TrainPipelineConfig, PEFT, hub push, model cards, training utils.
"""
import abc
import json
import logging
import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, TypeVar

import packaging
import safetensors
from safetensors.torch import load_model as load_model_as_safetensor
from torch import Tensor, nn

from .constants import ACTION, OBS_STATE
from .types import FeatureType, PolicyFeature

T_Config = TypeVar("T_Config", bound="PreTrainedConfig")
T_Policy = TypeVar("T_Policy", bound="PreTrainedPolicy")

SAFETENSORS_SINGLE_FILE = "model.safetensors"
CONFIG_NAME = "config.json"

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Minimal device helpers (replaces lerobot.utils.device_utils)
# ---------------------------------------------------------------------------
import torch


def auto_select_torch_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def is_torch_device_available(device: str) -> bool:
    try:
        torch.device(device)
        if "cuda" in device:
            return torch.cuda.is_available()
        if device == "mps":
            return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# PreTrainedConfig — loads from config.json, no draccus needed
# ---------------------------------------------------------------------------
# Registry for subclass lookup by "type" field in config.json
_CONFIG_REGISTRY: dict[str, type] = {}


@dataclass
class PreTrainedConfig(abc.ABC):
    n_obs_steps: int = 1
    input_features: dict[str, PolicyFeature] | None = field(default_factory=dict)
    output_features: dict[str, PolicyFeature] | None = field(default_factory=dict)
    device: str | None = None
    use_amp: bool = False
    pretrained_path: Path | None = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Auto-register subclasses that define 'name'
        name = getattr(cls, "_config_name", None)
        if name:
            _CONFIG_REGISTRY[name] = cls

    def __post_init__(self) -> None:
        if not self.device or not is_torch_device_available(self.device):
            auto_device = auto_select_torch_device()
            logger.warning(f"Device '{self.device}' is not available. Switching to '{auto_device}'.")
            self.device = auto_device.type

    @property
    def type(self) -> str:
        return getattr(self, "_config_name", self.__class__.__name__)

    @property
    def robot_state_feature(self) -> PolicyFeature | None:
        if not self.input_features:
            return None
        for ft_name, ft in self.input_features.items():
            if ft.type is FeatureType.STATE and ft_name == OBS_STATE:
                return ft
        return None

    @property
    def env_state_feature(self) -> PolicyFeature | None:
        if not self.input_features:
            return None
        for _, ft in self.input_features.items():
            if ft.type is FeatureType.ENV:
                return ft
        return None

    @property
    def image_features(self) -> dict[str, PolicyFeature]:
        if not self.input_features:
            return {}
        return {key: ft for key, ft in self.input_features.items() if ft.type is FeatureType.VISUAL}

    @property
    def action_feature(self) -> PolicyFeature | None:
        if not self.output_features:
            return None
        for ft_name, ft in self.output_features.items():
            if ft.type is FeatureType.ACTION and ft_name == ACTION:
                return ft
        return None

    @classmethod
    def from_pretrained(cls, pretrained_name_or_path: str | Path, **kwargs) -> "PreTrainedConfig":
        """Load config from a local directory or HuggingFace Hub repo."""
        model_id = str(pretrained_name_or_path)
        config_file = None

        if Path(model_id).is_dir():
            candidate = os.path.join(model_id, CONFIG_NAME)
            if os.path.isfile(candidate):
                config_file = candidate
        else:
            # Try downloading from HuggingFace Hub
            try:
                from huggingface_hub import hf_hub_download
                config_file = hf_hub_download(repo_id=model_id, filename=CONFIG_NAME)
            except Exception as e:
                raise FileNotFoundError(f"{CONFIG_NAME} not found in {model_id}") from e

        if config_file is None:
            raise FileNotFoundError(f"{CONFIG_NAME} not found in {model_id}")

        with open(config_file) as f:
            raw = json.load(f)

        return cls._from_dict(raw, **kwargs)

    @classmethod
    def _from_dict(cls, raw: dict[str, Any], **overrides) -> "PreTrainedConfig":
        """Instantiate config from a dict (loaded from JSON). Handles PolicyFeature deserialization."""
        # Look up the right subclass from the "type" field
        config_type = raw.pop("type", None)
        target_cls = _CONFIG_REGISTRY.get(config_type, cls) if config_type else cls

        # Deserialize PolicyFeature dicts
        for key in ("input_features", "output_features"):
            if key in raw and isinstance(raw[key], dict):
                raw[key] = {
                    name: PolicyFeature(type=FeatureType(v["type"]), shape=tuple(v["shape"]))
                    for name, v in raw[key].items()
                }

        # Deserialize NormalizationMode dicts
        if "normalization_mapping" in raw and isinstance(raw["normalization_mapping"], dict):
            from .types import NormalizationMode
            raw["normalization_mapping"] = {
                k: NormalizationMode(v) for k, v in raw["normalization_mapping"].items()
            }

        # Apply overrides
        raw.update(overrides)

        # Filter to only fields the dataclass accepts
        valid_fields = {f.name for f in fields(target_cls)}
        filtered = {k: v for k, v in raw.items() if k in valid_fields}

        return target_cls(**filtered)


# ---------------------------------------------------------------------------
# PreTrainedPolicy — nn.Module with from_pretrained for loading weights
# ---------------------------------------------------------------------------
class PreTrainedPolicy(nn.Module, abc.ABC):
    config_class = None
    name = None

    def __init__(self, config: PreTrainedConfig, *args, **kwargs):
        super().__init__()
        self.config = config

    @classmethod
    def from_pretrained(
        cls,
        pretrained_name_or_path: str | Path,
        *,
        config: PreTrainedConfig | None = None,
        strict: bool = False,
        **kwargs,
    ) -> "PreTrainedPolicy":
        """Load policy weights from a local directory or HuggingFace Hub repo.
        The policy is set to eval mode by default.
        """
        if config is None:
            config = cls.config_class.from_pretrained(pretrained_name_or_path, **kwargs)

        model_id = str(pretrained_name_or_path)
        instance = cls(config)

        if os.path.isdir(model_id):
            model_file = os.path.join(model_id, SAFETENSORS_SINGLE_FILE)
        else:
            try:
                from huggingface_hub import hf_hub_download
                model_file = hf_hub_download(repo_id=model_id, filename=SAFETENSORS_SINGLE_FILE)
            except Exception as e:
                raise FileNotFoundError(f"{SAFETENSORS_SINGLE_FILE} not found in {model_id}") from e

        # Load weights
        load_kwargs = {"strict": strict}
        if packaging.version.parse(safetensors.__version__) >= packaging.version.parse("0.4.3"):
            load_kwargs["device"] = config.device

        missing, unexpected = load_model_as_safetensor(instance, model_file, **load_kwargs)
        if missing:
            logger.warning(f"Missing key(s) when loading model: {missing}")
        if unexpected:
            logger.warning(f"Unexpected key(s) when loading model: {unexpected}")

        instance.to(config.device)
        instance.eval()
        return instance

    # Abstract methods — forward/select_action are implemented by ACTPolicy
    @abc.abstractmethod
    def reset(self): ...

    @abc.abstractmethod
    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict | None]: ...

    @abc.abstractmethod
    def select_action(self, batch: dict[str, Tensor]) -> Tensor: ...
