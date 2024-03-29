"""Functions related to the model setups."""

from omegaconf import DictConfig

from .protocols import ModelSetup
from .wav2vec2 import Wav2Vec2ModelSetup
from .whisper import WhisperModelSetup


def load_model_setup(cfg: DictConfig) -> ModelSetup:
    """Get the model setup for the given configuration.

    Args:
        cfg (DictConfig):
            The Hydra configuration object.

    Returns:
        ModelSetup:
            The model setup.
    """
    model_type: str = cfg.model.type
    match model_type:
        case "wav2vec2":
            return Wav2Vec2ModelSetup(cfg)
        case "whisper":
            return WhisperModelSetup(cfg)
        case _:
            raise ValueError(f"Unsupported model type: {model_type!r}")
