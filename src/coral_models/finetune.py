"""Functions related to the finetuning of Wav2Vec 2.0 models on ASR datasets."""

from functools import partial
import logging
from typing import Callable

from omegaconf import DictConfig
from transformers import EarlyStoppingCallback, TrainerCallback
from wandb.sdk.wandb_init import init as wandb_init
from wandb.sdk.wandb_run import finish as wandb_finish

from .data import load_data
from .model_setup import load_model_setup
from .protocols import ModelSetup
from .utils import disable_tqdm

logger = logging.getLogger(__package__)


def prepare_dataset_example(example: dict, processor: Callable) -> dict:
    """Prepare a dataset example for the model.

    Args:
        example: The example from the dataset.
        processor: The processor to use.

    Returns:
        The prepared example.
    """
    # Prepare audio
    audio = example["audio"]
    sr = audio["sampling_rate"]
    processed = processor(audio["array"], sampling_rate=sr)
    if "input_values" in processed:
        example["input_values"] = processed.input_values[0]
        example["num_seconds"] = len(example["input_values"]) / sr
    if "input_features" in processed:
        example["input_features"] = processed.input_features[0]
        example["num_seconds"] = len(example["input_features"]) / sr

    # Prepare transcriptions
    example["labels"] = processor(text=example["text"], truncation=True).input_ids
    example["input_length"] = len(example["labels"])

    return example


def finetune(cfg: DictConfig) -> None:
    """Finetune a model on a dataset.

    Args:
        cfg (DictConfig):
            The Hydra cfguration object.
    """
    model_setup: ModelSetup = load_model_setup(cfg)
    processor = model_setup.load_processor()
    processor.save_pretrained(cfg.model_dir)
    model = model_setup.load_model()
    dataset = load_data(cfg)

    dataset = dataset.map(
        function=partial(prepare_dataset_example, processor=processor),
        remove_columns=dataset["train"].column_names,
    )
    dataset = dataset.filter(
        function=lambda example: example["num_seconds"] <= cfg.max_seconds_per_example
    )

    if cfg.wandb:
        wandb_init(
            project=cfg.wandb_project,
            group=cfg.wandb_group,
            name=cfg.wandb_name,
            config=dict(cfg),
        )

    if "val" not in dataset:
        logger.info("No validation set found. Disabling early stopping.")

    trainer = model_setup.load_trainer_class()(
        model=model,
        data_collator=model_setup.load_data_collator(),
        args=model_setup.load_training_arguments(),
        compute_metrics=model_setup.load_compute_metrics(),
        train_dataset=dataset["train"],
        eval_dataset=dataset["val"] if "val" in dataset else None,
        tokenizer=getattr(processor, "tokenizer"),
        callbacks=load_early_stopping_callback(cfg) if "val" in dataset else None,
    )

    with disable_tqdm():
        trainer.train(resume_from_checkpoint=cfg.resume_from_checkpoint)
    wandb_finish()

    model.save_pretrained(cfg.model_dir)
    if cfg.push_to_hub:
        trainer.push_to_hub()


def load_early_stopping_callback(cfg: DictConfig) -> list[TrainerCallback]:
    """Load the early stopping callback for the trainer.

    Args:
        cfg (DictConfig):
            The Hydra configuration object.

    Returns:
        list of TrainerCallback:
            The callbacks.
    """
    callbacks: list[TrainerCallback] = list()
    if cfg.early_stopping:
        early_stopping_callback = EarlyStoppingCallback(
            early_stopping_patience=cfg.early_stopping_patience
        )
        callbacks = [early_stopping_callback]
    return callbacks
