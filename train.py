import os
from typing import Optional

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from data.VerbaLightGCN_data_module import VerbaLightGCNDataModule
import logging
from logging.handlers import RotatingFileHandler
import shutil
import torch
from scipy.sparse import coo_matrix
torch.serialization.add_safe_globals([coo_matrix])

def setup_logging(log_dir: str = "logs/train", wipe_existing: bool = True) -> logging.Logger:
    # Ensure log directory exists
    # delete the log dir if it exists (for clean runs)
    if wipe_existing and os.path.exists(log_dir):
        shutil.rmtree(log_dir, ignore_errors=True)
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "train.log")
    
    py_logger = logging.getLogger()  # root logger so Lightning also uses it
    py_logger.setLevel(logging.INFO)  # Enable INFO, WARNING, ERROR, and CRITICAL logging

    # Remove default handlers (avoid duplicate logs)
    for h in py_logger.handlers[:]:
        py_logger.removeHandler(h)

    # Rotating file handler
    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=100 * 1024 * 1024,  # 100 MB
        backupCount=5
    )
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)

    # Console handler (to see logs in terminal/nohup.out)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    py_logger.addHandler(file_handler)
    py_logger.addHandler(console_handler)

    return py_logger

@hydra.main(config_path="config", config_name="VerbaLightGCN_config", version_base=None)
def train(config: DictConfig) -> None:
    """Train a model using the provided configuration.

    Args:
        config: Configuration dictionary
    """
    # Set random seed for reproducibility
    resume_ckpt: Optional[str] = config.training.pretrain_path if config.training.pretrain_path else None
    _py_logger = setup_logging(
        f"{config.paths.log_dir}/{config.logging.wandb_run_name}",
        wipe_existing=(resume_ckpt is None),
    )

    pl.seed_everything(config.training.seed)

    # Initialize data module
    data_module: VerbaLightGCNDataModule = hydra.utils.instantiate(config.data)

    user_num, item_num = data_module.get_user_item_num()

    model_config = OmegaConf.to_container(config.model, resolve=True)
    model_config['user_num'] = int(user_num)
    model_config['item_num'] = int(item_num)
    model_config['interaction_matrix'] = data_module._create_sparse_matrix()
    model_config['item_attributes'] = data_module.item_attributes

    model_class = hydra.utils.get_class(config.model._target_)
    model = model_class(**model_config)
    
    callbacks_config = config.callbacks_vlgcn

    callbacks = []
    if callbacks_config.get("checkpoint"):
        callbacks.append(hydra.utils.instantiate(callbacks_config.checkpoint))
    if callbacks_config.get("checkpoint_llm_head"):
        callbacks.append(hydra.utils.instantiate(callbacks_config.checkpoint_llm_head))
    if callbacks_config.get("early_stopping"):
        callbacks.append(hydra.utils.instantiate(callbacks_config.early_stopping))
    if callbacks_config.get("lr_monitor"):
        callbacks.append(hydra.utils.instantiate(callbacks_config.lr_monitor))

    # Initialize logger
    wandb_logger = hydra.utils.instantiate(config.logger)

    # Initialize trainer
    trainer = hydra.utils.instantiate(config.trainer, logger=wandb_logger, callbacks=callbacks)
    # Train model
    if config.training.mode == "train":
        trainer.validate(model, data_module)
        if config.training.pretrain_path:
            trainer.fit(model, data_module, ckpt_path = config.training.pretrain_path, weights_only=False)
        else:
            trainer.fit(model, data_module, ckpt_path=resume_ckpt,  weights_only=False)
        # Load best checkpoint before testing
        best_model_path = trainer.checkpoint_callback.best_model_path
        trainer.strategy.strict_loading = False
        trainer.test(model, data_module, ckpt_path=best_model_path, weights_only=False)
    elif config.training.mode == "test":
        trainer.strategy.strict_loading = False
        trainer.test(model, data_module, ckpt_path=config.training.pretrain_path)
    else:
        raise ValueError(f"Unknown mode: {config.training.mode}")


if __name__ == "__main__":
    train() 