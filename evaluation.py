import os
from typing import Any, Dict

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from data.VerbaLightGCN_data_module import VerbaLightGCNDataModule
import logging
from logging.handlers import RotatingFileHandler
import shutil
from models.VerbaLightGCN import VerbaLightGCN

@hydra.main(config_path="config", config_name="VerbaLightGCN_config", version_base=None)
def evaluation(config: DictConfig) -> None:
    """Train a model using the provided configuration.

    Args:
        config: Configuration dictionary
    """
    # Set random seed for reproducibility 
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
    
    # model.item_attributes = data_module.item_attributes

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
    logger = hydra.utils.instantiate(config.logger)

    # Initialize trainer
    trainer = hydra.utils.instantiate(config.trainer, logger=logger, callbacks=callbacks)
    # Train model

    trainer.strategy.strict_loading = False
    trainer.test(model, data_module, ckpt_path=config.training.pretrain_path, weights_only=False)


if __name__ == "__main__":
    evaluation() 