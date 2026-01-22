from typing import Any, Dict, List, Optional, Tuple

import json
import logging
import os
import networkx as nx
import numpy as np
import pandas as pdnvidi
import pytorch_lightning as pl
import scipy.sparse as sp
import torch
import torch.nn as nn
from openai import OpenAI
from pydantic import BaseModel
from torch.amp import autocast
from torch.serialization import safe_globals
from torchmetrics import MeanSquaredError, MeanAbsoluteError
from torchmetrics.classification import MulticlassF1Score, MulticlassAccuracy, MulticlassAUROC
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
from metrics.retreval_metrics import (
    RetrievalMRR,
    RetrievalPrecision, 
    RetrievalRecall,
    RetrievalNormalizedDCG
)
from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_uniform_initialization
from recbole.model.loss import BPRLoss, EmbLoss
from recbole.utils import InputType
from scipy.sparse import coo_matrix
import pandas as pd
# Set up logger
logger = logging.getLogger(__name__)

class DummyOptimizer(torch.optim.Adam):
    """A dummy optimizer that does nothing during step."""
    
    def step(self, closure=None):
        """Override step method to do nothing.
        
        Args:
            closure: Optional closure for recomputing loss
        """
        pass

class RankingResponse(BaseModel):
    AggregatedUserProfile: str
    AggregatedCandidate1: str
    AggregatedCandidate2: str
    Explanation: str
    Output: int


class UpdateDescriptionResponse(BaseModel):
    Explanation: str
    UpdatedUserProfile: str
    UpdatedItem1Profile: str
    UpdatedItem2Profile: str

class ItemProfileResponse(BaseModel):
    ItemProfile: str

class UserProfileResponse(BaseModel):
    UserProfile: str

class VerbaLightGCN(pl.LightningModule):
    def __init__(
        self,
        prompt_path: str,
        user_num: int,
        item_num: int,
        embedding_dim: int = 64,
        num_layers: int = 2,
        reg_weight: float = 1e-5,
        learning_rate: float = 1e-3,
        checkpoint_dir: str = "checkpoints",
        wandb_run_name: str = "llm-ranking",
        **kwargs: Any,
    ) -> None:
        """Initialize the LLM ranking model.

        Args:
            item_info_path: Path to item description JSON file
            user_profile_path: Path to user profile file
            item_profile_path: Path to item profile file
            interaction_path: Path to interaction sequence file
            prompt_path: Path to prompt templates
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
            checkpoint_dir: Directory to save checkpoints
            wandb_run_name: Name of the wandb run
            **kwargs: Additional arguments
        """
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        
        # Set up checkpoint paths
        self.checkpoint_dir = checkpoint_dir
        self.wandb_run_name = wandb_run_name
        self.checkpoint_path = os.path.join(self.checkpoint_dir, self.wandb_run_name)
        os.makedirs(self.checkpoint_path, exist_ok=True)
        self.item_attributes = self.hparams.item_attributes
        self.max_profile_length = self.hparams.max_profile_length
        self.tensor_parallel_size = self.hparams.tensor_parallel_size
        self.max_num_seqs = self.hparams.max_num_seqs
        self.save_content = self.hparams.save_content
        self.without_neighbor = self.hparams.without_neighbor
        self.user_num = user_num
        self.item_num = item_num
        # Storage variables for full sort evaluation acceleration
        self.restore_user_e = None
        self.restore_item_e = None

        # ========== LLM model ==========
        self.llm = LLM(model=self.hparams.llm_model, max_model_len=32768, max_num_batched_tokens=32768 * 3,enable_chunked_prefill=True, max_num_seqs=self.max_num_seqs, tensor_parallel_size=self.tensor_parallel_size)
        
        
        if self.hparams.user_profile_path is not None:
            with open(self.hparams.user_profile_path, 'r') as f:
                user_profile = json.load(f)
            self.user_memory = {int(k): [v] for k, v in user_profile.items()}
            logger.info(f"Loaded {len(self.user_memory)} user profiles")
            self.user_profile_ori = {int(k): v for k, v in user_profile.items()}
        else:
            self.user_memory = {}
            self.user_profile_ori = {}
        if self.hparams.item_profile_path is not None:
            with open(self.hparams.item_profile_path, 'r') as f:
                item_profile = json.load(f)
                self.item_memory = {int(k): [v] for k, v in item_profile.items()}
            logger.info(f"Loaded {len(self.item_memory)} item profiles")
        else:
            logger.info("No item profile path provided, using item attributes to initialize item profile")
            self.item_memory = {}
            for item, item_info in self.item_attributes.items():
                self.item_memory[item] = ["Item attributes: " + ", ".join(json.loads(item_info)['Item key characteristics'])]

        # Initialize metrics
        self.compute_f1 = MulticlassF1Score(num_classes=2, average='macro')  # or 'micro'
        self.compute_acc = MulticlassAccuracy(num_classes=2)
        self.compute_auc = MulticlassAUROC(num_classes=2)
        self.dummy_param = torch.nn.Parameter(torch.tensor(0.0))

        
        self._load_prompts()
        self.monitor_epoch = -1
        self.log_training = True

        with open(self.hparams.item_description_path, 'r') as f:
            self.item_description = json.load(f)
        self.user_hist_cache = {}
    
    def _batch_predict(self, batch: Dict[str, Any]):
        
        conversation = []
        for p in batch['predict_prompt']:
            conversation.append([{"role": "system", "content": "You are a helpful assistant."},
                                {"role": "user", "content": p}])

        guided_decoding_params = GuidedDecodingParams(json=RankingResponse.model_json_schema())
        outputs = self.llm.chat(conversation, sampling_params=SamplingParams(max_tokens=1024, guided_decoding=guided_decoding_params, temperature=self.hparams.temperature, top_p=self.hparams.top_p),
                            use_tqdm=False)

        preds, rs_explanations = [], []
        for idx, output in enumerate(outputs):
            res = {'Output': -1, 'Explanation': 'N/A'}
            res["Explanation"] = output.outputs[0].text
            try:
                res = json.loads(output.outputs[0].text)
                res["Explanation"] = output.outputs[0].text
            except json.JSONDecodeError:
                pass

            output = int(res['Output'])
            if output > 2 or output < 1:  # In some case output return item ID, so we need to convert it to 1 and 2
                if output == batch['item_1'][idx]:
                    output = 1
                elif output == batch['item_2'][idx]:
                    output = 2
                else:
                    output = -1
            preds.append(output)
            rs_explanations.append(res['Explanation'])
        if self.trainer.testing or self.trainer.global_step % 10 == 0:
            logger.info(batch['predict_prompt'][0])
            logger.info(outputs[0].outputs[0].text)
        return preds, rs_explanations

    def _call_vllm_inference(self, conversation, max_tokens=4096, guided_decoding_params=None):
        outputs = self.llm.chat(conversation, sampling_params=SamplingParams(max_tokens=max_tokens, guided_decoding=guided_decoding_params, temperature=self.hparams.temperature, top_p=self.hparams.top_p), use_tqdm=False)
        return outputs

    def batch_refinement_profile(self, promts_list):
        false_prompt_indices = [i for i in range(len(promts_list)) if promts_list[i] != "None"]
        conversation = []
        for p in [promts_list[i] for i in false_prompt_indices]:
            conversation.append([{"role": "system", "content": "You are a helpful assistant."},
                                {"role": "user", "content": p[:32768]}])
    
        guided_decoding_params = GuidedDecodingParams(json=UpdateDescriptionResponse.model_json_schema())
        outputs = self._call_vllm_inference(conversation, max_tokens=4096, guided_decoding_params=guided_decoding_params)
        profile_update_response = [-1] * len(promts_list)
        for i, output in enumerate(outputs):
            try:
                res = json.loads(output.outputs[0].text)
                new_res = {
                    "UpdatedUserProfile": res['UpdatedUserProfile'],
                    "UpdatedItem1Profile": res['UpdatedItem1Profile'],
                    "UpdatedItem2Profile": res['UpdatedItem2Profile'],
                    "Explanation": res['Explanation']
                }
            except json.JSONDecodeError:
                new_res = -1
            profile_update_response[false_prompt_indices[i]] = new_res
        return profile_update_response
    
    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Forward pass of the model.

        Args:
            batch: Batch of data containing user and item information

        Returns:
            Dictionary containing model predictions and explanations
        """
        pred_prompts, optimizer_prompts = self._construct_prompts(batch)
        batch['predict_prompt'] = pred_prompts
        llm_predictions, llm_explanations = self._batch_predict(batch)

        return {"llm_predictions": llm_predictions, "llm_explanations": llm_explanations, "optimizer_prompts": optimizer_prompts}
    
    

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Training step.

        Args:
            batch: Batch of data
            batch_idx: Index of the batch

        Returns:
            Loss tensor
        """
        ## summary CF profile
        if self.max_profile_length > 1:  # Skip summarization when max_profile_length == 1
            num_long_profile_user = len([k for k,v in self.user_memory.items() if len(v) >=3])
            num_long_profile_item = len([k for k,v in self.item_memory.items() if len(v) >=3])
            if (self.trainer.global_step % self.max_profile_length == 0) and self.trainer.training:
                logger.info(f"Long CF user profile: {num_long_profile_user} / {len(self.user_memory)}")
                logger.info(f"Long CF item profile: {num_long_profile_item} / {len(self.item_memory)}")
                self.summarize_memory(summary_user_mem=True, summary_item_mem=True, max_profile_length=self.max_profile_length)

        outputs = self(batch)
        llm_predictions = outputs['llm_predictions']
        self.optimizer_step(outputs, batch)
        optimizer = self.optimizers()
        optimizer.step()
        optimizer.zero_grad()

        llm_predictions = torch.tensor(llm_predictions).to(self.device)
        metrics_llm = self._compute_metric(llm_predictions[torch.where(llm_predictions != -1)] , batch["label"][torch.where(llm_predictions != -1)])
        logger.info(metrics_llm)
        
        if self.trainer.global_step % 100 == 0:
            self.on_train_epoch_end()
        self.log_dict(metrics_llm, on_step=True, on_epoch=True)
    

    def optimizer_step(self, outputs: Dict[str, Any], batch: Dict[str, Any]) -> None:
        # OPTIMIZE LLM
        optimizer_prompt_list = []
        aggragted_user_profile = []
        for i in range(len(batch['predict_prompt'])):
            if 'cf_label' in outputs['optimizer_prompts'][i]:
                raise ValueError("Please check the prompt, cf_label is not allowed in the prompt")
            
            try:
                explaination = json.loads(outputs['llm_explanations'][i])
                aggregated_user = explaination['AggregatedUserProfile']
                aggregated_item1 = explaination['AggregatedCandidate1']
                aggregated_item2 = explaination['AggregatedCandidate2']
                explaination = explaination['Explanation']

                if batch['label'][i].item() != outputs['llm_predictions'][i]:
                    optimizer_prompt_template = self.refinement_prompt_template
                else:
                    optimizer_prompt_template = self.refinement_prompt_template_true

                user_id = batch['user_id'][i].item()
                item_1 = batch['item_1'][i].item()
                item_2 = batch['item_2'][i].item()
                label = batch['label'][i].item()
                if not self.without_neighbor:
                    neighbor_item = [i for i in batch["neighbor_item"][i].cpu().tolist() if i != 0 and i in self.item_attributes]
                    neighbor_user_1 = [i for i in batch["neighbor_user_1"][i].cpu().tolist() if i != 0 and i in self.user_memory]
                    neighbor_user_2 = [i for i in batch["neighbor_user_2"][i].cpu().tolist() if i != 0 and i in self.user_memory]
                else:
                    neighbor_item = []
                    neighbor_user_1 = []
                    neighbor_user_2 = []
                optimizer_prompt = optimizer_prompt_template.replace("{{{user_profile}}}", self.user_memory.get(user_id, ["None"])[-1])\
                                        .replace("{{{user_hist}}", self.construct_neighbor_item_feature(neighbor_item))\
                                        .replace("{{{item_1_information}}}", self.item_attributes[item_1])\
                                        .replace("{{{item_1_profile}}}", self.item_memory.get(item_1, ["None"])[-1])\
                                        .replace("{{{user_like_item_1}}}", self.construct_neighbor_user_feature(neighbor_user_1))\
                                        .replace("{{{item_2_information}}}", self.item_attributes[item_2])\
                                        .replace("{{{item_2_profile}}}", self.item_memory.get(item_2, ["None"])[-1])\
                                        .replace("{{{user_like_item_2}}}", self.construct_neighbor_user_feature(neighbor_user_2))\
                                        .replace("{{{gt}}}", str(label))\
                                        .replace("{{{false_candidate}}}", str(3 - label))\
                                        .replace("{{{true_candidate_title}}}", json.loads(self.item_attributes[item_1 if label == 1 else item_2])['Title'])\
                                        .replace("{{{false_candidate_title}}}", json.loads(self.item_attributes[item_1 if label == 2 else item_2])['Title'])\
                                        .replace("{{{item_1_description}}}", self.item_description['description'][str(item_1)])\
                                        .replace("{{{item_2_description}}}", self.item_description['description'][str(item_2)])\
                                        .replace("{{{rs_pred}}}", str(outputs['llm_predictions'][i]))\
                                        .replace("{{{rs_explanation}}}", explaination)\
                                        .replace("{{{aggregated_user}}}", aggregated_user)\
                                        .replace("{{{aggregated_item1}}}", aggregated_item1)\
                                        .replace("{{{aggregated_item2}}}", aggregated_item2)

                optimizer_prompt = optimizer_prompt.replace('\\"', '')
            except Exception as e:
                logger.warning("Error in parsing llm explanation: " + str(e) + "\n" + outputs['llm_explanations'][i])
                explaination = {
                    "AggregatedUserProfile": "No information",
                    "AggregatedCandidate1": "No information",
                    "AggregatedCandidate2": "No information",
                    "Explanation": "No information"
                }
                optimizer_prompt = "None"
                aggregated_user = "No information"

            
            optimizer_prompt_list.append(optimizer_prompt)
            aggragted_user_profile.append(aggregated_user)
            

        refinement_outputs = self.batch_refinement_profile(optimizer_prompt_list)
        logger.info(optimizer_prompt_list[0])
        logger.info(refinement_outputs[0])
        for i, update_description in enumerate(refinement_outputs):
            user_id, item_1, item_2, label = batch['user_id'][i].item(), batch['item_1'][i].item(), batch['item_2'][i].item(), batch['label'][i].item()
            if update_description == -1:
                self.train_content['user_profile_update'].append("None")
                self.train_content['item_1_profile_update'].append("None")
                self.train_content['item_2_profile_update'].append("None")
                self.train_content['llm_predictions'].append(outputs['llm_predictions'][i])
                self.train_content['llm_explanations'].append(outputs['llm_explanations'][i])
                self.train_content['optimizer_prompts'].append(optimizer_prompt_list[i])
                continue
            if isinstance(update_description, list):
                update_description = update_description[0]
            
            logger.info("Explanation: " + update_description['Explanation'])
            if label != outputs['llm_predictions'][i] or self.trainer.current_epoch == 0: # Only update profile if it false
                logger.info(f"update user {user_id} profile " + self.user_memory.get(user_id, ["None"])[-1] + " -> " + update_description['UpdatedUserProfile'])
                update_user_profile = update_description['UpdatedUserProfile']
                if self.max_profile_length == 1:
                    self.user_memory[user_id] = [update_user_profile]
                else:
                    self.user_memory[user_id] = self.user_memory.get(user_id, []) + [update_user_profile]

                if label == 1: # Update the positive item profile
                    logger.info(f"update item {item_1} profile " + self.item_memory.get(item_1, ["None"])[-1] + " -> " + update_description['UpdatedItem1Profile'])
                    update_item_1_profile = update_description['UpdatedItem1Profile']
                    update_item_2_profile = "None"
                    if self.max_profile_length == 1:
                        self.item_memory[item_1] = [update_item_1_profile]
                    else:
                        self.item_memory[item_1] = self.item_memory.get(item_1, []) + [update_item_1_profile]
                else:
                    logger.info(f"update item {item_2} profile " + self.item_memory.get(item_2, ["None"])[-1] + " -> " + update_description['UpdatedItem2Profile'])
                    update_item_2_profile = update_description['UpdatedItem2Profile']
                    update_item_1_profile = "None"
                    if self.max_profile_length == 1:
                        self.item_memory[item_2] = [update_item_2_profile]
                    else:
                        self.item_memory[item_2] = self.item_memory.get(item_2, []) + [update_item_2_profile]
            else:
                update_user_profile = "None"
                update_item_1_profile = "None"
                update_item_2_profile = "None"
            if self.log_training == True:
                self.train_content['user_profile_update'].append(update_user_profile)
                self.train_content['item_1_profile_update'].append(update_item_1_profile)
                self.train_content['item_2_profile_update'].append(update_item_2_profile)
                self.train_content['llm_predictions'].append(outputs['llm_predictions'][i])
                self.train_content['llm_explanations'].append(outputs['llm_explanations'][i])
                self.train_content['optimizer_prompts'].append(optimizer_prompt_list[i])
    
    def _construct_prompts(self, batch: Dict[str, Any]) -> List[str]:
        predict_prompt_list = []
        optimizer_prompt_list = []
        for i in range(len(batch['user_id'])):
            user_id = batch["user_id"][i].item()
            item_1 = batch["item_1"][i].item()
            item_2 = batch["item_2"][i].item()
            label = batch["label"][i].item()
            if not self.without_neighbor:
                neighbor_item = [i for i in batch["neighbor_item"][i].cpu().tolist() if i != 0 and i in self.item_attributes]
                # self.user_hist_cache[user_id] = neighbor_item
                neighbor_user_1 = [i for i in batch["neighbor_user_1"][i].cpu().tolist() if i != 0 and i in self.user_memory]
                neighbor_user_2 = [i for i in batch["neighbor_user_2"][i].cpu().tolist() if i != 0 and i in self.user_memory]
            else:
                neighbor_item = []
                neighbor_user_1 = []
                neighbor_user_2 = []
            self.user_hist_cache[user_id] = self.user_hist_cache.get(user_id, []) + [item_1 if label == 1 else item_2]

            predict_prompt = self.prompt_template.replace("{{{user_profile}}}", self.user_memory.get(user_id, ["None"])[-1])\
                                    .replace("{{{user_hist}}", self.construct_neighbor_item_feature(neighbor_item))\
                                    .replace("{{{item_1_information}}}", self.item_attributes[item_1])\
                                    .replace("{{{item_1_profile}}}", self.item_memory.get(item_1, ["None"])[-1])\
                                    .replace("{{{user_like_item_1}}}", self.construct_neighbor_user_feature(neighbor_user_1))\
                                    .replace("{{{item_2_information}}}", self.item_attributes[item_2])\
                                    .replace("{{{item_2_profile}}}", self.item_memory.get(item_2, ["None"])[-1])\
                                    .replace("{{{user_like_item_2}}}", self.construct_neighbor_user_feature(neighbor_user_2))\
                                    .replace("{{{item_1_description}}}", self.item_description['description'][str(item_1)])\
                                    .replace("{{{item_2_description}}}", self.item_description['description'][str(item_2)])
            predict_prompt_list.append(predict_prompt.replace('\\"', ''))
            optimizer_prompt_list.append("None")

            if self.log_training == True and self.trainer.training:
                self.train_content['user_id'].append(user_id)
                self.train_content['user_profile'].append(self.user_memory.get(user_id, ["None"])[-1])
                self.train_content['neighbor_item'].append(neighbor_item)
                self.train_content['user_hist'].append(self.construct_neighbor_item_feature(neighbor_item))
                self.train_content['item_1_information'].append(self.item_attributes[item_1])
                self.train_content['item_2_information'].append(self.item_attributes[item_2])
                self.train_content['item_1_profile'].append(self.item_memory.get(item_1, ["None"])[-1])
                self.train_content['item_2_profile'].append(self.item_memory.get(item_2, ["None"])[-1])
                self.train_content['neighbor_user_1'].append(self.construct_neighbor_user_feature(neighbor_user_1))
                self.train_content['neighbor_user_2'].append(self.construct_neighbor_user_feature(neighbor_user_2))
                self.train_content['user_like_item_1'].append(self.construct_neighbor_user_feature(neighbor_user_1))
                self.train_content['user_like_item_2'].append(self.construct_neighbor_user_feature(neighbor_user_2))
                self.train_content['predict_prompt'].append(predict_prompt)
                self.train_content['label'].append(label)
        return predict_prompt_list, optimizer_prompt_list
    
    def construct_neighbor_item_feature(self, item_list):
        hist_text = []
        for i in item_list:
            item_profile = self.item_memory.get(i, ["None"])[-1]
            hist_text.append(f"Item {i}: {json.loads(self.item_attributes[i])['Title']} (Item profile: {item_profile})")
        return "\n".join(hist_text) if hist_text else "No information"

    def construct_neighbor_user_feature(self, user_list):
        if len(user_list) == 0:
            return "No information"
        return "\n".join([f"User {i} profile: {self.user_memory.get(i, ["None"])[-1]}" for i in user_list])
    
    def on_train_epoch_start(self):
        if self.log_training == True:
            self.train_content = {
                "user_id": [],
                "user_profile": [],
                "neighbor_item": [],
                "user_hist": [],
                "item_1_information": [],
                "item_2_information": [],
                "item_1_profile": [],
                "item_2_profile": [],
                "neighbor_user_1": [],
                "neighbor_user_2": [],
                "user_like_item_1": [],
                "user_like_item_2": [],
                "predict_prompt": [],
                "label": [],
                "user_profile_update": [],
                "item_1_profile_update": [],
                "item_2_profile_update": [],
                "llm_predictions": [],
                "llm_explanations": [],
                "optimizer_prompts": [],
            }
    def on_train_epoch_end(self):
        ### Save to parquet self.train_content
        os.makedirs(os.path.join(self.hparams['output_dir'], self.hparams['wandb_run_name']), exist_ok=True)
        for k in self.train_content:
            logger.info(f"{k}: {len(self.train_content[k])}")
        df = pd.DataFrame(self.train_content)
        if self.save_content:
            df.to_parquet(os.path.join(self.hparams['output_dir'], self.hparams['wandb_run_name'], f"train_{self.trainer.global_step}.parquet"))

    def on_validation_epoch_start(self):
        if self.trainer.global_step > 0 and self.max_profile_length > 1: # Only summarize profile when training and before validation epoch, skip if max_profile_length == 1
            self.summarize_memory(summary_user_mem=True, summary_item_mem=True)
        self.val_content={
            "user_id":[],
            "item_1":[],
            "item_2":[],
            "label":[],
            "predict_prompt":[],
            "neighbor_item":[],
            "neighbor_user_1":[],
            "neighbor_user_2":[],
            "llm_pred":[],
            "explanation":[],
            "optimizer_prompt":[],
        }

    def validation_step(self, batch, batch_idx = None):
        outputs = self(batch)
        return outputs
        
    def on_validation_batch_end(self, outputs, batch, batch_idx):
        for i in range(len(batch['predict_prompt'])):
            self.val_content["user_id"].append(batch["user_id"][i].item())
            self.val_content["item_1"].append(batch["item_1"][i].item())
            self.val_content["item_2"].append(batch["item_2"][i].item())
            self.val_content["label"].append(batch["label"][i].item())
            self.val_content["neighbor_item"].append(batch["neighbor_item"][i].cpu().tolist())
            self.val_content["neighbor_user_1"].append(batch["neighbor_user_1"][i].cpu().tolist())
            self.val_content["neighbor_user_2"].append(batch["neighbor_user_2"][i].cpu().tolist())
            self.val_content["predict_prompt"].append(batch["predict_prompt"][i])
            self.val_content["llm_pred"].append(outputs["llm_predictions"][i])
            self.val_content["explanation"].append(outputs["llm_explanations"][i])
            self.val_content["optimizer_prompt"].append(outputs["optimizer_prompts"][i])

    def on_validation_epoch_end(self):
        llm_metric = self.compute_metric(self.val_content, pred_col = 'llm_pred', label_col = 'label', mode = 'val')
        df=pd.DataFrame(self.val_content)

        metric = {**llm_metric}
        self.log_dict(metric, on_epoch=True)
        
        if self.save_content:
            os.makedirs(os.path.join(self.hparams['output_dir'], self.hparams['wandb_run_name']), exist_ok=True)
            df.to_parquet(os.path.join(self.hparams['output_dir'], self.hparams['wandb_run_name'], f"val_{self.trainer.global_step}.parquet"))

    def on_test_epoch_start(self):
        """Initialize test content dictionary at the start of test epoch."""
        self.test_content = {
            "user_id": [],
            "item_1": [],
            "item_2": [],
            "label": [],
            "predict_prompt": [],
            "neighbor_item": [],
            "neighbor_user_1": [],
            "neighbor_user_2": [],
            "llm_pred": [],
            "explanation": [],
            "optimizer_prompt": [],
        }
    def test_step(self, batch, batch_idx=None):
        """Process a single test batch.

        Args:
            batch: Batch of data
            batch_idx: Index of the batch

        Returns:
            Model outputs for the batch
        """
        outputs = self(batch)
        return outputs

    def on_test_batch_end(self, outputs, batch, batch_idx):
        """Process test batch results.

        Args:
            outputs: Model outputs
            batch: Input batch
            batch_idx: Index of the batch
        """
        for i in range(len(batch['predict_prompt'])):
            self.test_content["user_id"].append(batch["user_id"][i].item())
            self.test_content["item_1"].append(batch["item_1"][i].item())
            self.test_content["item_2"].append(batch["item_2"][i].item())
            self.test_content["label"].append(batch["label"][i].item())
            self.test_content["neighbor_item"].append(batch["neighbor_item"][i].cpu().tolist())
            self.test_content["neighbor_user_1"].append(batch["neighbor_user_1"][i].cpu().tolist())
            self.test_content["neighbor_user_2"].append(batch["neighbor_user_2"][i].cpu().tolist())
            self.test_content["predict_prompt"].append(batch["predict_prompt"][i])
            self.test_content["llm_pred"].append(outputs["llm_predictions"][i])
            self.test_content["explanation"].append(outputs["llm_explanations"][i])
            self.test_content["optimizer_prompt"].append(outputs["optimizer_prompts"][i])
    def on_test_epoch_end(self):
        """Process test results at the end of test epoch."""
        llm_metric = self.compute_metric(self.test_content, pred_col = 'llm_pred', label_col = 'label', mode='test')
        df = pd.DataFrame(self.test_content)

        metric = {**llm_metric}
        self.log_dict(metric, on_epoch=True)
        
        if self.save_content:
            os.makedirs(os.path.join(self.hparams['output_dir'], self.hparams['wandb_run_name']), exist_ok=True)
            df.to_parquet(os.path.join(self.hparams['output_dir'], self.hparams['wandb_run_name'], f"test_{self.trainer.global_step}.parquet"))

    def _compute_metric(self, pred, label):
        """Compute metrics for predictions and labels.

        Args:
            pred: Predictions tensor
            label: Labels tensor

        Returns:
            Dictionary containing metric values
        """
        pred = torch.tensor(pred, device=self.device) - 1  # sTorch metric only accept label begin 0
        label = torch.tensor(label, device=self.device) - 1
        f1 = self.compute_f1(pred, label)
        acc = self.compute_acc(pred, label)
        
        pred_logit = torch.zeros((pred.shape[0], 2), dtype=torch.float32)
        pred_logit[np.arange(len(pred)), pred] = 1.0
        pred_logit = pred_logit.to(self.device)
        auc = self.compute_auc(pred_logit, label)  # pass label, not one-hot!

        metric = {
            "f1": f1.item(),
            "acc": acc.item(),
            "auc": auc.item(),
        }
        return metric

    def compute_metric(self, content, pred_col, label_col, mode='train'):
        """Compute metrics for a batch of data.

        Args:
            batch: Batch data
            mode: Mode (train/val/test)

        Returns:
            Dictionary containing metric values
        """
        filter_pred = []
        filter_label = []
        pred = content[pred_col]
        label = content[label_col]
        for i in range(len(pred)):
            filter_label.append(label[i])
            if pred[i] != -1:
                filter_pred.append(pred[i])
            else:
                filter_pred.append(3-label[i])
                
        self.metric = self._compute_metric(filter_pred, filter_label)
        logger.info(f"{mode} Step: {self.trainer.global_step}, Metric: {self.metric}")
        metrics = {f"{mode}/{pred_col}_{k}": v for k,v in self.metric.items()}
        return metrics

    def on_save_checkpoint(self, checkpoint) -> None:
        checkpoint['user_profile'] = self.user_memory
        checkpoint['item_profile'] = self.item_memory
        checkpoint['prompt_template'] = self.prompt_template
        checkpoint['refinement_prompt_template'] = self.refinement_prompt_template
        checkpoint['user_summary_prompt'] = self.user_summary_prompt
        checkpoint['item_summary_prompt'] = self.item_summary_prompt
        checkpoint['refinement_prompt_template_true'] = self.refinement_prompt_template_true
        
        # Disable save Qwen weight
        checkpoint['state_dict'] = {k: v for k, v in checkpoint['state_dict'].items() if 'model' not in k}

    def on_load_checkpoint(self, checkpoint) -> None:
        self.user_memory = checkpoint['user_profile']
        self.item_memory = checkpoint['item_profile']

        self.prompt_template = checkpoint['prompt_template']
        self.refinement_prompt_template = checkpoint['refinement_prompt_template']
        self.user_summary_prompt = checkpoint['user_summary_prompt']
        self.item_summary_prompt = checkpoint['item_summary_prompt']
        self.refinement_prompt_template_true = checkpoint['refinement_prompt_template_true']
        self.on_train_epoch_start()

    def _load_prompts(self):
        """Load prompt templates."""
        with open(os.path.join(self.hparams.prompt_path, 'predict_prompt.txt'), 'r') as f:
            self.prompt_template = f.read()
        with open(os.path.join(self.hparams.prompt_path, 'optimizer_prompt_false.txt'), 'r') as f:
            self.refinement_prompt_template = f.read()
        with open(os.path.join(self.hparams.prompt_path, 'optimizer_prompt_true.txt'), 'r') as f:
            self.refinement_prompt_template_true = f.read()
        with open(os.path.join(self.hparams.prompt_path, 'summary_user_mem_prompt.txt'), 'r') as f:
            self.user_summary_prompt = f.read()
        with open(os.path.join(self.hparams.prompt_path, 'summary_item_mem_prompt.txt'), 'r') as f:
            self.item_summary_prompt = f.read()

    def configure_optimizers(self):
        """Configure optimizers.

        Returns:
            Optimizer for the model parameters
        """
        return DummyOptimizer([self.dummy_param], lr=0.0)
    
    def on_train_start(self):
        self.trainer.logger.experiment.log_code("./models/", include_fn=lambda path: path.endswith(os.path.basename(__file__)))
        self.trainer.logger.experiment.log_code(self.hparams.prompt_path, include_fn=lambda path: path.endswith(f".txt"))
    
    def summarize_memory(self, summary_user_mem=False, summary_item_mem=False, max_profile_length=2):
        if self.max_profile_length == 1:
            return

        def build_summary_user_prompt(profile_dict, entity_type):
            ids_to_summarize = [eid for eid, ch in profile_dict.items() if len(ch) >= max_profile_length and eid in self.user_hist_cache]
            prompts = []
            for eid in ids_to_summarize:
                memory = "- " + "\n- ".join([profile for profile in profile_dict[eid]])
                if eid in self.user_profile_ori:
                    memory = f"- {self.user_profile_ori[eid]}\n" + memory

                user_hist = []
                for i in self.user_hist_cache[eid][-30:]: # Only use the last 30 items
                    his_item_info = {
                        "title": self.item_description['title'][str(i)],
                        "description": self.item_memory[i][-1] if i in self.item_memory else ', '.join(json.loads(self.item_attributes[i])['Item key characteristics'])
                    }
                    user_hist.append(his_item_info)
                user_hist = "\n".join([json.dumps(his_item_info) for his_item_info in user_hist])
                prompt = self.user_summary_prompt + "\n INPUT \n" + "Item that user has bought:\n" + user_hist + "\nuser preference summaries arranged in chronological order:\n" + memory
                prompts.append(prompt.replace('\\"', ''))
            return ids_to_summarize, prompts
        
        def build_summary_item_prompt(profile_dict, entity_type):
            ids_to_summarize = [eid for eid, ch in profile_dict.items() if len(ch) >= max_profile_length]
            prompts = []
            for eid in ids_to_summarize:
                memory = "- " + "\n- ".join([profile for profile in profile_dict[eid]])
                item_info = json.loads(self.item_attributes[eid])
                item_info['description'] = self.item_description['description'][str(eid)]
                item_info['target_customer_preferences'] = memory
                prompt = self.item_summary_prompt + "\n INPUT \n" + json.dumps(item_info)
                prompts.append(prompt.replace('\\"', ''))
            return ids_to_summarize, prompts

        item_ids, item_prompts = build_summary_item_prompt(self.item_memory, "item")

        if len(item_prompts) > 0 and summary_item_mem:
            logger.info(f"Sample summary item prompts (item_id: {item_ids[0]}): {item_prompts[0]}")
            logger.info(f"Summarizing {len(item_prompts)} item collaborative filtering profiles.")
            item_summaries = self.batch_summarize_profiles(item_prompts, max_tokens=512, schema=ItemProfileResponse)
            for eid, summary in zip(item_ids, item_summaries):
                if summary != "None":
                    logger.info(f"Item {eid} CF profile summarized to: {summary}")
                    self.item_memory[eid] = [summary['ItemProfile']]
        else:
            logger.info("No item collaborative filtering profiles to summarize.")

        user_ids, user_prompts = build_summary_user_prompt(self.user_memory, "user")
        if len(user_prompts) > 0 and summary_user_mem:
            logger.info(f"Sample summary user prompts (user_id: {user_ids[0]}): {user_prompts[0]}")
            logger.info(f"Summarizing {len(user_prompts)} user collaborative filtering profiles.")
            user_summaries = self.batch_summarize_profiles(user_prompts, max_tokens=512, schema=UserProfileResponse)
            
            for eid, summary in zip(user_ids, user_summaries):
                logger.info(f"User {eid} CF profile summarized to: {summary}")
                if summary != "None":
                    self.user_memory[eid] = [summary['UserProfile']]
        else:
            logger.info("No user collaborative filtering profiles to summarize.")

    def batch_summarize_profiles(self, prompts: list, max_tokens: int = 512, chunk_size: int = 12, schema: BaseModel = UserProfileResponse) -> list:
        """
        Summarize a list of profile prompts using LLM in batches of chunk_size.
        Returns a list of summary strings.
        """
        if schema is not None:
            guided_decoding_params = GuidedDecodingParams(json=schema.model_json_schema())
        else:
            guided_decoding_params = None
        if not prompts:
            return []
        all_summaries = []
        for chunk_start in range(0, len(prompts), chunk_size):
            chunk = prompts[chunk_start:chunk_start + chunk_size]
            conversation = [
                [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": p[:32768]}
                ] for p in chunk
            ]
            outputs = self.llm.chat(conversation, sampling_params=SamplingParams(max_tokens=1024, guided_decoding=guided_decoding_params),
                                use_tqdm=False)
            for out in outputs:
                if schema is None:
                    all_summaries.append(out.outputs[0].text.strip())
                else:
                    try:
                        text = json.loads(out.outputs[0].text.strip())
                        all_summaries.append(text)
                    except Exception as e:
                        logger.error(f"Error summarizing profile: {e}")
                        all_summaries.append("None")
        return all_summaries