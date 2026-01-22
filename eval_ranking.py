import sys
sys.path.append("../")
import os
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams
import torch
import pandas as pd
import json
from tqdm import tqdm
import random

from transformers import AutoModelForCausalLM, AutoTokenizer
from torchmetrics.classification import MulticlassF1Score, MulticlassAccuracy, MulticlassAUROC
from data.VerbaLightGCN_data_module import VerbaLightGCNDataModule
import time
import numpy as np
from datasets import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import argparse

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)
set_seed()

from pydantic import BaseModel
class aggProfileResponse(BaseModel):
    SemanticView: str
    CollaborativeView: str


def load_item_attributes(item_info_path):
    if "kw" in item_info_path:
        with open(item_info_path, 'r') as f:
            item_attributes_data = json.load(f)
        item_attributes = {}
        for k,v in item_attributes_data.items():
            item_attributes[int(k)] = {"Title": v['Title'] if 'Title' in v else v['item_name'], "Item key characteristics": v['feature'] if 'feature' in v else v['features']}
    elif "desciption" in item_info_path:
        with open(item_info_path, 'r') as f:
            item_description = json.load(f)
        if 'title' in item_description:
            item_attributes = {}
            for item_id in list(item_description['title'].keys()):
                attr = {
                    "Title": item_description['title'][item_id].replace("&amp;", "").replace("&quot;", "").replace("&apos;", "").replace("&lt;", "").replace("&gt;", ""),
                    "description": item_description['description'][item_id].replace("&amp;", "").replace("&quot;", "").replace("&apos;", "").replace("&lt;", "").replace("&gt;", ""),
                }
                if item_description['brand'][item_id] != '':
                    attr['brand'] = item_description['brand'][item_id]
                item_id = int(item_id)
                item_attributes[item_id] = attr
        else:
            item_attributes = {int(k): v for k, v in item_description.items()}
    return item_attributes

def vllm_batch_inferences(model, df, col_name, structure=None, batch_size = 32):
    response = []
    for idx in tqdm(range(0, len(df), batch_size)):
        batch = df[idx: idx + batch_size][col_name].tolist()
        conversation = []
        for p in batch:
            conversation.append([{"role": "system", "content": "You are a helpful assistant."},
                                {"role": "user", "content": p}])
            
        if structure is not None:
            guided_decoding_params = GuidedDecodingParams(json=structure.model_json_schema())
        else:
            guided_decoding_params = None
        outputs = model.chat(conversation, sampling_params=SamplingParams(max_tokens=4096, guided_decoding=guided_decoding_params, temperature=0, top_p=0.95),
                            use_tqdm=False)

        if structure is not None:
            for idx, output in enumerate(outputs):
                try:
                    res = json.loads(output.outputs[0].text)
                    response.append(res)
                except json.JSONDecodeError:
                    response.append(None)
        else:
            response.extend([i.outputs[0].text for i in outputs])
    df['response'] = response
    return df


def generate_item_prompt(row, item_attributes, item_profile, user_profile, agg_item_prompt):
    item_information = json.dumps(item_attributes[row['item_id']])
    profile = json.dumps(item_profile[row['item_id']][0])
    user_like_item = "\n".join([f"User {i}: " + json.dumps(user_profile.get(i, [{"Like": [], "Dislike": []}])[0]) for i in row['user_id']])
    return agg_item_prompt.replace("{{{item_information}}}", item_information)\
                            .replace("{{{item_profile}}}", profile)\
                            .replace("{{{user_like_item}}}", user_like_item).replace('\\"', '')

def generate_user_prompt(row, user_profile, item_attributes, agg_item_profile, agg_user_prompt):
    profile = json.dumps(user_profile.get(row['user_id'], [{"Like": [], "Dislike": []}])[0])
    user_hist = "\n".join([f"Item {i}: {item_attributes[i]['Title']} (profile: {agg_item_profile[i]})" for i in row['item_id']])
    return agg_user_prompt.replace("{{{user_profile}}}", profile)\
                            .replace("{{{user_hist}}}", user_hist).replace('\\"', '')

def aggregate_profile(config, model, df, item_df, item_attributes, item_profile, user_profile, agg_item_prompt, agg_user_prompt):
    item_df['prompt'] = item_df.apply(lambda x: generate_item_prompt(x, item_attributes, item_profile, user_profile, agg_item_prompt), axis=1)
    item_df = vllm_batch_inferences(model, item_df, 'prompt', aggProfileResponse)
    agg_item_profile = dict(zip(item_df.item_id, item_df.response))

    df['prompt'] = df.apply(lambda x: generate_user_prompt(x, user_profile, item_attributes, agg_item_profile, agg_user_prompt), axis=1)

    df = vllm_batch_inferences(model, df, 'prompt', aggProfileResponse)
    agg_user_profile = dict(zip(df.user_id, df.response))

    return agg_item_profile, agg_user_profile

def recall_at_k(y_true, y_score, k):
    top_k_indices = np.argsort(-y_score, axis=1)[:, :k]
    recalls = []
    for i in range(y_true.shape[0]):
        try:
            true_labels = y_true[i, top_k_indices[i]]
        except:
            print(y_true.shape, y_score.shape, top_k_indices.shape)
            exit(0)
        recalls.append(np.sum(true_labels) / np.sum(y_true[i]))
    return np.mean(recalls)

def mrr_at_k(y_true, y_score, k):
    mrrs = []
    for i in range(y_true.shape[0]):
        sorted_indices = np.argsort(-y_score[i])
        rank = 0
        for j in range(k):
            if y_true[i, sorted_indices[j]] == 1:
                rank = j + 1
                break
        mrrs.append(1.0 / rank if rank > 0 else 0)
    return np.mean(mrrs)

def ndcg_at_k(y_true, y_score, k):

    def dcg_at_k(r, k):
        """计算DCG@k的值"""
        r = np.asarray(r)[:k]
        return np.sum(r / np.log2(np.arange(2, r.size + 2)))
    
    ndcgs = []
    for i in range(y_true.shape[0]):
        sorted_indices = np.argsort(-y_score[i])
        predicted_relevance = y_true[i, sorted_indices][:k]
        dcg_value = dcg_at_k(predicted_relevance, k)
        
        sorted_true_relevance = np.sort(y_true[i])[::-1]  # 从大到小排序
        idcg_value = dcg_at_k(sorted_true_relevance, k)
        
        ndcgs.append(dcg_value / (idcg_value + 1e-7))  # 防止除以0
    
    return np.mean(ndcgs)

def get_results(tokenizer, model, dataloader):

    yes_id = tokenizer.convert_tokens_to_ids("Yes")
    no_id = tokenizer.convert_tokens_to_ids("No")

    yes_logits, no_logits, labels = [], [], []
    output_logits = []
    decoded_texts = []
    all_results_over_time = []
    steps = [] 
    with torch.no_grad():
        step = 0
        total_elements = len(dataloader)
        start_time = time.time()

        for batch in tqdm(dataloader):
            step += 1
            if (step + 1) % 10 == 0:
                elapsed_time = time.time() - start_time
                avg_time_per_loop = elapsed_time / (step + 1)
                remaining_elements = total_elements - (step + 1)
                est_time_remaining = avg_time_per_loop * remaining_elements
                print(f"Processed {step+1} steps. Estimated time remaining: {est_time_remaining / 60:.2f} minutes.")

            labels.append(batch["label"])
            batch_size = len(batch["label"])

            # Forward pass with deterministic decoding
            outputs = model.generate(
                    batch['input_ids'],
                    max_new_tokens=10,
                    num_return_sequences=1,
                    do_sample=False,
                    output_scores=True,  # Set to True to get the logits
                    return_dict_in_generate=True, # Returns a GenerationOutput object for easier access
                    output_logits=True,
                    attention_mask = batch['attention_mask'])

            logits = outputs.logits  # tuple of tensors
            sequences = outputs.sequences
            for i, seq in enumerate(outputs.sequences):
                gen_tokens = seq[batch['last_position'][i]:]  # strip off input part
                decoded_texts.append(tokenizer.decode(gen_tokens, skip_special_tokens=True))


            for i in range(batch_size):
                # take logits of the first step
                yes_logits.append(logits[0][i, yes_id].item())
                no_logits.append(logits[0][i, no_id].item())

            if step % 10 == 0:
                results, _, _ = _compute_metric(yes_logits, no_logits, labels, is_intermediate = True)
                all_results_over_time.append(results)
                steps.append(step)
                print(results)
        
    results, output_logits, no_probs = _compute_metric(yes_logits, no_logits, labels)
    print("FINAL METRIC:", results)
    return results, output_logits, no_probs, decoded_texts
    
def _compute_metric(yes_logits, no_logits, labels, is_intermediate = False):
    if is_intermediate:
        tested_user = len(yes_logits) // 20
        yes_logits = yes_logits[:tested_user * 20]
        no_logits = no_logits[:tested_user * 20]
        labels = labels[:tested_user * 20]
    
    # Convert logits → probabilities
    yes_logits = torch.tensor(yes_logits).reshape(-1)
    no_logits = torch.tensor(no_logits).reshape(-1)
    yes_probs = F.softmax(torch.stack([yes_logits, no_logits], dim=-1), dim=-1)[:, 0].cpu().numpy()
    no_probs = F.softmax(torch.stack([yes_logits, no_logits], dim=-1), dim=-1)[:, 1].cpu().numpy()

    labels = torch.concat(labels).cpu().numpy()
    y_true = labels.reshape(-1, 20)
    y_score = yes_probs.reshape(-1, 20)

    output_logits = yes_logits.cpu().numpy()

    k_list = [1, 5, 10, 15, 20]
    results = {}
    for k in k_list:
        results[f"recall@{k}"] = recall_at_k(y_true, y_score, k)
        # results[f"mrr@{k}"] = mrr_at_k(y_true, y_score, k)
        results[f"ndcg@{k}"] = ndcg_at_k(y_true, y_score, k)

    return results, yes_probs.reshape(-1), no_probs

def collate_fn(batch):
    input_ids = [torch.tensor(item['input_ids'][0]) for item in batch]
    label = [item['label'] for item in batch]
    last_position = [item['last_position'] for item in batch]
    attention_mask = [torch.tensor(item['attention_mask'][0]) for item in batch]
    return {
        'input_ids': torch.stack(input_ids).to("cuda"),
        'label': torch.tensor(label).to("cuda"),
        'last_position': torch.tensor(last_position).to("cuda"),
        'attention_mask': torch.stack(attention_mask).to("cuda")
    }

def encode(row, tokenizer, max_input_length = 1024):
    temp_data = {}
    # text = row['prompt']

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": row['prompt']}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    input_ids = tokenizer(text, return_tensors="pt", padding='max_length', max_length=max_input_length, truncation=True)
    # input_ids = tokenizer.encode(text, return_tensors="pt", padding='max_length', truncation=True)
    last_position = max_input_length #min(len(tokenizer.encode(text)) - 1, 2047)
    temp_data['input_ids'] = input_ids['input_ids']
    temp_data['label'] = row['label']
    temp_data['last_position'] = last_position
    temp_data['attention_mask'] = input_ids['attention_mask']
    return temp_data

def generate_ranking_prompt(row, tokenizer, ranking_prompt_template, agg_user_profile, agg_item_profile, item_attributes, ranking_prompt):
    user_preference = agg_user_profile[str(row['user_id'])]
    history_items = [i for i in row['neighbor_item'] if i != 0]
    history_items_text = '\n'.join(f"{i}. Title: {item_attributes[i]['Title']}. Profile: {item_attributes[i]['Item key characteristics']}" for i in history_items)
    candidate_text = f"Title: {item_attributes[row['candidate']]['Title']}. Profile: {agg_item_profile[str(row['candidate'])]}"
    prompt = ranking_prompt_template.replace("{{{user_preference}}}", json.dumps(user_preference))\
                                    .replace("{{{history_items}}}", history_items_text)\
                                    .replace("{{{next_item}}}", candidate_text)
    return prompt

def aggregate_stage(config):
    model = LLM(model="Qwen/Qwen2.5-7B-Instruct", max_model_len=32768, max_num_batched_tokens=32768 * 3, enable_chunked_prefill=True, max_num_seqs=12)

    # LOAD CKPT AND DATA
    ckpt = torch.load(config["ckpt_path"], weights_only = False)
    user_profile = ckpt['user_profile']
    item_profile = ckpt['item_profile']
    df = pd.read_parquet(config["interaction_path"])
    item_df = df[['user_id','train']].rename(columns = {'train':'item_id'}).explode('item_id').groupby('item_id').agg({'user_id': list}).reset_index()
    with open(os.path.join(config["prompt_path"], "aggreate_item_profile.txt"), 'r') as f:
        agg_item_prompt = f.read()
    with open(os.path.join(config["prompt_path"], "aggreate_user_profile.txt"), 'r') as f:
        agg_user_prompt = f.read()
    item_attributes = load_item_attributes(config["item_info_path"])

def ranking(config):
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    # AGGREGATE PROFILE
    print("LOAD PROFILE FROM ", config["save_agg_profile"])
    with open(config["save_agg_profile"], 'r') as f:
        agg_profile = json.load(f)
    agg_item_profile = agg_profile["agg_item_profile"]
    agg_user_profile = agg_profile["agg_user_profile"]
        
    # RANKING
    with open(os.path.join(config["prompt_path"], "pointwise_ranking_template.txt"), 'r') as f:
        ranking_prompt = f.read()

    data_module = VerbaLightGCNDataModule(
            interaction_path=config["interaction_path"],
            item_info_path=config["item_info_path"],
            user_profile_path=None,
            item_profile_path=None,
            batch_size=48,
            num_workers=4,
            ranking_mode=True
        )
    data_loader = data_module.test_dataloader()

    test_df = {
        "user_id": [],
        "neighbor_item": [],
        "pos_index": [],
        "candidate": []
    }
    for batch in data_loader:
        for i in range(len(batch["user_id"])):
            test_df['user_id'].append(batch["user_id"][i].item())
            test_df['neighbor_item'].append(batch["neighbor_item"][i].cpu().tolist())
            test_df['pos_index'].append(batch["pos_index"][i].item())
            test_df['candidate'].append(batch["candidate"][i].cpu().tolist())
    test_df = pd.DataFrame.from_dict(test_df)
    test_df = test_df

    item_attributes = load_item_attributes(config["item_info_path"])
    test_df['pos_item_id'] = test_df.apply(lambda row: row['candidate'][row['pos_index'] - 1], axis=1)
    infer_df = test_df.explode('candidate')
    infer_df['label'] = infer_df.apply(lambda row: row['pos_item_id'] == row['candidate'], axis=1)
    infer_df['prompt'] = infer_df.apply(lambda row: generate_ranking_prompt(row, tokenizer, ranking_prompt, agg_user_profile, agg_item_profile, item_attributes, ranking_prompt), axis=1)

    print("SAMPLE PROMPT:")
    print(infer_df['prompt'].iloc[99])

    test_dataset = Dataset.from_pandas(infer_df)
    test_dataset = test_dataset.map(encode, fn_kwargs={'tokenizer': tokenizer}, batched=False, num_proc=16)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    results, output_logits, no_probs, decoded_texts = get_results(tokenizer, model, test_dataloader)
    print(results)

    infer_df['pred_logit'] = output_logits
    infer_df['no_probs'] = no_probs
    infer_df['decoded_texts'] = decoded_texts
    infer_df = infer_df.sort_values(by='pred_logit', ascending=False)
    test_df = test_df.merge(infer_df.groupby('user_id').agg(rank_pred = ('candidate', list), score_pred = ('pred_logit', list)).reset_index(), on='user_id')
    test_df.to_parquet(config["save_ranking_result_path"])


if __name__ == "__main__":

    #Toys v4
    config = {
        "ckpt_path": "ckpt/toys/VerbaLightGCN_k3/VerbaLightGCN_k3-epoch=04-val_f1=0.00.ckpt",
        "interaction_path": "dataset/Toys_v1_small_fix/small_data_interaction.parquet",
        "item_info_path": "dataset/Toys_v1_small_fix/item_kw.json",
        "prompt_path": "prompts/toys",
    }


    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--mode", type=str, default="aggregate_profile")
    args = parser.parse_args()

    config["save_agg_profile"] = os.path.join( os.path.dirname(config['ckpt_path']), "agg_profile.json")
    config["save_ranking_result_path"] = os.path.join( os.path.dirname(config['ckpt_path']), "ranking_result.parquet")
    print(config)
    if args.mode == "aggregate_profile":
        aggregate_stage(config)
    elif args.mode == "ranking":
        ranking(config)
    else:
        raise ValueError(f"Invalid model name: {args.model_name}")
