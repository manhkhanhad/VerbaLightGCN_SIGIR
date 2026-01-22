from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, AutoModel, BitsAndBytesConfig
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
tqdm.pandas()
import torch
import os
import time
import gc
import json
print(torch.__version__)
import importlib.util
from datasets import Dataset
from torch.utils.data import DataLoader
from utils import * 
from transformers.pipelines.pt_utils import KeyDataset
from ollama import chat
from ollama import ChatResponse
import torch.nn as nn
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams
from pydantic import BaseModel
import argparse

def call_ollama(prompt, model_name):
    response: ChatResponse = chat(model=model_name, messages=[
        {
            'role': 'user',
            'content': prompt,
        },
        ])
    return response['message']['content']

def ollama_inference(config, prompt_dataset):
    prompt_dataset['reasoning'] = prompt_dataset['prompt'].progress_apply(lambda x: ollama_inference(x, config.model_name))
    prompt_dataset.to_parquet(config.save_path)
    return prompt_dataset

def vllm_inference(config, prompt_dataset):
    class Response(BaseModel):
        summarize_user_preference: str
        prediction: str
        reasoning: str

    dataset = Dataset.from_pandas(prompt_dataset)
    dataloader = DataLoader(dataset, batch_size=12, shuffle=False)
    model = LLM(model=config.model_name, max_model_len=32768, max_num_batched_tokens=32768 * 3,enable_chunked_prefill=True, max_num_seqs=12, tensor_parallel_size=1)

    prediction_df = {"user_id": [], "item_id": [], "prompt": [], "output": [], "label": []}
    with torch.no_grad():
        step = 0
        total_elements = len(dataloader)
        start_time = time.time()

        for batch in tqdm(dataloader):
            print(batch["prompt"][0])
            step += 1
            if (step + 1) % 10 == 0:
                elapsed_time = time.time() - start_time
                avg_time_per_loop = elapsed_time / (step + 1)

                remaining_elements = total_elements - (step + 1)
                estimated_time_remaining = avg_time_per_loop * remaining_elements
                print(f"Processed {step+1} steps. Estimated time remaining: {estimated_time_remaining / 60:.2f} minutes.")

            prompt = batch["prompt"]
            conservation = []
            for i in range(len(prompt)):
                conservation.append([{"role": "user", "content": prompt[i]}])

            sampling_params = SamplingParams(max_tokens=300, temperature=0, top_p=0.95, guided_decoding=GuidedDecodingParams(json=Response.model_json_schema()))

            outputs = model.chat(conservation, sampling_params, use_tqdm=False)
            for i, output in enumerate(outputs):
                prediction_df["user_id"].append(batch["user_id"][i].item())
                prediction_df["item_id"].append(batch["item_id"][i].item())
                prediction_df["prompt"].append(batch["prompt"][i])
                prediction_df["output"].append(output.outputs[0].text)
                prediction_df["label"].append(batch["label"][i])
                print(output.outputs[0].text)
                print("=" * 20)
    
    res_df = pd.DataFrame(prediction_df)
    res_df['output'] =  res_df.output.apply(lambda x: parse(x))
    res_df['pred'] =  res_df.output.apply(lambda x: convert_lable(x))
    res_df['reasoning'] =  res_df.output.apply(lambda x: construct_pred_text(x))
    res_df['output'] =  res_df.output.apply(lambda x: json.dumps(x))

    return res_df

def hf_inference(config, prompt_dataset):
    dataset = Dataset.from_pandas(prompt_dataset)
    dataloader = DataLoader(dataset, batch_size=12, shuffle=False)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.padding_side = "left"

    prediction_df = {"user_id": [], "item_id": [], "prompt": [], "output": []}
    with torch.no_grad():
        step = 0
        total_elements = len(dataloader)
        start_time = time.time()

        for batch in tqdm(dataloader):
            print(batch["prompt"][0])
            step += 1
            if (step + 1) % 10 == 0:
                elapsed_time = time.time() - start_time
                avg_time_per_loop = elapsed_time / (step + 1)
                remaining_elements = total_elements - (step + 1)
                estimated_time_remaining = avg_time_per_loop * remaining_elements
                print(f"Processed {step+1} steps. Estimated time remaining: {estimated_time_remaining / 60:.2f} minutes.")

            # Build chat prompts
            conservation = []
            for i in range(len(batch["prompt"])):
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": batch["prompt"][i]}
                ]
                messages = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                conservation.append(messages)

            # Tokenize batch
            inputs = tokenizer(
                conservation,
                return_tensors="pt",
                padding="max_length",
                max_length=2048,
                truncation=True
            ).to("cuda")

            # Generate
            outputs = model.generate(
                **inputs,
                max_new_tokens=300,
                temperature=0.02,
                top_p=0.9
            )

            # Extract only generated tokens
            gen_texts = tokenizer.batch_decode(
                outputs[:, inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )

            # Store results
            for i, decoded_text in enumerate(gen_texts):
                prediction_df["user_id"].append(batch["user_id"][i].item())
                prediction_df["item_id"].append(batch["item_id"][i].item())
                prediction_df["prompt"].append(batch["prompt"][i])
                prediction_df["output"].append(decoded_text)

                print(decoded_text)
                print("=" * 20)

    return pd.DataFrame(prediction_df)
    

def construct_neighbor_item_feature(item_list, item_attributes, item_memory):
    item_list_text = []
    for i in item_list:
        if i in item_attributes:
            if 'Item key characteristics' in json.loads(item_attributes[i]):
                item_list_text.append(f"Item {i}: {json.loads(item_attributes[i])['Title']} (Favorited features: {json.loads(item_attributes[i])['Item key characteristics']})")
            else:
                item_list_text.append(f"Item {i}: {json.loads(item_attributes[i])['Title']} (Favorited features: {json.loads(item_attributes[i])['feature']})")
    return "\n".join(item_list_text)

def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interaction_data_path", type=str, default="data/toys/test_data.parquet")
    parser.add_argument("--item_info_path", type=str, default="data/toys/item_kw.json")
    parser.add_argument("--ckpt", type=str, default="ckpt/toys_data_v1_tune/prompt_v15_VLGCN_VOT_run5/prompt_v15_VLGCN_VOT_run5-epoch=04-val_f1=0.00.ckpt")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--model_type", type=str, default="vllm")
    parser.add_argument("--prompt_path", type=str, default="prompts/toys")
    parser.add_argument("--save_path", type=str, default="./outputs/toys/prediction.parquet")
    return parser.parse_args()

def main():
    config = parse_arg()
    # Load data
    df = pd.read_parquet(config.interaction_data_path)
    # df = df[df.label == True]
    item_attributes = load_item_attributes(config.item_info_path)
    ckpt = torch.load(config.ckpt, weights_only=False)
    user_memory = ckpt['user_profile']
    item_memory = ckpt['item_profile']

    # Load prompt
    with open(config.prompt_path, 'r') as f:
        prompt_template = f.read()
    # Creare prompt_dataset
    data_dict = {"user_id": [], "item_id": [], "prompt": [], "label": []}
    for idx, row in tqdm(df.iterrows()):
        user_id = row['user_id']
        user_hist_item = [i for i in row['neighbor_item'] if i != 0]
        target_item_id = row['item_id']
        label = "LIKE" if row['label'] == 1 else "DISLIKE"
        user_history = construct_neighbor_item_feature(user_hist_item, item_attributes, item_memory)
        user_profile = user_memory.get(user_id, [""])[-1]
        item_profile = item_memory.get(target_item_id, [""])[-1]
        
        #Item infor KW
        if 'Item key characteristics' in json.loads(item_attributes[target_item_id]):
            item_information = json.loads(item_attributes[target_item_id])['Title'] + "(Item key characteristics: " + ", ".join(json.loads(item_attributes[target_item_id])['Item key characteristics']) + ")"
        else:
            item_information = json.loads(item_attributes[target_item_id])['item_name'] + "(Item key characteristics: " + ", ".join(json.loads(item_attributes[target_item_id])['feature']) + ")"

        target_item_title = json.loads(item_attributes[target_item_id])['Title']
        prompt = prompt_template.replace("{{{user_hist}}}", str(user_history)).replace("{{{user_profile}}}", "User preferences features:" + str(user_profile)).replace("{{{item_profile}}}", str(item_profile)).replace("{{{item_information}}}", item_information).replace("{{{target_item_title}}}", target_item_title)
        # ADD: user also bought item
                
        data_dict["user_id"].append(row['user_id'])
        data_dict["item_id"].append(target_item_id)
        data_dict["prompt"].append(prompt)
        data_dict["label"].append(label)
    prompt_dataset = pd.DataFrame(data_dict)
    # inferences using Ollama
    if config.model_type == "ollama":
        prompt_dataset = ollama_inference(config, prompt_dataset)
    elif config.model_type == "hf":
        prompt_dataset = hf_inference(config, prompt_dataset)
    elif config.model_type == "vllm":
        prompt_dataset = vllm_inference(config, prompt_dataset)
    prompt_dataset.to_parquet(config.save_path)


    print("============= EVALUATION =============")
    print("Accuracy: ", len(prompt_dataset[prompt_dataset.label == prompt_dataset.pred]) / len(prompt_dataset))
    print("Predict distribution: ", prompt_dataset.pred.value_counts())

if __name__ == "__main__":
    main()