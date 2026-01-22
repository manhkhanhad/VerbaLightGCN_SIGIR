import json
import torch
from transformers import DataCollatorForLanguageModeling # Keep this import

def load_item_attributes(item_info_path):
    if "kw" in item_info_path:
        with open(item_info_path, 'r') as f:
            item_attributes_data = json.load(f)
        item_attributes = {}
        for k,v in item_attributes_data.items():
            item_attributes[int(k)] = json.dumps({"Title": v['Title'] if 'Title' in v else v['item_name'], "Item key characteristics": v['feature']})
    elif "desciption" in item_info_path:
        with open(item_info_path, 'r') as f:
            item_description = json.load(f)
        item_attributes = {}
        for item_id in list(item_description['title'].keys()):
            attr = {
                "Title": item_description['title'][item_id].replace("&amp;", "").replace("&quot;", "").replace("&apos;", "").replace("&lt;", "").replace("&gt;", ""),
                "description": item_description['description'][item_id].replace("&amp;", "").replace("&quot;", "").replace("&apos;", "").replace("&lt;", "").replace("&gt;", ""),
            }
            if 'brand' in item_description: 
                if item_description['brand'][item_id] != '':
                    attr['brand'] = item_description['brand'][item_id]
            if 'category' in item_description:
                if len(item_description['category'][item_id]) > 0:
                    attr['category'] = item_description['category'][item_id]
            item_id = int(item_id)
            item_attributes[item_id] = json.dumps(attr)
    return item_attributes




def create_data_collator_for_chat_training(tokenizer):
    """
    Creates a data collator that masks labels to -100 except for the assistant's response.
    This version is more robust for chat-formatted data by directly using tokenized sequences
    for identifying turns.

    Args:
        tokenizer: The tokenizer used for encoding the data, expecting chat template support.

    Returns:
        A DataCollatorForLanguageModeling instance with custom label masking.
    """
    class CustomDataCollator(DataCollatorForLanguageModeling):
        def torch_call(self, features):
            # Pad inputs to the longest sequence in the batch
            # padding="max_length" is generally good if you've already set it in tokenization
            # Otherwise, padding=True for dynamic padding to the longest in the batch
            batch = self.tokenizer.pad(
                features,
                return_tensors="pt",
                padding=True, # Dynamically pad to the longest sequence in the batch
            )

            # batch = {key: torch.stack([f[key] for f in features]) for key in features[0].keys()} If you don't need padding use this line.

            labels = batch["input_ids"].clone()

            # Define the exact token sequences for start/end of assistant's turn
            # These are specific to Qwen's chat template tokenization
            # <|im_start|>assistant\n  -> [151644, 872, 198]
            # <|im_end|>                  -> [151645]

            # Use tokenizer.encode to get the exact token IDs for the prompts
            # This is more reliable than guessing individual token IDs for subwords/special chars
            assistant_prompt_ids = self.tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False, return_tensors="pt")[0]
            im_end_token_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>") # This one is a known special token

            for i, input_ids_tensor in enumerate(batch["input_ids"]):
                # Initialize all labels to -100 (ignore by default)
                labels[i, :] = -100

                # Convert input_ids_tensor to a standard Python list for easier slicing/comparison
                # This makes the search part simpler if you prefer list operations.
                # However, for efficiency with large tensors, keeping it as tensor is better.
                # Let's stick to tensor operations for robustness and performance.

                # Find the start of the assistant's response within the current sequence
                assistant_start_idx = -1

                # Iterate through the input_ids to find the sequence `assistant_prompt_ids`
                # We need to use tensor comparison here.
                len_prompt = len(assistant_prompt_ids)
                for j in range(len(input_ids_tensor) - len_prompt + 1):
                    if torch.equal(input_ids_tensor[j : j + len_prompt], assistant_prompt_ids.to(input_ids_tensor.device)):
                        assistant_start_idx = j + len_prompt # Start of the actual response
                        break # Found the first occurrence of assistant turn

                # Find the corresponding end of the assistant's turn (<|im_end|>)
                assistant_end_idx = -1
                if assistant_start_idx != -1:
                    # Search for im_end_token_id ONLY AFTER the assistant's prompt start
                    for j in range(assistant_start_idx, len(input_ids_tensor)):
                        if input_ids_tensor[j] == im_end_token_id:
                            assistant_end_idx = j
                            break

                # If the assistant's response part is successfully identified, unmask it
                # Ensure start is before end and both are valid indices
                if (assistant_start_idx != -1 and
                    assistant_end_idx != -1 and
                    assistant_start_idx < assistant_end_idx):

                    labels[i, assistant_start_idx : assistant_end_idx] = input_ids_tensor[assistant_start_idx : assistant_end_idx]

                # Handle cases where the sequence is truncated and assistant_end_idx might be beyond max_length
                # In such cases, if assistant_start_idx is found, but end is not (due to truncation),
                # we should still train on the available part of the assistant's response until max_length.
                elif assistant_start_idx != -1 and assistant_end_idx == -1:
                     labels[i, assistant_start_idx:] = input_ids_tensor[assistant_start_idx:]


            batch["labels"] = labels
            return batch

    return CustomDataCollator(tokenizer=tokenizer, mlm=False)

def parse(x):
    try: 
        json_data =  json.loads(x)
        return json_data
    except:
        return x

def convert_lable(x):
    if isinstance(x, dict):
        pred = x['prediction']
        if pred == "Yes":
            return "LIKE"
        elif pred == "No":
            return "DISLIKE"
        else:
            return "n/a"
    else:
        return "n/a"

def construct_pred_text(x):
    if isinstance(x, dict):
        return f"Prediction: {x['prediction']}\n Summarize User Preferences: {x['summarize_user_preference']}\n Explanation: {x['reasoning']}"
    else:
        return "n/a"