#python -m torch.distributed.launch --nproc_per_node=2 testing.py
#!pip install transformers
#!pip install git+https://github.com/huggingface/accelerate
#!pip install bitsandbytes
#!pip install peft
#!pip install datasets
import torch
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import os
from accelerate import PartialState

# If you are using 4000 series gpu
os.environ["NCCL_P2P_DISABLE"]="1"
os.environ["NCCL_IB_DISABLE"]="1"
multi_gpu = True

model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
max_token = 256

# Instantiate the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
tokenizer.add_special_tokens({'pad_token': '[PAD]'}) # mixtral does not have default padding token
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

# Tokenization functions
def tokenize_function(row):
    return tokenizer(row["dialog"])

def is_shorter_than_max_token(row):
    """
    Return if a given row has more than max_token number of tokens
    """
    return len(row['input_ids']) <= max_token

def split_conversation(conversation): 
    """
    Split conversation into turns
    """
    return [conversation[:i+2] for i in range(0, len(conversation), 2) if i+2 <= len(conversation)]

def format_conversation(conversation: list[str]) -> str:
    formatted_conversation = ""
    
    # Check if the conversation has more than two turns
    if len(conversation) > 2:
        # Process all but the last two turns
        for i in range(len(conversation) - 2):
            if i % 2 == 0:
                formatted_conversation += "<Past User>" + conversation[i] + "\n"
            else:
                formatted_conversation += "<Past Assistant>" + conversation[i] + "\n"
    
    # Process the last two turns
    if len(conversation) >= 2:
        formatted_conversation += "<User>" + conversation[-2] + "\n"
        formatted_conversation += "<Assistant>" + conversation[-1]
    
    return formatted_conversation

def convert_to_conversation(row):
    conversation_list = row["dialog"]
    
    conversation = format_conversation(conversation_list)
    conversation += "</s>"
    return {"dialog": conversation.strip()}

# Load and tokenize dataset
dataset = load_dataset("daily_dialog")

# Split into multiple turns of conversation
split_dataset = dataset.map(lambda x: {'dialog': split_conversation(x['dialog'])})

# Flatten dataset
flatten_dataset_train = [item for row in split_dataset["train"]["dialog"] for item in row]
flatten_dataset_valid = [item for row in split_dataset["validation"]["dialog"] for item in row]
flatten_dataset_test = [item for row in split_dataset["test"]["dialog"] for item in row]

flatten_dataset_train = Dataset.from_dict({'dialog': flatten_dataset_train})
flatten_dataset_valid = Dataset.from_dict({'dialog': flatten_dataset_valid})
flatten_dataset_test = Dataset.from_dict({'dialog': flatten_dataset_test})

dataset = DatasetDict({
    'train': flatten_dataset_train,
    'validation': flatten_dataset_valid,
    'test': flatten_dataset_test
})

# Change to conversational manner
dataset = dataset.map(convert_to_conversation)

# Tokenize dataset
dataset = dataset.map(tokenize_function)

# Filter conversation longer than tok`en limit
dataset = dataset.filter(is_shorter_than_max_token)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
    
if multi_gpu:
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map={"": PartialState().process_index}, quantization_config=bnb_config)
else:
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", quantization_config=bnb_config)
model.resize_token_embeddings(len(tokenizer))
model = prepare_model_for_kbit_training(model)

# LORA config
config = LoraConfig(
    r=16, 
    lora_alpha=32, #alpha scaling
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, config)

if multi_gpu:
    training_args = TrainingArguments(
        output_dir="output_dir",
        per_device_train_batch_size=10,
        gradient_accumulation_steps=10,
        num_train_epochs=1,
        learning_rate=1e-4,
        evaluation_strategy="steps",
        eval_steps=0.25,
        warmup_steps=50,
        weight_decay=1e-3,
        optim="paged_adamw_32bit",
        group_by_length=True,
        lr_scheduler_type="cosine",
        ddp_find_unused_parameters=False
    )
else:
    training_args = TrainingArguments(
        output_dir="output_dir",
        per_device_train_batch_size=10,
        gradient_accumulation_steps=10,
        num_train_epochs=1,
        learning_rate=1e-4,
        evaluation_strategy="steps",
        eval_steps=0.25,
        warmup_steps=50,
        weight_decay=1e-3,
        optim="paged_adamw_32bit",
        group_by_length=True,
        lr_scheduler_type="cosine",
    )

trainer = Trainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    args=training_args,
    data_collator=data_collator,
    tokenizer=tokenizer
)

trainer.train()

# Saving model
trainer.save_model("ChadGPT")

# Loading model
from peft import PeftModel, PeftConfig
peft_model_id = "ChadGPT"
config = PeftConfig.from_pretrained(peft_model_id)
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, return_dict=True, load_in_8bit=True, device_map='auto')
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
model.resize_token_embeddings(len(tokenizer) + 1)

# Load the Lora model
model = PeftModel.from_pretrained(model, peft_model_id)