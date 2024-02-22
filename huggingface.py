from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

model_name = "microsoft/phi-2"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
model = AutoModelForCausalLM.from_pretrained(model_name, load_in_8bit=True, trust_remote_code=True)

inputs = tokenizer("Hello", return_tensors="pt", return_attention_mask=False)
outputs = model.generate(**inputs, max_length=20)
text = tokenizer.batch_decode(outputs)[0]
print(text)