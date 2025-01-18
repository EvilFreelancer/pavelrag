import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

checkpoint = "jinaai/reader-lm-1.5b"

dtype = torch.bfloat16
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=dtype,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

device = "cuda"  # for GPU usage or "cpu" for CPU usage
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(
    checkpoint,
    quantization_config=quantization_config,
).to(device)

# example html content
with open("test.html", "r") as f:
    html_content = f.read()

messages = [{"role": "user", "content": html_content}]
input_text = tokenizer.apply_chat_template(messages, tokenize=False)

# print(input_text)

inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)
outputs = model.generate(inputs, max_new_tokens=4096, temperature=0, do_sample=False, repetition_penalty=1.08)

print(tokenizer.decode(outputs[0]))
