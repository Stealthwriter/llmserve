import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import huggingface_hub  
import uvicorn


# Get model name and Hugging Face token from environment variables
model_name = os.getenv("MODEL")
hf_token = os.getenv("HF_TOKEN")

print(hf_token)
# Ensure the environment variables are set
if not model_name or not hf_token:
    raise ValueError("Environment variables MODEL and HF_TOKEN must be set")

huggingface_hub.login(hf_token)

app = FastAPI()

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True,
    padding_side='left'
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map={"": "cuda:0"},
    trust_remote_code=True,
    attn_implementation="flash_attention_2",
    token=True,
)
model.cuda().eval()

model.generation_config.max_new_tokens = 256
model.generation_config.do_sample = True

class BatchSentenceRequest(BaseModel):
    sentences: list[str]
    temperature: float

def generate_responses(sentences: list[str], temperature: float) -> list[str]:
    messages_list = [
        [{"role": "user", "content": sentence}] for sentence in sentences
    ]

    inputs = tokenizer.apply_chat_template(
        messages_list,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        padding=True,
        truncation=True,
    ).to("cuda")

    model.generation_config.temperature = temperature

    tokens = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        renormalize_logits=True
    )

    # Slice the tokens to remove the initial part corresponding to the input sequence for each batch
    tokens = tokens[:, inputs["input_ids"].shape[1]:]

    answers = tokenizer.batch_decode(tokens, skip_special_tokens=True)
    return answers

@app.post("/processBatch")
async def process_batch(request: BatchSentenceRequest):
    responses = generate_responses(request.sentences, request.temperature)
   
    return {"responses": responses}
