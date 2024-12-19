from torch import nn
import transformers
from transformers.models.llama.modeling_llama import *
from transformers import (AutoTokenizer, AutoModelForCausalLM, AutoConfig, LlamaForCausalLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments)
from datasets import load_dataset
import huggingface_hub
from huggingface_hub import login
import wandb
from huggingface_hub import create_repo, HfApi
import datasets
print("Libraries Imported Successfully! \n")

class BitLinear(nn.Linear):
    def forward(self, x):
        w = self.weight # a weight tensor with shape [d, k]
        x = x.to(w.device)
        RMSNorm = LlamaRMSNorm(x.shape[-1]).to(w.device)
        x_norm = RMSNorm(x)
        # A trick for implementing Straight−Through−Estimator (STE) using detach()
        x_quant = x_norm + (activation_quant(x_norm) - x_norm).detach()
        w_quant = w + (weight_quant(w) - w).detach()
        y = F.linear(x_quant, w_quant)
        return y

def activation_quant(x):
    scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
    y = (x * scale).round().clamp_(-128, 127) / scale
    return y

def weight_quant(w):
    scale = 1.0 / w.abs().mean().clamp_(min=1e-5)
    u = (w * scale).round().clamp_(-1, 1) / scale
    return u
    
def convert_to_bitnet(model, copy_weights):
    for name, module in model.named_modules():
        # Replace linear layers with BitNet
        if isinstance(module, LlamaSdpaAttention) or isinstance(module, LlamaMLP):
            for child_name, child_module in module.named_children():
                if isinstance(child_module, nn.Linear):
                    bitlinear = BitLinear(child_module.in_features, child_module.out_features, child_module.bias is not None).to(device="cuda:0")
                    if copy_weights:
                        bitlinear.weight = child_module.weight
                        if child_module.bias is not None:
                            bitlinear.bias = child_module.bias
                    setattr(module, child_name, bitlinear)
        # Remove redundant input_layernorms
        elif isinstance(module, LlamaDecoderLayer):
            for child_name, child_module in module.named_children():
                if isinstance(child_module, LlamaRMSNorm) and child_name == "input_layernorm":
                    setattr(module, child_name, nn.Identity().to(device="cuda:0"))

def tokenize(element):
    outputs = tokenizer(
        element["content"],
        truncation=False,
        max_length=CONTEXT_LENGTH,
        return_overflowing_tokens=True,
        return_length=True,
    )
    # Combine all tokens
    combined = []
    for tokenized_doc in outputs['input_ids']:
        combined += tokenized_doc + [tokenizer.eos_token_id]
    # Chunk
    input_batch = []
    for i in range(0, len(combined) - CONTEXT_LENGTH, CONTEXT_LENGTH):
        input_batch.append(combined[i:i+CONTEXT_LENGTH])
    return {"input_ids": input_batch}


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device} \n")

model_name = "ai4bharat/Airavata"

print("Loading Model....")
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(device)
print("Model Loaded Successfully! \n")

MODEL_CONFIG = model_name
HEADS = 32
DIMENSIONS = 4096
LAYERS = 32
INTERMEDIATE_SIZE= 11008
CONTEXT_LENGTH = 4096
NEW_MODEL = "Bitnet-Airavata-7B"

HF_TOKEN = "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"
login(token=HF_TOKEN)

print("Loading Dataset....")
ds = load_dataset("Hindi-data-hub/odaigen_hindi_pre_trained_sp")
print("Dataset Loaded Successfully! \n")

WANDB_TOKEN = "c38a73d44262eca3349959c90973a0247ff28afb"
wandb.login(key=WANDB_TOKEN)

DATASET = ds
BATCH_SIZE = 2000000
LEARNING_RATE = 1.2e-4
EPOCHS = 2

print("Tokenizing Data....")
tokenized_data = ds.map(
    tokenize, batched=True, remove_columns=ds["train"].column_names,
)
print("Data Tokenized Successfully! \n")

total_tokens = tokenized_data['train'].num_rows * CONTEXT_LENGTH
print(f"Training on {total_tokens:_} tokens")

config = AutoConfig.from_pretrained(
    MODEL_CONFIG,
    vocab_size=len(tokenizer),
    n_ctx=CONTEXT_LENGTH,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
)

config.hidden_size = DIMENSIONS
config.max_position_embeddings = DIMENSIONS
config.num_attention_heads = HEADS
config.num_hidden_layers = LAYERS
config.num_key_value_heads = HEADS
config.intermediate_size = INTERMEDIATE_SIZE

model = LlamaForCausalLM(config)
convert_to_bitnet(model, copy_weights=False)

model_size = sum(t.numel() for t in model.parameters())
print(f"Model size: {model_size/1000**2:.1f}M parameters")

tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

output_path = "./out_7B"
args = TrainingArguments(
    output_dir=output_path,
    per_device_train_batch_size=BATCH_SIZE,
    logging_steps=100,
    gradient_accumulation_steps=2,
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    warmup_steps=500,
    lr_scheduler_type="cosine",
    learning_rate=LEARNING_RATE,
    save_steps=0.25,
    fp16=True,
    report_to="wandb"
)

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=args,
    data_collator=data_collator,
    train_dataset=tokenized_data["train"],
    resume_from_checkpoint=True,
)

print("Training Model....")
trainer.train()
print("Model Trained Successfully! \n")

trainer.save_model(f"{output_path}/final_model")
folder = "./out/final_model"

HUGGINGFACE_ID = "abhiakshat"
api = HfApi()
create_repo(
    repo_id = f"{HUGGINGFACE_ID}/{NEW_MODEL}",
    repo_type="model",
    exist_ok=True,
    token=HF_TOKEN,
)

api.upload_folder(
    folder_path=folder,
    repo_type="model",
    repo_id=f"{HUGGINGFACE_ID}/{NEW_MODEL}",
    token=HF_TOKEN,
)