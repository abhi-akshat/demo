import torch
import transformers
import huggingface_hub
import wandb
import datasets

device = "cuda" if torch.cuda.is_available() else "cpu"

print("DEVICE: ", device)
print("TORCH: ", torch.__version__)
print("TRANSFORMERS: ",transformers.__version__)
print("HF: ",huggingface_hub.__version__)
print("WANDB: ",wandb.__version__)
print("DATASETS: ",datasets.__version__)

print("All Systems Running!")

