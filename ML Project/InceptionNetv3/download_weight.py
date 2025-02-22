import torch
from huggingface_hub import snapshot_download
model_path = '/home/gfl-ml-2025/Music/plant_not_plant/pretrained_model/'   # The local directory to save downloaded checkpoint
# download pretrained model for donloading the weight to start fine tuning
snapshot_download("google/vit-base-patch16-224", local_dir=model_path)
