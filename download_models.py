import os
from huggingface_hub import hf_hub_download
import shutil


ckpt_save_root = "checkpoint"

flux_lora_dir = os.path.join(ckpt_save_root, "flux_lora")
os.makedirs(flux_lora_dir, exist_ok=True)

lrm_dir = os.path.join(ckpt_save_root, "lrm")
os.makedirs(lrm_dir, exist_ok=True)

zero123_plus_dir = os.path.join(ckpt_save_root, "zero123++")

flux_lora_pth = hf_hub_download(repo_id="LTT/Kiss3DGen", filename="rgb_normal.safetensors", repo_type="model", local_dir="./checkpoint/flux_lora", local_dir_use_symlinks=False)

model_ckpt_path = hf_hub_download(repo_id="LTT/PRM", filename="final_ckpt.ckpt", repo_type="model", local_dir="./checkpoint/lrm", local_dir_use_symlinks=False)

unet_ckpt_path = hf_hub_download(repo_id="LTT/Kiss3DGen", filename="flexgen.ckpt", repo_type="model", local_dir="./checkpoint/zero123++", local_dir_use_symlinks=False)
