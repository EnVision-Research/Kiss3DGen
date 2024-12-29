import os
from huggingface_hub import hf_hub_download
import shutil


ckpt_save_root = "checkpoint"

flux_lora_dir = os.path.join(ckpt_save_root, "flux_lora")
os.makedirs(flux_lora_dir, exist_ok=True)

lrm_dir = os.path.join(ckpt_save_root, "lrm")
os.makedirs(lrm_dir, exist_ok=True)

zero123_plus_dir = os.path.join(ckpt_save_root, "zero123++")

flux_lora_pth = hf_hub_download(repo_id="LTT/Kiss3DGen", filename="rgb_normal_large.safetensors", repo_type="model")
shutil.move(flux_lora_pth, os.path.join(flux_lora_dir, "rgb_normal_large.safetensors"))

model_ckpt_path = hf_hub_download(repo_id="LTT/PRM", filename="final_ckpt.ckpt", repo_type="model")
shutil.move(model_ckpt_path, os.path.join(lrm_dir, "final_ckpt.ckpt"))

unet_ckpt_path = hf_hub_download(repo_id="LTT/Kiss3DGen", filename="flexgen.ckpt", repo_type="model")
shutil.move(unet_ckpt_path, os.path.join(zero123_plus_dir, "flexgen.ckpt"))