
import subprocess
import gradio as gr
from PIL import Image
import requests
import torch
import uuid
import shutil
import json
import yaml
from transformers import AutoProcessor, AutoModelForCausalLM
from tqdm import tqdm
import os



if __name__ == "__main__":
    # url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg?download=true"
    image = Image.open("examples/cartoon_dinosaur.png").convert("RGB")
    breakpoint()
    run_captioning(image)

