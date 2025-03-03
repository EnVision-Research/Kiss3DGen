# The kiss3d pipeline wrapper for inference

import os
# import spaces
import numpy as np
import random
import torch
import yaml
import uuid
from typing import Union, Any, Dict
from einops import rearrange
from PIL import Image
import time

from pipeline.utils import logger, TMP_DIR, OUT_DIR
from pipeline.utils import lrm_reconstruct, isomer_reconstruct, preprocess_input_image

import torch
import torchvision
from torch.nn import functional as F

# for reconstruction model
from omegaconf import OmegaConf
from models.lrm.utils.train_util import instantiate_from_config
from models.lrm.utils.render_utils import rotate_x, rotate_y
# 
from utils.tool import get_background
# for florence2
from transformers import AutoProcessor, AutoModelForCausalLM, AutoTokenizer
from models.llm.llm import load_llm_model, get_llm_response

from pipeline.custom_pipelines import FluxPriorReduxPipeline, FluxControlNetImg2ImgPipeline, FluxImg2ImgPipeline
from diffusers import FluxPipeline, DiffusionPipeline, EulerAncestralDiscreteScheduler, FluxTransformer2DModel
from diffusers.models.controlnets.controlnet_flux import FluxMultiControlNetModel, FluxControlNetModel
from diffusers.schedulers import FlowMatchHeunDiscreteScheduler
from huggingface_hub import hf_hub_download


def convert_flux_pipeline(exist_flux_pipe, target_pipe, **kwargs):
    new_pipe = target_pipe(
        scheduler = exist_flux_pipe.scheduler,
        vae = exist_flux_pipe.vae,
        text_encoder = exist_flux_pipe.text_encoder,
        tokenizer = exist_flux_pipe.tokenizer,
        text_encoder_2 = exist_flux_pipe.text_encoder_2,
        tokenizer_2 = exist_flux_pipe.tokenizer_2,
        transformer = exist_flux_pipe.transformer,
        **kwargs
    )
    return new_pipe

# @spaces.GPU
def init_wrapper_from_config(config_path):
    with open(config_path, 'r') as config_file:
        config_ = yaml.load(config_file, yaml.FullLoader)

    dtype_ = {
        'fp8': torch.float8_e4m3fn,
        'bf16': torch.bfloat16,
        'fp16': torch.float16,
        'fp32': torch.float32
    }
    
    # init flux_pipeline
    logger.info('==> Loading Flux model ...')
    flux_device = config_['flux'].get('device', 'cpu')
    flux_base_model_pth = config_['flux'].get('base_model', None)
    flux_dtype = config_['flux'].get('dtype', 'bf16')
    flux_controlnet_pth = config_['flux'].get('controlnet', None)
    flux_lora_pth = config_['flux'].get('lora', None)
    flux_redux_pth = config_['flux'].get('redux', None)
    
    if flux_base_model_pth.endswith('safetensors'):
        flux_pipe = FluxImg2ImgPipeline.from_single_file(flux_base_model_pth, torch_dtype=dtype_[flux_dtype], )
    else:
        flux_pipe = FluxImg2ImgPipeline.from_pretrained(flux_base_model_pth, torch_dtype=dtype_[flux_dtype])
    
    # load flux model and controlnet
    if flux_controlnet_pth is not None:
        flux_controlnet = FluxControlNetModel.from_pretrained(flux_controlnet_pth, torch_dtype=torch.bfloat16)
        flux_pipe = convert_flux_pipeline(flux_pipe, FluxControlNetImg2ImgPipeline, controlnet=[flux_controlnet])

    flux_pipe.scheduler = FlowMatchHeunDiscreteScheduler.from_config(flux_pipe.scheduler.config)
        

    # breakpoint()
    # load lora weights
    if not os.path.exists(flux_lora_pth):
        flux_lora_pth = hf_hub_download(repo_id="LTT/Kiss3DGen", filename="rgb_normal.safetensors", repo_type="model")
    flux_pipe.load_lora_weights(flux_lora_pth)
    flux_pipe.to(device=flux_device)

    # load redux model
    flux_redux_pipe = None
    if flux_redux_pth is not None:
        flux_redux_pipe = FluxPriorReduxPipeline.from_pretrained(flux_redux_pth, torch_dtype=torch.bfloat16, )
        flux_redux_pipe.text_encoder = flux_pipe.text_encoder
        flux_redux_pipe.text_encoder_2 = flux_pipe.text_encoder_2
        flux_redux_pipe.tokenizer = flux_pipe.tokenizer
        flux_redux_pipe.tokenizer_2 = flux_pipe.tokenizer_2

        flux_redux_pipe.to(device=flux_device)

    logger.warning(f"GPU memory allocated after load flux model on {flux_device}: {torch.cuda.memory_allocated(device=flux_device) / 1024**3} GB")

    # init multiview model
    logger.info('==> Loading multiview diffusion model ...')
    multiview_device = config_['multiview'].get('device', 'cpu')
    multiview_pipeline = DiffusionPipeline.from_pretrained(
        config_['multiview']['base_model'], 
        custom_pipeline=config_['multiview']['custom_pipeline'],
        torch_dtype=torch.float16,
    )
    multiview_pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
        multiview_pipeline.scheduler.config, timestep_spacing='trailing'
    )
    
    unet_ckpt_path = config_['multiview'].get('unet', None)
    if not os.path.exists(unet_ckpt_path):
        unet_ckpt_path = hf_hub_download(repo_id="LTT/Kiss3DGen", filename="flexgen.ckpt", repo_type="model")
    state_dict = torch.load(unet_ckpt_path, map_location='cpu')
    multiview_pipeline.unet.load_state_dict(state_dict, strict=True)

    multiview_pipeline.to(multiview_device)
    logger.warning(f"GPU memory allocated after load multiview model on {multiview_device}: {torch.cuda.memory_allocated(device=multiview_device) / 1024**3} GB")

    # load caption model
    logger.info('==> Loading caption model ...')
    caption_device = config_['caption'].get('device', 'cpu')
    caption_model = AutoModelForCausalLM.from_pretrained(config_['caption']['base_model'], \
                    torch_dtype=torch.bfloat16, trust_remote_code=True).to(caption_device)
    caption_processor = AutoProcessor.from_pretrained(config_['caption']['base_model'], trust_remote_code=True)
    logger.warning(f"GPU memory allocated after load caption model on {caption_device}: {torch.cuda.memory_allocated(device=caption_device) / 1024**3} GB")

    # load reconstruction model
    logger.info('==> Loading reconstruction model ...')
    recon_device = config_['reconstruction'].get('device', 'cpu')
    recon_model_config = OmegaConf.load(config_['reconstruction']['model_config'])
    recon_model = instantiate_from_config(recon_model_config.model_config)
    model_ckpt_path = config_['reconstruction']['base_model']
    if not os.path.exists(model_ckpt_path):
        model_ckpt_path = hf_hub_download(repo_id="LTT/PRM", filename="final_ckpt.ckpt", repo_type="model")
    state_dict = torch.load(model_ckpt_path, map_location='cpu')['state_dict']
    state_dict = {k[14:]: v for k, v in state_dict.items() if k.startswith('lrm_generator.')}
    recon_model.load_state_dict(state_dict, strict=True)
    recon_model.to(recon_device)
    recon_model.init_flexicubes_geometry(recon_device, fovy=50.0)
    recon_model.eval()
    logger.warning(f"GPU memory allocated after load reconstruction model on {recon_device}: {torch.cuda.memory_allocated(device=recon_device) / 1024**3} GB")

    # load llm
    llm_configs = config_.get('llm', None)
    if llm_configs is not None:
        logger.info('==> Loading LLM ...')
        llm_device = llm_configs.get('device', 'cpu')
        llm, llm_tokenizer = load_llm_model(llm_configs['base_model'])
        llm.to(llm_device)
        logger.warning(f"GPU memory allocated after load llm model on {llm_device}: {torch.cuda.memory_allocated(device=llm_device) / 1024**3} GB")
    else:
        llm, llm_tokenizer = None, None

    return kiss3d_wrapper(
        config = config_,
        flux_pipeline = flux_pipe,
        flux_redux_pipeline=flux_redux_pipe,
        multiview_pipeline = multiview_pipeline,
        caption_processor = caption_processor,
        caption_model = caption_model,
        reconstruction_model_config = recon_model_config,
        reconstruction_model = recon_model,
        llm_model = llm,
        llm_tokenizer = llm_tokenizer
    )

def seed_everything(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"Random seed set to {seed}")

class kiss3d_wrapper(object):
    def __init__(self,
        config: Dict,
        flux_pipeline: Union[FluxPipeline, FluxControlNetImg2ImgPipeline],
        flux_redux_pipeline: FluxPriorReduxPipeline,
        multiview_pipeline: DiffusionPipeline,
        caption_processor: AutoProcessor,
        caption_model: AutoModelForCausalLM,
        reconstruction_model_config: Any,
        reconstruction_model: Any,
        llm_model: AutoModelForCausalLM = None,
        llm_tokenizer: AutoTokenizer = None
    ):
        self.config = config
        self.flux_pipeline = flux_pipeline
        self.flux_redux_pipeline = flux_redux_pipeline
        self.multiview_pipeline = multiview_pipeline
        self.caption_model = caption_model
        self.caption_processor = caption_processor
        self.recon_model_config = reconstruction_model_config
        self.recon_model = reconstruction_model
        self.llm_model = llm_model
        self.llm_tokenizer = llm_tokenizer

        self.to_512_tensor = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize((512, 512), interpolation=2),
        ])

        self.renew_uuid()

    def renew_uuid(self):
        self.uuid = uuid.uuid4()

    def context(self):
        if self.config['use_zero_gpu']:
            pass
            # import spaces
            # return spaces.GPU()
        else:
            return torch.no_grad()

    def get_image_caption(self, image):
        """
        image: PIL image or path of PIL image
        """
        torch_dtype = torch.bfloat16
        caption_device = self.config['caption'].get('device', 'cpu')

        if isinstance(image, str):  # If image is a file path
            image = preprocess_input_image(Image.open(image))
        elif not isinstance(image, Image.Image):
            raise NotImplementedError('unexpected image type')

        prompt = "<MORE_DETAILED_CAPTION>"
        inputs = self.caption_processor(text=prompt, images=image, return_tensors="pt").to(caption_device, torch_dtype)

        generated_ids = self.caption_model.generate(
                input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"], max_new_tokens=1024, num_beams=3
            )

        generated_text = self.caption_processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = self.caption_processor.post_process_generation(
            generated_text, task=prompt, image_size=(image.width, image.height)
        )
        caption_text = parsed_answer["<MORE_DETAILED_CAPTION>"] # .replace("The image is ", "")

        logger.info(f"Auto caption result: \"{caption_text}\"")

        caption_text = self.get_detailed_prompt(caption_text)

        return caption_text

    def get_detailed_prompt(self, prompt, seed=None):
        if self.llm_model is not None:
            detailed_prompt = get_llm_response(self.llm_model, self.llm_tokenizer, prompt, seed=seed)

            logger.info(f"LLM refined prompt result: \"{detailed_prompt}\"")
            return detailed_prompt
        return prompt

    def del_llm_model(self):
        # logger.warning('This function is now deprecated and will take no effect')
        
        # raise NotImplementedError()
        # del llm.model
        # del llm.tokenizer
        self.llm_model = None
        self.llm_tokenizer = None

    def generate_multiview(self, image, seed=None, num_inference_steps=None):
        seed = seed or self.config['multiview'].get('seed', 0)
        mv_device = self.config['multiview'].get('device', 'cpu')

        generator = torch.Generator(device=mv_device).manual_seed(seed)
        with self.context():
            mv_image = self.multiview_pipeline(image, 
                                               num_inference_steps=num_inference_steps or self.config['multiview']['num_inference_steps'], 
                                               width=512*2, 
                                               height=512*2,
                                               generator=generator).images[0]
        return mv_image

    def reconstruct_from_multiview(self, mv_image, lrm_render_radius=4.15):
        """
        mv_image: PIL.Image
        """
        recon_device = self.config['reconstruction'].get('device', 'cpu')

        rgb_multi_view = np.asarray(mv_image, dtype=np.float32) / 255.0
        rgb_multi_view = torch.from_numpy(rgb_multi_view).squeeze(0).permute(2, 0, 1).contiguous().float()     # (3, 1024, 2048)
        rgb_multi_view = rearrange(rgb_multi_view, 'c (n h) (m w) -> (n m) c h w', n=2, m=2).unsqueeze(0).to(recon_device)

        with self.context():
            vertices, faces, lrm_multi_view_normals, lrm_multi_view_rgb, lrm_multi_view_albedo = \
            lrm_reconstruct(self.recon_model, self.recon_model_config.infer_config,
                            rgb_multi_view, name=self.uuid, render_radius=lrm_render_radius)

        return rgb_multi_view, vertices, faces, lrm_multi_view_normals, lrm_multi_view_rgb, lrm_multi_view_albedo
    
    def generate_reference_3D_bundle_image_zero123(self, image, use_mv_rgb=False, save_intermediate_results=True):
        """
        input: image, PIL.Image
        return: ref_3D_bundle_image, Tensor of shape (3, 1024, 2048)
        """
        mv_image = self.generate_multiview(image)

        if save_intermediate_results:
            mv_image.save(os.path.join(TMP_DIR, f'{self.uuid}_mv_image.png'))

        rgb_multi_view, vertices, faces, lrm_multi_view_normals, lrm_multi_view_rgb, lrm_multi_view_albedo = self.reconstruct_from_multiview(mv_image)

        if use_mv_rgb:
            # ref_3D_bundle_image = torchvision.utils.make_grid(torch.cat([rgb_multi_view[0, [3, 0, 1, 2], ...].cpu(), (lrm_multi_view_normals.cpu() + 1) / 2], dim=0), nrow=4, padding=0) # range [0, 1]
            # breakpoint()
            # rgb_ = torch.cat([rgb_multi_view[0, [3, 0, 1, 2], ...].cpu(), lrm_multi_view_rgb.cpu()], dim=0)
            # ref_3D_bundle_image = torchvision.utils.make_grid(torch.cat([rgb_[[0, 5, 2, 7], ...], (lrm_multi_view_normals.cpu() + 1) / 2], dim=0), nrow=4, padding=0) # range [0, 1]
            ref_3D_bundle_image = torchvision.utils.make_grid(torch.cat([rgb_multi_view.squeeze(0).detach().cpu()[[3,0,1,2]], (lrm_multi_view_normals.cpu() + 1) / 2], dim=0), nrow=4, padding=0) # range [0, 1]
        else:
            ref_3D_bundle_image = torchvision.utils.make_grid(torch.cat([lrm_multi_view_rgb.cpu(), (lrm_multi_view_normals.cpu() + 1) / 2], dim=0), nrow=4, padding=0) # range [0, 1]
        
        ref_3D_bundle_image = ref_3D_bundle_image.clip(0., 1.)
        
        if save_intermediate_results:
            save_path = os.path.join(TMP_DIR, f'{self.uuid}_ref_3d_bundle_image.png')
            torchvision.utils.save_image(ref_3D_bundle_image, save_path)

            logger.info(f"Save reference 3D bundle image to {save_path}")

            return ref_3D_bundle_image, save_path

        return ref_3D_bundle_image

    def generate_3d_bundle_image_controlnet(self, 
                                 prompt, 
                                 image=None,
                                 strength=1.0, 
                                 control_image=[],
                                 control_mode=[],
                                 control_guidance_start=None,
                                 control_guidance_end=None,
                                 controlnet_conditioning_scale=None,
                                 lora_scale=1.0,
                                 num_inference_steps=None,
                                 seed=None,
                                 redux_hparam=None,
                                 save_intermediate_results=True,
                                 **kwargs):
        control_mode_dict = {
            'canny': 0,
            'tile': 1,
            'depth': 2,
            'blur': 3,
            'pose': 4,
            'gray': 5,
            'lq': 6,
        } # for https://huggingface.co/InstantX/FLUX.1-dev-Controlnet-Union only

        flux_device = self.config['flux'].get('device', 'cpu')
        seed = seed or self.config['flux'].get('seed', 0)
        num_inference_steps = num_inference_steps or self.config['flux'].get('num_inference_steps', 20)

        generator = torch.Generator(device=flux_device).manual_seed(seed)

        if image is None:
            image = torch.zeros((1, 3, 1024, 2048), dtype=torch.float32, device=flux_device)

        hparam_dict = {
            'prompt': 'A grid of 2x4 multi-view image, elevation 5. White background.',
            'prompt_2': ' '.join(['A grid of 2x4 multi-view image, elevation 5. White background.', prompt]),
            'image': image,
            'strength': strength,
            'num_inference_steps': num_inference_steps,
            'guidance_scale': 3.5,
            'num_images_per_prompt': 1,
            'width': 2048,
            'height': 1024,
            'output_type': 'np',
            'generator': generator,
            'joint_attention_kwargs': {"scale": lora_scale}
        }
        hparam_dict.update(kwargs)

        # do redux
        if redux_hparam is not None:
            assert self.flux_redux_pipeline is not None
            assert 'image' in redux_hparam.keys()
            redux_hparam_ = {
                'prompt': hparam_dict.pop('prompt'),
                'prompt_2': hparam_dict.pop('prompt_2'),
            }
            redux_hparam_.update(redux_hparam)

            with self.context():
                redux_output = self.flux_redux_pipeline(**redux_hparam_)
    
            hparam_dict.update(redux_output)

         # append controlnet hparams
        if len(control_image) > 0:
            assert isinstance(self.flux_pipeline, FluxControlNetImg2ImgPipeline)
            assert len(control_mode) == len(control_image) # the count of image should be the same as control mode
            
            flux_ctrl_net = self.flux_pipeline.controlnet.nets[0]
            self.flux_pipeline.controlnet = FluxMultiControlNetModel([flux_ctrl_net for _ in control_mode])

            ctrl_hparams = {
                'control_mode': [control_mode_dict[mode_] for mode_ in control_mode],
                'control_image': control_image,
                'control_guidance_start': control_guidance_start or [0.0 for i in range(len(control_image))],
                'control_guidance_end': control_guidance_end or [1.0 for i in range(len(control_image))],
                'controlnet_conditioning_scale': controlnet_conditioning_scale or [1.0 for i in range(len(control_image))],
            }

            hparam_dict.update(ctrl_hparams)

        with self.context():
            gen_3d_bundle_image = self.flux_pipeline(**hparam_dict).images
        
        gen_3d_bundle_image_ = torch.from_numpy(gen_3d_bundle_image).squeeze(0).permute(2, 0, 1).contiguous().float()     # (3, 1024, 2048)

        if save_intermediate_results:
            save_path = os.path.join(TMP_DIR, f'{self.uuid}_gen_3d_bundle_image.png')
            torchvision.utils.save_image(gen_3d_bundle_image_, save_path)
            logger.info(f"Save generated 3D bundle image to {save_path}")
            return gen_3d_bundle_image_, save_path

        return gen_3d_bundle_image_

    def preprocess_controlnet_cond_image(self, image, control_mode, save_intermediate_results=True, **kwargs):
        """
        image: Tensor of shape (c, h, w), range [0., 1.]
        """
        if control_mode in ['tile', 'lq']:
            _, h, w = image.shape
            down_scale = kwargs.get('down_scale', 4)
            down_up = torchvision.transforms.Compose([
                torchvision.transforms.Resize((h // down_scale, w // down_scale), interpolation=2), # 1 for lanczos and 2 for bilinear
                torchvision.transforms.Resize((h, w), interpolation=2),
                torchvision.transforms.ToPILImage()
            ])
            preprocessed = down_up(image)
        elif control_mode == 'blur':
            kernel_size = kwargs.get('kernel_size', 51)
            sigma = kwargs.get('sigma', 2.0)
            blur = torchvision.transforms.Compose([
                    torchvision.transforms.ToPILImage(),
                    torchvision.transforms.GaussianBlur(kernel_size, sigma),
                ])
            preprocessed = blur(image)
        else:
            raise NotImplementedError(f'Unexpected control mode {control_mode}')

        if save_intermediate_results:
            save_path = os.path.join(TMP_DIR, f'{self.uuid}_{control_mode}_controlnet_cond.png')
            preprocessed.save(save_path)
            logger.info(f'Save image to {save_path}')
    
        return preprocessed

    def generate_3d_bundle_image_text(self, 
                                      prompt,
                                      image=None, 
                                      strength=1.0,
                                      lora_scale=1.0,
                                      num_inference_steps=None,
                                      seed=None,
                                      redux_hparam=None,
                                      save_intermediate_results=True,
                                      **kwargs):
        
        """
        return: gen_3d_bundle_image, torch.Tensor of shape (3, 1024, 2048), range [0., 1.]
        """
        
        if isinstance(self.flux_pipeline, FluxImg2ImgPipeline):
            flux_pipeline = self.flux_pipeline
        else:
            flux_pipeline = convert_flux_pipeline(self.flux_pipeline, FluxImg2ImgPipeline)

        flux_device = self.config['flux'].get('device', 'cpu')
        seed = seed or self.config['flux'].get('seed', 0)
        num_inference_steps = num_inference_steps or self.config['flux'].get('num_inference_steps', 20)

        if image is None:
            image = torch.zeros((1, 3, 1024, 2048), dtype=torch.float32, device=flux_device)

        generator = torch.Generator(device=flux_device).manual_seed(seed)


        hparam_dict = {
            'prompt': 'A grid of 2x4 multi-view image, elevation 5. White background.',
            'prompt_2': ' '.join(['A grid of 2x4 multi-view image, elevation 5. White background.', prompt]),
            'image': image,
            'strength': strength,
            'num_inference_steps': num_inference_steps,
            'guidance_scale': 3.5,
            'num_images_per_prompt': 1,
            'width': 2048,
            'height': 1024,
            'output_type': 'np',
            'generator': generator,
            'joint_attention_kwargs': {"scale": lora_scale}
        }
        hparam_dict.update(kwargs)

        # do redux
        if redux_hparam is not None:
            assert self.flux_redux_pipeline is not None
            assert 'image' in redux_hparam.keys()
            redux_hparam_ = {
                'prompt': hparam_dict.pop('prompt'),
                'prompt_2': hparam_dict.pop('prompt_2'),
            }
            redux_hparam_.update(redux_hparam)

            with self.context():
                redux_output = self.flux_redux_pipeline(**redux_hparam_)
    
            hparam_dict.update(redux_output)


        with self.context():
            gen_3d_bundle_image = flux_pipeline(**hparam_dict).images

        gen_3d_bundle_image_ = torch.from_numpy(gen_3d_bundle_image).squeeze(0).permute(2, 0, 1).contiguous().float()     # (3, 1024, 2048)

        if save_intermediate_results:
            save_path = os.path.join(TMP_DIR, f'{self.uuid}_gen_3d_bundle_image.png')
            torchvision.utils.save_image(gen_3d_bundle_image_, save_path)
            logger.info(f"Save generated 3D bundle image to {save_path}")
            return gen_3d_bundle_image_, save_path

        return gen_3d_bundle_image_
    
    def reconstruct_3d_bundle_image(self, 
        image, 
        lrm_render_radius=4.15, 
        isomer_radius=4.5, 
        reconstruction_stage1_steps=0,
        reconstruction_stage2_steps=20,
        save_intermediate_results=True):
        """
        image: torch.Tensor, range [0., 1.], (3, 1024, 2048)
        """
        recon_device = self.config['reconstruction'].get('device', 'cpu')

        # split rgb and normal
        images = rearrange(image, 'c (n h) (m w) -> (n m) c h w', n=2, m=4) # (3, 1024, 2048) -> (8, 3, 512, 512)
        rgb_multi_view, normal_multi_view = images.chunk(2, dim=0)
        multi_view_mask = get_background(normal_multi_view).to(recon_device)
        print(f'shape images: {images.shape}')
        # breakpoint()
        rgb_multi_view = rgb_multi_view.to(recon_device) * multi_view_mask + (1 - multi_view_mask)
        end = time.time()
        with self.context():
            vertices, faces, lrm_multi_view_normals, lrm_multi_view_rgb, lrm_multi_view_albedo = \
            lrm_reconstruct(self.recon_model, self.recon_model_config.infer_config,
                            rgb_multi_view.unsqueeze(0).to(recon_device), name=self.uuid, 
                            input_camera_type='kiss3d', render_3d_bundle_image=save_intermediate_results,
                            render_azimuths=[0, 90, 180, 270],
                            render_radius=lrm_render_radius)
        print(f'lrm_reconstruct time: {time.time() - end}')
        if save_intermediate_results:
            recon_3D_bundle_image = torchvision.utils.make_grid(torch.cat([lrm_multi_view_rgb.cpu(), (lrm_multi_view_normals.cpu() + 1) / 2], dim=0), nrow=4, padding=0).unsqueeze(0) # range [0, 1]        
            torchvision.utils.save_image(recon_3D_bundle_image, os.path.join(TMP_DIR, f'{self.uuid}_lrm_recon_3d_bundle_image.png'))

        recon_mesh_paths = [os.path.join(TMP_DIR, f"{self.uuid}_isomer_recon_mesh.glb"), os.path.join(TMP_DIR, f"{self.uuid}_isomer_recon_mesh.obj")]
        
        return isomer_reconstruct(rgb_multi_view=rgb_multi_view,
                                  normal_multi_view=normal_multi_view,
                                  multi_view_mask=multi_view_mask,
                                  vertices=vertices,
                                  faces=faces,
                                  save_paths=recon_mesh_paths,
                                  radius=isomer_radius,
                                  reconstruction_stage1_steps=int(reconstruction_stage1_steps),
        reconstruction_stage2_steps=int(reconstruction_stage2_steps)
        )


def run_text_to_3d(k3d_wrapper,
                   prompt,
                   init_image_path=None):
    # ======================================= Example of text to 3D generation ======================================

    # Renew The uuid
    k3d_wrapper.renew_uuid()

    # FOR Text to 3D (also for image to image) with init image
    init_image = None
    if init_image_path is not None:
        init_image = Image.open(init_image_path)

    # refine prompt
    logger.info(f"Input prompt: \"{prompt}\"")
    
    prompt = k3d_wrapper.get_detailed_prompt(prompt)
    end = time.time()
    gen_3d_bundle_image, gen_save_path = k3d_wrapper.generate_3d_bundle_image_text(prompt, 
                                                                                   image=init_image, 
                                                                                   strength=1.0, 
                                                                                   save_intermediate_results=True)
    print(f"3d bundle image generation time: {time.time() - end}")
    # recon from 3D Bundle image
    recon_mesh_path = k3d_wrapper.reconstruct_3d_bundle_image(gen_3d_bundle_image, save_intermediate_results=False,
                                                              isomer_radius=4.2, reconstruction_stage2_steps=50)

    return gen_save_path, recon_mesh_path

def image2mesh_preprocess(k3d_wrapper, input_image_, seed, use_mv_rgb=True):
    seed_everything(seed)

    # Renew The uuid
    k3d_wrapper.renew_uuid()

    # FOR IMAGE TO 3D: generate reference 3D bundle image from a single input image
    input_image__ = Image.open(input_image_) if isinstance(input_image_, str) else input_image_

    input_image = preprocess_input_image(input_image__)
    input_image_save_path = os.path.join(TMP_DIR, f'{k3d_wrapper.uuid}_input_image.png')
    input_image.save(input_image_save_path)

    reference_3d_bundle_image, reference_save_path = k3d_wrapper.generate_reference_3D_bundle_image_zero123(input_image, use_mv_rgb=use_mv_rgb)
    caption = k3d_wrapper.get_image_caption(input_image)

    return input_image_save_path, reference_save_path, caption

def image2mesh_main(k3d_wrapper, input_image, reference_3d_bundle_image, caption, seed, strength1=0.5, strength2=0.95, enable_redux=True, use_controlnet=True):
    seed_everything(seed)

    if enable_redux:
        redux_hparam = {
            'image': k3d_wrapper.to_512_tensor(input_image).unsqueeze(0).clip(0., 1.),
            'prompt_embeds_scale': 1.0,
            'pooled_prompt_embeds_scale': 1.0,
            'strength': strength1
        }
    else:
        redux_hparam = None

    if use_controlnet:
        # prepare controlnet condition
        control_mode = ['tile']
        control_image = [k3d_wrapper.preprocess_controlnet_cond_image(reference_3d_bundle_image, mode_, down_scale=1, kernel_size=51, sigma=2.0) for mode_ in control_mode]
        control_guidance_start = [0.0]
        control_guidance_end = [0.3]
        controlnet_conditioning_scale = [0.3]


        gen_3d_bundle_image, gen_save_path = k3d_wrapper.generate_3d_bundle_image_controlnet(
            prompt=caption,
            image=reference_3d_bundle_image.unsqueeze(0),
            strength=strength2,
            control_image=control_image, 
            control_mode=control_mode,
            control_guidance_start=control_guidance_start,
            control_guidance_end=control_guidance_end,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            lora_scale=1.0,
            redux_hparam=redux_hparam
        )
    else:
        gen_3d_bundle_image, gen_save_path = k3d_wrapper.generate_3d_bundle_image_text(
            prompt=caption,
            image=reference_3d_bundle_image.unsqueeze(0),
            strength=strength2,
            lora_scale=1.0,
            redux_hparam=redux_hparam
        )

    # recon from 3D Bundle image
    recon_mesh_path = k3d_wrapper.reconstruct_3d_bundle_image(gen_3d_bundle_image, save_intermediate_results=False)

    return gen_save_path, recon_mesh_path


def run_image_to_3d(k3d_wrapper, input_image_path, enable_redux=True, use_mv_rgb=True, use_controlnet=True):
    # ======================================= Example of image to 3D generation ======================================

    # Renew The uuid
    k3d_wrapper.renew_uuid()

    # FOR IMAGE TO 3D: generate reference 3D bundle image from a single input image
    input_image = preprocess_input_image(Image.open(input_image_path))
    input_image.save(os.path.join(TMP_DIR, f'{k3d_wrapper.uuid}_input_image.png'))

    reference_3d_bundle_image, reference_save_path = k3d_wrapper.generate_reference_3D_bundle_image_zero123(input_image, use_mv_rgb=use_mv_rgb)
    # breakpoint()
    caption = k3d_wrapper.get_image_caption(input_image)

    if enable_redux:
        redux_hparam = {
            'image': k3d_wrapper.to_512_tensor(input_image).unsqueeze(0).clip(0., 1.),
            'prompt_embeds_scale': 1.0,
            'pooled_prompt_embeds_scale': 1.0,
            'strength': 0.5
        }
    else:
        redux_hparam = None

    if use_controlnet:
        # prepare controlnet condition
        control_mode = ['tile']
        control_image = [k3d_wrapper.preprocess_controlnet_cond_image(reference_3d_bundle_image, mode_, down_scale=1, kernel_size=51, sigma=2.0) for mode_ in control_mode]
        control_guidance_start = [0.0]
        control_guidance_end = [0.65]
        controlnet_conditioning_scale = [0.6]


        gen_3d_bundle_image, gen_save_path = k3d_wrapper.generate_3d_bundle_image_controlnet(
            prompt=caption,
            image=reference_3d_bundle_image.unsqueeze(0),
            strength=.95,
            control_image=control_image, 
            control_mode=control_mode,
            control_guidance_start=control_guidance_start,
            control_guidance_end=control_guidance_end,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            lora_scale=1.0,
            redux_hparam=redux_hparam
        )
    else:
        gen_3d_bundle_image, gen_save_path = k3d_wrapper.generate_3d_bundle_image_text(
            prompt=caption,
            image=reference_3d_bundle_image.unsqueeze(0),
            strength=.95,
            lora_scale=1.0,
            redux_hparam=redux_hparam
        )

    # recon from 3D Bundle image
    recon_mesh_path = k3d_wrapper.reconstruct_3d_bundle_image(gen_3d_bundle_image, save_intermediate_results=False,
                                                              isomer_radius=4.15, reconstruction_stage2_steps=50)

    return gen_save_path, recon_mesh_path

def run_redux_image_to_3d(k3d_wrapper, input_image_path, prompt=None, init_image_path=None, strength=1.0):
    # ======================================= Example of image to 3D generation ======================================

    # Renew The uuid
    k3d_wrapper.renew_uuid()

    # FOR IMAGE TO 3D: generate reference 3D bundle image from a single input image
    input_image = preprocess_input_image(Image.open(input_image_path))
    input_image.save(os.path.join(TMP_DIR, f'{k3d_wrapper.uuid}_input_image.png'))

    # breakpoint()
    if prompt is None:
        caption = k3d_wrapper.get_image_caption(input_image)
    else:
        caption = prompt

    redux_hparam = {
        'image': k3d_wrapper.to_512_tensor(input_image).unsqueeze(0).clip(0., 1.),
        'prompt_embeds_scale': 1.0,
        'pooled_prompt_embeds_scale': 1.0,
        'strength': strength
    }

    init_image = None
    if init_image_path is not None:
        init_image = Image.open(init_image_path)
    gen_3d_bundle_image, gen_save_path = k3d_wrapper.generate_3d_bundle_image_text(
            prompt=caption,
            image=init_image, 
            strength=0.95,
            lora_scale=1.0,
            redux_hparam=redux_hparam
        )

    # recon from 3D Bundle image
    recon_mesh_path = k3d_wrapper.reconstruct_3d_bundle_image(gen_3d_bundle_image, save_intermediate_results=False,
                                                              isomer_radius=4.15, reconstruction_stage2_steps=50)

    return gen_save_path, recon_mesh_path


if __name__ == "__main__":
    k3d_wrapper = init_wrapper_from_config('./pipeline_config/default.yaml')

    os.system(f'rm -rf {TMP_DIR}/*')
    # os.system(f'rm -rf {OUT_DIR}/3d_bundle/*')

    enable_redux = True
    use_mv_rgb = True
    use_controlnet = True

    img_folder = './examples'
    for img_ in os.listdir(img_folder):
        name, _ = os.path.splitext(img_)
        print("Now processing:", name)

        gen_save_path, recon_mesh_path = run_image_to_3d(k3d_wrapper, os.path.join(img_folder, img_), enable_redux, use_mv_rgb, use_controlnet)

        os.system(f'cp -f {gen_save_path} {OUT_DIR}/3d_bundle/{name}_3d_bundle.png')
        os.system(f'cp -f {recon_mesh_path} {OUT_DIR}/3d_bundle/{name}.obj')