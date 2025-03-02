from pipeline.kiss3d_wrapper_3d_to_3d import init_wrapper_from_config, run_3d_to_3d
import os
import shutil
from pipeline.utils import logger, TMP_DIR, OUT_DIR
import time

if __name__ == "__main__":
    k3d_wrapper = init_wrapper_from_config('./pipeline/pipeline_config/redux_lora.yaml')
    # k3d_wrapper = None

    os.system(f'rm -rf {TMP_DIR}/*')

    save_dir = '3d_to_3d'

    shutil.rmtree(os.path.join(OUT_DIR, save_dir), ignore_errors=True)
    os.makedirs(os.path.join(OUT_DIR, save_dir), exist_ok=True)

    use_controlnet = True
    refine_prompt = False

    prompts = open('./examples/mesh/prompts.txt', 'r').readlines()
    # prompts = open('./examples/mesh/prompts_edit.txt', 'r').readlines()

    prompts_dict = {prompt.split(';')[0]: prompt.split(';')[1].strip() for prompt in prompts}
    
    mesh_folder = './examples/mesh/enhancement_3d_meshes'
    for mesh_name in os.listdir(mesh_folder):
        print("Now processing:", mesh_name)
        print("Prompt:", prompts_dict[mesh_name])
        end = time.time()
        gen_save_path, recon_mesh_path = run_3d_to_3d(k3d_wrapper, os.path.join(mesh_folder, mesh_name, f'{mesh_name}.obj'), prompts_dict[mesh_name], use_controlnet, refine_prompt)
        print(f"3d to 3d generation time: {time.time() - end} s")
        # run_3d_to_3d(k3d_wrapper, os.path.join(mesh_folder, mesh_name, f'{mesh_name}.obj'), enable_redux, use_mv_rgb, use_controlnet)

        os.system(f'cp -f {gen_save_path} {OUT_DIR}/{save_dir}/{mesh_name}_3d_bundle.png')
        os.system(f'cp -f {recon_mesh_path} {OUT_DIR}/{save_dir}/{mesh_name}.glb')
