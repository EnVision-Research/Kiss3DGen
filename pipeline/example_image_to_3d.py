from pipeline.kiss3d_wrapper import init_wrapper_from_config, run_text_to_3d, run_image_to_3d
import os
from pipeline.utils import logger, TMP_DIR, OUT_DIR

if __name__ == "__main__":
    k3d_wrapper = init_wrapper_from_config('./pipeline/pipeline_config/default.yaml')

    os.system(f'rm -rf {TMP_DIR}/*')
    os.makedirs(os.path.join(OUT_DIR, 'image_to_3d'), exist_ok=True)


    enable_redux = True
    use_mv_rgb = True
    use_controlnet = True

    img_folder = './examples'
    for img_ in os.listdir(img_folder):
        name, _ = os.path.splitext(img_)
        print("Now processing:", name)

        gen_save_path, recon_mesh_path = run_image_to_3d(k3d_wrapper, os.path.join(img_folder, img_), enable_redux, use_mv_rgb, use_controlnet)
        os.system(f'cp -f {gen_save_path} {OUT_DIR}/image_to_3d/{name}_3d_bundle.png')
        os.system(f'cp -f {recon_mesh_path} {OUT_DIR}/image_to_3d/{name}.glb')
