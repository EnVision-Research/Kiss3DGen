from pipeline.kiss3d_wrapper import init_wrapper_from_config, run_redux_image_to_3d, run_text_to_3d, run_image_to_3d
import os
from pipeline.utils import logger, TMP_DIR, OUT_DIR
import time
if __name__ == "__main__":
    k3d_wrapper = init_wrapper_from_config('./pipeline/pipeline_config/default.yaml')

    os.system(f'rm -rf {TMP_DIR}/*')
    os.makedirs(os.path.join(OUT_DIR, 'redux_image_to_3d'), exist_ok=True)


    enable_redux = True

    init_image_path = './init_3d_Bundle/3.png'
    # init_image_path = None
    img_folder = './examples/redux'
    for img_ in os.listdir(img_folder):
        # if not (img_.endswith('.png') or img_.endswith('.jpg') or img_.endswith('.webp')):
        #     continue
        if not img_.endswith('.webp'):
            continue
        name, _ = os.path.splitext(img_)
        print("Now processing:", name)
        end = time.time()
        prompt=None
        redux_image_strength = 0.95
        gen_save_path, recon_mesh_path = run_redux_image_to_3d(k3d_wrapper, os.path.join(img_folder, img_), init_image_path=init_image_path, prompt=prompt, strength=redux_image_strength)
        print(f"image to 3d time: {time.time() - end}")
        os.system(f'cp -f {gen_save_path} {OUT_DIR}/redux_image_to_3d/{name}_3d_bundle.png')
        os.system(f'cp -f {recon_mesh_path} {OUT_DIR}/redux_image_to_3d/{name}.glb')
