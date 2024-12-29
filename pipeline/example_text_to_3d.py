from pipeline.kiss3d_wrapper import init_wrapper_from_config, run_text_to_3d, run_image_to_3d

if __name__ == "__main__":
    k3d_wrapper = init_wrapper_from_config('/hpc2hdd/home/jlin695/code/github/Kiss3DGen/pipeline/pipeline_config/default.yaml')

    run_text_to_3d(k3d_wrapper, prompt='A doll of a girl in Harry Potter')

