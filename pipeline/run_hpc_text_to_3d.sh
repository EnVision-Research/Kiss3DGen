source /hpc2ssd/softwares/anaconda3/bin/activate kiss3dgen
module load cuda/12.1 compilers/gcc-11.1.0 compilers/icc-2023.1.0 cmake/3.27.0
export CXX=$(which g++)
export CC=$(which gcc)
export CPLUS_INCLUDE_PATH=/hpc2ssd/softwares/cuda/cuda-12.1/targets/x86_64-linux/include:$CPLUS_INCLUDE_PATH
export CUDA_LAUNCH_BLOCKING=1
export NCCL_TIMEOUT=3600
export CUDA_VISIBLE_DEVICES="0,1"

python ./pipeline/example_text_to_3d.py