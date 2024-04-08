#!/bin/sh
#SBATCH --time=5
#SBATCH --job-name=nguyesyd_mt
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:6

UNAME=nguyesyd
EXPERIMENT_NAME=exp_th_1_gpt2xl_evalonly
IMG_TAG=mt

IMG_NAME="$UNAME/$EXPERIMENT_NAME:$IMG_TAG"

CONTAINER_NAME="$UNAME"_"$EXPERIMENT_NAME"

start=$(date +%s.%N)

echo ------------------------
echo Building image $IMG_NAME ...
echo ------------------------
docker build -t $IMG_NAME /cluster/home/nguyesyd/mtDGX 
echo ------------------------
echo Starting container $CONTAINER_NAME ...
echo ------------------------


nvidia-docker run -i --rm --gpus all -e CUDA_VISIBLE_DEVICES=4 \
    --user $(id -u):$(id -g) \
    --volume /cluster/home/nguyesyd/mtDGX/cache:/app/cache \
    --volume /cluster/home/nguyesyd/mtDGX/cache:/root/.cache/huggingface \
    --volume /cluster/home/nguyesyd/mtDGX/data/model-outputs/checkpoints/llama2:/app/data/model-outputs/checkpoints/llama2 \
    --volume /cluster/home/nguyesyd/mtDGX/data/model-outputs/checkpoints/tiny-llama:/app/data/model-outputs/checkpoints/tiny-llama \
    --volume /cluster/home/nguyesyd/mtDGX/data/pvqa/CLIPLLAMA:/app/data/pvqa/CLIPLLAMA \
    --volume /cluster/home/nguyesyd/mtDGX/data/model-outputs/checkpoints/gpt2-xl:/app/data/model-outputs/checkpoints/gpt2-xl \
    --volume /cluster/home/nguyesyd/mtDGX/data/model-outputs/checkpoints/gpt2:/app/data/model-outputs/checkpoints/gpt2 \
    --volume /cluster/home/nguyesyd/mtDGX/data/pvqa/CLIPGPT2:/app/data/pvqa/CLIPGPT2 \
    --name $CONTAINER_NAME \
    $IMG_NAME

echo ------------------------
echo Removing image... 
echo ------------------------
docker rmi $IMG_NAME

end=$(date +%s.%N)
runtime=$(echo "$end - $start" | bc)
echo ------------------------
echo "Execution time: $runtime seconds"
echo ------------------------


# ----------------------------------------
# sbatch - this is how it has to be done!
# ----------------------------------------
# sbatch /cluster/home/nguyesyd/mtDGX/docker_submit.sh


# --------
# MANUALLY
# --------
# 1) docker build -t nguyesyd_mt /cluster/home/nguyesyd/data/test_torch_docker/mtDGX

# 2) get a  shell
# srun --pty --job-name=mt6 --ntasks=1 --cpus-per-task=4 --mem=64G --gres=gpu:6 bash

# 3) nvidia-docker run -it --rm --user $(id -u):$(id -g) --name nguyesyd_mt --volume /cluster/home/nguyesyd/data/test_torch_docker/mtDGX/results:/mtDGX/results --volume /cluster/home/nguyesyd/data/test_torch_docker/mtDGX/data:/mtDGX/data nguyesyd_mt /bin/bash

# 4) python3 LLaMA_VQA/main.py