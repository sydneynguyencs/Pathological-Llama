import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import argparse
import numpy as np
import random
import os

from PVQAmodel import PVQAmodel

def set_random_seeds(random_seed=0):
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="llama2", choices=("llama2", "tiny-llama", "gpt2-xl", "microsoft/biogpt","stanford-crfm/BioMedLM"))
    parser.add_argument("--notes", type=str, default="")
    parser.add_argument("--question_type", type=str, default="oa", choices=("oa", "closed","both"))
    parser.add_argument("--task", type=str, default="classification", choices=("generative", "classification", "explainability"))

    parser.add_argument("--setting", type=str, default="lora", choices=("lora", "frozen",'prefixtuning',"p_tuning","prompttuning", "unfrozen"))
    parser.add_argument("--ablation", type=str, default="none", choices=("remove_question", "remove_visual",'replace_visual',"swap"))
    parser.add_argument("--mapping_type", type=str, default="MLP")
    parser.add_argument("--prefix_length", type=int, default=6)
    parser.add_argument("--dataset_path", type=str, default="data/")
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--dataset", type=str, default='pvqa/CLIPLLAMA', choices=('pvqa/CLIPGPT2', 'pvqa/CLIPLLAMA', 'pathvqa', 'ovqa', 'slake'))
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=600)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--iters_to_accumulate", type=int, default=4)
    parser.add_argument("--validation_step", type=int, default=1000)
    parser.add_argument("--out_dir", default="data/model-outputs/checkpoints")
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--eval", dest="eval", action="store_true")
    parser.add_argument("--verbose", dest="verbose", action="store_true")
    parser.add_argument("--early_stopping", type=bool, default=True)
    parser.add_argument("--tqdm", type=bool, default=False)
    parser.add_argument("--like_test", type=bool, default=False)
    
    parser.add_argument("--wandb", type=bool, default=False)
    # Add distributed training arguments
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=1)  # Total number of processes
    parser.add_argument("--rank", type=int, default=0)  # Rank of the current process
    parser.add_argument("--nodes", type=int, default=0)
    parser.add_argument("--gpus_per_node", type=int, default=torch.cuda.device_count())  # GPUs per node
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()
    set_random_seeds(args.seed)
    return args


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '8600'
    dist.init_process_group("nccl", rank=rank, world_size=world_size) # try gloo


def cleanup():
    dist.destroy_process_group()


def main_worker(rank, world_size, args):
    print(f"Running DDP on rank {rank}.")
    args.rank = rank
    args.world_size = world_size
    setup(rank=rank, world_size=world_size)

    model = PVQAmodel(args)
    # if not args.eval:
    #     model.train_model()
    model.eval_task(mode="test") # dataset choices: "test", "train", "val"
    cleanup()

"""
    1) setup the process group, which is three lines of code and needs no modification;
    2) split the dataloader to each process in the group, which can be easily achieved by torch.utils.data.DistributedSampler or any customized sampler;
    3) wrap our model with DDP, which is one line of code and barely needs modification;
    4) train/test our model, which is the same as is on 1 gpu;
    5) clean up the process groups (like free in C), which is one line of code.
"""
if __name__ == "__main__":
    args = parse_argument() 
    world_size = args.gpus_per_node
    print(f"{world_size} GPUs in use.")
    mp.spawn(
        main_worker,
        args=(world_size,args,),
        nprocs=world_size,
        join=True
    )
