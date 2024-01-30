#!/usr/bin/env python3
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def all_gather_dict(gathered_list, local_dict):
    dist.barrier()

    # Convert the local_dict to a list of tuples for all_gather_object
    local_list = list(local_dict.items())

    # All gather the list of tuples
    # gathered_list = [None for _ in range(dist.get_world_size())]
    torch.distributed.all_gather_object(gathered_list, local_list)
    # for i, x in enumerate(gathered_list):
    #     print(f"{i}: {x}")

    # Convert the gathered list of tuples back to a dictionary
    def tuple2dict(tuple_items):
        out_dict = {}
        for (key, val) in tuple_items:
            out_dict[key] = val
        return out_dict
    gathered_dict = tuple2dict(gathered_list[0])
    for i, tuple_items in enumerate(gathered_list):
        if i==0:
            continue
        temp_dict = tuple2dict(tuple_items)
        for key in gathered_dict.keys():
            gathered_dict[key].update(temp_dict[key])
    print(gathered_dict)
    # Ensure that the gathered dictionary has the same structure on all processes
    dist.barrier()

    return gathered_dict

def main(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'  # 设置主进程的地址
    os.environ['MASTER_PORT'] = '12345'  # 设置主进程的端口
    os.environ['CUDA_VISIBLE_DEVICES'] = str(rank)  # 每个进程使用不同的 GPU


    num_gpus = torch.cuda.device_count()
    print(f"rank{rank}: num_gpus={num_gpus}")
    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    print(f'rank{rank}: dist env initialized')
    if rank == 0:
        data = {
            0: {
                0: [1,2], 
                1: [3,4]
            },
            1: {
                0: [1,2], 
                1: [3,4]
            },
        }
    else:
        data = {
            0: {
                2: [5,6], 
                3: [7,8]
            },
            1: {
                2: [1,2], 
                3: [3,4]
            },
        }
    output = [None for _ in range(2)]
    # dist.all_gather_object(output, data)
    output = all_gather_dict(output, data)
    print(f"rank{rank} output: {output}")
    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = 2  # 进程总数
    mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)

# CUDA_VISIBLE_DEVICES=0,1 python test_all_gather.py