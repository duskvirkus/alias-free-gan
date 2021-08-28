import torch

def print_gpu_memory_stats(gpu_index):
    r = torch.cuda.memory_reserved(gpu_index)
    a = torch.cuda.memory_allocated(gpu_index)
    f = r-a

    print('memory stats for gpu: ', gpu_index, ' allocated: ', a, ' free: ', f)