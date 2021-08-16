import hashlib
import torch

def sha1_hash(filename):

   h = hashlib.sha1()

   with open(filename,'rb') as file:

       chunk = 0
       while chunk != b'':
           chunk = file.read(1024)
           h.update(chunk)

   return h.hexdigest()

def print_gpu_memory_stats(gpu_index):
    r = torch.cuda.memory_reserved(gpu_index)
    a = torch.cuda.memory_allocated(gpu_index)
    f = r-a

    print('memory stats for gpu: ', gpu_index, ' allocated: ', a, ' free: ', f)