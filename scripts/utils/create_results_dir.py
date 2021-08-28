import os
import re

def create_results_dir():
    results_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../results')
    os.makedirs(results_root, exist_ok=True)

    max_num = -1

    for root, subdirs, files in os.walk(results_root):
        for subdir in subdirs:
            numbers = re.findall('[0-9]+', subdir)
            if numbers:
                if (int(numbers[0]) > max_num):
                    max_num = int(numbers[0])
    
    max_num += 1

    results_dir = os.path.join(results_root, 'training-' + str(max_num).zfill(6))
    os.makedirs(results_dir)
    return results_dir