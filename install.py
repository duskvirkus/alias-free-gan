import pip
import os
import re
from sys import platform

_all_ = [
    "pytorch-lightning",
    "pytorch-lightning-bolts",
    "wandb",
    "ninja",
    "pytest",
    "numpy",
    "scipy",
    "nltk",
    "lmdb",
    "cython",
    "pydantic",
    "pyhocon",
    "opencv-python-headless",
    "setuptools",
    "pytorch==1.9",
    "torchvision==0.10",
]

def install(packages):
    for package in packages:
        os.system(f'python -m pip install {package}')


def get_cuda_version():
    path = '/usr/local/cuda/version.txt'
    if os.path.isfile(path):
        with open(path, 'r') as f:
            line = f.readline().strip()
            regex = r'[0-9]+.[0-9]+'
            match = re.search(regex, line)
            if match:
                version = match.group(0)
                version = ''.join(version.split('.'))
                version = 'cu' + version
                return version
    return None     

def install_arrayfire_wheel(cuda_version):
    install_successful = False
    installation_failed = False
    while not install_successful and not installation_failed:
        to_install = 'arrayfire==3.8.0'
        if cuda_version is not None:
            to_install += '+' + cuda_version
        
        code = os.system(f'python -m pip install {to_install} -f https://repo.arrayfire.com/python/wheels/3.8.0/')
        if code == 0:
            install_successful = True
        else:
            print(f'Failed to install {to_install}')
            print('trying next cuda version')
            if int(cuda_version[-1]) == 9:
                installation_failed = True
            cuda_version = cuda_version[:-1] + str(int(cuda_version[-1]) + 1)



if __name__ == '__main__':

    install(_all_)

    cuda_version = get_cuda_version()
    install_arrayfire_wheel(cuda_version)
