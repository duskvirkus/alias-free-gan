import pip
import os
import re
import subprocess
from sys import platform

_all_ = [
    "pytorch-lightning",
    "pytorch-lightning-bolts",
    "wandb",
    "ninja",
    "pytest",
    "pydantic",
    "pyhocon",
    "opencv-python-headless",
    "opensimplex",
]

non_colab = [
    "numpy",
    "scipy",
    "nltk",
    "lmdb",
    "cython",
    "setuptools",
]

non_ci_and_colab = [
    "torch>1.9.0",
    "torchvision>0.10.0",
]

colab_tpu = [
    "cloud-tpu-client==0.10.0",
    "https://storage.googleapis.com/tpu-pytorch/wheels/torch_xla-1.9-cp37-cp37m-linux_x86_64.whl",
]

def apt_install(package_name):
    subprocess.run(["apt-get", "-y", "install", package_name])

def install(packages):
    all_packages = ''
    for package in packages:
        all_packages += package + ' '
    os.system('python3 -m pip install %s' % all_packages)

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
        
        code = os.system('python3 -m pip install %s -f https://repo.arrayfire.com/python/wheels/3.8.0/' % to_install)
        if code == 0:
            install_successful = True
        else:
            print('Failed to install %s' % to_install)
            print('trying next cuda version')
            if int(cuda_version[-1]) == 9:
                installation_failed = True
            cuda_version = cuda_version[:-1] + str(int(cuda_version[-1]) + 1)



if __name__ == '__main__':

    # apt_install('wget')

    install(_all_)

    if not ('COLAB_GPU' in os.environ or 'COLAB_TPU_ADDR' in os.environ):
        install(non_colab)

    if not ('COLAB_GPU' in os.environ or 'COLAB_TPU_ADDR' in os.environ or 'CI_RUNNING' in os.environ):
        install(non_ci_and_colab)

    if 'COLAB_TPU_ADDR' in os.environ and 'CI_RUNNING' not in os.environ:
        install(colab_tpu)

    if 'CI_RUNNING' in os.environ and not os.path.isdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'alias-free-gan-ci-files')):
        subprocess.run(['wget', '-O', 'alias-free-gan-ci-files.zip', 'https://aliasfreegan.sfo3.cdn.digitaloceanspaces.com/alias-free-gan-ci-files.zip'])
        subprocess.run(['unzip', 'alias-free-gan-ci-files.zip'])

    # cuda_version = get_cuda_version()
    # install_arrayfire_wheel(cuda_version)
