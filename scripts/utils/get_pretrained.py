import os
import json
import subprocess

from utils.sha1_hash import sha1_hash

class ModelNameNotFoundException(Exception):
    """Model name not found in pretrained models."""
    pass

def load_pretrained_models_json():
    """
    Loads pretrained_model.json data and returns it.
    """
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, 'pretrained_models.json')) as json_file:
        data = json.load(json_file)

        return data['pretrained_models']

class PretrainedModelInformation:

    def __init__(
        self,
        model_name: str,
        creator: str,
        model_architecture: str,
        description: str,
        model_size: int,
        wget_url: str,
        sha1: str
    ):
        # constructor args
        self.model_name = model_name
        self.creator = creator
        self.model_architecture = model_architecture
        self.description = description
        self.model_size = model_size
        self.wget_url = wget_url
        self.sha1 = sha1

        # meta data
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.model_path = os.path.join(project_root, 'pretrained', self.model_name + '.pt')
        os.makedirs(os.path.join(project_root, 'pretrained'), exist_ok=True)
        self.download_attempts = 0

    def download_model(self) -> bool:
        """
        Downloads model and checks if it's valid. If it fails the download will be attempted once more before False is returned indicating that the download failed.
        """
        while self.download_attempts < 2:
            if self.is_downloaded():
                if self.is_valid():
                    return True
                else:
                    print(f'Deleting download of {self.model_name} sha1 sum failed.')
                    os.remove(self.model_path)

            print(f'Attempting to download {self.model_name} from {self.wget_url}')
            sys_call = subprocess.run(["wget", "--progress=bar:force", "-O", self.model_path, self.wget_url], capture_output=True)
            print(sys_call)
            
            self.download_attempts += 1
        return False

    def is_downloaded(self) -> bool:
        return os.path.isfile(self.model_path)

    def is_valid(self) -> bool:
        sha1_hash_val = sha1_hash(self.model_path)
        return sha1_hash_val == self.sha1


def get_pretrained_model_from_name(
    to_load_name: str,
    verbose: bool = False
) -> str:
    """
    Loads information about avialible models and performs downloads if nesscary. Will return the PretrainedModelInformation which has model_path attribute if successful.

    Raises an exception if:
        - pretrained_name is not defined in pretrained_models.json
        - sha1 check fails more than once (covers unsuccessful downloads)
    """

    # construct model list first time this function is called
    if not hasattr(get_pretrained_model_from_name, "model_list"):
        pretrained_models = load_pretrained_models_json()

        get_pretrained_model_from_name.model_list = []

        for pretrained in pretrained_models:

            get_pretrained_model_from_name.model_list.append(PretrainedModelInformation(
                pretrained['model_name'],
                pretrained['creator'],
                pretrained['model_architecture'],
                pretrained['description'],
                pretrained['model_size'],
                pretrained['wget_url'],
                pretrained['sha1']
            ))

    assert len(get_pretrained_model_from_name.model_list) == len(pretrained_models)

    to_load_index = None
    for i in range(len(get_pretrained_model_from_name.model_list)):
        if to_load_name == get_pretrained_model_from_name.model_list[i].model_name:
            to_load_index = i
            break

    if to_load_index is None:
        raise ModelNameNotFoundException()

    ret_model = get_pretrained_model_from_name.model_list[to_load_index]

    if not ret_model.download_model():
        raise Exception(f'Download failed!')

    return ret_model
