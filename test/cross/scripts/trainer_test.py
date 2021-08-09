import subprocess
import os

import pytest

def test_trainer_help():
    trainer_script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../..', 'scripts/trainer.py')
    p = subprocess.run(["python", trainer_script_path, "--help"])
    assert p.returncode == 0    
