import os

def test_tpu_available():

    assert('TPU_IP_ADDRESS' in os.environ)
    assert('XRT_TPU_CONFIG' in os.environ)

