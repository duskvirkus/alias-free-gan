import re
import os


def test_tpu_script():

    # run tpu_script.py
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../scripts/tpu_setup.py')
    exec(open(script_path).read())

    assert(os.environ['USE_CPU_OP'] == '1')

    tpu_ip = os.environ["TPU_IP_ADDRESS"]
    tpu_ip_match = re.match('\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', tpu_ip)
    assert(tpu_ip_match is not None)

    xrt_config = os.environ["XRT_TPU_CONFIG"]
    xrt_config_match = re.match('tpu_worker;0;\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}:\d{1,4}', xrt_config)
    assert(xrt_config_match is not None)
