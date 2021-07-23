import re
import os

os.environ['USE_CPU_OP'] = '1'
print(f'USE_CPU_OP set to: {os.environ["USE_CPU_OP"]}')

tpu_ip = re.match('grpc\://((\d{1,3}\.){3}\d{1,3})\:\d{4}',
             os.environ.get('TPU_NAME')).group(1)
os.environ['TPU_IP_ADDRESS'] = tpu_ip

if 'COLAB_TPU_ADDR' in os.environ:
    print('Setting up for Colab TPUs')
    os.environ['XRT_TPU_CONFIG'] = f"tpu_worker;0;{os.environ['COLAB_TPU_ADDR']}"
    print(f'COLAB_TPU_ADDR is: {os.environ["COLAB_TPU_ADDR"]}')
else:
    print('Setting up for Google Cloud TPUs')

print(f'TPU_IP_ADDRESS set to: {os.environ["TPU_IP_ADDRESS"]}')
print(f'XRT_TPU_CONFIG set to: {os.environ["XRT_TPU_CONFIG"]}')