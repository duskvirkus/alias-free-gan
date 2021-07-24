#!/bin/bash

export USE_CPU_OP=1

[[ $TPU_NAME =~ [0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3} ]]
export TPU_IP_ADDRESS="${BASH_REMATCH[0]}"

[[ $TPU_NAME =~ [0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}:[0-9]{4} ]]
export XRT_TPU_CONFIG="tpu_worker;0;${BASH_REMATCH[0]}"

echo "USE_CPU_OP     = ${USE_CPU_OP}"
echo "TPU_IP_ADDRESS = ${TPU_IP_ADDRESS}"
echo "XRT_TPU_CONFIG = ${XRT_TPU_CONFIG}"
