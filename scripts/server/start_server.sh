#!/bin/bash

# set path and first
lightx2v_path=/Wan_models/LightX2V
model_path=/Wan_models/models/wan2.2_models

export CUDA_VISIBLE_DEVICES=0

# set environment variables
source ${lightx2v_path}/scripts/base/base.sh


# Start API server with distributed inference service
python -m lightx2v.server \
--model_cls wan2.2_moe_distill \
--task i2v \
--model_path $model_path \
--config_json ${lightx2v_path}/configs/wan22/wan_moe_i2v_distill.json \
--port 8105

echo "Service stopped"
