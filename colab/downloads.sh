#!/usr/bin/env bash
############################################################## 
# This script is used to download resources for experiments
############################################################## 

## Download attention analysis library

attn_analysis_DIR=$(pwd)/attention-analysis
if [ ! -d ${attn_analysis_DIR}  ]; then
    echo "Create attention analysis library"
    git clone https://github.com/clarkkev/attention-analysis.git ${attn_analysis_DIR}
fi

## Download BertViz library

bertviz_DIR=$(pwd)/bertviz
if [ ! -d ${bertviz_DIR}  ]; then
    echo "Create BertViz library"
    git clone https://github.com/jessevig/bertviz.git ${bertviz_DIR}
fi

## Download apex from source

apex_DIR=$(pwd)/apex
if [ ! -d ${apex_DIR}  ]; then
    echo "Create apex from source"
    git clone https://github.com/NVIDIA/apex.git ${apex_DIR}
fi

## Download MT-DNN requirements

mt_dnn_DIR=$(pwd)/new-mt-dnn
pip install -r "${mt_dnn_DIR}/requirements.txt"
pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
pip install transformers==3.5.1
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" $apex_DIR