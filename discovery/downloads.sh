#!/usr/bin/env bash
############################################################## 
# This script is used to download resources for experiments
############################################################## 

## Get Gigaword-5 corpus

gigaword_eng_DIR=$(pwd)/gigaword_eng
echo ${gigaword_eng_DIR}
if [ ! -d "${gigaword_eng_DIR}"  ]; then
    echo "Please see the following link to obtain the Gigaword-5 corpus. https://catalog.ldc.upenn.edu/LDC2003T05. You should recieve a file called 'gigaword_eng_LDC2003T05.tgz'"
    exit 1
fi

## Unzip Gigaword-5 corpus

gunzip -r "${gigaword_eng_DIR}"

## Process Gigaword-5 corpus

gigaword_txt_DIR=$(pwd)/gigaword_txt
if [ ! -d "${gigaword_txt_DIR}"  ]; then
    echo "Process Gigaword-5 corpus."
    mkdir "${gigaword_txt_DIR}"
fi
pip install tqdm
pip install bs4
pwd=$(pwd)
python build_gigaword_dir.py --pwd "${pwd}"