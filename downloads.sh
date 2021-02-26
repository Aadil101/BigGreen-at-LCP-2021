#!/usr/bin/env bash
############################################################## 
# This script is used to download resources for experiments
############################################################## 

cd macbook
pip install -r requirements.txt
sh downloads.sh

cd ../colab
pip install -r requirements.txt
sh downloads.sh

cd ../discovery
pip install -r requirements.txt

cd ..