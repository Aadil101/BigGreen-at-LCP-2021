#!/usr/bin/env bash
############################################################## 
# This script is used to download resources for experiments
############################################################## 

## Download ELMo models

ELMo_DIR=$(pwd)/lib/ELMo
if [ ! -d ${ELMo_DIR}  ]; then
    echo "Create folder ELMo_DIR"
    mkdir ${ELMo_DIR}
fi

wget -nc -q https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json -O "${ELMo_DIR}/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json"
wget -nc -q https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5 -O "${ELMo_DIR}/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5"

## Download infersent models

encoder_DIR=$(pwd)/lib/encoder
if [ ! -d ${encoder_DIR}  ]; then
    echo "Create folder encoder_DIR"
    mkdir ${encoder_DIR}
fi

wget -nc -q https://dl.fbaipublicfiles.com/infersent/infersent1.pkl -O "${encoder_DIR}/infersent1.pkl"
wget -nc -q https://dl.fbaipublicfiles.com/infersent/infersent2.pkl -O "${encoder_DIR}/infersent2.pkl"

infersent_DIR=$(pwd)/lib/InferSent/
if [ ! -d "${infersent_DIR}"  ]; then
    echo "Clone InferSent repo"
    git clone https://github.com/facebookresearch/InferSent.git ${infersent_DIR}
fi

## Download GloVe pretrained word embeddings

glove_DIR=$(pwd)/lib/glove
if [ ! -d ${glove_DIR}  ]; then
    echo "Create folder glove_DIR"
    mkdir ${glove_DIR}
fi

if [ ! -f "${glove_DIR}/glove.6B.300d.txt" ]; then
    if [ ! -f "${glove_DIR}/glove.6B.zip" ]; then
        wget -nc -q http://nlp.stanford.edu/data/glove.6B.zip -O "${glove_DIR}/glove.6B.zip"
    fi
    unzip "${glove_DIR}/glove.6B.zip" -d "${glove_DIR}/glove.6B.300d.txt"
    rm "${glove_DIR}/glove.6B.zip"
fi

## Download Stanford CoreNLP

lib=$(pwd)/lib
if [ ! -d "${lib}/stanford-corenlp-4.2.0"  ]; then
    if [ ! -f "${lib}/stanford-corenlp-latest.zip" ]; then
        wget -nc -q https://nlp.stanford.edu/software/stanford-corenlp-latest.zip -O "${lib}/stanford-corenlp-4.2.0.zip"
    fi
    unzip "${lib}/stanford-corenlp-4.2.0.zip" -d "${lib}"
fi

