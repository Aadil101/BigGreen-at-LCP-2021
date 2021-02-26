import argparse
import json
import os
import torch
import sys
from torch.utils.data import DataLoader
from data_utils.task_def import TaskType
from experiments.exp_def import TaskDefs, EncoderModelType
from torch.utils.data import Dataset, DataLoader, BatchSampler
from mt_dnn.batcher import SingleTaskDataset, Collater
from mt_dnn.model import MTDNNModel
from data_utils.metrics import calc_metrics
from mt_dnn.inference import eval_model
import tasks
from experiments.exp_def import TaskDef, EncoderModelType
from pretrained_models import *
import urllib
import requests
#!test -d bertviz_repo && echo "FYI: bertviz_repo directory already exists, to pull latest version uncomment this line: !rm -r bertviz_repo"
# !rm -r bertviz_repo # Uncomment if you need a clean pull from repo
#!test -d bertviz_repo || git clone https://github.com/jessevig/bertviz bertviz_repo
if not '/content/gdrive/My Drive/Colab Notebooks/cs99/bertviz_repo' in sys.path:
  sys.path += ['/content/gdrive/My Drive/Colab Notebooks/cs99/bertviz_repo']
from bertviz import head_view
import IPython
from IPython.display import Javascript
from tqdm import tqdm
from bertviz.transformers_neuron_view import BertConfig, BertTokenizer

def call_html():
  display(IPython.core.display.HTML('''
        <script src="/static/components/requirejs/require.js"></script>
        <script>
          requirejs.config({
            paths: {
              base: '/static/base',
              "d3": "https://cdnjs.cloudflare.com/ajax/libs/d3/3.5.8/d3.min",
              jquery: '//ajax.googleapis.com/ajax/libs/jquery/2.0.0/jquery.min',
            },
          });
        </script>
        '''))

def expand_cell():
  display(Javascript('''google.colab.output.setIframeHeight(0, true, {maxHeight: 5000})'''))

def listRightIndex(alist, value):
    return len(alist) - alist[-1::-1].index(value)

def load_model_for_viz_2(task_def_path, checkpoint_path, model_type='bert-base-cased', do_lower_case=False, use_cuda=True):
  # load task info
  task = os.path.splitext(os.path.basename(task_def_path))[0]
  task_defs = TaskDefs(task_def_path)
  assert task in task_defs._task_type_map
  assert task in task_defs._data_type_map
  assert task in task_defs._metric_meta_map
  prefix = task.split('_')[0]
  task_def = task_defs.get_task_def(prefix)
  data_type = task_defs._data_type_map[task]
  task_type = task_defs._task_type_map[task]
  metric_meta = task_defs._metric_meta_map[task]
  # load model
  assert os.path.exists(checkpoint_path)
  if use_cuda:
    state_dict = torch.load(checkpoint_path)
  else:
    state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
  config = state_dict['config']
  config["cuda"] = use_cuda
  task_def = task_defs.get_task_def(prefix)
  task_def_list = [task_def]
  config['task_def_list'] = task_def_list
  ## temp fix
  config['fp16'] = False
  config['answer_opt'] = 0
  config['adv_train'] = False
  del state_dict['optimizer']
  config['output_attentions'] = True
  config['output_hidden_states'] = True
  config['local_rank'] = -1
  encoder_type = config.get('encoder_type', EncoderModelType.BERT)
  root = os.path.basename(task_def_path)
  literal_model_type = model_type.split('-')[0].upper()
  encoder_model = EncoderModelType[literal_model_type]
  literal_model_type = literal_model_type.lower()
  mt_dnn_suffix = literal_model_type
  if 'base' in model_type:
      mt_dnn_suffix += "_base"
  elif 'large' in model_type:
      mt_dnn_suffix += "_large"
  # load config and tokenizer
  config = BertConfig.from_dict(config)
  config_class, model_class, tokenizer_class = MODEL_CLASSES[literal_model_type]
  tokenizer = tokenizer_class.from_pretrained(model_type, do_lower_case=do_lower_case)
  return config, tokenizer

def load_model_for_viz_1(task_def_path, checkpoint_path, input_path, model_type='bert-base-cased', do_lower_case=False, use_cuda=True):
  # load task info
  task = os.path.splitext(os.path.basename(task_def_path))[0]
  task_defs = TaskDefs(task_def_path)
  assert task in task_defs._task_type_map
  assert task in task_defs._data_type_map
  assert task in task_defs._metric_meta_map
  prefix = task.split('_')[0]
  task_def = task_defs.get_task_def(prefix)
  data_type = task_defs._data_type_map[task]
  task_type = task_defs._task_type_map[task]
  metric_meta = task_defs._metric_meta_map[task]
  # load model
  assert os.path.exists(checkpoint_path)
  state_dict = torch.load(checkpoint_path)
  config = state_dict['config']
  config["cuda"] = use_cuda
  device = torch.device("cuda" if use_cuda else "cpu")
  task_def = task_defs.get_task_def(prefix)
  task_def_list = [task_def]
  config['task_def_list'] = task_def_list
  ## temp fix
  config['fp16'] = False
  config['answer_opt'] = 0
  config['adv_train'] = False
  #del state_dict['optimizer']
  config['output_attentions'] = True
  config['local_rank'] = -1
  model = MTDNNModel(config, device, state_dict=state_dict)
  encoder_type = config.get('encoder_type', EncoderModelType.BERT)
  root = os.path.basename(task_def_path)
  literal_model_type = model_type.split('-')[0].upper()
  encoder_model = EncoderModelType[literal_model_type]
  literal_model_type = literal_model_type.lower()
  mt_dnn_suffix = literal_model_type
  if 'base' in model_type:
      mt_dnn_suffix += "_base"
  elif 'large' in model_type:
      mt_dnn_suffix += "_large"
  # load tokenizer
  config_class, model_class, tokenizer_class = MODEL_CLASSES[literal_model_type]
  tokenizer = tokenizer_class.from_pretrained(model_type, do_lower_case=do_lower_case)
  # load data
  prep_input = input_path
  test_data_set = SingleTaskDataset(prep_input, False, maxlen=512, task_id=0, task_def=task_def)
  collater = Collater(is_train=False, encoder_type=encoder_type)
  test_data = DataLoader(test_data_set, batch_size=1, collate_fn=collater.collate_fn, pin_memory=True)
  idx = 0
  results = []
  for batch_meta, batch_data in tqdm(test_data):
    if idx < 360:
      idx += 1
      continue   
    batch_meta, batch_data = Collater.patch_data(device, batch_meta, batch_data)
    model.network.eval()
    task_id = batch_meta['task_id']
    task_def = TaskDef.from_dict(batch_meta['task_def'])
    task_type = task_def.task_type
    task_obj = tasks.get_task_obj(task_def)
    inputs = batch_data[:batch_meta['input_len']]
    if len(inputs) == 3:
      inputs.append(None)
      inputs.append(None)
    inputs.append(task_id)
    input_ids = inputs[0]
    token_type_ids = inputs[1]
    attention = model.mnetwork.module.bert(input_ids, token_type_ids=token_type_ids)[-1]
    batch_size = batch_data[0].shape[0]
    for i in range(batch_size):
      attention = tuple([item[i:i+1,:,:,:] for item in attention])
      input_id_list = input_ids[i].tolist()
      tokens = tokenizer.convert_ids_to_tokens(input_id_list)
      idx_sep = listRightIndex(tokens, '[SEP]')+1
      tokens = tokens[:idx_sep]
      attention = tuple([item[:, :, :idx_sep, :idx_sep] for item in attention])
      results.append((attention, tokens))
    idx += batch_size 
  return results

def load_model_for_viz_0(task_def_path, checkpoint_path, input_path, model_type='bert-base-cased', do_lower_case=False, use_cuda=True):
    # load task info
  task = os.path.splitext(os.path.basename(task_def_path))[0]
  task_defs = TaskDefs(task_def_path)
  assert task in task_defs._task_type_map
  assert task in task_defs._data_type_map
  assert task in task_defs._metric_meta_map
  prefix = task.split('_')[0]
  task_def = task_defs.get_task_def(prefix)
  data_type = task_defs._data_type_map[task]
  task_type = task_defs._task_type_map[task]
  metric_meta = task_defs._metric_meta_map[task]
  # load model
  assert os.path.exists(checkpoint_path)
  state_dict = torch.load(checkpoint_path)
  config = state_dict['config']
  config["cuda"] = use_cuda
  task_def = task_defs.get_task_def(prefix)
  task_def_list = [task_def]
  config['task_def_list'] = task_def_list
  ####### temp fix #######
  config['fp16'] = False
  config['answer_opt'] = 0
  config['adv_train'] = False
  del state_dict['optimizer']
  #########################
  model = MTDNNModel(config, state_dict=state_dict)
  encoder_type = config.get('encoder_type', EncoderModelType.BERT)
  root = os.path.basename(task_def_path)
  literal_model_type = model_type.split('-')[0].upper()
  encoder_model = EncoderModelType[literal_model_type]
  literal_model_type = literal_model_type.lower()
  mt_dnn_suffix = literal_model_type
  if 'base' in model_type:
      mt_dnn_suffix += "_base"
  elif 'large' in model_type:
      mt_dnn_suffix += "_large"
  # load tokenizer
  config_class, model_class, tokenizer_class = MODEL_CLASSES[literal_model_type]
  tokenizer = tokenizer_class.from_pretrained(model_type, do_lower_case=do_lower_case)
  # load data
  prep_input = input_path
  test_data_set = SingleTaskDataset(prep_input, False, maxlen=512, task_id=0, task_def=task_def)
  collater = Collater(is_train=False, encoder_type=encoder_type)
  test_data = DataLoader(test_data_set, batch_size=1, collate_fn=collater.collate_fn, pin_memory=True)
  idx = 0
  results = []
  return model.mnetwork.module.bert, config, test_data

def get_indices_excluding_separator_tokens(sentence, aspect_word, tokenizer):
  tokens = tokenizer.tokenize(sentence)
  result = list(range(len(tokens)+1))
  tokens = tokenizer.tokenize(aspect_word)
  result.extend(list(range(result[-1]+2, result[-1]+2+len(tokens))))
  return result

def get_tf(word, exact=False):
    encoded_query = urllib.parse.quote(word) 
    params = {'corpus': 'eng-us', 'query': encoded_query, 'format': 'json'} 
    params = '&'.join('{}={}'.format(name, value) for name, value in params.items()) 
    response = requests.get('https://api.phrasefinder.io/search?' + params) 
    response = json.loads(response.text)
    if 'phrases' not in response:
        return 0
    if exact:
        word_lst = word.split(' ')
        tks_lst = [(i, [tk['tt'] for tk in phrase['tks']]) for i, phrase in enumerate(response['phrases'])]
        for i, tks in tks_lst:
            if tks == word_lst:
                return response['phrases'][i]['mc']
        else:
            return 0
    else:
        return sum([phrase['mc'] for phrase in response['phrases']])