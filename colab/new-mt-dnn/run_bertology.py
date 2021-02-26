#!/usr/bin/env python3
# Copyright 2018 CMU and The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Bertology: this script shows how you can explore the internals of the models in the library to:
    - compute the entropy of the head attentions
    - compute the importance of each head
    - prune (remove) the low importance head.
    Some parts of this script are adapted from the code of Michel et al. (http://arxiv.org/abs/1905.10650)
    which is available at https://github.com/pmichel31415/are-16-heads-really-better-than-1
"""
import argparse
import logging
import os
from datetime import datetime

import numpy as np
import pandas as pd
import json
import torch
from torch.utils.data import DataLoader, SequentialSampler, Subset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    GlueDataset,
    default_data_collator,
    glue_compute_metrics,
    glue_output_modes,
    glue_processors,
    set_seed,
)
from transformers.trainer_utils import is_main_process
from experiments.exp_def import TaskDefs, EncoderModelType
from pretrained_models import *
from mt_dnn.model import MTDNNModel
from mt_dnn.batcher import SingleTaskDataset, MultiTaskDataset, Collater, MultiTaskBatchSampler, DistMultiTaskBatchSampler, DistSingleTaskBatchSampler
from experiments.exp_def import TaskDef
import tasks
from scipy.stats import pearsonr
from mt_dnn.loss import *
import pickle as pkl

def entropy(p):
    """ Compute the entropy of a probability distribution """
    plogp = p * torch.log(p)
    plogp[p == 0] = 0
    return -plogp.sum(dim=-1)

def print_2d_tensor(tensor):
    """ Print a 2D tensor """
    print("lv, h >\t" + "\t".join(f"{x + 1}" for x in range(len(tensor))))
    for row in range(len(tensor)):
        if tensor.dtype != torch.long:
            print(f"layer {row + 1}:\t" + "\t".join(f"{x:.5f}" for x in tensor[row].cpu().data))
        else:
            print(f"layer {row + 1}:\t" + "\t".join(f"{x:d}" for x in tensor[row].cpu().data))

def compute_heads_importance(
    args, model, eval_dataloader, compute_entropy=True, compute_importance=True, head_mask=None, actually_pruned=False, verbose=True
):
    """This method shows how to compute:
    - head attention entropy
    - head importance scores according to http://arxiv.org/abs/1905.10650
    """
    # Prepare our tensors
    device = torch.device("cuda" if args['cuda'] else "cpu")
    n_layers = model.mnetwork.module.bert.config.num_hidden_layers
    n_heads = model.mnetwork.module.bert.config.num_attention_heads
    head_importance = torch.zeros(n_layers, n_heads).to(device)
    attn_entropy = torch.zeros(n_layers, n_heads).to(device)

    if head_mask is None:
        head_mask = torch.ones(n_layers, n_heads).to(device)
    
    head_mask.requires_grad_(requires_grad=True)
    # If actually pruned attention multi-head, set head mask to None to avoid shape mismatch
    if actually_pruned:
        head_mask = None

    preds = None
    labels = None
    tot_tokens = 0.0

    for batch_meta, batch_data in tqdm(eval_dataloader):
        for i in range(len(batch_data[1])):
            batch_data[1][i] = batch_data[1][i].to(device)
        y = batch_data[batch_meta['label']]
        y = model._to_cuda(y)

        # Do a forward pass (not with torch.no_grad() since we need gradients for importance score - see below)
        batch_meta, batch_data = Collater.patch_data(device, batch_meta, batch_data)
        logits, attention = model.update(batch_meta, batch_data, head_mask=head_mask)

        if compute_entropy:
            for layer, attn in enumerate(attention):
                masked_entropy = entropy(attn.detach()) * batch_data[batch_meta['mask']].float().unsqueeze(1)
                attn_entropy[layer] += masked_entropy.sum(-1).sum(0).detach()
        
        if compute_importance:
            head_importance += head_mask.grad.abs().detach()

        # Also store our logits/labels if we want to compute metrics afterwards
        if preds is None:
            preds = logits.detach().cpu().numpy()
            labels = y.detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            labels = np.append(labels, y.detach().cpu().numpy(), axis=0)

        tot_tokens += batch_data[batch_meta['mask']].float().detach().sum().data

    # Normalize
    attn_entropy /= tot_tokens
    head_importance /= tot_tokens
    # Layerwise importance normalization
    if not args['dont_normalize_importance_by_layer']:
        exponent = 2
        norm_by_layer = torch.pow(torch.pow(head_importance, exponent).sum(-1), 1 / exponent)
        head_importance /= norm_by_layer.unsqueeze(-1) + 1e-20

    if not args['dont_normalize_global_importance']:
        head_importance = (head_importance - head_importance.min()) / (head_importance.max() - head_importance.min())

    # Print/save matrices
    np.save(os.path.join(args['output_dir'], "attn_entropy.npy"), attn_entropy.detach().cpu().numpy())
    np.save(os.path.join(args['output_dir'], "head_importance.npy"), head_importance.detach().cpu().numpy())

    if verbose:
      print("Attention entropies")
      print_2d_tensor(attn_entropy)
      print("Head importance scores")
      print_2d_tensor(head_importance)
      print("Head ranked by importance scores")
    head_ranks = torch.zeros(head_importance.numel(), dtype=torch.long, device=device)
    head_ranks[head_importance.view(-1).sort(descending=True)[1]] = torch.arange(
        head_importance.numel(), device=device
    )
    head_ranks = head_ranks.view_as(head_importance)
    if verbose:
      print_2d_tensor(head_ranks)

    return attn_entropy, head_importance, preds, labels

def mask_heads(args, model, eval_dataloader):
    """This method shows how to mask head (set some heads to zero), to test the effect on the network,
    based on the head importance scores, as described in Michel et al. (http://arxiv.org/abs/1905.10650)
    """
    _, head_importance, preds, labels = compute_heads_importance(args, model, eval_dataloader, compute_entropy=False)
    preds = np.squeeze(preds)
    original_score = pearsonr(preds, labels)[0]
    print("Pruning: original score: %f, threshold: %f", original_score, original_score * args['masking_threshold'])

    new_head_mask = torch.ones_like(head_importance)
    num_to_mask = max(1, int(new_head_mask.numel() * args['masking_amount']))

    current_score = original_score
    while current_score >= original_score * args['masking_threshold']:
        head_mask = new_head_mask.clone()  # save current head mask
        # heads from least important to most - keep only not-masked heads
        head_importance[head_mask == 0.0] = float("Inf")
        current_heads_to_mask = head_importance.view(-1).sort()[1]

        if len(current_heads_to_mask) <= num_to_mask:
            break

        # mask heads
        current_heads_to_mask = current_heads_to_mask[:num_to_mask]
        print("Heads to mask: %s", str(current_heads_to_mask.tolist()))
        new_head_mask = new_head_mask.view(-1)
        new_head_mask[current_heads_to_mask] = 0.0
        new_head_mask = new_head_mask.view_as(head_mask)
        new_head_mask = new_head_mask.clone().detach()
        print_2d_tensor(new_head_mask)

        # Compute metric and head importance again
        _, head_importance, preds, labels = compute_heads_importance(
            args, model, eval_dataloader, compute_entropy=False, head_mask=new_head_mask
        )
        preds = np.squeeze(preds)
        current_score = pearsonr(preds, labels)[0]
        print("Masking: current score: %f, remaining heads %d (%.1f percents)", current_score, new_head_mask.sum(), new_head_mask.sum() / new_head_mask.numel() * 100)

    print("Final head mask")
    print_2d_tensor(head_mask)
    np.save(os.path.join(args['output_dir'], "head_mask.npy"), head_mask.detach().cpu().numpy())

    return head_mask.detach()


def prune_heads(args, model, eval_dataloader, head_mask):
    """This method shows how to prune head (remove heads weights) based on
    the head importance scores as described in Michel et al. (http://arxiv.org/abs/1905.10650)
    """
    # Try pruning and test time speedup
    # Pruning is like masking but we actually remove the masked weights
    before_time = datetime.now()
    _, _, preds, labels = compute_heads_importance(
        args, model, eval_dataloader, compute_entropy=False, compute_importance=False, head_mask=head_mask
    )
    preds = np.squeeze(preds)
    score_masking = pearsonr(preds, labels)[0]
    original_time = datetime.now() - before_time

    original_num_params = sum(p.numel() for p in model.mnetwork.parameters())
    heads_to_prune = dict(
        (layer, (1 - head_mask[layer].long()).nonzero().squeeze().tolist()) for layer in range(len(head_mask))
    )

    assert sum(len(h) for h in heads_to_prune.values()) == (1 - head_mask.long()).sum().item()
    model.mnetwork.module.bert.prune_heads(heads_to_prune)
    pruned_num_params = sum(p.numel() for p in model.mnetwork.parameters())

    before_time = datetime.now()
    _, _, preds, labels = compute_heads_importance(
        args,
        model,
        eval_dataloader,
        compute_entropy=False,
        compute_importance=False,
        head_mask=None,
        actually_pruned=True,
    )
    preds = np.squeeze(preds)
    score_pruning = pearsonr(preds, labels)[0]
    new_time = datetime.now() - before_time

    print("Pruning: original num of params: %.2e, after pruning %.2e (%.1f percents)", original_num_params, pruned_num_params, pruned_num_params / original_num_params * 100)
    print("Pruning: score with masking: %f score with pruning: %f", score_masking, score_pruning)
    print("Pruning: speed ratio (new timing / original timing): %f percents", original_time / new_time * 100)

def initialize_distributed(args):
    """Initialize torch.distributed."""
    args['rank'] = int(os.getenv('RANK', '0'))
    args['world_size'] = int(os.getenv("WORLD_SIZE", '1'))

    if os.getenv('OMPI_COMM_WORLD_LOCAL_RANK'):
        # We are using (OpenMPI) mpirun for launching distributed data parallel processes
        local_rank = int(os.getenv('OMPI_COMM_WORLD_LOCAL_RANK'))
        local_size = int(os.getenv('OMPI_COMM_WORLD_LOCAL_SIZE'))
        args['local_rank'] = local_rank
        args['rank'] = nodeid * local_size + local_rank
        args['world_size'] = num_nodes * local_size
    #args.batch_size = args.batch_size * args.world_size

    device = args['rank'] % torch.cuda.device_count()
    if args['local_rank'] is not None:
        device = args['local_rank']
    torch.cuda.set_device(device)
    device = torch.device('cuda', args['local_rank'])
    # Call the init process
    init_method = 'tcp://'
    master_ip = os.getenv('MASTER_ADDR', 'localhost')
    master_port = os.getenv('MASTER_PORT', '6600')
    init_method += master_ip + ':' + master_port
    torch.distributed.init_process_group(
        backend=args['backend'],
        world_size=args['world_size'], rank=args['rank'],
        init_method=init_method)
    return device

def main():
    parser = argparse.ArgumentParser()
    #   Required parameters
    parser.add_argument("--task_def", type=str, required=True, default="experiments/glue/glue_task_def.yml")
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--task_id", type=int, default=0, help="the id of this task when training")
    parser.add_argument("--checkpoint", default='mt_dnn_models/bert_model_base_uncased.pt', type=str)
    parser.add_argument(
        "--output_dir",
        default='/content/gdrive/My Drive/Colab Notebooks/cs99/mt-dnn/checkpoints/bert-cased_lcp-single_2020-12-23T2029/',
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--prep_input", 
        default='/content/gdrive/My Drive/Colab Notebooks/cs99/mt-dnn/data_complex/bert_base_cased/lcp_dev.json',
        type=str,
        required=True,
    )
    parser.add_argument(
        '--bert_model_type', 
        default='bert-base-cased',
        type=str, 
        help="What type of bert model should we be using",
    )

    # Other parameters
    parser.add_argument(
        "--config_name",
        default="",
        type=str,
        help="Pretrained config name or path if not the same as model_name_or_path",
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name_or_path",
    )
    parser.add_argument(
        "--cache_dir",
        default=None,
        type=str,
        help="Where do you want to store the pre-trained models downloaded from huggingface.co",
    )
    parser.add_argument(
        "--data_subset", type=int, default=-1, help="If > 0: limit the data to a subset of data_subset instances."
    )
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Whether to overwrite data in output directory"
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )

    parser.add_argument(
        "--dont_normalize_importance_by_layer", action="store_true", help="Don't normalize importance score by layers"
    )
    parser.add_argument(
        "--dont_normalize_global_importance",
        action="store_true",
        help="Don't normalize all importance scores between 0 and 1",
    )

    parser.add_argument(
        "--try_masking", action="store_true", help="Whether to try to mask head until a threshold of accuracy."
    )
    parser.add_argument(
        "--masking_threshold",
        default=0.9,
        type=float,
        help="masking threshold in term of metrics (stop masking when metric < threshold * original metric value).",
    )
    parser.add_argument(
        "--masking_amount", default=0.1, type=float, help="Amount to heads to masking at each masking step."
    )
    parser.add_argument("--metric_name", default="acc", type=str, help="Metric to use for head masking.")

    parser.add_argument(
        "--max_seq_length",
        default=512,
        type=int,
        help="The maximum total input sequence length after WordPiece tokenization. \n"
        "Sequences longer than this will be truncated, sequences shorter padded.",
    )
    # temp fix: technically these parameters should've already bin in checkpoint's config...
    parser.add_argument("--world_size", type=int, default=1, help="For distributed training: world size")

    parser.add_argument("--batch_size", default=8, type=int, help="Batch size.")
    parser.add_argument("--seed", type=int, default=2018)
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available(), help='whether to use GPU acceleration.')
    parser.add_argument("--server_ip", type=str, default="", help="Can be used for distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="Can be used for distant debugging.")
    parser.add_argument("--do_proper", type=str, default=False, help="Can be used for distant debugging.")
    parser.add_argument("--do_improper", type=str, default=False, help="Can be used for distant debugging.")
    args = parser.parse_args()

    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup devices and distributed training
    device = torch.device("cuda")
    if args.local_rank > -1:
        device = initialize_distributed(args)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    # load task info
    task = args.task
    task_defs = TaskDefs(args.task_def)
    assert args.task in task_defs._task_type_map
    assert args.task in task_defs._data_type_map
    assert args.task in task_defs._metric_meta_map
    prefix = task.split('_')[0]
    task_def = task_defs.get_task_def(prefix)
    data_type = task_defs._data_type_map[args.task]
    task_type = task_defs._task_type_map[args.task]
    metric_meta = task_defs._metric_meta_map[args.task]
    # load model
    checkpoint_path = args.checkpoint
    assert os.path.exists(checkpoint_path)
    if args.cuda:
        state_dict = torch.load(checkpoint_path)
    else:
        state_dict = torch.load(checkpoint_path, map_location="cpu")
    opt = state_dict['config']
    args.bin_on = False
    opt.update(vars(args))
    model = MTDNNModel(opt, device=device, state_dict=state_dict)

    # Load pretrained model and tokenizer
    # Load data
    data = pd.read_csv('data_complex/lcp_test.tsv', sep='\t', header=None, names=['idx', 'complexity', 'sentence', 'token'])
    data['complexity'] = np.load('/content/gdrive/My Drive/Colab Notebooks/cs99/from_macbook/single_test_labels.npy')
    data['class'] = pd.cut(data['complexity'], labels=[1,2,3,4,5], bins=[0,0.2,0.4,0.6,0.8,1], include_lowest=True)
    data['sent_len'] = data['sentence'].str.len()
    with open('/content/gdrive/My Drive/Colab Notebooks/cs99/new-mt-dnn/checkpoints/bert-cased_lcp-single_2021-01-19T0309/lcp_test_scores_epoch_4.json', 'r') as file:
        single_dev_bert_scores = json.load(file)
        data['finetuned_complexity'] = single_dev_bert_scores['scores']
        data['finetuned_error'] = data['finetuned_complexity']-data['complexity']
        data['finetuned_abs_error'] = (data['finetuned_complexity']-data['complexity']).abs()
    with open('/content/gdrive/My Drive/Colab Notebooks/cs99/new-mt-dnn/checkpoints/bert-cased_lcp-single_2021-01-19T0309/pretrained.json', 'r') as file:
        single_dev_bert_scores = json.load(file)
        data['pretrained_complexity'] = single_dev_bert_scores['scores']
        data['pretrained_error'] = data['pretrained_complexity']-data['complexity']
        data['pretrained_abs_error'] = (data['pretrained_complexity']-data['complexity']).abs()
    data['improvement'] = data['pretrained_abs_error']-data['finetuned_abs_error']
    data['proper'] = data['token'].apply(lambda x: x[0].isupper())
    # Distributed training:
    # download model & vocab.
    printable = opt['local_rank'] in [-1, 0]
    encoder_type = opt.get('encoder_type', EncoderModelType.BERT)
    collater = Collater(is_train=True, encoder_type=encoder_type, max_seq_len=opt['max_seq_len'], do_padding=opt['do_padding'])
    dev_data = SingleTaskDataset(opt['prep_input'], True, maxlen=opt['max_seq_len'], task_id=opt['task_id'], task_def=task_def, printable=printable)
    if args.do_proper:
      dev_data._data = np.array(dev_data._data)[data[data['proper']]['idx'].to_numpy()].tolist()
    if args.do_improper:
      dev_data._data = np.array(dev_data._data)[data[~data['proper']]['idx'].to_numpy()].tolist()
    dev_data_loader = DataLoader(dev_data, batch_size=opt['batch_size_eval'], collate_fn=collater.collate_fn, pin_memory=opt['cuda'])
    
    # Compute head entropy and importance score
    results = []
    for seed in tqdm(range(2010+1, 2020+1)):# Set seeds
      set_seed(seed)
      attn_entropy, head_importance, preds, labels = compute_heads_importance(opt, model, dev_data_loader)
      results.append((attn_entropy, head_importance))
    pkl.dump(results, open('checkpoints/bert-cased_lcp-single_2021-01-19T0309/results.pkl', 'wb'))

    # Try head masking (set heads to zero until the score goes under a threshold)
    # and head pruning (remove masked heads and see the effect on the network)
    if args.try_masking and args.masking_threshold > 0.0 and args.masking_threshold < 1.0:
        head_mask = mask_heads(opt, model, dev_data_loader)
    #   prune_heads(opt, model, dev_data_loader, head_mask)

if __name__ == "__main__":
    main()
