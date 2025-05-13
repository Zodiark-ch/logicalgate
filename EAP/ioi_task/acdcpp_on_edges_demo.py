#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from IPython import get_ipython
ipython = get_ipython()
if ipython is not None:
    ipython.magic("%load_ext autoreload")
    ipython.magic("%autoreload 2")

import os
import sys
# 获取当前脚本所在目录的上一级目录（项目根目录）
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 将项目根目录添加到 sys.path
sys.path.append(project_root)
print(sys.path)
import torch
import re
from collections import OrderedDict
import acdc
from utils.prune_utils import get_3_caches, split_layers_and_heads
from acdc.TLACDCExperiment import TLACDCExperiment
from acdc.acdc_utils import TorchIndex, EdgeType
import numpy as np
import torch as t
from torch import Tensor
import einops
import itertools

from transformer_lens import HookedTransformer, ActivationCache

import tqdm.notebook as tqdm
import plotly
from rich import print as rprint
from rich.table import Table

from jaxtyping import Float, Bool
from typing import Callable, Tuple, Union, Dict, Optional

device = t.device('cuda') if t.cuda.is_available() else t.device('cpu')
print(f'Device: {device}')


# # Model Setup

# In[ ]:


model = HookedTransformer.from_pretrained(
    'gpt2-small',
    center_writing_weights=False,
    center_unembed=False,
    fold_ln=False,
    device=device,
)
model.set_use_hook_mlp_in(True)
model.set_use_split_qkv_input(True)
model.set_use_attn_result(True)


# # Dataset Setup

# In[ ]:


from ioi_dataset import IOIDataset, format_prompt, make_table
import json
N = 25
clean_dataset = IOIDataset(
    prompt_type='mixed',
    N=N,
    tokenizer=model.tokenizer,
    prepend_bos=False,
    seed=1,
    device=device
)
corr_dataset = clean_dataset.gen_flipped_prompts('ABC->XYZ, BAB->XYZ')

make_table(
  colnames = ["IOI prompt", "IOI subj", "IOI indirect obj", "ABC prompt"],
  cols = [
    map(format_prompt, clean_dataset.sentences),
    model.to_string(clean_dataset.s_tokenIDs).split(),
    model.to_string(clean_dataset.io_tokenIDs).split(),
    map(format_prompt, clean_dataset.sentences),
  ],
  title = "Sentences from IOI vs ABC distribution",
)


# # Metric Setup

# In[ ]:


def ave_logit_diff(
    logits: Float[Tensor, 'batch seq d_vocab'],
    ioi_dataset: IOIDataset,
    per_prompt: bool = False
):
    '''
        Return average logit difference between correct and incorrect answers
    '''
    # Get logits for indirect objects
    io_logits = logits[range(logits.size(0)), ioi_dataset.word_idx['end'], ioi_dataset.io_tokenIDs]
    s_logits = logits[range(logits.size(0)), ioi_dataset.word_idx['end'], ioi_dataset.s_tokenIDs]
    # Get logits for subject
    logit_diff = io_logits - s_logits
    return logit_diff if per_prompt else logit_diff.mean()

with t.no_grad():
    clean_logits = model(clean_dataset.toks)
    corrupt_logits = model(corr_dataset.toks)
    clean_logit_diff = ave_logit_diff(clean_logits, clean_dataset).item()
    corrupt_logit_diff = ave_logit_diff(corrupt_logits, corr_dataset).item()

def ioi_metric(
    logits: Float[Tensor, "batch seq_len d_vocab"],
    corrupted_logit_diff: float = corrupt_logit_diff,
    clean_logit_diff: float = clean_logit_diff,
    ioi_dataset: IOIDataset = clean_dataset
 ):
    patched_logit_diff = ave_logit_diff(logits, ioi_dataset)
    return (patched_logit_diff - corrupted_logit_diff) / (clean_logit_diff - corrupted_logit_diff)

def ioi_metric_denoising(
    logits: Float[Tensor, "batch seq_len d_vocab"],
    corrupted_logit_diff: float = corrupt_logit_diff,
    clean_logit_diff: float = clean_logit_diff,
    ioi_dataset: IOIDataset = corr_dataset
 ):
    patched_logit_diff = ave_logit_diff(logits, ioi_dataset)
    return (patched_logit_diff - corrupted_logit_diff) / (clean_logit_diff - corrupted_logit_diff)

def negative_ioi_metric(logits: Float[Tensor, "batch seq_len d_vocab"]):
    return -ioi_metric(logits)

def negative_ioi_metric_denoising(logits: Float[Tensor, "batch seq_len d_vocab"]):
    return -ioi_metric_denoising(logits)
    
# Get clean and corrupt logit differences
with t.no_grad():
    clean_metric = ioi_metric(clean_logits, corrupt_logit_diff, clean_logit_diff, clean_dataset)
    corrupt_metric = ioi_metric_denoising(corrupt_logits, corrupt_logit_diff, clean_logit_diff, corr_dataset)
    # clean_metric_denoising= ioi_metric_denoising(clean_logits, corrupt_logit_diff, clean_logit_diff, clean_dataset)
    # corrupt_metric_denoising = ioi_metric_denoising(corrupt_logits, corrupt_logit_diff, clean_logit_diff, corr_dataset)

print(f'Clean direction: {clean_logit_diff}, Corrupt direction: {corrupt_logit_diff}')
print(f'Clean metric: {clean_metric}, Corrupt metric: {corrupt_metric}')
#print(f'Clean metric denoising: {clean_metric_denoising}, Corrupt metric denoising: {corrupt_metric_denoising}')

# # Run Experiment

# In[ ]:


# get the 2 fwd and 1 bwd caches; cache "normalized" and "result" of attn layers
clean_cache, corrupted_cache, clean_grad_cache = get_3_caches(
    model, 
    clean_dataset.toks,
    corr_dataset.toks,
    metric=negative_ioi_metric,
    mode = "edge",
)

corrupted_cache_denoising, clean_cache_denoising, clean_grad_cache_denoising = get_3_caches(
    model, 
    corr_dataset.toks,
    clean_dataset.toks,
    metric=negative_ioi_metric_denoising,
    mode = "edge",
)


# In[ ]:


clean_head_act = split_layers_and_heads(clean_cache.stack_head_results(), model=model)
corr_head_act = split_layers_and_heads(corrupted_cache.stack_head_results(), model=model)
corr_head_act_denoising = split_layers_and_heads(corrupted_cache_denoising.stack_head_results(), model=model)
clean_head_act_denoising = split_layers_and_heads(clean_cache_denoising.stack_head_results(), model=model)

# In[ ]:


stacked_grad_act = torch.zeros(
    3, # QKV
    model.cfg.n_layers,
    model.cfg.n_heads,
    clean_head_act.shape[-3], # Batch
    clean_head_act.shape[-2], # Seq
    clean_head_act.shape[-1], # D
)

for letter_idx, letter in enumerate("qkv"):
    for layer_idx in range(model.cfg.n_layers):
        stacked_grad_act[letter_idx, layer_idx] = einops.rearrange(clean_grad_cache[f"blocks.{layer_idx}.hook_{letter}_input"], "batch seq n_heads d -> n_heads batch seq d")

stacked_grad_act_denoising = torch.zeros(
    3, # QKV
    model.cfg.n_layers,
    model.cfg.n_heads,
    clean_head_act.shape[-3], # Batch
    clean_head_act.shape[-2], # Seq
    clean_head_act.shape[-1], # D
)

for letter_idx, letter in enumerate("qkv"):
    for layer_idx in range(model.cfg.n_layers):
        stacked_grad_act_denoising[letter_idx, layer_idx] = einops.rearrange(clean_grad_cache_denoising[f"blocks.{layer_idx}.hook_{letter}_input"], "batch seq n_heads d -> n_heads batch seq d")

# In[ ]:


results = {}
results_denoising={}
for upstream_layer_idx in range(model.cfg.n_layers):
    for upstream_head_idx in range(model.cfg.n_heads):
        for downstream_letter_idx, downstream_letter in enumerate("qkv"):
            for downstream_layer_idx in range(upstream_layer_idx+1, model.cfg.n_layers):
                for downstream_head_idx in range(model.cfg.n_heads):
                    results[(upstream_layer_idx,upstream_head_idx,downstream_letter,downstream_layer_idx,downstream_head_idx,)] = (stacked_grad_act[downstream_letter_idx, downstream_layer_idx, downstream_head_idx].cpu() * (clean_head_act[upstream_layer_idx, upstream_head_idx] - corr_head_act[upstream_layer_idx, upstream_head_idx]).cpu()).sum()
                    
                    results_denoising[
                        (
                            upstream_layer_idx,
                            upstream_head_idx,
                            downstream_letter,
                            downstream_layer_idx,
                            downstream_head_idx,
                        )
                    ] = (stacked_grad_act_denoising[downstream_letter_idx, downstream_layer_idx, downstream_head_idx].cpu() * (corr_head_act_denoising[upstream_layer_idx, upstream_head_idx] - clean_head_act_denoising[upstream_layer_idx, upstream_head_idx]).cpu()).sum()


# In[ ]:
    results_bool = {}
    for key in results.keys():
        results_bool[key] = results[key] if results[key].abs() > results_denoising[key].abs() else results_denoising[key]
    
sorted_results = sorted(results.items(), key=lambda x: x[1].abs(), reverse=True)
sorted_results_denoising = sorted(results_denoising.items(), key=lambda x: x[1].abs(), reverse=True)
sort_results_bool = sorted(results_bool.items(), key=lambda x: x[1].abs(), reverse=True)
# In[ ]:
def convert_and_save_results(sorted_results, edge, output_file):
    saved_edges = []
    for i in range(edge):
        n1 = sorted_results[i][0][0]
        n2 = sorted_results[i][0][1]
        n3 = sorted_results[i][0][2]
        n4 = sorted_results[i][0][3]
        n5 = sorted_results[i][0][4]
        f1 = sorted_results[i][1].item()

        s1 = f"blocks.{n1}.attn.hook_result"
        s2 = [None, None, n2]
        s3 = f"blocks.{n4}.hook_{n3}_input"
        s4 = [None, None, n5]

        saved_edges.append([[s3, s4, s1, s2], f1])

    with open(output_file, 'w') as f:
        json.dump(saved_edges, f, indent=4)

# Example usage

edge=5000
output_file = "noising_top{}_edges_1.json".format(edge)
convert_and_save_results(sorted_results, edge, output_file)

output_file = "denoising_top{}_edges_1.json".format(edge)

convert_and_save_results(sorted_results_denoising, edge, output_file)

output_file = "bool_top{}_edges_1.json".format(edge)
convert_and_save_results(sort_results_bool, edge, output_file)

def convert_and_save_to_ordered_dict(edges: int, data, path: str):
    """
    Convert sorted_results to OrderedDict format and save as a .pth file.

    Args:
        edges (int): Number of top edges to save.
        data: The sorted_results data.
        path (str): Path to save the .pth file.
    """
    ret = OrderedDict()
    for i in range(edges):
        n1, n2, n3, n4, n5 = data[i][0]
        s1 = f"blocks.{n1}.attn.hook_result"
        s2 = (None, None, n2)
        s3 = f"blocks.{n4}.hook_{n3}_input"
        s4 = (None, None, n5)
        ret[(s3, s4, s1, s2)] = True

    torch.save(ret, path)

# Example usage
output_path = "ioi_subgraph_noising_{}.pth".format(edge)
convert_and_save_to_ordered_dict(edge, sorted_results, output_path)

output_path = "ioi_subgraph_denoising_{}.pth".format(edge)
convert_and_save_to_ordered_dict(edge, sorted_results_denoising, output_path)

output_path = "ioi_subgraph_bool_{}.pth".format(edge)
convert_and_save_to_ordered_dict(edge, sort_results_bool, output_path)

print("Top {} most important edges:".format(edge))
for i in range(edge):
    print(
        f"{sorted_results[i][0][0]}:{sorted_results[i][0][1]} -> {sorted_results[i][0][3]}:{sorted_results[i][0][4]} | Value: {sorted_results[i][1]}",
    )
print("Top {} most important edges with OR gate and Addition gate:".format(edge))   
for i in range(edge):
    print(
        f"{sorted_results_denoising[i][0][0]}:{sorted_results_denoising[i][0][1]} -> {sorted_results_denoising[i][0][3]}:{sorted_results_denoising[i][0][4]} | Value: {sorted_results_denoising[i][1]}",
    )
print("Top {} most important edges with bool gate:".format(edge))   
for i in range(edge):
    print(
        f"{sort_results_bool[i][0][0]}:{sort_results_bool[i][0][1]} -> {sort_results_bool[i][0][3]}:{sort_results_bool[i][0][4]} | Value: {sort_results_bool[i][1]}",
    )
