# %% [markdown]
# <h1>ACDC Main Demo</h1>
#
# <p>This notebook (which doubles as a script) shows several use cases of ACDC</p>
#
# <p>The codebase is built on top of https://github.com/neelnanda-io/TransformerLens (source version)</p>
#
# <h3>Setup:</h3>
# <p>Janky code to do different setup when run in a Colab notebook vs VSCode (adapted from e.g <a href="https://github.com/neelnanda-io/TransformerLens/blob/5c89b7583e73ce96db5e46ef86a14b15f303dde6/demos/Activation_Patching_in_TL_Demo.ipynb">this notebook</a>)</p>

#%%
import sys
import os
import time
import random
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    import google.colab

    IN_COLAB = True
    print("Running as a Colab notebook")

    import subprocess # to install graphviz dependencies
    command = ['apt-get', 'install', 'graphviz-dev']
    subprocess.run(command, check=True)


    os.mkdir("ims/")

    from IPython import get_ipython
    ipython = get_ipython()

    ipython.run_line_magic( # install ACDC
        "pip",
        "install git+https://github.com/ArthurConmy/Automatic-Circuit-Discovery.git@d89f7fa9cbd095202f3940c889cb7c6bf5a9b516",
    )

except Exception as e:
    IN_COLAB = False
    print("Running outside of colab")

    import numpy # crucial to not get cursed error
    import plotly

    plotly.io.renderers.default = "colab"  # added by Arthur so running as a .py notebook with #%% generates .ipynb notebooks that display in colab
    # disable this option when developing rather than generating notebook outputs

    import os # make images folder
    if not os.path.exists("ims/"):
        os.mkdir("ims/")

    from IPython import get_ipython

    ipython = get_ipython()
    if ipython is not None:
        print("Running as a notebook")
        ipython.run_line_magic("load_ext", "autoreload")  # type: ignore
        ipython.run_line_magic("autoreload", "2")  # type: ignore
    else:
        print("Running as a script")

# %% [markdown]
# <h2>Imports etc</h2>

#%%
import wandb
import IPython
from IPython.display import Image, display
import torch
import gc
from tqdm import tqdm
import networkx as nx
import os
import torch
import huggingface_hub
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import einops
from tqdm import tqdm
import yaml
import pandas
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer

import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from transformer_lens.hook_points import HookedRootModule, HookPoint
from transformer_lens.HookedTransformer import (
    HookedTransformer,
)
try:
    from acdc.tracr_task.utils import (
        get_all_tracr_things,
        get_tracr_model_input_and_tl_model,
    )
except Exception as e:
    print(f"Could not import `tracr` because {e}; the rest of the file should work but you cannot use the tracr tasks")
from acdc.docstring.utils import get_all_docstring_things
from acdc.logic_gates.utils import get_all_logic_gate_things
from acdc.acdc_utils import (
    make_nd_dict,
    reset_network,
    shuffle_tensor,
    cleanup,
    ct,
    TorchIndex,
    Edge,
    EdgeType,
)  # these introduce several important classes !!!

from acdc.TLACDCCorrespondence import TLACDCCorrespondence
from acdc.TLACDCInterpNode import TLACDCInterpNode
from acdc.TLACDCExperiment import TLACDCExperiment

from acdc.acdc_utils import (
    kl_divergence,
)
from acdc.ioi.utils import (
    get_all_ioi_things,
    get_gpt2_small,
)
from acdc.induction.utils import (
    get_all_induction_things,
    get_validation_data,
    get_good_induction_candidates,
    get_mask_repeat_candidates,
)
from acdc.greaterthan.utils import get_all_greaterthan_things

from acdc.acdc_graphics import (
    build_colorscheme,
    show
)
import argparse



torch.autograd.set_grad_enabled(False)

# %% [markdown]
# <h2>ACDC Experiment Setup</h2>
# <p>We use a `parser to set all the options for the ACDC experiment.
# This is still usable in notebooks! We can pass a string to the parser, see below.
# We'll reproduce </p>

#%%
parser = argparse.ArgumentParser(description="Used to launch ACDC runs. Only task and threshold are required")


task_choices = ['ioi', 'docstring', 'induction', 'tracr-reverse', 'tracr-proportion', 'greaterthan', 'or_gate']
parser.add_argument('--task', type=str, required=False, choices=task_choices, default="ioi",help=f'Choose a task from the available options: {task_choices}')
parser.add_argument('--threshold', type=float, required=False, default="0.01",help='Value for THRESHOLD')
parser.add_argument('--first-cache-cpu', type=str, required=False, default="False", help='Value for FIRST_CACHE_CPU (the old name for the `online_cache`)')
parser.add_argument('--second-cache-cpu', type=str, required=False, default="False", help='Value for SECOND_CACHE_CPU (the old name for the `corrupted_cache`)')
parser.add_argument('--zero-ablation', action='store_true', help='Use zero ablation')
parser.add_argument('--using-wandb', action='store_true', help='Use wandb')
parser.add_argument('--wandb-entity-name', type=str, required=False, default="remix_school-of-rock", help='Value for WANDB_ENTITY_NAME')
parser.add_argument('--wandb-group-name', type=str, required=False, default="default", help='Value for WANDB_GROUP_NAME')
parser.add_argument('--wandb-project-name', type=str, required=False, default="acdc", help='Value for WANDB_PROJECT_NAME')
parser.add_argument('--wandb-run-name', type=str, required=False, default=None, help='Value for WANDB_RUN_NAME')
parser.add_argument("--wandb-dir", type=str, default="/tmp/wandb")
parser.add_argument("--wandb-mode", type=str, default="online")
parser.add_argument('--indices-mode', type=str, default="normal")
parser.add_argument('--names-mode', type=str, default="reverse")
parser.add_argument('--device', type=str, default="cuda")
parser.add_argument('--reset-network', type=int, default=0, help="Whether to reset the network we're operating on before running interp on it")
parser.add_argument('--metric', type=str, default="logit_diff", help="Which metric to use for the experiment")
parser.add_argument('--torch-num-threads', type=int, default=0, help="How many threads to use for torch (0=all)")
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument("--max-num-epochs",type=int, default=100_000)
parser.add_argument('--single-step', action='store_true', help='Use single step, mostly for testing')
parser.add_argument("--abs-value-threshold", action='store_true', help='Use the absolute value of the result to check threshold')

if ipython is not None:
    # We are in a notebook
    # you can put the command you would like to run as the ... in r"""..."""
    args = parser.parse_args(
        [line.strip() for line in r"""--task=induction\
--zero-ablation\
--threshold=0.71\
--indices-mode=reverse\
--first-cache-cpu=False\
--second-cache-cpu=False\
--max-num-epochs=100000""".split("\\\n")]
    )
else:
    # read from command line
    args = parser.parse_args()

# Process args

if args.torch_num_threads > 0:
    torch.set_num_threads(args.torch_num_threads)
torch.manual_seed(args.seed)

TASK = args.task
if args.first_cache_cpu is None: # manage default
    ONLINE_CACHE_CPU = True
elif args.first_cache_cpu.lower() == "false":
    ONLINE_CACHE_CPU = False
elif args.first_cache_cpu.lower() == "true":
    ONLINE_CACHE_CPU = True
else: 
    raise ValueError(f"first_cache_cpu must be either True or False, got {args.first_cache_cpu}")
if args.second_cache_cpu is None:
    CORRUPTED_CACHE_CPU = True
elif args.second_cache_cpu.lower() == "false":
    CORRUPTED_CACHE_CPU = False
elif args.second_cache_cpu.lower() == "true":
    CORRUPTED_CACHE_CPU = True
else:
    raise ValueError(f"second_cache_cpu must be either True or False, got {args.second_cache_cpu}")
THRESHOLD = args.threshold  # only used if >= 0.0
ZERO_ABLATION = True if args.zero_ablation else False
USING_WANDB = True if args.using_wandb else False
WANDB_ENTITY_NAME = args.wandb_entity_name
WANDB_PROJECT_NAME = args.wandb_project_name
WANDB_RUN_NAME = args.wandb_run_name
WANDB_GROUP_NAME = args.wandb_group_name
INDICES_MODE = args.indices_mode
NAMES_MODE = args.names_mode
DEVICE = args.device
RESET_NETWORK = args.reset_network
SINGLE_STEP = True if args.single_step else False

#%% [markdown] 
# <h2>Setup Task</h2>

#%%

second_metric = None  # some tasks only have one metric
use_pos_embed = TASK.startswith("tracr")

if TASK == "ioi":
    num_examples = 100
    things_noising = get_all_ioi_things(
        num_examples=num_examples, device=DEVICE, metric_name=args.metric,patching="noising"
    )
    things_denoising = get_all_ioi_things(
        num_examples=num_examples, device=DEVICE, metric_name=args.metric,patching="denoising"
    )
elif TASK == "or_gate":
    num_examples = 1
    seq_len = 1

    things = get_all_logic_gate_things(
        mode="OR",
        num_examples=num_examples,
        seq_len=seq_len,
        device=DEVICE,
    )
elif TASK == "tracr-reverse":
    num_examples = 6
    things = get_all_tracr_things(
        task="reverse",
        metric_name=args.metric,
        num_examples=num_examples,
        device=DEVICE,
    )
elif TASK == "tracr-proportion":
    num_examples = 50
    things = get_all_tracr_things(
        task="proportion",
        metric_name=args.metric,
        num_examples=num_examples,
        device=DEVICE,
    )
elif TASK == "induction":
    num_examples = 10 if IN_COLAB else 50
    seq_len = 300
    things = get_all_induction_things(
        num_examples=num_examples, seq_len=seq_len, device=DEVICE, metric=args.metric
    )
elif TASK == "docstring":
    num_examples = 50
    seq_len = 41
    things = get_all_docstring_things(
        num_examples=num_examples,
        seq_len=seq_len,
        device=DEVICE,
        metric_name=args.metric,
        correct_incorrect_wandb=True,
    )
elif TASK == "greaterthan":
    num_examples = 100
    things = get_all_greaterthan_things(
        num_examples=num_examples, metric_name=args.metric, device=DEVICE
    )
else:
    raise ValueError(f"Unknown task {TASK}")


#%% [markdown]
# <p> Let's define the four most important objects for ACDC experiments:

#%%

validation_metric_nosing = things_noising.validation_metric # metric we use (e.g KL divergence)
toks_int_values_nosing = things_noising.validation_data # clean data x_i
toks_int_values_other_nosing = things_noising.validation_patch_data # corrupted data x_i'
tl_model_nosing = things_noising.tl_model # transformerlens model

validation_metric_denosing = things_denoising.validation_metric # metric we use (e.g KL divergence)
toks_int_values_denosing = things_denoising.validation_data # clean data x_i
toks_int_values_other_denosing = things_denoising.validation_patch_data # corrupted data x_i'
tl_model_denosing = things_denoising.tl_model # transformerlens model

if RESET_NETWORK:
    reset_network(TASK, DEVICE, tl_model_nosing)
    reset_network(TASK, DEVICE, tl_model_denosing)

#%% [markdown]
# <h2>Setup ACDC Experiment</h2>

#%%
# Make notes for potential wandb run
try:
    with open(__file__, "r") as f:
        notes = f.read()
except Exception as e:
    notes = "No notes generated, expected when running in an .ipynb file. Error is " + str(e)

tl_model_nosing.reset_hooks()
tl_model_denosing.reset_hooks()

# Save some mem
gc.collect()
torch.cuda.empty_cache()

# Setup wandb if needed
if WANDB_RUN_NAME is None or IPython.get_ipython() is not None:
    WANDB_RUN_NAME = f"{ct()}{'_randomindices' if INDICES_MODE=='random' else ''}_{THRESHOLD}{'_zero' if ZERO_ABLATION else ''}"
else:
    assert WANDB_RUN_NAME is not None, "I want named runs, always"

tl_model_nosing.reset_hooks()
tl_model_denosing.reset_hooks()
exp_nosing = TLACDCExperiment(
    model=tl_model_nosing,
    threshold=THRESHOLD,
    using_wandb=USING_WANDB,
    wandb_entity_name=WANDB_ENTITY_NAME,
    wandb_project_name=WANDB_PROJECT_NAME,
    wandb_run_name=WANDB_RUN_NAME,
    wandb_group_name=WANDB_GROUP_NAME,
    wandb_notes=notes,
    wandb_dir=args.wandb_dir,
    wandb_mode=args.wandb_mode,
    wandb_config=args,
    zero_ablation=ZERO_ABLATION,
    abs_value_threshold=args.abs_value_threshold,
    ds=toks_int_values_nosing,
    ref_ds=toks_int_values_other_nosing,
    metric=validation_metric_nosing,
    second_metric=second_metric,
    verbose=True,
    indices_mode=INDICES_MODE,
    names_mode=NAMES_MODE,
    corrupted_cache_cpu=CORRUPTED_CACHE_CPU,
    hook_verbose=False,
    online_cache_cpu=ONLINE_CACHE_CPU,
    add_sender_hooks=True,
    use_pos_embed=use_pos_embed,
    add_receiver_hooks=False,
    remove_redundant=True,
    show_full_index=use_pos_embed,
)

exp_denosing = TLACDCExperiment(
    model=tl_model_denosing,
    threshold=THRESHOLD,
    using_wandb=USING_WANDB,
    wandb_entity_name=WANDB_ENTITY_NAME,
    wandb_project_name=WANDB_PROJECT_NAME,
    wandb_run_name=WANDB_RUN_NAME,
    wandb_group_name=WANDB_GROUP_NAME,
    wandb_notes=notes,
    wandb_dir=args.wandb_dir,
    wandb_mode=args.wandb_mode,
    wandb_config=args,
    zero_ablation=ZERO_ABLATION,
    abs_value_threshold=args.abs_value_threshold,
    ds=toks_int_values_denosing,
    ref_ds=toks_int_values_other_denosing,
    metric=validation_metric_denosing,
    second_metric=second_metric,
    verbose=True,
    indices_mode=INDICES_MODE,
    names_mode=NAMES_MODE,
    corrupted_cache_cpu=CORRUPTED_CACHE_CPU,
    hook_verbose=False,
    online_cache_cpu=ONLINE_CACHE_CPU,
    add_sender_hooks=True,
    use_pos_embed=use_pos_embed,
    add_receiver_hooks=False,
    remove_redundant=True,
    show_full_index=use_pos_embed,
)

# %% [markdown]
# <h2>Run steps of ACDC: iterate over a NODE in the model's computational graph</h2>
# <p>WARNING! This will take a few minutes to run, but there should be rolling nice pictures too : )</p>

#%%

import datetime
exp_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

for i in range(args.max_num_epochs):
    if exp_nosing.current_node is None or exp_denosing.current_node is None:
        show(
            exp_nosing.corr,
            f"ims/ACDC_img_{exp_time}.png",

        )
        break

    start_step_time = time.time()
    exp_nosing.step_idx += 1
    exp_denosing.step_idx += 1

    exp_nosing.update_cur_metric(recalc_edges=False)
    exp_nosing_initial_metric = exp_nosing.cur_metric
    
    exp_denosing.update_cur_metric(recalc_edges=False)
    exp_denosing_initial_metric = exp_denosing.cur_metric
    
    exp_nosing_cur_metric = exp_nosing_initial_metric
    exp_denosing_cur_metric = exp_denosing_initial_metric
    
    if exp_denosing.verbose:
        print("New metric (noising and denoising):", exp_nosing_cur_metric, exp_denosing_cur_metric)
        
    if exp_nosing.current_node.incoming_edge_type.value != EdgeType.PLACEHOLDER.value:
            exp_nosing_added_receiver_hook = exp_nosing.add_receiver_hook(exp_nosing.current_node, override=True, prepend=True)

    if exp_nosing.current_node.incoming_edge_type.value == EdgeType.DIRECT_COMPUTATION.value:
            # basically, because these nodes are the only ones that act as both receivers and senders
            exp_nosing_added_sender_hook = exp_nosing.add_sender_hook(exp_nosing.current_node, override=True)
    
    if exp_denosing.current_node.incoming_edge_type.value != EdgeType.PLACEHOLDER.value:
            exp_denosing_added_receiver_hook = exp_denosing.add_receiver_hook(exp_denosing.current_node, override=True, prepend=True)

    if exp_denosing.current_node.incoming_edge_type.value == EdgeType.DIRECT_COMPUTATION.value:
            # basically, because these nodes are the only ones that act as both receivers and senders
            exp_denosing_added_sender_hook = exp_denosing.add_sender_hook(exp_denosing.current_node, override=True)
    
    exp_nosing_is_this_node_used = False
    exp_denosing_is_this_node_used = False
    if exp_nosing.current_node.name in ["blocks.0.hook_resid_pre", "hook_pos_embed", "hook_embed"]:
        exp_nosing_is_this_node_used = True
            
    if exp_denosing.current_node.name in ["blocks.0.hook_resid_pre", "hook_pos_embed", "hook_embed"]:
        exp_denosing_is_this_node_used = True
        
    exp_nosing_sender_names_list = list(exp_nosing.corr.edges[exp_nosing.current_node.name][exp_nosing.current_node.index])

    if exp_nosing.names_mode == "random":
            random.shuffle(exp_nosing_sender_names_list)
    elif exp_nosing.names_mode == "reverse":
        exp_nosing_sender_names_list = list(reversed(exp_nosing_sender_names_list))

        
    for sender_name in exp_nosing_sender_names_list:
            exp_nosing_sender_indices_list = list(exp_nosing.corr.edges[exp_nosing.current_node.name][exp_nosing.current_node.index][sender_name])#attention head？

            if exp_nosing.indices_mode == "random":
                random.shuffle(sender_indices_list)
            elif exp_nosing.indices_mode == "reverse":
                sender_indices_list = list(reversed(sender_indices_list))

            for sender_index in exp_nosing_sender_indices_list:#每个attention head里的qkv？若是其他则是空
                edge = exp_nosing.corr.edges[exp_nosing.current_node.name][exp_nosing.current_node.index][sender_name][sender_index]
                edge_denosing = exp_denosing.corr.edges[exp_denosing.current_node.name][exp_denosing.current_node.index][sender_name][sender_index]
                cur_parent = exp_nosing.corr.graph[sender_name][sender_index]

                if edge.edge_type == EdgeType.PLACEHOLDER:
                    exp_nosing_is_this_node_used = True
                    exp_denosing_is_this_node_used = True
                    continue # include by default

                if exp_denosing.verbose:
                    print(f"\nNode: {cur_parent=} ({exp_denosing.current_node=})\n")
                    
                edge.present = False
                edge_denosing.present = False

                if edge.edge_type == EdgeType.ADDITION:
                    exp_nosing_added_sender_hook = exp_nosing.add_sender_hook(
                        cur_parent,
                    )
                    exp_denosing_added_sender_hook = exp_denosing.add_sender_hook(
                        cur_parent,
                    )
                else:
                    added_sender_hook = False
                    
                exp_nosing_old_metric = exp_nosing.cur_metric
                exp_denosing_old_metric = exp_denosing.cur_metric
                exp_nosing.update_cur_metric(recalc_edges=False)
                exp_denosing.update_cur_metric(recalc_edges=False)
                exp_nosing_evaluated_metric = exp_nosing.cur_metric
                exp_denosing_evaluated_metric = exp_denosing.cur_metric
                if exp_nosing.verbose:
                    print(
                        "Noising: Metric after removing connection to",
                        sender_name,
                        sender_index,
                        "is",
                        exp_nosing_evaluated_metric,
                        "(and current metric " + str(exp_nosing_old_metric) + ")",
                    )
                    print(
                        "Denoising: Metric after removing connection to",
                        sender_name,
                        sender_index,
                        "is",
                        exp_denosing_evaluated_metric,
                        "(and current metric " + str(exp_denosing_old_metric) + ")",
                    )
                exp_nosing_result = exp_nosing_evaluated_metric - exp_nosing_old_metric
                exp_denosing_result = exp_denosing_evaluated_metric - exp_denosing_old_metric
                if exp_nosing.abs_value_threshold:
                    exp_nosing_result = abs(exp_nosing_result)
                    exp_denosing_result = abs(exp_denosing_result)
                if exp_nosing_result < exp_nosing.threshold and exp_denosing_result < exp_denosing.threshold:
                    edge.effect_size = exp_nosing_result
                    if exp_nosing.verbose:
                        print("...so removing connection")
                    exp_nosing.corr.remove_edge(
                        exp_nosing.current_node.name, exp_nosing.current_node.index, sender_name, sender_index
                    )
                    exp_denosing.corr.remove_edge(
                        exp_denosing.current_node.name, exp_denosing.current_node.index, sender_name, sender_index
                    )
                    
                else: # include this edge in the graph
                    if exp_nosing_result > exp_nosing.threshold:
                        edge.effect_size = exp_nosing_result
                    elif exp_denosing_result > exp_denosing.threshold:
                        edge.effect_size = exp_denosing_result
                    else:
                        edge.effect_size = max(exp_nosing_result, exp_denosing_result)
                    exp_nosing.cur_metric = exp_nosing_old_metric
                    exp_denosing.cur_metric = exp_denosing_old_metric

                    exp_nosing_is_this_node_used = True
                    exp_denosing_is_this_node_used = True
                    if exp_nosing.verbose:
                        print("...so keeping connection")
                    edge.present = True
                    edge_denosing.present = True
                exp_nosing.update_cur_metric(recalc_edges=False, recalc_metric=False)
                exp_denosing.update_cur_metric(recalc_edges=False, recalc_metric=False)
            
            exp_nosing.update_cur_metric(recalc_metric=True, recalc_edges=True)
            exp_denosing.update_cur_metric(recalc_metric=True, recalc_edges=True)
        
    if not exp_nosing_is_this_node_used and exp_nosing.remove_redundant:
        if exp_nosing.verbose:
            print("Removing redundant node", exp_nosing.current_node)
        exp_nosing.remove_redundant_node(exp_nosing.current_node)
        exp_denosing.remove_redundant_node(exp_denosing.current_node)
        
    if exp_nosing_is_this_node_used and exp_nosing.current_node.incoming_edge_type.value != EdgeType.PLACEHOLDER.value:
            fname = f"ims/img_new_{exp_nosing.step_idx}.png"
            show(
                exp_nosing.corr,
                fname=fname,
                show_full_index=exp_nosing.show_full_index,
            )
            
    exp_nosing.increment_current_node()#将current_node指向下一个节点
    exp_nosing.update_cur_metric(recalc_metric=True, recalc_edges=True) # so we log the correct state...
    exp_denosing.increment_current_node()#将current_node指向下一个节点
    exp_denosing.update_cur_metric(recalc_metric=True, recalc_edges=True) # so we log the correct state..
    
    show(
        exp_nosing.corr,
        f"ims/img_new_{i+1}.png",
        show_full_index=False,
    )

    if IN_COLAB or ipython is not None:
        # so long as we're not running this as a script, show the image!
        display(Image(f"ims/img_new_{i+1}.png"))

    print(i, "-" * 50)
    print(exp_nosing.count_no_edges())

    if i == 0:
        exp_nosing.save_edges("edges.json")

    if exp_nosing.current_node is None or exp_denosing.current_node is None:
        show(
            exp_nosing.corr,
            f"ims/ACDC_img_{exp_time}.png",

        )
        break

exp_nosing.save_edges("bool_reverse_0.01_edges.json")

if USING_WANDB:
    edges_fname = f"edges.pth"
    exp_nosing.save_edges(edges_fname)
    artifact = wandb.Artifact(edges_fname, type="dataset")
    artifact.add_file(edges_fname)
    wandb.log_artifact(artifact)
    os.remove(edges_fname)
    wandb.finish()

# %% [markdown]
# <h2>Save the final subgraph of the model</h2>
# <p>There are more than `exp.count_no_edges()` here because we include some "placeholder" edges needed to make ACDC work that don't actually matter</p>
# <p>Also note that the final image has more than 12 edges, because the edges from a0.0_q and a0.0_k are not connected to the input</p>
# <p>We recover minimal induction machinery! `embed -> a0.0_v -> a1.6k`</p>

#%%
exp_nosing.save_subgraph('ioi_subgraph_bool_0.01_reverse.pth',
    return_it=True,
)
out = exp_nosing.call_metric_with_corr(exp_nosing.corr, things_noising.test_metrics["kl_div"], things_noising.test_data)
print(out)
# %%
