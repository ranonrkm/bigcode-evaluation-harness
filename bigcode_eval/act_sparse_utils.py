import os
import sys
import types
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.llama.modeling_llama import LlamaMLP
from tqdm import tqdm

########################### CATS ###########################
def mlp_cats_forward(self, x):
    if self.config.pretraining_tp > 1:
        slice = self.intermediate_size // self.config.pretraining_tp
        gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
        up_proj_slices = self.up_proj.weight.split(slice, dim=0)
        down_proj_slices = self.down_proj.weight.split(slice, dim=1)

        gate_proj = torch.cat(
            [F.linear(x, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1
        )
        up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)

        post_act = self.act_fn(gate_proj)
        # use self.thresh to sparsify the intermediate states
        mask = post_act.abs() < self.thresh
        post_act[mask] = 0.0

        intermediate_states = (post_act * up_proj).split(slice, dim=2)
        down_proj = [
            F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.config.pretraining_tp)
        ]
        down_proj = sum(down_proj)
    else:
        post_act = self.act_fn(self.gate_proj(x))
        # use self.thresh to sparsify the intermediate states
        mask = post_act.abs() < self.thresh
        if self.check_cats_sparsity:
            self.cats_sparsity+=(mask.sum().item() / mask.numel())
        
        post_act[mask] = 0.0
        down_proj = self.down_proj(post_act * self.up_proj(x))

    return down_proj

def catsify_draft(draft, sparsity=0.5): # add sparsity argument
    
    cats_dict = {
        "meta-llama/Llama-2-7b-hf": 
            {
                "thresh_50" : [0.019775390625, 0.03759765625, 0.04931640625, 0.0634765625, 0.08154296875, 0.0966796875, 0.11328125, 0.1240234375, 0.1259765625, 0.12890625, 0.1318359375, 0.13671875, 0.1376953125, 0.142578125, 0.1435546875, 0.1484375, 0.1591796875, 0.1640625, 0.16796875, 0.1669921875, 0.173828125, 0.173828125, 0.1767578125, 0.17578125, 0.177734375, 0.1806640625, 0.1845703125, 0.1875, 0.1904296875, 0.193359375, 0.1982421875, 0.197265625],
                "thresh_70" : [0.030517578125, 0.057373046875, 0.07568359375, 0.09521484375, 0.11962890625, 0.1396484375, 0.16015625, 0.173828125, 0.1787109375, 0.1826171875, 0.1865234375, 0.19140625, 0.1923828125, 0.1982421875, 0.2001953125, 0.2060546875, 0.2158203125, 0.220703125, 0.224609375, 0.224609375, 0.2314453125, 0.232421875, 0.2353515625, 0.234375, 0.2373046875, 0.2392578125, 0.2421875, 0.2451171875, 0.248046875, 0.25, 0.255859375, 0.259765625]
            },
        "meta-llama/Llama-2-7b-chat-hf":
            {
                "thresh_50":  [0.0201416015625, 0.037109375, 0.050048828125, 0.06298828125, 0.08056640625, 0.09423828125, 0.1103515625, 0.11865234375, 0.1201171875, 0.12451171875, 0.1259765625, 0.1279296875, 0.12890625, 0.1318359375, 0.1337890625, 0.138671875, 0.1484375, 0.15234375, 0.15625, 0.15625, 0.1630859375, 0.1650390625, 0.1689453125, 0.1689453125, 0.171875, 0.173828125, 0.1787109375, 0.181640625, 0.1845703125, 0.1875, 0.193359375, 0.1982421875],
                "thresh_70":  [0.031005859375, 0.056396484375, 0.076171875, 0.0947265625, 0.11865234375, 0.13671875, 0.1572265625, 0.16796875, 0.171875, 0.1767578125, 0.1796875, 0.181640625, 0.1826171875, 0.1875, 0.189453125, 0.1953125, 0.205078125, 0.2099609375, 0.2138671875, 0.21484375, 0.2216796875, 0.224609375, 0.228515625, 0.23046875, 0.232421875, 0.2353515625, 0.240234375, 0.2431640625, 0.24609375, 0.2490234375, 0.25390625, 0.263671875]
            },
        "meta-llama/Meta-Llama-3-8B":
            {
                "thresh_50":  [0.03076171875, 0.03662109375, 0.04736328125, 0.07080078125, 0.10205078125, 0.1162109375, 0.125, 0.125, 0.126953125, 0.1337890625, 0.12890625, 0.1298828125, 0.12890625, 0.1396484375, 0.1455078125, 0.1640625, 0.16015625, 0.1650390625, 0.1591796875, 0.15625, 0.1572265625, 0.1650390625, 0.1650390625, 0.1611328125, 0.1611328125, 0.1630859375, 0.16796875, 0.1748046875, 0.1806640625, 0.1796875, 0.189453125, 0.181640625],
                "thresh_70":  [0.048095703125, 0.056884765625, 0.072265625, 0.10498046875, 0.1435546875, 0.1591796875, 0.171875, 0.1728515625, 0.1787109375, 0.185546875, 0.1806640625, 0.181640625, 0.1826171875, 0.1923828125, 0.2021484375, 0.2177734375, 0.2138671875, 0.2177734375, 0.2119140625, 0.208984375, 0.2099609375, 0.2177734375, 0.216796875, 0.2138671875, 0.2138671875, 0.2158203125, 0.220703125, 0.2275390625, 0.2333984375, 0.234375, 0.244140625, 0.25]
            },
        "meta-llama/Meta-Llama-3-8B-Instruct":
            {
                "thresh_50": [0.0306396484375, 0.041015625, 0.050537109375, 0.0703125, 0.09912109375, 0.1123046875, 0.125, 0.125, 0.12890625, 0.1328125, 0.1318359375, 0.1298828125, 0.1279296875, 0.138671875, 0.14453125, 0.1611328125, 0.1572265625, 0.1640625, 0.15625, 0.154296875, 0.154296875, 0.16015625, 0.1591796875, 0.154296875, 0.15234375, 0.154296875, 0.16015625, 0.1669921875, 0.171875, 0.1708984375, 0.1826171875, 0.1748046875],
                "thresh_70": [0.048095703125, 0.06298828125, 0.076171875, 0.1044921875, 0.1396484375, 0.1552734375, 0.171875, 0.1728515625, 0.1787109375, 0.1845703125, 0.181640625, 0.181640625, 0.181640625, 0.19140625, 0.19921875, 0.2158203125, 0.2099609375, 0.2158203125, 0.20703125, 0.2060546875, 0.205078125, 0.2119140625, 0.2099609375, 0.20703125, 0.205078125, 0.20703125, 0.2138671875, 0.2197265625, 0.224609375, 0.2275390625, 0.2392578125, 0.2421875]
            },
        "neuralmagic/Llama-2-7b-pruned70-retrained":
            {
                "thresh_50": [0.0267333984375, 0.041015625, 0.05126953125, 0.064453125, 0.08349609375, 0.1005859375, 0.119140625, 0.1328125, 0.1357421875, 0.1416015625, 0.14453125, 0.1513671875, 0.15234375, 0.1591796875, 0.16015625, 0.1650390625, 0.17578125, 0.1806640625, 0.1826171875, 0.1787109375, 0.1826171875, 0.1796875, 0.1806640625, 0.1787109375, 0.1787109375, 0.181640625, 0.1845703125, 0.1884765625, 0.1923828125, 0.1943359375, 0.1982421875, 0.197265625],
                "thresh_70": [0.04052734375, 0.0625, 0.07861328125, 0.09765625, 0.12353515625, 0.1455078125, 0.16796875, 0.185546875, 0.189453125, 0.1962890625, 0.201171875, 0.2080078125, 0.2099609375, 0.216796875, 0.21875, 0.224609375, 0.232421875, 0.2353515625, 0.2373046875, 0.2353515625, 0.23828125, 0.2373046875, 0.2373046875, 0.2373046875, 0.23828125, 0.240234375, 0.2421875, 0.24609375, 0.25, 0.251953125, 0.255859375, 0.259765625]
            },
        "neuralmagic/Llama-2-7b-ultrachat200k-pruned_70":
            {
                "thresh_50": [0.035400390625, 0.035400390625, 0.044921875, 0.057861328125, 0.07373046875, 0.08984375, 0.10595703125, 0.11962890625, 0.125, 0.130859375, 0.1328125, 0.138671875, 0.140625, 0.1455078125, 0.1474609375, 0.154296875, 0.1640625, 0.1708984375, 0.1728515625, 0.169921875, 0.1728515625, 0.17578125, 0.1767578125, 0.1708984375, 0.1748046875, 0.1748046875, 0.1796875, 0.181640625, 0.18359375, 0.185546875, 0.1923828125, 0.189453125],
                "thresh_70": [0.049560546875, 0.0546875, 0.0693359375, 0.087890625, 0.1103515625, 0.130859375, 0.1513671875, 0.1689453125, 0.1767578125, 0.1826171875, 0.1865234375, 0.1923828125, 0.1953125, 0.201171875, 0.203125, 0.2099609375, 0.2197265625, 0.224609375, 0.2275390625, 0.224609375, 0.228515625, 0.23046875, 0.2314453125, 0.2275390625, 0.23046875, 0.23046875, 0.234375, 0.236328125, 0.2392578125, 0.2412109375, 0.2490234375, 0.25390625]
            }
    }
    
    assert draft.config._name_or_path in cats_dict.keys(), "Model not supported for catsification yet"
    thresh_50 = cats_dict[draft.config._name_or_path]['thresh_50']
    thresh_70 = cats_dict[draft.config._name_or_path]['thresh_70']

    print('sparsity for cats', sparsity)
    layer_count = 0
    
    if sparsity == 0.5:
        threshold_layer = thresh_50
    elif sparsity == 0.7:
        threshold_layer = thresh_70
    else:
        raise NotImplementedError
    
    for name, module in draft.named_modules():
        if isinstance(module, LlamaMLP):
            module.thresh = threshold_layer[layer_count]
            module.forward = types.MethodType(mlp_cats_forward, module)
            module.check_cats_sparsity = True
            module.cats_sparsity = 0.0
            layer_count += 1
    assert layer_count == 32

    return draft

def check_cats_sparsity(draft):
    check_cats_list = []
    for name, module in draft.named_modules():
        if isinstance(module, LlamaMLP):
            check_cats_list.append(module.cats_sparsity)
    cats_sparsity = np.mean(check_cats_list)
    return cats_sparsity 

########################### Griffin ###########################
def select_neurons(neuron_stat, method, k):
    if method == 'topk':
        weight, indices = torch.topk(neuron_stat, k, dim=-1)
    elif method == 'topk_sample':
        topk_weight, topk_indices = torch.topk(neuron_stat, k // 2, dim=-1)
        neuron_stat_clone = neuron_stat.clone()
        neuron_stat_clone.scatter_(index=topk_indices, dim=1, value=0)
        sampled_indices = torch.multinomial(neuron_stat_clone, k // 2, replacement=False)
        indices = torch.cat((topk_indices, sampled_indices), dim=-1)
        weight = torch.cat((topk_weight, torch.gather(neuron_stat, 1, sampled_indices)), dim=-1)
    elif method == 'sample':
        indices = torch.multinomial(neuron_stat, k, replacement=False)
        weight = torch.gather(neuron_stat, 1, indices)
    elif method == 'random': 
        indices = torch.multinomial(torch.ones_like(neuron_stat), k, replacement=False)
        weight = torch.gather(neuron_stat, 1, indices)
    else:
        raise NotImplementedError

    return weight, indices

def prepare_reduced_weights(self, topk_indices):
    assert topk_indices.size(0) == 1 # only support batch size 1
    topk_indices = topk_indices[0]

    self.gate_proj_reduced.weight.copy_(self.gate_proj.weight[topk_indices])
    self.up_proj_reduced.weight.copy_(self.up_proj.weight[topk_indices])
    self.down_proj_reduced.weight.copy_(self.down_proj.weight[:, topk_indices])
    

def mlp_griffin_forward(self, x):
    k_factor = self.k_factor
    if (self.mode == 'gen' and x.shape[1] > 1): 
        int_states = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        # Griffin expert selection
        # Generation prefilling stage 
        if k_factor > 0.0:
            k = int(int_states.shape[-1] * k_factor)
            neuron_stat = ((int_states / int_states.norm(dim=-1).unsqueeze(-1))).norm(dim=1) # B, D
            tok_weight, topk_indices = select_neurons(neuron_stat, self.config.selection_method, k)
            self.prepare_reduced_weights(topk_indices)

        down_proj = self.down_proj(int_states)
    else:
        # subsequent generation stage or acceptance rate stage 
        if k_factor == 0.0:
            down_proj = 0. * x
        else:
            down_proj = self.down_proj_reduced(self.act_fn(self.gate_proj_reduced(x)) * self.up_proj_reduced(x))

    return down_proj

def griffinify_draft(draft, k_factor=0.5, selection_method='topk'):
    # k_factor: retain top k_factors (higher means lesser sparsity)
    layer_count = 0
    for name, module in draft.named_modules():
        if isinstance(module, LlamaMLP):
            module.expert_selection = True

            module.k_factor = k_factor
            module.config.selection_method = selection_method

            hidden_size = module.hidden_size
            intermediate_size = module.intermediate_size
            k = int(intermediate_size * k_factor)
            
            module.gate_proj_reduced = torch.nn.Linear(hidden_size, k, bias=False, 
                                                        device=module.gate_proj.weight.device,
                                                        dtype=module.gate_proj.weight.dtype)
            module.up_proj_reduced = torch.nn.Linear(hidden_size, k, bias=False, 
                                                     device=module.up_proj.weight.device,
                                                     dtype=module.up_proj.weight.dtype)
            module.down_proj_reduced = torch.nn.Linear(k, hidden_size, bias=False, 
                                                        device=module.down_proj.weight.device,
                                                        dtype=module.down_proj.weight.dtype)

            module.prepare_reduced_weights = types.MethodType(prepare_reduced_weights, module)
            module.forward = types.MethodType(mlp_griffin_forward, module)
            module.mode = "gen"     # mode: gen, lm

            layer_count += 1
    assert layer_count == 32

    return draft

def set_mlp_mode(draft, mode):
    for name, module in draft.named_modules():
        if isinstance(module, LlamaMLP):
            module.mode = mode