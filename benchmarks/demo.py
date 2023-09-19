import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig
import torch.nn as nn
from functools import partial

model_name_or_path = '/nvme/Baichuan2-13B-Chat'

a_dict_orig = {}
a_dict_summary = {}
questions = [
    "我怎样才能学会烹饪？",
    "我怎么才能更好地理解和接纳自己？",
    "你有什么好书推荐吗？"
]

def get_kv_hook(module, input, output, name):
    past_key_value = output[2]
    key_states, value_states = past_key_value[0][:,:, -1], past_key_value[1][:,:, -1]
    min_k, max_k = key_states.min().item(), key_states.max().item()
    min_v, max_v = value_states.min().item(), value_states.max().item()
    
    name_k = name + '.k'
    name_v = name + '.v'
    if name_k in a_dict_orig.keys():
        a_dict_orig[name_k].append((min_k, max_k))
    else:
        a_dict_orig[name_k] = [(min_k, max_k)]
    if name_v in a_dict_orig.keys():
        a_dict_orig[name_v].append((min_v, max_v))
    else:
        a_dict_orig[name_v] = [(min_v, max_v)]

def travase(m, prefix=''):

    for name, child in m.named_children():
        full_child_name = f'{prefix}.{name}' if len(prefix) else name
        if name == 'self_attn':
            child.register_forward_hook(partial(get_kv_hook, name=full_child_name))
        else:
            travase(child, full_child_name)

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)
travase(model)
model.generation_config = GenerationConfig.from_pretrained(model_name_or_path)

for q in questions:
    print('\n')
    print(q)
    messages = [{"role": "user", "content": q}]
    response = model.chat(tokenizer, messages)
    print(response)
    print('\n')

for k, v in a_dict_orig.items():
    min_val, max_val = v[0]
    for x in v:
        min_val = min(x[0], min_val)
        max_val = max(x[1], max_val)
    a_dict_summary[k] = (min_val, max_val)

for k, v in a_dict_summary.items():
    print(k, v)