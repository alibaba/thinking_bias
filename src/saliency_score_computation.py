import os
import json
import numpy as np
import random
from tqdm import tqdm
from typing import List, Tuple, Union

import torch 
import torch.nn.functional as F

from src.utils import *

MCQ_templates = {
    # For Llama2 chat models
    "system_prompt": (
        "[SYS] You are a helpful assistant and have the knowledge of a great number of celebrities and their stories. [/SYS]\n"
        "[INST] {instruction} [/INST]\n\nHere is my answer:\n\n"
    ),

    "instruction_prompt": "Below is a multiple-choice question. Please answer this question with the letter corresponding to the correct option, such as A/B/C/D.\n{question}",

    "n2d": ["Given the following descriptions, which one matches your knowledge about {name}?",
            "Please select the most appropriate descriptions about {name} from the following options.",
            "What is the most appropriate description of {name}?",
            "Regarding {name}, which of these descriptions are most applicable?",
            "Identify the correct descriptions of {name} from the options provided."],
    "d2n": ["Who is {description}?", 
            "Please select the name of the person who is {description}.",
            "Match the description \"{description}\" with the correct person's name.",
            "Who is the individual described as {description}?",
            "Select the person who is {description} from the following options."]
}


def get_model_activation(model, input_ids, 
                         target_modules: Union[str, List[str]]):
    """
    Get models' activations from target modules on input token sequence.

    Args:
        target_modules: List of target modules on each layer.
            attn:  return attention weights;
            mlp:   return mlp outputs;
            layer: return hidden states
    """

    if isinstance(target_modules, str):
        target_modules = [target_modules]

    activations = {}
    for m in target_modules:
        activations[m] = {}
    
    # get activations from hook
    def get_activation(layer_name, module_name):
        def hook(module, inputs, outputs):
            if module_name == 'attn':
                # (batch_size, num_heads, q_len, k_len)
                activations[module_name][layer_name] = outputs[1]
            elif module_name == 'mlp':
                # (batch_size, q_len, hidden_dims)
                activations[module_name][layer_name] = {"inputs": inputs[0][0].detach(), "outputs": outputs[0].detach()}
            elif module_name == 'layer':
                # (batch_size, q_len, hidden_dims)
                activations[module_name][layer_name] = outputs[0].detach()
            
        return hook 
    
    def remove_hooks(hooks):
        for h in hooks:
            h.remove()

    
    hooks = []

    # For the ease of activation acquirement
    model.config.output_attentions = True
    model.config._attn_implementation = "eager"
        
    for i in range(len(model.model.layers)):
        for target_module in target_modules:

            layer_name = f"Layer_{i}"

            if target_module == 'layer':
                module = model.model.layers[i]
            elif target_module == 'mlp':
                module = model.model.layers[i].mlp
            elif target_module == 'attn':
                module = model.model.layers[i].self_attn
            
            hooks.append(module.register_forward_hook(get_activation(layer_name, target_module)))

    output = model(input_ids)

    remove_hooks(hooks)
    return activations, output


def get_saliency_maps(pred, label, attention_maps, score_type=0):
    '''
        pred:  (batch_size, num_categroies)
        label: (batch_size, )
        attn_maps: dict[str, Tensor]
        score_type: 0 -- sum then abs
                    1 -- abs then sum
    '''
    for k, v in attention_maps.items():
        attention_maps[k].retain_grad()
    
    loss = F.cross_entropy(pred, label)
    loss.backward()

    saliency_maps = {}

    for k, v in attention_maps.items():
        attn_map = v.detach()
        attn_grad = v.grad.detach()
        
        if score_type == 0:
            saliency_map = torch.sum(attn_map * attn_grad, dim=1).squeeze()
            saliency_maps[k] = torch.abs(saliency_map)
        elif score_type == 1:
            saliency_maps[k] = torch.sum(torch.abs(attn_map * attn_grad), dim=1).squeeze()
    
    return saliency_maps


def compute_saliency_score(input_ids, tokenizer, model_output, activations, 
                           q_word: str, options: List[str]):
    """
    Compute the saliency scores of the names/descriptions given in the question and 4 options based on model activations and final output.
    """

    logits = model_output['logits'][0, -1]
    probs = [logits[tokenizer.vocab["A"]].cpu().item(), 
             logits[tokenizer.vocab["B"]].cpu().item(), 
             logits[tokenizer.vocab["C"]].cpu().item(), 
             logits[tokenizer.vocab["D"]].cpu().item(),]
    
    pred = model_output['logits'][0, -1].unsqueeze(0)
    chosen_opt_idx = np.argmax(probs)
    chosen_opt = ["A", "B", "C", "D"][chosen_opt_idx]
    label = torch.tensor([tokenizer.vocab[chosen_opt]]).to(pred.device)

    saliency_maps = get_saliency_maps(pred, label, activations['attn'])
    
    # Locate the position of each target phrases in the token sequence
    q_word_ids_span = None,
    opts_ids_span_list = []

    q_word_ids_span = get_ids_span(q_word, input_ids, tokenizer)
    for opt in options:
        opts_ids_span_list.append(get_ids_span(opt, input_ids, tokenizer))

    # Begin computation
    q_word_mean_saliency_scores = []
    opts_mean_saliency_scores   = [[] for _ in opts_ids_span_list]


    for k, v in saliency_maps.items():
        q_word_scores = v[-1, q_word_ids_span]
        q_word_mean_saliency_scores.append(torch.mean(q_word_scores).cpu().item())

        for i, opt_ids_span in enumerate(opts_ids_span_list):
            opt_scores = v[-1, opt_ids_span]
            opts_mean_saliency_scores[i].append(torch.mean(opt_scores).cpu().item())
            

    # (N, num_layers)
    q_word_mean_saliency_scores = np.array(q_word_mean_saliency_scores, dtype=np.float32)
    opts_mean_saliency_scores   = np.array(opts_mean_saliency_scores,   dtype=np.float32)

    mean_score = np.stack([q_word_mean_saliency_scores, np.max(opts_mean_saliency_scores, axis=0)], axis=0)

    return mean_score/(np.sum(mean_score, axis=0)+1e-12)


def MCQs_saliency_score_computation(model, tokenizer, 
                                    name_and_descriptions: List[Tuple[str, str]],
                                    num_mcqs_per_template: int=4,
                                    seed: int=1234):
    """
    Construct MCQ inputs online and compute the saliency scores of names and description.

    Args:
        names (`List[Tuple[str, str]]`):
            Different celebrities' names and the paired descriptions for MCQs construction and saliency score computation.
        num_mcqs_per_templates (`int`):
            One template can be used to construct multiple MCQs by changing the composition of the options.
            Thus the total number of MCQs used for saliency score computation and for statistical analysis is:
                Number of MCQs = Number of names (descriptions) x Number of templates x Number of MCQs per templates
    """
    np.random.seed(seed)
    random.seed(seed)

    n2d_saliency_scores, d2n_saliency_scores = [], []

    all_names, all_descriptions = [d['name'] for d in name_and_descriptions], [d['description'] for d in name_and_descriptions]

    for pair_idx in tqdm(range(len(name_and_descriptions))):
        nd_pair = name_and_descriptions[pair_idx]
        name, desc = nd_pair['name'], nd_pair['description']

        for n2d_template, d2n_template in zip(MCQ_templates['n2d'], MCQ_templates['d2n']):
            for opt_idx in range(num_mcqs_per_template):

                n2d_options = np.random.choice(all_descriptions[:pair_idx]+all_descriptions[pair_idx+1:], 
                                               size=3, replace=False).tolist()
                d2n_options = np.random.choice(all_names[:pair_idx]+all_names[pair_idx+1:], 
                                               size=3, replace=False).tolist()
                
                n2d_options.insert(opt_idx, desc)
                d2n_options.insert(opt_idx, name)

                n2d_input_text = warp_mul_c_question(MCQ_templates["system_prompt"], MCQ_templates["instruction_prompt"], 
                                                     n2d_template, name, n2d_options, q_type="n2d")
                d2n_input_text = warp_mul_c_question(MCQ_templates["system_prompt"], MCQ_templates["instruction_prompt"], 
                                                     d2n_template, desc, d2n_options, q_type="d2n")
                
                n2d_input_ids = tokenizer(n2d_input_text, return_tensors="pt")["input_ids"].to("cuda")
                d2n_input_ids = tokenizer(d2n_input_text, return_tensors="pt")["input_ids"].to("cuda")

                # Compute model activations on n2d questions
                clear_all_grads(model)
                activations, output = get_model_activation(model, n2d_input_ids, ["attn"])
                n2d_score = compute_saliency_score(n2d_input_ids, tokenizer, output, activations, name, n2d_options)


                # Compute model activations on d2n questions
                clear_all_grads(model)
                activations, output = get_model_activation(model, d2n_input_ids, ["attn"])
                d2n_score = compute_saliency_score(d2n_input_ids, tokenizer, output, activations, desc, d2n_options)

                n2d_saliency_scores.append(n2d_score)
                d2n_saliency_scores.append(d2n_score)

    return n2d_saliency_scores, d2n_saliency_scores