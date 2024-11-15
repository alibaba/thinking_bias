import os
import json
import jsonlines
import argparse

from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import set_seed, AutoTokenizer


PROMPT_TEMPLATE = {
    "llama2": "<s>[INST] {input} [/INST]\n",
    "llama3": "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
}


def judge_MCQ_correctness(pred, gt):
    pred, gt = pred.strip(), gt.strip()

    gt_opt = gt[1]

    if 'I choose option ' in pred:
        pred = pred.split('I choose option ')[1].split(' ')[0]
    elif 'I choose ' in pred:
        pred = pred.split('I choose ')[1].split(' ')[0]
    else:
        return 0

    correct = int(gt_opt in pred)
    
    return correct


def compute_openQA_recall(pred, gt):
    pred, gt = pred.strip(), gt.strip()
    
    import string
    from rouge import Rouge
    import nltk
    from nltk import word_tokenize
    nltk.download('punkt')
    nltk.download('punkt_tab')
    
    exclude_voc = [ "a", "of", "the", "and", "is", "are", "be", "to", "new", "on", "in", "at", "an", "for"]

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    

    pred = remove_punc(pred)
    tokenized_pred = word_tokenize(pred)
    tokenized_pred = [w for w in tokenized_pred if w.lower() not in exclude_voc]
    pred = ' '.join(tokenized_pred)


    if isinstance(gt, str):
        gts = [gt]
    else:
        gts = gt
    gts = [remove_punc(s) for s in gts]
    tokenized_refs  = [word_tokenize(s) for s in gts]
    tokenized_refs  = [ [w for w in tokenized_ref  if w.lower() not in exclude_voc] for tokenized_ref in tokenized_refs]
    gts = [' '.join(s) for s in tokenized_refs]

    rouge = Rouge()
    rouge_scores = rouge.get_scores(hyps=[pred]*len(gts), refs=gts)
    recall = max([rouge_scores[i]["rouge-1"]["r"] for i in range(len(rouge_scores))])
    
    return recall


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_name_or_path", default="", type=str, help="model name or path")
    parser.add_argument("--task_type", default="", type=str, help="task type, MCQ or open-QA")
    parser.add_argument("--template_type", default="", type=str, help="input prompt template")
    parser.add_argument("--input_file", default="", type=str, help="path to test input, MCQs or open-QAs")
    parser.add_argument("--output_file", default="", type=str, help="path for test result saving")
    parser.add_argument("--tensor_parallel_size", default=8, type=int, help="vllm tensor parallel size")
    parser.add_argument("--gpu_memory_utilization", default=0.9, type=float, help="vllm gpu memory utilization")
    parser.add_argument("--generate_max_new_tokens", default=60, type=int, help="max new tokens to generate")
    parser.add_argument("--seed", default=1234, type=int, help="random seed")
    parser.add_argument("--batch_size", default=32, type=int, help="inference batch size")
    return parser


if __name__ == '__main__':
    args = make_parser().parse_args()

    set_seed(args.seed)

    assert args.task_type in ['MCQ', 'open-QA']
    assert args.template_type in PROMPT_TEMPLATE.keys()

    input_template = PROMPT_TEMPLATE[args.template_type]

    input_data = []

    with jsonlines.open(args.input_file, 'r') as f:
        for l in f:
            input_data.append(l)
    
    # Load vLLM model
    llm = LLM(model=args.model_name_or_path,
              trust_remote_code=True,
              gpu_memory_utilization=args.gpu_memory_utilization,
              tensor_parallel_size=args.tensor_parallel_size)
    
    # Use greedy decoding
    sampling_params = SamplingParams(n=1, temperature=0.0,
                                     max_tokens=args.generate_max_new_tokens)
    
    # Show inference progress bar
    pbar = range(0, len(input_data), args.batch_size)
    pbar = tqdm(pbar)

    output_data = []

    # Begin batch inference
    for i in pbar:

        prompts = [e["question"] for e in input_data[i:i+args.batch_size]]

        # Apply chat templates
        prompts = [input_template.format(input=p) for p in prompts]

        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
        output_data.extend([{**info, "pred":output.outputs[0].text} for info, output in zip(input_data[i:i+args.batch_size], outputs)])


    # Begin evaluation
    if args.task_type == 'MCQ':
        eval_func = judge_MCQ_correctness
    elif args.task_type == 'open-QA':
        eval_func = compute_openQA_recall

    count_dict = {}
    for d in output_data:
        group, task = d['group'], d['task']
        if group not in count_dict.keys():
            count_dict[group] = {}
        if task not in count_dict[group].keys():
            count_dict[group][task] = [0, 0]
        
        count_dict[group][task][0] += eval_func(d['pred'], d['answer'])
        count_dict[group][task][1] += 1

    scores = {}
    for group in count_dict.keys():
        scores[group] = {}
        for task in count_dict[group].keys():
            scores[group][task] = count_dict[group][task][0] / count_dict[group][task][1]

    result_data = {
        "scores": scores,
        "count_dict": count_dict,
        "model_outputs": output_data
    }
    
    print('#'*100)
    print()
    print(f"Save result to {args.output_file}")
    print()
    print('#'*100)


    with open(args.output_file, 'w') as f:
        f.write(json.dumps(result_data, indent=4, ensure_ascii=False))