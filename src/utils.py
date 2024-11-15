from typing import List

def clear_all_grads(model):
    for param in model.parameters():
        if param.grad is not None:
            param.grad.zero_()


def make_options(options: List[str], 
                 sep_word: str='\n'):
    opt_strs = []
    for i, o in enumerate(options):
        opt_strs.append(f"{chr(ord('A')+i)}: {o[0].upper()+o[1:]}.")
    return sep_word.join(opt_strs)


def warp_mul_c_question(sys_prompt: str, instruction_prompt: str, 
                        MCQ_template: str, q_word: str, options: List[str],
                        opt_sep_word: str='\n', q_type: str="n2d"):
    assert q_type in ["n2d", "d2n"]
    if q_type == "n2d":
        question = MCQ_template.format(name=q_word)
    elif q_type == "d2n":
        question = MCQ_template.format(description=q_word)

    mcq = f"Question: {question}\nOptions:{opt_sep_word}{make_options(options, sep_word=opt_sep_word)}"
    return sys_prompt.format(instruction=instruction_prompt.format(question=mcq))


def get_ids_span(target_str, input_ids, tokenizer):

    def find_subsequence_span(s, ss):
        for i in range(len(s)-len(ss)+1):
            if s[i:i+len(ss)] == ss:
                return range(i, i+len(ss))
        return None
    
    input_ids = input_ids[0].cpu().tolist()
    
    target_sub_ids = tokenizer(target_str, return_tensors="pt")["input_ids"][0][1:].tolist()
    target_ids_span = find_subsequence_span(input_ids, target_sub_ids)
    if target_ids_span is None:
        target_sub_ids = tokenizer(target_str[0].upper()+target_str[1:], return_tensors="pt")["input_ids"][0][1:].tolist()
        target_ids_span = find_subsequence_span(input_ids, target_sub_ids)
    if target_ids_span is None:
        target_sub_ids = tokenizer(target_str+'.', return_tensors="pt")["input_ids"][0][1:].tolist()
        target_ids_span = find_subsequence_span(input_ids, target_sub_ids)
    if target_ids_span is None:
        target_sub_ids = tokenizer(target_str[0].upper()+target_str[1:]+'.', return_tensors="pt")["input_ids"][0][1:].tolist()
        target_ids_span = find_subsequence_span(input_ids, target_sub_ids)
    if target_ids_span is None:
        target_sub_ids = tokenizer("\""+target_str+"\"", return_tensors="pt")["input_ids"][0][1:].tolist()
        target_ids_span = find_subsequence_span(input_ids, target_sub_ids)
    if target_ids_span is None:
        target_sub_ids = tokenizer(target_str+",", return_tensors="pt")["input_ids"][0][1:].tolist()
        target_ids_span = find_subsequence_span(input_ids, target_sub_ids)

    if target_ids_span is None:
        print(target_str)
        print(target_sub_ids)
        print(input_ids)

    assert target_ids_span is not None

    return target_ids_span
    