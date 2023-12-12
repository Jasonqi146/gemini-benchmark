# NOTE:
# You can find an updated, more robust and feature-rich implementation
# in Zeno Build
# - Zeno Build: https://github.com/zeno-ml/zeno-build/
# - Implementation: https://github.com/zeno-ml/zeno-build/blob/main/zeno_build/models/providers/openai_utils.py

import openai
import asyncio
from typing import Any
import base64
import requests
from io import BytesIO
from PIL import Image 
import json
import os
import re
import torch as th
from tqdm import tqdm
from openai import AsyncOpenAI
from torch.utils.data import DataLoader
from litellm import acompletion
import yaml
import click
sys.path.append('../utils')
from reasoning_utils import * 

os.environ["OPENAI_API_KEY"] = "#######"


class BBHDataset(th.utils.data.Dataset):
    def __init__(self, examples):
        self.examples = examples
        self.qns = [ex["question"] for ex in self.examples]
        self.ans = [ex["answer"] for ex in self.examples]
        self.ids = [ex["qid"] for ex in self.examples]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        qn = self.qns[idx]
        ans = self.ans[idx]
        qid = self.ids[idx]
        
        return qid, qn, ans



@click.command()
@click.option("--task", default='object_counting', type=str)
@click.option("--model", default='gpt-3.5-turbo', type=str)
def main(task, model):
    
    with open(f"lm-evaluation-harness/lm_eval/tasks/bbh/cot_fewshot/{task}.yaml", 'r') as stream:
        data_loaded = yaml.safe_load(stream)
    prompt = data_loaded['doc_to_text']
    
    question_answer_list = []
    
    test_examples = get_examples(task, N=10)
    
    test_dset = BBHDataset(test_examples)
    test_loader = DataLoader(test_dset, batch_size=8, shuffle=True)

    for idx, (qid, qn, ans) in tqdm(enumerate(test_loader), total=len(test_loader)):
        
        mlist = []
        for q in qn:
            q_prompt = prompt.replace("{{input}}", "{input}").format(input=q)
            
            mlist.append([{"role": "system", "content": "You are a helpful assistant."},
                          {"role": "user", "content": q_prompt}])
        
        predictions = asyncio.run(
            dispatch_openai_requests(
                messages_list=mlist,
                model=model,
                temperature=0.3,
                max_tokens=512,
                top_p=1.0,
            )
        )

        for i, (response, qi, q, a) in enumerate(zip(predictions, qid, qn, ans)):
            question_answer_list.append({'qid': qi.item(),
                                         'prompt': prompt.replace("{{input}}", "{input}").format(input=q),
                                         'question': q,
                                         'answer': a,
                                         'generated_text': response.choices[0].message.content})
            
    if not os.path.exists(f'/home/sakter/courses/Fall_2023/openai/outputs/bbh/{task}'):
        os.makedirs(f'/home/sakter/courses/Fall_2023/openai/outputs/bbh/{task}')
    
    question_answer_list = get_answer(question_answer_list)
    
    with open(f'/home/sakter/courses/Fall_2023/openai/outputs/bbh/{task}/output.jsonl', 'w') as f:
        for d in question_answer_list:
            json.dump(d, f)
            f.write('\n')
        
    return


if __name__ == "__main__":
    main()
