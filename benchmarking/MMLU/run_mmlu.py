from litellm import acompletion
from tqdm import tqdm
from utils import *
import pandas as pd
import asyncio
import litellm
import json
import openai

# parse arguments
import argparse
import os
import re
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
from typing import Type
import threading

SOTOPIA_FORMAT = "Prompt after formatting:\nImagine you are James Davis, your task is to act/speak as James Davis would, keeping in mind James Davis's social goal.\nYou can find James Davis's background and goal in the 'Here is the context of the interaction' field.\nNote that James Davis's secret and goal is only visible to you.\nYou should try your best to achieve James Davis's goal in a way that align with their character traits.\nAdditionally, maintaining the conversation's naturalness and realism is essential (e.g., do not repeat what other people has already said before).\n\nHere is the context of this interaction:\nScenario: A professor is asking question for a very good student who knows anything.\nParticipants: Leo Brown and James Davis\nLeo Brown's background: Leo Brown is a 51-year-old male professor. He/him pronouns. Leo Brown often spends his time compiling interesting multiple choice questions to ask students. \nJames Davis's background: James Davis is a 20-year-old male college student. He/him pronouns. James Davis is a really good student and he know the answer to almost all questions. Personality and values description: James Davis, who is somewhat reserved and anxious, highly prioritizes care and protection for others. He is cautious when making decisions. James's secrets: He has a secret identity as a stand-up comedian.\nLeo Brown's goal: Unknown\nJames Davis's goal: Answer the multiple choice question given by your professor correctly (Extra information: you knows the answer to everything in the world, so the questions should be easy to you.)\nConversation Starts:\n\nTurn #0: Leo Brown said: \"Hey James, I'd love to ask you an intersting multiple choice question. Here is the question: {}\"\n\nYou are at Turn #1."
class MMLUResponse(BaseModel):
    choice: str = Field(
        description="The actual choice of the correct answer. Must be one of A, B, C, D."
    )
    reasoning: str = Field(
        description="A short explanation that supports the choice of answer."
    )

class MMLUResponsePydanticOutputParser(PydanticOutputParser[MMLUResponse]):
    def __init__(self, pydantic_object: Type[BaseModel] = MMLUResponse) -> None:
        super(MMLUResponsePydanticOutputParser, self).__init__(
            pydantic_object=pydantic_object
        )

    def parse(self, text: str) -> MMLUResponse:
        text = re.sub(r",\s*(\)|\])", r"\1", text)
        return super().parse(text)

    def get_format_instructions(self) -> str:
        format_instruction = super().get_format_instructions()
        return format_instruction

async def get_response(prompt: str, model: str):
    if "gemini" in model:
        response = await acompletion(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "Follow the given examples and answer the question.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            safety_settings=[
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_NONE",
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_NONE",
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_NONE",
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_NONE",
                },
            ],
        )
    elif model.startswith("custom"):
        global client, parser
        response = get_response_custom_model(prompt, model[7:], client = client)
    elif model.startswith("sotopia"):
        global client, parser
        response = get_response_custom_model(prompt, model[8:], client = client)
    else:
        response = await acompletion(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "Follow the given examples and answer the question.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
    return response


def get_response_custom_model(prompt: str, model: str = "gpt-3.5-turbo", base_url: str = "http://0.0.0.0:8013/v1", client: openai.OpenAI = None):
    return chat_with_timeout({ "client": client, "model": model, "prompt": prompt, "timeout": 30 })

def chat_with_timeout(api_parameters):
    timeout = api_parameters.pop("timeout", None)
    result = [None]

    api_thread = threading.Thread(target=api_call, args=(result, api_parameters))
    api_thread.start()
    api_thread.join(timeout=timeout)  # Set the timeout for the API call

    if api_thread.is_alive():
        # If the API call is still running after the timeout, terminate it
        print("API call timeout, retrying...")

        api_thread.join(timeout=timeout + 1)  # Retry
        if api_thread.is_alive(): 
            print("API call still hanging, retry failed.")
            return {
                "choices": [
                    {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "API Timeout"
                    },
                    "finish_reason": "timeout"
                    }
                ],
            }
            
    return result[0]

def api_call(result, api_parameters):
    completion = api_parameters['client'].chat.completions.create(
        model=api_parameters['model'],
        messages=[
            {
                "role": "system",
                "content": "Follow the given examples and answer the question.",
            },
            {"role": "user", "content": api_parameters['prompt']},
        ],
        temperature=0,
        max_tokens=256,
    )
    response = {'choices': [dict(choice) for choice in completion.choices]}
    for choice in response['choices']:
        choice['message'] = dict(choice['message'])
    result[0] = response

def main(args, tasks=TASKS):
    if "gpt" in args.model_name:
        # gpt evaluation
        os.environ["OPENAI_API_KEY"] = args.openai_api_key
    elif "gemini" in args.model_name:
        # gemini evaluation
        litellm.vertex_project = ""  # Your Project ID
        litellm.vertex_location = ""  # Your Project Location
        litellm.drop_params = True
    elif args.model_name.startswith("custom"):
        global client
        client = openai.OpenAI(api_key="anything", base_url="http://0.0.0.0:8013/v1")
    elif args.model_name.startswith("sotopia"):
        global client
        global parser
        client = openai.OpenAI(api_key="anything", base_url="http://0.0.0.0:8013/v1")
        parser = MMLUResponsePydanticOutputParser()
    else:
        args.model_name = "together_ai/mistralai/Mixtral-8x7B-Instruct-v0.1"
    if args.cot:
        mmlu_cot_prompt = json.load(open("data/mmlu-cot.json"))

    all_acc = 0
    all_number = 0
    accs_json = {}
    method_name = "cot" if args.cot else "simple"
    outputs_file = open(f"outputs/{args.model_name}_{method_name}_outputs.json", "a")
    for task in tqdm(tasks):
        print("Testing %s ..." % task)
        acc = 0
        dev_df = pd.read_csv(
            os.path.join("data", "dev", task + "_dev.csv"), header=None
        )[: args.num_examples]
        test_df = pd.read_csv(
            os.path.join("data", "test", task + "_test.csv"), header=None
        )
        for i in tqdm(range(test_df.shape[0])):
            if args.cot:
                # chain of thought
                q = test_df.iloc[i, 0] + "\n"
                for j, letter in enumerate(["A", "B", "C", "D"]):
                    q += "(" + letter + ") " + str(test_df.iloc[i, j + 1]) + " "
                q += "\nA: Let's think step by step."

                prompt = mmlu_cot_prompt[task] + "\n\n" + q
                label = test_df.iloc[i, test_df.shape[1] - 1]

                response = asyncio.run(get_response(prompt, args.model_name))
                ans_model = response["choices"][0]["message"]["content"]
                ans_, residual = extract_ans(ans_model)

                ans_model, correct = test_answer_mmlu_(ans_, label)
                if correct:
                    acc += 1
            else:
                # simple prompting
                k = args.num_examples
                prompt_end = format_example(test_df, i, include_answer=False)
                train_prompt = gen_prompt(dev_df, task, k)
                prompt = train_prompt + prompt_end
                if args.model_name.startswith("sotopia"):
                    prompt = SOTOPIA_FORMAT.format(prompt) + parser.get_format_instructions()
                    
                label = test_df.iloc[i, test_df.shape[1] - 1]
                response = asyncio.run(get_response(prompt, args.model_name))
                
                # 0 means the answer character [A, B, C, D] (sometimes model will output more)
                if args.model_name.startswith("sotopia"):
                    try:
                        ans_model = parser.parse(response["choices"][0]["message"]["content"]).choice
                    except:
                        ans_model = "*"
                        print("Use * as default answer")
                else:
                    ans_model = response["choices"][0]["message"]["content"][0]

                correct = ans_model == label
                if correct:
                    acc += 1
            outputs_file.write(
                json.dumps(
                    {
                        "task": task,
                        "correct": correct,
                        "prediction": ans_model,
                        "label": label,
                        "response": response["choices"][0]["message"]["content"] if response else None,
                        "question": test_df.iloc[i, 0],
                        "A": test_df.iloc[i, 1],
                        "B": test_df.iloc[i, 2],
                        "C": test_df.iloc[i, 3],
                        "D": test_df.iloc[i, 4],
                        "prompt": prompt,
                    }
                )
                + "\n"
            )
        print("%s acc %.4f" % (task, acc / test_df.shape[0]))
        accs_json[task] = acc / test_df.shape[0]
        all_acc += acc
        all_number += test_df.shape[0]
    accs_json["all"] = all_acc / all_number
    json.dump(
        accs_json, open(f"outputs/{args.model_name}_{method_name}_accs.json", "w")
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--openai_api_key", type=str)
    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt-3.5-turbo",
        # choices=["gpt-3.5-turbo", "gpt-4-1106-preview", "gemini-pro", "mixtral", "custom"],
    )
    parser.add_argument("--cot", action="store_true")
    parser.add_argument(
        "--num_examples",
        type=int,
        default=5,
        help="Number of examples included in the current prompt input. ",
    )
    # parser.add_argument(
    #     "--api_base_url",
    #     type=str,
    #     default="http://0.0.0.0:8000",
    #     help="API base url for custom model",
    # )
    args = parser.parse_args()
    main(args)
