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
from typing import Literal, Type
import threading
from langchain.schema.output_parser import OutputParserException
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate,
)

# os.environ['OPENAI_API_KEY'] = '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'

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
    return chat_c_threaded({ "client": client, "model": model, "prompt": prompt, "timeout": 30 })

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

def chat_c_threaded(api_parameters):
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

    # The API call has finished within the timeout or retried successfully
    return result[0]

def _return_fixed_model_version(
    model_name: Literal["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"]
) -> str:
    return {
        "gpt-3.5-turbo": "gpt-3.5-turbo-0613",
        "gpt-4": "gpt-4-0613",
        "gpt-4-turbo": "gpt-4-1106-preview",
    }[model_name]

def obtain_chain(
    model_name: str,
    template: str,
    input_variables: list[str],
    temperature: float = 0.7,
    max_retries: int = 6,
) -> LLMChain:
    """
    Using langchain to sample profiles for participants
    """
    match model_name:
        case "gpt-3.5-turbo" | "gpt-4" | "gpt-4-turbo":
            human_message_prompt = HumanMessagePromptTemplate(
                prompt=PromptTemplate(
                    template=template,
                    input_variables=input_variables,
                )
            )
            chat_prompt_template = ChatPromptTemplate.from_messages(
                [human_message_prompt]
            )
            chat = ChatOpenAI(
                model_name=_return_fixed_model_version(model_name),
                temperature=temperature,
                max_retries=max_retries,
            )
            chain = LLMChain(llm=chat, prompt=chat_prompt_template)
            return chain
        case _:
            raise ValueError(f"Invalid model name: {model_name}")

def format_bad_output(
    ill_formed_output: str,
    format_instructions: str,
    model_name: str = "gpt-3.5-turbo",
) -> str:
    template = """
    Given the string that can not be parsed by json parser, reformat it to a string that can be parsed by json parser.
    Original string: {ill_formed_output}

    Format instructions: {format_instructions}

    Please only generate the JSON:
    """
    chain = obtain_chain(
        model_name=model_name,
        template=template,
        input_variables=re.findall(r"{(.*?)}", template),
    )
    input_values = {
        "ill_formed_output": ill_formed_output,
        "format_instructions": format_instructions,
    }
    reformat = chain.predict([], **input_values)
    print(f"Reformated output: {reformat}")
    return reformat


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
                if args.model_name.startswith("custom"):
                    prompt = SOTOPIA_FORMAT.format(prompt) + parser.get_format_instructions()
                label = test_df.iloc[i, test_df.shape[1] - 1]
                # try:
                response = asyncio.run(get_response(prompt, args.model_name))
                # 0 means the answer character [A, B, C, D] (sometimes model will output more)
                if args.model_name.startswith("custom"):
                    try:
                        ans_model = parser.parse(response["choices"][0]["message"]["content"]).choice
                    except OutputParserException as ope:
                        print(f"Error in response {ope}")
                        ans_model = parser.parse(
                            format_bad_output(response["choices"][0]["message"]["content"], parser.get_format_instructions())
                        ).choice
                else:
                    ans_model = response["choices"][0]["message"]["content"][0]
                # except:
                #     print("Error in response")
                #     ans_model = "*"
                #     print("Use * as default answer")

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
