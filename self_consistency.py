import argparse
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from datasets import load_dataset
from tqdm import tqdm
import json
from termcolor import colored
from collections import Counter
from utils import (parse_answer,
                   summarize_message,
                   generate_answer,
                   most_frequent,
                   EMAD_USER_MSG)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_type", type=str, choices=['ea', 'eu'])
    parser.add_argument("--model_type", type=str, default='gpt-3.5-turbo-0125')
    parser.add_argument("--num_sc", type=int)
    parser.add_argument("--emotion", type=int, default=0)
    args = parser.parse_args()

    # setup llm
    load_dotenv()
    os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
    llm = ChatOpenAI(model=args.model_type)

    # setup dataset
    if args.dataset_type == 'ea':
        dataset = load_dataset("SahandSab/EmoBench", "emotional_application")
    else:
        dataset = load_dataset("SahandSab/EmoBench", "emotional_understanding")
    dataset = dataset['train']
    dataset = [inst for inst in dataset if inst['language'] == 'en']

    for idx, inst in enumerate(dataset):
        if args.dataset_type == 'ea':
            scenario = inst['scenario']
            subject = inst['subject']
            choices = inst['choices']
            choices = [f"{key}. {choices[key]}" for key in choices]
            choices = ' '.join(choices)
            label = inst['label']
            print(
                f"{idx+1} / {len(dataset)}\nScenario: {scenario}\nsubject: {subject}\nChoices: {choices}\n")
            result = []
            for i in range(args.num_sc):
                sc_prompts = ChatPromptTemplate.from_messages([
                    ('user', EMAD_USER_MSG)
                ])
                chain = sc_prompts | llm
                output = chain.invoke({
                    'scenario': scenario,
                    'subject': subject,
                    'choices': choices
                }).content
                output = parse_answer(output)
                print(f"#{i+1} try: {output}")
                result.append(output)
            final_output = most_frequent(result)
            print(f"Final output: {final_output}\tGold label: {label}")
            log_file_path = f"./logs/sc_{args.dataset_type}_{args.model_type}_num:{args.num_sc}.jsonl"
            with open(log_file_path, "a+") as f:
                json_inst = {
                    'scenario': scenario,
                    'subject': subject,
                    'choices': inst['choices'],
                    'pred': final_output,
                    'label': label,
                    'result': result
                }
                f.write(json.dumps(json_inst)+'\n')

                
