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
                   EMAD_USER_MSG,
                   EUMAD_USER_MSG)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_type", type=str, choices=['ea', 'eu'])
    parser.add_argument("--model_type", type=str, default='gpt-3.5-turbo-0125')
    args = parser.parse_args()

    # setup llm
    load_dotenv()
    os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
    llm = ChatOpenAI(model=args.model_type)

    # setup dataset
    if args.dataset_type == 'ea':
        dataset = load_dataset("SahandSab/EmoBench", "emotional_application")
        sc_prompts = ChatPromptTemplate.from_messages([
            ('user', EMAD_USER_MSG)
        ])
    else:
        dataset = load_dataset("SahandSab/EmoBench", "emotional_understanding")
        sc_prompts = ChatPromptTemplate.from_messages([
            ('user', EUMAD_USER_MSG)
        ])
    dataset = dataset['train']
    dataset = [inst for inst in dataset if inst['language'] == 'en']

    chain = sc_prompts | llm

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
            output = chain.invoke({
                'scenario': scenario,
                'subject': subject,
                'choices': choices
            }).content
            final_output = parse_answer(output)
            print(f"Final output: {final_output}\tGold label: {label}")
            log_file_path = f"./logs/base_{args.dataset_type}_{args.model_type}.jsonl"
            with open(log_file_path, "a+") as f:
                json_inst = {
                    'scenario': scenario,
                    'subject': subject,
                    'choices': inst['choices'],
                    'pred': final_output,
                    'label': label
                }
                f.write(json.dumps(json_inst)+'\n')
        else:
            scenario = inst['scenario']
            subject = inst['subject']
            e_choices = inst['emotion choices']
            e_choices = [f"{key}. {e_choices[key]}" for key in e_choices]
            e_choices = ' '.join(e_choices)
            e_label = inst['emotion label']
            print(
                f"{idx+1} / {len(dataset)}\nScenario: {scenario}\nsubject: {subject}\nChoices: {e_choices}\n")
            output = chain.invoke({
                'scenario': scenario,
                'subject': subject,
                'choices': e_choices
            }).content
            final_output = parse_answer(output)
            print(f"Final output: {final_output}\tGold label: {e_label}")
            log_file_path = f"./logs/base_{args.dataset_type}_{args.model_type}.jsonl"
            with open(log_file_path, "a+") as f:
                json_inst = {
                    'scenario': scenario,
                    'subject': subject,
                    'choices': inst['emotion choices'],
                    'pred': final_output,
                    'label': e_label
                }
                f.write(json.dumps(json_inst)+'\n')

