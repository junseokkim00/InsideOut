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
                   mad_summarize_message,
                   generate_answer,
                   most_frequent,
                   selectedEmotions,
                   ES_SYS_MSG,
                   ES_USER_MSG,
                   EMAD_SYS_MSG,
                   EMAD_USER_MSG)

emotion_color = {
    'joy': 'yellow',
    'trust': 'light_green',
    'fear': 'green',
    'surprise': 'light_blue',
    'sadness': 'blue',
    'disgust': 'magenta',
    'anger': 'red',
    'anticipation': 'light_red' 
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_type", type=str, choices=['ea', 'eu'])
    parser.add_argument("--model_type", type=str, default='gpt-3.5-turbo-0125')
    parser.add_argument("--num_agents", type=int, default=3)
    parser.add_argument("--num_round", type=int, default=2)
    parser.add_argument("--summarize", type=int, default=1)
    args = parser.parse_args()

    # setup llm
    load_dotenv()
    os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
    llm = ChatOpenAI(model=args.model_type)

    # setup dataset
    print("Dataset prep")
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

            # Configure MAD
            agent_contexts = []
            for _ in range(args.num_agents):
                messages = [('user', EMAD_USER_MSG)]
                agent_contexts.append(messages)
            # run MAD
            result_history=[]
            for round in range(args.num_round):
                if round != 0:
                    for idx, agent_context in enumerate(agent_contexts):
                        if args.summarize:
                            summary = mad_summarize_message(llm=llm,
                                                        agent_contexts=agent_contexts,
                                                        idx=idx)
                            prefix_string = f"Here is a summary of responses from other emotions: {summary}"
                        else:
                            prefix_string = "Here are a list of opinions from different agents: "
                            for i, agent in enumerate(agent_contexts):
                                if i!=idx:
                                    agent_response = agent[-1][-1]
                                    response = f"\n\n One agent response: ```{agent_response}```"
                                    prefix_string = prefix_string + response
                        prefix_string = prefix_string + \
                            "\n\nUse these opinions carefully as additional advice, can you provide an updated answer? Examine your solution and that other agents step by step. Put your answer in the form of (A), (B), (C), or (D) at the end of your response (Choose only one choice)."
                        agent_context.append(('user', prefix_string))
                for idx, agent_context in enumerate(agent_contexts):
                    completion = generate_answer(agent_context=agent_context,
                                                 llm=llm,
                                                 kwargs={
                                                     'scenario': scenario,
                                                     'subject': subject,
                                                     'choices': choices
                                                 })
                    print(
                        f"{idx}'s {round + 1} / {args.num_round} response: {completion}")
                    assistant_message = ('ai', completion)
                    agent_context.append(assistant_message)
                result = []
                for agent_context in agent_contexts:
                    input_str = agent_context[-1][-1]
                    answer = parse_answer(input_str)
                    result.append(answer)
                    final_output = most_frequent(result)
                result_history.append(result)
                if len(list(Counter(result).keys())) == 1:
                    print(f"Flush! in round {round + 1}")
                    break
            print(f"Final output: {final_output}\tGold label: {label}")
            log_file_path = f"./logs/mad_{args.dataset_type}_{args.model_type}_round:{args.num_round}_summarize:{args.summarize}"
            log_file_path+='.jsonl'
            with open(log_file_path, "a+") as f:
                json_inst = {
                    'scenario': scenario,
                    'subject': subject,
                    'choices': inst['choices'],
                    'pred': final_output,
                    'label': label,
                    'round': round+1,
                    'result_history': result_history
                }
                f.write(json.dumps(json_inst)+'\n')
