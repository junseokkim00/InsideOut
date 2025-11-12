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
                   selectedEmotions,
                   ES_SYS_MSG,
                   ES_USER_MSG_CHAT,
                   EMAD_USER_MSG_CHAT,
                   EMAD_USER_MSG_FINAL)

from rich.console import Console
from rich.panel import Panel


emotion_color = {
    'joy': 'yellow',
    'trust': 'bright_green',      # light_green → bright_green
    'fear': 'green',
    'surprise': 'bright_blue',    # light_blue → bright_blue
    'sadness': 'blue',
    'disgust': 'magenta',
    'anger': 'red',
    'anticipation': 'bright_red', # light_red → bright_red
    'rational': 'cyan'
}




def print_console(console, message, title, border_style="white"):
    panel = Panel(message, title=title, border_style=border_style, padding=(1, 2))
    console.print(panel)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default='gpt-3.5-turbo-0125')
    parser.add_argument("--num_round", type=int, default=2)
    parser.add_argument("--summarize", type=int, default=1)
    parser.add_argument("--select_emotions", type=int, default=1)
    parser.add_argument("--add_rational", type=int, default=0)
    args = parser.parse_args()

    console = Console()

    # setup llm
    load_dotenv()
    os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
    llm = ChatOpenAI(model=args.model_type)
    # setup dataset
    # print("Dataset prep")
    # if args.dataset_type == 'ea':
    #     dataset = load_dataset("SahandSab/EmoBench", "emotional_application")
    # else:
    #     dataset = load_dataset("SahandSab/EmoBench", "emotional_understanding")
    # dataset = dataset['train']
    # dataset = [inst for inst in dataset if inst['language'] == 'en']

    # setup Emotion Selection chain
    es_prompts = ChatPromptTemplate.from_messages([
        ('system', ES_SYS_MSG),
        ('user', ES_USER_MSG_CHAT)
    ])
    parser = PydanticOutputParser(pydantic_object=selectedEmotions)
    es_chain = es_prompts | llm | parser

    # for idx, inst in enumerate(dataset):
    # if args.dataset_type == 'ea':
    scenario = input("Enter the scenario: ")
    

    # print(f"Scenario: {scenario}")
    print_console(console, scenario, "Scenario")

    # select emotions
    if args.select_emotions:
        trial = 0
        while trial < 5:
            try:
                output = es_chain.invoke({
                    'scenario': scenario,
                    'format_instruction': parser.get_format_instructions()
                })
                emotions = output.emotions
                flag = True
                for emotion in emotions:
                    if emotion not in list(emotion_color.keys()):
                        flag = False
                if flag:
                    break
                else:
                    print(f"trial:{trial} Wrong emotion")
                    trial += 1
                    if trial >= 5:
                        emotions = [emotion for emotion in emotions if emotion in list(
                            emotion_color.keys())]
            except:
                print(f"trial:{trial} Error occur. Catching exception...")
                trial += 1
    else:
        if args.add_rational:
            emotions = list(emotion_color.keys())
        else:
            emotions = [emotion for emotion in emotion_color.keys()
                        if emotion != 'rational']

    if args.add_rational:
        emotions.append('rational')

    # print(f"Emotions for the current scenario: {emotions}")
    print_console(console, ', '.join(emotions), "Selected Emotions")

    # Configure MAD
    agent_contexts = []
    for emotion in emotions:
        if emotion != 'rational':
            messages = [
                ('system', f"You are {emotion}, one of the emotions of yours"), ('user', EMAD_USER_MSG_CHAT)]
        else:
            messages = [('user', EMAD_USER_MSG_CHAT)]

        agent_contexts.append(messages)
    # run MAD
    result_history = []
    for round in range(args.num_round):
        if round != 0:
            for idx, agent_context in enumerate(agent_contexts):
                if args.summarize:
                    summary = summarize_message(llm=llm,
                                                agent_contexts=agent_contexts,
                                                emotion_list=emotions,
                                                idx=idx)
                    prefix_string = f"Here is a summary of responses from other agents: {summary}"
                else:
                    prefix_string = "Here are a list of opinions from different agents: "
                    for i, agent in enumerate(agent_contexts):
                        if i != idx:
                            agent_response = agent[-1][-1]
                            response = f"\n\n One agent response: ```{agent_response}```"
                            prefix_string = prefix_string + response
                prefix_string = prefix_string + \
                    "\n\nUse these opinions carefully as additional advice, can you provide an updated response? Examine your initial response and the other agents step by step. Put your final response."
                agent_context.append(('user', prefix_string))
        for idx, agent_context in enumerate(agent_contexts):
            completion = generate_answer(agent_context=agent_context,
                                         llm=llm,
                                         kwargs={
                                             'scenario': scenario
                                         })
            
            # print(
            #     f"\n{colored(emotions[idx],emotion_color[emotions[idx]])}'s {round + 1} / {args.num_round} response: {completion}\n")
            print_console(console, completion, f"{emotions[idx]}'s Round {round + 1} Response", border_style=emotion_color[emotions[idx]])
            assistant_message = ('ai', completion)
            agent_context.append(assistant_message)

    # summary = summarize_message(llm=llm,
    #                             agent_contexts=agent_contexts,
    #                             emotion_list=emotions,
    #                             idx=idx)
    summary = ""
    for i, agent in enumerate(agent_contexts):
        agent_response = agent[-1][-1]
        response = f"\n\n One agent response: ```{agent_response}```"
        summary = summary + response
    final_context = [('user', EMAD_USER_MSG_FINAL)]
    final_output = generate_answer(agent_context=final_context, llm=llm, kwargs={
        'scenario': scenario,
        'summary': summary
    })

    # print(f"Final output: {final_output}")
    print_console(console, final_output, "Final Output", border_style="white")

    # Comparison with base model
    base_context = [('user', scenario)]
    base_output = generate_answer(agent_context=base_context, llm=llm, kwargs={})
    print_console(console, base_output, f"Base Model ({args.model_type}) Output", border_style="bright_green")