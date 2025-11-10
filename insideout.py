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
                   selectedEmotions,
                   ES_SYS_MSG,
                   ES_USER_MSG,
                   EMAD_SYS_MSG,
                   EMAD_USER_MSG,
                   EUMAD_USER_MSG)

emotion_color = {
    'joy': 'yellow',
    'trust': 'light_green',
    'fear': 'green',
    'surprise': 'light_blue',
    'sadness': 'blue',
    'disgust': 'magenta',
    'anger': 'red',
    'anticipation': 'light_red',
    'rational': 'cyan'
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_type", type=str, choices=['ea', 'eu'])
    parser.add_argument("--model_type", type=str, default='gpt-3.5-turbo-0125')
    parser.add_argument("--num_round", type=int, default=2)
    parser.add_argument("--summarize", type=int, default=1)
    parser.add_argument("--select_emotions", type=int, default=1)
    parser.add_argument("--add_rational", type=int, default=0)
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

    # setup Emotion Selection chain
    es_prompts = ChatPromptTemplate.from_messages([
        ('system', ES_SYS_MSG),
        ('user', ES_USER_MSG)
    ])
    parser = PydanticOutputParser(pydantic_object=selectedEmotions)
    es_chain = es_prompts | llm | parser

    dataset = dataset[4:5]
    # print(dataset)
    for idx, inst in enumerate(dataset):
        if args.dataset_type == 'ea':
            scenario = inst['scenario']
            subject = inst['subject']
            choices = inst['choices']
            choices_dict = {chr(idx+ord('A')): emotion for idx, emotion in enumerate(choices)}
            choices = {emotion: chr(idx+ord('A')) for idx, emotion in enumerate(choices)}
            label = choices[inst['label']]
            choices = [f"({choices[key]}). {key}" for key in choices]
            choices = ' '.join(choices)
            print(
                f"{idx+1} / {len(dataset)}\nScenario: {scenario}\nsubject: {subject}\nChoices: {choices}\n")

            # select emotions
            if args.select_emotions:
                trial=0
                while trial < 5:
                    try:
                        output = es_chain.invoke({
                            'scenario': scenario,
                            'subject': subject,
                            'format_instruction': parser.get_format_instructions()
                        })
                        emotions = output.emotions
                        flag=True
                        for emotion in emotions:
                            if emotion not in list(emotion_color.keys()):
                                flag=False
                        if flag:
                            break
                        else:
                            print(f"trial:{trial} Wrong emotion")
                            trial+=1
                            if trial >= 5:
                                emotions = [emotion for emotion in emotions if emotion in list(emotion_color.keys())]
                    except:
                        print(f"trial:{trial} Error occur. Catching exception...")
                        trial+=1
            else:
                if args.add_rational:
                    emotions = list(emotion_color.keys())
                else:
                    emotions = [emotion for emotion in emotion_color.keys() if emotion != 'rational']
            
            if args.add_rational:
                emotions.append('rational')
            
            
            print(f"Emotions for instance #{idx+1}: {emotions}")

            # Configure MAD
            agent_contexts = []
            for emotion in emotions:
                if emotion != 'rational':
                    messages = [
                        ('system', f"You are {emotion}, one of the emotions of yours"), ('user', EMAD_USER_MSG)]
                else:
                    messages = [('user', EMAD_USER_MSG)]

                agent_contexts.append(messages)
            # run MAD
            result_history=[]
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
                                if i!=idx:
                                    agent_response = agent[-1][-1]
                                    response = f"\n\n One agent response: ```{agent_response}```"
                                    prefix_string = prefix_string + response
                        prefix_string = prefix_string + \
                            "\n\nUse these opinions carefully as additional advice, can you provide an updated answer? Examine your solution and that other agents step by step. Put your answer in the form of (A), (B), (C), or (D) at the end of your response (Choose only one choice)."
                        agent_context.append(('user', prefix_string))
                for idx, agent_context in enumerate(agent_contexts):
                    while True:
                        completion = generate_answer(agent_context=agent_context,
                                                    llm=llm,
                                                    kwargs={
                                                        'scenario': scenario,
                                                        'subject': subject,
                                                        'choices': choices
                                                    })
                        answer_cand = parse_answer(completion)
                        conditional = [answer_cand == choice for choice in list(choices_dict.keys())]
                        if any(conditional):
                            break
                        else:
                            print("Completion does not contain any answer. Trying again...")
                    print(
                        f"\n{colored(emotions[idx],emotion_color[emotions[idx]])}'s {round + 1} / {args.num_round} response: {completion}\n")
                    assistant_message = ('ai', completion)
                    agent_context.append(assistant_message)
                result = []
                for agent_context in agent_contexts:
                    input_str = agent_context[-1][-1]
                    answer = parse_answer(input_str)
                    result.append(answer)
                    final_output = most_frequent(result)
                print(f"\n\nRound {round+1} result: {result}\n\n")
                result_history.append(result)
                if len(list(Counter(result).keys())) == 1:
                    print(f"Flush! in round {round + 1}")
                    break
                else:
                    print("No flush yet, continuing to next round...")
            print(f"Final output: {final_output}\tGold label: {label}")
            log_file_path = f"./logs/{args.dataset_type}_{args.model_type}_round:{args.num_round}_summarize:{args.summarize}"
            if args.select_emotions:
                log_file_path+="_select_emotions"
            if args.add_rational:
                log_file_path+="_add_rational"
            log_file_path = "./result"
            log_file_path+='.jsonl'
            with open(log_file_path, "a+") as f:
                json_inst = {
                    'scenario': scenario,
                    'subject': subject,
                    'choices': inst['choices'],
                    'pred': final_output,
                    'label': label,
                    'emotions': emotions,
                    'round': round+1,
                    'result_history': result_history
                }
                f.write(json.dumps(json_inst)+'\n')
        else:
            scenario = inst['scenario']
            subject = inst['subject']
            e_choices = inst['emotion_choices']
            e_choices_dict = {chr(idx+ord('A')): emotion for idx, emotion in enumerate(e_choices)}
            e_choices = {emotion: chr(idx+ord('A')) for idx, emotion in enumerate(e_choices)}
            e_label = e_choices[inst['emotion_label']]
            e_choices = [f"({e_choices[key]}). {key}" for key in e_choices]
            
            e_choices = ' '.join(e_choices)
            

            if args.select_emotions:
                trial=0
                while trial < 5:
                    try:
                        output = es_chain.invoke({
                            'scenario': scenario,
                            'subject': subject,
                            'format_instruction': parser.get_format_instructions()
                        })
                        emotions = output.emotions
                        flag=True
                        for emotion in emotions:
                            if emotion not in list(emotion_color.keys()):
                                flag=False
                        if flag:
                            break
                        else:
                            print(f"trial:{trial} Wrong emotion")
                            trial+=1
                            if trial >= 5:
                                emotions = [emotion for emotion in emotions if emotion in list(emotion_color.keys())]
                    except:
                        print(f"trial:{trial} Error occur. Catching exception...")
                        trial+=1
            else:
                if args.add_rational:
                    emotions = list(emotion_color.keys())
                else:
                    emotions = [emotion for emotion in emotion_color.keys() if emotion != 'rational']
            
            if args.add_rational:
                emotions.append('rational')

            print(f"Emotions for instance #{idx+1}: {emotions}")

            agent_contexts=[]
            for emotion in emotions:
                messages = [('system', f"You are {emotion}, one of the emotions of yours"), ('user', EUMAD_USER_MSG)]
                agent_contexts.append(messages)
            
            result_history=[]
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
                                if i!=idx:
                                    agent_response = agent[-1][-1]
                                    response = f"\n\n One agent response: ```{agent_response}```"
                                    prefix_string = prefix_string + response
                        prefix_string = prefix_string + \
                            "\n\nUse these opinions carefully as additional advice, can you provide an updated answer? Examine your solution and that other agents step by step. Put your answer in the form (X) at the end of your response (Choose only one choice)."
                        agent_context.append(('user', prefix_string))
                for idx, agent_context in enumerate(agent_contexts):
                    while True:
                        completion = generate_answer(agent_context=agent_context,
                                                    llm=llm,
                                                    kwargs={
                                                        'scenario': scenario,
                                                        'subject': subject,
                                                        'choices': e_choices
                                                    })
                        answer_cand = parse_answer(completion)
                        conditional = [answer_cand == choice for choice in list(e_choices_dict.keys())]
                        if any(conditional):
                            break
                        else:
                            print("Completion does not contain any answer. Trying again...")
                    print(
                        f"{colored(emotions[idx],emotion_color[emotions[idx]])}'s {round + 1} / {args.num_round} response: {completion}")
                    assistant_message = ('ai', completion)
                    agent_context.append(assistant_message)
                result = []
                for agent_context in agent_contexts:
                    input_str = agent_context[-1][-1]
                    answer = parse_answer(input_str)
                    result.append(answer)
                    final_output = most_frequent(result)
                result_history.append(result)
                print(f"\n\nRound {round+1} result: {result}\n\n")
                if len(list(Counter(result).keys())) == 1:
                    print(f"Flush! in round {round + 1}")
                    break
                else:
                    print("No flush yet, continuing to next round...")
            print(f"Final output: {final_output}\tGold label: {e_label}")
            log_file_path = f"./logs/{args.dataset_type}_{args.model_type}_round:{args.num_round}_summarize:{args.summarize}"
            if args.select_emotions:
                log_file_path+="_select_emotions"
            if args.add_rational:
                log_file_path+="_add_rational"
            log_file_path = "./result"
            log_file_path+='.jsonl'
            with open(log_file_path, "a+") as f:
                json_inst = {
                    'scenario': scenario,
                    'subject': subject,
                    'choices': inst['emotion_choices'],
                    'pred': final_output,
                    'label': e_label,
                    'emotions': emotions,
                    'round': round+1,
                    'result_history': result_history
                }
                f.write(json.dumps(json_inst)+'\n')

