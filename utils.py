import re
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from collections import Counter

def parse_answer(input_str):
    pattern = r'\((\w)\)'
    matches = re.findall(pattern, input_str)
    solution = None
    for match_str in matches[::-1]:
        solution = match_str.upper()
        if solution:
            break
    return solution

def summarize_message(llm, agent_contexts, emotion_list, idx):
    prefix_string = "Here are a list of opinions from different agents: "
    for i, (agent, emotion) in enumerate(zip(agent_contexts, emotion_list)):
        if i!=idx:
            agent_response = agent[-1][-1]
            response = f"\n\n {emotion} response: ```{agent_response}```"
            prefix_string = prefix_string + response
    prefix_string = prefix_string + "\n\n Write a summary of the different opinions from each of the individual agent."
    agent_context = [('user', prefix_string)]
    completion = generate_answer(agent_context=agent_context, llm=llm, kwargs={})
    return completion


def mad_summarize_message(llm, agent_contexts, idx):
    prefix_string = "Here are a list of opinions from different agents: "
    for i, agent in enumerate(agent_contexts):
        if i!=idx:
            agent_response = agent[-1][-1]
            response = f"\n\n One agent response: ```{agent_response}```"    
            prefix_string = prefix_string + response
    prefix_string = prefix_string + "\n\n Write a summary of the different opinions from each of the individual agent."
    agent_context = [('user', prefix_string)]
    completion = generate_answer(agent_context=agent_context, llm=llm, kwargs={})
    return completion

def generate_answer(agent_context, llm, kwargs):
    prompts = ChatPromptTemplate.from_messages(agent_context)
    chain = prompts | llm
    output = chain.invoke(kwargs)
    content = output.content
    return content

def most_frequent(result):
    return Counter(result).most_common(1)[0][0]


class selectedEmotions(BaseModel):
    emotions: list[str] = Field(description="a list of activated emotions")

class finalResult(BaseModel):
    explanation: str = Field(description="explanation for the final response")
    response: str = Field(description="final response for the given scenario")

ES_SYS_MSG = """You are an emotion manager, and your job is to tell the user which emotions should be activated to the given situation. According to Robert Plutchik's emotion wheel, you have eight base emotions which are the following:
- 'joy'
- 'trust'
- 'fear'
- 'surprise'
- 'sadness'
- 'disgust'
- 'anger'
- 'anticipation'"""

ES_USER_MSG = """Based on the given scenario, tell me {subject}'s base emotions that could be activated.
Here is a list of base emotions that can be selected: ['joy', 'trust', 'fear', 'surprise', 'sadness', 'disgust', 'anger', 'anticipation']
Also follow the format instruction when responsing.

format instruction: {format_instruction}
scenario: {scenario}
emotions: """

ES_USER_MSG_CHAT = """Based on the given scenario, tell me the user's base emotions that could be activated.
Here is a list of base emotions that can be selected: ['joy', 'trust', 'fear', 'surprise', 'sadness', 'disgust', 'anger', 'anticipation']
Also follow the format instruction when responsing.

format instruction: {format_instruction}
scenario: {scenario}
emotions: """

EMAD_SYS_MSG = "You are {emotion}, one of the emotions of yours."

# EMAD_USER_MSG = """You are a debater. Hello and welcome to the debate competition. It's not necessary to fully agree with each other's perspectives, as our objective is to find the correct response. Given a situation, subject, and possible choices that the given subject could execute, please choose the most proper choice.
# Explain your answer, putting the answer in the form (A~D) at the end of your response.
# Situation: {scenario}
# Subject: {subject}
# Possible choices: {choices}

# Your response: """

EMAD_USER_MSG = """Given a situation, subject, and possible choices that the given subject could execute, please choose the most proper choice.
Explain your answer, putting the answer in the form of (A), (B), (C), or (D) at the end of your response (Choose only one choice among the possible choices).
Situation: {scenario}
Subject: {subject}
Possible choices: {choices}

Your response: """

EMAD_USER_MSG_CHAT = """Given a situation, please generate the most appropriate response.
Explain why your response is the most appropriate.
Situation: {scenario}
Your response: """

EMAD_USER_MSG_FINAL = """Given a situation and the summary from other agents, please generate the most appropriate response.
Situation: {scenario}
Here is a summary of other agents' responses: {summary}
Your response: """

EUMAD_USER_MSG = """Scenario:\n{scenario}\nQuestion: What emotion(s) would {subject} ultimately feel in this situation?\nChoices:\n{choices}\n"
Explain your answer, and provide only one answer in the form (X) at the end of your response.
Your response: """