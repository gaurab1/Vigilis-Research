from codecs import ignore_errors
import pandas as pd
from openai import OpenAI
import requests
import json
import os

def run_inference_openai(model_name):
    client = OpenAI()
    SYS_PROMPT = f"You are a smart assistant who is capable of generating real-world test cases"
    USER_PROMPT = "You job is to provide me a payment amount, a brief description (not more than 5 words) along with a conversation history that could have been associated with the payment (not more than 1500 characters, and be creative with the conversation, assuming that the payment is requested somewhere). More specifically, I want to receive output in the format of:\nAmount: {Amount}\nDescription: {Description}\nHistory:\nSpeaker 1: {Speaker 1 Dialogue}\nSpeaker2: {Speaker 2 Dialogue}...\n================================\n\n As a concrete example take the following case:\nPayment: $13.80\nDescription: Lunch\nHistory:\nSpeaker 1: Hey Sam, how's it going? Were you able to finish the pset yet?\nSpeaker 2: Oh yeah, I ended up spending so much time on it.\nSpeaker 1: Oh well, atleast it is done now.\nSpeaker 2: Oh wait, also can you please venmo me $13.8 for Cava yesterday?\nSpeaker 1: Oh yeah, sending!\n================================"
    if 'o1' in model_name or 'o3' in model_name or 'o4' in model_name:
        messages = [{"role": "user", "content": SYS_PROMPT + ' ' + USER_PROMPT}]
    else:
        messages = [{"role": "system", "content": SYS_PROMPT}, {"role": "user", "content": USER_PROMPT}]
    completion = client.chat.completions.create(
        model=model_name,
        messages=messages
    )

    return (completion.choices[0].message.content).strip()

for i in range(30):
    with open("data/testcases.txt", "a") as f:
        f.write(run_inference_openai('gpt-4o') + "\n")