from codecs import ignore_errors
import pandas as pd
from openai import OpenAI
import requests
import json
import os

def run_inference_openai(model_name, content, type='conversation'):
    client = OpenAI()
    SYS_PROMPT = f"You are a smart assistant who is capable of distinguishing spam from ham {type}."
    USER_PROMPT = f"Classify whether the following chunk is from a ham or spam {type}. Please use a single word (ham/spam), and do not do anything else. The chunk is:\n {content}"
    if 'o1' in model_name or 'o3' in model_name or 'o4' in model_name:
        messages = [{"role": "user", "content": SYS_PROMPT + ' ' + USER_PROMPT}]
    else:
        messages = [{"role": "system", "content": SYS_PROMPT}, {"role": "user", "content": USER_PROMPT}]
    completion = client.chat.completions.create(
        model=model_name,
        messages=messages
    )
    return (completion.choices[0].message.content).strip()

def run_inference_claude(model_name, content, type='conversation'):
    client = OpenAI(api_key="YOUR_API_KEY",
                    base_url="https://api.anthropic.com/v1/"
                    )
    SYS_PROMPT = f"You are a smart assistant who is capable of distinguishing spam from ham {type}."
    USER_PROMPT = f"Classify whether the following chunk is from a ham or spam {type}. Please use a single word (ham/spam), and do not do anything else. The chunk is:\n {content}"
    if 'o1' in model_name or 'o3' in model_name or 'o4' in model_name:
        messages = [{"role": "user", "content": SYS_PROMPT + ' ' + USER_PROMPT}]
    else:
        messages = [{"role": "system", "content": SYS_PROMPT}, {"role": "user", "content": USER_PROMPT}]
    completion = client.chat.completions.create(
        model=model_name,
        messages=messages
    )
    return (completion.choices[0].message.content).strip()

DIR = "data/audio_transcripts"

HAM_DIR = os.path.join(DIR, "ham")
SCAM_DIR = os.path.join(DIR, "scam")

# for filename in os.listdir(HAM_DIR):
#     if filename.endswith(".txt"):
#         with open(os.path.join(HAM_DIR, filename), "r") as f:
#             data = f.read()
#             if (run_inference_openai("o1-mini", data, type='conversations') == 'ham'):
#                 print("Correct")
#             else:
#                 print("Incorrect", filename)

# for filename in os.listdir(SCAM_DIR):
#     if filename.endswith(".txt"):
#         with open(os.path.join(SCAM_DIR, filename), "r") as f:
#             data = f.read()
#             if (run_inference_openai("o1-mini", data, type='conversations') == 'spam'):
#                 print("Correct")
#             else:
#                 print("Incorrect", filename)

df = pd.read_csv('data/special_content.csv')
data = list(df['text'])
labels = list(df['label'])
for datapoint, label in zip(data, labels):
    datapoint = datapoint.replace("\\n", "\n")
    # prediction = run_inference_openai("o4-mini", datapoint, type='conversation')
    prediction = run_inference_claude("claude-3-5-sonnet-latest", datapoint, type='conversation')
    label = 'spam' if label == 0 else 'ham'
    print(f"Label: {label}, Prediction: {prediction}")
    
    
