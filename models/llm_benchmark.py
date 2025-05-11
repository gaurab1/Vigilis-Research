import pandas as pd
from openai import OpenAI
import requests
import json

# Load and preprocess datasets
email_ds = pd.read_csv('data/spamassassin.csv')
text_ds = pd.read_csv('data/spam.csv', encoding='latin-1')

# Shuffle the datasets
text_ds = text_ds.sample(frac=1, random_state=42)[['label', 'text']]
email_ds = email_ds.sample(frac=1, random_state=42)[['label', 'text']]

def run_inference_openai(model_name, content, type='text message'):
    client = OpenAI()
    SYS_PROMPT = f"You are a developed human who is capable of distinguishing spam from ham {type}."
    USER_PROMPT = f"Classify whether the following {type} is ham or spam. Please use a single word (ham/spam), and do not do anything else. The sentence is:\n {content}"
    if 'o1' in model_name or 'o3' in model_name:
        messages = [{"role": "user", "content": SYS_PROMPT + ' ' + USER_PROMPT}]
    else:
        messages = [{"role": "system", "content": SYS_PROMPT}, {"role": "user", "content": USER_PROMPT}]
    completion = client.chat.completions.create(
        model=model_name,
        messages=messages
    )
    return (completion.choices[0].message.content).strip()

def run_multishot_inference_openai(model_name, content, spam_examples=None, ham_examples=None, type='text message'):
    client = OpenAI()
    SYS_PROMPT = f"You are a developed human who is capable of distinguishing spam from ham {type}."
    if spam_examples:
        SYS_PROMPT += f"\nSome examples of spam {type}:\n" + '\n'.join(spam_examples)
    if ham_examples:
        SYS_PROMPT += f"\nSome examples of ham {type}:\n" + '\n'.join(ham_examples)
    USER_PROMPT = f"Classify whether the following {type} is ham or spam. Please use a single word (ham/spam), and do not do anything else. The sentence is:\n {content}"
    if 'o1' in model_name or 'o3' in model_name:
        messages = [{"role": "user", "content": SYS_PROMPT + ' ' + USER_PROMPT}]
    else:
        messages = [{"role": "system", "content": SYS_PROMPT}, {"role": "user", "content": USER_PROMPT}]
    completion = client.chat.completions.create(
        model=model_name,
        messages=messages
    )
    return (completion.choices[0].message.content).strip()

def run_inference_ollama(content, type='text message'):
    api_url = "http://localhost:11434/api/generate"
    
    # Prepare the prompt similar to the OpenAI version
    system_prompt = f"You are a developed human who is capable of distinguishing spam from ham {type}."
    user_prompt = f"Classify whether the following {type} is ham or spam. Please use a single word (ham/spam), and do not do anything else. The sentence is:\n {content}"
    
    full_prompt = f"{system_prompt}\n\n{user_prompt}"
    
    payload = {
        "model": "deepseek-r1:7b",
        "prompt": full_prompt,
        "stream": False,
    }

    response = requests.post(api_url, json=payload)
    response.raise_for_status()  # Raise exception for HTTP errors
        
    # Parse the response
    result = response.json()
        
    # Extract the generated text
    generated_text = result.get("response", "").strip()
    generated_text = generated_text.split("\n")[-1]
        
    if "ham" in generated_text.lower():
        return "ham"
    elif "spam" in generated_text.lower():
        return "spam"
    else:
        return generated_text

def run_multishot_inference_ollama(content, spam_examples=None, ham_examples=None, type='text message'):
    api_url = "http://localhost:11434/api/generate"
    
    # Prepare the prompt similar to the OpenAI version
    system_prompt = f"You are a developed human who is capable of distinguishing spam from ham {type}."
    if spam_examples:
        system_prompt += f"\nSome examples of spam {type}:\n" + '\n'.join(spam_examples)
    if ham_examples:
        system_prompt += f"\nSome examples of ham {type}:\n" + '\n'.join(ham_examples)
    user_prompt = f"Classify whether the following {type} is ham or spam. Please use a single word (ham/spam), and do not do anything else. The sentence is:\n {content}"
    
    full_prompt = f"{system_prompt}\n\n{user_prompt}"
    
    payload = {
        "model": "deepseek-r1:7b",
        "prompt": full_prompt,
        "stream": False,
    }

    response = requests.post(api_url, json=payload)
    response.raise_for_status()  # Raise exception for HTTP errors
        
    # Parse the response
    result = response.json()
        
    # Extract the generated text
    generated_text = result.get("response", "").strip()
    generated_text = generated_text.split("\n")[-1]
        
    if "ham" in generated_text.lower():
        return "ham"
    elif "spam" in generated_text.lower():
        return "spam"
    else:
        return generated_text

def evaluate(model_name='ollama', type='text messages', test_count=500):
    log_string = ""
    correct = 0
    ds = text_ds if type == 'text messages' else email_ds
    spam_examples = list(ds[ds['label'] == 0]['text'].iloc[test_count: test_count + 10])
    ham_examples = list(ds[ds['label'] == 1]['text'].iloc[test_count: test_count + 10])
    
    for i in range(test_count):
        # print(i)
        text = ds['text'].iloc[i]
        label = 'spam' if ds['label'].iloc[i] == 0 else 'ham'
        if model_name == 'ollama':
            prediction = run_inference_ollama(text, type=type)
        else:
            prediction = run_multishot_inference_openai(model_name, text, spam_examples, ham_examples, type=type)
        if prediction == label:
            correct += 1
        log_string += f"Label: {label}, Prediction: {prediction}\n"
        print(f"Label: {label}, Prediction: {prediction}")
    
    with open(f"{model_name}_{type}_multishot.log", "w") as f:
        f.write(log_string)
    
    print(f"{model_name} {type} accuracy: {correct/test_count:.4f}")

if __name__ == '__main__':
    evaluate(model_name='ollama', type='emails')
    # evaluate(model_name='gpt-4o-mini', type='emails')

        

