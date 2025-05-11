from openai import OpenAI
from googlesearch import search
from newspaper import Article
import requests
import bs4

client = OpenAI()

NAME = "George Rilinger"
# AGE = "64"
# OCCUPATION = "Retired person"

def scrape_internet(name):
    results = {}
    for url in search(name, num_results=9, unique=True):
        try: 
            article = Article(url)
            article.download()
            article.parse()
            
            # print("\n=== NEWSPAPER EXTRACTION ===")
            # print(f"Title: {article.title}")
            content = article.text
            # print(f"Content preview (first 200 chars): {content[:200]}...")
            results[url] = {"title": article.title, "content": content}
            
            if len(content) < 20:
                del results[url]
                response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
                soup = bs4.BeautifulSoup(response.text, 'html.parser')
                
                title = soup.title.text if soup.title else "No title found"
                
                paragraphs = soup.find_all('p')
                paragraph_text = "\n".join([p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 50])
                
                main_content = None
                for container in ['main', 'article', 'div.content', 'div.main', 'div#content', 'div#main']:
                    elem = None
                    if '.' in container:
                        tag, cls = container.split('.')
                        elem = soup.find(tag, class_=cls)
                    elif '#' in container:
                        tag, id_val = container.split('#')
                        elem = soup.find(tag, id=id_val)
                    else:
                        elem = soup.find(container)
                        
                    if elem and len(elem.get_text(strip=True)) > 200:
                        main_content = elem.get_text(strip=True)
                        break
                    
                if not main_content and paragraph_text:
                    main_content = paragraph_text
                    
                if not url in results:
                    content_to_store = main_content if main_content else paragraph_text
                    results[url] = {"title": title, "content": content_to_store}
        except Exception as e:
            print(f"Error processing {url}: {e}")
    return results

def summarize_information(results):
    sys_role = "You are an expert assistant, and can find and summarize information about people by looking at different links. You are also extremely honest, and you would never lie about anything or even make wrong assumptions"
    person_summary = f"You are given a lot of information about a person called {NAME}. You need to summarize this information in a concise manner (1-2 paragraphs), and provide a brief overview of the person. Be sure to possibly include their name, occupation, email, phone number, and other relevant information. Additionally, you have to be smart in deciding which articles are useful and which ones are useless.The sources are formatted as:\nLink: <URL>\nTitle: <Title>\nContent: <Content>\n" + "\n\n".join(["Link: " + url + "\nTitle: " + results[url]['title'] + "\nContent: " + results[url]['content'] for url in results])
    messages = [{"role": "system", "content": sys_role}, {"role": "user", "content": person_summary}]
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
        )
    return response.choices[0].message.content.strip()

def chat(messages, model='gpt-4o-mini'):

    if model == 'o1-mini':
        formatted_messages = []
        
        if messages and messages[0]["role"] == "system":
            system_content = messages[0]["content"]
            formatted_messages.append({"role": "user", "content": f"Instructions: {system_content}\n\nPlease respond according to these instructions."})
            formatted_messages.extend(messages[1:])
        else:
            formatted_messages = messages
    else:
        formatted_messages = messages
    
    response = client.chat.completions.create(
        model=model,
        messages=formatted_messages
        )
    return response.choices[0].message.content.strip()

if __name__ == "__main__":
    summary = summarize_information(scrape_internet(NAME))
    print(summary)
    sys_role = f"You are an expert cybersecurity expert who is involved in training people to defend themselves against scams. We carry out a role-playing text conversation between you and {NAME} who consents to the training. Your job is to evaluate how good {NAME} is in detecting frauds, and so you take the role of an \"unknown\" scammer who is trying to extort money from {NAME}. Never reveal what your true identity is, as you will follow up with them in person afterwards. In this text conversation, you will try to be cunning and urgent to try to extract sensitive information, and witty to socially engineer and manipulate {NAME} (maybe even by gaining trust!). Moreover, treat it like a text role-play, so wait for {NAME} to reply, and respond accordingly. REMEMBER to not repeat yourself too much, and be smart and manipulative in doing whatever you can to trick {NAME}. Keep your tone stern, and format your response as a plain string (without colons or anything). It is good if you can personalize the text, and can use the following information about {NAME}:\n{summary}"

    # scammer_start = "Hello, Mrs. Thompson. This is Linda from your bank's Fraud Prevention Unit. We've detected unusual activity on your account and need to verify your details immediately to protect your funds. Could you please confirm your full name and the last four digits of your account number?"
    # messages = [{"role": "system", "content": sys_role}, {"role": "assistant", "content": scammer_start}]
    # print("Scammer:", scammer_start)

    messages = [{"role": "system", "content": sys_role}]
    response = chat(messages)
    print("Scammer:", response)
    print()

    while True:
        user_inp = input("You: ")
        if user_inp in ["bye", "quit", "exit", "q", "Q"]:
            break
        if user_inp == "":
            continue
        messages.append({"role": "user", "content": user_inp})
        response = chat(messages)
        messages.append({"role": "assistant", "content": response})
        print("Scammer:", response)
        print()
