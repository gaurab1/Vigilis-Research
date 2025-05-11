from googlesearch import search
import bs4
import requests
import re
from openai import OpenAI

from newspaper import Article

query = "George Rilinger"
results = {}

for url in search(query, num_results=9, unique=True):
    try: 
        article = Article(url)
        article.download()
        article.parse()
        
        print("\n=== NEWSPAPER EXTRACTION ===")
        print(f"Title: {article.title}")
        
        content = article.text
        print(f"Content preview (first 200 chars): {content[:200]}...")
        results[url] = {"title": article.title, "content": content}
        
        if len(content) < 20:
            del results[url]
            response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
            soup = bs4.BeautifulSoup(response.text, 'html.parser')
            
            title = soup.title.text if soup.title else "No title found"
            print("\n=== BEAUTIFULSOUP EXTRACTION ===")
            print(f"Title: {title}")
            
            paragraphs = soup.find_all('p')
            paragraph_text = "\n".join([p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 50])
            
            print(f"Paragraph content preview: {paragraph_text[:200]}...")
            
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
                    print(f"Found main content in <{container}>")
                    break
                    
            if main_content:
                print(f"Main container content preview: {main_content[:200]}...")
                
            if not main_content and paragraph_text:
                main_content = paragraph_text
                
            if not url in results:
                content_to_store = main_content if main_content else paragraph_text
                results[url] = {"title": title, "content": content_to_store}
            
    except Exception as e:
        print(f"Error processing {url}: {e}")

print("\n=== SUMMARY ===")
print(f"Successfully processed {len(results)} URLs")

client = OpenAI()
sys_role = f"You are an expert assistant, and can find and summarize information about people by looking at different links. You are also extremely honest, and you would never lie about anything or even make wrong assumptions"
person_summary = "You are given a lot of information about a person called {NAME}. You need to summarize this information in a concise manner (1-2 paragraphs), and provide a brief overview of the person. Be sure to possibly include their name, occupation, email, phone number, and other relevant information. Additionally, you have to be smart in deciding which articles are useful and which ones are useless.The sources are formatted as:\nLink: <URL>\nTitle: <Title>\nContent: <Content>\n" + "\n\n".join(["Link: " + url + "\nTitle: " + results[url]['title'] + "\nContent: " + results[url]['content'] for url in results])
print(person_summary)

def summary():
    messages = [{"role": "system", "content": sys_role}, {"role": "user", "content": person_summary}]
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
        )
    return response.choices[0].message.content.strip()

print("\n=== PERSON SUMMARY ===")
print(chat())