import sys
from bs4 import BeautifulSoup

if len(sys.argv) < 2:
    print('Usage: python print_cleaned_text.py <html_file>')
    sys.exit(1)

file_path = sys.argv[1]
with open(file_path, 'r', encoding='utf-8') as f:
    html_content = f.read()

soup = BeautifulSoup(html_content, 'html.parser')
for element in soup.find_all(['script', 'style', 'nav', 'footer', 'header', 'noscript', 'aside', 'sidebar', 'menu', 'picture', 'source', 'track', 'textarea', 'canvas', 'svg', 'figure', 'figcaption']):
    element.decompose()
for div in soup.find_all('div', class_=lambda x: x and any(term in str(x).lower() for term in ['cookie', 'popup', 'ad', 'social', 'share', 'related', 'sidebar', 'menu', 'footer', 'header'])):
    div.decompose()

main_content = None
selectors = ['article', 'main', 'div[class*=content]', 'div[class*=story]', 'div[class*=case-study]']
for selector in selectors:
    main_content = soup.select_one(selector)
    if main_content:
        break
content = []
if main_content:
    for tag in main_content.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li']):
        text = tag.get_text(strip=True)
        if text and not any(phrase in text.lower() for phrase in ['cookie', 'privacy', 'terms', 'subscribe', 'newsletter']):
            content.append(text)
clean_text = '\n'.join(content)

if not clean_text.strip():
    meta_desc = soup.find('meta', attrs={'name': 'description'})
    if meta_desc and meta_desc.get('content'):
        clean_text = meta_desc['content'].strip()
        print('Used <meta name="description"> as fallback')
if not clean_text.strip():
    title_tag = soup.find('title')
    if title_tag and title_tag.text.strip():
        clean_text = title_tag.text.strip()
        print('Used <title> as fallback')
if not clean_text.strip():
    all_p = [p.get_text(strip=True) for p in soup.find_all('p') if p.get_text(strip=True)]
    if all_p:
        clean_text = '\n'.join(all_p)
        print('Used all <p> tags as fallback')

print(f'Cleaned text length: {len(clean_text)}')
print('Sample cleaned text:', clean_text[:300].replace('\n', ' '))
print('--- FULL CLEANED TEXT ---')
print(clean_text) 