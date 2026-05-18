from collections import deque
import hashlib
import os
from urllib.parse import urljoin

from bs4 import BeautifulSoup
from markdownify import markdownify
import requests


def fetch_page(url):
    try:
        response = requests.get(url, timeout=10)

        if response.status_code != 200:
            print(f'Return code for {url} is not 200')
            return None

        return response.text
    except Exception as e:
        print(f'Error fetching {url}: {e}')
        return None

def extract_content(page):
    soup = BeautifulSoup(page, 'html.parser')

    for tag in soup(['nav', 'footer', 'script', 'style']):
        tag.decompose()

    main = soup.find('div', {'role': 'main'})
    return main

def save_raw(url, content, raw_dir):
    content_str = str(content)
    markdown = markdownify(content_str, heading_style='ATX')

    file_hash = hashlib.md5(url.encode()).hexdigest()
    path = os.path.join(raw_dir, f'{file_hash}.md')

    with open(path, 'w', encoding='utf-8') as f:
        f.write(markdown)


def crawl(base_url, raw_dir):

    queue = deque([base_url])
    visited = set()

    while queue:
        url = queue.popleft()

        if url in visited:
            continue

        visited.add(url)

        page = fetch_page(url)
        if not page:
            continue

        content = extract_content(page)
        if not content:
            continue

        save_raw(url, content, raw_dir)

        soup = BeautifulSoup(page, "html.parser")

        for a in soup.find_all("a", href=True):
            link = urljoin(url, a["href"])

            if link.startswith(base_url):
                if link not in visited:
                    queue.append(link)


if __name__=='__main__':
    from constants import BASE_URL, RAW_DIR

    crawl(base_url=BASE_URL, raw_dir=RAW_DIR)