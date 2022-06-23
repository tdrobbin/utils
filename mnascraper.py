"""
exmple usage from directory of mna.xls:

python mnascraper.py get_all_deal_prices ./mna.xlsx 

# requirements.txt
requests
pandas
lxml
beautifulsoup4
loguru
tqdm
fire
"""

from ctypes import resize
from urllib import response
from webbrowser import get
import requests
import pandas as pd
import numpy as np
import re
from urllib.parse import quote_plus
import urllib
from bs4 import BeautifulSoup
from loguru import logger
from tqdm.auto import tqdm
from pathlib import Path
import fire
import sys


TARGET_DOMAINS = [
        'businesswire.com',
        'globenewswire.com',
        'reuters.com',
        'bloomberg.com',
        'wsj.com',
        'barrons.com',
        'pehub.com',
        'mergr.com',
        'globest.com'
    ]


FIREFOX_HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:98.0) Gecko/20100101 Firefox/98.0",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
        "Cache-Control": "max-age=0",
    }



def get_google_search_links(qry, start=None, end=None):
    """
    take a search query and return a list of links from first page of query search results

    example search qry:
    https://www.google.com/search?q=natus+medical+acquisition&tbs=cdr:1,cd_min:4/7/2022,cd_max:4/21/2022
    
    # mostly from website below with some modifications for time range searches and headers
    # https://practicaldatascience.co.uk/data-science/how-to-scrape-google-search-results-using-python
    """

    if start is not None or end is not None:
        assert start is not None and end is not None

        start = pd.Timestamp(start).strftime("%m/%d/%Y")
        end = pd.Timestamp(end).strftime("%m/%d/%Y")

        url = f"https://www.google.com/search?q={quote_plus(qry)}&tbs=cdr:1,cd_min:{quote_plus(start)},cd_max:{quote_plus(end)}"
    else:
        url = f"https://www.google.com/search?q={quote_plus(qry)}"

    logger.debug(f'searching for: {qry}')
    response = requests.get(url, headers=FIREFOX_HEADERS)
    bs = BeautifulSoup(response.content.decode('utf-8'), features='lxml')
    
    links = [a.get('href') for a in bs.find_all('a') if a.get('href') is not None]
    links = [link for link in links if re.search('^https://', link) is not None]

    google_domains = (
        'https://www.google.', 
        'https://google.', 
        'https://webcache.googleusercontent.', 
        'http://webcache.googleusercontent.', 
        'https://policies.google.',
        'https://support.google.',
        'https://maps.google.'
    )

    links = [link for link in links if not link.startswith(google_domains)]

    return links


def parse_links(links):
    """
    take a list of links and find match for story from news domain in preference
    of order of TARGET_DOMAINS. once found best article, parse for relevent text
    on deal price and return string
    """
    target_link = None
    for link in links:
        for domain in TARGET_DOMAINS:
            if re.search(domain, link) is not None:
                target_link = link
                break
        if target_link is not None:
            break
    
    if target_link is None:
        logger.debug('target link NOT found')
        return None, None
    
    logger.debug(f'target link found for domain: {domain}')
    logger.debug(f'target link: {link}')

    # now get price per share which usually is in the pattern of
    # $33.50 in cash for each share
    regex = r'\$\d{1,6}(\.)?(\d{0,2})?.{0,30}\s(each|per)\sshare'

    response = requests.get(target_link, headers=FIREFOX_HEADERS)
    bs = BeautifulSoup(response.content.decode('utf-8'), features='lxml')
    target_text = re.search(regex, bs.find('body').text)

    if target_text is not None:
        logger.debug(f'target text found: {target_text}')
    else:
        logger.debug(f'target text NOT found')

    return target_link, target_text.group()


def parse_target_text(target_text):
    """
    take string returned from parse_links and get float representing deal price
    """
    # just extract currency substring
    regex = r'^\$(\d|\.)+\s'
    currency = re.search(regex, target_text).group().strip()

    return float(currency.replace('$', ''))


def get_deal_info(target, date, acquirer=None):
    query = target + ' acquisition'

    start = pd.Timestamp(date) - pd.offsets.Day(7)
    end = pd.Timestamp(date) + pd.offsets.Day(7)

    links = get_google_search_links(query, start, end)
    target_link, target_text = parse_links(links)

    if target_link is not None and target_text is not None:
        price = parse_target_text(target_text)
    else:
        price = np.nan

    return dict(
        target=target,
        date=date,
        acquirer=acquirer,
        query_start=start,
        query_end=end,
        query=query,
        link=target_link,
        text=target_text,
        price=price
    )


def get_all_deal_prices(fpath, write_excel=True, logging_level='INFO'):
    """
    fpath is excel file with following columns:
    target, date, acquierer
    """
    logger.remove()
    logger.add(sys.stderr, level=logging_level)

    df = pd.read_excel(fpath)
    logger.info(f'searching web for deal info for {len(df)} targets')

    all_info = []
    for i, deal in tqdm(df.iterrows(), total=len(df)):
        info = get_deal_info(target=deal.target, date=deal.date)
        all_info.append(info)

    all_info = pd.DataFrame(all_info)
    
    if write_excel:
        fpath = Path(fpath)
        output_fpath = fpath.parent / 'mna_scraped_deal_prices.xlsx'
        all_info.to_excel(output_fpath)
        logger.info(f'data saved: {output_fpath}')

    else:
        return all_info


if __name__ == '__main__':
    fire.Fire()