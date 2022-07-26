

import pandas as pd
import datetime
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
from loguru import logger
from tqdm.auto import tqdm

def get_main_url_info():
    return \
    [
        {'url': 'https://seekingalpha.com/etfs-and-funds/etf-tables/key_markets',
        'category': 'Key Market ETFs'},
        {'url': 'https://seekingalpha.com/etfs-and-funds/etf-tables/bonds',
        'category': 'Bond ETFs'},
        {'url': 'https://seekingalpha.com/etfs-and-funds/etf-tables/countries',
        'category': 'Country ETFs'},
        {'url': 'https://seekingalpha.com/etfs-and-funds/etf-tables/currencies',
        'category': 'Currency ETFs'},
        {'url': 'https://seekingalpha.com/etfs-and-funds/etf-tables/dividends',
        'category': 'Dividend ETFs'},
        {'url': 'https://seekingalpha.com/etfs-and-funds/etf-tables/emerging_markets',
        'category': 'Emerging Market ETFs'},
        {'url': 'https://seekingalpha.com/etfs-and-funds/etf-tables/global_and_regions',
        'category': 'Global & Regional ETFs'},
        {'url': 'https://seekingalpha.com/etfs-and-funds/etf-tables/growth_vs_value',
        'category': 'Growth vs. Value ETFs'},
        {'url': 'https://seekingalpha.com/etfs-and-funds/etf-tables/market_cap',
        'category': 'Market Cap ETFs'},
        {'url': 'https://seekingalpha.com/etfs-and-funds/etf-tables/real_estate',
        'category': 'Real Estate ETFs'},
        {'url': 'https://seekingalpha.com/etfs-and-funds/etf-tables/sectors',
        'category': 'Sector ETFs'},
        {'url': 'https://seekingalpha.com/etfs-and-funds/etf-tables/strategies',
        'category': 'ETF Strategies'},
        {'url': 'https://seekingalpha.com/etfs-and-funds/etf-tables/smart_beta',
        'category': 'Smart Beta'},
        {'url': 'https://seekingalpha.com/etfs-and-funds/etf-tables/themes_and_subsectors',
        'category': 'Themes & Subsectors ETFs'}
  ]


def fetch_url_tables(url, category, browser):
    # import ipdb; ipdb.set_trace()
    page = browser.new_page()
    page.goto(url)
    page.wait_for_timeout(5000)
    bs = BeautifulSoup(page.content(), features='lxml')
    page.close

    scs = bs.find_all('section', attrs={'data-test-id':'card-container'})
    dfs = []
    for s in scs:
        subcat = s.find('h1', attrs={'data-test-id':'cards-title'}).text
        df = pd.read_html(s.decode())[0]
        df = df.drop(columns=['Day Range', '52 Week Range'])
        df['category'] = category
        df['subcategory'] = subcat
        df['url'] = url
        dfs.append(df)

    df = pd.concat(dfs, axis=0)
    
    cols = ['Today', '1 Month', 'YTD', '1 Year', '3 Years']
    df[cols] = df[cols].applymap(lambda x: x.replace('%', ''))
    df[cols] = df[cols].replace('-', None)
    df[cols] = df[cols].astype(float).div(100)

    return df


def fetch_all_tables():
    url_info = get_main_url_info()

    dfs = []
    with sync_playwright() as p:
        browser = p.firefox.launch(headless=True)
        for info in tqdm(url_info):
            dfi = fetch_url_tables(info['url'], info['category'], browser)
            dfs.append(dfi)
    
    df = pd.concat(dfs, axis=0)
    df['timestamp'] = pd.Timestamp.today().date()
    # import pdb; pdb.set_trace()

    df = df.rename(columns=lambda x: x.replace(' ', '').replace('"', ''))
    df = df.rename(columns={'index': 'index_name'})

    try:
        df = df.drop(columns=['index'])
    except:
        pass

    return df


# if __name__ == '__main__':
#     df = fetch_all_tables()
#     df.to_csv('~/Downloads/seekingalpha_etf_info.csv')
#     logger.info('saved: ~/Downloads/seekingalpha_etf_info.csv')
    # url_info = get_main_url_info()
    # # import pdb; pdb.set_trace()
    
    # dfs = []
    # with sync_playwright() as p:
    #     browser = p.firefox.launch(headless=False)
    #     for info in url_info[:2]:
    #         dfi = fetch_url_tables(info['url'], info['category'], browser)
    #         dfs.append(dfi)
    
    # df = pd.concat(dfs, axis=0)
    # df['timestamp'] = pd.Timestamp.today().date()

    # print(df)



