import time
import json
import requests as rq
import datetime as dt
from bs4 import BeautifulSoup as bs
from tqdm import tqdm
from selenium import webdriver
import pandas as pd
from pandas_datareader import data as web


def convert_isin_to_ticker(infile="data/raw_isin.txt"):
    """
    convert a list of ISINs to a dict of ISIN:ticker
    """
    with open(infile, 'r') as f:
        isin_list = [f.strip() for f in f.readlines()]

    isin_mapper = {}

    base_url = "https://finance.yahoo.com"

    user_agent = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.50 Safari/537.36"
    options = webdriver.ChromeOptions()
    options.add_argument("headless")
    options.add_argument(f"user-agent={user_agent}")
    driver = webdriver.Chrome(options=options)
    driver.get(base_url)
    time.sleep(3)
    print(f"Opened base search page. Beginning ticker searches..")

    for isin in tqdm(isin_list):
        try:
            input_element = driver.find_element_by_id('yfin-usr-qry')
            input_element.clear()
            input_element.send_keys(isin)

            time.sleep(2)
            html = driver.page_source
            if "<span>Symbols</span>" in html:
                soup = bs(driver.page_source)
                first_res = soup.find("div", {
                    "role": "link",
                    "data-test": "srch-sym"
                })
                if not first_res:
                    print(f"{isin} missing")
                    continue
                ticker = first_res.find(class_="C(black)").text
                isin_mapper[isin] = ticker
            else:
                print(f"{isin} missing")

        except Exception as e:
            print(f"Error for {isin_list.index(isin)}. {isin}: {e}")
            print("Restting search")
            driver.get(base_url)
            time.sleep(2)
            continue
    driver.close

    with open("data/isin_mapper.json", "w") as f:
        json.dump(isin_mapper, f)

    return isin_mapper
