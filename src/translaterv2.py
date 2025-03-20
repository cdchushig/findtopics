import pandas as pd
import numpy as np
import json
from concurrent.futures import ThreadPoolExecutor
import argparse
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from utils.loader import load_tweets_by_language
import requests
import io
from utils.loader import save_tweets_translated
import utils.consts as consts
import logging
import coloredlogs

logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', logger=logger)


def save_file_with_tweets(file_content):
    file_bytes = io.BytesIO(file_content.encode("utf-8"))
    output_file_path = "output_file.txt"
    with open(output_file_path, "wb") as file:
        file.write(file_bytes.getvalue())


def translate_libretranslate_file_request(file_content, source_lang, target_lang):

    url = "http://localhost:5000/translate_file"
    files = {
        "file": ("file.txt", io.BytesIO(file_content.encode("utf-8")), "text/plain")
    }
    data = {
        "source": source_lang,
        "target": target_lang,
        "api_key": ""
    }

    try:
        response = requests.post(url, files=files, data=data)
        if response.status_code == 200:
            translated_file_url = response.json().get("translatedFileUrl")
            if translated_file_url:
                translated_response = requests.get(translated_file_url)
                if translated_response.status_code == 200:
                    return translated_response.text.splitlines()
                else:
                    return [f"Error downloading file: {translated_response.status_code}"]
            else:
                return ["Error: URL of translated file not found."]
        else:
            return [f"HTTP {response.status_code}: {response.text}"]
    except Exception as e:
        return {"error": str(e)}


def translate_libretranslate(text, source_lang="ca", target_lang="es"):
    if isinstance(text, str) and text.strip():
        try:
            response = requests.post(
                "http://localhost:5000/translate",
                data=json.dumps({
                    "q": text,
                    "source": source_lang,
                    "target": target_lang,
                    "format": "text",
                    "api_key": ""
                }),
                headers={"Content-Type": "application/json"}
            )
            if response.status_code == 200:
                print("Texto original: ", text)
                print("Respuesta de la API:", response.json())
                return response.json().get("translatedText", text)
            else:
                print(f"Error {response.status_code}: {response.text}")
                return text
        except Exception as e:
            return print('fallo: ', text)
    return text


def translate_texts_parallel(df, text_column, source_lang="ca", target_lang="es", max_workers=10):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        translations = list(
            executor.map(lambda x: translate_libretranslate(x, source_lang, target_lang),
                         df[text_column].dropna())
        )
    return pd.Series(translations, index=df[text_column].dropna().index)


def translate_google_selenium(text):
    if isinstance(text, str) and text.strip():
        try:
            chrome_options = Options()
            chrome_options.add_argument("--headless")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--no-sandbox")
            service = Service(executable_path='PATH_TO_CHROMEDRIVER')
            driver = webdriver.Chrome(service=service, options=chrome_options)

            driver.get("https://translate.google.com/?sl=ca&tl=es")

            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.XPATH, "//textarea[@aria-label='Source text']"))
            )

            input_box = driver.find_element(By.XPATH, "//textarea[@aria-label='Source text']")
            input_box.clear()
            input_box.send_keys(text)

            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.XPATH, "//span[@class='VIiyi']/span"))
            )

            output_box = driver.find_element(By.XPATH, "//span[@class='VIiyi']/span")
            translation = output_box.text

            driver.quit()
            return translation
        except Exception as e:
            return f"Error: {e}"
    return text


def parse_arguments(parser):
    parser.add_argument('--src_lang', default='basque', type=str)
    parser.add_argument('--dataset', default='prg', type=str)
    parser.add_argument('--target_lang', default='spanish', type=str)
    parser.add_argument('--mode', default='oneline', type=str)
    parser.add_argument('--n_batches', default=10, type=int)
    parser.add_argument('--n_jobs', default=4, type=int)
    return parser.parse_args()


parser = argparse.ArgumentParser(description='translator')
args = parse_arguments(parser)

df = load_tweets_by_language(args.dataset, args.src_lang)
logger.info('src-lang: {}, target-lang: {}, dataset: {}'.format(args.src_lang, args.target_lang, args.dataset))
logger.info('# num text to translate {}'.format(df.shape))
dfs = np.array_split(df, args.n_batches)

for i, df_batch in enumerate(dfs):
    logger.info('Processing df-sample {}'.format(i))
    df_sample = df_batch.copy()
    df_sample["text_cleaned"] = df_sample["text"]

    if args.mode == 'batch':
        file_content = "\n".join(df["text"].dropna().astype(str))
        save_file_with_tweets(file_content)
        translated_lines = translate_libretranslate_file_request(file_content, 'english', 'english')
        df_sample["text_translated"] = translated_lines
    else:
        # df_sample["text_translated"] = translate_texts_parallel(df_sample, "text", args.src_lang, args.target_lang,
        #                                                         max_workers=args.n_jobs)
        df_sample["text_translated"] = df_sample["text"].apply(
            lambda x: translate_libretranslate(
                x,
                source_lang=consts.DICT_TRANSLATION_LANGUAGES[args.src_lang],
                target_lang=consts.DICT_TRANSLATION_LANGUAGES[args.target_lang]
            )
        )

    save_tweets_translated(df_sample, args.dataset, args.src_lang, args.target_lang, i)
    print(f"Batch {i+1} processed and saved.")




