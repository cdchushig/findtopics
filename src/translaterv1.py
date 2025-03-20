import os
import numpy as np
from pathlib import Path
import pandas as pd
import argparse
import string
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
torch.cuda.empty_cache()
import utils.consts as consts
from utils.loader import load_tweets_by_language, save_dataframe_tweets_by_language, load_tweets_test
import emoji
import re
import logging
import coloredlogs

logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', logger=logger)


def load_tokenizer_and_model_translator():
    tokenizer = AutoTokenizer.from_pretrained(consts.MODEL_TRANSLATOR_CATALAN_SPANISH)
    model = AutoModelForCausalLM.from_pretrained(consts.MODEL_TRANSLATOR_CATALAN_SPANISH, attn_implementation='eager')

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    return tokenizer, model, device


def preprocess_emoji(text_tweet: str):
    return emoji.demojize(text_tweet)


def parse_arguments(parser):
    parser.add_argument('--src_lang', default='basque', type=str)
    parser.add_argument('--target_lang', default='spanish', type=str)
    parser.add_argument('--batch_size', default=500, type=int)
    parser.add_argument('--batch_id', default=2, type=int)
    return parser.parse_args()


parser = argparse.ArgumentParser(description='image builder')
args = parse_arguments(parser)

batch_size = args.batch_size

tokenizer, model, device = load_tokenizer_and_model_translator()

# Load tweets for translation
df_tweets_original = load_tweets_by_language(args.dataset, args.src_lang)
df_tweets = load_tweets_test(args.src_lang, args.target_lang, args.batch_id)
# df_tweets['text_cleaned'] = df_tweets['text'].apply(preprocess_emoji)
df_tweets_filtered = df_tweets_original.copy()
batches = np.array_split(df_tweets_filtered, len(df_tweets_filtered) // batch_size + 1)

for i, df_batch in enumerate(batches):
    list_tweets = df_batch['text_cleaned'].to_list()
    prompt = lambda x: f'[{string.capwords(args.src_lang)}] {x} \n[{string.capwords(args.target_lang)}]'
    list_prompts = [prompt(x) for x in list_tweets]

    list_translated_tweets_batch = []

    for sentence_prompt in list_prompts:
        input_ids = tokenizer(sentence_prompt, return_tensors='pt').input_ids.to(device)
        output_ids = model.generate(input_ids, max_new_tokens=300, max_length=2000, num_beams=5)
        input_length = input_ids.shape[1]

        generated_text = tokenizer.decode(output_ids[0, input_length:], skip_special_tokens=True).strip()
        list_translated_tweets_batch.append(generated_text)

    print(len(list_translated_tweets_batch))
    df_batch['text_translated'] = list_translated_tweets_batch
    print(df_batch)
    list_df_batches = [df_batch]
    save_tweets_translated(list_df_batches, args.src_lang, args.target_lang)
    print(f"Batch {i+1} processed and saved.")



