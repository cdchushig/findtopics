import pandas as pd
from tqdm.auto import tqdm
import argparse
from pysentimiento import create_analyzer
from pysentimiento.preprocessing import preprocess_tweet
from pathlib import Path

import utils.consts as consts


def parse_arguments(parser):
    parser.add_argument('--dataset', default='ldp', type=str)
    parser.add_argument('--language', default='basque', type=str)
    parser.add_argument('--n_jobs', default=4, type=int)
    return parser.parse_args()


parser = argparse.ArgumentParser(description='topic modeler')
args = parse_arguments(parser)

filename_tweets_topics_language = str(
    Path.joinpath(
        consts.PATH_PROJECT_DATA_ADDICTIONS, args.dataset, 'topics',
        '{}_otros_topics.csv'.format(args.language))
)

df_topics = pd.read_csv(filename_tweets_topics_language)


emotion_analyzer = create_analyzer(task="emotion", lang="es")

oraciones = [
    "Qué gran jugador es Messi",
    "Esto es muy pésimo",
    "No sé, cómo se llama?",
]

x = emotion_analyzer.predict(oraciones)
print(x)

