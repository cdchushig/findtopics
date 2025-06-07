from pathlib import Path
import nltk
import argparse
import pandas as pd
from bertopic.representation import KeyBERTInspired
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic import BERTopic
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import csv
import warnings
import logging
import coloredlogs
import csv

from plotter import text_eda_summary, plot_tweets_per_year

from find_topics_utils import load_merged_dataset, load_dataset_with_meta_keywords

warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', logger=logger)

EMBEDDING_MODEL_SENTENCE = 'paraphrase-multilingual-MiniLM-L12-v2'
PATH_PROJECT_DIR = Path(__file__).resolve().parents[1]
PATH_PROJECT_TOPICS = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'topics')
PATH_PROJECT_MODELS = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'models')
PATH_POST_SUICIDE_FIREARM_FILTERED = Path.joinpath(PATH_PROJECT_DIR, 'data', 'firearms', 'post_suicide_firearm_filtered.csv')
PATH_POST_SUICIDE_FILTERED = Path.joinpath(PATH_PROJECT_DIR, 'data', 'firearms', 'posts_suicide_filtered.csv')
PATH_POST_FIREARM_FILTERED = Path.joinpath(PATH_PROJECT_DIR, 'data', 'firearms', 'post_firearm_filtered.csv')


def has_match(df, cols):
    return df[cols].fillna('').apply(lambda row: any(len(str(cell)) > 0 for cell in row), axis=1)


def train_bertopic(documents: pd.DataFrame,
                   min_cluster_size: int,
                   min_samples: int,
                   cluster_selection_method: str,
                   cluster_selection_epsilon: float,
                   language: str,
                   dataset: str,
                   type_data: str
                   ):

    print('Training with ', documents.shape)
    documents['text'] = documents['text'].astype(str)

    summary = text_eda_summary(documents, 'text')

    embedding_model = SentenceTransformer(EMBEDDING_MODEL_SENTENCE)
    representation_model = KeyBERTInspired(top_n_words=20, nr_repr_docs=10, random_state=0)
    ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)
    # Stopwords for english
    stopword_en = nltk.corpus.stopwords.words('english')
    vectorizer_model = CountVectorizer(stop_words=stopword_en)
    # Train hdbscan
    hdbscan_model = HDBSCAN(min_cluster_size=min_cluster_size,
                            metric='euclidean',
                            min_samples=min_samples,
                            cluster_selection_method=cluster_selection_method,
                            cluster_selection_epsilon=cluster_selection_epsilon
                            )
    topic_model = BERTopic(representation_model=representation_model,
                           verbose=True,
                           embedding_model=embedding_model,
                           ctfidf_model=ctfidf_model,
                           hdbscan_model=hdbscan_model,
                           vectorizer_model=vectorizer_model,
                           )

    filename_model = str(
        Path.joinpath(
            PATH_PROJECT_MODELS,
            'model_bertopic_{}_{}_{}_{}_{}'.format(
                dataset,
                language,
                min_cluster_size,
                min_samples,
                cluster_selection_epsilon
            ))
    )

    topic_model.save(filename_model, serialization="pickle")

    topics, probs = topic_model.fit_transform(documents['text'])
    df = topic_model.get_topic_info()

    filename_topics = str(
        Path.joinpath(
            PATH_PROJECT_TOPICS,
            type_data,
            'topics_{}_{}_{}_{}_{}_{}'.format(
                dataset,
                type_data,
                language,
                min_cluster_size,
                min_samples,
                cluster_selection_epsilon
            ))
    )

    df.to_excel(f'{filename_topics}_words.xlsx')
    documents['Topic'] = topics
    documents.to_csv(f'{filename_topics}_topics.csv')
    return df, documents


def filter_rows_by_colname():
    df_post_filtered = df_keywords_report[
        df_keywords_report["lemmatized_firearm_match"] & df_keywords_report["lemmatized_suicide_match"]
    ]

    df_firearm = df_keywords_report[df_keywords_report["lemmatized_firearm_match"]]
    df_suicide = df_keywords_report[df_keywords_report["lemmatized_suicide_match"]]


def select_lms_data_for_training(type_data, df_post_filtered, df_firearm, df_suicide, df_keywords_report):
    if type_data == 'suicide':
        return df_suicide
    elif type_data == 'firearms':
        return df_firearm
    elif type_data == 'suicide_firearms':
        return df_post_filtered
    else: # all
        return df_keywords_report


def parse_arguments(parser):
    parser.add_argument('--language', default='english', type=str)
    parser.add_argument('--stop_words', default='en', type=str)
    parser.add_argument('--dataset', default='firearms', type=str)
    parser.add_argument('--n_jobs', default=4, type=int)
    parser.add_argument('--min_samples', default=1000, type=int)
    parser.add_argument('--min_cluster_size', default=10000, type=int)
    parser.add_argument('--cluster_selection_epsilon', default=0.5, type=float)
    parser.add_argument('--load_preprocessed_dataset', default=True, type=bool)
    parser.add_argument('--type_data', default='suicide_firearm', type=str)
    parser.add_argument('--keyword_list', default='twitter', type=str)
    return parser.parse_args()


parser = argparse.ArgumentParser(description='topic modeler')
args = parse_arguments(parser)

try:
    stopwords.words('english')
except LookupError:
    print("Downloading stopwords")
    nltk.download('stopwords')

if args.load_preprocessed_dataset:
    df_keywords_report = load_dataset_with_meta_keywords(args.keyword_list)

    firearm_cols = [
        "regular_firearm_match_summary",
        "lemmatized_firearm_match_summary",
        "stemmed_firearm_match_summary"
    ]
    suicide_cols = [
        "regular_suicide_match_summary",
        "lemmatized_suicide_match_summary",
        "stemmed_suicide_match_summary"
    ]

    # Identify which posts match firearm and/or suicide keywords
    firearm_match = has_match(df_keywords_report, firearm_cols)
    suicide_match = has_match(df_keywords_report, suicide_cols)

    df_firearm = df_keywords_report[firearm_match]
    df_suicide = df_keywords_report[suicide_match]
    df_post_filtered = df_keywords_report[firearm_match & suicide_match]

    df_for_training = select_lms_data_for_training(args.type_data, df_post_filtered, df_firearm, df_suicide,
                                                   df_keywords_report)
    filtered_ids = df_for_training["id"]

    # Filter posts
    df_raw = load_merged_dataset(args.keyword_list)
    df = df_raw[df_raw["id"].isin(filtered_ids)]
    print('Raw data, number of posts: ', df_raw.shape)
    print('Filtered data: number of posts', df_post_filtered.shape)
    print('Firearm data: number of posts', df_firearm.shape)
    print('Suicide data: number of posts', df_suicide.shape)

    df_post_filtered.to_csv(PATH_POST_SUICIDE_FIREARM_FILTERED,
                            index=False,
                            encoding='utf-8',
                            quotechar='"',
                            quoting=csv.QUOTE_ALL)
    df_firearm.to_csv(PATH_POST_FIREARM_FILTERED, index=False, encoding='utf-8', quotechar='"', quoting=csv.QUOTE_ALL)
    df_suicide.to_csv(PATH_POST_SUICIDE_FILTERED, index=False, encoding='utf-8', quotechar='"', quoting=csv.QUOTE_ALL)

else:
    df = load_merged_dataset(args.keyword_list)

min_cluster_size = args.min_cluster_size
min_samples = args.min_samples
cluster_selection_method = 'leaf'
cluster_selection_epsilon = args.cluster_selection_epsilon
topics_info, topic = train_bertopic(df,
                                    min_cluster_size,
                                    min_samples,
                                    cluster_selection_method,
                                    cluster_selection_epsilon,
                                    args.language,
                                    args.dataset,
                                    args.type_data
                                    )
