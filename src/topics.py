import pandas as pd
import nltk
from pathlib import Path
import argparse
from bertopic.representation import KeyBERTInspired
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic import BERTopic
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import utils.consts as consts

EMBEDDING_MODEL_SENTENCE = 'paraphrase-multilingual-MiniLM-L12-v2'


def train_bertopic(documents,
                   min_cluster_size,
                   min_samples,
                   cluster_selection_method,
                   cluster_selection_epsilon,
                   language,
                   language_stop_words,
                   dataset,
                   name
                   ):

    documents['text'] = documents['text'].astype(str)

    embedding_model = SentenceTransformer(EMBEDDING_MODEL_SENTENCE)
    representation_model = KeyBERTInspired(top_n_words=20, nr_repr_docs=10, random_state=0)
    ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)
    # vectorizer_model = dict_stopwords[language]
    if language_stop_words == 'es':
        stopword_es = nltk.corpus.stopwords.words('spanish')
        vectorizer_model = CountVectorizer(stop_words=stopword_es)
    elif language_stop_words == 'en':
        stopword_en = nltk.corpus.stopwords.words('english')
        vectorizer_model = CountVectorizer(stop_words=stopword_en)
    elif language_stop_words == 'ca':
        stopword_ca = nltk.corpus.stopwords.words('catalan')
        vectorizer_model = CountVectorizer(stop_words=stopword_ca)
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
            consts.PATH_PROJECT_MODELS,
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
            consts.PATH_PROJECT_TOPICS,
            'topics_{}_{}_{}_{}_{}'.format(
                dataset,
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


def parse_arguments(parser):
    parser.add_argument('--language', default='basque', type=str)
    parser.add_argument('--stop_words', default='es', type=str)
    parser.add_argument('--dataset', default='prg', type=str)
    parser.add_argument('--n_jobs', default=4, type=int)
    parser.add_argument('--min_samples', default=1000, type=int)
    parser.add_argument('--min_cluster_size', default=10000, type=int)
    parser.add_argument('--cluster_selection_epsilon', default=0.5, type=float)
    return parser.parse_args()


parser = argparse.ArgumentParser(description='topic modeler')
args = parse_arguments(parser)

try:
    stopwords.words('english')
except LookupError:
    print("Downloading stopwords")
    nltk.download('stopwords')

filename_tweets_language = str(
    Path.joinpath(
        consts.PATH_PROJECT_DATA_ADDICTIONS, args.dataset, 'csv', args.language,
        'df_tweets_{}_{}.parquet'.format(args.dataset, args.language))
)

print(filename_tweets_language)
df = pd.read_parquet(filename_tweets_language)
print(df.shape)
min_cluster_size = args.min_cluster_size  # 100 y 2000 (entre 4 / 5 / 6 topics en este caso de 290.000 tweets)
min_samples = args.min_samples # 100 1000
cluster_selection_method = 'leaf' # eon
cluster_selection_epsilon = args.cluster_selection_epsilon  # 0.01, 1.5
topics_info, topic = train_bertopic(df,
                                    min_cluster_size,
                                    min_samples,
                                    cluster_selection_method,
                                    cluster_selection_epsilon,
                                    args.language,
                                    args.stop_words,
                                    args.dataset,
                                    filename_tweets_language
                                    )
