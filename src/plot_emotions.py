import pandas as pd
import argparse
import utils.consts as cons
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import dask.dataframe as dd
from transformers import pipeline
emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)


def save_emotion_file(ddf, type_dataset):
    str_filename_emotions = '{}_{}.csv'.format('emotions', type_dataset)
    path_data_emotions = Path.joinpath(cons.PATH_PROJECT_REPORTS, 'topics', type_dataset, str_filename_emotions)
    df_result = ddf.compute()
    df_result.to_csv(path_data_emotions, index=False)


def plot_emotion_distribution(df,
                              type_dataset,
                              emotion_column="Emotion",
                              title="Emotion Distribution",
                              figsize=(10, 6),
                              flag_save_figure=True
                              ):

    plt.figure(figsize=figsize)
    sns.countplot(data=df, x=emotion_column, order=df[emotion_column].value_counts().index, palette="pastel")
    plt.title(title)
    plt.xlabel("Emotion")
    plt.ylabel("Number of Tweets")
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    if flag_save_figure:
        path_figure = str(Path.joinpath(cons.PATH_PROJECT_REPORTS_FIGURES, 'emotion_distribution_{}'.format(type_dataset)))
        plt.savefig(path_figure)
    else:
        plt.show()

def load_emotion_files(type_dataset):
    path_data_emotions = Path.joinpath(cons.PATH_PROJECT_REPORTS, 'topics', type_dataset,
                                       '{}_{}.csv'.format('emotions', type_dataset))
    df = pd.read_csv(path_data_emotions)
    return df

def load_topics_files(type_dataset):

    dict_map_descriptions = {
        'firearms': cons.FILENAME_DATA_TOPICS_FIREARMS,
        'suicide': cons.FILENAME_DATA_TOPICS_SUICIDE,
        'suicide_firearms': cons.FILENAME_DATA_TOPICS_SUICIDE_FIREARMS
    }

    topics_descriptions = 'LABELS_{}_words.xlsx'.format(dict_map_descriptions[type_dataset])
    topics_tweets = '{}_topics.csv'.format(dict_map_descriptions[type_dataset])

    path_data_topic_descriptions = Path.joinpath(cons.PATH_PROJECT_REPORTS, 'topics', type_dataset, topics_descriptions)
    path_data_topic_tweets = Path.joinpath(cons.PATH_PROJECT_REPORTS, 'topics', type_dataset, topics_tweets)

    df_topic_descriptions = pd.read_excel(path_data_topic_descriptions, sheet_name=0)
    df_topic_tweets = pd.read_csv(path_data_topic_tweets)

    return df_topic_descriptions, df_topic_tweets

def get_id_topics_for_type_dataset(type_dataset):

    if type_dataset == 'firearms':
        return [0, 1, 3, 4, 5]
    elif type_dataset == 'suicide':
        return [0, 1, 2, 3, 4, 5]
    else: # firearms and suicide
        return [0, 1, 2, 3, 4]


def get_topic_descriptions(df_topic_descriptions, list_id_topics):
    df_filtered = df_topic_descriptions[df_topic_descriptions["Topic"].isin(list_id_topics)][["Topic", "Label"]]
    return df_filtered


def plot_topics_temporal_evolution(df_tweets,
                                   type_dataset,
                                   ids_topics,
                                   df_match_descriptions,
                                   date_column="date",
                                   topic_column="Topic",
                                   resolution='month',
                                   flag_save_figure=True
                                   ):

    df = df_tweets[df_tweets["Topic"].isin(ids_topics)]
    df[date_column] = pd.to_datetime(df[date_column], errors='coerce')

    if resolution == "month":
        df["time"] = df[date_column].dt.to_period("M").dt.to_timestamp()
        title = "Monthly Tweet Count by Topic"
    elif resolution == "year":
        df["time"] = df[date_column].dt.to_period("Y").dt.to_timestamp()
        title = "Yearly Tweet Count by Topic"
    else:
        raise ValueError("Resolution must be either 'month' or 'year'")

    grouped = df.groupby(["time", topic_column]).size().reset_index(name="tweet_count")
    pivot_df = grouped.pivot(index="time", columns=topic_column, values="tweet_count").fillna(0)

    topic_description_map = df_match_descriptions.set_index(topic_column)["Label"].to_dict()
    print(topic_description_map)
    pivot_df.rename(columns=topic_description_map, inplace=True)

    pivot_df.plot(kind="line", marker="o", figsize=(14, 6))
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Number of Tweets")
    plt.grid(True)
    plt.legend(title="Topic", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    if flag_save_figure:
        path_figure = str(Path.joinpath(cons.PATH_PROJECT_REPORTS_FIGURES, 'topics_temporal_{}'.format(type_dataset)))
        plt.savefig(path_figure)
        plt.close()
    else:
        plt.show()


def plot_emotions_temporal_evolution(df,
                                     type_dataset,
                                     date_column="date",
                                     emotion_column="Emotion",
                                     resolution='month',
                                     flag_save_figure=True
                                     ):

    df[date_column] = pd.to_datetime(df[date_column], errors='coerce')

    if resolution == "month":
        df["time"] = df[date_column].dt.to_period("M").dt.to_timestamp()
        title = "Monthly Tweet Count by Topic"
    elif resolution == "year":
        df["time"] = df[date_column].dt.to_period("Y").dt.to_timestamp()
        title = "Yearly Tweet Count by Topic"
    else:
        raise ValueError("Resolution must be either 'month' or 'year'")

    grouped = df.groupby(["time", emotion_column]).size().reset_index(name="tweet_count")
    pivot_df = grouped.pivot(index="time", columns=emotion_column, values="tweet_count").fillna(0)

    pivot_df.plot(kind="line", marker="o", figsize=(14, 6))
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Number of Tweets")
    plt.grid(True)
    plt.legend(title="Topic", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    if flag_save_figure:
        path_figure = str(Path.joinpath(cons.PATH_PROJECT_REPORTS_FIGURES, 'emotions_temporal_{}'.format(type_dataset)))
        plt.savefig(path_figure)
        plt.close()
    else:
        plt.show()


def get_top_emotion(text):
    try:
        results = emotion_classifier(text)
        emotions = results[0]
        top_emotion = max(emotions, key=lambda x: x['score'])
        return top_emotion['label']
    except Exception as e:
        return "Error"


def generate_emotions(df_tweets, ids_topics, n_jobs):
    df_topic_tweets = df_tweets[df_tweets["Topic"].isin(ids_topics)]

    ddf = dd.from_pandas(df_topic_tweets, npartitions=n_jobs)
    ddf["Emotion"] = ddf["text"].map_partitions(
        lambda partition: partition.apply(get_top_emotion), meta=('Emotion', 'object')
    )
    save_emotion_file(ddf, args.type_dataset)


def parse_arguments(parser):
    parser.add_argument('--language', default='english', type=str)
    parser.add_argument('--stop_words', default='en', type=str)
    parser.add_argument('--n_jobs', default=4, type=int)
    parser.add_argument('--min_samples', default=1000, type=int)
    parser.add_argument('--min_cluster_size', default=10000, type=int)
    parser.add_argument('--cluster_selection_epsilon', default=0.5, type=float)
    parser.add_argument('--load_preprocessed_dataset', default=True, type=bool)
    parser.add_argument('--type_dataset', default='suicide_firearms', type=str)
    parser.add_argument('--resolution', default='year', type=str)
    parser.add_argument('--keyword_list', default='twitter', type=str)
    return parser.parse_args()


parser = argparse.ArgumentParser(description='emotion modeler')
args = parse_arguments(parser)

df_topic_descriptions, df_topic_tweets = load_topics_files(args.type_dataset)
list_id_topics_per_dataset = get_id_topics_for_type_dataset(args.type_dataset)
df_match_descriptions = get_topic_descriptions(df_topic_descriptions, list_id_topics_per_dataset)
print(df_match_descriptions)
plot_topics_temporal_evolution(df_topic_tweets, args.type_dataset, list_id_topics_per_dataset, df_match_descriptions, resolution=args.resolution)

# Generate emotions
generate_emotions(df_topic_tweets, list_id_topics_per_dataset, args.n_jobs)
df_emotions = load_emotion_files(args.type_dataset)
path_emotions = str(Path.joinpath(cons.PATH_PROJECT_REPORTS, 'emotions', 'df_emotions_{}.csv'.format(args.type_dataset)))
df_emotions.to_csv(path_emotions, index=False)

plot_emotion_distribution(df_emotions, args.type_dataset, emotion_column='Emotion')
plot_emotions_temporal_evolution(df_emotions, args.type_dataset, resolution=args.resolution)