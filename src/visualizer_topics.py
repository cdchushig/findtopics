import os
import pandas as pd
import argparse
from utils.loader import load_dataset_by_name, make_topic_fusion, assign_topic_name_per_therapy, get_dataset_id
import utils.consts as consts
from utils.loader import load_list_datasets
from utils.plotter import plot_number_tweets_per_therapy, plot_number_tweets_stacked_per_topic, \
    plot_temporal_evolution_tweets, plot_radar_plot, plot_number_tweets_stacked_language, plot_radar_plot_all, plot_number_tweets_per_language

import logging
import coloredlogs

logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', logger=logger)


def split_save_data_by_topics(df: pd.DataFrame, therapy_name: str, language: str):
    list_topics = df['Topic'].unique()
    for specific_topic in list_topics:
        df_topic = df[df['Topic'] == specific_topic]
        # filename_data_topic = f"{therapy_name}_{language}_{specific_topic}.csv"
        filename_data_topic = 'therapy_{}_topic_{}_{}.csv'.format(therapy_name, specific_topic, language)
        df_topic.to_csv(
            str(os.path.join(consts.PATH_PROJECT_REPORTS_PSYCHOTHERAPY, 'csvs', filename_data_topic)),
            index=False
        )


# def create_years_loop(list_therapy_ids: list, language: str):
#     for dataset_name in list_therapy_ids:
#         dataset_id = get_dataset_id(dataset_name)
#         path_dataset = Path.joinpath(consts.PATH_PROJECT_DATA_PSYCHOTHERAPY, '{}_{}.csv'.format(dataset_id, language))
#         df = pd.read_csv(str(path_dataset))
#         df.drop_duplicates()
#         df = df.reset_index(drop=True)
#         pathx = str(os.path.join(consts.PATH_PROJECT_REPORTS_PSYCHOTHERAPY, 'csvs'))
#         create_years(df, pathx, language, '')


def split_data_per_topics(list_therapy_names: list, list_datasets: list, language: str, flag_save_data: bool = False):

    list_therapy_topics = []

    for idx, therapy_name in enumerate(list_therapy_names):
        df_therapy = list_datasets[idx][1].copy()
        df_therapy_new_topics = make_topic_fusion(df_therapy, therapy_name, language=language)
        df_topic_counts = df_therapy_new_topics['Topic'].value_counts().reset_index()
        df_topic_counts_sort = df_topic_counts.sort_values(by=['count'], ascending=False)
        df_topic_counts_sort = df_topic_counts_sort.iloc[:2, :]
        list_topics_new_filtered = df_topic_counts_sort['Topic'].unique()
        df_therapy_new_topics_filtered = df_therapy_new_topics[df_therapy_new_topics['Topic'].isin(list_topics_new_filtered)]

        if flag_save_data:
            split_save_data_by_topics(df_therapy_new_topics_filtered, therapy_name=therapy_name, language=language)

        for topic_id in list_topics_new_filtered:
            item_list = (therapy_name,
                         topic_id,
                         df_therapy_new_topics_filtered[df_therapy_new_topics_filtered['Topic'] == topic_id].shape[0],
                         language
                         )
            list_therapy_topics.append(item_list)

    return list_therapy_topics


def parse_arguments(parser):
    parser.add_argument('--language', default='english', type=str)
    parser.add_argument('--project', default='additions', choices=['psychotherapy', 'addictions'], type=str)
    parser.add_argument('--dataset', default='ldp', choices=['ldp', 'prg'], type=str)
    parser.add_argument('--type_data', default='na', choices=['otros', 'casas', 'na'], type=str)
    parser.add_argument('--plot_temporal', default=False, type=bool)
    parser.add_argument('--unique_language', default=False, type=bool)
    parser.add_argument('--plot_radar', default=False, type=bool)
    parser.add_argument('--plot_radar_all', default=False, type=bool)
    parser.add_argument('--plot_radar_language', default=False, type=bool)
    parser.add_argument('--plot_tweets_therapy', default=False, type=bool)
    parser.add_argument('--plot_tweets_language', default=False, type=bool)
    parser.add_argument('--show_legend', default=False, type=bool)
    parser.add_argument('--type_image', default='png', type=str)
    parser.add_argument('--save_data', default=False, type=bool)
    return parser.parse_args()


parser = argparse.ArgumentParser(description='topic modeling plotter')
args = parse_arguments(parser)

if args.project == 'psychotherapy':
    dict_therapy_fullnames = consts.DICT_THERAPY_FULL_NAME_EN
    list_therapy_names = consts.LIST_THERAPIES
else: # addictions
    dict_therapy_fullnames = None
    list_therapy_names = None

# if args.plot_temporal_topics:
#     plot_temporal_evolution_tweets(list_therapy_names, language=args.language, flag_save_figure=True, show_legend=True)

if args.plot_temporal:
    plot_temporal_evolution_tweets(list_therapy_names,
                                   language=args.language,
                                   project=args.project,
                                   hue_var='therapy_id',
                                   dataset_name=args.dataset,
                                   type_data=args.type_data,
                                   unique_language=args.unique_language,
                                   flag_save_figure=True,
                                   show_legend=True
                                   )

if args.plot_tweets_therapy:
    plot_number_tweets_per_therapy(list_therapy_names, language=args.language, flag_save_figure=True)

if args.plot_tweets_language:
    plot_number_tweets_per_language(args.dataset, language=args.language, type_data=args.type_data, flag_save_figure=True)

list_datasets_en = load_list_datasets(list_therapy_names, language='en')
list_datasets_es = load_list_datasets(list_therapy_names, language='es')
list_therapy_topics_es = split_data_per_topics(list_therapy_names, list_datasets_en, language='en', flag_save_data=args.save_data)
list_therapy_topics_en = split_data_per_topics(list_therapy_names, list_datasets_es, language='es', flag_save_data=args.save_data)
list_therapy_topics_all = list_therapy_topics_en + list_therapy_topics_es

list_languages = ['en', 'es']
list_data = []
for therapy_name, topic_id, _, specific_language in list_therapy_topics_all:
    filename_data_topic = 'therapy_{}_topic_{}_{}.csv'.format(therapy_name, topic_id, specific_language)
    path_filename = str(os.path.join(consts.PATH_PROJECT_REPORTS_PSYCHOTHERAPY, 'csvs', filename_data_topic))
    df_therapy_topic = pd.read_csv(path_filename)
    topic_description = assign_topic_name_per_therapy(therapy_name, topic_id, language=specific_language)
    list_all_emotions = ['joy', 'anger', 'surprise', 'neutral', 'sadness', 'fear', 'disgust']
    list_emotions = df_therapy_topic['emotion'].value_counts().index

    for specific_emotion in list_emotions:
        df_therapy_emotion = df_therapy_topic[df_therapy_topic['emotion'] == specific_emotion]
        number_tweets_therapy_emotion = df_therapy_emotion.shape[0]
        list_data.append((
            therapy_name, df_therapy_topic.shape[0], topic_id, topic_description,
            specific_language, specific_emotion, number_tweets_therapy_emotion
        ))

    list_missing_emotions = list(set(list_all_emotions) - set(list_emotions))
    for specific_emotion in list_missing_emotions:
        list_data.append((
            therapy_name, df_therapy_topic.shape[0], topic_id, topic_description,
            specific_language, specific_emotion, 0
        ))

df_data = pd.DataFrame(list_data)
df_data = df_data.rename(columns={
    0: 'therapy_id',
    1: 'number_tweets_topic',
    2: 'topic_id',
    3: 'topic_description',
    4: 'language',
    5: 'emotion',
    6: 'number_tweets_topic_emotion'
})

df_data = df_data.drop(df_data[df_data['emotion'] == 'neutral'].index)
df_data['therapy_fullname'] = df_data['therapy_id'].apply(lambda x: dict_therapy_fullnames[x])
df_data['therapy_unique'] = df_data[['therapy_id', 'language']].apply(lambda x: '{}_{}'.format(x[0], x[1]), axis=1)

plot_number_tweets_stacked_per_topic(df_data, language=args.language, flag_save_figure=True)
plot_number_tweets_stacked_language(df_data, flag_save_figure=True)

if args.plot_radar:
    plot_radar_plot(df_data, therapy_id='acceptance', id_labels='emotion', language=args.language, id_splits='topic_description', ax_external=None, flag_save_figure=True)
    plot_radar_plot(df_data, therapy_id='cognitive', id_labels='emotion', language=args.language, id_splits='topic_description', ax_external=None, flag_save_figure=True)
    # plot_radar_plot(df_data, therapy_id='family', id_labels='emotion', language=args.language, id_splits='topic_description', ax_external=None, flag_save_figure=True)
    plot_radar_plot(df_data, therapy_id='narrative', id_labels='emotion', language=args.language, id_splits='topic_description', ax_external=None, flag_save_figure=True)
    plot_radar_plot(df_data, therapy_id='psycho', id_labels='emotion', language=args.language, id_splits='topic_description', ax_external=None, flag_save_figure=True)

if args.plot_radar_all:
    plot_radar_plot_all(df_data, type_image=args.type_image, flag_save_figure=True)

if args.plot_radar_language:
    dfx = df_data.groupby(['language', 'emotion'], as_index=False).sum()
    plot_radar_plot(dfx,
                    therapy_id=None,
                    id_labels='emotion',
                    language=args.language,
                    id_splits='language',
                    ax_external=None,
                    flag_save_figure=True
                    )

