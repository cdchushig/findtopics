import os
import pandas as pd
import logging
import coloredlogs
from pathlib import Path

import utils.consts as consts

logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', logger=logger)


def save_parquet_dataframe(df_tweets: pd.DataFrame, dataset: str, language_of_tweets: str):
    df_tweets['_id'] = df_tweets['_id'].astype(str)

    filename_tweets_language = str(
        Path.joinpath(consts.PATH_PROJECT_DATA_ADDICTIONS, 'tweets_{}_{}.parquet'.format(
            dataset, language_of_tweets
        ))
    )

    df_tweets.to_parquet(filename_tweets_language, engine='pyarrow')


def save_dataframe_tweets_by_language(df_tweets: pd.DataFrame, language_of_tweets: str):
    filename_tweets_language = str(
        Path.joinpath(consts.PATH_PROJECT_DATA_ADDICTIONS, 'tweets_{}_translated.parquet'.format(
            language_of_tweets
        ))
    )

    df_tweets.to_parquet(filename_tweets_language, engine='pyarrow')


def load_tweets_test(src_lang: str, target_lang: str, batch_id: str):
    filename_tweets_language = str(
        Path.joinpath(consts.PATH_PROJECT_DATA_ADDICTIONS, 'df_tweets_{}_{}_{}.csv'.format(src_lang, target_lang, batch_id))
    )

    df_tweets_language = pd.read_csv(filename_tweets_language)

    return df_tweets_language


def save_tweets_translated(df_tweets: pd.DataFrame, dataset: str, src_lang: str, target_lang: str, batch_id: int):

    csv_pathfile = str(
        Path.joinpath(consts.PATH_PROJECT_DATA_ADDICTIONS, 'df_tweets_{}_{}_{}_{}.csv'.format(dataset, src_lang, target_lang, batch_id))
    )

    # if os.path.exists(csv_pathfile):
    #     df_tweets = pd.read_csv(csv_pathfile)
    # else:
    #     df_tweets = pd.DataFrame()

    # for df_batch in list_total_df_batches:
    #     df_tweets = pd.concat([df_tweets, df_batch], ignore_index=True)

    df_tweets.to_csv(csv_pathfile, index=False)


def load_tweets_by_language(dataset: str, language_of_tweets: str):
    filename_tweets_language = str(
        Path.joinpath(consts.PATH_PROJECT_DATA_ADDICTIONS, 'tweets_{}_{}.parquet'.format(
            dataset, language_of_tweets
        ))
    )

    df_tweets_language = pd.read_parquet(filename_tweets_language)

    return df_tweets_language


def load_list_datasets(list_therapy_names: list, language):

    list_datasets = []
    for therapy_name in list_therapy_names:
        df_therapy = load_dataset_by_name(therapy_name, language)
        list_datasets.append((therapy_name, df_therapy))

    return list_datasets


def get_dataset_id(dataset_name: str):
    dict_map_datasets = consts.DICT_DATASETS_THERAPY
    return dict_map_datasets[dataset_name]


def load_dataset_by_name(dataset_name: str, language: str):
    dataset_id = get_dataset_id(dataset_name)
    path_dataset = Path.joinpath(consts.PATH_PROJECT_DATA_PSYCHOTHERAPY, '{}_{}.csv'.format(dataset_id, language))
    df_data = pd.read_csv(str(path_dataset))
    return df_data


def load_dataset_addictions_by_name(dataset_name: str, language: str, type_data: str = ''):

    if dataset_name == 'ldp':
        topics_filename = 'topics_{}_{}_{}.csv'.format(dataset_name, language, type_data)
    else:
        topics_filename = 'topics_{}_{}.csv'.format(dataset_name, language)

    path_dataset = Path.joinpath(
        consts.PATH_PROJECT_DATA_ADDICTIONS,
        dataset_name,
        'csv',
        language,
        topics_filename
    )
    logger.info('Loading dataset in {}'.format(path_dataset))
    if language == 'spanish':
        df_data = pd.read_csv(str(path_dataset), encoding="utf-8", sep=",", on_bad_lines="skip", low_memory=False)
    else:
        df_data = pd.read_csv(str(path_dataset))
    return df_data


def make_topic_fusion(df_original: pd.DataFrame, dataset_name: str, language: str):
    dataset_id = get_dataset_id(dataset_name)

    df = df_original.copy()

    if dataset_id == consts.DATASET_NAME_ACCEPTANCE_COMMITMENT_THERAPY and language == 'en':
        df.loc[df['Topic'] == 0, 'Topic'] = 1
        df.loc[df['Topic'] == 1, 'Topic'] = 1
        df.loc[df['Topic'] == -1, 'Topic'] = 0
    elif dataset_id == consts.DATASET_NAME_COGNITIVE_BEHAVIOURAL_THERAPY and language == 'en':
        df.loc[df['Topic'] == 1, 'Topic'] = 2
        df.loc[df['Topic'] == 0, 'Topic'] = 1
        df.loc[df['Topic'] == -1, 'Topic'] = 0
    elif dataset_id == consts.DATASET_NAME_FAMILY_COUPLE_THERAPY and language == 'en':
        df.loc[df['Topic'] == 1, 'Topic'] = 0
        df.loc[df['Topic'] == 2, 'Topic'] = 1
        df.loc[df['Topic'] == 3, 'Topic'] = 1
        df.loc[df['Topic'] == 4, 'Topic'] = 1
        df.drop(df[(df['Topic'] == -1)].index, inplace=True)
        df.drop(df[(df['Topic'] == 6)].index, inplace=True)
        df.drop(df[(df['Topic'] == 5)].index, inplace=True)
    elif dataset_id == consts.DATASET_NAME_NARRATIVE_THERAPY and language == 'en':
        df.loc[df['Topic'] == 1, 'Topic'] = 0
        df.loc[df['Topic'] == -1, 'Topic'] = 0
        df.loc[df['Topic'] == 2, 'Topic'] = 1
    elif dataset_id == consts.DATASET_NAME_PSYCHO_THERAPY and language == 'en':
        df.loc[df['Topic'] == 1, 'Topic'] = 2
        df.loc[df['Topic'] == -1, 'Topic'] = 1
    elif dataset_id == consts.DATASET_NAME_ACCEPTANCE_COMMITMENT_THERAPY and language == 'es':
        df.loc[df['Topic'] == -1, 'Topic'] = 0
        df.loc[df['Topic'] == 2, 'Topic'] = 1
    elif dataset_id == consts.DATASET_NAME_COGNITIVE_BEHAVIOURAL_THERAPY and language == 'es':
        df.loc[df['Topic'] == 1, 'Topic'] = 2
        df.loc[df['Topic'] == 0, 'Topic'] = 1
        df.loc[df['Topic'] == -1, 'Topic'] = 0
    elif dataset_id == consts.DATASET_NAME_FAMILY_COUPLE_THERAPY and language == 'es':
        df.loc[df['Topic'] == -1, 'Topic'] = 0
        df.loc[df['Topic'] == 5, 'Topic'] = 1
        df.loc[df['Topic'] == 3, 'Topic'] = 2
        df.loc[df['Topic'] == 4, 'Topic'] = 2
    elif dataset_id == consts.DATASET_NAME_NARRATIVE_THERAPY and language == 'es':
        df.loc[df['Topic'] == 2, 'Topic'] = 1
        df.loc[df['Topic'] == -1, 'Topic'] = 0
        df.loc[df['Topic'] == 3, 'Topic'] = 1
        df.loc[df['Topic'] == 4, 'Topic'] = 1
    elif dataset_id == consts.DATASET_NAME_PSYCHO_THERAPY and language == 'es':
        df.loc[df['Topic'] == -1, 'Topic'] = 0
        print('holaaa', df['Topic'].value_counts())
    else:
        print('not identified')

    return df


def assign_topic_name_per_therapy(dataset_name, topic_id, language):
    dict_topic_names = get_topic_name_per_therapy(dataset_name, language)
    return dict_topic_names[topic_id]


def get_topic_name_per_therapy(dataset_name, language):

    dataset_id = get_dataset_id(dataset_name)

    if dataset_id == consts.DATASET_NAME_COGNITIVE_BEHAVIOURAL_THERAPY and language == 'en':
        dict_topic_names = {
            # 0: 'Professionals offering courses and therapy', # do not remove
            0: 'Professionals offering therapy',
            1: 'Personal experiences',
            2: 'Clinical indications'
        }
    elif dataset_id == consts.DATASET_NAME_COGNITIVE_BEHAVIOURAL_THERAPY and language == 'es':
        dict_topic_names = {
            0: 'Clinical indications',
            1: 'Personal experiences',
            2: 'CBT for depression and OCD'
        }

    elif dataset_id == consts.DATASET_NAME_NARRATIVE_THERAPY and language == 'en':
        dict_topic_names = {
            # 0: 'Outreach on narrative therapy', # do not remove
            0: 'Outreach on therapy',
            1: 'Professionals offering therapy'
        }

    elif dataset_id == consts.DATASET_NAME_NARRATIVE_THERAPY and language == 'es':
        dict_topic_names = {
            # 0: 'Outreach on narrative therapy', # do not remove
            0: 'Outreach on therapy',
            1: 'Professionals offering therapy'
        }

    elif dataset_id == consts.DATASET_NAME_PSYCHO_THERAPY and language == 'es':
        dict_topic_names = {
            0: 'Unawareness of therapy',
            1: 'Professionals offering therapy',
            2: 'mal'
        }

    elif dataset_id == consts.DATASET_NAME_PSYCHO_THERAPY and language == 'en':
        dict_topic_names = {
            0: 'Personal experiences',
            1: 'Clinical indications',
            # 2: 'Outreach on psychodynamic therapy' # do not remove
            2: 'Outreach on therapy'
        }

    elif dataset_id == consts.DATASET_NAME_ACCEPTANCE_COMMITMENT_THERAPY and language == 'es':
        dict_topic_names = {
            0: 'Personal experiences',
            1: 'Clinical indications'
        }

    elif dataset_id == consts.DATASET_NAME_ACCEPTANCE_COMMITMENT_THERAPY and language == 'en':
        dict_topic_names = {
            # 1: 'Professionals offering courses and therapy', # do not remove
            0: 'Professionals offering therapy',
            1: 'Clinical indications'
        }

    elif dataset_id == consts.DATASET_NAME_FAMILY_COUPLE_THERAPY and language == 'es':
        dict_topic_names = {
            0: 'Users requesting couples or family therapy',
            1: 'Professionals offering therapy',
            2: 'Efficacy in sexual dysfunctions'
        }

    elif dataset_id == consts.DATASET_NAME_FAMILY_COUPLE_THERAPY and language == 'en':
        dict_topic_names = {
            # 0: 'Professionals offering family or couples therapy', # do not remove
            0: 'Professionals offering therapy',
            # 1: 'Outreach on family and couples therapy' # do not remove
            1: 'Outreach on therapy'
        }

    return dict_topic_names