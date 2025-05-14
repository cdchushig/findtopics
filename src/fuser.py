import pandas as pd
import argparse
from pathlib import Path
import utils.consts as consts
import logging
import coloredlogs
import csv
from io import StringIO

from utils.preprocessing import remove_wrong_lines

logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', logger=logger)


def read_fix_error_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read().replace('\\', '\\\\').replace('""', '\\"')
        reader = csv.reader(StringIO(content), doublequote=False, escapechar='\\')
        rows = [row for row in reader]
    df = pd.DataFrame(rows)
    return df


def read_custom_csv(file_path, id):

    with open(file_path, "r", encoding="utf-8") as f:
        file_text = f.read()

    csv_reader = csv.reader(StringIO(file_text), delimiter=",", quotechar='"', doublequote=True)
    rows = list(csv_reader)

    df = pd.DataFrame(rows[1:], columns=rows[0])

    numeric_columns = [
        "user_followers_count", "user_following_count", "user_listed_count",
        "user_tweets_count", "retweets", "favorites", "replies", "impressions",
        "user_following_count_numeric", "user_followers_count_numeric"
    ]

    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")  # convierte o pone NaN si hay error

    filename_df = str(
        Path.joinpath(consts.PATH_PROJECT_DATA_FIREARMS, 'df_partial_cleaned_{}.csv'.format(id))
    )

    df.to_csv(filename_df, index=False, encoding="utf-8")


def parse_arguments(parser):
    parser.add_argument('--src_lang', default='english', type=str)
    parser.add_argument('--target_lang', default='spanish', type=str)
    parser.add_argument('--mode', default='oneline', type=str)
    parser.add_argument('--dataset', default='prg', type=str)
    parser.add_argument('--n_partitions', default=12, type=int)
    parser.add_argument('--n_jobs', default=4, type=int)
    return parser.parse_args()


parser = argparse.ArgumentParser(description='translator')
args = parse_arguments(parser)


list_error_file = [str(Path.joinpath(
    consts.PATH_PROJECT_DATA_FIREARMS,
    'df_partial_errors_{}.csv'.format(i)))
    for i in range(1, 7)
]

# list_dfs_error = [pd.read_csv(path_file, sep=',', engine='python', skipinitialspace=True) for path_file in list_error_file]
# list_dfs_error = [read_fix_error_file(path_file) for path_file in list_error_file]
# list_dfs_error = [read_custom_csv(path_file, idx) for idx, path_file in enumerate(list_error_file)]

n_partitions = args.n_partitions
file_list = [str(Path.joinpath(
    consts.PATH_PROJECT_DATA_FIREARMS,
    'VET_CSV_Content_(1)-{}.csv'.format(i)))
    for i in range(1, n_partitions + 1)
]

list_dataframes = []
for idx, path_filename in enumerate(file_list):
    df_raw = pd.read_csv(path_filename, engine="python", quotechar='"',  doublequote=True, encoding='utf-8')
    df_raw.columns = df_raw.columns.str.replace(r"<.*?>", "", regex=True)
    df_pre = remove_wrong_lines(df_raw, idx)
    logger.info('Partial {}, raw samples {}, pre samples {}'.format(idx, df_raw.shape, df_pre.shape))
    list_dataframes.append(df_pre)

list_dataframe_post = []
for df in list_dataframes:
    df.loc[:, 'user_followers_count'] = pd.to_numeric(df['user_followers_count'], errors='coerce')
    df.loc[:, 'user_following_count'] = pd.to_numeric(df['user_following_count'], errors='coerce')
    df.loc[:, 'user_listed_count'] = pd.to_numeric(df['user_listed_count'], errors='coerce')
    df.loc[:, 'user_tweets_count'] = pd.to_numeric(df['user_tweets_count'], errors='coerce')
    df.loc[:, 'retweets'] = pd.to_numeric(df['retweets'], errors='coerce')
    df.loc[:, 'favorites'] = pd.to_numeric(df['favorites'], errors='coerce')
    df.loc[:, 'user_verified'] = df['user_verified'].astype(str)
    df.loc[:, 'replies'] = pd.to_numeric(df['replies'], errors='coerce')
    df.loc[:, 'impressions'] = pd.to_numeric(df['impressions'], errors='coerce')
    list_dataframe_post.append(df)

combined_df = pd.concat(list_dataframe_post, ignore_index=True)

filename_df = str(
    Path.joinpath(consts.PATH_PROJECT_DATA_FIREARMS, f'df_tweets_lms_raw.parquet')
)

combined_df.to_parquet(filename_df, engine='pyarrow')
