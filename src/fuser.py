import pandas as pd
import argparse
from pathlib import Path
import utils.consts as consts
import logging
import coloredlogs
import csv
from io import StringIO

logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', logger=logger)


def remove_wrong_lines(df: pd.DataFrame, id: int):
    numeric_columns = ['user_following_count', 'user_followers_count']

    # Create a boolean mask for invalid rows
    invalid_rows = pd.Series(False, index=df.index)

    for col in numeric_columns:
        # Try to convert to numeric, NaN if it fails
        df[col + '_numeric'] = pd.to_numeric(df[col], errors='coerce')

        # Mark rows as invalid if conversion fails but original value is not empty
        invalid_rows |= df[col + '_numeric'].isna() & df[col].notna()

    # Extract the bad rows
    bad_rows = df[invalid_rows]
    logger.info(f"Found {len(bad_rows)} invalid rows with non-numeric values in numeric columns.")

    filename_df = str(
        Path.joinpath(consts.PATH_PROJECT_DATA_FIREARMS, 'df_partial_errors_{}.csv'.format(id))
    )

    bad_rows.to_csv(filename_df, index=False)

    df_clean = df[~invalid_rows]

    return df_clean


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
#     df['verified'] = df['verified'].astype(str)
    # df['totalRetweets'] = pd.to_numeric(df['totalRetweets'], errors='coerce')
#     df['followers'] = pd.to_numeric(df['followers'], errors='coerce')
#     df['following'] = pd.to_numeric(df['following'], errors='coerce')
#     df['lists'] = pd.to_numeric(df['lists'], errors='coerce')
#     df['statusesCount'] = pd.to_numeric(df['statusesCount'], errors='coerce')
#     df['impressions'] = pd.to_numeric(df['impressions'], errors='coerce')
#     df['isImage'] = df['isImage'].astype(str)
#     df['_id'] = df['_id'].astype(str)
#
    list_dataframe_post.append(df)

combined_df = pd.concat(list_dataframe_post, ignore_index=True)

filename_df = str(
    Path.joinpath(consts.PATH_PROJECT_DATA_FIREARMS, f'df_tweets_lms_raw.parquet')
)

print('xxxxxxxxxxxx', combined_df.shape)

combined_df.to_parquet(filename_df, engine='pyarrow')
