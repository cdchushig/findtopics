import pandas as pd
import argparse
from pathlib import Path

import utils.consts as consts


def parse_arguments(parser):
    parser.add_argument('--src_lang', default='catalan', type=str)
    parser.add_argument('--target_lang', default='spanish', type=str)
    parser.add_argument('--dataset', default='prg', type=str)
    parser.add_argument('--mode', default='oneline', type=str)
    parser.add_argument('--n_batches', default=10, type=int)
    parser.add_argument('--n_jobs', default=4, type=int)
    return parser.parse_args()


parser = argparse.ArgumentParser(description='translator')
args = parse_arguments(parser)

df = pd.read_csv(str(
    Path.joinpath(
        consts.PATH_PROJECT_DATA_ADDICTIONS, args.dataset, 'csv', args.src_lang,
        'df_tweets_{}_{}.csv'.format(args.dataset, args.src_lang))),
    delimiter=',', on_bad_lines='skip'
)

print(df.shape)

df['totalRetweets'] = pd.to_numeric(df['totalRetweets'], errors='coerce')
df['favorites'] = pd.to_numeric(df['favorites'], errors='coerce')
df['replies'] = pd.to_numeric(df['replies'], errors='coerce')
df['impressions'] = pd.to_numeric(df['impressions'], errors='coerce')
df['followers'] = pd.to_numeric(df['followers'], errors='coerce')
df['following'] = pd.to_numeric(df['following'], errors='coerce')
df['lists'] = pd.to_numeric(df['lists'], errors='coerce')
df['statusesCount'] = pd.to_numeric(df['statusesCount'], errors='coerce')
df['impressions'] = pd.to_numeric(df['impressions'], errors='coerce')
df['isImage'] = df['isImage'].astype(str)
df['_id'] = df['_id'].astype(str)
df['verified'] = df['verified'].astype(str)

filename_df = str(
    Path.joinpath(consts.PATH_PROJECT_DATA_ADDICTIONS, args.dataset, 'csv', args.src_lang,
                  f'df_tweets_{args.dataset}_{args.src_lang}.parquet')
)

# df.to_parquet(filename_df, engine='pyarrow')


# file_list = [
#     str(Path.joinpath(consts.PATH_PROJECT_DATA_ADDICTIONS, f'df_tweets_{args.src_lang}_{args.target_lang}_{i}.csv'))
#     for i in range(10)
# ]
#
# dataframes = [pd.read_csv(file, on_bad_lines='skip', engine='python') for file in file_list]
# xx = [print(df.shape) for df in dataframes]

# dataframes_post = []
#
# for df in dataframes:
#     df['totalRetweets'] = pd.to_numeric(df['totalRetweets'], errors='coerce')
#     df['favorites'] = pd.to_numeric(df['favorites'], errors='coerce')
#     df['replies'] = pd.to_numeric(df['replies'], errors='coerce')
#     df['impressions'] = pd.to_numeric(df['impressions'], errors='coerce')
#     df['followers'] = pd.to_numeric(df['followers'], errors='coerce')
#     df['following'] = pd.to_numeric(df['following'], errors='coerce')
#     df['lists'] = pd.to_numeric(df['lists'], errors='coerce')
#     df['statusesCount'] = pd.to_numeric(df['statusesCount'], errors='coerce')
#     df['impressions'] = pd.to_numeric(df['impressions'], errors='coerce')
#     df['isImage'] = df['isImage'].astype(str)
#     df['_id'] = df['_id'].astype(str)
#     df['verified'] = df['verified'].astype(str)
#
#     dataframes_post.append(df)


# df8 = pd.read_csv(str(Path.joinpath(consts.PATH_PROJECT_DATA_ADDICTIONS, f'df_tweets_{args.src_lang}_{args.target_lang}_{8}.csv')))
# df9 = pd.read_csv(str(Path.joinpath(consts.PATH_PROJECT_DATA_ADDICTIONS, f'df_tweets_{args.src_lang}_{args.target_lang}_{8}.csv')))
# dataframes.append(df8)
# dataframes.append(df9)

# combined_df = pd.concat(dataframes_post, ignore_index=True)
# print(combined_df.shape)
#
# filename_df = str(
#     Path.joinpath(consts.PATH_PROJECT_DATA_ADDICTIONS, 'prg', 'csv', args.src_lang,
#                   f'df_tweets_{args.dataset}_{args.src_lang}.parquet')
# )
#
# combined_df.to_parquet(filename_df, engine='pyarrow')
