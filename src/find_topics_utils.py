import pandas as pd
from pathlib import Path

PATH_PROJECT_DIR = Path(__file__).resolve().parents[1]
PATH_KEYWORDS_FILE = Path.joinpath(PATH_PROJECT_DIR, 'data', 'firearms', 'final_keywords_v2_final_no_mm.xlsx')
PATH_FAKE_FINAL_POSTS = Path.joinpath(PATH_PROJECT_DIR, 'data', 'firearms', 'fake_final_posts.csv')
PATH_PROJECT_DATA_FIREARMS = Path.joinpath(PATH_PROJECT_DIR, 'data', 'firearms')
PATH_FINAL_REPORTS_FILE = Path.joinpath(PATH_PROJECT_DIR, 'data', 'firearms', 'final_posts_with_organized_keywords.csv')


def load_and_merge_csv_files():
    file_list = sorted(PATH_PROJECT_DATA_FIREARMS.glob("VET CSV Content (1)-*.csv"))

    if not file_list:
        raise FileNotFoundError(f"Not found CSV files in {PATH_PROJECT_DATA_FIREARMS}")

    df_list = [pd.read_csv(file, encoding='utf-8') for file in file_list]
    df_list_processed = []

    for idx, df_partial_data in enumerate(df_list):
        print(idx, df_partial_data.shape)
        df_partial_data.columns = df_partial_data.columns.str.replace(r'<.*?>', '', regex=True)
        df_partial_data['user_followers_count'] = pd.to_numeric(df_partial_data['user_followers_count'], errors='coerce')
        df_partial_data['user_following_count'] = pd.to_numeric(df_partial_data['user_following_count'], errors='coerce')
        df_partial_data['user_listed_count'] = pd.to_numeric(df_partial_data['user_listed_count'], errors='coerce')
        df_partial_data['user_tweets_count'] = pd.to_numeric(df_partial_data['user_tweets_count'], errors='coerce')
        df_partial_data = df_partial_data.drop(['user_verified', 'retweets', 'favorites', 'replies', 'impressions'], axis=1)
        df_list_processed.append(df_partial_data)

    df_merged = pd.concat(df_list_processed, ignore_index=True)
    print(df_merged.head())

    output_file = PATH_PROJECT_DATA_FIREARMS / "merged_vet_data.parquet"
    df_merged.to_parquet(output_file, index=False, engine='pyarrow')


def load_merged_dataset():
    parquet_file = PATH_PROJECT_DATA_FIREARMS / "merged_vet_data.parquet"
    df_data = pd.read_parquet(parquet_file, engine='pyarrow')
    return df_data

