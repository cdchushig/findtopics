import pandas as pd
import logging
import coloredlogs
from pathlib import Path
import utils.consts as consts

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

