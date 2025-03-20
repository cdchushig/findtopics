import pandas as pd
from nltk.stem import WordNetLemmatizer
import csv
from find_topics_utils import load_merged_dataset, load_and_merge_csv_files
from find_topics_utils import PATH_KEYWORDS_FILE, PATH_FINAL_REPORTS_FILE

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Define exceptions for multi-word terms
exceptions = ["AK-47", "AR-15", "AK 47", "AR 15"]


def split_and_group_terms(terms):
    """Extract and lemmatize terms, handling exceptions"""
    grouped_terms = []
    for term in terms.dropna().astype(str):
        if term in exceptions:
            grouped_terms.append([term.lower()])  # Treat exceptions as single terms
        else:
            words = term.lower().split()
            grouped_terms.append([lemmatizer.lemmatize(word) for word in words])
    return grouped_terms


def lemmatize_text(text):
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])


def check_group_match(title, selftext, grouped_terms):
    """Function to check if all words in a group appear across title and selftext"""
    matches = []
    for group in grouped_terms:
        if all(word in title or word in selftext for word in group):
            matches.append(" ".join(group))  # Combine the group for better readability
    return matches


def preprocessing_key_terms(posts_df: pd.DataFrame, column_name_post: str = 'text'):
    terms_df = pd.read_excel(PATH_KEYWORDS_FILE, sheet_name="terms")
    fire_terms = split_and_group_terms(terms_df["firearm_terms"])
    suicide_terms = split_and_group_terms(terms_df["suicide_terms"])

    posts_df["title"] = ""

    # Preprocess text columns
    posts_df["title"] = posts_df["title"].fillna("").astype(str).str.lower()
    posts_df[column_name_post] = posts_df[column_name_post].fillna("").astype(str).str.lower()

    # Lemmatize the title and selftext fields
    posts_df["lemmatized_title"] = posts_df["title"].apply(lemmatize_text)
    posts_df["lemmatized_selftext"] = posts_df[column_name_post].apply(lemmatize_text)

    # Apply match-checking logic for lemmatized text
    posts_df["lemmatized_firearm_matches"] = posts_df.apply(
        lambda row: check_group_match(row["lemmatized_title"], row["lemmatized_selftext"], fire_terms),
        axis=1,
    )
    posts_df["lemmatized_suicide_matches"] = posts_df.apply(
        lambda row: check_group_match(row["lemmatized_title"], row["lemmatized_selftext"], suicide_terms),
        axis=1,
    )
    # Add TRUE/FALSE for whether both lemmatized terms were matched
    posts_df["lemmatized_both_matches"] = posts_df["lemmatized_firearm_matches"].apply(bool) & posts_df[
        "lemmatized_suicide_matches"].apply(bool)

    # Apply match-checking logic for regular text
    posts_df["regular_firearm_matches"] = posts_df.apply(
        lambda row: check_group_match(row["title"], row[column_name_post], fire_terms),
        axis=1,
    )
    posts_df["regular_suicide_matches"] = posts_df.apply(
        lambda row: check_group_match(row["title"], row[column_name_post], suicide_terms),
        axis=1,
    )

    # Add TRUE/FALSE for whether both regular terms were matched
    posts_df["regular_both_matches"] = posts_df["regular_firearm_matches"].apply(bool) & posts_df[
        "regular_suicide_matches"].apply(bool)

    # Add summary fields
    posts_df["lemmatized_firearm_match_summary"] = posts_df["lemmatized_firearm_matches"].apply(", ".join)
    posts_df["lemmatized_suicide_match_summary"] = posts_df["lemmatized_suicide_matches"].apply(", ".join)
    posts_df["regular_firearm_match_summary"] = posts_df["regular_firearm_matches"].apply(", ".join)
    posts_df["regular_suicide_match_summary"] = posts_df["regular_suicide_matches"].apply(", ".join)

    # Add TRUE/FALSE for individual matches
    posts_df["lemmatized_firearm_match"] = posts_df["lemmatized_firearm_matches"].apply(bool)
    posts_df["lemmatized_suicide_match"] = posts_df["lemmatized_suicide_matches"].apply(bool)
    posts_df["regular_firearm_match"] = posts_df["regular_firearm_matches"].apply(bool)
    posts_df["regular_suicide_match"] = posts_df["regular_suicide_matches"].apply(bool)
    return posts_df


# Loading raw data
load_and_merge_csv_files()
df = load_merged_dataset()

# Load keywords from Excel
terms_df = pd.read_excel(PATH_KEYWORDS_FILE, sheet_name="terms")
posts_df = preprocessing_key_terms(df)

# Save the updated dataset
posts_df.to_csv(PATH_FINAL_REPORTS_FILE, index=False, encoding='utf-8', quotechar='"', quoting=csv.QUOTE_ALL)
