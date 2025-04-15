import pandas as pd
from nltk.stem import WordNetLemmatizer, PorterStemmer
import time
from datetime import timedelta
import argparse
import re
import csv
import unicodedata
from tqdm import tqdm
from find_topics_utils import load_merged_dataset, load_and_merge_csv_files
from find_topics_utils import PATH_KEYWORDS_FILE, PATH_FINAL_REPORTS_FILE, PATH_FINAL_FILTERED_OUTPUT_FILE
tqdm.pandas()  # Enables progress bar with .progress_apply()

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

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


def extract_terms(terms):
    return [term.lower() for term in terms.dropna().astype(str)]


def stem_text(text):
    return " ".join([
        word if stemmer.stem(word) == "of" else stemmer.stem(word)
        for word in text.split()
    ])


def lemmatize_terms(term_list):
    """
    Function to lemmatize phrases
    """
    lemmatized_list = []
    for term in term_list:
        lemmatized_term = " ".join([lemmatizer.lemmatize(word) for word in term.split()])
        lemmatized_list.append(lemmatized_term)
    return lemmatized_list


def stem_terms(term_list):
    """
    Function to stem phrases
    """
    stemmed_list = []
    for term in term_list:
        stemmed_phrase = " ".join([
            word if stemmer.stem(word) == "of" else stemmer.stem(word)
            for word in term.split()
        ])
        stemmed_list.append(stemmed_phrase)
    return stemmed_list


def check_exact_phrase_match(title, selftext, term_list):
    """
    Function to check for exact phrase match
    """
    matches = []
    for term in term_list:
        regex_pattern = r'(^|\s|[^\w])' + re.escape(term) + r'($|\s|[^\w])'
        if re.search(regex_pattern, title) or re.search(regex_pattern, selftext):
            matches.append(term)
    return matches


def clean_text(text):
    """
    Clean text by normalizing and removing accents
    """
    if isinstance(text, str):
        text = unicodedata.normalize("NFKD", text)
        text = ''.join([c for c in text if not unicodedata.combining(c)])
        return text.strip()
    return text


def preprocessing_key_terms_v2(posts_df: pd.DataFrame, terms_df: pd.DataFrame, column_name_post: str = 'text'):
    fire_terms = extract_terms(terms_df["firearm_terms"])
    suicide_terms = extract_terms(terms_df["suicide_terms"])
    lemmatized_fire_terms = lemmatize_terms(fire_terms)
    lemmatized_suicide_terms = lemmatize_terms(suicide_terms)
    stemmed_fire_terms = stem_terms(fire_terms)
    stemmed_suicide_terms = stem_terms(suicide_terms)

    posts_df.columns = posts_df.columns.str.lower()
    posts_df.rename(columns={column_name_post: 'selftext'}, inplace=True)
    posts_df["title"] = ""

    print(f"Number of rows before processing: {len(posts_df)}")

    # Clean title/selftext
    posts_df["title"] = posts_df["title"].fillna("").astype(str).str.lower().apply(clean_text)
    posts_df["selftext"] = posts_df["selftext"].fillna("").astype(str).str.lower().apply(clean_text)

    posts_df["lemmatized_title"] = posts_df["title"].apply(lemmatize_text)
    posts_df["lemmatized_selftext"] = posts_df["selftext"].apply(lemmatize_text)
    posts_df["stemmed_title"] = posts_df["title"].apply(stem_text)
    posts_df["stemmed_selftext"] = posts_df["selftext"].apply(stem_text)

    # Output filenames
    output_file = PATH_FINAL_REPORTS_FILE
    filtered_output_file = PATH_FINAL_FILTERED_OUTPUT_FILE

    total_start = time.time()

    # Process match types
    for match_type in ["regular", "lemmatized", "stemmed"]:
        start = time.time()
        print(f"\nStarting {match_type} matching...")

        if match_type == "regular":
            title_col = "title"
            selftext_col = "selftext"
            fire_list = fire_terms
            suicide_list = suicide_terms
        elif match_type == "lemmatized":
            title_col = "lemmatized_title"
            selftext_col = "lemmatized_selftext"
            fire_list = lemmatized_fire_terms
            suicide_list = lemmatized_suicide_terms
        elif match_type == "stemmed":
            title_col = "stemmed_title"
            selftext_col = "stemmed_selftext"
            fire_list = stemmed_fire_terms
            suicide_list = stemmed_suicide_terms

        posts_df[f"{match_type}_firearm_matches"] = posts_df.progress_apply(
            lambda row: check_exact_phrase_match(row[title_col], row[selftext_col], fire_list),
            axis=1,
        )
        posts_df[f"{match_type}_suicide_matches"] = posts_df.progress_apply(
            lambda row: check_exact_phrase_match(row[title_col], row[selftext_col], suicide_list),
            axis=1,
        )

        posts_df[f"{match_type}_firearm_match_summary"] = posts_df[f"{match_type}_firearm_matches"].apply(
            lambda x: ", ".join(x) if x else ""
        )
        posts_df[f"{match_type}_suicide_match_summary"] = posts_df[f"{match_type}_suicide_matches"].apply(
            lambda x: ", ".join(x) if x else ""
        )

        posts_df[f"{match_type}_both_matches"] = (
                (posts_df[f"{match_type}_firearm_match_summary"].str.strip() != "") &
                (posts_df[f"{match_type}_suicide_match_summary"].str.strip() != "")
        )

        # Export after each pass
        posts_df.to_csv(
            output_file,
            encoding="latin1",
            index=False,
            quoting=csv.QUOTE_MINIMAL,
            sep=",",
            doublequote=True
        )

        elapsed = timedelta(seconds=time.time() - start)
        print(f"{match_type.capitalize()} matching complete. Time taken: {elapsed}")

    # Final count
    print(f"\nNumber of rows after processing: {len(posts_df)}")

    # Filtered file (any both_matches = True)
    filtered_df = posts_df[
        posts_df.get("regular_both_matches", False) |
        posts_df.get("lemmatized_both_matches", False) |
        posts_df.get("stemmed_both_matches", False)
        ]

    filtered_df.to_csv(
        filtered_output_file,
        encoding="latin1",
        index=False,
        quoting=csv.QUOTE_MINIMAL,
        sep=",",
        doublequote=True
    )

    print(f"\nFiltered dataset saved to {filtered_output_file}")
    print(f"Total elapsed time: {timedelta(seconds=time.time() - total_start)}")


def preprocessing_key_terms_v1(posts_df: pd.DataFrame, terms_df: pd.DataFrame, column_name_post: str = 'text'):
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


def parse_arguments(parser):
    parser.add_argument('--merge_data', default=False, type=bool)
    return parser.parse_args()


parser = argparse.ArgumentParser(description='topic modeler')
args = parse_arguments(parser)

if args.merge_data:
    load_and_merge_csv_files()

# Loading raw data
df = load_merged_dataset()

# Load keywords from Excel
terms_df = pd.read_excel(PATH_KEYWORDS_FILE, sheet_name="terms")
preprocessing_key_terms_v2(df, terms_df)

# posts_df = preprocessing_key_terms_v1(df, terms_df)
# Save the updated dataset
# posts_df.to_csv(PATH_FINAL_REPORTS_FILE, index=False, encoding='utf-8', quotechar='"', quoting=csv.QUOTE_ALL)
