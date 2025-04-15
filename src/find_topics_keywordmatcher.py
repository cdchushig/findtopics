import pandas as pd
import csv
import re
import time
from datetime import timedelta
from nltk.stem import WordNetLemmatizer, PorterStemmer
import unicodedata
from tqdm import tqdm
tqdm.pandas()  # Enables progress bar with .progress_apply()

# Initialize the lemmatizer & stemmer
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

# Load keywords from Excel
keywords_file = "LMS Keywords v3.2 Reddit FINAL.xlsx"
terms_df = pd.read_excel(keywords_file, sheet_name="terms")


def lemmatize_text(text):
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])


def stem_text(text):
    return " ".join([
        word if stemmer.stem(word) == "of" else stemmer.stem(word)
        for word in text.split()
    ])


def extract_terms(terms):
    return [term.lower() for term in terms.dropna().astype(str)]


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


def clean_text(text):
    """
    Clean text by normalizing and removing accents
    """
    if isinstance(text, str):
        text = unicodedata.normalize("NFKD", text)
        text = ''.join([c for c in text if not unicodedata.combining(c)])
        return text.strip()
    return text


# Load search terms
fire_terms = extract_terms(terms_df["firearm_terms"])
suicide_terms = extract_terms(terms_df["suicide_terms"])
lemmatized_fire_terms = lemmatize_terms(fire_terms)
lemmatized_suicide_terms = lemmatize_terms(suicide_terms)
stemmed_fire_terms = stem_terms(fire_terms)
stemmed_suicide_terms = stem_terms(suicide_terms)

# Load Reddit posts
posts_file = "Final_All_Reddit_Posts_2015-2025_After_Removed_Group_1.csv"
posts_df = pd.read_csv(posts_file, encoding="latin1")
posts_df.columns = posts_df.columns.str.lower()
posts_df.rename(columns={'content': 'selftext'}, inplace=True)

print(f"Number of rows before processing: {len(posts_df)}")

# Clean title/selftext
posts_df["title"] = posts_df["title"].fillna("").astype(str).str.lower().apply(clean_text)
posts_df["selftext"] = posts_df["selftext"].fillna("").astype(str).str.lower().apply(clean_text)

posts_df["lemmatized_title"] = posts_df["title"].apply(lemmatize_text)
posts_df["lemmatized_selftext"] = posts_df["selftext"].apply(lemmatize_text)
posts_df["stemmed_title"] = posts_df["title"].apply(stem_text)
posts_df["stemmed_selftext"] = posts_df["selftext"].apply(stem_text)

# Function to check for exact phrase match
def check_exact_phrase_match(title, selftext, term_list):
    matches = []
    for term in term_list:
        regex_pattern = r'(^|\s|[^\w])' + re.escape(term) + r'($|\s|[^\w])'
        if re.search(regex_pattern, title) or re.search(regex_pattern, selftext):
            matches.append(term)
    return matches

# Output filenames
output_file = "Final_All_Reddit_Posts_2015-2025_After_Removed_Group_1_with_matches.csv"
filtered_output_file = "LMS_Boolean_TRUE_both_Group_5.csv"

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