from pathlib import Path

PATH_PROJECT_DIR = Path(__file__).resolve().parents[2]

PATH_PROJECT_DATA = Path.joinpath(PATH_PROJECT_DIR, 'data')
PATH_PROJECT_DATA_PSYCHOTHERAPY = Path.joinpath(PATH_PROJECT_DIR, 'data', 'psychotherapy')
PATH_PROJECT_DATA_ADDICTIONS = Path.joinpath(PATH_PROJECT_DIR, 'data', 'addictions')
PATH_PROJECT_DATA_FIREARMS = Path.joinpath(PATH_PROJECT_DIR, 'data', 'firearms_twitter')

PATH_PROJECT_REPORTS = Path.joinpath(PATH_PROJECT_DIR, 'reports')
PATH_PROJECT_MODELS = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'models')
PATH_PROJECT_TOPICS = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'topics')
PATH_PROJECT_REPORTS_PSYCHOTHERAPY = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'psychotherapy')
PATH_PROJECT_REPORTS_FIGURES = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'figures')

DATASET_NAME_COGNITIVE_BEHAVIOURAL_THERAPY = 'cognitive_behavioural_therapy_topic'
DATASET_NAME_PSYCHO_THERAPY = 'psychoanadynamic_therapy_merged_topic'
DATASET_NAME_NARRATIVE_THERAPY = 'narrative_therapy_topic'
DATASET_NAME_FAMILY_COUPLE_THERAPY = 'familycouple_therapy_merged_topic'
DATASET_NAME_ACCEPTANCE_COMMITMENT_THERAPY = 'acceptance_commitment_therapy_topic'

FILENAME_DATA_TOPICS_FIREARMS = 'topics_firearms_firearms_english_10000_1000_0.5'
FILENAME_DATA_TOPICS_SUICIDE = 'topics_firearms_suicide_english_200_40_0.1'
FILENAME_DATA_TOPICS_SUICIDE_FIREARMS = 'topics_firearms_suicide_firearms_english_200_40_0.1'

DICT_DATASETS_THERAPY = {
    'cognitive': DATASET_NAME_COGNITIVE_BEHAVIOURAL_THERAPY,
    'psycho': DATASET_NAME_PSYCHO_THERAPY,
    'narrative': DATASET_NAME_NARRATIVE_THERAPY,
    'family': DATASET_NAME_FAMILY_COUPLE_THERAPY,
    'acceptance': DATASET_NAME_ACCEPTANCE_COMMITMENT_THERAPY
}

DICT_THERAPY_FULL_NAME_EN = {
    'cognitive': 'Cognitive behavioural',
    'psycho': 'Psychoanalytic',
    'narrative': 'Narrative',
    'family': 'Family and couples',
    'acceptance': 'Acceptance and commitment'
}

DICT_THERAPY_FULL_NAME_ES = {
    'cognitive': 'Cognitivo conductual',
    'psycho': 'Psicoanálisis',
    'narrative': 'Terapia narrativa',
    'family': 'Terapia familiar y de pareja',
    'acceptance': 'Terapia de aceptación y compromiso'
}

# LIST_THERAPIES = ['cognitive', 'psycho', 'narrative', 'family', 'acceptance']
LIST_THERAPIES = ['cognitive', 'psycho', 'narrative', 'acceptance']

MODEL_TRANSLATOR_CATALAN_SPANISH = 'BSC-LT/salamandra-2b'
MODEL_TRANSLATOR_BASQUE_SPANISH = 'BSC-LT/salamandra-2b'

DICT_TRANSLATION_LANGUAGES = {
    'basque': 'eu',
    'catalan': 'ca',
    'spanish': 'es',
    'english': 'en'
}


dict_prg_spanish = {
    1: 'Consumption',
    2: 'Criminal',
    3: 'Social media',
    4: 'LGTB',
    5: 'Web pages'
}

dict_prg_english = {
    1: 'Scandals',
    2: 'Teen porn',
    3: 'Social media',
    4: 'Preferences',
}

dict_prg_basque = {
    1: 'Explicit content',
    2: 'Content related to porn',
    3: 'Social, legal and educational discussion',
    4: 'Actors and actresses',
}

dict_prg_catalan = {
    1: 'Violent/ilegal porn',
    2: 'Conventional porn',
    3: 'Spanish porn',
    4: 'Producers',
}

dict_ldp_spanish_casas = {
    1: 'Football',
    2: 'Codere (advertising)',
    3: 'Codere (links)'
}

dict_ldp_english_casas = {
    1: 'Football betting',
    2: 'Other sports betting'
}

dict_ldp_spanish_otros = {
    1: 'Betting',
    2: 'Football',
    3: 'Addictions'
}

dict_ldp_english_otros = {
    1: 'Addictions',
    2: 'Prevention'
}

dict_ldp_basque_otros = {
    1: 'Gambling',
    2: 'Football'
}

dict_addiction_language = {
    'prg_spanish': dict_prg_spanish,
    'prg_english': dict_prg_english,
    'prg_basque': dict_prg_basque,
    'prg_catalan': dict_prg_catalan,
    'ldp_spanish_casas': dict_ldp_spanish_casas,
    'ldp_spanish_otros': dict_ldp_spanish_otros,
    'ldp_english_casas': dict_ldp_english_casas,
    'ldp_english_otros': dict_ldp_english_otros,
    'ldp_basque_otros': dict_ldp_basque_otros
}