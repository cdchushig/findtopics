import re
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



def plot_topic_distribution_percent(df, topic_col='Topic', count_col='Count', name_col='Name', title="Percentage of tweets per topic"):
    """
    Genera una gráfica de barras horizontal de temas basada en porcentajes.

    Parámetros:
    - df: DataFrame con columnas 'Topic', 'Count', 'Name'
    - topic_col: Nombre de la columna con IDs de tópico
    - count_col: Nombre de la columna con los conteos
    - name_col: Nombre de la columna con los nombres descriptivos del tópico
    - title: Título del gráfico
    """
    # Eliminar el topic -1 si existe
    df_filtered = df[df[topic_col] != -1].copy()

    # Calcular porcentaje
    total = df_filtered[count_col].sum()
    df_filtered['Percentage'] = (df_filtered[count_col] / total) * 100

    # Ordenar por porcentaje
    df_filtered = df_filtered.sort_values(by='Percentage', ascending=True)

    # Estilo
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))

    # Gráfico
    ax = sns.barplot(
        x='Percentage',
        y=name_col,
        data=df_filtered,
        palette="viridis"
    )

    # # Etiquetas con porcentaje en cada barra
    # for i, row in df_filtered.iterrows():
    #     ax.text(row['Percentage'] + 0.2, i, f"{row['Percentage']:.1f}%", va='center')

    plt.title(title, fontsize=14)
    plt.xlabel("Porcentage (%)")
    plt.ylabel("Topic")
    plt.tight_layout()
    plt.show()


def plot_tweets_per_year(df, date_column='date'):
    """
    Genera un gráfico de barras del número de tweets por año.

    Parámetros:
    - df: DataFrame de pandas que contiene la columna de fechas.
    - date_column: Nombre de la columna con fechas (por defecto: 'date').

    Retorna:
    - Figura de matplotlib.
    """
    # Asegurar tipo datetime
    df[date_column] = pd.to_datetime(df[date_column], errors='coerce')

    # Eliminar filas sin fecha válida
    df = df.dropna(subset=[date_column])

    # Extraer el año
    df['year'] = df[date_column].dt.year

    # Contar tweets por año
    tweets_per_year = df['year'].value_counts().sort_index()

    # Graficar
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.barplot(x=tweets_per_year.index, y=tweets_per_year.values, palette="Blues_d")

    plt.title("Número de Tweets por Año")
    plt.xlabel("Año")
    plt.ylabel("Número de Tweets")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def text_eda_summary(df, column):
    """
    Performs EDA on a text column of a pandas DataFrame.

    Parameters:
        df (pd.DataFrame): DataFrame containing the text column.
        column (str): Name of the column to analyze.

    Returns:
        summary_df (pd.DataFrame): Summary statistics per row.
    """
    # Drop NA
    texts = df[column].dropna()

    # Basic Metrics
    summary_df = pd.DataFrame()
    summary_df['text'] = texts
    summary_df['num_words'] = texts.apply(lambda x: len(x.split()))
    summary_df['num_chars'] = texts.apply(len)
    summary_df['avg_word_length'] = texts.apply(
        lambda x: sum(len(word) for word in x.split()) / len(x.split()) if len(x.split()) > 0 else 0
    )
    summary_df['capital_ratio'] = texts.apply(
        lambda x: sum(1 for c in x if c.isupper()) / len(x) if len(x) > 0 else 0
    )

    # Word Frequency
    all_words = " ".join(texts).lower()
    words = re.findall(r'\b\w+\b', all_words)
    word_counts = Counter(words)
    top_words = pd.DataFrame(word_counts.most_common(10), columns=['word', 'count'])

    # Text Stats Summary
    print("Summary Statistics:")
    print(summary_df[['num_words', 'num_chars', 'avg_word_length', 'capital_ratio']].describe())

    # Plotting
    plt.figure(figsize=(14, 10))
    plt.subplot(2, 2, 1)
    sns.histplot(summary_df['num_words'], bins=20, kde=True)
    plt.title('Number of Words per Entry')

    plt.subplot(2, 2, 2)
    sns.histplot(summary_df['num_chars'], bins=20, kde=True)
    plt.title('Number of Characters per Entry')

    plt.subplot(2, 2, 3)
    sns.histplot(summary_df['avg_word_length'], bins=20, kde=True)
    plt.title('Average Word Length')

    plt.subplot(2, 2, 4)
    sns.barplot(data=top_words, x='count', y='word', palette='viridis')
    plt.title('Top 10 Most Common Words')

    plt.tight_layout()
    plt.show()

    return summary_df


# path_topic = '/home/cdchushig/kawsai/findtopics/reports/topics/suicide/' + 'topics_firearms_suicide_firearms_english_200_100_0.1_words.xlsx'
path_topic_tweets = '/home/cdchushig/kawsai/findtopics/reports/topics/suicide_firearms/' + 'topics_firearms_suicide_firearms_english_200_100_0.1_topics.csv'

dfx = pd.read_csv(path_topic_tweets)
plot_topic_distribution_percent(dfx)
