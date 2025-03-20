import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.model_selection import cross_val_score

np.random.seed(42)
series = np.sin(np.linspace(0, 50, 500)) + np.random.normal(0, 0.1, 500)
labels = np.random.choice([0, 1], size=(500,), p=[0.5, 0.5])


def generar_caracteristicas_ventanas(series, labels, window_size, step_size):
    features = []
    target = []

    for start in range(0, len(series) - window_size + 1, step_size):
        ventana = series[start:start + window_size]
        row = {
            "mean": np.mean(ventana),
            "std": np.std(ventana),
            "min": np.min(ventana),
            "max": np.max(ventana),
            "skew": pd.Series(ventana).skew(),
            "kurt": pd.Series(ventana).kurt()
        }
        features.append(row)
        # Etiqueta central de la ventana
        target.append(labels[start + window_size // 2])

    return pd.DataFrame(features), np.array(target)

def ensemble_feature_selection(X, y, n_features=3, cv=5):
    feature_names = X.columns
    n_total_features = len(feature_names)

    # 1. Filtrado estadístico (ANOVA F-test)
    skb_f_test = SelectKBest(score_func=f_classif, k=n_total_features)
    skb_f_test.fit(X, y)
    f_test_scores = skb_f_test.scores_

    # 2. Filtrado estadístico (Mutual Information)
    skb_mi = SelectKBest(score_func=mutual_info_classif, k=n_total_features)
    skb_mi.fit(X, y)
    mi_scores = skb_mi.scores_

    # 3. Importancia basada en modelos (Random Forest)
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X, y)
    rf_importances = rf.feature_importances_

    # 4. Ensamblar los puntajes (Ranking promedio)
    scores = np.vstack([f_test_scores, mi_scores, rf_importances])
    mean_scores = np.mean(scores, axis=0)

    # Ordenar por el puntaje promedio
    feature_ranking = sorted(
        zip(feature_names, mean_scores), key=lambda x: x[1], reverse=True
    )

    # Seleccionar las mejores características
    selected_features = [feature for feature, score in feature_ranking[:n_features]]

    return selected_features, feature_ranking


window_size = 50
step_size = 10
X, y = generar_caracteristicas_ventanas(series, labels, window_size, step_size)
selected_features, feature_ranking = ensemble_feature_selection(X, y, n_features=3)
print(f"Selected features: {selected_features}")
print("Feature rankings:")
for feature, score in feature_ranking:
    print(f"{feature}: {score:.4f}")
