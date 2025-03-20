import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# Generar una serie temporal de ejemplo
np.random.seed(42)
series = np.sin(np.linspace(0, 50, 500)) + np.random.normal(0, 0.1, 500)
labels = np.random.choice([0, 1], size=(500,), p=[0.5, 0.5])  # Etiquetas simuladas


# Función para generar ventanas deslizantes y extraer características
def generar_caracteristicas_ventanas(series, labels, window_size, step_size):
    features = []
    target = []

    for start in range(0, len(series) - window_size + 1, step_size):
        ventana = series[start:start + window_size]
        # Extraer estadísticas de la ventana
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


# Generar características
window_size = 50
step_size = 10
X, y = generar_caracteristicas_ventanas(series, labels, window_size, step_size)


# Wrapper de selección de características (Forward Selection)
def forward_selection(X, y, model, scoring="accuracy", cv=5):
    n_features = X.shape[1]
    selected_features = []
    remaining_features = list(range(n_features))
    best_score = 0

    while remaining_features:
        scores = []
        for feature in remaining_features:
            current_features = selected_features + [feature]
            X_subset = X.iloc[:, current_features].values
            # Evaluar el modelo con validación cruzada
            score = np.mean(cross_val_score(model, X_subset, y, cv=cv, scoring=scoring))
            scores.append((score, feature))

        # Seleccionar la mejor característica
        scores.sort(reverse=True, key=lambda x: x[0])
        best_new_score, best_new_feature = scores[0]

        if best_new_score > best_score:
            best_score = best_new_score
            selected_features.append(best_new_feature)
            remaining_features.remove(best_new_feature)
            print(f"Feature {X.columns[best_new_feature]} added, score: {best_score:.4f}")
        else:
            break  # Si no mejora, parar

    return selected_features, best_score


# Modelo base
model = RandomForestClassifier(random_state=42)

# Ejecutar el wrapper
selected_features, best_score = forward_selection(X, y, model)
print(f"Selected features: {[X.columns[i] for i in selected_features]}")
print(f"Best score: {best_score:.4f}")
