# Librerías estándar
import os
import gzip
import pickle
import zipfile
import json

# Manipulación y análisis de datos
import pandas as pd
import numpy as np

# Scikit-learn
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


# 1.1 Cargar los datos
train_data = pd.read_csv(
    "files/input/train_data.csv.zip", index_col=False, compression="zip"
)
test_data = pd.read_csv(
    "files/input/test_data.csv.zip", index_col=False, compression="zip"
)

# 1.2 Organizar los datasets:


def limpiar(df):
    df = df.rename(columns={"default payment next month": "default"})
    df.drop("ID", axis=1, inplace=True)
    df["EDUCATION"] = df["EDUCATION"].apply(lambda x: 4 if x > 4 else x)
    df = df.query("MARRIAGE > 0 and EDUCATION > 0")
    df = df.dropna()
    return df


train_data = limpiar(train_data)
test_data = limpiar(test_data)

x_train = train_data.drop(columns=["default"])
y_train = train_data["default"]

x_test = test_data.drop(columns=["default"])
y_test = test_data["default"]

Categoria = ["SEX", "EDUCATION", "MARRIAGE"]


transformer = ColumnTransformer(
    transformers=[("ohe", OneHotEncoder(dtype=int), Categoria)], remainder="passthrough"
)

pipeline = Pipeline(
    steps=[
        ("transformer", transformer),
        ("clasi", RandomForestClassifier(n_jobs=-1, random_state=17)),
    ]
)

pipeline

# Precisión:
pipeline.fit(x_train, y_train)
print("Precisión:", pipeline.score(x_test, y_test))

param_grid = {
    "clasi__n_estimators": [180],
    "clasi__max_features": ["sqrt"],
    "clasi__min_samples_split": [10],
    "clasi__min_samples_leaf": [2],
    "clasi__bootstrap": [True],
    "clasi__max_depth": [None],
}

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=10,
    scoring="balanced_accuracy",
    n_jobs=-1,
    refit=True,
    verbose=True,
)
grid_search.fit(x_train, y_train)

os.makedirs("files/models", exist_ok=True)

with gzip.open("files/models/model.pkl.gz", "wb") as file:
    pickle.dump(grid_search, file)


def cargar_modelo_y_predecir(data, modelo_path="files/models/model.pkl.gz"):
    try:
        with gzip.open(modelo_path, "rb") as file:
            estimator = pickle.load(file)
        return estimator.predict(data)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"No se encontró el archivo de modelo en la ruta especificada: {modelo_path}"
        )
    except Exception as e:
        raise RuntimeError(f"Error al cargar el modelo o realizar predicciones: {e}")


# Uso de la función
y_train_pred = cargar_modelo_y_predecir(x_train)
y_test_pred = cargar_modelo_y_predecir(x_test)

import os
import json
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    balanced_accuracy_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


def metricas(dict_metricas):
    models_dir = "files/output"
    os.makedirs(models_dir, exist_ok=True)

    if os.path.exists("files/output/metrics.json"):
        with open("files/output/metrics.json", mode="r") as file:
            if len(file.readlines()) >= 4:
                os.remove("files/output/metrics.json")

    with open("files/output/metrics.json", mode="a") as file:
        file.write(str(dict_metricas).replace("'", '"') + "\n")


def evaluacion(dataset, y_true, y_pred):
    accuracy = float(accuracy_score(y_true, y_pred))
    precision = float(precision_score(y_true, y_pred))
    balanced_accuracy = float(balanced_accuracy_score(y_true, y_pred))
    recall = float(recall_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred))
    metrics = {
        "type": "metrics",
        "dataset": dataset,
        "precision": precision,
        "balanced_accuracy": balanced_accuracy,
        "recall": recall,
        "f1_score": f1,
    }

    metricas(metrics)


metrics_train = evaluacion("train", y_train, y_train_pred)
metrics_test = evaluacion("test", y_test, y_test_pred)


def matriz_confusion(dataset, y_true, y_pred):
    matriz = confusion_matrix(y_true, y_pred)
    matrix_confusion = {
        "type": "cm_matrix",
        "dataset": dataset,
        "true_0": {
            "predicted_0": int(matriz[0, 0]),
            "predicted_1": int(matriz[0, 1]),
        },
        "true_1": {"predicted_0": int(matriz[1, 0]), "predicted_1": int(matriz[1, 1])},
    }

    metricas(json.dumps(matrix_confusion))


metrics_train_cm = matriz_confusion("train", y_train, y_train_pred)
metrics_test_cm = matriz_confusion("test", y_test, y_test_pred)
