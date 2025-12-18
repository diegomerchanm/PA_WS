# src/train_price_model.py

import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Pour importer config.settings
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import DATA_PROCESSED_PATH  # dossier data/processed


def load_training_data():
    """Charger les données nettoyées pour entraîner le modèle de prix."""
    data_file = os.path.join(DATA_PROCESSED_PATH, "immobilier_clean.csv")
    if not os.path.exists(data_file):
        raise FileNotFoundError(
            f"{data_file} introuvable. Lance d'abord src/cleaner.py."
        )

    df = pd.read_csv(data_file)

    # On enlève les lignes avec prix ou surface manquants ou non positifs
    df = df[(df["price"] > 0) & (df["surface"] > 0)]
    df = df.dropna(subset=["price", "surface", "city", "region"])

    # On se concentre sur des colonnes simples et robustes
    feature_cols = [
        "city",
        "region",
        "surface",
        "rooms",
        "bedrooms",
        "property_type",
        "has_elevator",
        "is_furnished",
        "parking",
    ]

    # S'il manque certaines colonnes (ex: bedrooms), on les crée par défaut
    for col in feature_cols:
        if col not in df.columns:
            if col in ["rooms", "bedrooms"]:
                df[col] = 0
            elif col in ["has_elevator", "is_furnished", "parking"]:
                df[col] = False
            else:
                df[col] = np.nan

    # Types
    df["rooms"] = df["rooms"].fillna(0).astype(int)
    df["bedrooms"] = df["bedrooms"].fillna(0).astype(int)
    df["has_elevator"] = df["has_elevator"].fillna(False).astype(bool)
    df["is_furnished"] = df["is_furnished"].fillna(False).astype(bool)
    df["parking"] = df["parking"].fillna(False).astype(bool)

    X = df[feature_cols].copy()
    y = df["price"].astype(float)

    return X, y


def build_model():
    """Construire un pipeline sklearn pour la prédiction de prix."""
    # Liste des features
    numeric_features = ["surface", "rooms", "bedrooms"]
    categorical_features = ["city", "region", "property_type", "has_elevator", "is_furnished", "parking"]

    # Préprocesseur
    numeric_transformer = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    # Modèle de régression
    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )

    # Pipeline complet
    clf = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    return clf


def train_and_save_model():
    """Entraîner le modèle et le sauvegarder dans data/processed/price_model.pkl."""
    print("Chargement des données d'entraînement...")
    X, y = load_training_data()

    print(f" Données chargées : {len(X)} lignes")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(" Construction du pipeline modèle...")
    model = build_model()

    print(" Entraînement du modèle...")
    model.fit(X_train, y_train)

    print(" Évaluation sur le jeu de test...")
    y_pred = model.predict(X_test)

    rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))
    mae = np.mean(np.abs(y_pred - y_test))
    r2 = 1 - np.sum((y_pred - y_test) ** 2) / np.sum((y_test - y_test.mean()) ** 2)

    print(f" RMSE : {rmse:,.0f} €")
    print(f" MAE  : {mae:,.0f} €")
    print(f" R²   : {r2:.3f}")

    # Sauvegarde du modèle
    model_path = os.path.join(DATA_PROCESSED_PATH, "price_model.pkl")
    joblib.dump(model, model_path)
    print(f" Modèle sauvegardé dans : {model_path}")


if __name__ == "__main__":
    train_and_save_model()
