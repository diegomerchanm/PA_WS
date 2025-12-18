# PA_WS - Analyse Marché Immobilier

Projet d'analyse du marché immobilier français par web scraping (API Bien'ici).

## Installation

git clone [URL]
cd PA_WS
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

## Exécution

python src/scraper.py --pages 50
python src/cleaner.py
python src/analyzer.py
python src/price_model.py
streamlit run dashboard/app.py

## Structure

PA_WS/
├── config/          # Configuration
├── data/            # Données (raw + processed)
├── src/             # Scripts pipeline
├── dashboard/       # Application Streamlit
└── notebooks/       # Exploration

## Scripts principaux

- scraper.py : Collecte données via API
- cleaner.py : Nettoyage et géocodage
- analyzer.py : Analyses statistiques
- price_model.py : Modèle prédictif (RandomForest)
- dashboard/app.py : Interface interactive

## Technologies

pandas, numpy, scikit-learn, streamlit, folium, geopy

## Auteurs

Diego Merchan, Perveena Sivayanama
Master Data Analytics - Paris 1 Panthéon-Sorbonne
2025-2026