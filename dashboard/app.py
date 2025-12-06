# dashboard/app.py

import os
import sys
import json
import pandas as pd
import altair as alt
import streamlit as st
import folium
import joblib
from folium.plugins import MarkerCluster
from streamlit.components.v1 import html

# Pour importer config.settings depuis dashboard/
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import DATA_PROCESSED_PATH  # chemin vers data/processed


# =========================
# Chargement des données
# =========================

@st.cache_data
def load_data():
    """Charger le fichier immobilier_clean.csv et préparer quelques colonnes."""
    data_file = os.path.join(DATA_PROCESSED_PATH, "immobilier_clean.csv")

    if not os.path.exists(data_file):
        st.error(f"Fichier introuvable : {data_file}. Lance d'abord le cleaner (src/cleaner.py).")
        return None

    df = pd.read_csv(data_file)

    # Colonnes indispensables
    expected_cols = ["city", "region", "price", "surface", "rooms"]
    for col in expected_cols:
        if col not in df.columns:
            st.error(f"Colonne manquante dans les données : {col}")
            return None

    # Nettoyage léger / typage
    df["rooms"] = df["rooms"].fillna(0).astype(int)

    # Créer price_per_m2 si elle n'existe pas
    if "price_per_m2" not in df.columns:
        df["price_per_m2"] = (df["price"] / df["surface"]).round(2)

    return df

@st.cache_data
def load_analysis_results():
    """Charger le fichier analysis_results.json produit par analyzer.py."""
    path = os.path.join(DATA_PROCESSED_PATH, "analysis_results.json")
    if not os.path.exists(path):
        st.warning("Le fichier analysis_results.json est introuvable. Lance d'abord src/analyzer.py.")
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Impossible de lire analysis_results.json : {e}")
        return None

@st.cache_resource
def load_price_model():
    """Charger le modèle de prix entraîné."""
    model_path = os.path.join(DATA_PROCESSED_PATH, "price_model.pkl")
    if not os.path.exists(model_path):
        st.warning("Modèle de prix introuvable. Lance d'abord : python src/train_price_model.py")
        return None

    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle : {e}")
        return None

# =========================
# Carte Folium
# =========================

def create_folium_map(df):
    """
    Créer une carte Folium (HTML) à partir d'un DataFrame
    contenant latitude / longitude et quelques infos.
    """
    if df.empty:
        return None

    # On garde seulement les lignes avec coordonnées
    if "latitude" not in df.columns or "longitude" not in df.columns:
        return None

    df_map = df.dropna(subset=["latitude", "longitude"]).copy()
    if df_map.empty:
        return None

    # Centre = moyenne des lat/lon
    center_lat = df_map["latitude"].mean()
    center_lon = df_map["longitude"].mean()

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=6,        
        tiles="OpenStreetMap"
    )

    marker_cluster = MarkerCluster().add_to(m)

    for _, row in df_map.iterrows():
        lat = row["latitude"]
        lon = row["longitude"]
        price = row.get("price", None)
        city = row.get("city", "")
        surface = row.get("surface", None)
        rooms = row.get("rooms", None)
        url = row.get("url", None)

        # Texte au survol
        if price is not None:
            tooltip = f"{price:.0f} € - {city}"
        else:
            tooltip = city

        # Contenu de la popup
        popup_lines = []
        if city:
            popup_lines.append(f"<b>Ville :</b> {city}")
        if price is not None:
            popup_lines.append(f"<b>Prix :</b> {price:.0f} €")
        if surface is not None:
            popup_lines.append(f"<b>Surface :</b> {surface:.1f} m²")
        if rooms is not None:
            popup_lines.append(f"<b>Pièces :</b> {int(rooms)}")
        if url:
            popup_lines.append(f'<a href="{url}" target="_blank">Voir l\'annonce</a>')

        popup_html = "<br>".join(popup_lines) if popup_lines else "Détails indisponibles"

        folium.CircleMarker(
            location=[lat, lon],
            radius=4,
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=tooltip,
            fill=True,
            fill_opacity=0.7,
        ).add_to(marker_cluster)

    # On renvoie le HTML de la carte
    return m._repr_html_()


# =========================
# Sidebar (filtres)
# =========================

def sidebar_filters(df):
    """Créer les filtres dans la sidebar et retourner le DataFrame filtré."""

    st.sidebar.header("Filtres")

    # --- Région ---
    regions = sorted(df["region"].dropna().unique().tolist())
    regions_options = ["Toutes les régions"] + regions
    selected_region = st.sidebar.selectbox(
        "Région",
        regions_options,
        index=0
    )

    # Filtre temporaire pour les villes (affichage des choix)
    if selected_region == "Toutes les régions":
        df_region = df.copy()
    else:
        df_region = df[df["region"] == selected_region]

    # --- Ville ---
    cities = sorted(df_region["city"].dropna().unique().tolist())
    cities_options = ["Toutes les villes"] + cities
    selected_city = st.sidebar.selectbox(
        "Ville",
        cities_options,
        index=0
    )

    # --- Surface ---
    min_surface = float(df_region["surface"].min())
    max_surface = float(df_region["surface"].max())
    surface_range = st.sidebar.slider(
        "Surface (m²)",
        min_value=float(int(min_surface)),
        max_value=float(int(max_surface) + 1),
        value=(min_surface, max_surface),
        step=1.0
    )

    # --- Prix ---
    min_price = float(df_region["price"].min())
    max_price = float(df_region["price"].max())
    price_range = st.sidebar.slider(
        "Prix (€)",
        min_value=float(int(min_price)),
        max_value=float(int(max_price) + 1),
        value=(min_price, max_price),
        step=50.0
    )

    # --- Nombre de pièces ---
    min_rooms = int(df_region["rooms"].min())
    max_rooms = int(df_region["rooms"].max())
    rooms_range = st.sidebar.slider(
        "Nombre de pièces",
        min_value=min_rooms,
        max_value=max_rooms,
        value=(min_rooms, max_rooms),
        step=1
    )

    # =====================
    # Application des filtres
    # =====================
    df_filtered = df_region[
        (df_region["surface"] >= surface_range[0]) &
        (df_region["surface"] <= surface_range[1]) &
        (df_region["price"] >= price_range[0]) &
        (df_region["price"] <= price_range[1]) &
        (df_region["rooms"] >= rooms_range[0]) &
        (df_region["rooms"] <= rooms_range[1])
    ]

    # Filtre ville
    if selected_city != "Toutes les villes":
        df_filtered = df_filtered[df_filtered["city"] == selected_city]

    return df_filtered, selected_region, selected_city, price_range, surface_range, rooms_range


# =========================
# Page 1 : Exploration
# =========================

def page_exploration(df):
    """Page d'exploration interactive (ta page actuelle)."""

    # Filtres
    df_filtered, region, city, price_range, surface_range, rooms_range = sidebar_filters(df)

    st.markdown("### Filtres appliqués")
    filtres_txt = []
    if region != "Toutes les régions":
        filtres_txt.append(f"Région : **{region}**")
    if city != "Toutes les villes":
        filtres_txt.append(f"Ville : **{city}**")
    filtres_txt.append(f"Prix : **{int(price_range[0])}€ - {int(price_range[1])}€**")
    filtres_txt.append(f"Surface : **{int(surface_range[0])}m² - {int(surface_range[1])}m²**")
    filtres_txt.append(f"Pièces : **{rooms_range[0]} - {rooms_range[1]}**")

    st.markdown(" • ".join(filtres_txt))

    # Si aucune donnée après filtres
    if df_filtered.empty:
        st.warning("Aucun bien ne correspond aux filtres.")
        return

    # =========================
    # Indicateurs clés (KPI)
    # =========================
    st.markdown("### Indicateurs clés")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Nombre de biens",
            len(df_filtered)
        )

    with col2:
        st.metric(
            "Prix moyen",
            f"{df_filtered['price'].mean():.0f} €"
        )

    with col3:
        st.metric(
            "Surface moyenne",
            f"{df_filtered['surface'].mean():.1f} m²"
        )

    with col4:
        st.metric(
            "Prix moyen au m²",
            f"{df_filtered['price_per_m2'].mean():.1f} €/m²"
        )

    # =========================
    # Graphiques
    # =========================
    st.markdown("### Graphiques")

    col_left, col_right = st.columns((2, 1))

    # Graphique 1 : Prix moyen par ville (top 10)
    with col_left:
        st.subheader("Prix moyen par ville (Top 10)")

        city_stats = (
            df_filtered.groupby("city")["price"]
            .mean()
            .sort_values(ascending=False)
            .head(10)
            .round(0)
        )

        st.bar_chart(city_stats)

    # Graphique 2 : Histogramme des prix (Altair)
    with col_right:
        st.subheader("Distribution des prix")

        hist = (
            alt.Chart(df_filtered)
            .mark_bar(opacity=0.7)
            .encode(
                alt.X("price:Q", bin=alt.Bin(maxbins=20), title="Prix (€)"),
                alt.Y("count()", title="Nombre d'annonces")
            )
            .properties(height=300)
        )

        st.altair_chart(hist, use_container_width=True)

    # =========================
    # Carte Folium intégrée (HTML)
    # =========================
    st.markdown("### Carte des biens (Folium)")

    map_html = create_folium_map(df_filtered)
    if map_html is not None:
        html(map_html, height=600, scrolling=False)
    else:
        st.info("Aucune coordonnée disponible pour les biens filtrés ou colonnes 'latitude'/'longitude' manquantes.")

    # =========================
    # Tableau détaillé
    # =========================
    st.markdown("### Détails des biens filtrés")
    cols_to_show = [
        "city",
        "region",
        "price",
        "surface",
        "rooms",
        "price_per_m2",
        "url",
    ]
    existing_cols = [c for c in cols_to_show if c in df_filtered.columns]

    st.dataframe(
        df_filtered[existing_cols].sort_values("price", ascending=False)
    )


# =========================
# Insights avancés
# =========================
def page_insights(df, results):
    """Page Insights : utilise analysis_results.json pour afficher des analyses avancées."""
    st.header("Insights avancés du marché")

    if results is None:
        st.info("Aucun résultat d'analyse disponible pour l'instant.")
        return

    # --- Vue d'ensemble ---
    meta = results.get("metadata", {})
    total_props = meta.get("total_properties", len(df))
    analysis_date = meta.get("analysis_date", "N/A")
    data_source = meta.get("data_source", "inconnu")

    st.subheader("Vue d'ensemble du marché")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Nombre de biens analysés", total_props)
    with col2:
        st.metric("Prix moyen", f"{df['price'].mean():.0f} €")
    with col3:
        st.metric("Prix médian", f"{df['price'].median():.0f} €")
    with col4:
        st.metric("Prix moyen au m²", f"{df['price_per_m2'].mean():.1f} €/m²")

    st.caption(f"Source : {data_source} — Analyse du {analysis_date}")

    st.markdown("---")

    # A. Relation surface / prix au m²
    st.subheader(" Relation entre surface et prix au m²")

    df_scatter = df[["surface", "price_per_m2", "city", "region", "property_type"]].dropna()

    # On limite à un certain nombre de points pour éviter un graphe illisible
    max_points = 5000
    if len(df_scatter) > max_points:
        df_scatter = df_scatter.sample(max_points, random_state=42)

    scatter = (
        alt.Chart(df_scatter)
        .mark_circle(opacity=0.4)
        .encode(
            x=alt.X("surface:Q", title="Surface (m²)"),
            y=alt.Y("price_per_m2:Q", title="Prix au m² (€)"),
            color=alt.Color("region:N", title="Région", legend=None),
            tooltip=[
                "surface",
                "price_per_m2",
                "city",
                "region",
                "property_type",
            ],
        )
        .properties(height=350)
        .interactive()
    )

    st.altair_chart(scatter, use_container_width=True)

    st.caption(
        "On observe généralement que les petites surfaces ont un prix au m² plus élevé, "
        "tandis que les grandes surfaces ont un prix au m² plus faible."
    )

    st.markdown("---")

    # B. Prix au m² selon la localisation (région / ville)
    st.subheader(" Prix au m² selon la localisation")

    col_loc1, col_loc2 = st.columns(2)

    # B1. Prix au m² par région
    with col_loc1:
        st.markdown("#### Par région")

        df_region = (
            df.dropna(subset=["region", "price_per_m2"])
            .groupby("region")["price_per_m2"]
            .mean()
            .sort_values(ascending=False)
            .reset_index()
        )

        chart_region = (
            alt.Chart(df_region)
            .mark_bar()
            .encode(
                x=alt.X("price_per_m2:Q", title="Prix moyen au m² (€)"),
                y=alt.Y("region:N", sort="-x", title="Région"),
                tooltip=["region", "price_per_m2"],
            )
            .properties(height=300)
        )

        st.altair_chart(chart_region, use_container_width=True)

    # B2. Prix au m² par ville (Top N)
    with col_loc2:
        st.markdown("#### Top villes (prix au m²)")

        df_city = (
            df.dropna(subset=["city", "price_per_m2"])
            .groupby("city")["price_per_m2"]
            .mean()
            .sort_values(ascending=False)
            .reset_index()
        )

        top_n = st.slider(
            "Nombre de villes à afficher",
            min_value=5,
            max_value=min(30, len(df_city)),
            value=10,
        )

        df_city_top = df_city.head(top_n)

        chart_city = (
            alt.Chart(df_city_top)
            .mark_bar()
            .encode(
                x=alt.X("price_per_m2:Q", title="Prix moyen au m² (€)"),
                y=alt.Y("city:N", sort="-x", title="Ville"),
                tooltip=["city", "price_per_m2"],
            )
            .properties(height=300)
        )

        st.altair_chart(chart_city, use_container_width=True)

    st.markdown("---")

    # Segments de prix
    st.subheader(" Segments de prix")

    seg = results.get("price_segmentation", {})
    if seg:
        seg_df = pd.DataFrame(
            [
                {
                    "segment": name,
                    "avg_price": data.get("avg_price"),
                    "avg_price_m2": data.get("avg_price_m2"),
                    "avg_surface": data.get("avg_surface"),
                    "avg_rooms": data.get("avg_rooms"),
                    "market_share": data.get("market_share"),
                    "count": data.get("count"),
                }
                for name, data in seg.items()
            ]
        )

        col_a, col_b = st.columns((2, 1))
        with col_a:
            chart_seg = (
                alt.Chart(seg_df)
                .mark_bar()
                .encode(
                    x=alt.X("segment:N", title="Segment"),
                    y=alt.Y("market_share:Q", title="Part de marché (%)"),
                    tooltip=["segment", "avg_price", "avg_price_m2", "avg_surface", "avg_rooms", "market_share", "count"],
                )
                .properties(height=300)
            )
            st.altair_chart(chart_seg, use_container_width=True)

        with col_b:
            st.dataframe(
                seg_df.set_index("segment")[["avg_price", "avg_price_m2", "avg_surface", "avg_rooms", "count", "market_share"]]
            )
    else:
        st.info("Pas encore de segmentation de prix dans les résultats.")

    st.markdown("---")

    # Types de biens
    st.subheader("Appartements vs Maisons")

    prop_types = results.get("property_type_comparison", {})
    if prop_types:
        pt_df = pd.DataFrame(
            [
                {
                    "type": name,
                    "avg_price": data.get("avg_price"),
                    "median_price": data.get("median_price"),
                    "avg_price_m2": data.get("avg_price_m2"),
                    "avg_surface": data.get("avg_surface"),
                    "avg_rooms": data.get("avg_rooms"),
                    "count": data.get("count"),
                    "market_share": data.get("market_share"),
                }
                for name, data in prop_types.items()
            ]
        )

        col_t1, col_t2 = st.columns((2, 1))
        with col_t1:
            chart_types = (
                alt.Chart(pt_df)
                .mark_bar()
                .encode(
                    x=alt.X("type:N", title="Type de bien"),
                    y=alt.Y("avg_price_m2:Q", title="Prix moyen au m²"),
                    color="type:N",
                    tooltip=["type", "avg_price", "median_price", "avg_price_m2", "avg_surface", "avg_rooms", "count", "market_share"],
                )
                .properties(height=300)
            )
            st.altair_chart(chart_types, use_container_width=True)

        with col_t2:
            st.dataframe(pt_df.set_index("type"))
    else:
        st.info("Pas encore de comparaison par type de bien dans les résultats.")

    st.markdown("---")

    # Clusters
    st.subheader(" Typologies de biens (clusters)")

    clusters = results.get("clusters", {})
    if clusters:
        clust_df = pd.DataFrame(
            [
                {
                    "cluster": name,
                    "label": data.get("name"),
                    "avg_price": data.get("avg_price"),
                    "avg_price_m2": data.get("avg_price_m2"),
                    "avg_surface": data.get("avg_surface"),
                    "avg_rooms": data.get("avg_rooms"),
                    "typical_city": data.get("typical_city"),
                    "count": data.get("count"),
                }
                for name, data in clusters.items()
            ]
        )

        st.dataframe(clust_df.set_index("cluster"))

        chart_clust = (
            alt.Chart(clust_df)
            .mark_circle(size=120)
            .encode(
                x=alt.X("avg_surface:Q", title="Surface moyenne (m²)"),
                y=alt.Y("avg_price_m2:Q", title="Prix moyen au m²"),
                color="label:N",
                size="count:Q",
                tooltip=["cluster", "label", "avg_price", "avg_surface", "avg_rooms", "avg_price_m2", "typical_city", "count"],
            )
            .properties(height=300)
        )
        st.altair_chart(chart_clust, use_container_width=True)
    else:
        st.info("Pas encore de clusters dans les résultats.")

    st.markdown("---")

    # Paris vs reste
    st.subheader(" Paris vs reste de la France")

    pvr = results.get("paris_vs_rest", {})
    paris = pvr.get("paris")
    rest = pvr.get("rest_of_france")
    comp = pvr.get("comparison")

    if paris and rest and comp:
        col_p1, col_p2, col_p3 = st.columns(3)
        with col_p1:
            st.metric("Prix moyen à Paris", f"{paris['avg_price']:.0f} €")
        with col_p2:
            st.metric("Prix moyen hors Paris", f"{rest['avg_price']:.0f} €")
        with col_p3:
            st.metric("Surcoût Paris (%)", f"{comp['price_difference_%']:.1f} %")

        st.caption(
            f"Paris représente {paris['market_share_%']:.1f}% des biens, "
            f"avec un prix/m² {comp['price_m2_difference_%']:.1f}% plus élevé."
        )
    else:
        st.info("Pas encore d'analyse Paris vs reste dans les résultats.")

    st.markdown("---")

    # Bons plans
    st.subheader("Top 'bons plans' (Value Score)")

    best_values = results.get("best_value_properties", [])
    if best_values:
        n = st.slider("Nombre de biens à afficher", min_value=5, max_value=min(20, len(best_values)), value=10)
        bv_df = pd.DataFrame(best_values[:n])
        st.dataframe(bv_df)
    else:
        st.info("Pas encore de 'best value properties' dans les résultats.")

# =========================
# Prédiction
# =========================

def page_prediction(df):
    """Page de prédiction du prix d'un bien."""
    st.header(" Prédiction du prix d'un bien immobilier")

    model = load_price_model()
    if model is None:
        st.info("Aucun modèle de prix disponible pour l'instant.")
        return

    st.markdown(
        "Renseigne les caractéristiques du bien ci-dessous pour obtenir "
        "une estimation de son **prix** et de son **prix au m²**."
    )

    # Préparer des valeurs par défaut à partir des données
    cities = sorted(df["city"].dropna().unique().tolist())
    regions = sorted(df["region"].dropna().unique().tolist())
    prop_types = sorted(df["property_type"].dropna().unique().tolist()) if "property_type" in df.columns else ["flat", "house"]

    col1, col2 = st.columns(2)

    with col1:
        region = st.selectbox("Région", regions)
        # Filtrer les villes de la région choisie
        df_region = df[df["region"] == region]
        cities_region = sorted(df_region["city"].dropna().unique().tolist())
        city = st.selectbox("Ville", cities_region or cities)

        surface = st.number_input(
            "Surface (m²)",
            min_value=10.0,
            max_value=500.0,
            value=float(df["surface"].median()),
            step=1.0,
        )

        rooms = st.slider(
            "Nombre de pièces",
            min_value=1,
            max_value=8,
            value=int(df["rooms"].median()) if "rooms" in df.columns else 3,
        )

    with col2:
        bedrooms = st.slider(
            "Nombre de chambres",
            min_value=0,
            max_value=6,
            value=int(df["bedrooms"].median()) if "bedrooms" in df.columns else max(0, rooms - 1),
        )

        property_type = st.selectbox("Type de bien", prop_types)

        has_elevator = st.checkbox("Ascenseur", value=True)
        is_furnished = st.checkbox("Meublé", value=False)
        parking = st.checkbox("Parking", value=False)

    if st.button("Prédire le prix"):
        # Construire un DataFrame avec les mêmes colonnes que lors de l'entraînement
        data_input = pd.DataFrame(
            [
                {
                    "city": city,
                    "region": region,
                    "surface": surface,
                    "rooms": rooms,
                    "bedrooms": bedrooms,
                    "property_type": property_type,
                    "has_elevator": has_elevator,
                    "is_furnished": is_furnished,
                    "parking": parking,
                }
            ]
        )

        try:
            pred_price = float(model.predict(data_input)[0])
            price_per_m2 = pred_price / surface

            st.success(f" Prix estimé : **{pred_price:,.0f} €**".replace(",", " "))
            st.info(f" Prix estimé au m² : **{price_per_m2:,.0f} €/m²**".replace(",", " "))
        except Exception as e:
            st.error(f"Erreur lors de la prédiction : {e}")


# =========================
# Affichage principal
# =========================

def main():
    st.set_page_config(
        page_title="Tableau de bord Immobilier",
        layout="wide"
    )

    st.title(" Tableau de bord Immobilier - Bien'ici")

    df = load_data()
    if df is None:
        return

    # Choix de la page
    page = st.sidebar.radio(
        "Page",
        ["Exploration", "Insights avancés","Prédiction du prix"],
        index=0
    )

    if page == "Exploration":
        page_exploration(df)
    elif page == "Insights avancés":
        results = load_analysis_results()
        page_insights(df, results)
    else:
        page_prediction(df)


if __name__ == "__main__":
    main()
