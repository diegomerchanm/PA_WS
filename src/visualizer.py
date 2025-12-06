import os
import sys
import pandas as pd
import folium
from folium.plugins import MarkerCluster

# Pour pouvoir importer config.settings comme dans scraper/analyzer
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import DATA_PROCESSED_PATH


class MapVisualizer:
    """Create interactive map of real estate listings using Folium."""

    def __init__(self):
        # Fichier de données nettoyées
        self.data_file = os.path.join(DATA_PROCESSED_PATH, 'immobilier_clean.csv')
        # Fichier de sortie pour la carte
        self.output_file = os.path.join(DATA_PROCESSED_PATH, 'map_listings.html')
        self.df = None

    def load_data(self):
        """Load clean data with latitude & longitude."""
        print("Loading clean data for visualization.")

        if not os.path.exists(self.data_file):
            print(f"File not found: {self.data_file}")
            return False

        df = pd.read_csv(self.data_file)

        # On garde seulement les lignes avec coordonnées valides
        if 'latitude' not in df.columns or 'longitude' not in df.columns:
            print("Columns 'latitude' and 'longitude' not found in data.")
            return False

        df = df[df['latitude'].notna() & df['longitude'].notna()]

        if df.empty:
            print("No rows with valid latitude/longitude. Nothing to plot.")
            return False

        self.df = df
        print(f"Loaded {len(self.df)} records with coordinates")
        return True

    def _get_map_center(self):
        """Compute map center as mean of lat/lon."""
        center_lat = self.df['latitude'].mean()
        center_lon = self.df['longitude'].mean()
        return center_lat, center_lon

    def create_map(self):
        """Create the interactive Folium map."""
        if self.df is None or self.df.empty:
            print("No data loaded, aborting map creation.")
            return

        print("\nCreating interactive map...")

        # 1) Centre de la carte
        center_lat, center_lon = self._get_map_center()

        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=6,           # Zoom de départ (6 ~ France entière)
            tiles="OpenStreetMap"   # Fond de carte
        )

        # 2) Cluster pour éviter trop de points superposés
        marker_cluster = MarkerCluster().add_to(m)

        # 3) Ajout d'un marker par bien
        for _, row in self.df.iterrows():
            lat = row['latitude']
            lon = row['longitude']
            price = row.get('price', None)
            city = row.get('city', '')
            surface = row.get('surface', None)
            rooms = row.get('rooms', None)
            url = row.get('url', None)

            # Texte affiché au survol (tooltip)
            if price is not None:
                tooltip = f"{price:.0f} € - {city}"
            else:
                tooltip = city

            # Contenu de la popup (au clic)
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

        # 4) Sauvegarde de la carte
        os.makedirs(DATA_PROCESSED_PATH, exist_ok=True)
        m.save(self.output_file)

        print(f" Map saved to: {self.output_file}")
        print("Open this file in your browser to explore the map.")


def main():
    """Main entry point."""
    visualizer = MapVisualizer()
    if not visualizer.load_data():
        return
    visualizer.create_map()


if __name__ == "__main__":
    main()
