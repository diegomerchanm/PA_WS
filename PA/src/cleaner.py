# src/cleaner.py

import numpy as np
import os
import pandas as pd
import re
import sys
from datetime import datetime
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

# Add parent directory for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import DATA_RAW_PATH, DATA_PROCESSED_PATH, BIENICI_RAW_FILE

class DataCleaner:
    """Clean and process scraped real estate data"""
    
    def __init__(self):
        self.raw_file = os.path.join(DATA_RAW_PATH, BIENICI_RAW_FILE)
        self.clean_file = os.path.join(DATA_PROCESSED_PATH, 'immobilier_clean.csv')
        self.df = None
        self.geolocator = Nominatim(user_agent="immobilier_cleaner/1.0", timeout=10)
        self.geocode = RateLimiter(
            self.geolocator.geocode,
            min_delay_seconds=1,   
            error_wait_seconds=2,
            swallow_exceptions=True
        )
        self.geocode_cache = {}
        
    def load_data(self):
        """Load raw data from CSV"""
        print(" Loading raw data...")
        
        if not os.path.exists(self.raw_file):
            print(f"File not found: {self.raw_file}")
            return False
        
        self.df = pd.read_csv(self.raw_file)
        print(f"Loaded {len(self.df)} records")
        return True
    
    def remove_duplicates(self):
        """Remove duplicate listings by ID"""
        print("\nRemoving duplicates...")
        
        initial_count = len(self.df)
        self.df = self.df.drop_duplicates(subset=['id'], keep='first')
        removed = initial_count - len(self.df)
        
        print(f"Removed {removed} duplicates ({len(self.df)} remaining)")
    
    def filter_property_types(self):
        """Keep only flats and houses"""
        print("\nFiltering property types...")
        
        initial_count = len(self.df)
        self.df = self.df[self.df['property_type'].isin(['flat', 'house'])]
        removed = initial_count - len(self.df)
        
        print(f" Removed {removed} non-residential properties ({len(self.df)} remaining)")
    
    def clean_prices(self):
        """Clean price data and remove outliers"""
        print("\n Cleaning prices...")
        
        # Remove null prices
        initial_count = len(self.df)
        self.df = self.df[self.df['price'].notna()]
        self.df = self.df[self.df['price'] > 0]
        
        # Remove extreme outliers (up to 10000€ for luxury properties)
        self.df = self.df[self.df['price'] <= 10000]
        self.df = self.df[self.df['price'] >= 200]  # Minimum realistic rent
        
        removed = initial_count - len(self.df)
        print(f" Removed {removed} invalid/extreme prices ({len(self.df)} remaining)")
        print(f"   Price range: {self.df['price'].min():.0f}€ - {self.df['price'].max():.0f}€")
    
    def clean_surface(self):
        """Clean surface area data"""
        print("\n Cleaning surface areas...")
        
        initial_count = len(self.df)
        
        # Remove null or zero surfaces
        self.df = self.df[self.df['surface'].notna()]
        self.df = self.df[self.df['surface'] > 0]
        
        # Remove unrealistic surfaces (< 10m² or > 500m²)
        self.df = self.df[self.df['surface'] >= 10]
        self.df = self.df[self.df['surface'] <= 500]
        
        removed = initial_count - len(self.df)
        print(f" Removed {removed} invalid surfaces ({len(self.df)} remaining)")
        print(f"   Surface range: {self.df['surface'].min():.1f}m² - {self.df['surface'].max():.1f}m²")
    
    def calculate_price_per_m2(self):
        """Calculate price per square meter"""
        print("\n Calculating price per m²...")
        
        self.df['price_per_m2'] = (self.df['price'] / self.df['surface']).round(2)
        
        avg_price_m2 = self.df['price_per_m2'].mean()
        print(f" Average price per m²: {avg_price_m2:.2f}€")
    
    def clean_rooms(self):
        """Clean room data"""
        print("\n🛏️  Cleaning room data...")
        
        # Fill missing rooms with 1 (studio)
        self.df['rooms'] = self.df['rooms'].fillna(1)
        
        # Fill missing bedrooms with 0
        self.df['bedrooms'] = self.df['bedrooms'].fillna(0)
        
        # Ensure bedrooms <= rooms
        self.df['bedrooms'] = self.df.apply(
            lambda row: min(row['bedrooms'], row['rooms'] - 1) if row['rooms'] > 1 else 0,
            axis=1
        )
        
        print(f" Room data cleaned")
    
    def clean_dates(self):
        """Clean publication dates"""
        print("\n Cleaning dates...")
        
        # Convert ISO dates to simple format
        self.df['publication_date'] = pd.to_datetime(
            self.df['publication_date'], 
            errors='coerce'
        ).dt.strftime('%Y-%m-%d')
        
        # Replace invalid dates (1970-01-01) with NaN
        self.df.loc[self.df['publication_date'] == '1970-01-01', 'publication_date'] = None
        
        print(f" Dates formatted")
    
    def filter_main_cities(self):
        """Keep all cities - no filtering"""
        print("\n  Keeping all cities...")
        
        city_count = self.df['city'].nunique()
        print(f"  Total cities in dataset: {city_count}")
        print(f" No city filtering applied (keeping all {len(self.df)} records)")
    
    def add_region(self):
        """Add region based on postal code"""
        print("\n🗺️  Adding regions...")
        
        def get_region(postal_code):
            """Map postal code to region"""
            if pd.isna(postal_code):
                return 'Unknown'
            
            # Get first 2 digits
            dept = str(postal_code)[:2]
            
            regions = {
                '75': 'Île-de-France',
                '77': 'Île-de-France', '78': 'Île-de-France', 
                '91': 'Île-de-France', '92': 'Île-de-France',
                '93': 'Île-de-France', '94': 'Île-de-France', '95': 'Île-de-France',
                '69': 'Auvergne-Rhône-Alpes', '01': 'Auvergne-Rhône-Alpes',
                '07': 'Auvergne-Rhône-Alpes', '26': 'Auvergne-Rhône-Alpes',
                '38': 'Auvergne-Rhône-Alpes', '42': 'Auvergne-Rhône-Alpes',
                '73': 'Auvergne-Rhône-Alpes', '74': 'Auvergne-Rhône-Alpes',
                '13': 'Provence-Alpes-Côte d\'Azur', '04': 'Provence-Alpes-Côte d\'Azur',
                '05': 'Provence-Alpes-Côte d\'Azur', '06': 'Provence-Alpes-Côte d\'Azur',
                '83': 'Provence-Alpes-Côte d\'Azur', '84': 'Provence-Alpes-Côte d\'Azur',
                '31': 'Occitanie', '09': 'Occitanie', '11': 'Occitanie',
                '12': 'Occitanie', '30': 'Occitanie', '32': 'Occitanie',
                '34': 'Occitanie', '46': 'Occitanie', '48': 'Occitanie',
                '65': 'Occitanie', '66': 'Occitanie', '81': 'Occitanie', '82': 'Occitanie',
                '44': 'Pays de la Loire', '49': 'Pays de la Loire',
                '53': 'Pays de la Loire', '72': 'Pays de la Loire', '85': 'Pays de la Loire',
                '33': 'Nouvelle-Aquitaine', '16': 'Nouvelle-Aquitaine',
                '17': 'Nouvelle-Aquitaine', '19': 'Nouvelle-Aquitaine',
                '23': 'Nouvelle-Aquitaine', '24': 'Nouvelle-Aquitaine',
                '40': 'Nouvelle-Aquitaine', '47': 'Nouvelle-Aquitaine',
                '64': 'Nouvelle-Aquitaine', '79': 'Nouvelle-Aquitaine', '86': 'Nouvelle-Aquitaine', '87': 'Nouvelle-Aquitaine',
            }
            
            return regions.get(dept, 'Other')
        
        self.df['region'] = self.df['postal_code'].apply(get_region)
        print(f" Regions added")
        print(f"   Distribution: {self.df['region'].value_counts().to_dict()}")

    def geocode_missing_locations(self, max_rows=500):
        """
        Compléter les localisations manquantes avec Nominatim :
        - Ajoute latitude / longitude
        - Complète city / postal_code si possible
        """
        print("\n Geocoding missing locations with Nominatim...")

        if self.df is None or self.df.empty:
            print("ℹ  No data loaded, skipping geocoding.")
            return

        # S'assurer que les colonnes latitude / longitude existent
        if 'latitude' not in self.df.columns:
            self.df['latitude'] = np.nan
        if 'longitude' not in self.df.columns:
            self.df['longitude'] = np.nan

        # Lignes à traiter :
        # - city manquante ou vide
        # - OU postal_code manquant ou vide
        # - OU latitude/longitude manquantes
        mask = (
            self.df['city'].isna() |
            (self.df['city'].astype(str).str.strip() == "") |
            self.df['postal_code'].isna() |
            (self.df['postal_code'].astype(str).str.strip() == "") |
            self.df['latitude'].isna() |
            self.df['longitude'].isna()
        )

        candidates = self.df[mask].copy()

        if candidates.empty:
            print("  No rows need geocoding.")
            return

        # On limite pour respecter Nominatim (modifiable selon ton usage)
        total_to_geocode = len(candidates)
        candidates = candidates.head(max_rows)
        print(f"  {total_to_geocode} rows need geocoding, processing first {len(candidates)}.")

        updated = 0

        for idx, row in candidates.iterrows():
            address = self._build_address(row)
            if not address:
                # On n'a rien à envoyer à Nominatim
                continue

            # Cache : si on a déjà géocodé cette adresse, on réutilise
            if address in self.geocode_cache:
                location = self.geocode_cache[address]
            else:
                location = self.geocode(address)
                self.geocode_cache[address] = location

            if not location:
                continue

            # Coords
            lat = location.latitude
            lon = location.longitude

            # Adresse structurée retournée par Nominatim
            addr = location.raw.get("address", {})

            city = row['city']
            postal_code = row['postal_code']

            # Compléter city si manquante
            if (pd.isna(city) or str(city).strip() == ""):
                city = (
                    addr.get('city') or
                    addr.get('town') or
                    addr.get('village') or
                    addr.get('municipality')
                )

            # Compléter postal_code si manquant
            if (pd.isna(postal_code) or str(postal_code).strip() == ""):
                postal_code = addr.get('postcode')

            # Écriture dans le DataFrame principal
            self.df.at[idx, 'latitude'] = lat
            self.df.at[idx, 'longitude'] = lon
            if city:
                self.df.at[idx, 'city'] = city
            if postal_code:
                self.df.at[idx, 'postal_code'] = postal_code

            updated += 1

        print(f" Geocoding finished. Updated {updated} rows.")
    
    def reorder_columns(self):
        """Reorder columns for better readability"""
        print("\n Reordering columns...")
        
        column_order = [
            'id', 'title', 'city', 'postal_code', 'region',
            'price', 'charges', 'surface', 'price_per_m2',
            'rooms', 'bedrooms', 'property_type', 'ad_type',
            'floor', 'has_elevator', 'is_furnished', 'parking',
            'publication_date', 'scraped_date', 'url', 'source'
        ]
        
        # Only reorder columns that exist
        existing_cols = [col for col in column_order if col in self.df.columns]
        self.df = self.df[existing_cols]
        
        print(f" Columns reordered")
    
    def save_clean_data(self):
        """Save cleaned data to CSV"""
        print("\n Saving clean data...")
        
        # Create directory if not exists
        os.makedirs(DATA_PROCESSED_PATH, exist_ok=True)
        
        # Save
        self.df.to_csv(self.clean_file, index=False)
        
        print(f" Clean data saved: {self.clean_file}")
        print(f"   Final records: {len(self.df)}")
    
    def print_summary(self):
        """Print summary statistics"""
        print("\n" + "=" * 60)
        print(" CLEANING SUMMARY")
        print("=" * 60)
        
        print(f"\n Property Types:")
        print(self.df['property_type'].value_counts())
        
        print(f"\n Price Statistics:")
        print(f"   Mean: {self.df['price'].mean():.2f}€")
        print(f"   Median: {self.df['price'].median():.2f}€")
        print(f"   Min: {self.df['price'].min():.2f}€")
        print(f"   Max: {self.df['price'].max():.2f}€")
        
        print(f"\n Surface Statistics:")
        print(f"   Mean: {self.df['surface'].mean():.2f}m²")
        print(f"   Median: {self.df['surface'].median():.2f}m²")
        
        print(f"\n Price per m² Statistics:")
        print(f"   Mean: {self.df['price_per_m2'].mean():.2f}€/m²")
        print(f"   Median: {self.df['price_per_m2'].median():.2f}€/m²")
        
        print(f"\n  Top 10 Cities:")
        print(self.df['city'].value_counts().head(10))
        
        print("\n" + "=" * 60)
    
    def run(self):
        """Execute complete cleaning pipeline"""
        print(" Starting data cleaning pipeline")
        print("=" * 60)
        
        # Load data
        if not self.load_data():
            return
        
        # Cleaning steps
        self.remove_duplicates()
        self.filter_property_types()
        self.clean_prices()
        self.clean_surface()
        self.calculate_price_per_m2()
        self.clean_rooms()
        self.clean_dates()
        self.filter_main_cities()
        self.add_region()
        self.reorder_columns()
        
        # Save and summarize
        self.save_clean_data()
        self.print_summary()
        
        print("\n Cleaning complete!")


def main():
    """Main entry point"""
    cleaner = DataCleaner()
    cleaner.run()


if __name__ == '__main__':
    main()
