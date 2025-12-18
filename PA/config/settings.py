# config/settings.py

# Scraper behavior
SCRAPER_CONFIG = {
    'append_mode': True,              # Append to existing CSV
    'check_duplicates': True,         # Skip already scraped IDs
    'create_backups': False,          # Save dated backups
    'max_file_size_mb': 50,          # CSV size limit
}

# Bien'ici API settings
BIENICI_CONFIG = {
    'base_url': 'https://www.bienici.com/realEstateAds.json',
    'results_per_page': 24,
    'max_pages': 10,
    'request_delay': 2,               # Seconds between requests
    'timeout': 30,
}

# Search parameters
BIENICI_SEARCH = {
    'cities': ['paris', 'lyon', 'marseille'],
    'filter_type': 'rent',            # 'rent' or 'buy'
    'property_types': ['flat', 'house'],
    'max_price': None,
    'min_rooms': None,
}

# HTTP headers
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Accept': 'application/json',
    'Accept-Language': 'fr-FR,fr;q=0.9',
    'Referer': 'https://www.bienici.com/',
}

# File paths
DATA_RAW_PATH = 'data/raw/'
DATA_PROCESSED_PATH = 'data/processed/'
BIENICI_RAW_FILE = 'bienici_raw.csv'

# CSV columns to extract
LISTING_FIELDS = [
    'id', 'title', 'price', 'charges', 'surface', 'rooms', 'bedrooms',
    'city', 'postal_code', 'property_type', 'ad_type', 'floor',
    'has_elevator', 'is_furnished', 'parking', 'publication_date',
    'url', 'scraped_date', 'source'
]