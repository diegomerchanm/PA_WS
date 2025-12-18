# src/scraper.py

import os
import sys
import time
import json
import requests
import pandas as pd
from datetime import datetime
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import (
    SCRAPER_CONFIG, BIENICI_CONFIG, BIENICI_SEARCH, 
    HEADERS, DATA_RAW_PATH, BIENICI_RAW_FILE
)

class BienciScraper:
    """Scraper for Bien'ici real estate website"""
    
    def __init__(self):
        self.base_url = BIENICI_CONFIG['base_url']
        self.delay = BIENICI_CONFIG['request_delay']
        self.timeout = BIENICI_CONFIG['timeout']
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        
        # Load existing IDs to avoid duplicates
        self.existing_ids = self._load_existing_ids()
        self.new_listings = []
        
    def _load_existing_ids(self):
        """Load already scraped listing IDs"""
        filepath = os.path.join(DATA_RAW_PATH, BIENICI_RAW_FILE)
        
        if not SCRAPER_CONFIG['check_duplicates']:
            return set()
        
        if os.path.exists(filepath):
            try:
                df = pd.read_csv(filepath)
                ids = set(df['id'].tolist())
                print(f"  Loaded {len(ids)} existing IDs")
                return ids
            except Exception as e:
                print(f" Error loading existing IDs: {e}")
                return set()
        
        return set()
    
    def _build_api_params(self, city, page):
        """Build API request parameters"""
        
        offset = (page - 1) * BIENICI_CONFIG['results_per_page']
        size = BIENICI_CONFIG['results_per_page']
        filter_type = BIENICI_SEARCH['filter_type']
        
        # Format exactly like working example from Stack Overflow
        filters_str = (
            f'{{"size":{size},"from":{offset},"filterType":"{filter_type}",'
            f'"newProperty":false,"page":{page},"resultsPerPage":{size},'
            f'"maxAuthorizedResults":2400,"sortBy":"relevance","sortOrder":"desc",'
            f'"onTheMarket":[true],"showAllModels":false,'
            f'"blurInfoType":["disk","exact"]}}'
        )
        
        return {"filters": filters_str}
    
    def _fetch_page(self, city, page):
        """Fetch one page from API"""
        params = self._build_api_params(city, page)
        
        try:
            response = self.session.get(
                self.base_url,
                params=params,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            # Add delay to be respectful
            time.sleep(self.delay)
            
            return response.json()
        
        except requests.exceptions.RequestException as e:
            print(f" Error fetching page {page}: {e}")
            return None
    
    def _parse_listing(self, raw_data):
        """Parse API response into our format"""
        # Create unique ID
        listing_id = f"bienici-{raw_data.get('id', 'unknown')}"
        
        # Skip if already scraped
        if listing_id in self.existing_ids:
            return None
        
        # Extract data with safe defaults
        listing = {
            'id': listing_id,
            'title': raw_data.get('title', ''),
            'price': raw_data.get('price'),
            'charges': raw_data.get('charges'),
            'surface': raw_data.get('surfaceArea'),
            'rooms': raw_data.get('roomsQuantity'),
            'bedrooms': raw_data.get('bedroomsQuantity'),
            'city': raw_data.get('city', ''),
            'postal_code': raw_data.get('postalCode', ''),
            'property_type': raw_data.get('propertyType', ''),
            'ad_type': BIENICI_SEARCH['filter_type'],
            'floor': raw_data.get('floor'),
            'has_elevator': raw_data.get('hasElevator', False),
            'is_furnished': raw_data.get('isFurnished', False),
            'parking': raw_data.get('parkingPlacesCount', 0),
            'publication_date': raw_data.get('publicationDate', ''),
            'url': f"https://www.bienici.com/annonce/{raw_data.get('id', '')}",
            'scraped_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'source': 'bienici'
        }
        
        return listing
    
    def _save_to_csv(self):
        """Save scraped listings to CSV"""
        if not self.new_listings:
            print(" No new listings to save")
            return
        
        filepath = os.path.join(DATA_RAW_PATH, BIENICI_RAW_FILE)
        
        # Create directory if not exists
        os.makedirs(DATA_RAW_PATH, exist_ok=True)
        
        # Convert to DataFrame
        df = pd.DataFrame(self.new_listings)
        
        # Check if file exists for append mode
        file_exists = os.path.exists(filepath)
        
        # Save
        df.to_csv(
            filepath,
            mode='a' if (file_exists and SCRAPER_CONFIG['append_mode']) else 'w',
            header=not (file_exists and SCRAPER_CONFIG['append_mode']),
            index=False
        )
        
        print(f"{len(self.new_listings)} new listings saved to {filepath}")
    
    def run(self, max_pages=None):
        """Run the scraper"""
        max_pages = max_pages or BIENICI_CONFIG['max_pages']
        
        print(f"🚀 Starting Bien'ici scraper")
        print(f"📄 Max pages: {max_pages}")
        print(f"🔍 Filter: {BIENICI_SEARCH['filter_type']}")
        print(f"📍 Scraping all France (filter by city later in cleaner)")
        print("-" * 50)
        
        for page in tqdm(range(1, max_pages + 1), desc="Pages"):
            data = self._fetch_page(None, page)
            
            if not data or 'realEstateAds' not in data:
                print(f"\n No data on page {page}, stopping")
                break
            
            ads = data.get('realEstateAds', [])
            
            if not ads:
                print(f"\n No more listings on page {page}")
                break
            
            # Parse each listing
            for ad in ads:
                listing = self._parse_listing(ad)
                if listing:
                    self.new_listings.append(listing)
                    self.existing_ids.add(listing['id'])
        
        # Save all at once
        self._save_to_csv()
        
        print("\n" + "=" * 50)
        print(f"Scraping complete!")
        print(f"Total new listings: {len(self.new_listings)}")
        print(f" File: {os.path.join(DATA_RAW_PATH, BIENICI_RAW_FILE)}")
        print("=" * 50)


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Scrape Bien\'ici real estate data')
    parser.add_argument('--pages', type=int, help='Max pages to scrape')
    
    args = parser.parse_args()
    
    scraper = BienciScraper()
    scraper.run(max_pages=args.pages)


if __name__ == '__main__':
    main()
