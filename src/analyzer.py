# src/analyzer.py

import os
import sys
import json
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Add parent directory for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import DATA_PROCESSED_PATH


class RealEstateAnalyzer:
    """Advanced statistical analysis of real estate data"""
    
    def __init__(self):
        self.data_file = os.path.join(DATA_PROCESSED_PATH, 'immobilier_clean.csv')
        self.output_file = os.path.join(DATA_PROCESSED_PATH, 'analysis_results.json')
        self.df = None
        self.results = {}
        
    def load_data(self):
        """Load clean data"""
        print("📂 Loading clean data...")
        
        if not os.path.exists(self.data_file):
            print(f"❌ File not found: {self.data_file}")
            return False
        
        self.df = pd.read_csv(self.data_file)
        print(f"✅ Loaded {len(self.df)} records")
        return True
    
    # ==================== LEVEL 1: DESCRIPTIVE STATISTICS ====================
    
    def analyze_price_distribution(self):
        """Comprehensive price distribution analysis"""
        print("\n💰 Analyzing price distribution...")
        
        prices = self.df['price']
        
        self.results['price_distribution'] = {
            'mean': float(prices.mean()),
            'median': float(prices.median()),
            'mode': float(prices.mode()[0]) if len(prices.mode()) > 0 else None,
            'std': float(prices.std()),
            'min': float(prices.min()),
            'max': float(prices.max()),
            'percentiles': {
                '25': float(prices.quantile(0.25)),
                '50': float(prices.quantile(0.50)),
                '75': float(prices.quantile(0.75)),
                '90': float(prices.quantile(0.90)),
                '95': float(prices.quantile(0.95)),
                '99': float(prices.quantile(0.99))
            },
            'iqr': float(prices.quantile(0.75) - prices.quantile(0.25)),
            'coefficient_variation': float(prices.std() / prices.mean()),
            'skewness': float(prices.skew()),
            'kurtosis': float(prices.kurtosis())
        }
        
        print(f"   Mean: {prices.mean():.2f}€")
        print(f"   Median: {prices.median():.2f}€")
        print(f"   Std Dev: {prices.std():.2f}€")
    
    def analyze_price_per_m2(self):
        """Price per m² analysis"""
        print("\n📐 Analyzing price per m²...")
        
        price_m2 = self.df['price_per_m2']
        
        self.results['price_per_m2'] = {
            'mean': float(price_m2.mean()),
            'median': float(price_m2.median()),
            'std': float(price_m2.std()),
            'min': float(price_m2.min()),
            'max': float(price_m2.max()),
            'percentiles': {
                '25': float(price_m2.quantile(0.25)),
                '50': float(price_m2.quantile(0.50)),
                '75': float(price_m2.quantile(0.75)),
                '90': float(price_m2.quantile(0.90))
            }
        }
        
        print(f"   Mean: {price_m2.mean():.2f}€/m²")
        print(f"   Median: {price_m2.median():.2f}€/m²")
    
    def analyze_surface_distribution(self):
        """Surface area analysis"""
        print("\n🏠 Analyzing surface areas...")
        
        surface = self.df['surface']
        
        self.results['surface_distribution'] = {
            'mean': float(surface.mean()),
            'median': float(surface.median()),
            'std': float(surface.std()),
            'min': float(surface.min()),
            'max': float(surface.max()),
            'by_property_type': {}
        }
        
        # By property type
        for prop_type in self.df['property_type'].unique():
            subset = self.df[self.df['property_type'] == prop_type]['surface']
            self.results['surface_distribution']['by_property_type'][prop_type] = {
                'mean': float(subset.mean()),
                'median': float(subset.median()),
                'count': int(len(subset))
            }
        
        print(f"   Mean: {surface.mean():.2f}m²")
    
    # ==================== LEVEL 2: CORRELATIONS ====================
    
    def analyze_correlations(self):
        """Correlation analysis"""
        print("\n🔗 Analyzing correlations...")
        
        # Select numeric columns
        numeric_cols = ['price', 'surface', 'rooms', 'bedrooms', 'price_per_m2', 'floor']
        corr_data = self.df[numeric_cols].dropna()
        
        # Correlation matrix
        corr_matrix = corr_data.corr()
        
        self.results['correlations'] = {
            'price_vs_surface': float(corr_matrix.loc['price', 'surface']),
            'price_vs_rooms': float(corr_matrix.loc['price', 'rooms']),
            'price_vs_floor': float(corr_matrix.loc['price', 'floor']) if 'floor' in corr_matrix else None,
            'surface_vs_rooms': float(corr_matrix.loc['surface', 'rooms']),
            'matrix': corr_matrix.to_dict()
        }
        
        print(f"   Price vs Surface: {corr_matrix.loc['price', 'surface']:.3f}")
        print(f"   Price vs Rooms: {corr_matrix.loc['price', 'rooms']:.3f}")
    
    # ==================== LEVEL 3: GEOGRAPHIC ANALYSIS ====================
    
    def analyze_by_city(self):
        """City-level analysis"""
        print("\n🏙️  Analyzing by city...")
        
        city_stats = self.df.groupby('city').agg({
            'price': ['mean', 'median', 'count'],
            'price_per_m2': ['mean', 'median'],
            'surface': 'mean'
        }).round(2)
        
        # Top 10 most expensive (absolute price)
        top_price = city_stats.nlargest(10, ('price', 'mean'))
        
        # Top 10 most expensive (price per m²)
        top_price_m2 = city_stats.nlargest(10, ('price_per_m2', 'mean'))
        
        # Cities with most listings
        top_volume = city_stats.nlargest(10, ('price', 'count'))
        
        self.results['city_analysis'] = {
            'top_10_price': [
                {
                    'city': city,
                    'avg_price': float(row[('price', 'mean')]),
                    'median_price': float(row[('price', 'median')]),
                    'count': int(row[('price', 'count')])
                }
                for city, row in top_price.iterrows()
            ],
            'top_10_price_per_m2': [
                {
                    'city': city,
                    'avg_price_m2': float(row[('price_per_m2', 'mean')]),
                    'median_price_m2': float(row[('price_per_m2', 'median')]),
                    'count': int(row[('price', 'count')])
                }
                for city, row in top_price_m2.iterrows()
            ],
            'top_10_volume': [
                {
                    'city': city,
                    'count': int(row[('price', 'count')]),
                    'avg_price': float(row[('price', 'mean')])
                }
                for city, row in top_volume.iterrows()
            ]
        }
        
        print(f"   Analyzed {len(city_stats)} cities")
    
    def analyze_by_region(self):
        """Region-level analysis"""
        print("\n🗺️  Analyzing by region...")
        
        region_stats = self.df.groupby('region').agg({
            'price': ['mean', 'median', 'count'],
            'price_per_m2': ['mean', 'median'],
            'surface': 'mean'
        }).round(2)
        
        self.results['region_analysis'] = {
            region: {
                'avg_price': float(row[('price', 'mean')]),
                'median_price': float(row[('price', 'median')]),
                'avg_price_m2': float(row[('price_per_m2', 'mean')]),
                'avg_surface': float(row[('surface', 'mean')]),
                'count': int(row[('price', 'count')]),
                'market_share': float(row[('price', 'count')] / len(self.df) * 100)
            }
            for region, row in region_stats.iterrows()
        }
        
        print(f"   Analyzed {len(region_stats)} regions")
    
    def compare_property_types(self):
        """Compare flats vs houses"""
        print("\n🏘️  Comparing property types...")
        
        type_stats = self.df.groupby('property_type').agg({
            'price': ['mean', 'median', 'count'],
            'price_per_m2': ['mean', 'median'],
            'surface': ['mean', 'median'],
            'rooms': 'mean'
        }).round(2)
        
        self.results['property_type_comparison'] = {
            prop_type: {
                'avg_price': float(row[('price', 'mean')]),
                'median_price': float(row[('price', 'median')]),
                'avg_price_m2': float(row[('price_per_m2', 'mean')]),
                'avg_surface': float(row[('surface', 'mean')]),
                'avg_rooms': float(row[('rooms', 'mean')]),
                'count': int(row[('price', 'count')]),
                'market_share': float(row[('price', 'count')] / len(self.df) * 100)
            }
            for prop_type, row in type_stats.iterrows()
        }
        
        print(f"   Flat: {type_stats.loc['flat', ('price', 'mean')]:.2f}€")
        print(f"   House: {type_stats.loc['house', ('price', 'mean')]:.2f}€")
    
    # ==================== LEVEL 4: SEGMENTATION ====================
    
    def segment_by_price_range(self):
        """Segment properties by price range"""
        print("\n💎 Segmenting by price range...")
        
        # Define segments
        self.df['price_segment'] = pd.cut(
            self.df['price'],
            bins=[0, 600, 1200, 2500, 10000],
            labels=['Economic', 'Medium', 'Premium', 'Luxury']
        )
        
        segment_stats = self.df.groupby('price_segment').agg({
            'price': ['mean', 'count'],
            'price_per_m2': 'mean',
            'surface': 'mean',
            'rooms': 'mean'
        }).round(2)
        
        self.results['price_segmentation'] = {
            segment: {
                'avg_price': float(row[('price', 'mean')]),
                'avg_price_m2': float(row[('price_per_m2', 'mean')]),
                'avg_surface': float(row[('surface', 'mean')]),
                'avg_rooms': float(row[('rooms', 'mean')]),
                'count': int(row[('price', 'count')]),
                'market_share': float(row[('price', 'count')] / len(self.df) * 100)
            }
            for segment, row in segment_stats.iterrows()
        }
        
        print(f"   Segments created: {len(segment_stats)}")
    
    def segment_by_rooms(self):
        """Segment by number of rooms"""
        print("\n🛏️  Segmenting by rooms...")
        
        rooms_stats = self.df.groupby('rooms').agg({
            'price': ['mean', 'median', 'count'],
            'price_per_m2': 'mean',
            'surface': 'mean'
        }).round(2)
        
        self.results['rooms_segmentation'] = {
            f'{int(rooms)}_rooms': {
                'avg_price': float(row[('price', 'mean')]),
                'median_price': float(row[('price', 'median')]),
                'avg_price_m2': float(row[('price_per_m2', 'mean')]),
                'avg_surface': float(row[('surface', 'mean')]),
                'count': int(row[('price', 'count')]),
                'price_per_room': float(row[('price', 'mean')] / rooms)
            }
            for rooms, row in rooms_stats.iterrows()
        }
        
        print(f"   Analyzed {len(rooms_stats)} room categories")
    
    def analyze_amenities_impact(self):
        """Analyze impact of amenities on price"""
        print("\n🎯 Analyzing amenities impact...")
        
        # Elevator impact
        elevator_stats = self.df.groupby('has_elevator')['price'].agg(['mean', 'count'])
        
        # Furnished impact
        furnished_stats = self.df.groupby('is_furnished')['price'].agg(['mean', 'count'])
        
        # Parking impact
        parking_stats = self.df[self.df['parking'] > 0]['price'].mean()
        no_parking_stats = self.df[self.df['parking'] == 0]['price'].mean()
        
        self.results['amenities_impact'] = {
            'elevator': {
                'with_elevator': float(elevator_stats.loc[True, 'mean']) if True in elevator_stats.index else None,
                'without_elevator': float(elevator_stats.loc[False, 'mean']) if False in elevator_stats.index else None,
                'price_premium_%': float((elevator_stats.loc[True, 'mean'] / elevator_stats.loc[False, 'mean'] - 1) * 100) if (True in elevator_stats.index and False in elevator_stats.index) else None
            },
            'furnished': {
                'furnished': float(furnished_stats.loc[True, 'mean']) if True in furnished_stats.index else None,
                'unfurnished': float(furnished_stats.loc[False, 'mean']) if False in furnished_stats.index else None,
                'price_premium_%': float((furnished_stats.loc[True, 'mean'] / furnished_stats.loc[False, 'mean'] - 1) * 100) if (True in furnished_stats.index and False in furnished_stats.index) else None
            },
            'parking': {
                'with_parking': float(parking_stats),
                'without_parking': float(no_parking_stats),
                'price_premium_%': float((parking_stats / no_parking_stats - 1) * 100)
            }
        }
        
        print(f"   Elevator premium: {self.results['amenities_impact']['elevator'].get('price_premium_%', 0):.1f}%")
    
    # ==================== LEVEL 5: ADVANCED INSIGHTS ====================
    
    def calculate_value_score(self):
        """Calculate custom value score"""
        print("\n⭐ Calculating value scores...")
        
        # Normalize factors
        avg_price_by_city = self.df.groupby('city')['price'].transform('mean')
        
        # Value Score = (Surface × Rooms) / (Price × City_Factor)
        city_factor = self.df['price'] / avg_price_by_city
        self.df['value_score'] = (
            (self.df['surface'] * self.df['rooms']) / 
            (self.df['price'] * city_factor)
        )
        
        # Top 20 best values
        top_values = self.df.nlargest(20, 'value_score')[
            ['city', 'price', 'surface', 'rooms', 'price_per_m2', 'value_score', 'url']
        ]
        
        self.results['best_value_properties'] = [
            {
                'city': row['city'],
                'price': float(row['price']),
                'surface': float(row['surface']),
                'rooms': int(row['rooms']),
                'price_per_m2': float(row['price_per_m2']),
                'value_score': float(row['value_score']),
                'url': row['url']
            }
            for _, row in top_values.iterrows()
        ]
        
        print(f"   Top value score: {self.df['value_score'].max():.2f}")
    
    def identify_outliers(self):
        """Identify statistical outliers"""
        print("\n🔍 Identifying outliers...")
        
        # Most expensive properties
        top_expensive = self.df.nlargest(10, 'price')[
            ['city', 'price', 'surface', 'rooms', 'price_per_m2', 'url']
        ]
        
        # Highest price per m²
        top_price_m2 = self.df.nlargest(10, 'price_per_m2')[
            ['city', 'price', 'surface', 'price_per_m2', 'url']
        ]
        
        # Suspiciously cheap (below 2 std devs)
        mean_price = self.df['price'].mean()
        std_price = self.df['price'].std()
        suspiciously_cheap = self.df[self.df['price'] < (mean_price - 2 * std_price)]
        
        self.results['outliers'] = {
            'most_expensive': [
                {
                    'city': row['city'],
                    'price': float(row['price']),
                    'surface': float(row['surface']),
                    'price_per_m2': float(row['price_per_m2']),
                    'url': row['url']
                }
                for _, row in top_expensive.iterrows()
            ],
            'highest_price_per_m2': [
                {
                    'city': row['city'],
                    'price': float(row['price']),
                    'surface': float(row['surface']),
                    'price_per_m2': float(row['price_per_m2']),
                    'url': row['url']
                }
                for _, row in top_price_m2.iterrows()
            ],
            'suspiciously_cheap_count': int(len(suspiciously_cheap))
        }
        
        print(f"   Found {len(suspiciously_cheap)} suspiciously cheap properties")
    
    def cluster_properties(self):
        """K-means clustering of properties"""
        print("\n🎨 Clustering properties...")
        
        # Prepare features for clustering
        features = self.df[['price', 'surface', 'rooms', 'price_per_m2']].dropna()
        
        # Standardize
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # K-means with 4 clusters
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(features_scaled)
        
        # Add cluster to dataframe
        self.df.loc[features.index, 'cluster'] = clusters
        
        # Analyze clusters
        cluster_stats = self.df.groupby('cluster').agg({
            'price': 'mean',
            'surface': 'mean',
            'rooms': 'mean',
            'price_per_m2': 'mean',
            'city': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'N/A'
        }).round(2)
        
        # Name clusters based on characteristics
        cluster_names = []
        for idx, row in cluster_stats.iterrows():
            if row['price'] > 2000:
                name = "Luxury Properties"
            elif row['price_per_m2'] > 20 and row['surface'] < 50:
                name = "Urban Studios"
            elif row['rooms'] >= 4:
                name = "Family Homes"
            else:
                name = "Standard Rentals"
            cluster_names.append(name)
        
        self.results['clusters'] = {
            f'cluster_{int(i)}': {
                'name': cluster_names[idx],
                'avg_price': float(row['price']),
                'avg_surface': float(row['surface']),
                'avg_rooms': float(row['rooms']),
                'avg_price_m2': float(row['price_per_m2']),
                'typical_city': row['city'],
                'count': int((self.df['cluster'] == i).sum())
            }
            for idx, (i, row) in enumerate(cluster_stats.iterrows())
        }
        
        print(f"   Created 4 clusters")
    
    def compare_paris_vs_rest(self):
        """Epic comparison: Paris vs Rest of France"""
        print("\n⚔️  Paris vs Rest of France...")
        
        # Define Paris (includes arrondissements)
        paris_mask = self.df['city'].str.contains('Paris', case=False, na=False)
        
        paris = self.df[paris_mask]
        rest = self.df[~paris_mask]
        
        self.results['paris_vs_rest'] = {
            'paris': {
                'count': int(len(paris)),
                'market_share_%': float(len(paris) / len(self.df) * 100),
                'avg_price': float(paris['price'].mean()),
                'median_price': float(paris['price'].median()),
                'avg_price_m2': float(paris['price_per_m2'].mean()),
                'avg_surface': float(paris['surface'].mean())
            },
            'rest_of_france': {
                'count': int(len(rest)),
                'market_share_%': float(len(rest) / len(self.df) * 100),
                'avg_price': float(rest['price'].mean()),
                'median_price': float(rest['price'].median()),
                'avg_price_m2': float(rest['price_per_m2'].mean()),
                'avg_surface': float(rest['surface'].mean())
            },
            'comparison': {
                'price_difference_%': float((paris['price'].mean() / rest['price'].mean() - 1) * 100),
                'price_m2_difference_%': float((paris['price_per_m2'].mean() / rest['price_per_m2'].mean() - 1) * 100),
                'surface_difference_%': float((paris['surface'].mean() / rest['surface'].mean() - 1) * 100)
            }
        }
        
        print(f"   Paris is {self.results['paris_vs_rest']['comparison']['price_difference_%']:.1f}% more expensive")
    
    # ==================== SAVE RESULTS ====================
    
    def save_results(self):
        """Save all results to JSON"""
        print("\n💾 Saving results...")
        
        # Add metadata
        self.results['metadata'] = {
            'total_properties': int(len(self.df)),
            'analysis_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'data_source': 'bienici.com'
        }
        
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Results saved: {self.output_file}")
    
    def print_summary(self):
        """Print executive summary"""
        print("\n" + "=" * 70)
        print("📊 EXECUTIVE SUMMARY")
        print("=" * 70)
        
        print(f"\n📈 Market Overview:")
        print(f"   Total Properties: {len(self.df):,}")
        print(f"   Average Price: {self.df['price'].mean():.2f}€")
        print(f"   Median Price: {self.df['price'].median():.2f}€")
        print(f"   Average €/m²: {self.df['price_per_m2'].mean():.2f}€")
        
        print(f"\n🏆 Top 3 Most Expensive Cities:")
        for i, city_data in enumerate(self.results['city_analysis']['top_10_price'][:3], 1):
            print(f"   {i}. {city_data['city']}: {city_data['avg_price']:.2f}€")
        
        print(f"\n🎯 Market Segments:")
        for segment, data in self.results['price_segmentation'].items():
            print(f"   {segment}: {data['market_share']:.1f}% ({data['count']} properties)")
        
        print("\n" + "=" * 70)
    
    def run(self):
        """Execute complete analysis pipeline"""
        print("🚀 Starting EPIC Real Estate Analysis")
        print("=" * 70)
        
        # Load data
        if not self.load_data():
            return
        
        # Level 1: Descriptive Statistics
        self.analyze_price_distribution()
        self.analyze_price_per_m2()
        self.analyze_surface_distribution()
        
        # Level 2: Correlations
        self.analyze_correlations()
        
        # Level 3: Geographic Analysis
        self.analyze_by_city()
        self.analyze_by_region()
        self.compare_property_types()
        
        # Level 4: Segmentation
        self.segment_by_price_range()
        self.segment_by_rooms()
        self.analyze_amenities_impact()
        
        # Level 5: Advanced Insights
        self.calculate_value_score()
        self.identify_outliers()
        self.cluster_properties()
        self.compare_paris_vs_rest()
        
        # Save and summarize
        self.save_results()
        self.print_summary()
        
        print("\n🎉 Analysis complete!")


def main():
    """Main entry point"""
    analyzer = RealEstateAnalyzer()
    analyzer.run()


if __name__ == '__main__':
    main()