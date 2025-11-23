"""
ETL Preprocessing Pipeline

Data standardization:
- Normalize entity names (fuzzy matching for variations)
- Standardize date formats
- Clean location names
- Resolve organization name variations

Data enrichment:
- Add entity metadata (country codes, organization types)
- Calculate derived metrics
- Create surrogate keys
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
from datetime import datetime
from collections import defaultdict
import re

# Add scripts directory to path
sys.path.append(str(Path(__file__).parent))
from utils import (
    load_config, load_json, save_json,
    print_section, print_stats
)


class ETLPreprocessor:
    """Preprocess extracted data for data warehouse"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize preprocessor"""
        self.config = load_config(config_path)
        self.staging_dir = Path("data/staging")
        self.preprocessed_dir = Path("data/staging/preprocessed")
        self.preprocessed_dir.mkdir(parents=True, exist_ok=True)
        
        # Entity normalization mappings
        self.entity_normalization = defaultdict(str)
        self.surrogate_keys = {
            'speech': {},
            'person': {},
            'organization': {},
            'location': {},
            'date': {},
            'topic': {},
            'emotion': {}
        }
        
        # Country code mapping (simplified)
        self.country_codes = {
            'united states': 'US', 'usa': 'US', 'america': 'US',
            'china': 'CN', 'russia': 'RU', 'russian federation': 'RU',
            'iran': 'IR', 'north korea': 'KP', 'south korea': 'KR',
            'united kingdom': 'GB', 'uk': 'GB', 'britain': 'GB',
            'france': 'FR', 'germany': 'DE', 'italy': 'IT',
            'japan': 'JP', 'india': 'IN', 'israel': 'IL',
            'saudi arabia': 'SA', 'uae': 'AE', 'qatar': 'QA'
        }
    
    def find_latest_staging_file(self, pattern: str) -> Optional[Path]:
        """Find latest staging file matching pattern"""
        files = list(self.staging_dir.glob(pattern))
        if files:
            return max(files, key=lambda p: p.stat().st_mtime)
        return None
    
    def normalize_entity_name(self, entity_name: str, entity_type: str) -> str:
        """Normalize entity name for consistency"""
        if not entity_name:
            return ""
        
        # Basic cleaning
        normalized = entity_name.strip()
        normalized = re.sub(r'\s+', ' ', normalized)  # Multiple spaces to single
        
        # Title case for proper nouns
        if entity_type in ['PERSON', 'ORG', 'GPE', 'LOC', 'FAC']:
            # Preserve acronyms and special cases
            words = normalized.split()
            normalized_words = []
            for word in words:
                if word.isupper() and len(word) > 1:
                    normalized_words.append(word)
                else:
                    normalized_words.append(word.capitalize())
            normalized = ' '.join(normalized_words)
        
        # Store normalization mapping
        if entity_name != normalized:
            self.entity_normalization[entity_name] = normalized
        
        return normalized
    
    def standardize_date(self, date_str: str) -> Optional[str]:
        """Standardize date format to YYYY-MM-DD"""
        # Handle None, NaN, or empty values
        if date_str is None:
            return None
        
        # Convert to string if it's a number (handles NaN from pandas)
        if isinstance(date_str, (int, float)):
            # Check if it's NaN
            import math
            if isinstance(date_str, float) and math.isnan(date_str):
                return None
            # Convert number to string (if it's a timestamp or date number)
            date_str = str(int(date_str)) if isinstance(date_str, float) else str(date_str)
        
        # Convert to string if not already
        if not isinstance(date_str, str):
            date_str = str(date_str)
        
        # Strip whitespace
        date_str = date_str.strip()
        
        if not date_str or date_str.lower() in ['nan', 'none', 'null', '']:
            return None
        
        date_formats = [
            '%Y-%m-%d',
            '%Y/%m/%d',
            '%m/%d/%Y',
            '%d/%m/%Y',
            '%B %d, %Y',
            '%b %d, %Y',
            '%Y-%m-%d %H:%M:%S',
            '%Y-%m-%dT%H:%M:%S',
        ]
        
        for fmt in date_formats:
            try:
                dt = datetime.strptime(date_str, fmt)
                return dt.strftime('%Y-%m-%d')
            except:
                continue
        
        # If parsing fails, try to extract year-month-day pattern
        try:
            match = re.search(r'(\d{4})[-\/](\d{1,2})[-\/](\d{1,2})', date_str)
            if match:
                year, month, day = match.groups()
                dt = datetime(int(year), int(month), int(day))
                return dt.strftime('%Y-%m-%d')
        except:
            pass
        
        return None
    
    def get_country_code(self, location_name: str) -> Optional[str]:
        """Get country code for location"""
        location_lower = location_name.lower()
        
        # Direct lookup
        if location_lower in self.country_codes:
            return self.country_codes[location_lower]
        
        # Partial match
        for key, code in self.country_codes.items():
            if key in location_lower or location_lower in key:
                return code
        
        return None
    
    def infer_organization_type(self, org_name: str) -> str:
        """Infer organization type from name"""
        org_lower = org_name.lower()
        
        if any(term in org_lower for term in ['nato', 'un', 'united nations', 'eu', 'european union']):
            return 'INTERNATIONAL_ORG'
        elif any(term in org_lower for term in ['congress', 'senate', 'house', 'parliament']):
            return 'GOVERNMENT_BODY'
        elif any(term in org_lower for term in ['military', 'army', 'navy', 'air force', 'defense']):
            return 'MILITARY'
        elif any(term in org_lower for term in ['company', 'corp', 'inc', 'ltd', 'llc']):
            return 'CORPORATION'
        else:
            return 'OTHER'
    
    def create_surrogate_key(self, entity_type: str, natural_key: str) -> int:
        """Create surrogate key for entity"""
        if natural_key not in self.surrogate_keys[entity_type]:
            self.surrogate_keys[entity_type][natural_key] = len(self.surrogate_keys[entity_type]) + 1
        return self.surrogate_keys[entity_type][natural_key]
    
    def preprocess_speeches(self, speeches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Preprocess speech data"""
        print_section("PREPROCESSING SPEECHES")
        
        preprocessed = []
        
        for speech in speeches:
            processed = {}
            
            # Basic fields
            processed['speech_id'] = speech.get('speech_id', '')
            processed['title'] = speech.get('title', '').strip()
            processed['url'] = speech.get('url', '').strip()
            processed['location'] = speech.get('location', '').strip()
            processed['speaker'] = speech.get('speaker', 'Donald Trump').strip()
            
            # Standardize date
            date_str = speech.get('date', '')
            processed['date'] = self.standardize_date(date_str)
            processed['date_original'] = date_str
            
            # Timestamps
            processed['scraped_at'] = speech.get('scraped_at', '')
            processed['cleaned_at'] = speech.get('cleaned_at', '')
            processed['transformed_at'] = speech.get('transformed_at', '')
            
            # Create surrogate key
            natural_key = processed['speech_id'] or processed['title']
            processed['speech_surrogate_key'] = self.create_surrogate_key('speech', natural_key)
            
            preprocessed.append(processed)
        
        print(f"✓ Preprocessed {len(preprocessed)} speeches")
        return preprocessed
    
    def preprocess_entities(self, entity_catalog: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        """Preprocess entity catalog"""
        print_section("PREPROCESSING ENTITIES")
        
        entities = entity_catalog.get('entities', {})
        preprocessed = {
            'persons': [],
            'organizations': [],
            'locations': []
        }
        
        for entity_key, entity_data in entities.items():
            entity_type = entity_data.get('entity_type', '')
            entity_name = entity_data.get('entity_name', '')
            
            normalized_name = self.normalize_entity_name(entity_name, entity_type)
            
            if entity_type == 'PERSON':
                person = {
                    'person_name': normalized_name,
                    'normalized_name': normalized_name,
                    'first_mentioned_date': self.standardize_date(entity_data.get('first_mentioned_date', '')),
                    'last_mentioned_date': self.standardize_date(entity_data.get('last_mentioned_date', '')),
                    'mention_count': entity_data.get('frequency', 0),
                    'speech_count': len(entity_data.get('speech_ids', []))
                }
                natural_key = normalized_name
                person['person_surrogate_key'] = self.create_surrogate_key('person', natural_key)
                preprocessed['persons'].append(person)
            
            elif entity_type == 'ORG':
                org = {
                    'org_name': normalized_name,
                    'normalized_name': normalized_name,
                    'org_type': self.infer_organization_type(normalized_name),
                    'first_mentioned_date': self.standardize_date(entity_data.get('first_mentioned_date', '')),
                    'last_mentioned_date': self.standardize_date(entity_data.get('last_mentioned_date', '')),
                    'mention_count': entity_data.get('frequency', 0),
                    'speech_count': len(entity_data.get('speech_ids', []))
                }
                natural_key = normalized_name
                org['org_surrogate_key'] = self.create_surrogate_key('organization', natural_key)
                preprocessed['organizations'].append(org)
            
            elif entity_type in ['GPE', 'LOC', 'FAC']:
                location = {
                    'location_name': normalized_name,
                    'normalized_name': normalized_name,
                    'location_type': entity_type,
                    'country_code': self.get_country_code(normalized_name),
                    'first_mentioned_date': self.standardize_date(entity_data.get('first_mentioned_date', '')),
                    'last_mentioned_date': self.standardize_date(entity_data.get('last_mentioned_date', '')),
                    'mention_count': entity_data.get('frequency', 0),
                    'speech_count': len(entity_data.get('speech_ids', []))
                }
                natural_key = normalized_name
                location['location_surrogate_key'] = self.create_surrogate_key('location', natural_key)
                preprocessed['locations'].append(location)
        
        print(f"✓ Preprocessed {len(preprocessed['persons'])} persons")
        print(f"✓ Preprocessed {len(preprocessed['organizations'])} organizations")
        print(f"✓ Preprocessed {len(preprocessed['locations'])} locations")
        
        return preprocessed
    
    def create_date_dimension(self, dates: Set[str]) -> List[Dict[str, Any]]:
        """Create date dimension records"""
        print_section("CREATING DATE DIMENSION")
        
        date_dim = []
        
        for date_str in sorted(dates):
            if not date_str:
                continue
            
            try:
                dt = datetime.strptime(date_str, '%Y-%m-%d')
                
                date_record = {
                    'full_date': date_str,
                    'year': dt.year,
                    'quarter': (dt.month - 1) // 3 + 1,
                    'month': dt.month,
                    'day': dt.day,
                    'day_of_week': dt.weekday(),  # 0=Monday, 6=Sunday
                    'is_weekend': dt.weekday() >= 5,
                    'fiscal_year': dt.year if dt.month >= 10 else dt.year - 1,  # Oct-Sep fiscal year
                    'fiscal_quarter': ((dt.month - 1) // 3 + 1) if dt.month >= 10 else ((dt.month + 2) // 3 + 1)
                }
                
                natural_key = date_str
                date_record['date_surrogate_key'] = self.create_surrogate_key('date', natural_key)
                date_dim.append(date_record)
            except:
                continue
        
        print(f"✓ Created {len(date_dim)} date dimension records")
        return date_dim
    
    def preprocess_all(self) -> bool:
        """Preprocess all extracted data"""
        print_section("ETL PREPROCESSING PIPELINE")
        
        # Load staging data
        transformed_file = self.find_latest_staging_file('transformed_data_*.json')
        features_file = self.find_latest_staging_file('features_data_*.json')
        entity_file = self.find_latest_staging_file('entity_catalog_*.json')
        
        if not transformed_file:
            print("Error: No transformed data found in staging")
            return False
        
        # Load data
        transformed_data = load_json(str(transformed_file))
        if isinstance(transformed_data, list):
            speeches = transformed_data
        elif isinstance(transformed_data, dict) and 'speeches' in transformed_data:
            speeches = transformed_data['speeches']
        else:
            speeches = [transformed_data]
        
        # Preprocess speeches
        preprocessed_speeches = self.preprocess_speeches(speeches)
        
        # Extract unique dates
        dates = {s['date'] for s in preprocessed_speeches if s.get('date')}
        date_dimension = self.create_date_dimension(dates)
        
        # Preprocess entities if available
        preprocessed_entities = None
        if entity_file:
            entity_catalog = load_json(str(entity_file))
            if entity_catalog:
                preprocessed_entities = self.preprocess_entities(entity_catalog)
        
        # Save preprocessed data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        save_json(preprocessed_speeches, str(self.preprocessed_dir / f"preprocessed_speeches_{timestamp}.json"))
        save_json(date_dimension, str(self.preprocessed_dir / f"preprocessed_dates_{timestamp}.json"))
        
        if preprocessed_entities:
            save_json(preprocessed_entities, str(self.preprocessed_dir / f"preprocessed_entities_{timestamp}.json"))
        
        # Save surrogate key mappings
        save_json(dict(self.surrogate_keys), str(self.preprocessed_dir / f"surrogate_keys_{timestamp}.json"))
        
        print_section("PREPROCESSING SUMMARY")
        print(f"✓ Preprocessed {len(preprocessed_speeches)} speeches")
        print(f"✓ Created {len(date_dimension)} date dimension records")
        if preprocessed_entities:
            print(f"✓ Preprocessed entities: {len(preprocessed_entities.get('persons', []))} persons, "
                  f"{len(preprocessed_entities.get('organizations', []))} orgs, "
                  f"{len(preprocessed_entities.get('locations', []))} locations")
        
        return True


def main():
    """Main execution function"""
    preprocessor = ETLPreprocessor()
    success = preprocessor.preprocess_all()
    
    if success:
        print("\n✓ ETL preprocessing completed successfully")
        sys.exit(0)
    else:
        print("\n✗ ETL preprocessing failed")
        sys.exit(1)


if __name__ == "__main__":
    main()

