"""
Entity Extraction Script for Trump Speech Analysis

This script extracts and catalogs all entities from transformed speech data:
- PERSON, ORG, GPE, DATE, MONEY, NORP, FAC, LOC
- Creates comprehensive entity catalog with frequency, dates, co-occurrences
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Set
from collections import defaultdict, Counter
from datetime import datetime
import json

# Add scripts directory to path
sys.path.append(str(Path(__file__).parent))
from utils import (
    load_config, load_json, save_json,
    print_section, print_stats
)


class EntityExtractor:
    """Extract and catalog entities from transformed speech data"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize entity extractor"""
        self.config = load_config(config_path)
        self.entity_catalog = defaultdict(lambda: {
            'entity_name': '',
            'entity_type': '',
            'frequency': 0,
            'speech_ids': [],
            'first_mentioned_date': None,
            'last_mentioned_date': None,
            'mention_dates': [],
            'co_occurring_entities': defaultdict(int)
        })
        
    def extract_entities_from_speech(self, speech: Dict[str, Any]) -> Dict[str, List[str]]:
        """Extract entities from a single speech"""
        entities = {}
        
        # Get entities from transformed data
        entity_data = speech.get('entities', {})
        
        # Extract all entity types
        entity_types = ['PERSON', 'ORG', 'GPE', 'DATE', 'MONEY', 'NORP', 'FAC', 'LOC']
        
        for entity_type in entity_types:
            entities[entity_type] = entity_data.get(entity_type, [])
        
        return entities
    
    def normalize_entity_name(self, entity_name: str) -> str:
        """Normalize entity name for consistency"""
        # Basic normalization
        normalized = entity_name.strip()
        # Remove extra whitespace
        normalized = ' '.join(normalized.split())
        # Title case for consistency (can be enhanced with fuzzy matching)
        return normalized
    
    def parse_date(self, date_str: str) -> datetime:
        """Parse date string to datetime object"""
        if not date_str:
            return None
        
        # Try common date formats
        date_formats = [
            '%Y-%m-%d',
            '%Y/%m/%d',
            '%m/%d/%Y',
            '%d/%m/%Y',
            '%B %d, %Y',
            '%b %d, %Y',
            '%Y-%m-%d %H:%M:%S',
        ]
        
        for fmt in date_formats:
            try:
                return datetime.strptime(date_str, fmt)
            except:
                continue
        
        return None
    
    def build_entity_catalog(self, speeches: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Build comprehensive entity catalog from all speeches"""
        print_section("BUILDING ENTITY CATALOG")
        
        total_entities = 0
        
        for speech in speeches:
            speech_id = speech.get('speech_id', '')
            speech_date = speech.get('date', '')
            date_obj = self.parse_date(speech_date)
            
            # Extract entities from this speech
            entities = self.extract_entities_from_speech(speech)
            
            # Track entities mentioned in this speech
            speech_entities = set()
            
            for entity_type, entity_list in entities.items():
                for entity_name in entity_list:
                    if not entity_name or len(entity_name.strip()) < 2:
                        continue
                    
                    normalized = self.normalize_entity_name(entity_name)
                    entity_key = f"{entity_type}:{normalized}"
                    
                    # Update catalog
                    if entity_key not in self.entity_catalog:
                        self.entity_catalog[entity_key]['entity_name'] = normalized
                        self.entity_catalog[entity_key]['entity_type'] = entity_type
                    
                    catalog_entry = self.entity_catalog[entity_key]
                    catalog_entry['frequency'] += 1
                    
                    if speech_id and speech_id not in catalog_entry['speech_ids']:
                        catalog_entry['speech_ids'].append(speech_id)
                    
                    if date_obj:
                        if catalog_entry['first_mentioned_date'] is None:
                            catalog_entry['first_mentioned_date'] = speech_date
                        catalog_entry['last_mentioned_date'] = speech_date
                        catalog_entry['mention_dates'].append(speech_date)
                    
                    speech_entities.add(entity_key)
                    total_entities += 1
            
            # Track co-occurrences within this speech
            speech_entity_list = list(speech_entities)
            for i, entity1 in enumerate(speech_entity_list):
                for entity2 in speech_entity_list[i+1:]:
                    # Update co-occurrence counts
                    self.entity_catalog[entity1]['co_occurring_entities'][entity2] += 1
                    self.entity_catalog[entity2]['co_occurring_entities'][entity1] += 1
        
        print(f"✓ Processed {len(speeches)} speeches")
        print(f"✓ Extracted {total_entities} entity mentions")
        print(f"✓ Found {len(self.entity_catalog)} unique entities")
        
        return dict(self.entity_catalog)
    
    def get_entity_statistics(self) -> Dict[str, Any]:
        """Get statistics about extracted entities"""
        stats = {
            'total_unique_entities': len(self.entity_catalog),
            'entities_by_type': Counter(),
            'most_frequent_entities': [],
            'entities_with_dates': 0,
            'total_co_occurrences': 0
        }
        
        for entity_key, catalog_entry in self.entity_catalog.items():
            entity_type = catalog_entry['entity_type']
            stats['entities_by_type'][entity_type] += 1
            
            if catalog_entry['first_mentioned_date']:
                stats['entities_with_dates'] += 1
            
            stats['total_co_occurrences'] += len(catalog_entry['co_occurring_entities'])
        
        # Get most frequent entities
        sorted_entities = sorted(
            self.entity_catalog.items(),
            key=lambda x: x[1]['frequency'],
            reverse=True
        )
        
        stats['most_frequent_entities'] = [
            {
                'entity_name': entry['entity_name'],
                'entity_type': entry['entity_type'],
                'frequency': entry['frequency'],
                'speech_count': len(entry['speech_ids'])
            }
            for _, entry in sorted_entities[:20]
        ]
        
        return stats
    
    def save_catalog(self, output_path: str) -> bool:
        """Save entity catalog to JSON"""
        # Convert defaultdict to regular dict for JSON serialization
        catalog_dict = {}
        for key, value in self.entity_catalog.items():
            catalog_dict[key] = {
                'entity_name': value['entity_name'],
                'entity_type': value['entity_type'],
                'frequency': value['frequency'],
                'speech_ids': value['speech_ids'],
                'first_mentioned_date': value['first_mentioned_date'],
                'last_mentioned_date': value['last_mentioned_date'],
                'mention_dates': list(set(value['mention_dates'])),  # Unique dates
                'co_occurring_entities': dict(value['co_occurring_entities'])
            }
        
        catalog_data = {
            'extracted_at': datetime.now().isoformat(),
            'total_entities': len(catalog_dict),
            'statistics': self.get_entity_statistics(),
            'entities': catalog_dict
        }
        
        return save_json(catalog_data, output_path)


def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Extract entities from transformed speech data'
    )
    parser.add_argument(
        'input_file',
        help='Path to transformed speech data JSON file'
    )
    parser.add_argument(
        '--output',
        help='Output file path (default: data/entities/entity_catalog_TIMESTAMP.json)',
        default=None
    )
    
    args = parser.parse_args()
    
    # Load transformed data
    print_section("LOADING TRANSFORMED DATA")
    transformed_data = load_json(args.input_file)
    
    if not transformed_data:
        print("Error: Could not load transformed data")
        sys.exit(1)
    
    if isinstance(transformed_data, list):
        speeches = transformed_data
    elif isinstance(transformed_data, dict) and 'speeches' in transformed_data:
        speeches = transformed_data['speeches']
    else:
        print("Error: Unexpected data format")
        sys.exit(1)
    
    print(f"✓ Loaded {len(speeches)} speeches")
    
    # Extract entities
    extractor = EntityExtractor()
    catalog = extractor.build_entity_catalog(speeches)
    
    # Print statistics
    print_section("ENTITY STATISTICS")
    stats = extractor.get_entity_statistics()
    
    print("\nEntities by type:")
    for entity_type, count in stats['entities_by_type'].most_common():
        print(f"  {entity_type}: {count}")
    
    print(f"\nTotal unique entities: {stats['total_unique_entities']}")
    print(f"Entities with date information: {stats['entities_with_dates']}")
    print(f"Total co-occurrences: {stats['total_co_occurrences']}")
    
    print("\nTop 10 most frequent entities:")
    for i, entity in enumerate(stats['most_frequent_entities'][:10], 1):
        print(f"  {i}. {entity['entity_name']} ({entity['entity_type']}): "
              f"{entity['frequency']} mentions in {entity['speech_count']} speeches")
    
    # Save catalog
    if args.output:
        output_path = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("data/entities")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"entity_catalog_{timestamp}.json"
    
    print_section("SAVING ENTITY CATALOG")
    if extractor.save_catalog(str(output_path)):
        print(f"\n✓ Entity catalog saved to {output_path}")
    else:
        print("\n✗ Failed to save entity catalog")
        sys.exit(1)


if __name__ == "__main__":
    main()

