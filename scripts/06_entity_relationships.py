"""
Entity Relationship Mapping Script

This script maps relationships between:
- Speech → Entities (many-to-many)
- Entity → Entity co-occurrences
- Entity frequency by speech date
Generates entity network data for visualization
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Set, Tuple
from collections import defaultdict, Counter
from datetime import datetime
import json

# Add scripts directory to path
sys.path.append(str(Path(__file__).parent))
from utils import (
    load_config, load_json, save_json,
    print_section, print_stats
)


class EntityRelationshipMapper:
    """Map relationships between speeches and entities"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize relationship mapper"""
        self.config = load_config(config_path)
        self.speech_entity_map = defaultdict(set)  # speech_id -> set of entity_keys
        self.entity_speech_map = defaultdict(set)    # entity_key -> set of speech_ids
        self.entity_cooccurrence = defaultdict(Counter)  # entity_key -> Counter of co-occurring entities
        self.entity_date_frequency = defaultdict(Counter)  # entity_key -> Counter of dates
        self.entity_network = {
            'nodes': [],
            'edges': []
        }
    
    def load_entity_catalog(self, catalog_path: str) -> Dict[str, Dict[str, Any]]:
        """Load entity catalog"""
        catalog_data = load_json(catalog_path)
        
        if not catalog_data:
            return {}
        
        return catalog_data.get('entities', {})
    
    def load_transformed_data(self, transformed_path: str) -> List[Dict[str, Any]]:
        """Load transformed speech data"""
        transformed_data = load_json(transformed_path)
        
        if not transformed_data:
            return []
        
        if isinstance(transformed_data, list):
            return transformed_data
        elif isinstance(transformed_data, dict) and 'speeches' in transformed_data:
            return transformed_data['speeches']
        else:
            return []
    
    def extract_entity_key(self, entity_type: str, entity_name: str) -> str:
        """Create entity key from type and name"""
        normalized = entity_name.strip()
        normalized = ' '.join(normalized.split())
        return f"{entity_type}:{normalized}"
    
    def map_speech_entity_relationships(self, speeches: List[Dict[str, Any]]):
        """Map speech to entity relationships"""
        print_section("MAPPING SPEECH-ENTITY RELATIONSHIPS")
        
        for speech in speeches:
            speech_id = speech.get('speech_id', '')
            speech_date = speech.get('date', '')
            
            if not speech_id:
                continue
            
            # Extract entities from speech
            entities = speech.get('entities', {})
            speech_entities = set()
            
            for entity_type, entity_list in entities.items():
                for entity_name in entity_list:
                    if not entity_name or len(entity_name.strip()) < 2:
                        continue
                    
                    entity_key = self.extract_entity_key(entity_type, entity_name)
                    speech_entities.add(entity_key)
                    
                    # Map entity to speech
                    self.entity_speech_map[entity_key].add(speech_id)
                    
                    # Track entity frequency by date
                    if speech_date:
                        self.entity_date_frequency[entity_key][speech_date] += 1
            
            # Map speech to entities
            self.speech_entity_map[speech_id] = speech_entities
            
            # Track co-occurrences within this speech
            entity_list = list(speech_entities)
            for i, entity1 in enumerate(entity_list):
                for entity2 in entity_list[i+1:]:
                    self.entity_cooccurrence[entity1][entity2] += 1
                    self.entity_cooccurrence[entity2][entity1] += 1
        
        print(f"✓ Mapped {len(self.speech_entity_map)} speeches to entities")
        print(f"✓ Found {len(self.entity_speech_map)} unique entities")
    
    def build_entity_network(self, min_cooccurrence: int = 2):
        """Build entity network graph for visualization"""
        print_section("BUILDING ENTITY NETWORK")
        
        # Create nodes
        entity_ids = {}
        node_id = 0
        
        for entity_key in self.entity_speech_map.keys():
            entity_type, entity_name = entity_key.split(':', 1)
            entity_ids[entity_key] = node_id
            
            self.entity_network['nodes'].append({
                'id': node_id,
                'label': entity_name,
                'type': entity_type,
                'frequency': len(self.entity_speech_map[entity_key]),
                'entity_key': entity_key
            })
            node_id += 1
        
        # Create edges (only for significant co-occurrences)
        edge_id = 0
        processed_pairs = set()
        
        for entity1, cooccurrences in self.entity_cooccurrence.items():
            if entity1 not in entity_ids:
                continue
            
            for entity2, count in cooccurrences.items():
                if entity2 not in entity_ids:
                    continue
                
                if count < min_cooccurrence:
                    continue
                
                # Avoid duplicate edges
                pair = tuple(sorted([entity1, entity2]))
                if pair in processed_pairs:
                    continue
                processed_pairs.add(pair)
                
                self.entity_network['edges'].append({
                    'id': edge_id,
                    'source': entity_ids[entity1],
                    'target': entity_ids[entity2],
                    'weight': count,
                    'label': f"{count} co-occurrences"
                })
                edge_id += 1
        
        print(f"✓ Created network with {len(self.entity_network['nodes'])} nodes")
        print(f"✓ Created {len(self.entity_network['edges'])} edges")
    
    def get_relationship_statistics(self) -> Dict[str, Any]:
        """Get statistics about relationships"""
        stats = {
            'total_speeches': len(self.speech_entity_map),
            'total_entities': len(self.entity_speech_map),
            'total_relationships': sum(len(entities) for entities in self.speech_entity_map.values()),
            'avg_entities_per_speech': 0,
            'avg_speeches_per_entity': 0,
            'most_connected_entities': [],
            'strongest_cooccurrences': []
        }
        
        if stats['total_speeches'] > 0:
            stats['avg_entities_per_speech'] = stats['total_relationships'] / stats['total_speeches']
        
        if stats['total_entities'] > 0:
            stats['avg_speeches_per_entity'] = stats['total_relationships'] / stats['total_entities']
        
        # Most connected entities (appear in most speeches)
        entity_speech_counts = [
            (entity_key, len(speech_ids))
            for entity_key, speech_ids in self.entity_speech_map.items()
        ]
        entity_speech_counts.sort(key=lambda x: x[1], reverse=True)
        
        stats['most_connected_entities'] = [
            {
                'entity_key': entity_key,
                'entity_name': entity_key.split(':', 1)[1] if ':' in entity_key else entity_key,
                'entity_type': entity_key.split(':', 1)[0] if ':' in entity_key else 'UNKNOWN',
                'speech_count': count
            }
            for entity_key, count in entity_speech_counts[:20]
        ]
        
        # Strongest co-occurrences
        cooccurrence_list = []
        processed_pairs = set()
        
        for entity1, cooccurrences in self.entity_cooccurrence.items():
            for entity2, count in cooccurrences.items():
                pair = tuple(sorted([entity1, entity2]))
                if pair in processed_pairs:
                    continue
                processed_pairs.add(pair)
                
                if count >= 2:  # Only significant co-occurrences
                    cooccurrence_list.append({
                        'entity1': entity1.split(':', 1)[1] if ':' in entity1 else entity1,
                        'entity2': entity2.split(':', 1)[1] if ':' in entity2 else entity2,
                        'entity1_type': entity1.split(':', 1)[0] if ':' in entity1 else 'UNKNOWN',
                        'entity2_type': entity2.split(':', 1)[0] if ':' in entity2 else 'UNKNOWN',
                        'cooccurrence_count': count
                    })
        
        cooccurrence_list.sort(key=lambda x: x['cooccurrence_count'], reverse=True)
        stats['strongest_cooccurrences'] = cooccurrence_list[:20]
        
        return stats
    
    def save_relationships(self, output_path: str) -> bool:
        """Save relationship data to JSON"""
        # Convert sets to lists for JSON serialization
        speech_entity_list = {
            speech_id: list(entities)
            for speech_id, entities in self.speech_entity_map.items()
        }
        
        entity_speech_list = {
            entity_key: list(speech_ids)
            for entity_key, speech_ids in self.entity_speech_map.items()
        }
        
        entity_cooccurrence_dict = {
            entity_key: dict(counter)
            for entity_key, counter in self.entity_cooccurrence.items()
        }
        
        entity_date_frequency_dict = {
            entity_key: dict(counter)
            for entity_key, counter in self.entity_date_frequency.items()
        }
        
        relationship_data = {
            'extracted_at': datetime.now().isoformat(),
            'statistics': self.get_relationship_statistics(),
            'speech_entity_map': speech_entity_list,
            'entity_speech_map': entity_speech_list,
            'entity_cooccurrence': entity_cooccurrence_dict,
            'entity_date_frequency': entity_date_frequency_dict,
            'entity_network': self.entity_network
        }
        
        return save_json(relationship_data, output_path)


def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Map entity relationships from speech data'
    )
    parser.add_argument(
        'transformed_file',
        help='Path to transformed speech data JSON file'
    )
    parser.add_argument(
        '--catalog',
        help='Path to entity catalog JSON file (optional)',
        default=None
    )
    parser.add_argument(
        '--output',
        help='Output file path (default: data/entities/entity_relationships_TIMESTAMP.json)',
        default=None
    )
    parser.add_argument(
        '--min-cooccurrence',
        type=int,
        default=2,
        help='Minimum co-occurrence count for network edges (default: 2)'
    )
    
    args = parser.parse_args()
    
    # Load data
    print_section("LOADING DATA")
    speeches = EntityRelationshipMapper().load_transformed_data(args.transformed_file)
    
    if not speeches:
        print("Error: Could not load transformed data")
        sys.exit(1)
    
    print(f"✓ Loaded {len(speeches)} speeches")
    
    # Map relationships
    mapper = EntityRelationshipMapper()
    mapper.map_speech_entity_relationships(speeches)
    
    # Build network
    mapper.build_entity_network(min_cooccurrence=args.min_cooccurrence)
    
    # Print statistics
    print_section("RELATIONSHIP STATISTICS")
    stats = mapper.get_relationship_statistics()
    
    print(f"Total speeches: {stats['total_speeches']}")
    print(f"Total entities: {stats['total_entities']}")
    print(f"Total relationships: {stats['total_relationships']}")
    print(f"Average entities per speech: {stats['avg_entities_per_speech']:.2f}")
    print(f"Average speeches per entity: {stats['avg_speeches_per_entity']:.2f}")
    
    print("\nTop 10 most connected entities:")
    for i, entity in enumerate(stats['most_connected_entities'][:10], 1):
        print(f"  {i}. {entity['entity_name']} ({entity['entity_type']}): "
              f"{entity['speech_count']} speeches")
    
    print("\nTop 10 strongest co-occurrences:")
    for i, cooc in enumerate(stats['strongest_cooccurrences'][:10], 1):
        print(f"  {i}. {cooc['entity1']} ↔ {cooc['entity2']}: "
              f"{cooc['cooccurrence_count']} co-occurrences")
    
    # Save relationships
    if args.output:
        output_path = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("data/entities")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"entity_relationships_{timestamp}.json"
    
    print_section("SAVING RELATIONSHIPS")
    if mapper.save_relationships(str(output_path)):
        print(f"\n✓ Relationship data saved to {output_path}")
    else:
        print("\n✗ Failed to save relationship data")
        sys.exit(1)


if __name__ == "__main__":
    main()

