"""
ETL Extraction Pipeline

Extracts data from multiple sources:
- Raw scraped data
- Cleaned data
- Transformed NLP data
- Feature-engineered data
- Entity catalog

Validates data quality with completeness checks, data type validation, and referential integrity.
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import json

# Add scripts directory to path
sys.path.append(str(Path(__file__).parent))
from utils import (
    load_config, load_json, save_json, load_csv,
    print_section, print_stats
)


class ETLExtractor:
    """Extract data from all sources for data warehouse"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize ETL extractor"""
        self.config = load_config(config_path)
        self.paths = self.config.get('paths', {})
        self.extracted_data = {
            'raw_data': None,
            'cleaned_data': None,
            'transformed_data': None,
            'features_data': None,
            'entity_catalog': None,
            'extraction_metadata': {}
        }
        self.quality_report = {
            'extraction_timestamp': datetime.now().isoformat(),
            'sources': {},
            'quality_checks': {},
            'errors': [],
            'warnings': []
        }
    
    def find_latest_file(self, directory: str, pattern: str) -> Optional[Path]:
        """Find the most recent file matching a pattern"""
        dir_path = Path(directory)
        if not dir_path.exists():
            return None
        
        files = list(dir_path.glob(pattern))
        if files:
            return max(files, key=lambda p: p.stat().st_mtime)
        return None
    
    def extract_raw_data(self) -> bool:
        """Extract raw scraped data"""
        print_section("EXTRACTING RAW DATA")
        
        raw_dir = self.paths.get('raw_data', 'data/raw')
        raw_file = self.find_latest_file(raw_dir, 'trump_speeches_*.json')
        
        if not raw_file:
            raw_file = self.find_latest_file(raw_dir, 'trump_speeches_*.csv')
        
        if not raw_file:
            self.quality_report['warnings'].append("No raw data files found")
            print("⚠ No raw data files found")
            return False
        
        print(f"Found raw data file: {raw_file}")
        
        if raw_file.suffix == '.json':
            data = load_json(str(raw_file))
        else:
            # Load CSV and convert to dict format
            import pandas as pd
            df = pd.read_csv(raw_file)
            data = df.to_dict('records')
        
        if data:
            self.extracted_data['raw_data'] = data
            self.quality_report['sources']['raw_data'] = {
                'file': str(raw_file),
                'record_count': len(data) if isinstance(data, list) else 1,
                'extracted': True
            }
            print(f"✓ Extracted {len(data) if isinstance(data, list) else 1} raw records")
            return True
        else:
            self.quality_report['errors'].append(f"Failed to load raw data from {raw_file}")
            return False
    
    def extract_cleaned_data(self) -> bool:
        """Extract cleaned data"""
        print_section("EXTRACTING CLEANED DATA")
        
        cleaned_dir = self.paths.get('cleaned_data', 'data/cleaned')
        cleaned_file = self.find_latest_file(cleaned_dir, 'speeches_cleaned_*.json')
        
        if not cleaned_file:
            self.quality_report['warnings'].append("No cleaned data files found")
            print("⚠ No cleaned data files found")
            return False
        
        print(f"Found cleaned data file: {cleaned_file}")
        data = load_json(str(cleaned_file))
        
        if data:
            if isinstance(data, list):
                self.extracted_data['cleaned_data'] = data
                count = len(data)
            elif isinstance(data, dict) and 'speeches' in data:
                self.extracted_data['cleaned_data'] = data['speeches']
                count = len(data['speeches'])
            else:
                self.extracted_data['cleaned_data'] = [data]
                count = 1
            
            self.quality_report['sources']['cleaned_data'] = {
                'file': str(cleaned_file),
                'record_count': count,
                'extracted': True
            }
            print(f"✓ Extracted {count} cleaned records")
            return True
        else:
            self.quality_report['errors'].append(f"Failed to load cleaned data from {cleaned_file}")
            return False
    
    def extract_transformed_data(self) -> bool:
        """Extract transformed NLP data"""
        print_section("EXTRACTING TRANSFORMED DATA")
        
        transformed_dir = self.paths.get('transformed_data', 'data/transformed')
        transformed_file = self.find_latest_file(transformed_dir, 'speeches_nlp_features_*.json')
        
        if not transformed_file:
            self.quality_report['warnings'].append("No transformed data files found")
            print("⚠ No transformed data files found")
            return False
        
        print(f"Found transformed data file: {transformed_file}")
        data = load_json(str(transformed_file))
        
        if data:
            if isinstance(data, list):
                self.extracted_data['transformed_data'] = data
                count = len(data)
            elif isinstance(data, dict) and 'speeches' in data:
                self.extracted_data['transformed_data'] = data['speeches']
                count = len(data['speeches'])
            else:
                self.extracted_data['transformed_data'] = [data]
                count = 1
            
            self.quality_report['sources']['transformed_data'] = {
                'file': str(transformed_file),
                'record_count': count,
                'extracted': True
            }
            print(f"✓ Extracted {count} transformed records")
            return True
        else:
            self.quality_report['errors'].append(f"Failed to load transformed data from {transformed_file}")
            return False
    
    def extract_features_data(self) -> bool:
        """Extract feature-engineered data"""
        print_section("EXTRACTING FEATURES DATA")
        
        transformed_dir = self.paths.get('transformed_data', 'data/transformed')
        features_file = self.find_latest_file(transformed_dir, 'speeches_features_complete_*.csv')
        
        if not features_file:
            features_file = self.find_latest_file(transformed_dir, 'speeches_features_complete_*.json')
        
        if not features_file:
            self.quality_report['warnings'].append("No features data files found")
            print("⚠ No features data files found")
            return False
        
        print(f"Found features data file: {features_file}")
        
        if features_file.suffix == '.csv':
            import pandas as pd
            df = pd.read_csv(features_file)
            self.extracted_data['features_data'] = df.to_dict('records')
            count = len(df)
        else:
            data = load_json(str(features_file))
            if isinstance(data, list):
                self.extracted_data['features_data'] = data
                count = len(data)
            else:
                self.extracted_data['features_data'] = [data]
                count = 1
        
        self.quality_report['sources']['features_data'] = {
            'file': str(features_file),
            'record_count': count,
            'extracted': True
        }
        print(f"✓ Extracted {count} feature records")
        return True
    
    def extract_entity_catalog(self) -> bool:
        """Extract entity catalog"""
        print_section("EXTRACTING ENTITY CATALOG")
        
        entity_file = self.find_latest_file('data/entities', 'entity_catalog_*.json')
        
        if not entity_file:
            self.quality_report['warnings'].append("No entity catalog files found")
            print("⚠ No entity catalog files found")
            return False
        
        print(f"Found entity catalog file: {entity_file}")
        data = load_json(str(entity_file))
        
        if data:
            self.extracted_data['entity_catalog'] = data
            entity_count = len(data.get('entities', {}))
            
            self.quality_report['sources']['entity_catalog'] = {
                'file': str(entity_file),
                'entity_count': entity_count,
                'extracted': True
            }
            print(f"✓ Extracted {entity_count} entities")
            return True
        else:
            self.quality_report['errors'].append(f"Failed to load entity catalog from {entity_file}")
            return False
    
    def validate_data_quality(self) -> Dict[str, Any]:
        """Validate extracted data quality"""
        print_section("VALIDATING DATA QUALITY")
        
        checks = {
            'completeness': {},
            'data_types': {},
            'referential_integrity': {}
        }
        
        # Check completeness
        for source, data in self.extracted_data.items():
            if source == 'extraction_metadata':
                continue
            
            if data is None:
                checks['completeness'][source] = {
                    'status': 'missing',
                    'message': f'{source} not extracted'
                }
            else:
                if isinstance(data, list):
                    count = len(data)
                elif isinstance(data, dict):
                    count = len(data)
                else:
                    count = 1
                
                checks['completeness'][source] = {
                    'status': 'present',
                    'record_count': count
                }
        
        # Check data types and required fields
        if self.extracted_data['transformed_data']:
            sample = self.extracted_data['transformed_data'][0] if isinstance(self.extracted_data['transformed_data'], list) else self.extracted_data['transformed_data']
            required_fields = ['speech_id', 'entities', 'sentiment', 'emotions']
            
            missing_fields = [field for field in required_fields if field not in sample]
            checks['data_types']['transformed_data'] = {
                'has_required_fields': len(missing_fields) == 0,
                'missing_fields': missing_fields
            }
        
        if self.extracted_data['features_data']:
            sample = self.extracted_data['features_data'][0] if isinstance(self.extracted_data['features_data'], list) else self.extracted_data['features_data']
            required_fields = ['speech_id', 'word_count', 'sentiment_compound']
            
            missing_fields = [field for field in required_fields if field not in sample]
            checks['data_types']['features_data'] = {
                'has_required_fields': len(missing_fields) == 0,
                'missing_fields': missing_fields
            }
        
        # Check referential integrity (speech_ids should match across sources)
        if self.extracted_data['transformed_data'] and self.extracted_data['features_data']:
            transformed_ids = set()
            if isinstance(self.extracted_data['transformed_data'], list):
                transformed_ids = {s.get('speech_id') for s in self.extracted_data['transformed_data'] if s.get('speech_id')}
            else:
                if self.extracted_data['transformed_data'].get('speech_id'):
                    transformed_ids = {self.extracted_data['transformed_data']['speech_id']}
            
            features_ids = set()
            if isinstance(self.extracted_data['features_data'], list):
                features_ids = {s.get('speech_id') for s in self.extracted_data['features_data'] if s.get('speech_id')}
            else:
                if self.extracted_data['features_data'].get('speech_id'):
                    features_ids = {self.extracted_data['features_data']['speech_id']}
            
            common_ids = transformed_ids & features_ids
            checks['referential_integrity']['speech_id_match'] = {
                'transformed_count': len(transformed_ids),
                'features_count': len(features_ids),
                'common_count': len(common_ids),
                'match_rate': len(common_ids) / max(len(transformed_ids), len(features_ids), 1) * 100
            }
        
        self.quality_report['quality_checks'] = checks
        return checks
    
    def save_staging_data(self, output_dir: str = "data/staging") -> bool:
        """Save extracted data to staging area"""
        print_section("SAVING STAGING DATA")
        
        staging_dir = Path(output_dir)
        staging_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        saved_files = []
        
        for source, data in self.extracted_data.items():
            if source == 'extraction_metadata' or data is None:
                continue
            
            output_file = staging_dir / f"{source}_{timestamp}.json"
            
            if save_json(data, str(output_file)):
                saved_files.append(str(output_file))
                print(f"✓ Saved {source} to {output_file}")
        
        # Save quality report
        quality_report_file = staging_dir / f"extraction_quality_report_{timestamp}.json"
        if save_json(self.quality_report, str(quality_report_file)):
            saved_files.append(str(quality_report_file))
            print(f"✓ Saved quality report to {quality_report_file}")
        
        self.extracted_data['extraction_metadata'] = {
            'staging_directory': str(staging_dir),
            'timestamp': timestamp,
            'saved_files': saved_files
        }
        
        return len(saved_files) > 0
    
    def extract_all(self) -> bool:
        """Extract data from all sources"""
        print_section("ETL EXTRACTION PIPELINE")
        
        results = []
        results.append(self.extract_raw_data())
        results.append(self.extract_cleaned_data())
        results.append(self.extract_transformed_data())
        results.append(self.extract_features_data())
        results.append(self.extract_entity_catalog())
        
        # Validate quality
        self.validate_data_quality()
        
        # Save to staging
        self.save_staging_data()
        
        # Print summary
        print_section("EXTRACTION SUMMARY")
        print(f"Sources extracted: {sum(results)}/{len(results)}")
        print(f"Errors: {len(self.quality_report['errors'])}")
        print(f"Warnings: {len(self.quality_report['warnings'])}")
        
        if self.quality_report['errors']:
            print("\nErrors:")
            for error in self.quality_report['errors']:
                print(f"  - {error}")
        
        if self.quality_report['warnings']:
            print("\nWarnings:")
            for warning in self.quality_report['warnings']:
                print(f"  - {warning}")
        
        return sum(results) > 0


def main():
    """Main execution function"""
    extractor = ETLExtractor()
    success = extractor.extract_all()
    
    if success:
        print("\n✓ ETL extraction completed successfully")
        sys.exit(0)
    else:
        print("\n✗ ETL extraction failed")
        sys.exit(1)


if __name__ == "__main__":
    main()

