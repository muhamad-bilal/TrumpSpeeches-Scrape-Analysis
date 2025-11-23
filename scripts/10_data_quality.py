"""
Data Quality Validation Script

Validates warehouse data:
- Row counts
- Null checks
- Referential integrity
- Data range validations
Generates data quality report
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import json

# Add scripts directory to path
sys.path.append(str(Path(__file__).parent))
from utils import (
    load_config, load_json, save_json,
    print_section, print_stats
)


class DataQualityValidator:
    """Validate data quality for warehouse"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize validator"""
        self.config = load_config(config_path)
        self.quality_report = {
            'validation_timestamp': datetime.now().isoformat(),
            'checks': {},
            'errors': [],
            'warnings': [],
            'summary': {}
        }
    
    def validate_row_counts(self, data_sources: Dict[str, Any]) -> Dict[str, Any]:
        """Validate row counts across data sources"""
        print_section("VALIDATING ROW COUNTS")
        
        checks = {}
        
        # Check preprocessed speeches
        speeches_file = Path("data/staging/preprocessed")
        speeches_files = list(speeches_file.glob('preprocessed_speeches_*.json'))
        if speeches_files:
            speeches = load_json(str(max(speeches_files, key=lambda p: p.stat().st_mtime)))
            speech_count = len(speeches) if isinstance(speeches, list) else 1
            checks['speeches'] = {
                'count': speech_count,
                'status': 'pass' if speech_count > 0 else 'fail'
            }
            print(f"✓ Speeches: {speech_count}")
        
        # Check dates
        dates_files = list(speeches_file.glob('preprocessed_dates_*.json'))
        if dates_files:
            dates = load_json(str(max(dates_files, key=lambda p: p.stat().st_mtime)))
            date_count = len(dates) if isinstance(dates, list) else 1
            checks['dates'] = {
                'count': date_count,
                'status': 'pass' if date_count > 0 else 'fail'
            }
            print(f"✓ Dates: {date_count}")
        
        # Check entities
        entities_files = list(speeches_file.glob('preprocessed_entities_*.json'))
        if entities_files:
            entities = load_json(str(max(entities_files, key=lambda p: p.stat().st_mtime)))
            if entities:
                person_count = len(entities.get('persons', []))
                org_count = len(entities.get('organizations', []))
                location_count = len(entities.get('locations', []))
                checks['entities'] = {
                    'persons': person_count,
                    'organizations': org_count,
                    'locations': location_count,
                    'total': person_count + org_count + location_count,
                    'status': 'pass' if (person_count + org_count + location_count) > 0 else 'fail'
                }
                print(f"✓ Entities: {person_count} persons, {org_count} orgs, {location_count} locations")
        
        return checks
    
    def validate_null_checks(self, data: List[Dict[str, Any]], required_fields: List[str]) -> Dict[str, Any]:
        """Validate null values in required fields"""
        checks = {
            'total_records': len(data),
            'null_counts': {},
            'completeness': {}
        }
        
        for field in required_fields:
            null_count = sum(1 for record in data if not record.get(field))
            checks['null_counts'][field] = null_count
            checks['completeness'][field] = {
                'null_count': null_count,
                'completeness_rate': (len(data) - null_count) / len(data) * 100 if len(data) > 0 else 0,
                'status': 'pass' if null_count == 0 else 'warning' if null_count < len(data) * 0.1 else 'fail'
            }
        
        return checks
    
    def validate_referential_integrity(self) -> Dict[str, Any]:
        """Validate referential integrity"""
        print_section("VALIDATING REFERENTIAL INTEGRITY")
        
        checks = {
            'speech_date_mapping': {},
            'entity_speech_mapping': {}
        }
        
        # Check speech-date mapping
        speeches_file = Path("data/staging/preprocessed")
        speeches_files = list(speeches_file.glob('preprocessed_speeches_*.json'))
        dates_files = list(speeches_file.glob('preprocessed_dates_*.json'))
        
        if speeches_files and dates_files:
            speeches = load_json(str(max(speeches_files, key=lambda p: p.stat().st_mtime)))
            dates = load_json(str(max(dates_files, key=lambda p: p.stat().st_mtime)))
            
            if isinstance(speeches, list) and isinstance(dates, list):
                speech_dates = {s.get('date') for s in speeches if s.get('date')}
                date_set = {d.get('full_date') for d in dates if d.get('full_date')}
                
                missing_dates = speech_dates - date_set
                checks['speech_date_mapping'] = {
                    'speech_dates': len(speech_dates),
                    'date_dimension_dates': len(date_set),
                    'missing_dates': len(missing_dates),
                    'status': 'pass' if len(missing_dates) == 0 else 'warning'
                }
                
                if missing_dates:
                    print(f"⚠ {len(missing_dates)} speech dates not in date dimension")
                else:
                    print("✓ All speech dates present in date dimension")
        
        return checks
    
    def validate_data_ranges(self, data: List[Dict[str, Any]], field_ranges: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Validate data ranges"""
        checks = {}
        
        for field, range_def in field_ranges.items():
            min_val = range_def.get('min')
            max_val = range_def.get('max')
            
            values = [record.get(field) for record in data if record.get(field) is not None]
            
            if values:
                out_of_range = [v for v in values if (min_val is not None and v < min_val) or (max_val is not None and v > max_val)]
                checks[field] = {
                    'min_value': min(values),
                    'max_value': max(values),
                    'out_of_range_count': len(out_of_range),
                    'status': 'pass' if len(out_of_range) == 0 else 'fail'
                }
        
        return checks
    
    def validate_all(self) -> bool:
        """Run all validation checks"""
        print_section("DATA QUALITY VALIDATION")
        
        # Row counts
        self.quality_report['checks']['row_counts'] = self.validate_row_counts({})
        
        # Load preprocessed data for detailed checks
        speeches_file = Path("data/staging/preprocessed")
        speeches_files = list(speeches_file.glob('preprocessed_speeches_*.json'))
        
        if speeches_files:
            speeches = load_json(str(max(speeches_files, key=lambda p: p.stat().st_mtime)))
            if isinstance(speeches, list):
                # Null checks
                required_fields = ['speech_id', 'title', 'date']
                self.quality_report['checks']['null_checks'] = self.validate_null_checks(speeches, required_fields)
                
                # Data range checks
                field_ranges = {
                    'speech_surrogate_key': {'min': 1, 'max': None}
                }
                self.quality_report['checks']['data_ranges'] = self.validate_data_ranges(speeches, field_ranges)
        
        # Referential integrity
        self.quality_report['checks']['referential_integrity'] = self.validate_referential_integrity()
        
        # Generate summary
        self.quality_report['summary'] = self.generate_summary()
        
        # Print summary
        print_section("VALIDATION SUMMARY")
        summary = self.quality_report['summary']
        print(f"Total checks: {summary.get('total_checks', 0)}")
        print(f"Passed: {summary.get('passed', 0)}")
        print(f"Warnings: {summary.get('warnings', 0)}")
        print(f"Failed: {summary.get('failed', 0)}")
        
        return summary.get('failed', 0) == 0
    
    def generate_summary(self) -> Dict[str, Any]:
        """Generate validation summary"""
        summary = {
            'total_checks': 0,
            'passed': 0,
            'warnings': 0,
            'failed': 0
        }
        
        for check_type, check_data in self.quality_report['checks'].items():
            if isinstance(check_data, dict):
                if 'status' in check_data:
                    summary['total_checks'] += 1
                    status = check_data['status']
                    if status == 'pass':
                        summary['passed'] += 1
                    elif status == 'warning':
                        summary['warnings'] += 1
                    else:
                        summary['failed'] += 1
                elif isinstance(check_data, dict):
                    # Recursively check nested structures
                    for key, value in check_data.items():
                        if isinstance(value, dict) and 'status' in value:
                            summary['total_checks'] += 1
                            status = value['status']
                            if status == 'pass':
                                summary['passed'] += 1
                            elif status == 'warning':
                                summary['warnings'] += 1
                            else:
                                summary['failed'] += 1
        
        return summary
    
    def save_report(self, output_path: Optional[str] = None) -> bool:
        """Save quality report"""
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path("data/warehouse")
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"quality_report_{timestamp}.json"
        
        return save_json(self.quality_report, str(output_path))


def main():
    """Main execution function"""
    validator = DataQualityValidator()
    success = validator.validate_all()
    
    validator.save_report()
    
    if success:
        print("\n✓ Data quality validation passed")
        sys.exit(0)
    else:
        print("\n⚠ Data quality validation completed with warnings/failures")
        sys.exit(1)


if __name__ == "__main__":
    main()

