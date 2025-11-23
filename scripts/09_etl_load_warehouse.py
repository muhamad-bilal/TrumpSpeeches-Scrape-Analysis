"""
Data Warehouse Load Script

Transforms staging data to warehouse format:
- Load dimension tables (SCD Type 1)
- Load fact table
- Load bridge tables
Generates SQL INSERT statements
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


class WarehouseLoader:
    """Load preprocessed data into warehouse format and generate SQL"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize warehouse loader"""
        self.config = load_config(config_path)
        self.preprocessed_dir = Path("data/staging/preprocessed")
        self.sql_output_dir = Path("sql")
        self.sql_output_dir.mkdir(parents=True, exist_ok=True)
        
        self.sql_statements = {
            'dimensions': [],
            'fact': [],
            'bridges': []
        }
        
        self.surrogate_key_mappings = {}
    
    def find_latest_preprocessed_file(self, pattern: str) -> Optional[Path]:
        """Find latest preprocessed file matching pattern"""
        files = list(self.preprocessed_dir.glob(pattern))
        if files:
            return max(files, key=lambda p: p.stat().st_mtime)
        return None
    
    def escape_sql_string(self, value: Any) -> str:
        """Escape string for SQL"""
        if value is None:
            return 'NULL'
        if isinstance(value, bool):
            return '1' if value else '0'
        if isinstance(value, (int, float)):
            return str(value)
        
        # Escape single quotes
        escaped = str(value).replace("'", "''")
        return f"'{escaped}'"
    
    def generate_dim_speech_sql(self, speeches: List[Dict[str, Any]]) -> List[str]:
        """Generate SQL INSERT statements for dim_speech"""
        statements = []
        
        for speech in speeches:
            values = [
                speech.get('speech_surrogate_key', 'NULL'),
                self.escape_sql_string(speech.get('title', '')),
                self.escape_sql_string(speech.get('url', '')),
                self.escape_sql_string(speech.get('location', '')),
                self.escape_sql_string(speech.get('speaker', 'Donald Trump')),
                self.escape_sql_string(speech.get('date', '')),
                speech.get('duration_estimate') if speech.get('duration_estimate') else 'NULL',
                self.escape_sql_string(speech.get('scraped_at', '')),
                self.escape_sql_string(speech.get('cleaned_at', '')),
                self.escape_sql_string(speech.get('transformed_at', ''))
            ]
            
            sql = f"""INSERT INTO dim_speech (
    speech_surrogate_key, title, url, location, speaker, 
    speech_date, duration_estimate, scraped_at, cleaned_at, transformed_at
) VALUES (
    {values[0]}, {values[1]}, {values[2]}, {values[3]}, {values[4]},
    {values[5]}, {values[6]}, {values[7]}, {values[8]}, {values[9]}
)
ON DUPLICATE KEY UPDATE 
    title = VALUES(title),
    url = VALUES(url),
    location = VALUES(location),
    speech_date = VALUES(speech_date);"""
            
            statements.append(sql)
        
        return statements
    
    def generate_dim_person_sql(self, persons: List[Dict[str, Any]]) -> List[str]:
        """Generate SQL INSERT statements for dim_person"""
        statements = []
        
        for person in persons:
            values = [
                person.get('person_surrogate_key', 'NULL'),
                self.escape_sql_string(person.get('person_name', '')),
                self.escape_sql_string(person.get('normalized_name', '')),
                self.escape_sql_string(person.get('first_mentioned_date', '')),
                self.escape_sql_string(person.get('last_mentioned_date', '')),
                person.get('mention_count', 0),
                person.get('speech_count', 0)
            ]
            
            sql = f"""INSERT INTO dim_person (
    person_surrogate_key, person_name, normalized_name,
    first_mentioned_date, last_mentioned_date, mention_count, speech_count
) VALUES (
    {values[0]}, {values[1]}, {values[2]},
    {values[3]}, {values[4]}, {values[5]}, {values[6]}
)
ON DUPLICATE KEY UPDATE 
    mention_count = mention_count + VALUES(mention_count),
    speech_count = GREATEST(speech_count, VALUES(speech_count)),
    last_mentioned_date = VALUES(last_mentioned_date);"""
            
            statements.append(sql)
        
        return statements
    
    def generate_dim_organization_sql(self, organizations: List[Dict[str, Any]]) -> List[str]:
        """Generate SQL INSERT statements for dim_organization"""
        statements = []
        
        for org in organizations:
            values = [
                org.get('org_surrogate_key', 'NULL'),
                self.escape_sql_string(org.get('org_name', '')),
                self.escape_sql_string(org.get('normalized_name', '')),
                self.escape_sql_string(org.get('org_type', '')),
                self.escape_sql_string(org.get('first_mentioned_date', '')),
                self.escape_sql_string(org.get('last_mentioned_date', '')),
                org.get('mention_count', 0),
                org.get('speech_count', 0)
            ]
            
            sql = f"""INSERT INTO dim_organization (
    org_surrogate_key, org_name, normalized_name, org_type,
    first_mentioned_date, last_mentioned_date, mention_count, speech_count
) VALUES (
    {values[0]}, {values[1]}, {values[2]}, {values[3]},
    {values[4]}, {values[5]}, {values[6]}, {values[7]}
)
ON DUPLICATE KEY UPDATE 
    mention_count = mention_count + VALUES(mention_count),
    speech_count = GREATEST(speech_count, VALUES(speech_count)),
    last_mentioned_date = VALUES(last_mentioned_date);"""
            
            statements.append(sql)
        
        return statements
    
    def generate_dim_location_sql(self, locations: List[Dict[str, Any]]) -> List[str]:
        """Generate SQL INSERT statements for dim_location"""
        statements = []
        
        for location in locations:
            values = [
                location.get('location_surrogate_key', 'NULL'),
                self.escape_sql_string(location.get('location_name', '')),
                self.escape_sql_string(location.get('normalized_name', '')),
                self.escape_sql_string(location.get('location_type', '')),
                self.escape_sql_string(location.get('country_code', '')),
                self.escape_sql_string(location.get('first_mentioned_date', '')),
                self.escape_sql_string(location.get('last_mentioned_date', '')),
                location.get('mention_count', 0),
                location.get('speech_count', 0)
            ]
            
            sql = f"""INSERT INTO dim_location (
    location_surrogate_key, location_name, normalized_name, location_type, country_code,
    first_mentioned_date, last_mentioned_date, mention_count, speech_count
) VALUES (
    {values[0]}, {values[1]}, {values[2]}, {values[3]}, {values[4]},
    {values[5]}, {values[6]}, {values[7]}, {values[8]}
)
ON DUPLICATE KEY UPDATE 
    mention_count = mention_count + VALUES(mention_count),
    speech_count = GREATEST(speech_count, VALUES(speech_count)),
    last_mentioned_date = VALUES(last_mentioned_date);"""
            
            statements.append(sql)
        
        return statements
    
    def generate_dim_date_sql(self, dates: List[Dict[str, Any]]) -> List[str]:
        """Generate SQL INSERT statements for dim_date"""
        statements = []
        
        for date_rec in dates:
            values = [
                date_rec.get('date_surrogate_key', 'NULL'),
                self.escape_sql_string(date_rec.get('full_date', '')),
                date_rec.get('year', 'NULL'),
                date_rec.get('quarter', 'NULL'),
                date_rec.get('month', 'NULL'),
                date_rec.get('day', 'NULL'),
                date_rec.get('day_of_week', 'NULL'),
                '1' if date_rec.get('is_weekend', False) else '0',
                date_rec.get('fiscal_year', 'NULL'),
                date_rec.get('fiscal_quarter', 'NULL')
            ]
            
            sql = f"""INSERT INTO dim_date (
    date_surrogate_key, full_date, year, quarter, month, day,
    day_of_week, is_weekend, fiscal_year, fiscal_quarter
) VALUES (
    {values[0]}, {values[1]}, {values[2]}, {values[3]}, {values[4]}, {values[5]},
    {values[6]}, {values[7]}, {values[8]}, {values[9]}
)
ON DUPLICATE KEY UPDATE full_date = VALUES(full_date);"""
            
            statements.append(sql)
        
        return statements
    
    def generate_fact_speech_metrics_sql(self, features_data: List[Dict[str, Any]], 
                                         speech_mapping: Dict[str, int],
                                         date_mapping: Dict[str, int]) -> List[str]:
        """Generate SQL INSERT statements for fact_speech_metrics"""
        statements = []
        
        for feature in features_data:
            speech_id = speech_mapping.get(feature.get('speech_id', ''))
            speech_date = feature.get('date', '')
            date_id = date_mapping.get(speech_date) if speech_date and speech_date in date_mapping else None
            
            # Allow NULL date_id if no dates available, but require speech_id
            if not speech_id:
                continue  # Skip if we can't map to speech dimension
            
            # Build values list
            values = [
                speech_id,
                date_id,
                feature.get('word_count'),
                feature.get('sentence_count'),
                feature.get('char_count'),
                feature.get('sentiment_positive'),
                feature.get('sentiment_negative'),
                feature.get('sentiment_neutral'),
                feature.get('sentiment_compound'),
                feature.get('emotion_anger'),
                feature.get('emotion_fear'),
                feature.get('emotion_joy'),
                feature.get('emotion_sadness'),
                feature.get('emotion_surprise'),
                feature.get('emotion_disgust'),
                feature.get('emotion_trust'),
                feature.get('emotion_anticipation'),
                feature.get('flesch_kincaid_grade'),
                feature.get('flesch_reading_ease'),
                feature.get('gunning_fog'),
                feature.get('smog_index'),
                feature.get('coleman_liau_index'),
                feature.get('avg_sentence_length'),
                feature.get('type_token_ratio'),
                feature.get('lexical_diversity'),
                feature.get('unique_word_count'),
                feature.get('anaphora_count'),
                feature.get('repetition_density'),
                feature.get('contrast_marker_count'),
                feature.get('superlative_count'),
                feature.get('keywords_economy'),
                feature.get('keywords_security'),
                feature.get('keywords_immigration'),
                feature.get('keywords_foreign_policy'),
                feature.get('keywords_total'),
                feature.get('pronoun_i_count'),
                feature.get('pronoun_we_count'),
                feature.get('pronoun_you_count'),
                feature.get('pronoun_ratio_i_we'),
                feature.get('modal_verb_count'),
                feature.get('power_words'),
                feature.get('affiliation_words'),
                feature.get('power_affiliation_ratio'),
                feature.get('ner_person_count'),
                feature.get('ner_org_count'),
                feature.get('ner_gpe_count'),
                feature.get('ner_total_entities'),
                feature.get('adj_count'),
                feature.get('adv_count'),
                feature.get('question_count'),
                feature.get('exclamation_count')
            ]
            
            # Convert to SQL format
            sql_values = ', '.join([str(v) if v is not None else 'NULL' for v in values])
            
            sql = f"""INSERT INTO fact_speech_metrics (
    speech_id, date_id, word_count, sentence_count, char_count,
    sentiment_positive, sentiment_negative, sentiment_neutral, sentiment_compound,
    emotion_anger, emotion_fear, emotion_joy, emotion_sadness,
    emotion_surprise, emotion_disgust, emotion_trust, emotion_anticipation,
    flesch_kincaid_grade, flesch_reading_ease, gunning_fog, smog_index, coleman_liau_index,
    avg_sentence_length, type_token_ratio, lexical_diversity, unique_word_count,
    anaphora_count, repetition_density, contrast_marker_count, superlative_count,
    keywords_economy, keywords_security, keywords_immigration, keywords_foreign_policy, keywords_total,
    pronoun_i_count, pronoun_we_count, pronoun_you_count, pronoun_ratio_i_we,
    modal_verb_count, power_words, affiliation_words, power_affiliation_ratio,
    ner_person_count, ner_org_count, ner_gpe_count, ner_total_entities,
    adj_count, adv_count, question_count, exclamation_count
) VALUES ({sql_values})
ON DUPLICATE KEY UPDATE 
    word_count = VALUES(word_count),
    sentiment_compound = VALUES(sentiment_compound);"""
            
            statements.append(sql)
        
        return statements
    
    def load_all(self) -> bool:
        """Load all preprocessed data and generate SQL"""
        print_section("WAREHOUSE LOAD PIPELINE")
        
        # Load preprocessed data
        speeches_file = self.find_latest_preprocessed_file('preprocessed_speeches_*.json')
        dates_file = self.find_latest_preprocessed_file('preprocessed_dates_*.json')
        entities_file = self.find_latest_preprocessed_file('preprocessed_entities_*.json')
        surrogate_keys_file = self.find_latest_preprocessed_file('surrogate_keys_*.json')
        
        # Load features data from staging
        staging_dir = Path("data/staging")
        features_files = list(staging_dir.glob('features_data_*.json'))
        features_file = max(features_files, key=lambda p: p.stat().st_mtime) if features_files else None
        
        if not speeches_file or not dates_file:
            print("Error: Missing required preprocessed files")
            return False
        
        # Load data
        speeches = load_json(str(speeches_file))
        dates = load_json(str(dates_file))
        
        features_data = []
        if features_file:
            features_data = load_json(str(features_file))
            if isinstance(features_data, list):
                pass
            else:
                features_data = [features_data]
        
        # Load surrogate key mappings
        if surrogate_keys_file:
            self.surrogate_key_mappings = load_json(str(surrogate_keys_file))
        
        # Create mappings for fact table
        # Map speech_id from features to speech_surrogate_key from preprocessed speeches
        speech_mapping = {}
        for speech in speeches:
            speech_id_key = speech.get('speech_id', '')
            speech_surrogate = speech.get('speech_surrogate_key', 0)
            if speech_id_key:
                speech_mapping[speech_id_key] = speech_surrogate
        
        # Map dates - if no dates, create a default mapping or allow NULL dates
        date_mapping = {}
        if dates:
            date_mapping = {d['full_date']: d['date_surrogate_key'] for d in dates if d.get('full_date')}
        else:
            # If no dates, we'll use NULL for date_id (need to handle this in SQL)
            print("Warning: No dates found. Fact table records will use NULL date_id")
        
        # Generate SQL statements
        print_section("GENERATING SQL STATEMENTS")
        
        # Dimensions
        self.sql_statements['dimensions'].extend(self.generate_dim_speech_sql(speeches))
        if dates:  # Only add dates if they exist
            self.sql_statements['dimensions'].extend(self.generate_dim_date_sql(dates))
        
        if entities_file:
            entities = load_json(str(entities_file))
            if entities:
                self.sql_statements['dimensions'].extend(
                    self.generate_dim_person_sql(entities.get('persons', []))
                )
                self.sql_statements['dimensions'].extend(
                    self.generate_dim_organization_sql(entities.get('organizations', []))
                )
                self.sql_statements['dimensions'].extend(
                    self.generate_dim_location_sql(entities.get('locations', []))
                )
        
        # Fact table
        if features_data:
            print(f"Processing {len(features_data)} feature records...")
            print(f"Speech mapping has {len(speech_mapping)} entries")
            print(f"Date mapping has {len(date_mapping)} entries")
            
            fact_statements = self.generate_fact_speech_metrics_sql(
                features_data, speech_mapping, date_mapping
            )
            self.sql_statements['fact'].extend(fact_statements)
            print(f"Generated {len(fact_statements)} fact table INSERT statements")
        
        # Save SQL file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sql_file = self.sql_output_dir / f"04_load_data_{timestamp}.sql"
        
        with open(sql_file, 'w', encoding='utf-8') as f:
            f.write("-- ============================================================================\n")
            f.write("-- Data Warehouse Load Script\n")
            f.write("-- Generated: " + datetime.now().isoformat() + "\n")
            f.write("-- ============================================================================\n\n")
            
            f.write("-- Dimension Tables\n")
            f.write("-- ============================================================================\n\n")
            for stmt in self.sql_statements['dimensions']:
                f.write(stmt + "\n\n")
            
            f.write("\n-- Fact Table\n")
            f.write("-- ============================================================================\n\n")
            for stmt in self.sql_statements['fact']:
                f.write(stmt + "\n\n")
            
            f.write("\n-- Bridge Tables\n")
            f.write("-- ============================================================================\n")
            f.write("-- Bridge tables will be populated based on entity relationships\n")
            f.write("-- See entity_relationships data for co-occurrence information\n\n")
        
        print(f"✓ Generated SQL file: {sql_file}")
        print(f"  - {len(self.sql_statements['dimensions'])} dimension INSERT statements")
        print(f"  - {len(self.sql_statements['fact'])} fact INSERT statements")
        
        return True


def main():
    """Main execution function"""
    loader = WarehouseLoader()
    success = loader.load_all()
    
    if success:
        print("\n✓ Warehouse load SQL generation completed successfully")
        sys.exit(0)
    else:
        print("\n✗ Warehouse load SQL generation failed")
        sys.exit(1)


if __name__ == "__main__":
    main()

