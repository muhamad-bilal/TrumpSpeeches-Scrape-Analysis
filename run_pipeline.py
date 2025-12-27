"""
Master Pipeline Script for Trump Speech Analysis

This script runs the complete analysis pipeline from cleaned data to final results.
Usage: python run_pipeline.py <input_file>
"""

import sys
import subprocess
from pathlib import Path
import argparse
from datetime import datetime


def print_step(step_num, title):
    """Print formatted step header"""
    print("\n" + "=" * 70)
    print(f"STEP {step_num}: {title}")
    print("=" * 70 + "\n")


def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}\n")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        print(f"\n✓ {description} completed successfully\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ {description} failed with error code {e.returncode}\n")
        return False


def find_latest_file(directory, pattern):
    """Find the most recent file matching a pattern"""
    files = list(Path(directory).glob(pattern))
    if files:
        return max(files, key=lambda p: p.stat().st_mtime)
    return None


def main():
    parser = argparse.ArgumentParser(
        description='Run the complete Trump speech analysis pipeline'
    )
    parser.add_argument(
        'input_file',
        nargs='?',
        help='Input file with raw scraped transcripts (JSON or CSV). If not provided, will look for latest in data/raw/'
    )
    parser.add_argument(
        '--skip-cleaning',
        action='store_true',
        help='Skip data cleaning (use existing cleaned data)'
    )
    parser.add_argument(
        '--skip-transformation',
        action='store_true',
        help='Skip NLP transformation (use existing transformed data)'
    )
    parser.add_argument(
        '--skip-features',
        action='store_true',
        help='Skip feature engineering (use existing features)'
    )
    
    args = parser.parse_args()
    
    # Print header
    print("\n" + "=" * 70)
    print("TRUMP SPEECH ANALYSIS PIPELINE")
    print("=" * 70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # Determine input file
    if args.input_file:
        input_file = args.input_file
    else:
        print("\nNo input file specified. Looking for latest in data/raw/...")
        input_file = find_latest_file('data/raw', 'trump_speeches_*.json')
        if not input_file:
            input_file = find_latest_file('data/raw', 'trump_speeches_*.csv')
        
        if not input_file:
            print("Error: No input files found in data/raw/")
            print("Please run the scraper first or specify an input file.")
            sys.exit(1)
        
        print(f"✓ Found: {input_file}")
    
    # Verify input file exists
    if not Path(input_file).exists():
        print(f"Error: Input file not found: {input_file}")
        sys.exit(1)
    
    # Step 1: Data Cleaning
    if not args.skip_cleaning:
        print_step(1, "DATA CLEANING")
        
        success = run_command(
            ['python', 'scripts/01_data_cleaning.py', input_file],
            "Data cleaning"
        )
        
        if not success:
            print("Pipeline failed at cleaning step")
            sys.exit(1)
        
        # Find the cleaned output
        cleaned_file = find_latest_file('data/cleaned', 'speeches_cleaned_*.json')
    else:
        print_step(1, "SKIPPING DATA CLEANING (using existing data)")
        cleaned_file = find_latest_file('data/cleaned', 'speeches_cleaned_*.json')
    
    if not cleaned_file:
        print("Error: No cleaned data found")
        sys.exit(1)
    
    print(f"Using cleaned data: {cleaned_file}")
    
    # Step 2: NLP Transformation
    if not args.skip_transformation:
        print_step(2, "NLP TRANSFORMATION")
        
        success = run_command(
            ['python', 'scripts/02_data_transformation.py', str(cleaned_file)],
            "NLP transformation"
        )
        
        if not success:
            print("Pipeline failed at transformation step")
            sys.exit(1)
        
        # Find the transformed output
        transformed_file = find_latest_file('data/transformed', 'speeches_nlp_features_*.json')
    else:
        print_step(2, "SKIPPING NLP TRANSFORMATION (using existing data)")
        transformed_file = find_latest_file('data/transformed', 'speeches_nlp_features_*.json')
    
    if not transformed_file:
        print("Error: No transformed data found")
        sys.exit(1)
    
    print(f"Using transformed data: {transformed_file}")
    
    # Step 3: Feature Engineering
    if not args.skip_features:
        print_step(3, "FEATURE ENGINEERING")
        
        success = run_command(
            ['python', 'scripts/03_feature_engineering.py', str(transformed_file)],
            "Feature engineering"
        )
        
        if not success:
            print("Pipeline failed at feature engineering step")
            sys.exit(1)
        
        # Find the features output
        features_csv = find_latest_file('data/transformed', 'speeches_features_complete_*.csv')
    else:
        print_step(3, "SKIPPING FEATURE ENGINEERING (using existing data)")
        features_csv = find_latest_file('data/transformed', 'speeches_features_complete_*.csv')
    
    if not features_csv:
        print("Error: No feature data found")
        sys.exit(1)
    
    print(f"Using feature data: {features_csv}")
    
    # Step 4: Analysis Suite
    print_step(4, "COMPREHENSIVE ANALYSIS")
    
    success = run_command(
        ['python', 'scripts/04_analysis_suite.py', str(features_csv), 
         '--transformed', str(transformed_file)],
        "Comprehensive analysis"
    )
    
    if not success:
        print("Pipeline failed at analysis step")
        sys.exit(1)
    
    # Find the results output
    results_file = find_latest_file('data/results', 'analysis_results_*.json')
    
    # Print summary
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nOutput files:")
    print(f"  Cleaned data: {cleaned_file}")
    print(f"  Transformed data: {transformed_file}")
    print(f"  Features: {features_csv}")
    if results_file:
        print(f"  Analysis results: {results_file}")
    
    print("\nNext steps:")
    print("  - Open Jupyter notebooks for interactive analysis")
    print("  - Check data/results/ for detailed analysis outputs")
    print("  - Run notebooks/05_visualization_dashboard.ipynb for visualizations")
    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()

