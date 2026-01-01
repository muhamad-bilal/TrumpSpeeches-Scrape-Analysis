"""
Run the Flask API Server
========================
Simple script to start the API server.

Usage:
    python run_api.py

The API will be available at http://localhost:5000
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from api.app import app
from api.models.data_loader import get_data_loader

if __name__ == '__main__':
    print("=" * 60)
    print("Trump Speech Analytics API")
    print("=" * 60)

    # Pre-load data
    print("\nLoading data...")
    loader = get_data_loader()
    _ = loader.entity_profiles
    _ = loader.features_df

    stats = loader.get_stats()
    print(f"  Entities loaded: {stats['entity_profiles_count']}")
    print(f"  Speeches loaded: {stats['speeches_count']}")

    print("\n" + "=" * 60)
    print("Starting server...")
    print("API available at: http://localhost:5000")
    print("Documentation at: http://localhost:5000/api/docs")
    print("=" * 60 + "\n")

    # Run the Flask app
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True
    )
