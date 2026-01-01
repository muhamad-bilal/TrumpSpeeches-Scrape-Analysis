"""
Data Loader Module
==================
Handles loading all data files required by the prediction models.
"""

import json
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """Centralized data loader for all model data."""

    def __init__(self, base_path: Optional[str] = None):
        """
        Initialize the data loader.

        Args:
            base_path: Base path to the data directory. If None, uses relative path.
        """
        if base_path is None:
            # Get the project root (parent of api folder)
            self.base_path = Path(__file__).parent.parent.parent
        else:
            self.base_path = Path(base_path)

        self.data_path = self.base_path / 'data'

        # Cached data
        self._entity_profiles: Optional[pd.DataFrame] = None
        self._features_df: Optional[pd.DataFrame] = None
        self._trigger_words: Optional[pd.DataFrame] = None
        self._analysis_results: Optional[Dict] = None
        self._entity_catalog: Optional[Dict] = None
        self._baseline_sentiment: Optional[float] = None

        logger.info(f"DataLoader initialized with base path: {self.base_path}")

    def _get_latest_file(self, directory: Path, pattern: str) -> Optional[Path]:
        """Get the most recently modified file matching a pattern."""
        files = list(directory.glob(pattern))
        if not files:
            logger.warning(f"No files found matching {pattern} in {directory}")
            return None
        return max(files, key=lambda p: p.stat().st_mtime)

    @property
    def entity_profiles(self) -> Optional[pd.DataFrame]:
        """Load entity reaction profiles."""
        if self._entity_profiles is None:
            results_dir = self.data_path / 'results'

            # Try JSON first
            json_file = self._get_latest_file(results_dir, 'entity_reaction_profiles_*.json')
            if json_file:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if 'profiles' in data:
                        self._entity_profiles = pd.DataFrame(data['profiles'])
                    else:
                        self._entity_profiles = pd.DataFrame(data)
                logger.info(f"Loaded entity profiles from {json_file.name}")
            else:
                # Try CSV
                csv_file = self._get_latest_file(results_dir, 'entity_reaction_profiles_*.csv')
                if csv_file:
                    self._entity_profiles = pd.read_csv(csv_file)
                    logger.info(f"Loaded entity profiles from {csv_file.name}")

            # Compute baseline sentiment
            if self._entity_profiles is not None and 'avg_sentiment' in self._entity_profiles.columns:
                self._baseline_sentiment = self._entity_profiles['avg_sentiment'].mean()

        return self._entity_profiles

    @property
    def baseline_sentiment(self) -> float:
        """Get baseline sentiment (loads entity profiles if needed)."""
        if self._baseline_sentiment is None:
            _ = self.entity_profiles  # This will compute baseline
        return self._baseline_sentiment or 0.5

    @property
    def features_df(self) -> Optional[pd.DataFrame]:
        """Load speech features data."""
        if self._features_df is None:
            transformed_dir = self.data_path / 'transformed'

            json_file = self._get_latest_file(transformed_dir, 'speeches_features_complete_*.json')
            if json_file:
                with open(json_file, 'r', encoding='utf-8') as f:
                    self._features_df = pd.DataFrame(json.load(f))
                logger.info(f"Loaded features from {json_file.name}")
            else:
                csv_file = self._get_latest_file(transformed_dir, 'speeches_features_complete_*.csv')
                if csv_file:
                    self._features_df = pd.read_csv(csv_file)
                    logger.info(f"Loaded features from {csv_file.name}")

        return self._features_df

    @property
    def trigger_words(self) -> Optional[pd.DataFrame]:
        """Load trigger words data."""
        if self._trigger_words is None:
            results_dir = self.data_path / 'results'

            csv_file = self._get_latest_file(results_dir, 'trigger_words_*.csv')
            if csv_file:
                self._trigger_words = pd.read_csv(csv_file)
                logger.info(f"Loaded trigger words from {csv_file.name}")

        return self._trigger_words

    @property
    def analysis_results(self) -> Optional[Dict]:
        """Load analysis results."""
        if self._analysis_results is None:
            results_dir = self.data_path / 'results'

            json_file = self._get_latest_file(results_dir, 'analysis_results_*.json')
            if json_file:
                with open(json_file, 'r', encoding='utf-8') as f:
                    self._analysis_results = json.load(f)
                logger.info(f"Loaded analysis results from {json_file.name}")

        return self._analysis_results

    @property
    def entity_catalog(self) -> Optional[Dict]:
        """Load entity catalog."""
        if self._entity_catalog is None:
            entities_dir = self.data_path / 'entities'

            json_file = self._get_latest_file(entities_dir, 'entity_catalog_*.json')
            if json_file:
                with open(json_file, 'r', encoding='utf-8') as f:
                    self._entity_catalog = json.load(f)
                logger.info(f"Loaded entity catalog from {json_file.name}")

        return self._entity_catalog

    def get_stats(self) -> Dict[str, Any]:
        """Get summary statistics about loaded data."""
        stats = {
            'base_path': str(self.base_path),
            'entity_profiles_loaded': self._entity_profiles is not None,
            'entity_profiles_count': len(self._entity_profiles) if self._entity_profiles is not None else 0,
            'features_loaded': self._features_df is not None,
            'speeches_count': len(self._features_df) if self._features_df is not None else 0,
            'trigger_words_loaded': self._trigger_words is not None,
            'trigger_words_count': len(self._trigger_words) if self._trigger_words is not None else 0,
            'baseline_sentiment': self._baseline_sentiment
        }
        return stats

    def reload_all(self):
        """Force reload all data from disk."""
        self._entity_profiles = None
        self._features_df = None
        self._trigger_words = None
        self._analysis_results = None
        self._entity_catalog = None
        self._baseline_sentiment = None
        logger.info("All cached data cleared. Will reload on next access.")


# Singleton instance for shared access
_data_loader: Optional[DataLoader] = None

def get_data_loader(base_path: Optional[str] = None) -> DataLoader:
    """Get the singleton DataLoader instance."""
    global _data_loader
    if _data_loader is None:
        _data_loader = DataLoader(base_path)
    return _data_loader
