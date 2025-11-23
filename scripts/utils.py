"""
Utility functions for Trump speech analysis pipeline
"""
import json
import csv
import re
import yaml
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd


def load_config(config_path: str = "config.yaml") -> Dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_json(file_path: str) -> Any:
    """Load data from JSON file with error handling"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found - {file_path}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {file_path} - {e}")
        return None


def save_json(data: Any, file_path: str, indent: int = 2) -> bool:
    """Save data to JSON file with error handling"""
    try:
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
        print(f"✓ Saved to {file_path}")
        return True
    except Exception as e:
        print(f"Error saving JSON to {file_path}: {e}")
        return False


def load_csv(file_path: str) -> Optional[pd.DataFrame]:
    """Load CSV file into pandas DataFrame"""
    try:
        return pd.read_csv(file_path, encoding='utf-8')
    except FileNotFoundError:
        print(f"Error: File not found - {file_path}")
        return None
    except Exception as e:
        print(f"Error loading CSV {file_path}: {e}")
        return None


def save_csv(df: pd.DataFrame, file_path: str) -> bool:
    """Save DataFrame to CSV file"""
    try:
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(file_path, index=False, encoding='utf-8')
        print(f"✓ Saved to {file_path}")
        return True
    except Exception as e:
        print(f"Error saving CSV to {file_path}: {e}")
        return False


def get_stopwords(config: Dict) -> set:
    """Get combined stop words from NLTK and custom config"""
    try:
        from nltk.corpus import stopwords
        import nltk
        
        # Download if not already present
        try:
            stop_words = set(stopwords.words('english'))
        except LookupError:
            nltk.download('stopwords', quiet=True)
            stop_words = set(stopwords.words('english'))
        
        # Add custom Trump-specific stop words
        if 'nlp' in config and 'custom_stopwords' in config['nlp']:
            custom = config['nlp']['custom_stopwords']
            stop_words.update(custom)
        
        return stop_words
    except Exception as e:
        print(f"Warning: Could not load stopwords - {e}")
        return set()


def clean_whitespace(text: str) -> str:
    """Normalize whitespace in text"""
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)
    # Remove leading/trailing whitespace
    text = text.strip()
    return text


def normalize_quotes(text: str) -> str:
    """Normalize various quote characters"""
    # Replace curly quotes with straight quotes
    text = text.replace('"', '"').replace('"', '"')
    text = text.replace(''', "'").replace(''', "'")
    # Normalize apostrophes
    text = re.sub(r'[\u2018\u2019]', "'", text)
    return text


def remove_html_tags(text: str) -> str:
    """Remove HTML tags from text"""
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)


def extract_anaphora_patterns(text: str, patterns: List[str]) -> Dict[str, int]:
    """
    Extract anaphora patterns from text
    
    Args:
        text: Input text
        patterns: List of regex patterns to search for
    
    Returns:
        Dictionary mapping pattern to count
    """
    results = {}
    sentences = text.split('.')
    
    for pattern in patterns:
        count = 0
        regex = re.compile(pattern, re.IGNORECASE)
        for sentence in sentences:
            sentence = sentence.strip()
            if regex.match(sentence):
                count += 1
        results[pattern] = count
    
    return results


def detect_contrast_markers(text: str) -> int:
    """Count contrast markers (but, however, while, although, etc.)"""
    contrast_words = ['but', 'however', 'although', 'though', 'yet', 
                      'while', 'whereas', 'nevertheless', 'nonetheless']
    
    text_lower = text.lower()
    count = 0
    
    for word in contrast_words:
        # Use word boundaries to avoid partial matches
        pattern = r'\b' + word + r'\b'
        count += len(re.findall(pattern, text_lower))
    
    return count


def calculate_repetition_density(text: str, window: int = 1000) -> float:
    """
    Calculate repetition density per 1000 words
    
    Args:
        text: Input text
        window: Window size for calculation (default 1000 words)
    
    Returns:
        Percentage of repeated words per window
    """
    words = text.lower().split()
    
    if len(words) < 2:
        return 0.0
    
    # Count repeated consecutive words
    repeated = 0
    for i in range(len(words) - 1):
        if words[i] == words[i + 1]:
            repeated += 1
    
    # Calculate as percentage per window
    density = (repeated / len(words)) * window
    return density


def extract_modal_verbs(text: str) -> Dict[str, int]:
    """Count modal verbs (will, should, must, can, etc.)"""
    modals = ['will', 'would', 'should', 'shall', 'must', 'can', 
              'could', 'may', 'might', 'ought']
    
    text_lower = text.lower()
    results = {}
    
    for modal in modals:
        pattern = r'\b' + modal + r'\b'
        results[modal] = len(re.findall(pattern, text_lower))
    
    return results


def calculate_type_token_ratio(words: List[str]) -> float:
    """
    Calculate type-token ratio (lexical diversity)
    
    Args:
        words: List of words
    
    Returns:
        Ratio of unique words to total words
    """
    if not words:
        return 0.0
    
    unique_words = len(set(words))
    total_words = len(words)
    
    return unique_words / total_words


def extract_pronouns(text: str) -> Dict[str, int]:
    """
    Extract pronoun counts for psychological analysis
    
    Returns:
        Dictionary with pronoun counts (I, we, you, they, he, she, it)
    """
    pronouns = {
        'first_singular': ['i', "i'm", "i've", "i'll", "i'd"],
        'first_plural': ['we', "we're", "we've", "we'll", "we'd"],
        'second_person': ['you', "you're", "you've", "you'll", "you'd"],
        'third_plural': ['they', "they're", "they've", "they'll", "they'd"],
        'third_singular': ['he', 'she', 'it', "he's", "she's", "it's"]
    }
    
    text_lower = text.lower()
    results = {}
    
    for category, words in pronouns.items():
        count = 0
        for word in words:
            pattern = r'\b' + re.escape(word) + r'\b'
            count += len(re.findall(pattern, text_lower))
        results[category] = count
    
    return results


def validate_utf8(text: str) -> bool:
    """Validate that text is valid UTF-8"""
    try:
        text.encode('utf-8').decode('utf-8')
        return True
    except UnicodeError:
        return False


def create_speech_id(title: str, date: str) -> str:
    """
    Create a unique speech ID from title and date
    
    Args:
        title: Speech title
        date: Speech date
    
    Returns:
        Unique identifier (slug format)
    """
    # Combine title and date
    combined = f"{date}_{title}"
    
    # Convert to lowercase and replace spaces/special chars with hyphens
    speech_id = re.sub(r'[^\w\s-]', '', combined.lower())
    speech_id = re.sub(r'[-\s]+', '-', speech_id)
    speech_id = speech_id.strip('-')
    
    return speech_id


def print_section(title: str, width: int = 70):
    """Print a formatted section header"""
    print("\n" + "=" * width)
    print(title.center(width))
    print("=" * width)


def print_stats(label: str, value: Any, indent: int = 2):
    """Print formatted statistics"""
    spacing = " " * indent
    print(f"{spacing}{label}: {value}")


if __name__ == "__main__":
    # Test utilities
    print("Testing utility functions...")
    
    # Test config loading
    try:
        config = load_config()
        print("✓ Config loaded successfully")
    except:
        print("✗ Config loading failed")
    
    # Test text cleaning
    test_text = "This  is   a    test"
    cleaned = clean_whitespace(test_text)
    assert cleaned == "This is a test"
    print("✓ Whitespace cleaning works")
    
    # Test quote normalization
    test_quotes = '''"Hello" and 'world' '''
    normalized = normalize_quotes(test_quotes)
    print(f"✓ Quote normalization works: {normalized}")
    
    print("\n✓ All utility tests passed!")

