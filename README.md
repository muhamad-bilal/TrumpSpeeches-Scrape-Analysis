# Trump Speech Analysis Pipeline

A comprehensive NLP analysis pipeline for analyzing Donald Trump's speeches, implementing linguistic, rhetorical, emotional, and psychological profiling techniques.

## Project Overview

This project provides end-to-end analysis of Trump's speeches from Rev.com transcripts, including:
- Data cleaning and preprocessing
- Advanced NLP transformations
- Feature engineering
- Multi-dimensional analysis
- Interactive and publication-ready visualizations

## Directory Structure

```
.
├── data/
│   ├── raw/              # Scraped transcript files
│   ├── cleaned/          # Cleaned transcripts
│   ├── transformed/      # NLP-enriched data
│   └── results/          # Analysis outputs
├── scripts/
│   ├── utils.py                      # Utility functions
│   ├── 01_data_cleaning.py          # Data cleaning pipeline
│   ├── 02_data_transformation.py    # NLP transformation pipeline
│   ├── 03_feature_engineering.py    # Feature extraction
│   └── 04_analysis_suite.py         # Comprehensive analysis
├── notebooks/
│   ├── 01_exploratory_analysis.ipynb
│   ├── 02_linguistic_analysis.ipynb
│   ├── 03_rhetorical_analysis.ipynb
│   ├── 04_emotional_analysis.ipynb
│   └── 05_visualization_dashboard.ipynb
├── config.yaml           # Configuration settings
├── requirements.txt      # Python dependencies
└── README.md

```

## Installation

### 1. Clone or download this repository

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Download required NLP models

```bash
# Download spaCy model (transformer-based for best results)
python -m spacy download en_core_web_trf

# Or use smaller model if resources are limited
python -m spacy download en_core_web_sm
```

## Usage

### Pipeline Execution Order

Run the scripts in sequence:

#### Step 1: Scrape Data (if not already done)

```bash
python scrape_trump_specific.py
```

This will scrape transcripts from Rev.com and save to `data/raw/`.

#### Step 2: Clean Data

```bash
python scripts/01_data_cleaning.py data/raw/trump_speeches_XXXXXX.json
```

**What it does:**
- Removes HTML tags, metadata, timestamps
- Removes crowd reactions (applause, cheers, etc.)
- Standardizes speaker tags
- Normalizes punctuation and whitespace
- Removes duplicates and noise

**Output:** `data/cleaned/speeches_cleaned_TIMESTAMP.json` and `.csv`

#### Step 3: Transform Data with NLP

```bash
python scripts/02_data_transformation.py data/cleaned/speeches_cleaned_XXXXXX.json
```

**What it does:**
- Sentence segmentation and tokenization
- POS tagging and lemmatization
- Named Entity Recognition
- Sentiment analysis (VADER + transformers)
- Emotion classification
- Readability metrics
- N-gram extraction
- TF-IDF vectorization
- Embedding generation

**Output:** `data/transformed/speeches_nlp_features_TIMESTAMP.json`

#### Step 4: Engineer Features

```bash
python scripts/03_feature_engineering.py data/transformed/speeches_nlp_features_XXXXXX.json
```

**What it does:**
- Extracts 100+ linguistic, rhetorical, emotional, psychological features
- Computes derived metrics
- Creates analysis-ready dataset

**Output:** `data/transformed/speeches_features_complete_TIMESTAMP.csv` and `.json`

#### Step 5: Run Analysis Suite

```bash
python scripts/04_analysis_suite.py data/transformed/speeches_features_complete_XXXXXX.csv --transformed data/transformed/speeches_nlp_features_XXXXXX.json
```

**What it does:**
- Linguistic analysis (complexity, diversity, readability)
- Rhetorical analysis (anaphora, contrast, repetition)
- Political/thematic analysis (topic modeling, keyword clustering)
- Emotional analysis (sentiment trends, emotion distribution)
- Psychological profiling (pronouns, power language, confidence)
- Comparative trend analysis

**Output:** `data/results/analysis_results_TIMESTAMP.json`

## Analysis Components

### 1. Data Cleaning
- HTML & metadata removal
- Stop-word filtering (with Trump-specific additions)
- Speaker tag standardization
- Whitespace & punctuation normalization
- Duplicate & error removal
- UTF-8 encoding validation

### 2. Data Transformations
- **NLP Processing:** spaCy (en_core_web_trf) + transformers
- **Sentiment:** VADER + RoBERTa
- **Emotion:** NRCLex + DistilBERT (8 emotions)
- **Readability:** Flesch-Kincaid, Gunning Fog, SMOG, etc.
- **N-grams:** Unigrams, bigrams, trigrams
- **Embeddings:** sentence-transformers (all-MiniLM-L6-v2)

### 3. Feature Engineering

**Basic Attributes:**
- speech_id, title, date, location, speaker
- word_count, sentence_count, duration_estimate

**Derived Features:**
- **Linguistic:** sentence length, type-token ratio, lexical diversity, readability
- **Rhetorical:** anaphora patterns, repetition density, contrast markers
- **Political:** keyword clusters (economy, security, immigration, foreign policy)
- **Emotional:** sentiment scores, emotion distributions, volatility
- **Psychological:** pronoun ratios, modal verbs, power/affiliation ratio, certainty
- **NER:** top entities (people, organizations, countries)
- **Stylistic:** adjective/adverb ratio, questions, exclamations, superlatives

### 4. Analysis Suite

1. **Linguistic Analysis** - Complexity and vocabulary metrics
2. **Rhetorical Analysis** - Detection of rhetorical devices
3. **Political & Thematic Analysis** - Topic modeling (LDA + BERTopic)
4. **Emotional Analysis** - Sentiment timelines and emotion profiling
5. **Psychological Profiling** - Language-based personality insights
6. **Comparative Trend Analysis** - Temporal changes and patterns

### 5. Visualizations

**Interactive (Plotly):**
- Sentiment timeline with annotations
- Topic evolution diagrams
- Entity networks

**Publication-Ready (Matplotlib/Seaborn):**
- Word clouds
- Emotion heatmaps
- Readability trend lines
- N-gram frequency charts
- PCA/t-SNE embeddings

## Jupyter Notebooks

Explore the analysis interactively:

1. **01_exploratory_analysis.ipynb** - Initial data exploration
2. **02_linguistic_analysis.ipynb** - Deep dive into linguistic features
3. **03_rhetorical_analysis.ipynb** - Rhetorical device analysis
4. **04_emotional_analysis.ipynb** - Sentiment and emotion patterns
5. **05_visualization_dashboard.ipynb** - Comprehensive visualizations

## Configuration

Edit `config.yaml` to customize:
- File paths
- Stop words (including Trump-specific)
- NLP model settings
- Analysis parameters
- Visualization settings

## Expected Outcomes

1. **Quantitative Portrait** - Data-driven profile of Trump's rhetorical style
2. **Thematic Analysis** - Dominant themes and their evolution
3. **Emotional Insights** - Sentiment patterns and emotional shifts
4. **Psychological Profile** - Persuasion tactics and language use patterns
5. **Temporal Trends** - How speech characteristics change over time
6. **Benchmark Dataset** - Reusable workflow for similar analyses

## Key Features

- ✅ Fully automated pipeline
- ✅ Modular, reusable scripts
- ✅ Comprehensive feature extraction (100+ features)
- ✅ Multiple analysis dimensions
- ✅ Both interactive and static visualizations
- ✅ Configurable via YAML
- ✅ Progress tracking and error handling
- ✅ CSV and JSON output formats

## Technical Stack

- **NLP:** spaCy, NLTK, transformers, sentence-transformers
- **Sentiment:** VADER, RoBERTa
- **Emotion:** NRCLex, DistilBERT
- **Topic Modeling:** Gensim (LDA), BERTopic
- **ML:** scikit-learn, scipy
- **Visualization:** Plotly, Matplotlib, Seaborn, WordCloud
- **Data:** pandas, numpy

## Troubleshooting

### Memory Issues
- Use smaller spaCy model: `en_core_web_sm` instead of `en_core_web_trf`
- Process speeches in batches
- Reduce embedding dimensions

### Missing Models
```bash
python -m spacy download en_core_web_sm
python -m nltk.downloader punkt stopwords
```

### Import Errors
```bash
pip install --upgrade -r requirements.txt
```

## Citation

If you use this pipeline in academic work, please cite appropriately and reference the methodology.

## License

This project is for educational and research purposes.

## Authors

Data analysis pipeline implementing methodologies inspired by computational linguistics and political communication research.

---

**Note:** Ensure you have scraped data in `data/raw/` before running the pipeline. The scraper must be run separately to collect transcripts from Rev.com.

