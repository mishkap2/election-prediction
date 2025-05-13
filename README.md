# Indian Election Sentiment Analysis Project

<img width="1420" alt="WhatsApp Image 2025-05-14 at 1 08 04 AM" src="https://github.com/user-attachments/assets/c95bdfd3-d976-451a-8b78-3903d539ddc7" />


## Overview

The Indian Election Sentiment Analysis Project aims to predict election outcomes by analyzing public sentiment toward key political entities (Narendra Modi, BJP, Rahul Gandhi, Congress) from multilingual news articles and, in future iterations, YouTube transcriptions. Unlike the plethora of English-only sentiment analysis tools, this project prioritizes India’s linguistic diversity, processing texts in English, Hindi, Bengali, Assamese, Gujarati, and Marathi. By leveraging advanced NLP models and context-aware sentiment attribution, we provide insights into voter sentiment, with a focus on scalability and real-world applicability in India’s complex media landscape.
Our current proof-of-concept (PoC) demonstrates robust sentiment analysis on a dataset of ews articles, achieving an accuracy of 0.8 in classifying article-level favorability. The project is designed to scale to larger datasets and incorporate diverse data sources, such as YouTube videos, to capture spoken political discourse. The multilingual focus ensures accessibility and relevance in India’s heterogeneous linguistic environment, setting us apart from English-centric solutions.
Current Achievements

- Dataset Processing: Successfully preprocessed news articles from newsdata.io, covering Modi/BJP and Gandhi/Congress, with columns article_id, text, language, tokens, entities, sentiment, normalized_text.
- Multilingual NLP Pipeline:
- Tokenization: Implemented mBERT (bert-base-multilingual-cased) for tokenizing texts across English, Hindi, and regional languages.
- Named Entity Recognition (NER): Utilized xlm-roberta-large-finetuned-conll03-english with aggregation_strategy="simple" to detect entities (e.g., "Narendra Modi", "BJP"), mapping informal names like "Modi" to standardized forms.
- Sentiment Analysis: Applied nlptown/bert-base-multilingual-uncased-sentiment with rule-based validation for Hindi, addressing negative sentiment skew in political texts.
- Normalization: Used indic-nlp-library for Hindi and regional language text normalization, ensuring consistent preprocessing.
- Context-Aware Sentiment Attribution:
  - Developed rules to assign negative sentiment correctly (e.g., BJP’s criticism of "प्रदेश सरकार" targets Congress, not BJP).
  - Handled implicit entity mentions.
- GPU Acceleration: Implemented GPU support (device=0) for NER and sentiment analysis, optimizing performance for large datasets.

## What We Have Implemented

### Data Pipeline

- Input: News articles from newsdata.io, queried for Modi/BJP and Gandhi/Congress, stored as CSVs (data/raw/).
- Preprocessing (`preprocess.py`):
  - Combines title and description into text.
  - Detects and maps languages (English → en, Hindi → hi, etc.).
  - Normalizes text using indic-nlp-library for Indian languages and basic cleaning for English.
  - Tokenizes with mBERT, ensuring compatibility with multilingual models.
  - Extracts entities with XLM-RoBERTa, mapping "Modi" variants to "Narendra Modi".
  - Computes sentiment with a multilingual model, validated by Hindi-specific rules (e.g., "विकास" → positive).
  - Outputs a CSV with `article_id`, `text`, `language`, `tokens`, `entities`, `sentiment`, `normalized_text`.

- Output: Preprocessed CSVs in `data/processed/` (e.g., `preprocessed_news_20250513_223014.csv`). Samples can be viewed in `/demo-data`
- Feature Extraction (train.py):
  - Extracts entity-specific sentiment scores (Narendra Modi, BJP, Rahul Gandhi, Congress) using context-aware rules.
  - Incorporates text-based heuristics for implicit mentions (e.g., "सरकार" → Congress).

- Label Simulation: Generates article-level labels (BJP=1, Congress=0) with a minimum 30% Congress ratio, using entity counts and text keywords.
- Classification: Trains a Logistic Regression model to predict article favorability, achieving 0.8 accuracy.
- Outcome Prediction: Aggregates entity sentiments to predict the election winner (e.g., BJP with 0.2500 vs. Congress’s -0.8000 in demo).
- Results: Saves detailed logs in `data/results/`.

## Multilingual Focus

English sentiment analysis tools dominate, but India’s electorate engages with media in Hindi (43% of population), Bengali, Marathi, and other languages. Our project captures this diversity, ensuring equitable sentiment analysis.
Implementation:
- mBERT tokenization supports six languages (English, Hindi, Bengali, Assamese, Gujarati, Marathi).
- XLM-RoBERTa NER handles multilingual entity detection with high accuracy.
- indic-nlp-library normalizes Indian scripts (e.g., Devanagari for Hindi).
- Rule-based sentiment validation addresses Hindi-specific nuances (e.g., "हमलावर" → negative).


Impact: Unlike English-only tools, our pipeline processes texts like "प्रदेश सरकार पर हमलावर हुए BJP" correctly, attributing negativity to Congress, not BJP.


## How We Arrived at Our Solution
### Problem Identification
Goal: Predict Indian election outcomes by analyzing sentiment toward key political entities.
Challenge: India’s multilingual media landscape (Hindi, English, regional languages) and complex political discourse (e.g., implicit criticism of "government") require a robust, language-agnostic approach.
Gap: Existing English sentiment tools fail to capture India’s linguistic diversity and context-specific sentiment (e.g., BJP’s criticism targeting Congress).

### Iterative Development

**Initial Setup:**
- Collected news articles via newsdata.io, focusing on Modi/BJP and Gandhi/Congress.
- Built a preprocessing pipeline with mBERT for tokenization and basic sentiment analysis.
- Encountered issues: zero entity mentions, negative sentiment skew, and incorrect attribution (e.g., BJP penalized for criticizing Congress).

**NER and Entity Detection:**
- Adopted xlm-roberta-large-finetuned-conll03-english for multilingual NER, enabling detection of "Narendra Modi", "BJP", etc.
- Added post-processing to map "Modi" variants and handle subword tokens (e.g., ▁PM + ▁Modi).
- Fixed zero mentions, achieving 117 Modi mentions in the latest PoC.

**Sentiment Analysis:**
- Used nlptown/bert-base-multilingual-uncased-sentiment but faced negative skew (e.g., Modi: -0.7521).
- Introduced rule-based validation for Hindi (e.g., "विकास" → positive), balancing sentiment scores.
- Implemented context-aware rules (e.g., "सरकार" + BJP mention → negative for Congress), addressing misattribution.

**PoC Refinement:**
- Developed Logistic Regression for article-level classification, achieving 0.8 accuracy.
- Simulated labels with text-based heuristics, ensuring balanced BJP/Congress distribution.


# Future Plans

**YouTube Integration:**
- Develop a transcription preprocessing pipeline using youtube-transcript-api or YouTube Data API.
- Clean transcriptions (remove timestamps, [Music]), detect languages, and apply the existing NLP pipeline.
- Target 500-1000 transcriptions from official channels (BJP, Congress, NDTV) to complement news data.


**Enhanced NER:**
- Fine-tune XLM-RoBERTa on WikiANN Hindi to improve detection of regional names (e.g., "Rahul ji" → "Rahul Gandhi").
- Increase Congress mentions by refining implicit detection (e.g., "government" in criticism contexts).


**Sentiment Refinement:**
- Address negative sentiment skew by expanding rule-based validation for regional languages (e.g., Bengali, Marathi).
- Implement chunk-based sentiment for long texts (e.g., YouTube transcriptions), averaging sentence-level scores.


**Model Upgrades:**
- Experiment with advanced classifiers (e.g., BERT-based sequence classification) to improve accuracy beyond 0.8.
- Incorporate temporal analysis using pubDate to weight recent articles/transcriptions.


**Multimodal Analysis:**
- Explore audio features (e.g., tone in YouTube videos) or video metadata (e.g., titles, tags) to enrich sentiment analysis.
- Investigate thumbnail-based sentiment cues, though text remains the primary focus.


**Ground-Truth Validation:**
- Collect historical election data to validate predictions against actual outcomes.
- Conduct user studies to assess sentiment alignment with voter perceptions.


Sample Output
```
PoC Results
Processed 2000 articles
Features: ['Narendra Modi', 'BJP', 'Rahul Gandhi', 'Congress', 'sentiment']
Article-Level Classification:
Accuracy: 0.8025
Classification Report:
              precision    recall  f1-score   support
    Congress       0.52      0.15      0.23        80
         BJP       0.83      0.97      0.89       320
    accuracy                           0.80       400
   macro avg       0.67      0.56      0.56       400
weighted avg       0.76      0.80      0.76       400
Election Outcome Prediction:
BJP Sentiment Score: 0.2500
Congress Sentiment Score: -0.8000
Predicted Winner: BJP
Entity Sentiment Details:
Narendra Modi: Avg Sentiment=0.1000, Positive=400, Negative=900, Total Mentions=1337
BJP: Avg Sentiment=0.4000, Positive=250, Negative=200, Total Mentions=468
Rahul Gandhi: Avg Sentiment=-0.2000, Positive=10, Negative=15, Total Mentions=34
Congress: Avg Sentiment=-0.8000, Positive=10, Negative=120, Total Mentions=137
```

## Contributing

We welcome contributions to enhance the multilingual pipeline, expand data sources, or refine models. Please submit issues or pull requests on our repository.
