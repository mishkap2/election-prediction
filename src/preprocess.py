import pandas as pd
import re
import os
from indicnlp.tokenize import indic_tokenize
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
from transformers import AutoTokenizer, pipeline
import numpy as np
from datetime import datetime
import torch
import unicodedata

import config

# Language mapping: full names to ISO codes
LANGUAGE_MAP = {
    "english": "en",
    "hindi": "hi",
    "bengali": "bn",
    "assamese": "as",
    "gujarati": "gu",
    "marathi": "mr",
}


def load_and_clean_data(csv_paths):
    """
    Load and clean news article CSVs, using only title and description.

    Args:
        csv_paths (list): List of CSV file paths

    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    dfs = [pd.read_csv(path, encoding="utf-8") for path in csv_paths]
    df = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(df)} articles from CSVs")

    # Remove duplicates based on article_id
    df = df.drop_duplicates(subset=["article_id"], keep="first")
    print(f"After removing duplicates: {len(df)} articles")

    # Combine title and description
    df["text"] = df.apply(
        lambda x: (
            x["title"]
            + " "
            + (x["description"] if pd.notnull(x["description"]) else "")
        ).strip(),
        axis=1,
    )

    # Handle missing values
    df["language"] = df["language"].fillna("en").str.lower()
    df["text"] = df["text"].fillna("")

    # Map language names to ISO codes
    df["language"] = df["language"].map(LANGUAGE_MAP).fillna("en")

    return df[["article_id", "text", "language", "pubDate"]]


def sanitize_text(text):
    """
    Sanitize text to remove problematic characters.

    Args:
        text (str): Input text

    Returns:
        str: Sanitized text
    """
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(
        r"[^\x20-\x7E\u0900-\u097F\u0980-\u09FF\u0A80-\u0AFF\u0B00-\u0B7F\u0B80-\u0BFF]",
        " ",
        text,
    )
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_text(text, lang):
    """
    Normalize text for the given language.

    Args:
        text (str): Input text
        lang (str): Language ISO code (e.g., 'hi', 'bn')

    Returns:
        str: Normalized text
    """
    text = sanitize_text(text)

    if lang == "en":
        text = text.lower()
        text = re.sub(r"[^\w\s]", "", text)
        return text

    try:
        normalizer = IndicNormalizerFactory().get_normalizer(lang)
        text = normalizer.normalize(text)
        text = re.sub(r"\s+", " ", text).strip()
    except KeyError:
        print(
            f"Warning: Language '{lang}' not supported by indic-nlp. Using basic normalization."
        )
        text = re.sub(r"[^\w\s]", "", text)
    return text


def tokenize_text(text, tokenizer, max_length=512):
    """
    Tokenize text using mBERT tokenizer.

    Args:
        text (str): Input text
        tokenizer: mBERT tokenizer
        max_length (int): Max token length

    Returns:
        list: Token IDs
    """
    try:
        tokens = tokenizer(
            text,
            max_length=max_length,
            truncation=True,
            padding=False,
            return_tensors="pt",
        )["input_ids"][0].tolist()
        return tokens
    except Exception as e:
        print(f"Tokenization failed for text: {text[:50]}... Error: {str(e)}")
        return []


def perform_ner(text, ner_pipeline, max_length=512):
    """
    Perform Named Entity Recognition on text, aggregating subword tokens.

    Args:
        text (str): Input text
        ner_pipeline: NER pipeline
        max_length (int): Max token length

    Returns:
        list: List of entities (entity, label)
    """
    try:
        tokens = ner_pipeline.tokenizer(
            text, max_length=max_length, truncation=True, return_tensors="pt"
        )
        if len(tokens["input_ids"][0]) <= max_length:
            entities = ner_pipeline(text, aggregation_strategy="simple")
            processed_entities = []
            for entity in entities:
                word = entity["word"].replace("▁", "").strip()
                # Map Modi variants to Narendra Modi
                if word in ["Modi", "मोदी", "PM Modi", "पीएम मोदी"]:
                    word = "Narendra Modi"
                # Clean up subword artifacts and invalid entries
                if word and not word.startswith("#") and not word.isspace():
                    label = entity["entity_group"]  # e.g., PER, ORG
                    processed_entities.append((word, label))
            return [
                (word, label)
                for word, label in processed_entities
                if entity["score"] > 0.7
            ]
        else:
            print(f"Text too long for NER: {text[:50]}...")
            return []
    except Exception as e:
        print(f"NER failed for text: {text[:50]}... Error: {str(e)}")
        return []


def rule_based_sentiment(text):
    """
    Rule-based sentiment analysis for Hindi texts as a fallback.

    Args:
        text (str): Input text

    Returns:
        str: Sentiment label (positive, negative, neutral)
    """
    negative_keywords = ["हमलावर", "टैक्स थोप", "बंद किया", "प्रतिक्रिया", "आलोचना"]
    positive_keywords = ["विकास", "लोकप्रिय", "जीत", "सफलता", "प्रशंसा"]
    text = text.lower()
    if any(kw in text for kw in negative_keywords):
        return "negative"
    if any(kw in text for kw in positive_keywords):
        return "positive"
    return "neutral"


def compute_sentiment(text, sentiment_analyzer, max_length=512):
    """
    Compute sentiment for text using a multilingual model, with rule-based fallback.

    Args:
        text (str): Input text
        sentiment_analyzer: Sentiment analysis pipeline
        max_length (int): Max token length

    Returns:
        str: Sentiment label (positive, negative, neutral)
    """
    try:
        inputs = sentiment_analyzer.tokenizer(
            text,
            max_length=max_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        if len(inputs["input_ids"][0]) <= max_length:
            result = sentiment_analyzer(text, truncation=True, max_length=max_length)[0]
            label = result["label"]
            if label in ["1 star", "2 stars"]:
                sentiment = "negative"
            elif label in ["4 stars", "5 stars"]:
                sentiment = "positive"
            else:
                sentiment = "neutral"
            # Validate with rule-based sentiment for Hindi
            rule_sentiment = rule_based_sentiment(text)
            if sentiment == "negative" and rule_sentiment == "positive":
                return "positive"
            if sentiment == "positive" and rule_sentiment == "negative":
                return "negative"
            return sentiment
        else:
            print(f"Text too long for sentiment analysis: {text[:50]}...")
            return rule_based_sentiment(text)
    except Exception as e:
        print(f"Sentiment analysis failed for text: {text[:50]}... Error: {str(e)}")
        return rule_based_sentiment(text)


def preprocess_news_data(csv_paths, output_path):
    """
    Preprocess news article data for sentiment analysis, using title and description.

    Args:
        csv_paths (list): List of CSV file paths
        output_path (str): Path to save preprocessed CSV
    """
    device = config.DEVICE
    print(f"Using device: {'GPU' if device == 0 else 'CPU'}")

    # Load mBERT tokenizer for tokenization
    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

    # Load NER pipeline with xlm-roberta
    try:
        ner_pipeline = pipeline(
            "ner",
            model="xlm-roberta-large-finetuned-conll03-english",
            tokenizer="xlm-roberta-large-finetuned-conll03-english",
            device=device,
            aggregation_strategy="simple",
        )
    except Exception as e:
        print(f"Failed to load xlm-roberta-large-finetuned-conll03-english: {str(e)}")
        print("Falling back to dslim/bert-base-NER")
        ner_pipeline = pipeline(
            "ner",
            model="dslim/bert-base-NER",
            tokenizer="dslim/bert-base-NER",
            device=device,
            aggregation_strategy="simple",
        )

    # Load sentiment analysis pipeline
    sentiment_analyzer = pipeline(
        "sentiment-analysis",
        model="nlptown/bert-base-multilingual-uncased-sentiment",
        device=device,
    )

    # Load and clean data
    df = load_and_clean_data(csv_paths)

    # Normalize text
    df["normalized_text"] = df.apply(
        lambda x: normalize_text(x["text"], x["language"]), axis=1
    )

    # Tokenize text
    df["tokens"] = df["normalized_text"].apply(lambda x: tokenize_text(x, tokenizer))

    # Perform NER
    df["entities"] = df["normalized_text"].apply(lambda x: perform_ner(x, ner_pipeline))

    # Compute sentiment
    df["sentiment"] = df["normalized_text"].apply(
        lambda x: compute_sentiment(x, sentiment_analyzer)
    )

    # Save preprocessed data
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    output_file = f"{output_path}_{timestamp}.csv"
    df[
        [
            "article_id",
            "text",
            "language",
            "tokens",
            "entities",
            "sentiment",
            "normalized_text",
        ]
    ].to_csv(output_file, index=False, encoding="utf-8")

    print(f"Preprocessed {len(df)} articles. Saved to {output_file}")

    return df


if __name__ == "__main__":
    import glob
    csv_paths = glob.glob("data/raw/election_news_*.csv")
    output_path = "data/processed/preprocessed_news"
    preprocess_news_data(csv_paths, output_path)
