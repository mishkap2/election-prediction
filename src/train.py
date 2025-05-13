import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from collections import Counter
import ast
import os
from datetime import datetime
import re


def load_preprocessed_data(csv_path):
    """
    Load preprocessed news article CSV.

    Args:
        csv_path (str): Path to preprocessed CSV

    Returns:
        pd.DataFrame: Loaded DataFrame
    """
    df = pd.read_csv(csv_path, encoding="utf-8")
    print(f"Loaded {len(df)} preprocessed articles")
    return df


def extract_entity_sentiment(df):
    """
    Extract entity-specific sentiment scores with context-aware attribution.

    Args:
        df (pd.DataFrame): Preprocessed DataFrame

    Returns:
        pd.DataFrame: DataFrame with article-level entity sentiment features
        dict: Aggregated entity sentiment scores
    """
    sentiment_map = {"positive": 1, "neutral": 0, "negative": -1, "unknown": 0}
    key_entities = ["Narendra Modi", "BJP", "Rahul Gandhi", "Congress"]

    features = []
    entity_sentiments = {ent: [] for ent in key_entities}

    for _, row in df.iterrows():
        sentiment = sentiment_map[row["sentiment"]]
        entities = (
            ast.literal_eval(row["entities"])
            if isinstance(row["entities"], str)
            else row["entities"]
        )
        text = row["normalized_text"].lower()

        entity_counts = Counter([e[0] for e in entities if e[1] in ["PER", "ORG"]])

        bjp_mention = (
            entity_counts.get("Narendra Modi", 0) + entity_counts.get("BJP", 0) > 0
        )
        congress_mention = (
            entity_counts.get("Rahul Gandhi", 0) + entity_counts.get("Congress", 0) > 0
        )

        article_features = {ent: 0 for ent in key_entities}

        # Context-aware sentiment attribution
        criticism_keywords = ["हमलावर", "आलोचना", "विरोध", "टैक्स थोप", "बंद किया"]
        government_keywords = ["सरकार", "government", "प्रदेश सरकार"]

        # If BJP is criticizing government, negativity targets Congress
        if (
            bjp_mention
            and sentiment < 0
            and (
                any(kw in text for kw in government_keywords)
                or any(kw in text for kw in criticism_keywords)
            )
        ):
            article_features["Congress"] = sentiment
            entity_sentiments["Congress"].append(sentiment)
        # If Congress is criticizing government, negativity targets BJP
        elif (
            congress_mention
            and sentiment < 0
            and (
                any(kw in text for kw in government_keywords)
                or any(kw in text for kw in criticism_keywords)
            )
        ):
            article_features["BJP"] = sentiment
            entity_sentiments["BJP"].append(sentiment)
        else:
            # Default: assign sentiment to mentioned entities
            for ent in key_entities:
                if ent in entity_counts:
                    article_features[ent] = sentiment
                    entity_sentiments[ent].append(sentiment)
            # Implicit Congress mention for government criticism
            if (
                sentiment < 0
                and any(kw in text for kw in government_keywords)
                and not bjp_mention
            ):
                article_features["Congress"] = sentiment
                entity_sentiments["Congress"].append(sentiment)

        article_features["sentiment"] = sentiment
        features.append(article_features)

    feature_df = pd.DataFrame(features)

    aggregated_sentiments = {
        ent: {
            "avg_sentiment": np.mean(scores) if scores else 0,
            "positive_count": sum(1 for s in scores if s > 0),
            "negative_count": sum(1 for s in scores if s < 0),
            "total_mentions": len(scores),
        }
        for ent, scores in entity_sentiments.items()
    }

    return feature_df, aggregated_sentiments


def simulate_labels(df, min_congress_ratio=0.3):
    """
    Simulate article-level labels, ensuring minimum Congress labels.

    Args:
        df (pd.DataFrame): Preprocessed DataFrame
        min_congress_ratio (float): Minimum proportion of Congress labels

    Returns:
        np.ndarray: Simulated labels (1 for BJP, 0 for Congress)
    """
    labels = []
    sentiment_map = {"positive": 1, "neutral": 0, "negative": -1, "unknown": 0}

    for _, row in df.iterrows():
        sentiment = sentiment_map[row["sentiment"]]
        entities = (
            ast.literal_eval(row["entities"])
            if isinstance(row["entities"], str)
            else row["entities"]
        )
        text = row["normalized_text"].lower()

        entity_counts = Counter([e[0] for e in entities if e[1] in ["PER", "ORG"]])

        bjp_score = sentiment * (
            entity_counts.get("Narendra Modi", 0) + entity_counts.get("BJP", 0)
        )
        congress_score = sentiment * (
            entity_counts.get("Rahul Gandhi", 0) + entity_counts.get("Congress", 0)
        )

        # Context-aware labeling
        government_keywords = ["सरकार", "government", "प्रदेश सरकार"]
        criticism_keywords = ["हमलावर", "आलोचना", "विरोध"]

        if any(kw in text for kw in government_keywords) or any(
            kw in text for kw in criticism_keywords
        ):
            if entity_counts.get("BJP", 0) > 0 and sentiment < 0:
                labels.append(0)  # Negative toward Congress/government
            elif entity_counts.get("Congress", 0) > 0 and sentiment < 0:
                labels.append(1)  # Negative toward BJP/government
            else:
                if bjp_score > congress_score and bjp_score != 0:
                    labels.append(1)
                elif congress_score > bjp_score and congress_score != 0:
                    labels.append(0)
                else:
                    # Text-based heuristic
                    if "modi" in text or "bjp" in text:
                        labels.append(1)
                    elif "gandhi" in text or "congress" in text:
                        labels.append(0)
                    else:
                        labels.append(np.random.choice([0, 1]))
        else:
            if bjp_score > congress_score and bjp_score != 0:
                labels.append(1)
            elif congress_score > bjp_score and congress_score != 0:
                labels.append(0)
            else:
                if "modi" in text or "bjp" in text:
                    labels.append(1)
                elif "gandhi" in text or "congress" in text:
                    labels.append(0)
                else:
                    labels.append(np.random.choice([0, 1]))

    labels = np.array(labels)

    congress_count = np.sum(labels == 0)
    if congress_count / len(labels) < min_congress_ratio:
        n_needed = int(min_congress_ratio * len(labels)) - congress_count
        bjp_indices = np.where(labels == 1)[0]
        if len(bjp_indices) >= n_needed:
            flip_indices = np.random.choice(bjp_indices, n_needed, replace=False)
            labels[flip_indices] = 0

    print(
        f"Label distribution: BJP={np.sum(labels)}, Congress={len(labels) - np.sum(labels)}"
    )
    return labels


def train_and_predict(feature_df, labels):
    """
    Train Logistic Regression and predict article-level favorability.

    Args:
        feature_df (pd.DataFrame): Feature DataFrame
        labels (np.ndarray): Article labels

    Returns:
        dict: Model, predictions, and accuracy
    """
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        print(
            f"Warning: Only one class ({unique_labels[0]}) found. Skipping classification."
        )
        return {
            "model": None,
            "accuracy": 0.0,
            "report": "No classification performed due to single-class labels",
            "predictions": labels,
        }

    X_train, X_test, y_train, y_test = train_test_split(
        feature_df, labels, test_size=0.2, random_state=42
    )

    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test, y_pred, target_names=["Congress", "BJP"], labels=[0, 1], zero_division=0
    )

    return {
        "model": model,
        "accuracy": accuracy,
        "report": report,
        "predictions": model.predict(feature_df),
    }


def predict_election_outcome(aggregated_sentiments):
    """
    Predict election outcome based on entity-specific sentiment.

    Args:
        aggregated_sentiments (dict): Aggregated sentiment scores per entity

    Returns:
        dict: Predicted winner and sentiment details
    """
    bjp_sentiment = (
        aggregated_sentiments["Narendra Modi"]["avg_sentiment"]
        + aggregated_sentiments["BJP"]["avg_sentiment"]
    ) / 2
    congress_sentiment = (
        aggregated_sentiments["Rahul Gandhi"]["avg_sentiment"]
        + aggregated_sentiments["Congress"]["avg_sentiment"]
    ) / 2

    if bjp_sentiment == congress_sentiment == 0:
        winner = "No clear winner (insufficient entity mentions)"
    else:
        winner = "BJP" if bjp_sentiment > congress_sentiment else "Congress"

    return {
        "bjp_sentiment": bjp_sentiment,
        "congress_sentiment": congress_sentiment,
        "winner": winner,
        "details": aggregated_sentiments,
    }


def run_poc(csv_path, output_path):
    """
    Run PoC for election prediction based on entity-specific sentiment.

    Args:
        csv_path (str): Path to preprocessed CSV
        output_path (str): Path to save results
    """
    df = load_preprocessed_data(csv_path)

    feature_df, aggregated_sentiments = extract_entity_sentiment(df)
    print(f"Extracted features: {feature_df.columns.tolist()}")

    labels = simulate_labels(df)

    classification_results = train_and_predict(feature_df, labels)

    outcome = predict_election_outcome(aggregated_sentiments)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    output_file = f"{output_path}_{timestamp}.txt"

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"PoC Results\n")
        f.write(f"Processed {len(df)} articles\n")
        f.write(f"Features: {feature_df.columns.tolist()}\n")
        f.write(f"Article-Level Classification:\n")
        f.write(f"Accuracy: {classification_results['accuracy']:.4f}\n")
        f.write(f"Classification Report:\n{classification_results['report']}\n")
        f.write(f"Election Outcome Prediction:\n")
        f.write(f"BJP Sentiment Score: {outcome['bjp_sentiment']:.4f}\n")
        f.write(f"Congress Sentiment Score: {outcome['congress_sentiment']:.4f}\n")
        f.write(f"Predicted Winner: {outcome['winner']}\n")
        f.write(f"Entity Sentiment Details:\n")
        for ent, stats in outcome["details"].items():
            f.write(
                f"{ent}: Avg Sentiment={stats['avg_sentiment']:.4f}, "
                f"Positive={stats['positive_count']}, Negative={stats['negative_count']}, "
                f"Total Mentions={stats['total_mentions']}\n"
            )

    print(f"PoC completed. Results saved to {output_file}")
    print(f"Predicted Winner: {outcome['winner']}")

    return {
        "classification_results": classification_results,
        "election_outcome": outcome,
    }


if __name__ == "__main__":
    import glob
    csv_path = glob.glob("data/processed/*.csv")[0]
    output_path = "data/results/poc_results"
    run_poc(csv_path, output_path)
