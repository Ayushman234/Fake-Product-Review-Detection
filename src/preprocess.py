"""
preprocess.py — Data Loading & Cleaning Module
Fake Product Review Detection System

This module handles:
  1. Loading the dataset from CSV
  2. Validating expected columns
  3. Cleaning review text (lowercase, remove URLs, punctuation, extra spaces)
  4. Lemmatization (reducing words to base forms using NLTK WordNetLemmatizer)
  5. Removing missing values and duplicates
  6. Mapping labels (CG → Fake, OR → Genuine)
"""

import os
import re
import sys
import pandas as pd
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer

# Download required NLTK resources (safe to run multiple times)
nltk.download('wordnet',  quiet=True)
nltk.download('omw-1.4', quiet=True)

# Initialise lemmatizer once at module level (thread-safe)
_lemmatizer = WordNetLemmatizer()


# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
EXPECTED_COLUMNS = ['category', 'rating', 'label', 'text_']
LABEL_MAP = {'CG': 1, 'OR': 0}  # CG = Fake (1), OR = Genuine (0)
LABEL_NAME_MAP = {1: 'Fake', 0: 'Genuine'}


# ──────────────────────────────────────────────
# Text Cleaning
# ──────────────────────────────────────────────
def clean_text(text):
    """
    Clean a single review text string.
    
    Steps:
      - Convert to lowercase
      - Remove URLs
      - Remove HTML tags
      - Remove punctuation and special characters (keep letters, numbers, spaces)
      - Remove extra whitespace
      - Lemmatize each token (e.g. 'running' → 'run', 'bought' → 'buy')
    
    Note: Stopword removal is intentionally NOT applied here.
    Words like 'but', 'my', 'however', 'at' are strong genuine-review
    indicators (high negative weights in Logistic Regression) and must
    be preserved for the model to work correctly.
    
    Args:
        text (str): Raw review text.
    
    Returns:
        str: Cleaned and lemmatized review text.
    """
    if not isinstance(text, str):
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove punctuation and special characters (keep letters, numbers, spaces)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenize and lemmatize (no stopword removal — see docstring above)
    tokens = [
        _lemmatizer.lemmatize(token)
        for token in text.split()
    ]
    
    return ' '.join(tokens)


# ──────────────────────────────────────────────
# Dataset Loading & Validation
# ──────────────────────────────────────────────
def load_dataset(filepath):
    """
    Load the CSV dataset and validate that expected columns exist.
    
    Args:
        filepath (str): Path to the CSV file.
    
    Returns:
        pd.DataFrame: Loaded DataFrame.
    
    Raises:
        FileNotFoundError: If the CSV file doesn't exist.
        ValueError: If expected columns are missing.
    """
    # Check file exists
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"\n❌ Dataset not found at: {filepath}\n"
            f"   Please download the dataset from Kaggle and place it in the 'data/' folder.\n"
            f"   See README.md for download instructions."
        )
    
    # Load CSV
    print(f"📂 Loading dataset from: {filepath}")
    df = pd.read_csv(filepath)
    print(f"   Loaded {len(df):,} rows and {len(df.columns)} columns.\n")
    
    # Validate columns
    missing_cols = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"\n❌ Missing expected columns: {missing_cols}\n"
            f"   Found columns: {df.columns.tolist()}\n"
            f"   Expected columns: {EXPECTED_COLUMNS}\n"
            f"   Make sure you downloaded the correct dataset (Kaggle Fake Reviews Dataset)."
        )
    
    return df


# ──────────────────────────────────────────────
# Preprocessing Pipeline
# ──────────────────────────────────────────────
def preprocess_data(df):
    """
    Clean and preprocess the dataset.
    
    Steps:
      1. Keep only the relevant columns
      2. Drop rows with missing review text or labels
      3. Remove duplicate reviews
      4. Clean the review text
      5. Remove empty reviews (after cleaning)
      6. Map labels to numeric (CG=1, OR=0)
    
    Args:
        df (pd.DataFrame): Raw dataset.
    
    Returns:
        pd.DataFrame: Cleaned dataset with columns:
            - 'text_' (cleaned review text)
            - 'label' (numeric: 1=Fake, 0=Genuine)
            - 'label_name' (string: 'Fake' or 'Genuine')
            - 'category' (product category)
            - 'rating' (star rating)
    """
    print("🔧 Preprocessing data...")
    initial_count = len(df)
    
    # --- Step 1: Keep relevant columns ---
    df = df[EXPECTED_COLUMNS].copy()
    
    # --- Step 2: Handle missing values ---
    missing_text = df['text_'].isnull().sum()
    missing_label = df['label'].isnull().sum()
    if missing_text > 0 or missing_label > 0:
        print(f"   Dropping {missing_text} rows with missing text, {missing_label} with missing labels.")
    df = df.dropna(subset=['text_', 'label'])
    
    # --- Step 3: Remove duplicates ---
    duplicates = df.duplicated(subset=['text_'], keep='first').sum()
    if duplicates > 0:
        print(f"   Removing {duplicates} duplicate reviews.")
        df = df.drop_duplicates(subset=['text_'], keep='first')
    
    # --- Step 4: Validate labels ---
    valid_labels = set(LABEL_MAP.keys())
    invalid_mask = ~df['label'].isin(valid_labels)
    invalid_count = invalid_mask.sum()
    if invalid_count > 0:
        print(f"   Removing {invalid_count} rows with invalid labels.")
        df = df[~invalid_mask]
    
    # --- Step 5: Clean text ---
    print("   Cleaning review text...")
    df['text_'] = df['text_'].apply(clean_text)
    
    # --- Step 6: Remove empty reviews after cleaning ---
    empty_mask = df['text_'].str.strip() == ''
    empty_count = empty_mask.sum()
    if empty_count > 0:
        print(f"   Removing {empty_count} empty reviews after cleaning.")
        df = df[~empty_mask]
    
    # --- Step 7: Map labels ---
    df['label'] = df['label'].map(LABEL_MAP)
    df['label_name'] = df['label'].map(LABEL_NAME_MAP)
    
    # --- Summary ---
    final_count = len(df)
    removed = initial_count - final_count
    print(f"\n✅ Preprocessing complete!")
    print(f"   Rows: {initial_count:,} → {final_count:,} (removed {removed:,})")
    print(f"   Fake reviews:    {(df['label'] == 1).sum():,}")
    print(f"   Genuine reviews: {(df['label'] == 0).sum():,}\n")
    
    # Reset index
    df = df.reset_index(drop=True)
    
    return df


# ──────────────────────────────────────────────
# Main (for testing preprocessing standalone)
# ──────────────────────────────────────────────
if __name__ == "__main__":
    # Determine dataset path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    data_path = os.path.join(project_dir, "data", "fake_reviews_dataset.csv")
    
    try:
        df = load_dataset(data_path)
        df = preprocess_data(df)
        
        print("─" * 50)
        print("Sample of cleaned data:")
        print(df[['text_', 'label_name']].head())
        print()
        print("Data types:")
        print(df.dtypes)
        
    except (FileNotFoundError, ValueError) as e:
        print(e)
        sys.exit(1)
