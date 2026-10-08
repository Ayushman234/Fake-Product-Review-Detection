"""
train.py — Model Training & Evaluation Script
Fake Product Review Detection System

This script:
  1. Loads and preprocesses the dataset
  2. Performs Exploratory Data Analysis (EDA) with visualizations
  3. Splits data into train/test sets (stratified)
  4. Applies TF-IDF vectorization (fitted on train only)
  5. Trains a Logistic Regression model
  6. Evaluates with full metrics (accuracy, precision, recall, F1, confusion matrix)
  7. If accuracy < 87%, tunes hyperparameters with GridSearchCV
  8. Saves the trained model and TF-IDF vectorizer

Usage:
    python src/train.py
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    log_loss,
)
import joblib

# Import our preprocessing module
# (Add parent directory to path so we can import from src/)
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
sys.path.insert(0, project_dir)
from src.preprocess import load_dataset, preprocess_data

warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET_ACCURACY = 0.87

# Paths
DATA_PATH = os.path.join(project_dir, "data", "fake_reviews_dataset.csv")
MODELS_DIR = os.path.join(project_dir, "models")
RESULTS_DIR = os.path.join(project_dir, "results")
MODEL_PATH = os.path.join(MODELS_DIR, "logistic_regression_model.pkl")
VECTORIZER_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
CONFUSION_MATRIX_PATH = os.path.join(RESULTS_DIR, "confusion_matrix.png")
METRICS_PATH = os.path.join(RESULTS_DIR, "metrics.txt")
LOSS_GRAPH_PATH = os.path.join(RESULTS_DIR, "model_loss.png")
ACCURACY_GRAPH_PATH = os.path.join(RESULTS_DIR, "model_accuracy.png")

# Ensure output directories exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ══════════════════════════════════════════════
# STEP 1: EXPLORATORY DATA ANALYSIS
# ══════════════════════════════════════════════
def perform_eda(df):
    """
    Perform Exploratory Data Analysis and save visualizations.
    
    Creates:
      - Class distribution bar chart
      - Review length distribution histogram
      - Prints basic statistics
    """
    print("=" * 60)
    print("📊 EXPLORATORY DATA ANALYSIS")
    print("=" * 60)
    
    # --- Basic Statistics ---
    print(f"\n📋 Dataset Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"\n📋 Column Data Types:")
    print(df.dtypes.to_string())
    
    print(f"\n📋 Missing Values:")
    print(df.isnull().sum().to_string())
    
    print(f"\n📋 Duplicate Reviews: {df.duplicated(subset=['text_']).sum()}")
    
    # --- Class Distribution ---
    class_counts = df['label_name'].value_counts()
    print(f"\n📋 Class Distribution:")
    for label, count in class_counts.items():
        pct = count / len(df) * 100
        print(f"   {label}: {count:,} ({pct:.1f}%)")
    
    # --- Review Length Statistics ---
    df['review_length'] = df['text_'].str.len()
    df['word_count'] = df['text_'].str.split().str.len()
    
    print(f"\n📋 Review Length (characters):")
    print(f"   Mean:   {df['review_length'].mean():.0f}")
    print(f"   Median: {df['review_length'].median():.0f}")
    print(f"   Min:    {df['review_length'].min()}")
    print(f"   Max:    {df['review_length'].max()}")
    
    print(f"\n📋 Word Count:")
    print(f"   Mean:   {df['word_count'].mean():.0f}")
    print(f"   Median: {df['word_count'].median():.0f}")
    print(f"   Min:    {df['word_count'].min()}")
    print(f"   Max:    {df['word_count'].max()}")
    
    # --- Review Length by Class ---
    print(f"\n📋 Average Review Length by Class:")
    for label_name in ['Fake', 'Genuine']:
        subset = df[df['label_name'] == label_name]
        print(f"   {label_name}: {subset['review_length'].mean():.0f} chars, {subset['word_count'].mean():.0f} words")
    
    # --- Visualization 1: Class Distribution ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = ['#e74c3c', '#2ecc71']  # Red for Fake, Green for Genuine
    
    # Bar chart
    bars = axes[0].bar(class_counts.index, class_counts.values, color=colors, edgecolor='black', linewidth=0.5)
    axes[0].set_title('Class Distribution', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Review Type', fontsize=12)
    axes[0].set_ylabel('Count', fontsize=12)
    for bar, count in zip(bars, class_counts.values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 200,
                     f'{count:,}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Review length distribution
    for label_name, color in zip(['Fake', 'Genuine'], colors):
        subset = df[df['label_name'] == label_name]
        axes[1].hist(subset['word_count'], bins=50, alpha=0.6, label=label_name, color=color, edgecolor='black', linewidth=0.3)
    axes[1].set_title('Review Word Count Distribution', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Word Count', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].legend(fontsize=11)
    
    plt.tight_layout()
    eda_path = os.path.join(RESULTS_DIR, "eda_plots.png")
    plt.savefig(eda_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n📊 EDA plots saved to: {eda_path}")
    
    # --- Category Distribution ---
    if 'category' in df.columns:
        print(f"\n📋 Reviews per Category:")
        cat_counts = df['category'].value_counts()
        for cat, count in cat_counts.items():
            print(f"   {cat}: {count:,}")
    
    # Clean up temporary columns
    df.drop(columns=['review_length', 'word_count'], inplace=True, errors='ignore')
    
    print()
    return df


# ══════════════════════════════════════════════
# STEP 2: TRAIN/TEST SPLIT
# ══════════════════════════════════════════════
def split_data(df):
    """
    Split data into training and testing sets.
    Uses stratification to maintain class balance.
    """
    print("=" * 60)
    print("✂️  TRAIN/TEST SPLIT")
    print("=" * 60)
    
    X = df['text_']
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y  # Maintain class balance in both splits
    )
    
    print(f"\n   Training set: {len(X_train):,} reviews")
    print(f"   Testing set:  {len(X_test):,} reviews")
    print(f"\n   Training class distribution:")
    print(f"     Fake:    {(y_train == 1).sum():,} ({(y_train == 1).mean() * 100:.1f}%)")
    print(f"     Genuine: {(y_train == 0).sum():,} ({(y_train == 0).mean() * 100:.1f}%)")
    print(f"\n   Testing class distribution:")
    print(f"     Fake:    {(y_test == 1).sum():,} ({(y_test == 1).mean() * 100:.1f}%)")
    print(f"     Genuine: {(y_test == 0).sum():,} ({(y_test == 0).mean() * 100:.1f}%)")
    print()
    
    return X_train, X_test, y_train, y_test


# ══════════════════════════════════════════════
# STEP 3: TF-IDF VECTORIZATION
# ══════════════════════════════════════════════
def apply_tfidf(X_train, X_test, max_features=15000, ngram_range=(1, 2)):
    """
    Apply TF-IDF vectorization.
    
    IMPORTANT: The vectorizer is fitted ONLY on training data.
    Test data is only transformed (no fitting), preventing data leakage.
    
    Args:
        X_train: Training text data.
        X_test: Testing text data.
        max_features: Maximum number of features.
        ngram_range: Range of n-grams to consider.
    
    Returns:
        X_train_tfidf: TF-IDF transformed training data.
        X_test_tfidf: TF-IDF transformed testing data.
        vectorizer: Fitted TF-IDF vectorizer.
    """
    print("=" * 60)
    print("📝 TF-IDF VECTORIZATION")
    print("=" * 60)
    
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=2,           # Ignore terms appearing in fewer than 2 documents
        max_df=0.95,        # Ignore terms appearing in more than 95% of documents
        sublinear_tf=True,  # Apply sublinear TF scaling (1 + log(tf))
        strip_accents='unicode',
    )
    
    print(f"\n   Configuration:")
    print(f"     max_features:  {max_features:,}")
    print(f"     ngram_range:   {ngram_range}")
    print(f"     min_df:        2")
    print(f"     max_df:        0.95")
    print(f"     sublinear_tf:  True")
    
    # Fit on training data ONLY, then transform both
    print(f"\n   Fitting TF-IDF on training data...")
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    print(f"   Training matrix shape: {X_train_tfidf.shape}")
    print(f"   Testing matrix shape:  {X_test_tfidf.shape}")
    print(f"   Vocabulary size: {len(vectorizer.vocabulary_):,} terms")
    print()
    
    return X_train_tfidf, X_test_tfidf, vectorizer


# ══════════════════════════════════════════════
# STEP 4: MODEL TRAINING
# ══════════════════════════════════════════════
def train_model(X_train_tfidf, y_train, X_test_tfidf=None, y_test=None, C=1.0):
    """
    Train a Logistic Regression classifier.
    
    Also collects iterative loss convergence data and per-fold CV metrics
    so that loss/accuracy graphs can be generated afterwards.
    
    Args:
        X_train_tfidf: TF-IDF transformed training features.
        y_train: Training labels.
        X_test_tfidf: TF-IDF transformed test features (for loss/accuracy tracking).
        y_test: Test labels (for loss/accuracy tracking).
        C: Regularization strength (inverse).
    
    Returns:
        model: Trained Logistic Regression model.
        training_history: Dict with loss/accuracy data for graph generation.
    """
    print("=" * 60)
    print("🤖 TRAINING LOGISTIC REGRESSION")
    print("=" * 60)
    
    print(f"\n   Configuration:")
    print(f"     C (regularization): {C}")
    print(f"     solver:             lbfgs")
    print(f"     max_iter:           1000")
    
    # ── Cross-validation with per-fold metrics ──
    print(f"\n   Running 5-fold cross-validation on training data...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_accuracies = []
    cv_losses = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train_tfidf, y_train), 1):
        X_fold_train = X_train_tfidf[train_idx]
        y_fold_train = y_train.iloc[train_idx] if hasattr(y_train, 'iloc') else y_train[train_idx]
        X_fold_val = X_train_tfidf[val_idx]
        y_fold_val = y_train.iloc[val_idx] if hasattr(y_train, 'iloc') else y_train[val_idx]
        
        fold_model = LogisticRegression(C=C, max_iter=1000, random_state=RANDOM_STATE, solver='lbfgs')
        fold_model.fit(X_fold_train, y_fold_train)
        
        fold_acc = fold_model.score(X_fold_val, y_fold_val)
        fold_proba = fold_model.predict_proba(X_fold_val)
        fold_loss = log_loss(y_fold_val, fold_proba)
        
        cv_accuracies.append(fold_acc)
        cv_losses.append(fold_loss)
        print(f"     Fold {fold_idx}: Accuracy = {fold_acc:.4f}, Log-Loss = {fold_loss:.4f}")
    
    print(f"\n   CV Mean Accuracy:   {np.mean(cv_accuracies):.4f} ± {np.std(cv_accuracies):.4f}")
    print(f"   CV Mean Log-Loss:   {np.mean(cv_losses):.4f} ± {np.std(cv_losses):.4f}")
    
    # ── Track loss convergence across iterations ──
    print(f"\n   Tracking loss convergence across solver iterations...")
    iteration_steps = list(range(1, 21)) + list(range(25, 101, 5)) + list(range(150, 1001, 50))
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []
    actual_iters = []
    
    for max_it in iteration_steps:
        iter_model = LogisticRegression(
            C=C, max_iter=max_it, random_state=RANDOM_STATE, solver='lbfgs',
        )
        # Suppress convergence warnings during intermediate steps
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            iter_model.fit(X_train_tfidf, y_train)
        
        # Training metrics
        train_proba = iter_model.predict_proba(X_train_tfidf)
        t_loss = log_loss(y_train, train_proba)
        t_acc = iter_model.score(X_train_tfidf, y_train)
        train_losses.append(t_loss)
        train_accs.append(t_acc)
        actual_iters.append(max_it)
        
        # Test metrics (if provided)
        if X_test_tfidf is not None and y_test is not None:
            test_proba = iter_model.predict_proba(X_test_tfidf)
            te_loss = log_loss(y_test, test_proba)
            te_acc = iter_model.score(X_test_tfidf, y_test)
            test_losses.append(te_loss)
            test_accs.append(te_acc)
        
        # Early stop tracking if converged (loss not changing)
        if len(train_losses) > 3 and abs(train_losses[-1] - train_losses[-2]) < 1e-7:
            break
    
    print(f"   Convergence tracked over {len(actual_iters)} iteration checkpoints.")
    
    # ── Train final model on full training set ──
    print(f"\n   Training final model on full training set...")
    model = LogisticRegression(
        C=C, max_iter=1000, random_state=RANDOM_STATE, solver='lbfgs',
    )
    model.fit(X_train_tfidf, y_train)
    
    train_accuracy = model.score(X_train_tfidf, y_train)
    print(f"   Training Accuracy:  {train_accuracy:.4f}")
    print()
    
    # ── Build training history dict ──
    training_history = {
        'cv_accuracies': cv_accuracies,
        'cv_losses': cv_losses,
        'iterations': actual_iters,
        'train_losses': train_losses,
        'train_accs': train_accs,
        'test_losses': test_losses,
        'test_accs': test_accs,
        'final_train_accuracy': train_accuracy,
    }
    
    return model, training_history


# ══════════════════════════════════════════════
# STEP 5: MODEL EVALUATION
# ══════════════════════════════════════════════
def evaluate_model(model, X_test_tfidf, y_test):
    """
    Evaluate the model on the test set and generate all metrics.
    
    Returns:
        metrics (dict): Dictionary containing all evaluation metrics.
    """
    print("=" * 60)
    print("📈 MODEL EVALUATION")
    print("=" * 60)
    
    # Predictions
    y_pred = model.predict(X_test_tfidf)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"\n   📊 Test Set Results:")
    print(f"   ┌─────────────────────────────┐")
    print(f"   │ Accuracy:   {accuracy:.4f}          │")
    print(f"   │ Precision:  {precision:.4f}          │")
    print(f"   │ Recall:     {recall:.4f}          │")
    print(f"   │ F1-Score:   {f1:.4f}          │")
    print(f"   └─────────────────────────────┘")
    
    # Classification report
    target_names = ['Genuine (OR)', 'Fake (CG)']
    report = classification_report(y_test, y_pred, target_names=target_names)
    print(f"\n   Classification Report:\n")
    for line in report.split('\n'):
        print(f"   {line}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print(f"\n   Confusion Matrix Breakdown:")
    print(f"     True Negatives  (Genuine predicted as Genuine): {tn:,}")
    print(f"     False Positives (Genuine predicted as Fake):    {fp:,}")
    print(f"     False Negatives (Fake predicted as Genuine):    {fn:,}")
    print(f"     True Positives  (Fake predicted as Fake):       {tp:,}")
    
    # --- Save Confusion Matrix as Image ---
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=target_names,
        yticklabels=target_names,
        ax=ax,
        annot_kws={'size': 16},
        linewidths=0.5,
        linecolor='black',
    )
    ax.set_xlabel('Predicted Label', fontsize=13)
    ax.set_ylabel('Actual Label', fontsize=13)
    ax.set_title('Confusion Matrix — Fake Review Detection', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n   📊 Confusion matrix saved to: {CONFUSION_MATRIX_PATH}")
    
    # --- Save Metrics to Text File ---
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'true_positives': tp,
        'true_negatives': tn,
        'false_positives': fp,
        'false_negatives': fn,
    }
    
    with open(METRICS_PATH, 'w') as f:
        f.write("=" * 50 + "\n")
        f.write("Fake Product Review Detection — Evaluation Metrics\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Accuracy:   {accuracy:.4f} ({accuracy * 100:.2f}%)\n")
        f.write(f"Precision:  {precision:.4f} ({precision * 100:.2f}%)\n")
        f.write(f"Recall:     {recall:.4f} ({recall * 100:.2f}%)\n")
        f.write(f"F1-Score:   {f1:.4f} ({f1 * 100:.2f}%)\n\n")
        f.write(f"Confusion Matrix:\n")
        f.write(f"  True Negatives  (Genuine → Genuine): {tn}\n")
        f.write(f"  False Positives (Genuine → Fake):    {fp}\n")
        f.write(f"  False Negatives (Fake → Genuine):    {fn}\n")
        f.write(f"  True Positives  (Fake → Fake):       {tp}\n\n")
        f.write(f"Classification Report:\n{report}\n")
    
    print(f"   📄 Metrics saved to: {METRICS_PATH}")
    print()
    
    return metrics


# ══════════════════════════════════════════════
# STEP 5b: GENERATE LOSS & ACCURACY GRAPHS
# ══════════════════════════════════════════════
def generate_training_graphs(training_history, test_accuracy=None):
    """
    Generate and save model loss and accuracy graphs.
    
    Creates two plots:
      1. Model Loss — log-loss convergence over solver iterations
      2. Model Accuracy — per-fold CV accuracy + train vs test bar chart
    
    Args:
        training_history: Dict returned by train_model() with loss/accuracy data.
        test_accuracy: Final test set accuracy (for the bar chart comparison).
    """
    print("=" * 60)
    print("📉 GENERATING LOSS & ACCURACY GRAPHS")
    print("=" * 60)
    
    iterations = training_history['iterations']
    train_losses = training_history['train_losses']
    train_accs = training_history['train_accs']
    test_losses = training_history['test_losses']
    test_accs = training_history['test_accs']
    cv_accuracies = training_history['cv_accuracies']
    cv_losses = training_history['cv_losses']
    final_train_acc = training_history['final_train_accuracy']
    
    # ── GRAPH 1: Model Loss ──
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left: Loss convergence over iterations
    axes[0].plot(iterations, train_losses, 'o-', color='#e74c3c', linewidth=2,
                 markersize=3, label='Training Loss', alpha=0.9)
    if test_losses:
        axes[0].plot(iterations, test_losses, 's-', color='#3498db', linewidth=2,
                     markersize=3, label='Test Loss', alpha=0.9)
    axes[0].set_xlabel('Max Solver Iterations', fontsize=12)
    axes[0].set_ylabel('Log-Loss', fontsize=12)
    axes[0].set_title('Loss Convergence Over Iterations', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xscale('log')
    
    # Right: Per-fold CV loss
    fold_numbers = list(range(1, len(cv_losses) + 1))
    bars = axes[1].bar(fold_numbers, cv_losses, color='#e74c3c', edgecolor='black',
                       linewidth=0.5, alpha=0.85)
    mean_loss = np.mean(cv_losses)
    axes[1].axhline(y=mean_loss, color='#2c3e50', linestyle='--', linewidth=2,
                    label=f'Mean Loss = {mean_loss:.4f}')
    axes[1].set_xlabel('CV Fold', fontsize=12)
    axes[1].set_ylabel('Log-Loss', fontsize=12)
    axes[1].set_title('Cross-Validation Log-Loss Per Fold', fontsize=14, fontweight='bold')
    axes[1].set_xticks(fold_numbers)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, cv_losses):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                     f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.suptitle('Model Loss — Fake Review Detection', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(LOSS_GRAPH_PATH, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n   📉 Model loss graph saved to: {LOSS_GRAPH_PATH}")
    
    # ── GRAPH 2: Model Accuracy ──
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left: Accuracy convergence over iterations
    axes[0].plot(iterations, [a * 100 for a in train_accs], 'o-', color='#2ecc71',
                 linewidth=2, markersize=3, label='Training Accuracy', alpha=0.9)
    if test_accs:
        axes[0].plot(iterations, [a * 100 for a in test_accs], 's-', color='#9b59b6',
                     linewidth=2, markersize=3, label='Test Accuracy', alpha=0.9)
    axes[0].set_xlabel('Max Solver Iterations', fontsize=12)
    axes[0].set_ylabel('Accuracy (%)', fontsize=12)
    axes[0].set_title('Accuracy Convergence Over Iterations', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xscale('log')
    
    # Right: Per-fold CV accuracy + train vs test bar chart
    # Per-fold bars
    fold_numbers = list(range(1, len(cv_accuracies) + 1))
    bar_positions = fold_numbers + [len(cv_accuracies) + 2, len(cv_accuracies) + 3]
    bar_values = [a * 100 for a in cv_accuracies]
    bar_colors = ['#3498db'] * len(cv_accuracies)
    bar_labels_text = [f'Fold {i}' for i in fold_numbers]
    
    # Add train & test accuracy bars
    bar_values.append(final_train_acc * 100)
    bar_colors.append('#2ecc71')
    bar_labels_text.append('Train')
    
    if test_accuracy is not None:
        bar_values.append(test_accuracy * 100)
        bar_colors.append('#9b59b6')
        bar_labels_text.append('Test')
    
    bars = axes[1].bar(range(len(bar_values)), bar_values, color=bar_colors,
                       edgecolor='black', linewidth=0.5, alpha=0.85)
    mean_cv_acc = np.mean(cv_accuracies) * 100
    axes[1].axhline(y=mean_cv_acc, color='#2c3e50', linestyle='--', linewidth=2,
                    label=f'CV Mean = {mean_cv_acc:.2f}%')
    axes[1].set_xlabel('', fontsize=12)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].set_title('Per-Fold CV Accuracy & Train/Test Comparison', fontsize=14, fontweight='bold')
    axes[1].set_xticks(range(len(bar_values)))
    axes[1].set_xticklabels(bar_labels_text, fontsize=10, rotation=0)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3, axis='y')
    # Set y-axis to focus on relevant range
    min_val = min(bar_values)
    axes[1].set_ylim(max(0, min_val - 3), 100)
    for bar, val in zip(bars, bar_values):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
                     f'{val:.2f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.suptitle('Model Accuracy — Fake Review Detection', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(ACCURACY_GRAPH_PATH, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   📈 Model accuracy graph saved to: {ACCURACY_GRAPH_PATH}")
    print()


# ══════════════════════════════════════════════
# STEP 6: HYPERPARAMETER TUNING (if needed)
# ══════════════════════════════════════════════
def tune_model(X_train, y_train):
    """
    Use GridSearchCV to find optimal TF-IDF + Logistic Regression parameters.
    This function creates a pipeline and searches over key hyperparameters.
    
    Returns:
        best_vectorizer: Best TF-IDF vectorizer.
        best_model: Best Logistic Regression model.
        X_train_tfidf: Training data transformed by best vectorizer.
    """
    from sklearn.pipeline import Pipeline
    
    print("=" * 60)
    print("🔧 HYPERPARAMETER TUNING (GridSearchCV)")
    print("=" * 60)
    print("\n   Baseline accuracy was below 87%. Tuning hyperparameters...\n")
    
    # Create a pipeline
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,
            strip_accents='unicode',
        )),
        ('clf', LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
            solver='lbfgs',
        )),
    ])
    
    # Parameter grid
    param_grid = {
        'tfidf__max_features': [10000, 20000, 30000],
        'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
        'clf__C': [0.5, 1.0, 5.0, 10.0],
    }
    
    print(f"   Parameter grid:")
    for key, values in param_grid.items():
        print(f"     {key}: {values}")
    
    total_combos = 1
    for values in param_grid.values():
        total_combos *= len(values)
    print(f"\n   Total combinations: {total_combos}")
    print(f"   With 5-fold CV: {total_combos * 5} fits")
    print(f"\n   This may take a few minutes...\n")
    
    # Grid search with cross-validation
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,  # Use all CPU cores
        verbose=1,
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"\n   ✅ Best parameters found:")
    for key, value in grid_search.best_params_.items():
        print(f"     {key}: {value}")
    print(f"\n   Best CV Accuracy: {grid_search.best_score_:.4f}")
    
    # Extract best vectorizer and model from pipeline
    best_pipeline = grid_search.best_estimator_
    best_vectorizer = best_pipeline.named_steps['tfidf']
    best_model = best_pipeline.named_steps['clf']
    
    # Get transformed training data
    X_train_tfidf = best_vectorizer.transform(X_train)
    
    print()
    return best_vectorizer, best_model, X_train_tfidf


# ══════════════════════════════════════════════
# STEP 7: SAVE MODEL & VECTORIZER
# ══════════════════════════════════════════════
def save_artifacts(model, vectorizer):
    """Save the trained model and TF-IDF vectorizer using joblib."""
    print("=" * 60)
    print("💾 SAVING MODEL & VECTORIZER")
    print("=" * 60)
    
    joblib.dump(model, MODEL_PATH)
    print(f"\n   Model saved to:      {MODEL_PATH}")
    
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print(f"   Vectorizer saved to: {VECTORIZER_PATH}")
    print()


# ══════════════════════════════════════════════
# MAIN EXECUTION
# ══════════════════════════════════════════════
def main():
    """Main training pipeline."""
    print("\n" + "═" * 60)
    print("  🔍 FAKE PRODUCT REVIEW DETECTION SYSTEM")
    print("  Training Pipeline")
    print("═" * 60 + "\n")
    
    # ── Load & Preprocess ──
    try:
        df = load_dataset(DATA_PATH)
    except (FileNotFoundError, ValueError) as e:
        print(e)
        sys.exit(1)
    
    df = preprocess_data(df)
    
    # ── EDA ──
    df = perform_eda(df)
    
    # ── Train/Test Split ──
    X_train, X_test, y_train, y_test = split_data(df)
    
    # ── TF-IDF (Baseline) ──
    X_train_tfidf, X_test_tfidf, vectorizer = apply_tfidf(X_train, X_test)
    
    # ── Train Model (Baseline) ──
    model, training_history = train_model(X_train_tfidf, y_train, X_test_tfidf, y_test)
    
    # ── Evaluate (Baseline) ──
    metrics = evaluate_model(model, X_test_tfidf, y_test)
    
    # ── Generate Loss & Accuracy Graphs ──
    generate_training_graphs(training_history, test_accuracy=metrics['accuracy'])
    
    # ── Check if tuning is needed ──
    if metrics['accuracy'] < TARGET_ACCURACY:
        print(f"⚠️  Baseline accuracy ({metrics['accuracy']:.4f}) is below target ({TARGET_ACCURACY}).")
        print(f"   Starting hyperparameter tuning...\n")
        
        # Tune using GridSearchCV
        vectorizer, model, X_train_tfidf = tune_model(X_train, y_train)
        
        # Re-transform test data with the new vectorizer
        X_test_tfidf = vectorizer.transform(X_test)
        
        # Re-evaluate
        print("\n📈 RE-EVALUATION WITH TUNED MODEL:")
        metrics = evaluate_model(model, X_test_tfidf, y_test)
        
        if metrics['accuracy'] < TARGET_ACCURACY:
            print(f"\n⚠️  Even after tuning, accuracy ({metrics['accuracy']:.4f}) is below {TARGET_ACCURACY}.")
            print(f"   This may be a limitation of the dataset with Logistic Regression.")
            print(f"   The model has been saved with the best accuracy achieved.\n")
    else:
        print(f"✅ Accuracy ({metrics['accuracy']:.4f}) meets the target (>{TARGET_ACCURACY})!")
        print()
    
    # ── Save ──
    save_artifacts(model, vectorizer)
    
    # ── Final Summary ──
    print("═" * 60)
    print("  ✅ TRAINING COMPLETE!")
    print("═" * 60)
    print(f"\n  Final Accuracy: {metrics['accuracy'] * 100:.2f}%")
    print(f"\n  Saved files:")
    print(f"    📦 Model:            {MODEL_PATH}")
    print(f"    📦 Vectorizer:       {VECTORIZER_PATH}")
    print(f"    📊 Confusion Matrix: {CONFUSION_MATRIX_PATH}")
    print(f"    📉 Model Loss:       {LOSS_GRAPH_PATH}")
    print(f"    📈 Model Accuracy:   {ACCURACY_GRAPH_PATH}")
    print(f"    📄 Metrics:          {METRICS_PATH}")
    print(f"\n  Next step: Run 'python src/predict.py' to test predictions!")
    print()


if __name__ == "__main__":
    main()
