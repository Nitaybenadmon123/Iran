"""
Stage 23 - Topic Detection using BERTopic (Version 2)
Iranian Twitter Data Science Project

Follows course instructions exactly with:
- Improved hyperparameter ranges to reduce over-clustering
- Topic reduction after full model fit
- Exact output format requirements
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from tqdm import tqdm
import warnings
import os
import re
from typing import Tuple, List, Dict
warnings.filterwarnings('ignore')

# BERTopic and related libraries
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer

# Gensim for coherence calculation
from gensim.models import CoherenceModel
from gensim.corpora import Dictionary

# Set style
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10
sns.set_style("whitegrid")


def clean_text_for_topic(text):
    """
    Minimal cleaning for topic modeling:
    - Remove URLs
    - Remove RT
    - Remove @mentions
    - Convert #word -> word
    - Normalize whitespace
    Keep emojis and most punctuation.
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    
    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)
    
    # Remove RT
    text = re.sub(r'\bRT\b', '', text, flags=re.IGNORECASE)
    
    # Remove @mentions
    text = re.sub(r'@\w+', '', text)
    
    # Convert hashtags (#word -> word)
    text = re.sub(r'#(\w+)', r'\1', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def load_and_prepare_data(file_path='Posts_sentiment_emotion_cleaned.csv'):
    """
    PART A: Load data and prepare for topic modeling
    """
    print("\n" + "="*70)
    print("PART A — LOADING + CLEANING DATA")
    print("="*70)
    
    if not os.path.exists(file_path):
        print(f"File '{file_path}' not found, trying alternatives...")
        alternatives = ['Posts_sentiment_emotion.csv', 'Posts_sentiment.csv', 'posts.csv']
        for alt in alternatives:
            if os.path.exists(alt):
                file_path = alt
                break
        else:
            raise FileNotFoundError(f"No suitable input file found!")
    
    print(f"Loading: {file_path}")
    df = pd.read_csv(file_path)
    print(f"✓ Loaded {len(df):,} rows with {len(df.columns)} columns")
    
    # Convert created_at to datetime
    if 'created_at' in df.columns:
        df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
        print(f"✓ Converted 'created_at' to datetime")
    
    # Identify English text column
    if 'Text_en' in df.columns:
        text_col = 'Text_en'
    elif 'text_translated_en' in df.columns:
        text_col = 'text_translated_en'
    else:
        raise ValueError("No English text column found (Text_en or text_translated_en)")
    
    print(f"✓ Using text column: '{text_col}'")
    
    # Fill NA and create clean_text_topic column
    df[text_col] = df[text_col].fillna('')
    
    print("\nApplying minimal text cleaning for topic modeling...")
    df['clean_text_topic'] = df[text_col].apply(clean_text_for_topic)
    
    # Drop rows with empty clean_text_topic
    original_len = len(df)
    df = df[df['clean_text_topic'].str.strip().str.len() > 0].copy()
    dropped = original_len - len(df)
    
    print(f"✓ Cleaned {original_len:,} texts")
    print(f"✓ Dropped {dropped:,} empty texts after cleaning")
    print(f"✓ Final dataset: {len(df):,} tweets")
    
    if 'created_at' in df.columns and df['created_at'].notna().any():
        years = df['created_at'].dt.year.dropna().unique()
        years = sorted([int(y) for y in years if pd.notna(y)])
        print(f"✓ Year range: {min(years)} - {max(years)}")
    
    print(f"✓ Unique users: {df['username'].nunique() if 'username' in df.columns else 'N/A'}")
    
    return df, text_col


def sample_for_tuning(df, sample_fraction=0.075, random_state=42):
    """
    PART B: Sample for grid search (7.5%)
    """
    print("\n" + "="*70)
    print("PART B — SAMPLING FOR GRID SEARCH")
    print("="*70)
    
    sample_size = int(len(df) * sample_fraction)
    df_sample = df.sample(n=sample_size, random_state=random_state)
    
    print(f"Sample fraction: {sample_fraction*100:.1f}%")
    print(f"Sample size: {len(df_sample):,} tweets")
    print(f"Random state: {random_state}")
    
    return df_sample


def compute_coherence_score(topic_model, docs, top_n=10):
    """
    Compute coherence score (c_v) using gensim
    """
    try:
        topics_dict = topic_model.get_topics()
        
        if not topics_dict or len(topics_dict) <= 1:
            return 0.0
        
        # Remove outlier topic (-1)
        if -1 in topics_dict:
            del topics_dict[-1]
        
        if len(topics_dict) == 0:
            return 0.0
        
        # Extract top words for each topic
        topic_words = []
        for topic_id in sorted(topics_dict.keys()):
            words = [word for word, _ in topic_model.get_topic(topic_id)[:top_n]]
            if words:
                topic_words.append(words)
        
        if len(topic_words) < 2:
            return 0.0
        
        # Tokenize documents
        tokenized_docs = [doc.lower().split() for doc in docs if doc]
        
        if len(tokenized_docs) == 0:
            return 0.0
        
        # Create dictionary
        dictionary = Dictionary(tokenized_docs)
        
        # Compute coherence
        coherence_model = CoherenceModel(
            topics=topic_words,
            texts=tokenized_docs,
            dictionary=dictionary,
            coherence='c_v'
        )
        
        coherence_score = coherence_model.get_coherence()
        return coherence_score
    
    except Exception as e:
        print(f"      ⚠ Coherence calculation failed: {e}")
        return 0.0


def grid_search_hyperparameters(sample_texts, min_topic_sizes, n_neighbors_values):
    """
    PART C: Improved grid search with reduced over-clustering
    """
    print("\n" + "="*70)
    print("PART C — IMPROVED GRID SEARCH")
    print("="*70)
    
    print(f"Testing {len(min_topic_sizes)} × {len(n_neighbors_values)} = {len(min_topic_sizes) * len(n_neighbors_values)} combinations")
    print(f"min_topic_sizes: {min_topic_sizes}")
    print(f"n_neighbors: {n_neighbors_values}")
    print("\nEmbedding model: all-MiniLM-L6-v2")
    
    # Load embedding model once
    print("\nLoading SentenceTransformer...")
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    print("✓ Embedding model loaded")
    
    # Pre-compute embeddings for sample
    print(f"\nGenerating embeddings for {len(sample_texts):,} sample texts...")
    sample_embeddings = embedding_model.encode(sample_texts, show_progress_bar=True)
    print(f"✓ Embeddings shape: {sample_embeddings.shape}")
    
    results = []
    
    print("\n" + "-"*70)
    print("Running grid search...")
    print("-"*70)
    
    total_combinations = len(min_topic_sizes) * len(n_neighbors_values)
    pbar = tqdm(total=total_combinations, desc="Grid search progress")
    
    for min_topic_size in min_topic_sizes:
        for n_neighbors in n_neighbors_values:
            try:
                # Create UMAP model
                umap_model = UMAP(
                    n_neighbors=n_neighbors,
                    n_components=5,
                    min_dist=0.0,
                    metric='cosine',
                    random_state=42
                )
                
                # Create HDBSCAN model
                hdbscan_model = HDBSCAN(
                    min_cluster_size=min_topic_size,
                    metric='euclidean',
                    cluster_selection_method='eom',
                    prediction_data=True
                )
                
                # Create BERTopic model
                topic_model = BERTopic(
                    embedding_model=embedding_model,
                    umap_model=umap_model,
                    hdbscan_model=hdbscan_model,
                    vectorizer_model=CountVectorizer(stop_words="english", min_df=2),
                    calculate_probabilities=True,
                    verbose=False
                )
                
                # Fit model on sample
                topics, probs = topic_model.fit_transform(sample_texts, sample_embeddings)
                
                # Calculate metrics
                n_topics = len(set(topics)) - (1 if -1 in topics else 0)
                outlier_count = sum(1 for t in topics if t == -1)
                outlier_ratio = outlier_count / len(topics)
                
                # Calculate coherence
                coherence = compute_coherence_score(topic_model, sample_texts, top_n=10)
                
                results.append({
                    'min_topic_size': min_topic_size,
                    'n_neighbors': n_neighbors,
                    'n_topics': n_topics,
                    'coherence': coherence,
                    'outlier_ratio': outlier_ratio
                })
                
                pbar.set_postfix({
                    'min_size': min_topic_size,
                    'n_neigh': n_neighbors,
                    'topics': n_topics,
                    'coh': f"{coherence:.3f}",
                    'out': f"{outlier_ratio:.2%}"
                })
                
            except Exception as e:
                print(f"\n      ⚠ Error with min_size={min_topic_size}, n_neighbors={n_neighbors}: {e}")
                results.append({
                    'min_topic_size': min_topic_size,
                    'n_neighbors': n_neighbors,
                    'n_topics': 0,
                    'coherence': 0.0,
                    'outlier_ratio': 1.0
                })
            
            pbar.update(1)
    
    pbar.close()
    
    # Create results dataframe
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('coherence', ascending=False)
    
    print("\n" + "="*70)
    print("GRID SEARCH RESULTS (sorted by coherence)")
    print("="*70)
    print(results_df.to_string(index=False))
    
    # Save results
    os.makedirs('outputs/topic_modeling', exist_ok=True)
    results_file = 'outputs/topic_modeling/grid_search_results_v2.csv'
    results_df.to_csv(results_file, index=False)
    print(f"\n✓ Grid search results saved to {results_file}")
    
    return results_df


def select_best_configuration(results_df):
    """
    PART D: Model selection matching course criteria
    """
    print("\n" + "="*70)
    print("PART D — MODEL SELECTION")
    print("="*70)
    
    print("Selection criteria:")
    print("  1) 10 <= n_topics <= 60")
    print("  2) outlier_ratio <= 0.20")
    print("  3) Choose maximum coherence among candidates")
    
    # Filter by criteria
    candidates = results_df[
        (results_df['n_topics'] >= 10) &
        (results_df['n_topics'] <= 60) &
        (results_df['outlier_ratio'] <= 0.20)
    ].copy()
    
    if len(candidates) == 0:
        print("\n⚠ No candidates with outlier_ratio <= 0.20")
        print("   Relaxing to outlier_ratio <= 0.30...")
        candidates = results_df[
            (results_df['n_topics'] >= 10) &
            (results_df['n_topics'] <= 60) &
            (results_df['outlier_ratio'] <= 0.30)
        ].copy()
    
    if len(candidates) == 0:
        print("\n⚠ Still no candidates, using best coherence regardless of outlier ratio...")
        best_config = results_df.iloc[0]
    else:
        # Select best by coherence
        best_config = candidates.iloc[0]
    
    print("\n" + "="*70)
    print("✓ SELECTED CONFIGURATION:")
    print("="*70)
    print(f"  min_topic_size:    {int(best_config['min_topic_size'])}")
    print(f"  n_neighbors:       {int(best_config['n_neighbors'])}")
    print(f"  n_topics (sample): {int(best_config['n_topics'])}")
    print(f"  coherence:         {best_config['coherence']:.4f}")
    print(f"  outlier_ratio:     {best_config['outlier_ratio']:.2%}")
    print("="*70)
    
    return best_config


def fit_final_model_on_full_data(df, best_config):
    """
    PART E: Final model on full dataset with topic reduction
    """
    print("\n" + "="*70)
    print("PART E — FITTING FINAL MODEL ON FULL DATA")
    print("="*70)
    
    print(f"Total tweets for final model: {len(df):,}")
    print(f"\nUsing selected hyperparameters:")
    print(f"  min_topic_size: {int(best_config['min_topic_size'])}")
    print(f"  n_neighbors:    {int(best_config['n_neighbors'])}")
    
    # Prepare texts
    full_texts = df['clean_text_topic'].tolist()
    
    # Load embedding model
    print("\n1) Loading SentenceTransformer...")
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    print("   ✓ Model loaded")
    
    # Generate embeddings
    print(f"\n2) Generating embeddings for {len(full_texts):,} texts...")
    full_embeddings = embedding_model.encode(full_texts, show_progress_bar=True)
    print(f"   ✓ Embeddings shape: {full_embeddings.shape}")
    
    # Create final model
    print("\n3) Creating BERTopic model...")
    
    umap_model = UMAP(
        n_neighbors=int(best_config['n_neighbors']),
        n_components=5,
        min_dist=0.0,
        metric='cosine',
        random_state=42
    )
    
    hdbscan_model = HDBSCAN(
        min_cluster_size=int(best_config['min_topic_size']),
        metric='euclidean',
        cluster_selection_method='eom',
        prediction_data=True
    )
    
    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=CountVectorizer(stop_words="english", min_df=2),
        calculate_probabilities=True,
        verbose=False
    )
    
    # Fit model on full dataset
    print("\n4) Fitting model on full dataset...")
    topics, probs = topic_model.fit_transform(full_texts, full_embeddings)
    
    # Initial metrics
    initial_n_topics = len(set(topics)) - (1 if -1 in topics else 0)
    initial_outlier_count = sum(1 for t in topics if t == -1)
    initial_outlier_ratio = initial_outlier_count / len(topics)
    
    print(f"\n   ✓ Initial model fitted")
    print(f"     Initial n_topics: {initial_n_topics}")
    print(f"     Initial outlier_ratio: {initial_outlier_ratio:.2%}")
    
    # TOPIC REDUCTION: Reduce to manageable number (40 topics)
    print("\n5) Reducing topics to 40 for interpretability...")
    try:
        topic_model.reduce_topics(full_texts, nr_topics=40)
        
        # Re-get topics after reduction
        topics = topic_model.topics_
        
        n_topics_after_reduction = len(set(topics)) - (1 if -1 in topics else 0)
        print(f"   ✓ Topics reduced to {n_topics_after_reduction}")
    except Exception as e:
        print(f"   ⚠ Topic reduction failed: {e}")
        print("   Continuing with original topics...")
    
    # OUTLIER HANDLING
    outlier_count = sum(1 for t in topics if t == -1)
    outlier_ratio = outlier_count / len(topics)
    
    print(f"\n6) Handling outliers (current ratio: {outlier_ratio:.2%})...")
    
    if outlier_ratio > 0.15:
        print(f"   Outlier ratio > 0.15, applying outlier reduction...")
        try:
            new_topics = topic_model.reduce_outliers(full_texts, topics)
            
            new_outlier_count = sum(1 for t in new_topics if t == -1)
            new_outlier_ratio = new_outlier_count / len(new_topics)
            
            print(f"   ✓ New outlier ratio: {new_outlier_ratio:.2%}")
            print(f"     Reduced by: {(outlier_ratio - new_outlier_ratio):.2%}")
            
            topics = new_topics
            outlier_ratio = new_outlier_ratio
            
            # Update topics in model
            topic_model.update_topics(full_texts, topics=topics)
            
        except Exception as e:
            print(f"   ⚠ Outlier reduction failed: {e}")
            print("   Continuing with current topics...")
    else:
        print(f"   Outlier ratio <= 0.15, no reduction needed")
    
    # Final metrics
    final_n_topics = len(set(topics)) - (1 if -1 in topics else 0)
    final_outlier_ratio = outlier_ratio
    
    # Add to dataframe
    df['topic_id'] = topics
    
    # Get max probability for each document
    print("\n7) Computing topic probabilities...")
    
    # Get document-topic probabilities
    if hasattr(topic_model, 'probabilities_') and topic_model.probabilities_ is not None:
        doc_probs = topic_model.probabilities_
        topic_probs = []
        for i, prob_dist in enumerate(doc_probs):
            if topics[i] == -1:
                topic_probs.append(0.0)
            else:
                topic_probs.append(float(np.max(prob_dist)))
    else:
        # Fallback: use approximation
        topic_probs = [0.5 if t != -1 else 0.0 for t in topics]
    
    df['probability'] = topic_probs
    
    print(f"\n" + "="*70)
    print("✓ FINAL MODEL SUMMARY:")
    print("="*70)
    print(f"  Total tweets:          {len(df):,}")
    print(f"  Final n_topics:        {final_n_topics}")
    print(f"  Final outlier_ratio:   {final_outlier_ratio:.2%}")
    print(f"  Avg probability:       {np.mean(topic_probs):.3f}")
    print("="*70)
    
    return df, topic_model, final_n_topics, final_outlier_ratio


def export_final_output(df, text_col):
    """
    PART F: Export Data/Topics.csv with exact format
    """
    print("\n" + "="*70)
    print("PART F — EXPORTING FINAL OUTPUT")
    print("="*70)
    
    # Create output dataframe with EXACT columns
    output_df = pd.DataFrame()
    
    output_df['tweet_id'] = df['tweet_id'] if 'tweet_id' in df.columns else range(len(df))
    output_df['created_at'] = df['created_at'] if 'created_at' in df.columns else pd.NaT
    
    # Use Text_en if exists, else text_translated_en
    if text_col == 'Text_en':
        output_df['text_translated_en'] = df['Text_en']
    else:
        output_df['text_translated_en'] = df[text_col] if text_col in df.columns else df['clean_text_topic']
    
    output_df['topic_id'] = df['topic_id']
    output_df['probability'] = df['probability']
    
    # Create Data directory
    os.makedirs('Data', exist_ok=True)
    
    output_file = 'Data/Topics.csv'
    output_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    file_size_mb = os.path.getsize(output_file) / 1024 / 1024
    
    print(f"✓ Saved {len(output_df):,} tweets to {output_file}")
    print(f"✓ File size: {file_size_mb:.2f} MB")
    print(f"✓ Columns: {list(output_df.columns)}")
    
    return output_file


def save_topic_info_and_report(topic_model, df):
    """
    PART G: Topic reporting with info table and representative docs
    """
    print("\n" + "="*70)
    print("PART G — TOPIC REPORTING")
    print("="*70)
    
    # Get topic info
    topic_info = topic_model.get_topic_info()
    
    # Save topic info table
    info_file = 'outputs/topic_modeling/topic_info.csv'
    topic_info.to_csv(info_file, index=False, encoding='utf-8-sig')
    print(f"\n✓ Topic info table saved to {info_file}")
    
    print("\n" + "="*70)
    print("TOPIC INFO TABLE:")
    print("="*70)
    print(topic_info.to_string(index=False))
    
    # Get all topics (excluding -1)
    topics = sorted([t for t in topic_model.get_topics().keys() if t != -1])
    
    print("\n" + "="*70)
    print("DETAILED TOPIC ANALYSIS")
    print("="*70)
    
    for topic_id in topics:
        print(f"\n{'='*70}")
        print(f"TOPIC {topic_id}")
        print('='*70)
        
        # Get topic words
        topic_words = topic_model.get_topic(topic_id)
        n_tweets = len(df[df['topic_id'] == topic_id])
        
        print(f"\n📌 Top 10 Words:")
        for i, (word, score) in enumerate(topic_words[:10], 1):
            print(f"   {i:2d}. {word:20s} (score: {score:6.4f})")
        
        print(f"\n📈 Number of tweets: {n_tweets:,}")
        
        # Get representative documents
        print(f"\n📄 3 Representative Tweets:")
        try:
            repr_docs = topic_model.get_representative_docs(topic_id)
            for i, doc in enumerate(repr_docs[:3], 1):
                doc_display = doc[:120] + "..." if len(doc) > 120 else doc
                print(f"   {i}. {doc_display}")
        except Exception as e:
            print(f"   ⚠ Could not retrieve representative docs: {e}")
    
    return info_file


def create_visualizations(topic_model, df):
    """
    PART H: Create and save visualizations
    """
    print("\n" + "="*70)
    print("PART H — CREATING VISUALIZATIONS")
    print("="*70)
    
    output_dir = 'outputs/topic_modeling'
    os.makedirs(output_dir, exist_ok=True)
    
    created_files = []
    
    # 1) Intertopic distance map
    print("\n1) Creating intertopic distance map...")
    try:
        fig1 = topic_model.visualize_topics()
        file1 = f'{output_dir}/intertopic_distance_map.html'
        fig1.write_html(file1)
        created_files.append(file1)
        print(f"   ✓ Saved: {file1}")
    except Exception as e:
        print(f"   ⚠ Failed: {e}")
    
    # 2) Topic hierarchy
    print("\n2) Creating topic hierarchy...")
    try:
        fig2 = topic_model.visualize_hierarchy()
        file2 = f'{output_dir}/topic_hierarchy.html'
        fig2.write_html(file2)
        created_files.append(file2)
        print(f"   ✓ Saved: {file2}")
    except Exception as e:
        print(f"   ⚠ Failed: {e}")
    
    # 3) Bar chart of top 15 topics
    print("\n3) Creating bar chart of top 15 topics...")
    try:
        fig3 = topic_model.visualize_barchart(top_n_topics=15)
        file3 = f'{output_dir}/top_15_topics_barchart.html'
        fig3.write_html(file3)
        created_files.append(file3)
        print(f"   ✓ Saved: {file3}")
    except Exception as e:
        print(f"   ⚠ Failed: {e}")
    
    # 4) Topic trends over time
    print("\n4) Creating topic trends over time...")
    if 'created_at' in df.columns and df['created_at'].notna().any():
        try:
            # Group by month and topic
            df['year_month'] = df['created_at'].dt.to_period('M')
            
            monthly_topics = df.groupby(['year_month', 'topic_id']).size().reset_index(name='count')
            
            # Get top 5 topics by total count (excluding -1)
            topic_counts = df[df['topic_id'] != -1]['topic_id'].value_counts()
            top_5_topics = topic_counts.head(5).index.tolist()
            
            # Pivot for plotting
            pivot_data = monthly_topics[monthly_topics['topic_id'].isin(top_5_topics)].pivot(
                index='year_month',
                columns='topic_id',
                values='count'
            ).fillna(0)
            
            # Plot stacked area chart
            fig, ax = plt.subplots(figsize=(16, 8))
            
            # Convert period to timestamp for plotting
            x = [p.to_timestamp() for p in pivot_data.index]
            
            # Stacked area plot
            ax.stackplot(x, *[pivot_data[col].values for col in pivot_data.columns],
                        labels=[f'Topic {col}' for col in pivot_data.columns],
                        alpha=0.7)
            
            ax.set_xlabel('Month', fontsize=12, weight='bold')
            ax.set_ylabel('Number of Tweets', fontsize=12, weight='bold')
            ax.set_title('Top 5 Topics Over Time (Stacked Area Chart)\nStage 23 - Topic Detection',
                        fontsize=14, weight='bold')
            ax.legend(loc='upper left', fontsize=10)
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            file4 = f'{output_dir}/topic_trends_over_time.png'
            plt.savefig(file4, dpi=300, bbox_inches='tight')
            plt.close()
            
            created_files.append(file4)
            print(f"   ✓ Saved: {file4}")
            
        except Exception as e:
            print(f"   ⚠ Time trend visualization failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("   ⚠ No timestamp data available for time trends")
    
    print(f"\n✓ Created {len(created_files)} visualizations")
    
    return created_files


def main():
    """Main execution"""
    print("\n" + "="*70)
    print("STAGE 23 — TOPIC DETECTION USING BERTopic (V2)")
    print("Course Instructions Implementation")
    print("="*70)
    start_time = datetime.now()
    print(f"Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    try:
        # PART A: Load and prepare data
        df, text_col = load_and_prepare_data('Posts_sentiment_emotion_cleaned.csv')
        
        # PART B: Sample for tuning
        df_sample = sample_for_tuning(df, sample_fraction=0.075, random_state=42)
        sample_texts = df_sample['clean_text_topic'].tolist()
        
        # PART C: Grid search with improved parameters
        min_topic_sizes = [30, 40, 60]
        n_neighbors_values = [15, 25]
        
        results_df = grid_search_hyperparameters(
            sample_texts,
            min_topic_sizes,
            n_neighbors_values
        )
        
        # PART D: Select best configuration
        best_config = select_best_configuration(results_df)
        
        # PART E: Fit final model on full data
        df, topic_model, final_n_topics, final_outlier_ratio = fit_final_model_on_full_data(df, best_config)
        
        # PART F: Export final output
        output_file = export_final_output(df, text_col)
        
        # PART G: Topic info and reporting
        info_file = save_topic_info_and_report(topic_model, df)
        
        # PART H: Visualizations
        viz_files = create_visualizations(topic_model, df)
        
        # Save model
        print("\n" + "="*70)
        print("SAVING MODEL")
        print("="*70)
        model_file = 'outputs/topic_modeling/bertopic_model_v2'
        topic_model.save(
            model_file,
            serialization="safetensors",
            save_ctfidf=True,
            save_embedding_model="sentence-transformers/all-MiniLM-L6-v2"
        )
        print(f"✓ Model saved to {model_file}")
        
        # Calculate final coherence on sample for reference
        print("\n" + "="*70)
        print("COMPUTING FINAL METRICS")
        print("="*70)
        
        print("Computing coherence on sample...")
        sample_coherence = compute_coherence_score(topic_model, sample_texts[:1000], top_n=10)
        print(f"✓ Sample coherence: {sample_coherence:.4f}")
        
        # Final summary
        end_time = datetime.now()
        duration = end_time - start_time
        
        print("\n" + "="*70)
        print("✓ STAGE 23 COMPLETED SUCCESSFULLY")
        print("="*70)
        print(f"Duration: {duration}")
        
        print("\n📊 FINAL SUMMARY:")
        print("-"*70)
        print(f"  Total tweets:            {len(df):,}")
        print(f"  Selected min_topic_size: {int(best_config['min_topic_size'])}")
        print(f"  Selected n_neighbors:    {int(best_config['n_neighbors'])}")
        print(f"  Final n_topics:          {final_n_topics}")
        print(f"  Final outlier_ratio:     {final_outlier_ratio:.2%}")
        print(f"  Sample coherence (c_v):  {sample_coherence:.4f}")
        
        print("\n📁 OUTPUT FILES:")
        print(f"  Main CSV:    {output_file}")
        print(f"  Topic info:  {info_file}")
        print(f"  Model:       {model_file}")
        print(f"  Visuals:     {len(viz_files)} files in outputs/topic_modeling/")
        
        print("\n📈 OUTPUT STRUCTURE:")
        print(f"  Data/Topics.csv columns: tweet_id | created_at | text_translated_en | topic_id | probability")
        print(f"  Grid search results:     outputs/topic_modeling/grid_search_results_v2.csv")
        
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
