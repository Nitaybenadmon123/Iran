"""
Stage 21 - IMPROVED Sentiment Analysis for Iranian Twitter Data
Includes text cleaning and probability-based dominance rule to reduce neutral classifications
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
import warnings
import os
import re
warnings.filterwarnings('ignore')

# Set style for visualizations
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


def clean_text_for_nlp(text):
    """
    Clean text for NLP analysis
    - Removes URLs, RT, @mentions
    - Replaces hashtags with words
    - Keeps emojis and punctuation
    - Does NOT remove short texts
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    
    # Remove URLs (http, https, www)
    text = re.sub(r'http\S+|www\.\S+', '', text)
    
    # Remove RT
    text = re.sub(r'\bRT\b', '', text, flags=re.IGNORECASE)
    
    # Remove @mentions
    text = re.sub(r'@\w+', '', text)
    
    # Replace hashtags (#word -> word)
    text = re.sub(r'#(\w+)', r'\1', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def load_and_prepare_data(file_path='posts.csv'):
    """Load posts CSV and prepare data for sentiment analysis"""
    print("="*70)
    print("LOADING DATA")
    print("="*70)
    
    if not os.path.exists(file_path):
        # Try alternative file
        alt_file = 'Posts_sentiment.csv'
        if os.path.exists(alt_file):
            print(f"File '{file_path}' not found, loading '{alt_file}' instead...")
            file_path = alt_file
        else:
            raise FileNotFoundError(f"Neither '{file_path}' nor '{alt_file}' found!")
    
    print(f"Loading {file_path}...")
    df = pd.read_csv(file_path)
    
    print(f"Loaded {len(df):,} rows with {len(df.columns)} columns")
    
    # Identify the English text column
    if 'Text_en' in df.columns:
        text_col = 'Text_en'
    elif 'text_translated_en' in df.columns:
        text_col = 'text_translated_en'
    else:
        raise ValueError("No English text column found! Expected 'Text_en' or 'text_translated_en'")
    
    print(f"Using column '{text_col}' for sentiment analysis")
    
    # Convert created_at to datetime
    if 'created_at' in df.columns:
        df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
        print(f"Converted 'created_at' to datetime")
    
    # Handle missing values in text column
    df[text_col] = df[text_col].fillna('')
    
    # Apply text cleaning
    print("\nApplying text cleaning...")
    df['clean_text'] = df[text_col].apply(clean_text_for_nlp)
    
    # Filter out empty texts after cleaning
    df_valid = df[df['clean_text'].str.strip() != ''].copy()
    
    print(f"\nValid tweets with cleaned text: {len(df_valid):,}")
    print(f"Removed {len(df) - len(df_valid)} tweets with empty cleaned text")
    print(f"Unique users: {df_valid['username'].nunique()}")
    
    return df_valid


def perform_sentiment_analysis(df, batch_size=32):
    """Perform sentiment analysis using pysentimiento with probability-based reclassification"""
    print("\n" + "="*70)
    print("SENTIMENT ANALYSIS WITH IMPROVED CLASSIFICATION")
    print("="*70)
    
    print("Initializing pysentimiento sentiment analyzer...")
    
    try:
        from pysentimiento import create_analyzer
        analyzer = create_analyzer(task="sentiment", lang="en")
        print("✓ Sentiment analyzer loaded successfully")
    except ImportError:
        print("ERROR: pysentimiento not installed. Installing now...")
        import subprocess
        subprocess.run(["pip", "install", "pysentimiento"], check=True)
        from pysentimiento import create_analyzer
        analyzer = create_analyzer(task="sentiment", lang="en")
        print("✓ Sentiment analyzer installed and loaded")
    
    print(f"\nAnalyzing {len(df):,} tweets...")
    print(f"Batch size: {batch_size}")
    
    # Initialize result lists
    sentiment_labels_original = []
    sentiment_pos_probs = []
    sentiment_neg_probs = []
    sentiment_neu_probs = []
    
    # Process in batches with progress bar
    total_batches = (len(df) + batch_size - 1) // batch_size
    error_count = 0
    
    for i in tqdm(range(0, len(df), batch_size), desc="Processing batches", total=total_batches):
        batch = df.iloc[i:i+batch_size]['clean_text'].tolist()
        
        for text in batch:
            try:
                # Truncate very long texts to avoid memory issues
                text_truncated = text[:512] if len(text) > 512 else text
                
                # Skip empty texts
                if not text_truncated or text_truncated.isspace():
                    sentiment_labels_original.append('NEU')
                    sentiment_pos_probs.append(0.0)
                    sentiment_neg_probs.append(0.0)
                    sentiment_neu_probs.append(1.0)
                    continue
                
                # Get sentiment prediction
                result = analyzer.predict(text_truncated)
                
                # Extract label and probabilities
                sentiment_labels_original.append(result.output)
                sentiment_pos_probs.append(result.probas.get('POS', 0.0))
                sentiment_neg_probs.append(result.probas.get('NEG', 0.0))
                sentiment_neu_probs.append(result.probas.get('NEU', 0.0))
                
            except Exception as e:
                # Handle errors gracefully
                error_count += 1
                sentiment_labels_original.append('NEU')
                sentiment_pos_probs.append(0.0)
                sentiment_neg_probs.append(0.0)
                sentiment_neu_probs.append(1.0)
    
    if error_count > 0:
        print(f"\nWarning: {error_count} tweets had processing errors (defaulted to NEU)")
    
    # Add results to dataframe
    df['sentiment_label_original'] = sentiment_labels_original
    df['sentiment_positive'] = sentiment_pos_probs
    df['sentiment_negative'] = sentiment_neg_probs
    df['sentiment_neutral'] = sentiment_neu_probs
    
    print("\n✓ Initial sentiment analysis completed")
    
    # Apply DOMINANCE RULE
    print("\nApplying probability-based dominance rule...")
    print("Rule: If original_label == 'NEU' AND max_prob >= 0.40, use dominant class")
    
    sentiment_labels_fixed = []
    top_probs = []
    
    for idx, row in df.iterrows():
        probs = {
            'POS': row['sentiment_positive'],
            'NEG': row['sentiment_negative'],
            'NEU': row['sentiment_neutral']
        }
        
        top_class = max(probs, key=probs.get)
        top_prob = probs[top_class]
        
        # Apply dominance rule
        if row['sentiment_label_original'] == 'NEU' and top_prob >= 0.40:
            sentiment_labels_fixed.append(top_class)
        else:
            sentiment_labels_fixed.append(row['sentiment_label_original'])
        
        top_probs.append(top_prob)
    
    df['sentiment_label_fixed'] = sentiment_labels_fixed
    df['top_prob'] = top_probs
    
    print("✓ Dominance rule applied successfully")
    
    return df


def calculate_statistics(df):
    """Calculate and display sentiment statistics before and after improvement"""
    print("\n" + "="*70)
    print("SENTIMENT STATISTICS - BEFORE VS AFTER")
    print("="*70)
    
    # Original distribution
    original_counts = df['sentiment_label_original'].value_counts()
    original_pct = df['sentiment_label_original'].value_counts(normalize=True) * 100
    
    # Fixed distribution
    fixed_counts = df['sentiment_label_fixed'].value_counts()
    fixed_pct = df['sentiment_label_fixed'].value_counts(normalize=True) * 100
    
    print("\n📊 Sentiment Distribution Comparison:")
    print("-" * 70)
    print(f"{'Label':<10} {'Original Count':>15} {'Original %':>12} {'Fixed Count':>15} {'Fixed %':>12}")
    print("-" * 70)
    
    for label in ['POS', 'NEU', 'NEG']:
        orig_count = original_counts.get(label, 0)
        orig_pct_val = original_pct.get(label, 0.0)
        fixed_count = fixed_counts.get(label, 0)
        fixed_pct_val = fixed_pct.get(label, 0.0)
        
        print(f"{label:<10} {orig_count:>15,} {orig_pct_val:>11.2f}% {fixed_count:>15,} {fixed_pct_val:>11.2f}%")
    
    # Calculate reduction in neutral
    orig_neu = original_counts.get('NEU', 0)
    fixed_neu = fixed_counts.get('NEU', 0)
    neu_reduction = orig_neu - fixed_neu
    neu_reduction_pct = (neu_reduction / orig_neu * 100) if orig_neu > 0 else 0
    
    print("-" * 70)
    print(f"\n💡 Neutral Reduction:")
    print(f"   Original Neutral:  {orig_neu:,} ({original_pct.get('NEU', 0):.2f}%)")
    print(f"   Fixed Neutral:     {fixed_neu:,} ({fixed_pct.get('NEU', 0):.2f}%)")
    print(f"   Reduction:         {neu_reduction:,} tweets ({neu_reduction_pct:.2f}% reduction)")
    
    # Show examples where label changed
    print("\n📝 Example Tweets Where Label Changed (NEU -> POS/NEG):")
    print("-" * 70)
    
    changed = df[df['sentiment_label_original'] != df['sentiment_label_fixed']].copy()
    if len(changed) > 0:
        print(f"Total changed: {len(changed):,} tweets\n")
        
        for idx, (_, row) in enumerate(changed.head(5).iterrows(), 1):
            print(f"Example {idx}:")
            print(f"  Text: {row['clean_text'][:100]}...")
            print(f"  Original: {row['sentiment_label_original']} -> Fixed: {row['sentiment_label_fixed']}")
            print(f"  POS={row['sentiment_positive']:.3f}, NEU={row['sentiment_neutral']:.3f}, NEG={row['sentiment_negative']:.3f}")
            print()
    
    # Monthly trends
    if 'created_at' in df.columns:
        df['year_month'] = df['created_at'].dt.to_period('M')
        monthly_sentiment = df.groupby('year_month').agg({
            'sentiment_positive': 'mean',
            'sentiment_negative': 'mean',
            'sentiment_neutral': 'mean',
            'tweet_id': 'count'
        }).rename(columns={'tweet_id': 'tweet_count'})
        
        return monthly_sentiment
    
    return None


def create_visualizations(df, monthly_sentiment):
    """Create before/after comparison visualizations"""
    print("\n" + "="*70)
    print("CREATING VISUALIZATIONS")
    print("="*70)
    
    # Create output directory
    output_dir = 'outputs/sentiment_analysis_improved'
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Define colors
    colors = {'POS': '#2ecc71', 'NEU': '#95a5a6', 'NEG': '#e74c3c'}
    
    # 1. Bar Chart - BEFORE vs AFTER
    print("\n📊 Creating before vs after comparison...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # BEFORE
    original_counts = df['sentiment_label_original'].value_counts()
    labels_order = ['POS', 'NEU', 'NEG']
    original_values = [original_counts.get(label, 0) for label in labels_order]
    bar_colors = [colors[label] for label in labels_order]
    
    bars1 = ax1.bar(labels_order, original_values, color=bar_colors, edgecolor='black', linewidth=1.5)
    
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}',
                ha='center', va='bottom', fontsize=12, weight='bold')
    
    ax1.set_ylabel('Number of Tweets', fontsize=12, weight='bold')
    ax1.set_title('BEFORE: Original Classification', fontsize=13, weight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # AFTER
    fixed_counts = df['sentiment_label_fixed'].value_counts()
    fixed_values = [fixed_counts.get(label, 0) for label in labels_order]
    
    bars2 = ax2.bar(labels_order, fixed_values, color=bar_colors, edgecolor='black', linewidth=1.5)
    
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}',
                ha='center', va='bottom', fontsize=12, weight='bold')
    
    ax2.set_ylabel('Number of Tweets', fontsize=12, weight='bold')
    ax2.set_title('AFTER: Improved Classification', fontsize=13, weight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    fig.suptitle('Sentiment Distribution - Before vs After Dominance Rule\n(Stage 21 Improved)',
                 fontsize=15, weight='bold', y=1.02)
    
    plt.tight_layout()
    output_file = f'{output_dir}/sentiment_before_after_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_file}")
    plt.close()
    
    # 2. Monthly Trend - Using Fixed Labels
    if monthly_sentiment is not None and len(monthly_sentiment) > 0:
        print("📈 Creating monthly trends with improved labels...")
        fig, ax = plt.subplots(figsize=(14, 7))
        
        monthly_plot = monthly_sentiment.copy()
        monthly_plot.index = monthly_plot.index.to_timestamp()
        
        ax.plot(monthly_plot.index, monthly_plot['sentiment_positive'],
                marker='o', linewidth=2.5, color=colors['POS'], label='Positive', markersize=7)
        ax.plot(monthly_plot.index, monthly_plot['sentiment_neutral'],
                marker='s', linewidth=2.5, color=colors['NEU'], label='Neutral', markersize=7)
        ax.plot(monthly_plot.index, monthly_plot['sentiment_negative'],
                marker='^', linewidth=2.5, color=colors['NEG'], label='Negative', markersize=7)
        
        ax.set_xlabel('Month', fontsize=12, weight='bold')
        ax.set_ylabel('Average Sentiment Probability', fontsize=12, weight='bold')
        ax.set_title('Monthly Sentiment Trends - Improved Classification\n(Stage 21 Improved)',
                     fontsize=14, weight='bold', pad=20)
        ax.legend(loc='best', fontsize=11, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        output_file = f'{output_dir}/sentiment_monthly_trends_improved.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {output_file}")
        plt.close()
    
    print("\n✓ All visualizations created successfully")


def save_results(df, output_file='Posts_sentiment_improved.csv'):
    """Save the improved results to CSV file"""
    print("\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70)
    
    print(f"Saving to {output_file}...")
    
    # Save with all columns
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    file_size_mb = os.path.getsize(output_file) / 1024 / 1024
    
    print(f"✓ Saved {len(df):,} tweets to {output_file}")
    print(f"✓ File size: {file_size_mb:.2f} MB")
    print(f"✓ Columns: {len(df.columns)}")


def main():
    """Main execution function"""
    print("\n" + "="*70)
    print("STAGE 21 - IMPROVED SENTIMENT ANALYSIS")
    print("Iranian Twitter Data Science Project")
    print("="*70)
    start_time = datetime.now()
    print(f"Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    try:
        # Step 1: Load and clean data
        df = load_and_prepare_data('posts.csv')
        
        # Step 2: Perform sentiment analysis with dominance rule
        df = perform_sentiment_analysis(df, batch_size=32)
        
        # Step 3: Calculate statistics
        monthly_sentiment = calculate_statistics(df)
        
        # Step 4: Create visualizations
        create_visualizations(df, monthly_sentiment)
        
        # Step 5: Save results
        save_results(df, 'Posts_sentiment_improved.csv')
        
        # Final summary
        end_time = datetime.now()
        duration = end_time - start_time
        
        print("\n" + "="*70)
        print("✓ STAGE 21 IMPROVED COMPLETED SUCCESSFULLY")
        print("="*70)
        print(f"Started at:   {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Completed at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Duration:     {duration}")
        
        print("\n📁 Output Files:")
        print("-" * 70)
        print("  1. Posts_sentiment_improved.csv")
        print("  2. outputs/sentiment_analysis_improved/sentiment_before_after_comparison.png")
        print("  3. outputs/sentiment_analysis_improved/sentiment_monthly_trends_improved.png")
        print("="*70)
        
    except Exception as e:
        print("\n" + "="*70)
        print("❌ ERROR OCCURRED")
        print("="*70)
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        print("="*70)
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
