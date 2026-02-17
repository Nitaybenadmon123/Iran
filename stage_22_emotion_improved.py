"""
Stage 22 - IMPROVED Emotion Analysis for Iranian Twitter Data
Includes text cleaning and probability-based dominance rule to reduce "others" classifications
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


def load_and_prepare_data(file_path='Posts_sentiment_improved.csv'):
    """Load CSV and prepare data for emotion analysis"""
    print("="*70)
    print("LOADING DATA")
    print("="*70)
    
    if not os.path.exists(file_path):
        # Try alternative files
        alternatives = ['posts.csv', 'Posts_sentiment.csv']
        for alt_file in alternatives:
            if os.path.exists(alt_file):
                print(f"File '{file_path}' not found, loading '{alt_file}' instead...")
                file_path = alt_file
                break
        else:
            raise FileNotFoundError(f"No suitable input file found!")
    
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
    
    print(f"Using column '{text_col}' for emotion analysis")
    
    # Convert created_at to datetime
    if 'created_at' in df.columns:
        df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
        print(f"Converted 'created_at' to datetime")
    
    # Handle missing values in text column
    df[text_col] = df[text_col].fillna('')
    
    # Apply text cleaning if not already done
    if 'clean_text' not in df.columns:
        print("\nApplying text cleaning...")
        df['clean_text'] = df[text_col].apply(clean_text_for_nlp)
    else:
        print("\nUsing existing cleaned text column")
    
    # Filter out empty texts after cleaning
    df_valid = df[df['clean_text'].str.strip() != ''].copy()
    
    print(f"\nValid tweets with cleaned text: {len(df_valid):,}")
    print(f"Removed {len(df) - len(df_valid)} tweets with empty cleaned text")
    print(f"Unique users: {df_valid['username'].nunique()}")
    
    return df_valid


def perform_emotion_analysis(df, batch_size=32):
    """Perform emotion analysis using pysentimiento with probability-based reclassification"""
    print("\n" + "="*70)
    print("EMOTION ANALYSIS WITH IMPROVED CLASSIFICATION")
    print("="*70)
    
    print("Initializing pysentimiento emotion analyzer...")
    
    try:
        from pysentimiento import create_analyzer
        analyzer = create_analyzer(task="emotion", lang="en")
        print("✓ Emotion analyzer loaded successfully")
    except ImportError:
        print("ERROR: pysentimiento not installed. Installing now...")
        import subprocess
        subprocess.run(["pip", "install", "pysentimiento"], check=True)
        from pysentimiento import create_analyzer
        analyzer = create_analyzer(task="emotion", lang="en")
        print("✓ Emotion analyzer installed and loaded")
    
    print(f"\nAnalyzing {len(df):,} tweets...")
    print(f"Batch size: {batch_size}")
    
    # Initialize result lists
    emotion_labels_original = []
    emotion_joy_probs = []
    emotion_anger_probs = []
    emotion_sadness_probs = []
    emotion_fear_probs = []
    emotion_surprise_probs = []
    emotion_disgust_probs = []
    
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
                    emotion_labels_original.append('others')
                    emotion_joy_probs.append(0.0)
                    emotion_anger_probs.append(0.0)
                    emotion_sadness_probs.append(0.0)
                    emotion_fear_probs.append(0.0)
                    emotion_surprise_probs.append(0.0)
                    emotion_disgust_probs.append(0.0)
                    continue
                
                # Get emotion prediction
                result = analyzer.predict(text_truncated)
                
                # Extract label and probabilities
                emotion_labels_original.append(result.output)
                emotion_joy_probs.append(result.probas.get('joy', 0.0))
                emotion_anger_probs.append(result.probas.get('anger', 0.0))
                emotion_sadness_probs.append(result.probas.get('sadness', 0.0))
                emotion_fear_probs.append(result.probas.get('fear', 0.0))
                emotion_surprise_probs.append(result.probas.get('surprise', 0.0))
                emotion_disgust_probs.append(result.probas.get('disgust', 0.0))
                
            except Exception as e:
                # Handle errors gracefully
                error_count += 1
                emotion_labels_original.append('others')
                emotion_joy_probs.append(0.0)
                emotion_anger_probs.append(0.0)
                emotion_sadness_probs.append(0.0)
                emotion_fear_probs.append(0.0)
                emotion_surprise_probs.append(0.0)
                emotion_disgust_probs.append(0.0)
    
    if error_count > 0:
        print(f"\nWarning: {error_count} tweets had processing errors (defaulted to 'others')")
    
    # Add results to dataframe
    df['emotion_label_original'] = emotion_labels_original
    df['emotion_joy'] = emotion_joy_probs
    df['emotion_anger'] = emotion_anger_probs
    df['emotion_sadness'] = emotion_sadness_probs
    df['emotion_fear'] = emotion_fear_probs
    df['emotion_surprise'] = emotion_surprise_probs
    df['emotion_disgust'] = emotion_disgust_probs
    
    print("\n✓ Initial emotion analysis completed")
    
    # Apply DOMINANCE RULE
    print("\nApplying probability-based dominance rule...")
    print("Rule: If original_label == 'others' AND max_prob >= 0.35, use dominant emotion")
    
    emotion_labels_fixed = []
    top_emotions = []
    top_probs = []
    
    for idx, row in df.iterrows():
        probs = {
            'joy': row['emotion_joy'],
            'anger': row['emotion_anger'],
            'sadness': row['emotion_sadness'],
            'fear': row['emotion_fear'],
            'surprise': row['emotion_surprise'],
            'disgust': row['emotion_disgust']
        }
        
        top_emotion = max(probs, key=probs.get)
        top_prob = probs[top_emotion]
        
        # Apply dominance rule
        if row['emotion_label_original'] == 'others' and top_prob >= 0.35:
            emotion_labels_fixed.append(top_emotion)
        else:
            emotion_labels_fixed.append(row['emotion_label_original'])
        
        top_emotions.append(top_emotion)
        top_probs.append(top_prob)
    
    df['emotion_label_fixed'] = emotion_labels_fixed
    df['top_emotion'] = top_emotions
    df['top_prob'] = top_probs
    
    print("✓ Dominance rule applied successfully")
    
    return df


def calculate_statistics(df):
    """Calculate and display emotion statistics before and after improvement"""
    print("\n" + "="*70)
    print("EMOTION STATISTICS - BEFORE VS AFTER")
    print("="*70)
    
    # Original distribution
    original_counts = df['emotion_label_original'].value_counts()
    original_pct = df['emotion_label_original'].value_counts(normalize=True) * 100
    
    # Fixed distribution
    fixed_counts = df['emotion_label_fixed'].value_counts()
    fixed_pct = df['emotion_label_fixed'].value_counts(normalize=True) * 100
    
    print("\n📊 Emotion Distribution Comparison:")
    print("-" * 70)
    print(f"{'Emotion':<12} {'Original Count':>15} {'Original %':>12} {'Fixed Count':>15} {'Fixed %':>12}")
    print("-" * 70)
    
    all_emotions = set(list(original_counts.index) + list(fixed_counts.index))
    for emotion in sorted(all_emotions):
        orig_count = original_counts.get(emotion, 0)
        orig_pct_val = original_pct.get(emotion, 0.0)
        fixed_count = fixed_counts.get(emotion, 0)
        fixed_pct_val = fixed_pct.get(emotion, 0.0)
        
        print(f"{emotion:<12} {orig_count:>15,} {orig_pct_val:>11.2f}% {fixed_count:>15,} {fixed_pct_val:>11.2f}%")
    
    # Calculate reduction in "others"
    orig_others = original_counts.get('others', 0)
    fixed_others = fixed_counts.get('others', 0)
    others_reduction = orig_others - fixed_others
    others_reduction_pct = (others_reduction / orig_others * 100) if orig_others > 0 else 0
    
    print("-" * 70)
    print(f"\n💡 'Others' Reduction:")
    print(f"   Original 'Others': {orig_others:,} ({original_pct.get('others', 0):.2f}%)")
    print(f"   Fixed 'Others':    {fixed_others:,} ({fixed_pct.get('others', 0):.2f}%)")
    print(f"   Reduction:         {others_reduction:,} tweets ({others_reduction_pct:.2f}% reduction)")
    
    # Show examples where label changed
    print("\n📝 Example Tweets Where Label Changed (others -> specific emotion):")
    print("-" * 70)
    
    changed = df[df['emotion_label_original'] != df['emotion_label_fixed']].copy()
    if len(changed) > 0:
        print(f"Total changed: {len(changed):,} tweets\n")
        
        for idx, (_, row) in enumerate(changed.head(5).iterrows(), 1):
            print(f"Example {idx}:")
            print(f"  Text: {row['clean_text'][:100]}...")
            print(f"  Original: {row['emotion_label_original']} -> Fixed: {row['emotion_label_fixed']}")
            print(f"  Top emotion: {row['top_emotion']} (prob={row['top_prob']:.3f})")
            print(f"  joy={row['emotion_joy']:.3f}, anger={row['emotion_anger']:.3f}, " + 
                  f"sadness={row['emotion_sadness']:.3f}")
            print()
    
    # Top 10 tweets with highest emotion probability
    print("\n🔝 Top 10 Tweets with Highest Emotion Probability:")
    print("-" * 70)
    top_confident = df.nlargest(10, 'top_prob')
    for idx, (_, row) in enumerate(top_confident.iterrows(), 1):
        print(f"{idx:2d}. Emotion: {row['emotion_label_fixed']:<10} (prob={row['top_prob']:.3f})")
        print(f"    Text: {row['clean_text'][:80]}...")
        print()
    
    # Monthly trends
    if 'created_at' in df.columns:
        df['year_month'] = df['created_at'].dt.to_period('M')
        monthly_emotions = df.groupby('year_month').agg({
            'emotion_joy': 'mean',
            'emotion_anger': 'mean',
            'emotion_sadness': 'mean',
            'emotion_fear': 'mean',
            'emotion_surprise': 'mean',
            'emotion_disgust': 'mean',
            'tweet_id': 'count'
        }).rename(columns={'tweet_id': 'tweet_count'})
        
        return monthly_emotions
    
    return None


def create_visualizations(df, monthly_emotions):
    """Create before/after comparison and other visualizations"""
    print("\n" + "="*70)
    print("CREATING VISUALIZATIONS")
    print("="*70)
    
    # Create output directory
    output_dir = 'outputs/emotion_analysis_improved'
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Define colors for emotions
    emotion_colors = {
        'joy': '#FFD700',
        'anger': '#FF4444',
        'sadness': '#4169E1',
        'fear': '#800080',
        'surprise': '#FFA500',
        'disgust': '#228B22',
        'others': '#808080'
    }
    
    # 1. Bar Chart - BEFORE vs AFTER
    print("\n📊 Creating before vs after comparison...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    
    # BEFORE
    original_counts = df['emotion_label_original'].value_counts().sort_values(ascending=False)
    bar_colors1 = [emotion_colors.get(emotion, '#808080') for emotion in original_counts.index]
    
    bars1 = ax1.bar(range(len(original_counts)), original_counts.values, 
                    color=bar_colors1, edgecolor='black', linewidth=1.5)
    ax1.set_xticks(range(len(original_counts)))
    ax1.set_xticklabels(original_counts.index, rotation=45, ha='right')
    
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}',
                ha='center', va='bottom', fontsize=10, weight='bold')
    
    ax1.set_ylabel('Number of Tweets', fontsize=12, weight='bold')
    ax1.set_title('BEFORE: Original Classification', fontsize=13, weight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # AFTER
    fixed_counts = df['emotion_label_fixed'].value_counts().sort_values(ascending=False)
    bar_colors2 = [emotion_colors.get(emotion, '#808080') for emotion in fixed_counts.index]
    
    bars2 = ax2.bar(range(len(fixed_counts)), fixed_counts.values,
                    color=bar_colors2, edgecolor='black', linewidth=1.5)
    ax2.set_xticks(range(len(fixed_counts)))
    ax2.set_xticklabels(fixed_counts.index, rotation=45, ha='right')
    
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}',
                ha='center', va='bottom', fontsize=10, weight='bold')
    
    ax2.set_ylabel('Number of Tweets', fontsize=12, weight='bold')
    ax2.set_title('AFTER: Improved Classification', fontsize=13, weight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    fig.suptitle('Emotion Distribution - Before vs After Dominance Rule\n(Stage 22 Improved)',
                 fontsize=15, weight='bold', y=1.00)
    
    plt.tight_layout()
    output_file = f'{output_dir}/emotion_before_after_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_file}")
    plt.close()
    
    # 2. Pie Chart - AFTER (Fixed)
    print("📊 Creating pie chart with improved labels...")
    fig, ax = plt.subplots(figsize=(10, 8))
    
    plot_colors = [emotion_colors.get(emotion, '#808080') for emotion in fixed_counts.index]
    
    wedges, texts, autotexts = ax.pie(
        fixed_counts.values,
        labels=fixed_counts.index,
        colors=plot_colors,
        autopct='%1.1f%%',
        startangle=90,
        textprops={'fontsize': 11, 'weight': 'bold'}
    )
    
    for autotext in autotexts:
        autotext.set_color('white')
    
    ax.set_title('Improved Emotion Distribution\n(Stage 22 Improved)',
                 fontsize=14, weight='bold', pad=20)
    
    plt.tight_layout()
    output_file = f'{output_dir}/emotion_distribution_pie_improved.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_file}")
    plt.close()
    
    # 3. Monthly Trend - Using Fixed Labels
    if monthly_emotions is not None and len(monthly_emotions) > 0:
        print("📈 Creating monthly trends with improved labels...")
        fig, ax = plt.subplots(figsize=(14, 8))
        
        monthly_plot = monthly_emotions.copy()
        monthly_plot.index = monthly_plot.index.to_timestamp()
        
        ax.plot(monthly_plot.index, monthly_plot['emotion_joy'],
                marker='o', linewidth=2.5, color=emotion_colors['joy'], label='Joy', markersize=7)
        ax.plot(monthly_plot.index, monthly_plot['emotion_anger'],
                marker='s', linewidth=2.5, color=emotion_colors['anger'], label='Anger', markersize=7)
        ax.plot(monthly_plot.index, monthly_plot['emotion_sadness'],
                marker='^', linewidth=2.5, color=emotion_colors['sadness'], label='Sadness', markersize=7)
        ax.plot(monthly_plot.index, monthly_plot['emotion_fear'],
                marker='D', linewidth=2.5, color=emotion_colors['fear'], label='Fear', markersize=7)
        ax.plot(monthly_plot.index, monthly_plot['emotion_surprise'],
                marker='v', linewidth=2.5, color=emotion_colors['surprise'], label='Surprise', markersize=7)
        ax.plot(monthly_plot.index, monthly_plot['emotion_disgust'],
                marker='*', linewidth=2.5, color=emotion_colors['disgust'], label='Disgust', markersize=8)
        
        ax.set_xlabel('Month', fontsize=12, weight='bold')
        ax.set_ylabel('Average Emotion Probability', fontsize=12, weight='bold')
        ax.set_title('Monthly Emotion Trends - Improved Classification\n(Stage 22 Improved)',
                     fontsize=14, weight='bold', pad=20)
        ax.legend(loc='best', fontsize=11, framealpha=0.9, ncol=2)
        ax.grid(True, alpha=0.3)
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        output_file = f'{output_dir}/emotion_monthly_trends_improved.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {output_file}")
        plt.close()
    
    print("\n✓ All visualizations created successfully")


def save_results(df, output_file='Posts_sentiment_emotion_improved.csv'):
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
    print("STAGE 22 - IMPROVED EMOTION ANALYSIS")
    print("Iranian Twitter Data Science Project")
    print("="*70)
    start_time = datetime.now()
    print(f"Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    try:
        # Step 1: Load and prepare data
        df = load_and_prepare_data('Posts_sentiment_improved.csv')
        
        # Step 2: Perform emotion analysis with dominance rule
        df = perform_emotion_analysis(df, batch_size=32)
        
        # Step 3: Calculate statistics
        monthly_emotions = calculate_statistics(df)
        
        # Step 4: Create visualizations
        create_visualizations(df, monthly_emotions)
        
        # Step 5: Save results
        save_results(df, 'Posts_sentiment_emotion_improved.csv')
        
        # Final summary
        end_time = datetime.now()
        duration = end_time - start_time
        
        print("\n" + "="*70)
        print("✓ STAGE 22 IMPROVED COMPLETED SUCCESSFULLY")
        print("="*70)
        print(f"Started at:   {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Completed at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Duration:     {duration}")
        
        print("\n📁 Output Files:")
        print("-" * 70)
        print("  1. Posts_sentiment_emotion_improved.csv")
        print("  2. outputs/emotion_analysis_improved/emotion_before_after_comparison.png")
        print("  3. outputs/emotion_analysis_improved/emotion_distribution_pie_improved.png")
        print("  4. outputs/emotion_analysis_improved/emotion_monthly_trends_improved.png")
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
