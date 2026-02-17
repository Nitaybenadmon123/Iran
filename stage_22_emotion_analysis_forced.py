"""
Stage 22 - Emotion Analysis (Forced 6 Emotions Only)
Iranian Twitter Data Science Project

CRITICAL: Every tweet MUST be assigned to one of 6 emotions (no "others")
Uses argmax to force classification into: joy, anger, sadness, fear, surprise, disgust
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
import warnings
import os
warnings.filterwarnings('ignore')

# Set style for visualizations
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10


def load_and_prepare_data(file_path='Posts_sentiment.csv'):
    """Load CSV and prepare data for emotion analysis"""
    print("="*70)
    print("LOADING DATA")
    print("="*70)
    
    # Try multiple file options
    if not os.path.exists(file_path):
        alternatives = ['posts.csv', 'Posts_sentiment_improved.csv']
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
    
    # Keep all rows (do not filter)
    df_valid = df.copy()
    df_valid['text_for_analysis'] = df_valid[text_col]
    
    print(f"\nTotal tweets for analysis: {len(df_valid):,}")
    print(f"Unique users: {df_valid['username'].nunique()}")
    
    # Get year range for later
    if 'created_at' in df_valid.columns:
        years = df_valid['created_at'].dt.year.dropna().unique()
        years = sorted([int(y) for y in years if pd.notna(y)])
        print(f"Year range in data: {min(years)} - {max(years)}")
    
    return df_valid, text_col


def perform_emotion_analysis(df, text_col, batch_size=32):
    """
    Perform emotion analysis using pysentimiento
    FORCED: Every tweet gets one of 6 emotions (no "others")
    """
    print("\n" + "="*70)
    print("EMOTION ANALYSIS - FORCED 6-EMOTION CLASSIFICATION")
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
    print("⚠️  RULE: Every tweet MUST be assigned to one of 6 emotions (argmax)")
    
    # Initialize result lists
    emotion_labels = []
    emotion_top_probs = []
    emotion_joy_probs = []
    emotion_anger_probs = []
    emotion_sadness_probs = []
    emotion_fear_probs = []
    emotion_surprise_probs = []
    emotion_disgust_probs = []
    
    # The 6 allowed emotions
    emotion_keys = ['joy', 'anger', 'sadness', 'fear', 'surprise', 'disgust']
    
    error_count = 0
    
    # Process in batches with progress bar
    total_batches = (len(df) + batch_size - 1) // batch_size
    
    for i in tqdm(range(0, len(df), batch_size), desc="Processing batches", total=total_batches):
        batch = df.iloc[i:i+batch_size]['text_for_analysis'].tolist()
        
        for text in batch:
            try:
                # Truncate very long texts to 512 characters
                text_truncated = str(text)[:512] if len(str(text)) > 512 else str(text)
                
                # Handle empty texts
                if not text_truncated or text_truncated.isspace() or text_truncated == 'nan':
                    # Default: equal probabilities, choose sadness as default
                    probs_dict = {
                        'joy': 0.167,
                        'anger': 0.167,
                        'sadness': 0.167,
                        'fear': 0.167,
                        'surprise': 0.166,
                        'disgust': 0.166
                    }
                    chosen_emotion = 'sadness'
                    top_prob = 0.167
                else:
                    # Get emotion prediction
                    result = analyzer.predict(text_truncated)
                    
                    # Extract probabilities for the 6 emotions
                    probs_dict = {
                        'joy': result.probas.get('joy', 0.0),
                        'anger': result.probas.get('anger', 0.0),
                        'sadness': result.probas.get('sadness', 0.0),
                        'fear': result.probas.get('fear', 0.0),
                        'surprise': result.probas.get('surprise', 0.0),
                        'disgust': result.probas.get('disgust', 0.0)
                    }
                    
                    # FORCE: Choose the emotion with max probability among the 6
                    chosen_emotion = max(probs_dict, key=probs_dict.get)
                    top_prob = probs_dict[chosen_emotion]
                
                # Store results
                emotion_labels.append(chosen_emotion)
                emotion_top_probs.append(top_prob)
                emotion_joy_probs.append(probs_dict['joy'])
                emotion_anger_probs.append(probs_dict['anger'])
                emotion_sadness_probs.append(probs_dict['sadness'])
                emotion_fear_probs.append(probs_dict['fear'])
                emotion_surprise_probs.append(probs_dict['surprise'])
                emotion_disgust_probs.append(probs_dict['disgust'])
                
            except Exception as e:
                # Handle errors: default to sadness with equal probs
                error_count += 1
                emotion_labels.append('sadness')
                emotion_top_probs.append(0.167)
                emotion_joy_probs.append(0.167)
                emotion_anger_probs.append(0.167)
                emotion_sadness_probs.append(0.167)
                emotion_fear_probs.append(0.167)
                emotion_surprise_probs.append(0.166)
                emotion_disgust_probs.append(0.166)
    
    # Add results to dataframe
    df['emotion_label'] = emotion_labels
    df['emotion_top_prob'] = emotion_top_probs
    df['emotion_joy'] = emotion_joy_probs
    df['emotion_anger'] = emotion_anger_probs
    df['emotion_sadness'] = emotion_sadness_probs
    df['emotion_fear'] = emotion_fear_probs
    df['emotion_surprise'] = emotion_surprise_probs
    df['emotion_disgust'] = emotion_disgust_probs
    
    if error_count > 0:
        print(f"\n⚠️  Warning: {error_count} tweets had processing errors (defaulted to sadness)")
    
    print("\n✓ Emotion analysis completed successfully")
    print(f"✓ All {len(df):,} tweets classified into 6 emotions (no 'others')")
    
    return df, error_count


def calculate_statistics(df):
    """Calculate and display emotion statistics"""
    print("\n" + "="*70)
    print("EMOTION STATISTICS")
    print("="*70)
    
    # Overall emotion distribution
    emotion_counts = df['emotion_label'].value_counts()
    emotion_pct = df['emotion_label'].value_counts(normalize=True) * 100
    
    print("\n📊 Emotion Distribution (All Tweets):")
    print("-" * 70)
    print(f"{'Emotion':<12} {'Count':>10} {'Percentage':>12}")
    print("-" * 70)
    
    emotion_order = ['joy', 'anger', 'sadness', 'fear', 'surprise', 'disgust']
    for emotion in emotion_order:
        count = emotion_counts.get(emotion, 0)
        pct = emotion_pct.get(emotion, 0.0)
        print(f"{emotion:<12} {count:>10,} {pct:>11.2f}%")
    
    print("-" * 70)
    print(f"{'TOTAL':<12} {len(df):>10,} {100.0:>11.2f}%")
    
    # Average emotion probabilities per user (top 10)
    print("\n👤 Top 10 Users by Average Joy:")
    print("-" * 70)
    user_avg = df.groupby('username').agg({
        'emotion_joy': 'mean',
        'emotion_anger': 'mean',
        'emotion_sadness': 'mean',
        'emotion_fear': 'mean',
        'emotion_surprise': 'mean',
        'emotion_disgust': 'mean',
        'tweet_id': 'count'
    }).rename(columns={'tweet_id': 'tweet_count'})
    
    # Filter users with at least 10 tweets
    user_avg = user_avg[user_avg['tweet_count'] >= 10]
    
    top_joy = user_avg.sort_values('emotion_joy', ascending=False).head(10)
    for idx, (username, row) in enumerate(top_joy.iterrows(), 1):
        print(f"  {idx:2d}. @{username:25s} - Joy: {row['emotion_joy']:.3f} (n={int(row['tweet_count'])})")
    
    # Monthly trends
    if 'created_at' in df.columns:
        print("\n📅 Monthly Emotion Averages (Overall):")
        print("-" * 70)
        df['year_month'] = df['created_at'].dt.to_period('M')
        monthly = df.groupby('year_month').agg({
            'emotion_joy': 'mean',
            'emotion_anger': 'mean',
            'emotion_sadness': 'mean',
            'emotion_fear': 'mean',
            'emotion_surprise': 'mean',
            'emotion_disgust': 'mean',
            'tweet_id': 'count'
        }).rename(columns={'tweet_id': 'tweet_count'})
        
        # Show last 12 months
        print(monthly.tail(12).round(3).to_string())
        
        return monthly
    
    return None


def create_yearly_emotion_plots(df):
    """
    Create one PNG per year with 6 stacked subplots (one per emotion)
    showing monthly average probabilities
    """
    print("\n" + "="*70)
    print("CREATING YEARLY EMOTION TREND PLOTS")
    print("="*70)
    
    output_dir = 'outputs/emotion_analysis/by_year'
    os.makedirs(output_dir, exist_ok=True)
    
    if 'created_at' not in df.columns:
        print("⚠️  No 'created_at' column found, skipping yearly plots")
        return []
    
    # Get unique years
    df['year'] = df['created_at'].dt.year
    df['month'] = df['created_at'].dt.month
    years = sorted(df['year'].dropna().unique())
    
    print(f"Generating plots for {len(years)} years: {years}")
    
    created_files = []
    
    for year in years:
        year_int = int(year)
        print(f"\n  Processing year {year_int}...")
        
        # Filter data for this year
        df_year = df[df['year'] == year_int].copy()
        
        if len(df_year) == 0:
            print(f"    ⚠️  No data for year {year_int}, skipping")
            continue
        
        # Calculate monthly averages for this year
        monthly_year = df_year.groupby('month').agg({
            'emotion_fear': 'mean',
            'emotion_anger': 'mean',
            'emotion_sadness': 'mean',
            'emotion_joy': 'mean',
            'emotion_surprise': 'mean',
            'emotion_disgust': 'mean',
            'tweet_id': 'count'
        }).rename(columns={'tweet_id': 'tweet_count'})
        
        # Reindex to have all 12 months (fill missing with NaN)
        monthly_year = monthly_year.reindex(range(1, 13))
        
        # Create figure with 6 stacked subplots
        fig, axes = plt.subplots(6, 1, figsize=(14, 12), sharex=True)
        fig.suptitle(f'Emotion Trends - Year {year_int}\n(Stage 22 Analysis)', 
                     fontsize=16, weight='bold', y=0.995)
        
        emotions = [
            ('emotion_fear', 'Fear', '#800080', axes[0]),
            ('emotion_anger', 'Anger', '#FF4444', axes[1]),
            ('emotion_sadness', 'Sadness', '#4169E1', axes[2]),
            ('emotion_joy', 'Joy', '#FFD700', axes[3]),
            ('emotion_surprise', 'Surprise', '#FFA500', axes[4]),
            ('emotion_disgust', 'Disgust', '#228B22', axes[5])
        ]
        
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        for emotion_col, emotion_name, color, ax in emotions:
            values = monthly_year[emotion_col].values
            months = monthly_year.index.values
            
            # Plot line
            ax.plot(months, values, marker='o', linewidth=2.5, color=color, 
                   markersize=7, label=emotion_name)
            
            # Fill area under curve
            ax.fill_between(months, values, alpha=0.3, color=color)
            
            ax.set_ylabel('Avg Prob', fontsize=10, weight='bold')
            ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, max(0.5, values.max() * 1.1) if not all(pd.isna(values)) else 0.5)
        
        # Set x-axis labels only on bottom subplot
        axes[5].set_xlabel('Month', fontsize=11, weight='bold')
        axes[5].set_xticks(range(1, 13))
        axes[5].set_xticklabels(month_names, rotation=0)
        
        plt.tight_layout()
        
        # Save figure
        output_file = f'{output_dir}/emotion_trends_{year_int}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        created_files.append(output_file)
        print(f"    ✓ Saved: {output_file}")
        plt.close()
    
    print(f"\n✓ Created {len(created_files)} yearly emotion trend plots")
    return created_files


def create_overall_visualizations(df):
    """Create overall bar chart and pie chart for emotion distribution"""
    print("\n" + "="*70)
    print("CREATING OVERALL EMOTION VISUALIZATIONS")
    print("="*70)
    
    output_dir = 'outputs/emotion_analysis'
    os.makedirs(output_dir, exist_ok=True)
    
    emotion_colors = {
        'joy': '#FFD700',
        'anger': '#FF4444',
        'sadness': '#4169E1',
        'fear': '#800080',
        'surprise': '#FFA500',
        'disgust': '#228B22'
    }
    
    emotion_counts = df['emotion_label'].value_counts()
    
    # 1. Bar Chart
    print("\n📊 Creating bar chart...")
    fig, ax = plt.subplots(figsize=(12, 7))
    
    emotion_order = ['joy', 'anger', 'sadness', 'fear', 'surprise', 'disgust']
    values = [emotion_counts.get(e, 0) for e in emotion_order]
    colors = [emotion_colors[e] for e in emotion_order]
    
    bars = ax.bar(emotion_order, values, color=colors, edgecolor='black', linewidth=1.5)
    
    # Add count labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}',
                ha='center', va='bottom', fontsize=11, weight='bold')
    
    ax.set_xlabel('Emotion', fontsize=12, weight='bold')
    ax.set_ylabel('Number of Tweets', fontsize=12, weight='bold')
    ax.set_title('Emotion Distribution - All Tweets\n(Stage 22 Analysis - Forced 6 Emotions)',
                 fontsize=14, weight='bold', pad=20)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    bar_file = f'{output_dir}/emotion_distribution_bar.png'
    plt.savefig(bar_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {bar_file}")
    plt.close()
    
    # 2. Pie Chart
    print("📊 Creating pie chart...")
    fig, ax = plt.subplots(figsize=(10, 8))
    
    plot_colors = [emotion_colors[e] for e in emotion_order]
    plot_values = [emotion_counts.get(e, 0) for e in emotion_order]
    
    wedges, texts, autotexts = ax.pie(
        plot_values,
        labels=emotion_order,
        colors=plot_colors,
        autopct='%1.1f%%',
        startangle=90,
        textprops={'fontsize': 11, 'weight': 'bold'}
    )
    
    for autotext in autotexts:
        autotext.set_color('white')
    
    ax.set_title('Emotion Distribution - All Tweets\n(Stage 22 Analysis - Forced 6 Emotions)',
                 fontsize=14, weight='bold', pad=20)
    
    plt.tight_layout()
    pie_file = f'{output_dir}/emotion_distribution_pie.png'
    plt.savefig(pie_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {pie_file}")
    plt.close()
    
    return [bar_file, pie_file]


def save_results(df, output_file='Posts_sentiment_emotion.csv'):
    """Save the results to CSV file"""
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
    
    return output_file


def main():
    """Main execution function"""
    print("\n" + "="*70)
    print("STAGE 22 - EMOTION ANALYSIS (FORCED 6 EMOTIONS)")
    print("Iranian Twitter Data Science Project")
    print("="*70)
    start_time = datetime.now()
    print(f"Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    try:
        # Step 1: Load data
        df, text_col = load_and_prepare_data('Posts_sentiment.csv')
        
        # Step 2: Perform emotion analysis (forced 6 emotions)
        df, error_count = perform_emotion_analysis(df, text_col, batch_size=32)
        
        # Step 3: Calculate statistics
        monthly_stats = calculate_statistics(df)
        
        # Step 4: Create yearly emotion plots (6 stacked subplots per year)
        yearly_files = create_yearly_emotion_plots(df)
        
        # Step 5: Create overall visualizations
        overall_files = create_overall_visualizations(df)
        
        # Step 6: Save results
        output_csv = save_results(df, 'Posts_sentiment_emotion.csv')
        
        # Final summary
        end_time = datetime.now()
        duration = end_time - start_time
        
        print("\n" + "="*70)
        print("✓ STAGE 22 COMPLETED SUCCESSFULLY")
        print("="*70)
        print(f"Started at:   {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Completed at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Duration:     {duration}")
        
        print("\n📁 Output Files:")
        print("-" * 70)
        print(f"  CSV: {output_csv}")
        print(f"\n  Overall Visualizations:")
        for f in overall_files:
            print(f"    - {f}")
        print(f"\n  Yearly Trend Plots ({len(yearly_files)} files):")
        for f in yearly_files:
            print(f"    - {f}")
        
        print("\n📊 Processing Summary:")
        print("-" * 70)
        print(f"  Total tweets processed: {len(df):,}")
        print(f"  Inference errors: {error_count}")
        print(f"  All tweets classified: 6 emotions (no 'others')")
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
