"""
Stage 23 - Improved Topic Trends Visualization
Focus on 2022-2026 with normalized and smoothed charts
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['font.size'] = 10


def load_topic_names(topic_info_path='outputs/topic_modeling/topic_info.csv'):
    """Load topic names from topic_info.csv if available"""
    topic_names = {}
    
    if os.path.exists(topic_info_path):
        try:
            topic_info = pd.read_csv(topic_info_path)
            
            # Extract top 3 words for each topic
            for _, row in topic_info.iterrows():
                topic_id = row['Topic']
                
                # Skip outlier topic
                if topic_id == -1:
                    continue
                
                # Get representation (top words)
                if 'Representation' in row:
                    words = str(row['Representation']).replace('[', '').replace(']', '').replace("'", "").split(',')
                    words = [w.strip() for w in words[:3]]
                    label = f"Topic {topic_id} ({', '.join(words)})"
                elif 'Name' in row:
                    label = f"Topic {topic_id} ({row['Name']})"
                else:
                    label = f"Topic {topic_id}"
                
                topic_names[topic_id] = label
            
            print(f"✓ Loaded topic names for {len(topic_names)} topics")
        except Exception as e:
            print(f"⚠ Could not load topic names: {e}")
    else:
        print(f"⚠ Topic info file not found: {topic_info_path}")
    
    return topic_names


def create_improved_visualizations():
    """Create improved topic trend visualizations for 2022-2026"""
    
    print("\n" + "="*70)
    print("STAGE 23 - IMPROVED TOPIC TRENDS VISUALIZATION")
    print("="*70)
    
    # 1) Load Data/Topics.csv
    print("\n1) Loading Data/Topics.csv...")
    topics_file = 'Data/Topics.csv'
    
    if not os.path.exists(topics_file):
        print(f"❌ ERROR: {topics_file} not found!")
        return
    
    df = pd.read_csv(topics_file)
    print(f"   ✓ Loaded {len(df):,} tweets")
    
    # 2) Parse created_at as datetime
    print("\n2) Parsing created_at column...")
    df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
    
    # Remove rows with invalid dates
    df = df[df['created_at'].notna()].copy()
    print(f"   ✓ {len(df):,} tweets with valid dates")
    
    # 3) Filter to 2022-2026 only
    print("\n3) Filtering to years 2022-2026...")
    df['year'] = df['created_at'].dt.year
    df_filtered = df[(df['year'] >= 2022) & (df['year'] <= 2026)].copy()
    print(f"   ✓ {len(df_filtered):,} tweets in 2022-2026")
    
    # 4) Exclude topic_id == -1 (outliers)
    print("\n4) Excluding outlier topics (topic_id == -1)...")
    df_filtered = df_filtered[df_filtered['topic_id'] != -1].copy()
    print(f"   ✓ {len(df_filtered):,} tweets with valid topics")
    
    # 5) Identify Top 5 topics by count
    print("\n5) Identifying Top 5 topics...")
    topic_counts = df_filtered['topic_id'].value_counts()
    top_5_topics = topic_counts.head(5).index.tolist()
    
    print(f"   Top 5 topics:")
    for i, topic_id in enumerate(top_5_topics, 1):
        count = topic_counts[topic_id]
        pct = (count / len(df_filtered)) * 100
        print(f"      {i}. Topic {topic_id}: {count:,} tweets ({pct:.1f}%)")
    
    # 6) Build monthly time series
    print("\n6) Building monthly time series...")
    df_filtered['year_month'] = df_filtered['created_at'].dt.to_period('M')
    
    # Create pivot table for Top 5 topics
    monthly_data = []
    for ym, group in df_filtered.groupby('year_month'):
        row = {'year_month': ym}
        for topic_id in top_5_topics:
            count = len(group[group['topic_id'] == topic_id])
            row[f'topic_{topic_id}'] = count
        monthly_data.append(row)
    
    monthly_df = pd.DataFrame(monthly_data)
    monthly_df = monthly_df.sort_values('year_month')
    
    print(f"   ✓ Created time series with {len(monthly_df)} months")
    
    # Load topic names
    print("\n7) Loading topic names...")
    topic_names = load_topic_names()
    
    # Prepare labels
    topic_labels = {}
    for topic_id in top_5_topics:
        if topic_id in topic_names:
            topic_labels[topic_id] = topic_names[topic_id]
        else:
            topic_labels[topic_id] = f"Topic {topic_id}"
    
    # Prepare data for plotting
    x_dates = [p.to_timestamp() for p in monthly_df['year_month']]
    
    # Calculate percentages per month
    topic_cols = [f'topic_{tid}' for tid in top_5_topics]
    monthly_totals = monthly_df[topic_cols].sum(axis=1)
    
    # Handle zero totals
    monthly_totals = monthly_totals.replace(0, 1)
    
    pct_data = {}
    for topic_id in top_5_topics:
        col = f'topic_{topic_id}'
        pct_data[topic_id] = (monthly_df[col] / monthly_totals * 100).values
    
    # Output directory
    output_dir = 'outputs/topic_modeling'
    os.makedirs(output_dir, exist_ok=True)
    
    # ============================================================
    # A) NORMALIZED STACKED AREA CHART
    # ============================================================
    print("\n8) Creating normalized stacked area chart...")
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Prepare data for stackplot
    y_data = [pct_data[tid] for tid in top_5_topics]
    labels = [topic_labels[tid] for tid in top_5_topics]
    
    # Use a nice color palette
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    ax.stackplot(x_dates, *y_data, labels=labels, colors=colors, alpha=0.8)
    
    ax.set_xlabel('Month', fontsize=12, weight='bold')
    ax.set_ylabel('% of Monthly Tweets', fontsize=12, weight='bold')
    ax.set_title('Top 5 Topics Over Time (Monthly Share) — 2022–2026',
                 fontsize=14, weight='bold', pad=20)
    
    # Legend outside plot
    ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0), fontsize=9)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim(0, 100)
    
    # Rotate x-axis labels
    plt.xticks(rotation=45, ha='right')
    
    # Add data source note
    fig.text(0.99, 0.01, 'Based on BERTopic assignments (Stage 23)',
             ha='right', va='bottom', fontsize=8, style='italic', color='gray')
    
    plt.tight_layout()
    
    file_a = f'{output_dir}/topic_trends_over_time_2022_2026_share.png'
    plt.savefig(file_a, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✓ Saved: {file_a}")
    
    # ============================================================
    # B) SMOOTHED LINE CHART
    # ============================================================
    print("\n9) Creating smoothed line chart...")
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Apply 3-month rolling mean smoothing
    for i, topic_id in enumerate(top_5_topics):
        pct_series = pd.Series(pct_data[topic_id])
        smoothed = pct_series.rolling(window=3, center=True, min_periods=1).mean()
        
        ax.plot(x_dates, smoothed.values, 
                label=topic_labels[topic_id],
                color=colors[i],
                linewidth=2.5,
                marker='o',
                markersize=4,
                alpha=0.9)
    
    ax.set_xlabel('Month', fontsize=12, weight='bold')
    ax.set_ylabel('% of Monthly Tweets (Smoothed)', fontsize=12, weight='bold')
    ax.set_title('Top 5 Topics Over Time (Smoothed) — 2022–2026',
                 fontsize=14, weight='bold', pad=20)
    
    # Legend outside plot
    ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0), fontsize=9)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Rotate x-axis labels
    plt.xticks(rotation=45, ha='right')
    
    # Add data source note
    fig.text(0.99, 0.01, 'Based on BERTopic assignments (Stage 23) | 3-month rolling average',
             ha='right', va='bottom', fontsize=8, style='italic', color='gray')
    
    plt.tight_layout()
    
    file_b = f'{output_dir}/topic_trends_over_time_2022_2026_smoothed.png'
    plt.savefig(file_b, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✓ Saved: {file_b}")
    
    # Summary
    print("\n" + "="*70)
    print("✓ IMPROVED VISUALIZATIONS COMPLETED")
    print("="*70)
    print(f"  Period: 2022-2026")
    print(f"  Tweets analyzed: {len(df_filtered):,}")
    print(f"  Months in series: {len(monthly_df)}")
    print(f"  Top 5 topics: {top_5_topics}")
    print(f"\n  Output files:")
    print(f"    1) {file_a}")
    print(f"    2) {file_b}")
    print("="*70)


if __name__ == "__main__":
    create_improved_visualizations()
