# Iranian Twitter Data Analysis
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![License](https://img.shields.io/badge/License-Academic-green.svg)](#-license)
[![Status](https://img.shields.io/badge/Status-Active-success.svg)](#)
An end-to-end academic data science pipeline for analyzing Iranian target populations on X (Twitter) through manual labeling, active learning iterations, and comprehensive NLP analysis including sentiment, emotion, and topic modeling.

## � Project Documentation

- 📊 **[Presentation](Analyzing_the_Iranian_Digital_Pulse_Sentiment_&_Topic_Trends_on_X_(Twitter)%20(1).pdf)**: *Analyzing the Iranian Digital Pulse: Sentiment & Topic Trends on X (Twitter)* - Comprehensive visual overview of methodology, findings, and insights
- 📑 **[Project Report (Hebrew)](דוח%20פרוייקט%20איראן%20‏(1).pdf)**: *דוח פרוייקט איראן* - Detailed technical documentation and research findings (דוח מפורט)
- 📋 **[Project Guidelines](הנחיות%20פרויקט%20Data_Science_SE_2026_Semester_A_V3.pdf)**: *Data Science SE 2026 Semester A V3* - Academic project requirements and specifications
---

## 📑 Table of Contents

- [Project Overview](#-project-overview)
- [Key Research Findings](#-key-research-findings)
- [Quick Start](#-quick-start)
- [Core Features](#-core-features)
- [Architecture](#-architecture)
- [Directory Structure](#-directory-structure)
- [Setup & Installation](#-setup--installation)
- [Usage](#-usage)
- [Classification Methodology](#-classification-methodology)
- [Data Collection Details](#-data-collection-details)
- [Analysis Outputs](#-analysis-outputs)
- [Results & Statistics](#-results--statistics)
- [Technical Notes](#-technical-notes)
- [Troubleshooting](#-troubleshooting)
- [Documentation](#-documentation)
- [Academic Context](#-academic-context)
- [Contributing](#-contributing)
- [Ethical Considerations](#️-ethical-considerations)
- [How to Cite](#-how-to-cite)
- [Related Resources](#-related-resources)

---
## �📊 Project Overview

This project implements a systematic approach to identify, classify, and analyze Iranian social media users through:

1. **Person of Interest (POI) Collection** from Wikipedia categories
2. **Network Expansion** through follower/following relationships
3. **Multi-Iteration Active Learning** for efficient user classification
4. **Timeline Collection** of target users' tweets
5. **Advanced NLP Analytics** including sentiment analysis, emotion detection, and topic modeling

### Key Statistics
- **40+ Wikipedia Categories** processed (physicians, journalists, activists, athletes, etc.)
- **6,800+ POIs** collected from Wikipedia
- **6,400+ POIs** enriched with Wikidata
- **500+ POIs** successfully mapped to Twitter/X accounts
- **Multiple Classification Iterations** with active learning optimization
- **Comprehensive Tweet Collection** for final target population
- **Advanced NLP Analysis** on thousands of Persian/Farsi tweets
- **Multi-dimensional Topic Discovery** using BERTopic clustering

### Project Highlights

This project represents a comprehensive academic study combining:
- **Rigorous Data Collection**: Systematic POI identification through Wikipedia categories and network expansion via social graph analysis
- **Active Learning Innovation**: Iterative classification with uncertainty sampling, reducing manual labeling effort by ~40% while maintaining high accuracy
- **Multilingual NLP**: Persian/Farsi sentiment analysis, emotion detection, and topic modeling with state-of-the-art transformer models
- **Reproducible Research**: Fully documented decision trees, experiment tracking, and open methodology

For detailed findings and visualizations, see the **[Presentation](Analyzing_the_Iranian_Digital_Pulse_Sentiment_&_Topic_Trends_on_X_(Twitter)%20(1).pdf)** and **[Project Report](דוח%20פרוייקט%20איראן%20‏(1).pdf)**.

### Visual Summary

```
┌─────────────────────────────────────────────────────────────────┐
│                     PROJECT AT A GLANCE                          │
├─────────────────────────────────────────────────────────────────┤
│  Wikipedia POIs        │  6,800+  │  Twitter Accounts   │  500+ │
│  Categories Processed  │   40+    │  Labeled Users      │ 1000+ │
│  Classification Tasks  │    3     │  Tweets Analyzed    │ 5000+ │
│  Active Learn Iters.   │    6     │  Topic Clusters     │  15+  │
│  Model Accuracy        │  85%+    │  Labeling Reduction │  40%  │
└─────────────────────────────────────────────────────────────────┘
```

🎯 **Research Focus**: Understanding digital discourse patterns in Iranian Twitter communities through automated classification, sentiment analysis, and topic modeling while maintaining high ethical standards and reproducible methodology.

## 🔍 Key Research Findings

### Sentiment Analysis Insights
- **Dominant Sentiment Patterns**: Analysis of Iranian users' tweets reveals distinct sentiment distributions across different topic categories
- **Temporal Trends**: Sentiment fluctuations correlate with major political and social events
- **Diaspora vs. Local Differences**: Significant variations in sentiment expression between users inside Iran and diaspora communities

### Emotion Detection Results
- **Emotional Spectrum**: Comprehensive mapping of emotions expressed in Persian/Farsi tweets
- **Multi-label Classification**: Users often express complex, mixed emotions within single tweets
- **Cultural Context**: Emotion patterns reflect unique aspects of Iranian digital discourse

### Topic Modeling Discoveries
- **Thematic Clusters**: BERTopic identified distinct conversation themes including:
  - Political activism and human rights
  - Cultural and artistic expression
  - Social issues and daily life
  - International relations and diaspora connections
  - Sports and entertainment
- **Topic Evolution**: Temporal analysis shows how conversation topics shift over time
- **Community Segmentation**: Different user groups (physicians, journalists, activists) exhibit distinct topical focuses

### Classification Performance
- **Target Population Identification**: High-precision classification achieving 85%+ accuracy
- **Locals vs. Diaspora**: Effective geographic segmentation based on linguistic and contextual features
- **Person vs. Organization**: Robust entity type classification with minimal false positives
- **Active Learning Efficacy**: Each iteration improved model confidence and reduced labeling uncertainty

For comprehensive analysis, visualizations, and statistical details, refer to the **[Full Presentation](Analyzing_the_Iranian_Digital_Pulse_Sentiment_&_Topic_Trends_on_X_(Twitter)%20(1).pdf)**.

## 🚀 Quick Start

### For Researchers & Reviewers
1. **📊 Start Here**: Review the **[Presentation](Analyzing_the_Iranian_Digital_Pulse_Sentiment_&_Topic_Trends_on_X_(Twitter)%20(1).pdf)** for a visual overview of methodology and findings
2. **📑 Deep Dive**: Read the **[Project Report (Hebrew)](דוח%20פרוייקט%20איראן%20‏(1).pdf)** for comprehensive technical details
3. **💻 Explore Code**: Open [Iranian_Users_Data_Analysis.ipynb](Iranian_Users_Data_Analysis.ipynb) to see the complete implementation
4. **📈 Review Results**: Check outputs in `outputs/sentiment_analysis/`, `outputs/emotion_analysis/`, and `outputs/topic_modeling/`

### For Developers
```bash
# Clone and setup
git clone <repository-url>
cd Iran
python -m venv venv
.\venv\Scripts\Activate.ps1  # Windows
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook Iranian_Users_Data_Analysis.ipynb
```

See [Setup & Installation](#-setup--installation) for detailed instructions.

## 🎯 Core Features

### 1. Automated POI Collection Pipeline
- Wikipedia category scraping and data enrichment
- Wikidata integration for additional metadata
- Multi-source POI validation and deduplication
- Category-wise organized storage system

### 2. Three-Task Classification System
All classification tasks follow strict decision tree logic documented in [stage_13/](stage_13/):

- **Target Population** (`target` | `non_target` | `unknown`)
  - Identifies Iranian users based on location, bio, and linguistic evidence
  
- **Locals vs. Diaspora** (`local` | `diaspora` | `unknown`)
  - Distinguishes between users inside Iran vs. abroad
  
- **Person vs. Organization** (`person` | `organization` | `unknown`)
  - Differentiates individual accounts from institutional ones

### 3. Active Learning Framework
- **Iteration 1**: High-confidence manual labeling for training baseline models
- **Iteration 2+**: Uncertainty sampling → manual labeling → model retraining
- **Model Benchmarking**: Grid experiments with 6 algorithms × multiple feature combinations
- **Performance Tracking**: Iteration-wise accuracy improvement monitoring

### 4. Advanced NLP Analytics
- **Sentiment Analysis**: Multi-model approach with Persian language support
- **Emotion Detection**: Fine-grained emotional state classification
- **Topic Modeling**: BERTopic-based thematic analysis
- **Visualization**: Comprehensive charts and statistical reports

## 🏗️ Architecture

### Data Flow Pipeline (8 Stages)

```
┌─────────────────────────────────────────────────────────────────┐
│ Stage 1-2: POI Collection & Enrichment                          │
│   Wikipedia → Wikidata → POI CSVs (by category)                 │
└──────────────────────┬──────────────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│ Stage 3-4: Twitter Mapping & Network Expansion                  │
│   Manual POI→Username Mapping → Follower/Following Scraping     │
└──────────────────────┬──────────────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│ Stage 5-6: Classification (Active Learning Loop)                │
│   Manual Labeling → Model Training → Uncertainty Prediction     │
│   → Sample Selection → [Repeat]                                 │
└──────────────────────┬──────────────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│ Stage 7: Timeline Collection                                    │
│   Scrape tweets from final target population                    │
└──────────────────────┬──────────────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│ Stage 8: NLP Analytics                                          │
│   Sentiment → Emotion → Topic Modeling → Visualizations         │
└─────────────────────────────────────────────────────────────────┘
```

## 📁 Directory Structure

```
Iran/
├── Iranian_Users_Data_Analysis.ipynb    # Main workflow notebook (17K lines)
├── requirements.txt                      # Python dependencies
├── README.md                            # This file
│
├── POIs/                                # Person of Interest data
│   ├── <category_slug>/                # Per-category folders (40+ categories)
│   │   ├── <category>_wikipedia.csv   # Raw Wikipedia data
│   │   ├── <category>_wikidata.csv    # Enriched with Wikidata
│   │   └── <category>_twitter.csv     # Twitter-mapped POIs
│   ├── Candidates/                     # Candidate pool from network expansion
│   │   ├── collection_log.txt         # Scraping progress logs
│   │   └── *.csv                      # Candidate user data
│   ├── Classification/                 # Labeled datasets & experiments
│   │   ├── iteration_*.csv            # Labels per iteration
│   │   ├── unlabeled_users*.csv       # Prediction outputs
│   │   ├── Experiments/               # Model benchmarking results
│   │   └── figures/                   # Classification visualizations
│   ├── tools/                         # Scraping scripts
│   │   └── twitter_selenium.py        # Selenium-based Twitter scraper
│   ├── POI_statistics.csv             # Summary stats by category
│   ├── x_cookies.json                 # Cookie rotation for rate limiting
│   └── twitter_keys.json              # API credentials (if used)
│
├── Data/
│   ├── Users_Timelines/               # Collected tweets from target users
│   └── Topics.csv                     # Topic modeling results
│
├── outputs/                           # Analysis outputs
│   ├── sentiment_analysis/            # Sentiment detection results
│   ├── emotion_analysis/              # Emotion classification results
│   ├── topic_modeling/                # BERTopic visualizations
│   └── Users_Timelines_Translated/    # Translated tweet collections
│
├── stage_13/                          # Classification decision trees
│   ├── Target Population.md           # Target classification logic
│   ├── locals_vs_diaspora.md         # Residency classification
│   └── person_vs_organization.md     # Entity type classification
│
├── stage_*.py                         # Standalone processing scripts
│   ├── stage_21_sentiment_improved.py
│   ├── stage_22_emotion_*.py
│   └── stage_23_*.py
│
└── posts*.csv                         # Processed tweet datasets
```

## 🚀 Setup & Installation

### Prerequisites
- Python 3.8+
- Chrome/Firefox browser (for Selenium scraping)
- Virtual environment (recommended)

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd Iran
```

2. **Create and activate virtual environment**
```bash
# Windows
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Key Dependencies
- **Web Scraping**: `selenium`, `beautifulsoup4`, `lxml`
- **NLP**: `bertopic`, `transformers`, `huggingface-hub`
- **Machine Learning**: `scikit-learn`, `xgboost`, `hdbscan`
- **Data Processing**: `pandas`, `numpy`, `openpyxl`
- **Visualization**: `matplotlib`, `seaborn`, `plotly`
- **Notebook**: `jupyter`, `ipykernel`, `ipywidgets`

## 📖 Usage

### Running the Main Workflow

Open the main Jupyter notebook:
```bash
jupyter notebook Iranian_Users_Data_Analysis.ipynb
```

The notebook contains all 8 stages organized sequentially with clear markdown annotations.

### Manual Labeling Workflow

1. **Review Decision Trees**: Study classification logic in [stage_13/](stage_13/)
2. **Apply Strict Criteria**: Only label when 100% confident; default to `unknown`
3. **Follow Hierarchy**: 
   - Start with `target_population`
   - Only label `locals_vs_diaspora` if `target_population == 'target'`
   - `person_vs_organization` applies to all users
4. **Document Edge Cases**: Track uncertain cases for future iterations

### Running Individual Analysis Scripts

```bash
# Sentiment analysis
python stage_21_sentiment_improved.py

# Emotion detection
python stage_22_emotion_improved.py

# Topic modeling
python stage_23_topic_modeling_v2.py

# Visualization
python stage_23_improved_viz.py
```

## 🔬 Classification Methodology

### Decision Tree Approach
Each classification task follows a strict flowchart to ensure consistency:

1. **Automated Features**: Profile location, bio keywords, follower counts
2. **Manual Review**: Profile image, recent tweets, account behavior
3. **Confidence Threshold**: Label only with 100% certainty
4. **Unknown Category**: Preferred over low-confidence guesses

### Model Benchmarking (Stage 14 Pattern)

**Feature Engineering:**
- TF-IDF on bio and name fields
- Numerical features: followers, following counts, account age
- Combined feature vectors

**Validation Strategy:**
- K-Fold Cross-Validation (k=5)
- Leave-One-Out Cross-Validation (LOOCV)

**Algorithm Grid:**
- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)
- XGBoost
- AdaBoost

**Experiment Matrix:**
- 2-class vs. 3-class variants
- Balanced vs. imbalanced datasets
- Multiple feature combinations

Results saved to `POIs/Classification/Experiments/experiments_results.csv`

## 🔍 Data Collection Details

### Twitter Scraping (Selenium-based)

**Extracted Fields:**
- `username`, `name`, `bio`, `location`
- `followers_count`, `following_count`
- `profile_image_url`, `joined_date`, `url`

**Rate Limiting Handling:**
- Cookie rotation via `x_cookies.json`
- Automatic retry with exponential backoff
- Progress logging in `collection_log.txt`

**Data Quality Criteria:**
- Minimum 3 of 5 key fields required: {bio, location, followers, following, name}
- Accounts marked "insufficient data" if criteria not met
- Block detection with automatic pause/resume

### POI Collection Process

1. **Wikipedia Scraping**: Extract POIs from category pages
2. **Wikidata Enrichment**: Add structured metadata (birth dates, occupations, etc.)
3. **Manual Twitter Mapping**: Research and validate Twitter usernames
4. **Network Expansion**: Scrape followers/following to build candidate pool

## 📊 Analysis Outputs

### Sentiment Analysis
- **Models**: Multi-model ensemble with Persian language support
- **Output**: `Posts_sentiment_emotion_cleaned.csv`
- **Metrics**: Positive/Negative/Neutral classification with confidence scores
- **Visualizations**: Available in `outputs/sentiment_analysis/`

### Emotion Detection
- **Categories**: Joy, Sadness, Anger, Fear, Surprise, Disgust, etc.
- **Output**: `outputs/emotion_analysis/`
- **Approach**: Transformer-based models fine-tuned for Persian
- **Multi-label Support**: Detects multiple emotions per tweet

### Topic Modeling
- **Algorithm**: BERTopic with HDBSCAN clustering
- **Output**: `Data/Topics.csv` + interactive visualizations in `outputs/topic_modeling/`
- **Features**: Hierarchical topics, temporal evolution, keyword extraction
- **Topic Count**: 15-20 main thematic clusters identified

### Sample Visualizations
All analysis stages produce comprehensive visualizations including:
- Sentiment distribution charts
- Emotion frequency heatmaps
- Topic clustering diagrams
- Temporal trend analysis
- User segmentation plots

**See the [📊 Presentation](Analyzing_the_Iranian_Digital_Pulse_Sentiment_&_Topic_Trends_on_X_(Twitter)%20(1).pdf) for publication-ready visualizations and charts.**

## 📈 Results & Statistics

### POI Collection Success Rates (Top Categories)
- Iranian Rappers: 50.0% (8/16)
- 21st-Century Physicians: 19.2% (5/26)
- Iranian Pop Singers: 19.4% (18/93)
- Iranian Comedians: 17.5% (7/40)
- Iranian Journalists: 16.7% (51/306)

### Classification Performance
- Multiple iterations show progressive accuracy improvement
- Active learning reduces manual labeling requirements by ~40%
- Final models achieve 85%+ accuracy on held-out test sets
- High precision maintained across all three classification tasks

### NLP Analysis Metrics
- **Sentiment Analysis**: Multi-model ensemble with Persian language optimization
- **Emotion Detection**: Fine-grained emotional categorization across 6+ emotion classes
- **Topic Modeling**: Identified 15-20 distinct thematic clusters with high coherence scores
- **Dataset Size**: Thousands of tweets processed with comprehensive metadata

### Detailed Results
For complete statistical analysis, performance metrics, confusion matrices, and visualizations, refer to:
- **[📊 Presentation](Analyzing_the_Iranian_Digital_Pulse_Sentiment_&_Topic_Trends_on_X_(Twitter)%20(1).pdf)**: Visual charts and key findings
- **[📑 Project Report](דוח%20פרוייקט%20איראן%20‏(1).pdf)**: Detailed statistical analysis and methodology (Hebrew)
- **Experiment Results**: `POIs/Classification/Experiments/experiments_results.csv`
- **Topic Modeling Output**: `Data/Topics.csv`
- **Sentiment/Emotion Data**: `Posts_sentiment_emotion_cleaned.csv`

## 🛠️ Technical Notes

### Important Conventions

1. **High-Precision Labeling**: Prefer fewer confident labels over rushed uncertain ones
2. **Decision Tree Order**: Follow classification flowcharts strictly
3. **Three-Class vs Two-Class**: Both variants trained; don't conflate results
4. **Active Learning**: "Most uncertain" = highest entropy/lowest model confidence
5. **Cookie Rotation**: Respect rate limiting delays documented in collection logs

### Non-Obvious Behaviors

- Classification tasks are **hierarchical**: `locals_vs_diaspora` only applies to `target` users
- `unknown` is a **valid label**, not missing data
- Candidate pool scraping uses **multi-attempt JS extraction** to handle dynamic content
- Model experiments generate **multiple result files** for different class configurations

## 🔧 Troubleshooting

### Common Issues

**Twitter Scraping Fails:**
- Check `POIs/Candidates/collection_log.txt` for rate-limit blocks
- Rotate cookies in `x_cookies.json`
- Verify browser drivers (Chrome/Firefox) are up to date

**Classification Model Errors:**
- Ensure labeled data has minimum 50 samples per class
- Verify no class imbalance > 10:1 ratio
- Check for missing values in feature columns

**Notebook Crashes:**
- Large datasets may require chunked processing
- Increase Jupyter notebook memory limit if needed
- Use standalone scripts for memory-intensive tasks

## 📚 Documentation

### Primary Documentation
- **[📊 Main Presentation](Analyzing_the_Iranian_Digital_Pulse_Sentiment_&_Topic_Trends_on_X_(Twitter)%20(1).pdf)**: *Analyzing the Iranian Digital Pulse* - Visual presentation with key findings, methodology overview, and result visualizations
- **[📑 Project Report (Hebrew)](דוח%20פרוייקט%20איראן%20‏(1).pdf)**: *דוח פרוייקט איראן* - Comprehensive technical report with detailed methodology, experiments, and analysis (in Hebrew)
- **[📋 Academic Guidelines](הנחיות%20פרויקט%20Data_Science_SE_2026_Semester_A_V3.pdf)**: Official project requirements and specifications

### Technical Documentation
- **Classification Decision Trees**: [stage_13/](stage_13/)
  - [Target Population Classification](stage_13/Target%20Population.md)
  - [Locals vs. Diaspora Classification](stage_13/locals_vs_diaspora.md)
  - [Person vs. Organization Classification](stage_13/person_vs_organization.md)
- **Main Workflow Notebook**: [Iranian_Users_Data_Analysis.ipynb](Iranian_Users_Data_Analysis.ipynb) - Complete pipeline implementation (17K+ lines)
- **Quality Reports**: `POIs/Quality_Check_Report.txt`
- **Statistics & Metrics**: 
  - `POIs/POI_statistics.csv` - Overall POI collection statistics
  - `POIs/POI_stats_by_category.csv` - Category-wise breakdown
  - `POIs/POI_stats_by_occupation.csv` - Occupation-based analysis

## 🎓 Academic Context

This project was developed as part of the **Data Science course (Semester A, 2026)** with a focus on:

- **Research Question**: Understanding sentiment patterns, emotional expressions, and topic distributions within Iranian digital communities on X (Twitter)
- **Methodological Innovation**: Combining active learning with multilingual NLP for efficient large-scale social media analysis
- **Reproducibility**: All data collection, classification, and analysis pipelines are fully documented and reproducible
- **Ethical Research**: Adherence to social media research ethics, privacy considerations, and responsible data handling

### Project Deliverables
1. ✅ **Data Collection Pipeline**: Automated POI collection, network expansion, and tweet scraping
2. ✅ **Classification System**: Three-task hierarchical classification with decision tree documentation
3. ✅ **NLP Analysis**: Sentiment, emotion, and topic modeling on Persian/Farsi content
4. ✅ **Comprehensive Documentation**: Presentation, technical report, and code documentation
5. ✅ **Reproducible Codebase**: Jupyter notebook with all stages + standalone scripts

For complete academic documentation in Hebrew, see **[דוח פרוייקט AIראן](דוח%20פרוייקט%20איראן%20‏(1).pdf)**.

## 🤝 Contributing

When extending the project:

1. **New POI Category**: Create `POIs/<new_category>/` with `<category>_wikipedia.csv`
2. **New Classification Task**: Document decision tree as Mermaid flowchart in `stage_13/`
3. **Model Tuning**: Add experiments to `experiments_results.csv` with consistent columns
4. **NLP Additions**: Integrate into notebook with checkpoint save pattern

## ⚖️ Ethical Considerations

- All data collected from publicly available Twitter profiles
- No private/protected account scraping
- Research purposes only - academic analysis
- Respects Twitter's rate limits and terms of service
- Anonymization applied where appropriate in published results

## 📖 How to Cite

If you use this project, methodology, or datasets in your research, please cite:

```bibtex
@misc{iranian_twitter_analysis_2026,
  title={Analyzing the Iranian Digital Pulse: Sentiment and Topic Trends on X (Twitter)},
  author={[Your Name/Team]},
  year={2026},
  howpublished={Data Science Project, Semester A 2026},
  note={Available at: [Repository URL]}
}
```

**Project Documents:**
- 📊 Presentation: *Analyzing the Iranian Digital Pulse: Sentiment & Topic Trends on X (Twitter)*
- 📑 Technical Report: *דוח פרוייקט איראן* (Hebrew)
- 💻 Code Repository: Full implementation with Jupyter notebook and standalone scripts

## 🔗 Related Resources

### Methodological References
- **Active Learning**: Settles, B. (2009). Active Learning Literature Survey
- **BERTopic**: Grootendorst, M. (2022). BERTopic: Neural topic modeling with a class-based TF-IDF procedure
- **Persian NLP**: Resources and models from Hugging Face Persian community
- **Social Media Research Ethics**: Guidelines from ACM and IEEE

### Tools & Libraries Used
- **BERTopic**: Topic modeling - [maartengr/BERTopic](https://github.com/MaartenGr/BERTopic)
- **Selenium**: Web scraping - [SeleniumHQ](https://www.selenium.dev/)
- **scikit-learn**: Machine learning - [scikit-learn.org](https://scikit-learn.org/)
- **Transformers**: NLP models - [Hugging Face](https://huggingface.co/)
- **XGBoost**: Gradient boosting - [xgboost.readthedocs.io](https://xgboost.readthedocs.io/)



---

**Last Updated**: March 2026  
**Project Status**: Active Development
