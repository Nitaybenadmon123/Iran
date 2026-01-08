# Target Group State-of-Mind Analysis on X (Twitter) — Active Learning Pipeline

End-to-end data science project that identifies a target population on X (Twitter), builds high-quality labeled data with Active Learning, and analyzes the population’s state of mind using sentiment, emotion, and topic modeling.

## ✨ Project Highlights
- **Target population identification** from followers/following of selected POIs (Points of Interest)
- **Manual labeling + Active Learning iterations** (3 classification tasks)
- **Feature engineering**: TF-IDF text features + numerical account features
- **Model benchmarking**: K-Fold CV + LOOCV, balanced vs. imbalanced, 2 vs. 3 classes
- **Downstream NLP analytics**: Sentiment, Emotion, Topic Modeling (BERTopic)

---

## 🎯 Goal
Analyze the state of mind of a chosen **target group** based on their public discussions on X (Twitter), using a structured academic pipeline:
1) Build a reliable target-user dataset
2) Train robust classifiers with iterative labeling (Active Learning)
3) Collect user timelines
4) Extract sentiment/emotion/topics and produce insights & visualizations

---

## 🧠 Methodology (High-Level)
### 1) Select Target Population
Choose a target group (e.g., nationality/community/profession) and define the time range.

### 2) Collect POIs (Points of Interest)
Build curated POI lists from Wikipedia categories and enrich them with Wikidata fields.

### 3) Map POIs to X Usernames
Manually (and/or semi-automatically) map each POI to its official X username.

### 4) Collect POI Metadata + Connections
Fetch POI account metadata, then collect **followers/following** to build a large candidate pool.

### 5) Manual Labeling (Iteration 1)
Label a high-precision sample (only when 100% confident) for 3 tasks:
- `target_population`: target / non_target / unknown
- `locals_vs_diaspora`: local / diaspora / unknown
- `person_vs_organization`: person / organization / unknown

### 6) Benchmark Models (Stage 14)
Run a modular experiment grid:
- Feature sets: TF-IDF over different text fields + combinations with numeric features
- Validation: KFold (k=5) and LOOCV
- Variants: balanced/unbalanced, 2-class vs 3-class
- Algorithms: Logistic Regression, Decision Tree, Random Forest, SVM, XGBoost, AdaBoost

### 7) Active Learning Iterations (Iteration 2+)
Pick the best model, predict unlabeled users, sample the **most uncertain** users, label them, retrain, and repeat.

### 8) Collect Timelines + NLP Analysis
For final target users:
- Collect tweets in the selected date range
- Translate to English (if needed)
- Run sentiment, emotion detection, and topic modeling (BERTopic)
- Produce plots and a final report/presentation

---

## 🗂️ Repository Structure

ProjectRoot/

├── POIs/

│ ├── <category_1>/
│ ├── <category_2>/
│ └── ...
├── Candidates/

│ ├── POIs_candidate_connections.csv
│ ├── Candidates_user_data.csv
│ └── ...
├── Classification/

│ ├── iteration_1_.csv
│ ├── iteration_2_.csv
│ ├── iteration_3_*.csv
│ ├── experiments_results.csv
│ └── ...
├── Data/

│ ├── Users_Timelines/
│ ├── Posts/
│ └── Topics/
├── notebooks/
├── src/
└── README.md


---

## ⚙️ Setup
### Requirements
- Python 3.10+
- Recommended: create a virtual environment

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate

pip install -r requirements.txt

