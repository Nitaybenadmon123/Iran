# AI Agent Instructions: Iran Twitter Data Analysis Project

## 🎯 Project Overview
End-to-end academic data science pipeline analyzing Iranian target populations on X (Twitter) through manual labeling + active learning iterations, culminating in sentiment/emotion/topic modeling analysis.

## 🏗️ Architecture & Data Flow

**Core Pipeline (8 Stages):**
1. **POI Collection** → Wikipedia categories → enriched CSVs per category (stored in `POIs/<category_slug>/`)
2. **POI→Twitter Mapping** → Manual mapping to X usernames  
3. **Candidate Pool** → Scrape followers/following of POIs via Selenium (`POIs/tools/twitter_selenium.py`)
4. **Manual Labeling (Iteration 1)** → High-confidence labels for 3 classification tasks
5. **Model Benchmarking (Stage 14)** → Grid experiment with TF-IDF + numeric features, multiple algorithms
6. **Active Learning (Iterations 2+)** → Predict uncertain users → label → retrain loop
7. **Timeline Collection** → Scrape final target users' tweets
8. **NLP Analytics** → Sentiment, Emotion detection, BERTopic topic modeling

**Directory Structure:**
- `POIs/<category>/` → Wikipedia-derived CSVs per category (e.g., `iranian_physicians/`, `govt_ministers/`)
- `Candidates/` → Candidate pool data + collection logs
- `Classification/` → Labeled datasets per iteration + model experiments
- `Data/Users_Timelines/` → Tweet collections for final analysis
- `stage_13/` → Classification decision trees/flowcharts as visual guides

## 📋 Three Core Classification Tasks

All follow **decision tree logic** in `stage_13/`:

### 1. **target_population** (`stage_13/Target Population.md`)
- **Classes:** `target` | `non_target` | `unknown`
- **Decision Logic:** Location/bio profile → Iranian identity → linguistic evidence in tweets
- **Key:** Label only when 100% confident (prefer `unknown` over guessing)

### 2. **locals_vs_diaspora** (`stage_13/locals_vs_diaspora.md`)
- **Classes:** `local` | `diaspora` | `unknown`
- **Rule:** Only applies if `target_population == 'target'`; skip otherwise
- **Decision Logic:** Residence location → bio context → tweet content analysis
- **Key:** Diaspora = outside Iran; Local = inside Iran

### 3. **person_vs_organization** (`stage_13/person_vs_organization.md`)
- **Classes:** `person` | `organization` | `unknown`
- **Decision Logic:** Username keywords → profile image type → bio description → tweet patterns
- **Key:** Identifies individual vs. institutional accounts

## 🔧 Technical Patterns & Dependencies

**Twitter Scraping (Selenium):**
- Uses headless Chrome/Firefox via `twitter_selenium.py`
- Extracts: name, bio, location, URL, joined_date, followers, following, profile_image
- **Critical:** Handles dynamic content loading with scrolling + multi-attempt JS extraction
- Handles rate limiting via cookie rotation (`POIs/x_cookies.json`)
- Logs collection progress in `POIs/Candidates/collection_log.txt`

**Data Format:**
- CSVs with columns: `username`, `name`, `bio`, `location`, `followers`, `following`, `profile_image_url`
- Classification adds: `target_population`, `locals_vs_diaspora`, `person_vs_organization` columns
- Missing values/confidence represented as `null` or `unknown`

**Model Experiments (Stage 14 Pattern):**
- **Feature Engineering:** TF-IDF on bio/name + numerical features (followers, following count)
- **Validation:** K-Fold CV (k=5) + LOOCV variants
- **Grid:** {2-class vs 3-class} × {balanced vs. imbalanced} × {6 algorithms}
  - Algorithms: Logistic Regression, Decision Tree, Random Forest, SVM, XGBoost, AdaBoost
- **Output:** `Classification/experiments_results.csv` with performance metrics

**NLP Tools:**
- BERTopic for topic modeling
- Sentiment/emotion detection (specific library TBD in notebook)

## 📝 Key Files to Reference

- [README.md](../README.md) → Complete methodology + setup instructions
- [Iranian_Users_Data_Analysis.ipynb](../Iranian_Users_Data_Analysis.ipynb) → Main workflow notebook (17K lines)
- [stage_13/Target Population.md](../stage_13/Target Population.md) → Decision tree for target classification
- [stage_13/locals_vs_diaspora.md](../stage_13/locals_vs_diaspora.md) → Residency classification logic
- [stage_13/person_vs_organization.md](../stage_13/person_vs_organization.md) → Entity type classification logic
- [POIs/tools/twitter_selenium.py](../POIs/tools/twitter_selenium.py) → Twitter scraper implementation

## 🚀 Common Workflows

**When Labeling Data:**
- Follow decision tree in `stage_13/` strictly; only label if 100% confident
- Default to `unknown` on edge cases (avoid false positives)
- Check profile location → bio → tweet content in that order

**When Debugging Collection Logs:**
- Check `POIs/Candidates/collection_log.txt` for rate-limit blocks + retry timing
- "Insufficient data" = missing 2+ of {followers, following, bio}; skip these users
- "Block confirmed" = pause needed; resume with fresh cookies

**When Exploring Classification Data:**
- Iteration CSVs in `Classification/iteration_*.csv` contain cumulative labeled data
- Compare iteration results to track labeling progress + active learning uncertainty sampling
- Use `experiments_results.csv` to identify best-performing feature/algorithm combinations

## ⚠️ Non-Obvious Project Conventions

1. **High-Precision Labeling Over Coverage:** Prefer fewer confident labels → `unknown` over rushing through uncertain cases
2. **Decision Tree Order Matters:** Classification tasks follow a strict flowchart; deviation risks label inconsistency
3. **Three-Class vs Two-Class Variants:** Models trained on both; don't conflate results
4. **Active Learning Uncertainty:** "Most uncertain" means highest entropy/lowest confidence from classifier, not randomness
5. **Cookie Rotation for Rate Limiting:** Twitter blocking handled via `x_cookies.json` rotation; respect delays in logs
6. **POI Enrichment Pipeline:** Each POI category folder contains multiple CSV versions (base Wikipedia → enriched with Wikidata → Twitter mapped)

## 💡 When Extending the Project

- **New Category:** Create folder `POIs/<new_category>/` with `<category>_wikipedia.csv` naming convention
- **New Classification Task:** Document decision tree as Mermaid flowchart in `stage_13/` with same structure
- **Model Tuning:** Add experiments to `Classification/experiments_results.csv` with consistent column naming
- **NLP Additions:** Integrate into notebook with checkpoint save pattern (collect → process → export results)
