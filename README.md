# 🎮 Steam Review Sentiment Analysis

Analyze what players *really* think about a game — automatically, at scale — using NLP and machine learning on real Steam reviews.

## Overview

This project scrapes thousands of user reviews from the [Steam Web API](https://store.steampowered.com/appreviews/), preprocesses the text, labels each review as **positive** or **negative** using a lexicon-based approach, and trains classification models to predict sentiment.

The analysis was performed on **No Man's Sky** (App ID: 275850), collecting **10,000 English reviews**.

---

## 🔍 What This Project Does

| Step | Description |
|------|-------------|
| **Data Collection** | Scrapes up to 10,000 reviews via the Steam API using cursor-based pagination |
| **Preprocessing** | Cleans text (removes mentions, URLs, numbers, punctuation), normalizes slang words, tokenizes, removes stopwords, and applies stemming |
| **Labeling** | Assigns a polarity score using an English sentiment lexicon (positive/negative word lists), then classifies each review |
| **Visualization** | Generates pie charts, class distribution plots, text length histograms, and word clouds for overall, positive, and negative reviews |
| **Feature Extraction** | Transforms text using **TF-IDF** (top 200 features) and **Word2Vec** (100-dimensional embeddings) |
| **Classification** | Trains and evaluates Naive Bayes and Random Forest classifiers across multiple schemes |

---

## 📊 Results

| Scheme | Model | Features | Train Accuracy | Test Accuracy |
|--------|-------|----------|---------------|--------------|
| #1 | Naive Bayes | TF-IDF | 87.1% | **86.7%** |
| #2 | Random Forest | TF-IDF | 99.2% | **98.6%** |
| #3 | Random Forest | Word2Vec | 100% | **98.6%** |

> **Random Forest** consistently outperforms Naive Bayes, achieving ~98.6% test accuracy on both TF-IDF and Word2Vec features.

### Sentiment Distribution in the Dataset

- ✅ **Positive**: 9,835 reviews (98.4%)
- ❌ **Negative**: 165 reviews (1.6%)

---

## 🛠️ Tech Stack

- **Python** — pandas, NumPy, scikit-learn, NLTK, Gensim, Matplotlib, Seaborn, WordCloud
- **Steam Web API** — for live review scraping
- **Jupyter Notebook** — for interactive exploration and visualization

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install requests pandas nltk scikit-learn gensim matplotlib seaborn wordcloud
```

### Run the Notebook

1. Clone the repository
2. Open `Steam_Review_Sentiment_Analysis_Aditya_Nur_Huda.ipynb` in Jupyter
3. Set the `game_id` variable to any Steam App ID
4. Run all cells

```python
game_id = 275850  # Change this to analyze any game on Steam
df = get_steam_reviews(game_id, num_reviews=10000)
```

---

## 📁 Project Structure

```
.
├── Steam_Review_Sentiment_Analysis_Aditya_Nur_Huda.ipynb  # Main notebook
├── english-positive.csv                                    # Positive sentiment lexicon
├── english-negative.csv                                    # Negative sentiment lexicon
└── README.md
```

---

## 💡 Key Insights

- The overwhelming majority of No Man's Sky reviews are positive, reflecting its successful comeback after a rocky launch.
- **Random Forest + TF-IDF** provides the best balance of performance and interpretability.
- Slang normalization is a critical preprocessing step for gaming communities, where informal language is ubiquitous.

---

## 👤 Author

**Aditya Nur Huda**
