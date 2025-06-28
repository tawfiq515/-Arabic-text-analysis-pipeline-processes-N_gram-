# Arabic Fake News Linguistic Analysis

![WordCloud Example](path/to/sample_wordcloud.png)

A Python pipeline to analyze linguistic patterns in Arabic fact-checked claims, comparing n-gram distributions between real and fake news using perplexity metrics and visualizations.

## Features

- **Text Preprocessing**: Normalization, stopword removal, and Arabic-specific cleaning
- **N-gram Analysis**: Extracts uni-grams to 10-grams with frequency ranking
- **Perplexity Scoring**: Quantifies language model uncertainty (lower = more predictable)
- **Visualizations**: 
  - Interactive word clouds with custom coloring
  - Horizontal bar charts for n-gram frequencies
- **Optimal Pattern Detection**: Automatically identifies the most coherent n-gram type

## Key Findings
- **Best Perplexity**: 19.74 (Real news 3-grams)  
- **Distinctive Patterns**: Fake news showed higher lexical diversity but lower coherence

## Usage
```python
# Load data
df = pd.read_csv("AraFacts.csv")

# Preprocess claims
df['cleaned_claim'] = df['claim'].apply(preprocess)

# Compare n-grams
for n in [1,2,3]:
    plot_wordcloud(get_top_ngrams(fake_texts, n), f"Fake {n}-grams")
```
## Requirements
Python 3.8+
pandas, matplotlib, arabic-reshaper, python-bidi, nltk, wordcloud

## Data

Uses AraFacts dataset with labels:

True → Real

False/Partly-false → Fake
