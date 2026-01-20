# CLAUDE.md - AI Assistant Guide for Legal Text Retrieval COLIEE Project

## Project Overview

This repository implements a **Legal Information Retrieval System** for the COLIEE (Competition on Legal Information Extraction/Entailment) Task 3. The goal is to retrieve relevant Japanese Civil Code articles (in English translation) given legal queries.

**Key Facts:**
- **Domain:** Legal information retrieval
- **Dataset:** COLIEE 2020 - English translations of Japanese Civil Code
- **Task:** Multi-document retrieval (1-6 relevant articles per query)
- **Scale:** ~700 queries, 782 civil code articles
- **Primary Language:** Python 3.7+
- **Development Environment:** Jupyter Notebooks (Google Colab-based)

## Codebase Structure

```
/
├── README.md                              # Project overview
├── Information_Retrieval_Project__COLIEE_Task_3_.pdf  # Detailed research report
├── data/                                  # Dataset storage
│   └── readme.md                          # Dataset documentation
├── embeddings/                            # Word embeddings (law2vec, GloVe)
│   └── readme.md                          # Embeddings documentation
├── WMD/                                   # Legacy WMD implementation location
│   └── wmd.py                             # Standalone WMD script
└── src/                                   # Main source code
    ├── data_analaysis/                    # Data preprocessing and EDA
    │   ├── parse_statute_law.ipynb        # XML parsing for civil code
    │   └── data_analysis.ipynb            # Exploratory data analysis
    ├── classical_approach/                # Traditional IR methods
    │   └── classical_approach.ipynb       # TF-IDF and BM25 implementations
    └── modern_approach/                   # Deep learning approaches
        ├── DistilBERT/                    # Transformer-based classification
        │   ├── COLIEE_transformer.ipynb
        │   ├── create_classification_Dataset.ipynb
        │   ├── create_downsample_classification.ipynb
        │   └── readme.md
        └── WMD/                           # Word Mover's Distance retrieval
            └── wmd.py
```

### Directory Purposes

- **`/src/data_analaysis/`**: Data parsing, cleaning, tokenization, lemmatization, and EDA
- **`/src/classical_approach/`**: Baseline IR methods (TF-IDF, BM25)
- **`/src/modern_approach/DistilBERT/`**: Binary relevance classification using transformers
- **`/src/modern_approach/WMD/`**: Semantic similarity using word embeddings
- **`/data/`**: XML statute law files and query-article pairs (H18-H29 datasets)
- **`/embeddings/`**: Pre-trained embeddings (law2vec.200d, GoogleNews-300d)

## Key Technologies and Dependencies

### Core Libraries

```python
# NLP Processing
nltk                    # Tokenization, lemmatization, stopword removal
spacy                   # Advanced NLP preprocessing
gensim                  # Word2Vec, TF-IDF, similarity calculations
word_mover_distance     # WMD computation

# Machine Learning
transformers            # HuggingFace DistilBERT models
rank_bm25              # BM25 ranking function
scikit-learn           # ML utilities and metrics

# Data Processing
pandas                  # DataFrame manipulation
numpy                   # Numerical operations
pickle                  # Serialization

# Utilities
xml.etree.ElementTree  # XML parsing
matplotlib             # Visualization
```

### Development Environment

- **Platform:** Google Colab (cloud-based Jupyter notebooks)
- **Storage:** Google Drive integration for data/models
- **Execution:** Interactive notebook cells
- **No traditional build system** (no requirements.txt, setup.py, or pyproject.toml)

## Data Organization

### Data Files

The project uses several preprocessed pickle files:

1. **`cleaned_ground_truth.pkl`** - Query dataset
   - Columns: `ID`, `Query`, `Query_tokens`, `Query_lemma`, `Article_numbers`
   - Contains 695-716 queries with ground truth relevance labels

2. **`cleaned_extended_ground_truth.pkl`** - Query expansion dataset
   - Columns: `Expanded_query_tokens`, `Expanded_query_lemma`
   - Queries enhanced with synonyms for improved recall

3. **`cleaned_civil_code.pkl`** - Civil code articles
   - Columns: `Article_number`, `Article_description`, `Article_description_tokens`, `Article_description_lemmas`
   - Contains 776-782 articles from Japanese Civil Code

### Data Statistics

- **Queries:** 695-716 total
- **Articles:** 776-782 civil code articles
- **Relevant articles per query:** 1-6 (median: 1-2)
- **Ground truth pairs:** 901 total query-article relevance relations
- **Class imbalance:** 538/695 queries have only 1 relevant article

### Data Processing Pipeline

1. **Input:** XML statute law files (`statute_law.xml`, `riteval_H*.xml`)
2. **Parsing:** Extract article numbers, titles, and content
3. **Tokenization:** `word_tokenize` or `RegexpTokenizer`
4. **Cleaning:** Remove stopwords (NLTK English stopwords)
5. **Normalization:** Lemmatization (`WordNetLemmatizer`) or stemming (`PorterStemmer`)
6. **Feature Engineering:** N-grams (bigrams, trigrams), query expansion
7. **Serialization:** Save as pickle files for reuse

## Code Conventions

### Naming Conventions

- **Variables:** `snake_case` (e.g., `dataset_query_lemmas`, `ground_truth`)
- **Functions:** `snake_case` (e.g., `wmd_retrieval`, `law2vec_to_word2vec_format`)
- **Files:** `snake_case.py` or `descriptive_name.ipynb`
- **Constants:** `UPPER_CASE` (e.g., `REAL`)

### Common Patterns

1. **Article Number Normalization:**
   ```python
   # Hyphens: Convert underscores to hyphens
   article_number = re.sub("_", "-", article_number)
   # Spaces: Remove spaces from article numbers
   article_number = re.sub(" ", "", article_number)
   ```

2. **Ground Truth Parsing:**
   ```python
   ground_truth = []
   for line in range(len(df_query_list)):
       l = ast.literal_eval(df_query_list["Article_numbers"].iloc[line])
       ground_truth.append(l)
   ```

3. **TREC Output Format:**
   ```python
   # Format: query_id Q0 article_id rank score run_id
   print(f"{query_id} Q0 {article_id} {rank} {score} OVGU")
   ```

### Evaluation Metrics

All approaches use the same evaluation function:

```python
precision = true_positive / (true_positive + false_positive)
recall = true_positive / (true_positive + false_negative)
f2_score = (5 * precision * recall) / ((4 * precision) + recall)
```

**Note:** F2-score is used (emphasizes recall over precision), appropriate for legal retrieval where missing relevant articles is worse than retrieving some irrelevant ones.

## Working with the Codebase

### Prerequisites

1. **Google Colab Environment** or local Jupyter installation
2. **Google Drive** mounted (for data access in notebooks)
3. **Pre-downloaded data files:** Pickle files, embeddings, XML datasets

### Running Experiments

#### 1. Data Preprocessing (First Time Setup)

```bash
# Run notebooks in order:
1. src/data_analaysis/parse_statute_law.ipynb    # Parse XML to DataFrames
2. src/data_analaysis/data_analysis.ipynb        # Create cleaned pickle files
```

#### 2. Classical Approaches

```bash
# TF-IDF and BM25 retrieval
src/classical_approach/classical_approach.ipynb
```

**Key Parameters:**
- `top_n`: Number of top-ranked articles to retrieve (typically 2-5)
- `threshold`: Similarity threshold for multi-document retrieval (0.2-0.4)
- `features`: `lemmas`, `tokens`, `bigrams`, `trigrams`, or combinations

**Best Results:**
- **TF-IDF:** top-n=2, threshold=0.33 → F2=0.423
- **BM25:** rank=3, threshold=7 → F2=0.459

#### 3. Modern Approaches - Word Mover's Distance

```bash
# Run WMD retrieval
cd src/modern_approach/WMD/
python wmd.py
```

**Interactive Prompts:**
1. Embedding model: `law2vec` or `glove`
2. Number of articles to retrieve: `2-5`
3. Similarity threshold: `0.5-0.8`
4. Feature type: `1` (query lemmas) or `2` (query expansion)

**Important:** Update hardcoded file paths in `wmd.py`:
- Line 208: Path to `Law2Vec.200d.txt`
- Line 227: Path to `GoogleNews-vectors-negative300.bin.gz`

#### 4. Modern Approaches - DistilBERT

```bash
# Dataset creation
src/modern_approach/DistilBERT/create_classification_Dataset.ipynb
src/modern_approach/DistilBERT/create_downsample_classification.ipynb

# Model training and evaluation
src/modern_approach/DistilBERT/COLIEE_transformer.ipynb
```

**Note:** Trained models are stored externally on Google Drive (links in readme.md)

### Common Tasks

#### Adding a New Retrieval Method

1. Create a new notebook in appropriate directory (`classical_approach` or `modern_approach`)
2. Load preprocessed data:
   ```python
   df_query = pd.read_pickle("cleaned_ground_truth.pkl")
   df_articles = pd.read_pickle("cleaned_civil_code.pkl")
   ```
3. Implement retrieval logic following the pattern:
   ```python
   def new_retrieval_method(query, articles, article_numbers, top_n, threshold, ground_truth):
       # Compute similarities
       # Return: total_retrieved, true_positive, total_relevant
   ```
4. Use the standard evaluation function (see conventions above)
5. Output results in TREC format

#### Modifying Data Preprocessing

1. Edit `src/data_analaysis/parse_statute_law.ipynb` for XML parsing changes
2. Edit `src/data_analaysis/data_analysis.ipynb` for tokenization/lemmatization changes
3. Re-run notebooks to regenerate pickle files
4. Ensure downstream notebooks reference correct pickle files

#### Experimenting with Features

Common feature variants to try:
- **Tokens vs. Lemmas:** Compare `Query_tokens` vs. `Query_lemma`
- **Query Expansion:** Use `Expanded_query_lemmas` from extended ground truth
- **N-grams:** Add bigrams/trigrams to features
- **Combinations:** `lemmas + bigrams`, `tokens + trigrams`

## Retrieval Methods Comparison

| Method | Approach | Best F2-Score | Precision | Recall |
|--------|----------|---------------|-----------|--------|
| **TF-IDF** | Sparse matrix similarity (Gensim) | 0.423 | 0.349 | 0.446 |
| **BM25** | Okapi BM25 ranking | 0.459 | 0.335 | 0.506 |
| **WMD (law2vec)** | Word embeddings distance | ~0.4-0.5 | - | - |
| **WMD (GloVe)** | Word embeddings distance | ~0.4-0.5 | - | - |
| **DistilBERT** | Binary relevance classification | Variable | - | - |

## Important Notes for AI Assistants

### DO's

1. **Always read existing code before modifying** - Never propose changes to code you haven't examined
2. **Respect the experimental nature** - This is a research project with multiple approaches to compare
3. **Maintain evaluation consistency** - Use the same metrics (F2-score) across all methods
4. **Preserve data preprocessing** - Don't modify pickle file formats without updating all dependent notebooks
5. **Document hyperparameters** - Note top-n, thresholds, and feature choices in experiments
6. **Use TREC format for output** - Standard format: `query_id Q0 article_id rank score run_id`
7. **Handle article number formatting** - Apply hyphen/space normalization consistently

### DON'Ts

1. **Don't add production-grade features** - This is academic research code (avoid over-engineering)
2. **Don't assume file paths** - Update hardcoded Google Drive paths for local execution
3. **Don't modify ground truth** - The 901 query-article pairs are fixed competition data
4. **Don't change evaluation metrics** - F2-score is the standard for COLIEE Task 3
5. **Don't add type hints/extensive docstrings** - Match the existing informal notebook style
6. **Don't create new data files** - Work with existing pickle files unless preprocessing changes are required

### Key Considerations

1. **Google Colab Dependencies:**
   - Code expects Google Drive mounting: `from google.colab import drive; drive.mount('/content/drive')`
   - File paths reference Google Drive structure
   - For local execution, update all file paths

2. **Memory Management:**
   - Word embedding models are large (200-300MB)
   - Notebooks may require GPU/high RAM in Colab
   - Consider releasing embeddings when not needed: `del word_vectors`

3. **Data Sensitivity:**
   - Ground truth is competition data - treat as read-only
   - Pickle files contain preprocessed data - regenerate if source XML changes
   - Article numbers must match competition format exactly

4. **Reproducibility:**
   - Random seeds not set in all notebooks
   - Transformer models may vary between runs
   - Classical methods (TF-IDF, BM25) are deterministic

5. **Legal Domain:**
   - Use law2vec for domain-specific embeddings when possible
   - Legal text has specialized vocabulary and structure
   - Article numbers follow Japanese Civil Code format (e.g., "Article_3-1")

## Testing and Validation

### Ground Truth Validation

Always verify retrieval results against ground truth:

```python
if str(retrieved_article_number) in ground_truth[query_idx]:
    true_positive += 1
```

### Cross-Validation

- **Train/Test Split:** Use different years of COLIEE data (H18-H29)
- **Separate pickle files** for training and testing datasets
- **No data leakage:** Ensure test queries never seen during development

### Evaluation Checklist

Before finalizing any retrieval method:
- [ ] Calculate Precision, Recall, F2-score
- [ ] Compare against baseline (BM25 F2=0.459)
- [ ] Verify output format matches TREC standard
- [ ] Check article number formatting is consistent
- [ ] Validate against ground truth counts
- [ ] Document hyperparameters used

## Git Workflow

### Branch Strategy

- **Development branch:** `claude/add-claude-documentation-WyFiy`
- **Never push to main/master** without explicit permission
- **Create feature branches** for significant changes

### Commit Guidelines

1. Use descriptive commit messages
2. Commit after completing logical units of work (e.g., adding new retrieval method)
3. Don't commit large binary files (embeddings, models) - reference external storage instead
4. Include notebook output cells if they contain important results

### Push Requirements

```bash
git push -u origin claude/add-claude-documentation-WyFiy
```

## External Resources

- **Paper/Report:** `Information_Retrieval_Project__COLIEE_Task_3_.pdf`
- **COLIEE Competition:** http://www.coliee.org/
- **Law2Vec Embeddings:** Custom legal domain embeddings (200d)
- **GloVe Embeddings:** GoogleNews-vectors-negative300 (300d)

## Quick Reference

### File Locations

```python
# Preprocessed data
"cleaned_ground_truth.pkl"           # Queries with ground truth
"cleaned_extended_ground_truth.pkl"  # Expanded queries
"cleaned_civil_code.pkl"             # Civil code articles

# Embeddings
"Law2Vec.200d.txt"                   # Legal domain embeddings
"GoogleNews-vectors-negative300.bin.gz"  # General embeddings

# Raw data
"statute_law.xml"                    # Full civil code
"riteval_H*.xml"                     # Query-article pairs by year
```

### Common Code Snippets

**Load Data:**
```python
df_query = pd.read_pickle("cleaned_ground_truth.pkl")
df_articles = pd.read_pickle("cleaned_civil_code.pkl")
```

**Parse Ground Truth:**
```python
ground_truth = []
for line in range(len(df_query)):
    articles = ast.literal_eval(df_query["Article_numbers"].iloc[line])
    ground_truth.append(articles)
```

**Normalize Article Numbers:**
```python
article_number = re.sub("_", "-", article_number)
article_number = re.sub(" ", "", article_number)
```

**Evaluation:**
```python
precision = tp / (tp + fp)
recall = tp / (tp + fn)
f2_score = (5 * precision * recall) / ((4 * precision) + recall)
```

## Questions or Issues?

- **Code questions:** Review the research paper PDF for methodology details
- **Data questions:** Check readme files in `/data/` and `/embeddings/`
- **Results questions:** Compare against benchmarks in `classical_approach.ipynb`

---

**Last Updated:** 2026-01-20
**Project Status:** Research/Academic (COLIEE 2020 competition)
**Primary Contact:** Repository owner
