# Natural Language Processing (NLP) Assignments

This repository contains practical implementations of core Natural Language Processing pipelines and deep learning architectures, developed as part of coursework for the B.Tech CSE (AI & ML) program[cite: 1, 2].

## 📂 Repository Contents

### 1. Text Data & Basic NLP Pipeline (`nlp_pipeline.ipynb`)
* **Overview:** Covers foundational text preprocessing, data formatting, and traditional machine learning classification[cite: 1].
* **Key Tasks:**
  * Created and structured a text dataset of movie reviews with positive and negative labels saved in CSV format[cite: 1].
  * Implemented custom data preprocessing including lowercase conversion, punctuation removal, tokenization, stopword filtering using `scikit-learn`, and custom stemming[cite: 1].
  * Performed feature engineering using Bag of Words (`CountVectorizer`) and TF-IDF (`TfidfVectorizer` with n-grams)[cite: 1].
  * Trained, tested, and evaluated a Multinomial Naive Bayes classification pipeline, achieving high accuracy with confusion matrix and classification report metrics[cite: 1].

### 2. Word2Vec + LSTM & Transformer Fine-Tuning (`word2vec_lstm_distilbert.ipynb`)
* **Overview:** Explores advanced deep learning approaches for text classification using recurrent neural networks and pretrained transformers[cite: 2].
* **Key Tasks:**
  * **Word2Vec + LSTM Pipeline:** Loaded a subset of the IMDb dataset, cleaned review text, trained a custom Word2Vec model using `gensim`, built an embedding matrix, and trained a bidirectional LSTM model in PyTorch[cite: 2].
  * **DistilBERT Fine-Tuning:** Fine-tuned the pretrained `distilbert-base-uncased` transformer model using the Hugging Face `Trainer` API for binary sentiment classification[cite: 2].
  * **Evaluation & Comparison:** Evaluated both deep learning architectures using test accuracy, F1-scores, and confusion matrices to compare traditional recurrent modeling against modern transformer-based context understanding[cite: 2].

## 🛠️ Tech Stack
* **Python**[cite: 1, 2]
* **Libraries:** Pandas, NumPy, Scikit-learn, PyTorch, Gensim, Transformers (Hugging Face), Datasets, Matplotlib, Seaborn[cite: 1, 2]

---
Feel free to explore the notebooks and code implementations!
