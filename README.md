
# AI vs Human Text Classification

This project aims to develop a machine learning model capable of distinguishing between human-generated and AI-generated text. By leveraging various linguistic, syntactic, semantic, and additional text features, the model aims to accurately identify the authorship of given text samples. The implementation involves data preprocessing, feature extraction, model training with hyperparameter tuning, and evaluation using a voting classifier that combines multiple machine learning algorithms.

## Table of Contents

1. [Tech Used](#tech-used)
2. [Features](#features)
3. [FAQ](#faq)
4. [Acknowledgements](#acknowledgements)
5. [Authors](#authors)





## Tech Used

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)

![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)

![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)

![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)

![SciPy](https://img.shields.io/badge/SciPy-%230C55A5.svg?style=for-the-badge&logo=scipy&logoColor=%white)

![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)

![NLTK](https://img.shields.io/badge/NLTK-red?style=for-the-badge)

![SpaCy](https://img.shields.io/badge/SpaCy-09A3D5?style=for-the-badge)

![Gensim](https://img.shields.io/badge/Gensim-violet?style=for-the-badge)

![Textstat](https://img.shields.io/badge/Textstat-orange?style=for-the-badge)

![VADER Sentiment Analysis](https://img.shields.io/badge/VADER%20Sentiment%20Analysis-green?style=for-the-badge)






## Features

- Data Preparation:

  - Loading text data from CSV files.
  - Balancing the dataset by sampling equal amounts of human and AI-generated text.
  - Text Preprocessing:

- Converting text to lowercase.
  - Removing non-alphanumeric characters.
  - Tokenizing text.
  - Filtering out stopwords and punctuation.

- Linguistic Feature Extraction:

  - Word count and unique word count.
  - Type-Token Ratio (TTR).
  - Counts of nouns, verbs, and adjectives.
  - Sentence length statistics (average, maximum, minimum).
  - Passive voice detection.
  - Entity and conjunction counts.
  - Contraction and punctuation counts.

- Additional Feature Extraction:

  - Perplexity calculation using a bi-gram language model.
  - Readability scores (Flesch Reading Ease, SMOG Index).
  - Sentiment analysis using VADER.
  - Named entity recognition for persons, organizations, and locations.
  - Topic modeling using LDA.
  - N-gram frequency distributions (bigrams and trigrams).

- Feature Vectorization:

  - Transforming features into numerical form using TF-IDF vectorizers.
  - Combining linguistic and additional feature vectors into a single feature matrix.

- Model Training and Evaluation:

  - Splitting data into training and test sets.
  - Using a voting classifier that includes Random Forest, Logistic Regression, and Support Vector Machine classifiers.
  - Hyperparameter tuning with GridSearchCV.
  - Evaluating model performance using accuracy, precision, recall, and F1-score metrics.

- Prediction Function:

  - A function to predict the authorship of new input text based on extracted features and the trained model.
## FAQ
#### Q1: What is the purpose of this project?
A1: The project aims to create a machine learning model that can distinguish between human-generated and AI-generated text based on various linguistic and additional features extracted from the text.

#### Q2: What data is used in this project?
A2: The project uses a dataset containing samples of human-generated and AI-generated text. The data is balanced to ensure equal representation of both classes during training.

#### Q3: What features are extracted from the text?
A3: The features include lexical, syntactic, discourse, rhetorical, stylistic, perplexity, readability scores, sentiment analysis, named entity recognition, topic modeling, and n-gram frequency distributions.

#### Q4: How is the model trained?
A4: The model is trained using a voting classifier that combines Random Forest, Logistic Regression, and Support Vector Machine classifiers. Hyperparameter tuning is performed using GridSearchCV to find the best model parameters.

#### Q5: How is the performance of the model evaluated?
A5: The model's performance is evaluated using accuracy, precision, recall, and F1-score metrics on a test set.

#### Q6: Can the model predict the authorship of new text samples?
A6: Yes, the model includes a function that extracts features from new text samples, vectorizes them, and predicts whether the text is human-generated or AI-generated.
## Acknowledgements

 - [AI Vs Human Text Dataset](https://www.kaggle.com/datasets/shanegerami/ai-vs-human-text)


## Authors

- [@superiorsd10](https://www.github.com/superiorsd10)

