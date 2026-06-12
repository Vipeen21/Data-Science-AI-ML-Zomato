# Zomato Sentiment Analysis Capstone Project

## Project Overview
This repository contains a comprehensive, end-to-end Machine Learning capstone project based on Zomato restaurant reviews. The objective is to perform **Sentiment Analysis** to predict whether a customer's review is "Positive" or "Negative" using Natural Language Processing (NLP) techniques and Machine Learning classification algorithms.

The core analysis is documented and executed in `Sample_ML_Submission_Template_Completed.ipynb`.

## Datasets
The project utilizes two main datasets:
1. **`Zomato Restaurant names and Metadata.csv`**: Contains information about the restaurants including Name, Cost, Cuisines, Collections, and Timings.
2. **`Zomato Restaurant reviews.csv`**: Contains 10,000 customer reviews including the text of the Review, the Rating given (1-5), and information on uploaded pictures.

## Workflow & Features
The notebook is structured into the following key phases:

1. **Data Wrangling & Cleaning**
   - Merged metadata and reviews datasets based on restaurant name.
   - Handled missing values (dropped missing text reviews and ratings).
   - Cleaned string columns (like `Cost`).
   - Converted the numeric `Rating` (1 to 5) into a binary `Sentiment` target (Positive for $\geq$ 3.5, Negative for $<$ 3.5).

2. **Exploratory Data Analysis (EDA)**
   - Generated 15 meaningful charts (Univariate, Bivariate, Multivariate).
   - Explored relationships like Cost vs. Rating, Review Length vs. Sentiment, and the distribution of reviews across popular cuisines.
   - Answered critical business impact questions for every visualization.

3. **Hypothesis Testing**
   - Conducted rigorous statistical tests using `scipy.stats` (T-Tests and Chi-Square tests) to validate hypotheses regarding cost, review length, and the impact of user-uploaded pictures on ratings.

4. **NLP Feature Engineering**
   - Implemented an NLP pipeline to clean text data.
   - Techniques applied: Lowercasing, punctuation removal, regex cleaning (URLs/digits), stopword removal, and Lemmatization.
   - Extracted features using `TfidfVectorizer` (TF-IDF) capped at 5000 unigram/bigram features.

5. **Machine Learning Models**
   - Trained, tuned, and evaluated three distinct classification models:
     - **Logistic Regression** (Baseline & Interpretable)
     - **Random Forest Classifier** (Robust Tree Ensemble)
     - **XGBoost Classifier** (High-performance Boosting)
   - Used `GridSearchCV` for hyperparameter tuning.
   - Evaluated using Accuracy, Classification Reports (Precision, Recall, F1-Score), and Confusion Matrices.

6. **Model Deployment**
   - The best performing model pipeline (Estimator and TF-IDF Vectorizer) is serialized and exported using `joblib` for immediate deployment into a live server environment.

## Dependencies
To run the notebook successfully, you will need the following Python libraries installed:
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `xgboost`
- `nltk`
- `joblib`

*Note: The NLTK corpora (`stopwords` and `wordnet`) are downloaded automatically within the notebook.*

## How to Use
1. Clone or download the repository to your local machine.
2. Ensure you have Jupyter Notebook or Jupyter Lab installed.
3. Open `Sample_ML_Submission_Template_Completed.ipynb`.
4. Run the cells sequentially from top to bottom.

---
*This project was completed as part of the Innovexis Data Science & AI/ML Internship.*
