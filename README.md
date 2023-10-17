
# Movie Review Sentiment Analysis

This project performs binary sentiment analysis on movie reviews using a Naive Bayes classifier. The reviews are classified as either positive or negative.

## Data

The dataset used is the Large Movie Review Dataset from http://ai.stanford.edu/~amaas/data/sentiment/. It contains 25,000 highly polar movie reviews (12,500 positive and 12,500 negative).

The dataset is loaded and preprocessed by:

- Removing unwanted columns
- Cleaning special characters from text 
- Converting to lower case
- Removing stopwords
- Stemming words
Here is a Technologies section that can be added to the README:

## Technologies


- **NumPy** - For numerical processing and generating vectors/matrices
- **Pandas** - For loading, cleaning and preprocessing the textual data
- **Scikit-Learn** - For training and evaluating the Naive Bayes classifier
- **NLTK** - For text processing and cleansing like stemming, removing stopwords etc. 
- **Matplotlib** - For generating plots and visualizations 
- **Jupyter Notebooks** - As the main development environment to run experiments

The code is modularized into functions and classes to separate concerns. Key Python concepts like functions, objects and collections are leveraged.


## Models



- Naive Bayes classifier trained on bag-of-words features.

The Naive Bayes model calculates the prior and likelihood probabilities to classify reviews. 

## Usage

The main notebooks are:


- `Sentiment Analysis on Movie Reviews.ipynb`: Loads data, trains a Naive Bayes classifier and evaluates performance
- `text_classifier.py`: A simple CLI app to test sentiment prediction 


Type a movie review, it will predict whether it is positive or negative.

## Results

The Naive Bayes model achieves ~82% accuracy on the dataset. There is scope to further improve performance using regularization, better text preprocessing etc.

## References

Dataset: http://ai.stanford.edu/~amaas/data/sentiment/
