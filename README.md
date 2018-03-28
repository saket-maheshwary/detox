# Automated Detection of Toxic Comments on the Web

The term “toxic comment” indicates a rude, disrespectful or an unreasonable comment that is likely to make an individual leave a discussion. It includes identity hate, insult, hate speech, profanity, threats, and various ethnic, racial, or homophobic slurs. Each category can be considered abusive, and these categories are not mutually exclusive. My aim is to build a completely autonomous system to prevent such toxic comments from being posted on the web.

## Dependencies

* **[Scikit-learn](http://scikit-learn.org/stable/install.html)** for modelling
* **[NLTK](https://www.nltk.org/install.htmlhttps://www.nltk.org/)** for POS tagging
* **[Pattern library](https://github.com/clips/pattern)** for extracting polarity and subjectivity features
* **[Pandas](https://pandas.pydata.org/)** for data manipulation
* **[Numpy](http://www.numpy.org/)** for performing mathematical operations

## How to Run
```
python toxic_comment.py
```

## Implementation Details

This repo demonstrates the implementation of a model that is capable of detecting different types of toxicity like threats, obscenity, insults, and identity-based hate in the comments. This model was able to achieve ROC AUC score of **0.95** after 5 fold cross validation. 

### Dataset

In this repo, **data.csv** utilizes the training file obtained from [Kaggle](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data). It contains **159571** comments. The dataset contains large number of Wikipedia comments which have been labeled by human raters for toxic behavior. 
The types of toxicity are: toxic, severe_toxic, obscene, threat, insult and identity_hate.

### Feature Engineering

I have broadly categorized my feature engineering ideas into 3 groups.
* **N-gram based features :** Word n-gram (unigram, bigram and trigram), Char n-gram (upto 5 gram) and POS n-gram (unigram and bigram)
* **Features based on text statistics :** Count of sentences, Count of words, Count of punctuations, Diversity Count, Stopword count, Count of uppercase words, Average length of each word
* **Sentiment based features using Pattern library of CLiPS :** Subjectivity and Polarity

### Results

The results shown below are reported on Linear SVM classifier after 5 fold cross vaidation.

<img width="572" alt="screen shot 2018-03-27 at 5 46 57 am" src="https://user-images.githubusercontent.com/4180286/37997881-15f18aea-323a-11e8-96c6-2daf70e4fc4e.png">

## Next Steps (In Progress)

* GloVe Embeddings 
* LSTM

