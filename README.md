# Automated Detection of Toxic Comments on the Web

The term “toxic comment” indicates a rude, disrespectful or an unreasonable comment that is likely to make an individual leave a discussion. It includes identity hate, insult, hate speech, profanity, threats, and various ethnic, racial, or homophobic slurs. Each category can be considered abusive, and these categories are not mutually exclusive. My aim is to build a completely autonomous system to prevent such toxic comments from being posted on the web.

## Dependencies

* **Scikit-learn** for modelling
* **NLTK** for POS tagging
* **[Pattern library](https://github.com/clips/pattern)** for Polarity and Subjectivity features


## Implementation Details

This repo demonstrates the implementation of a model that is capable of detecting different types of toxicity like threats, obscenity, insults, and identity-based hate. This model was able to achieve ROC AUC score of **0.95** after 5 fold cross validation. 

### Dataset

**data.csv** utilizes the training file obtained from [Kaggle](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data). It contains **159571** comments. It contains large number of Wikipedia comments which have been labeled by human raters for toxic behavior. The types of toxicity are: toxic, severe_toxic, obscene, threat, insult and identity_hate.

### Feature Engineering
I have broadly categorized my feature engineering ideas into 3 groups.
* **N-gram based features :** Word n-gram (unigram, bigram and trigram), Char n-gram (upto 5 gram) and POS n-gram (unigram and bigram)

* **Features based on text statistics :** Count of sentences, Count of words, Count of punctuations, Diversity Count, Stopword count, Count of uppercase words, Average length of each word

* **Sentiment based features using Pattern library of CLiPS :** Subjectivity and Polarity

### Results
The results are reported on Linear SVM classifier after 5 fold cross vaidation.



