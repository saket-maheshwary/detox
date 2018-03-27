import os
import re
import sys
import nltk
import time
import string
import warnings
import numpy as np
import pandas as pd
from pattern.en import sentiment
from nltk.corpus import treebank
from nltk.corpus import stopwords
from sklearn.svm import LinearSVC
from nltk.tokenize import word_tokenize
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from nltk.stem.snowball import SnowballStemmer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction import DictVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from nltk.tag.sequential import ClassifierBasedPOSTagger
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

np.random.seed(127)

reload(sys)
sys.setdefaultencoding('utf8')
warnings.filterwarnings("ignore")

eng_stopwords = set(stopwords.words("english"))


# Counts the number of sentences seperated by '\n' in each comment

class countSentences(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, df, y=None):
        return self
    
    def transform(self, df, y=None):
        return df.apply(lambda x: len(re.findall("\n",str(x)))+1)


# Counts the number of words in each comment

class countWords(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, df, y=None):
        return self
    
    def transform(self, df, y=None):
        return df.apply(lambda x: len(str(x).split()))


# Calculates lexical diversity in each comment

class lexicalDiversity(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def diversity(self, text):
        return len(set(text)) / len(text)
    
    def fit(self, df, y=None):
        return self
    
    def transform(self, df, y=None):
        return df.apply(self.diversity)


# Counts the number of punctuations in each comment

class countPunctuations(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, df, y=None):
        return self
    
    def transform(self, df, y=None):
        return df.apply(lambda x: len([c for c in str(x) if c in string.punctuation]))


# Counts the number of upper case words in each comment

class countUppercase(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, df, y=None):
        return self
    
    def transform(self, df, y=None):
        return df.apply(lambda x: len([w for w in str(x).split() if w.isupper()]))


# Counts the number of stopwords in each comment

class countStopwords(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, df, y=None):
        return self
    
    def transform(self, df, y=None):
        return df.apply(lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))


# Average length of words in each comment

class averageLength(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, df, y=None):
        return self
    
    def transform(self, df, y=None):
        return df.apply(lambda x: np.mean([len(w) for w in str(x).split()]))


# POS Tag sequence for a given comment

class getPOSTags(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def POSTag(self, text):
        token_text = nltk.word_tokenize(unicode(text))
        return (' ').join([x[1] for x in nltk.pos_tag(token_text)])
    
    def fit(self, df, y=None):
        return self
    
    def transform(self, df, y=None):
        return df.apply(self.POSTag)


# Finds the Polarity in each comment using Pattern library of CLiPS

class findPolarity(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def polar(self, text):
        a,b = sentiment(text)
        return a
    
    def fit(self, df, y=None):
        return self
    
    def transform(self, df, y=None):
        return df.apply(self.polar)


# Finds the subjectivity in each comment using Pattern library of CLiPS

class findSubjectivity(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def subjective(self, text):
        a,b = sentiment(text)
        return b
    
    def fit(self, df, y=None):
        return self
    
    def transform(self, df, y=None):
        return df.apply(self.subjective)

                                                                    
# Converts the input to DataFrame.

class getDF(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, df, y=None):
        return self
    
    def transform(self, df, y=None):
        return pd.DataFrame(df)


# Converts the DataFrame to a List

class df2List(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, df, y=None):
        return self
    
    def transform(self, df, y=None):
        res = df.tolist()
        return res
                                    

class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']


pipe = Pipeline([
                     ('features', FeatureUnion([
                                                 ('sentence_count', Pipeline([
                                                                              ('extract', countSentences()),
                                                                              ('vect', getDF())
                                                                              ])),
                                                
                                                 ('word_count', Pipeline([
                                                                             ('extract', countWords()),
                                                                             ('vect', getDF())
                                                                             ])),
                                                
                                                 ('diversity_count', Pipeline([
                                                                             ('extract', lexicalDiversity()),
                                                                             ('vect', getDF())
                                                                             ])),

                                                 ('punctuation_count', Pipeline([
                                                                           ('extract', countPunctuations()),
                                                                           ('vect', getDF())
                                                                           ])),
                                                
                                                 ('uppercase_count', Pipeline([
                                                                           ('extract', countUppercase()),
                                                                           ('vect', getDF())
                                                                           ])),
                                                
                                                 ('stopword_count', Pipeline([
                                                                           ('extract', countStopwords()),
                                                                           ('vect', getDF())
                                                                           ])),
                                                
                                                 ('averageLength_count', Pipeline([
                                                                           ('extract', averageLength()),
                                                                           ('vect', getDF())
                                                                           ])),
                                                 ('pos_ngrams', Pipeline([
                                                                         ('extract', getPOSTags()),
                                                                         ('convert', df2List()),
                                                                         ('vect', CountVectorizer(ngram_range=(1,2),
                                                                                                  max_features= 1000)),
                                                                         ])),
                                                
                                                 ('polarity', Pipeline([
                                                                             ('extract', findPolarity()),
                                                                             ('vect', getDF())
                                                                             ])),
                                                
                                                 ('subjectivity', Pipeline([
                                                                             ('extract', findSubjectivity()),
                                                                             ('vect', getDF())
                                                                             ])),
                                                
                                                 ('word_ngram', TfidfVectorizer(sublinear_tf=True,
                                                                               strip_accents='unicode',
                                                                               analyzer='word',
                                                                               token_pattern=r'\w{1,}',
                                                                               ngram_range=(1, 3),
                                                                               lowercase=True,
                                                                               max_features= 10000)),
                                                
                                                 ('char_ngram', TfidfVectorizer(sublinear_tf=True,
                                                                               strip_accents='unicode',
                                                                               analyzer='char',
                                                                               ngram_range=(1, 5),
                                                                               lowercase=True,
                                                                               max_features=20000))
                                                
                                             ])
                      ),
  
                  ])


def main():
    start = time.time()
    data = pd.read_csv('data.csv').fillna(' ')
    text = data['comment_text']
    print ("\n")
    print ('Total number of comments: {}'.format((len(text))))

    print ("Fitting")
    pipe.fit(text)

    print ("Transforming")
    X = pipe.transform(text)
    
    #classifier = LogisticRegression()
    classifier = LinearSVC(C=1.0)
    
    print ("Evaluating")
    losses = []
    count = 0
    for class_name in class_names:
        y = data[class_name]
        cv_loss = np.mean(cross_val_score(classifier, X, y, cv=5, scoring='roc_auc'))
        losses.append(cv_loss)
        print('CV score for class {} is {}'.format(class_name, cv_loss))

    print('Total 5 fold CV score is {}'.format(np.mean(losses)))
    print('Total time taken for 5 fold CV is {} seconds'.format(time.time() - start))

if __name__ == "__main__":
    main()

