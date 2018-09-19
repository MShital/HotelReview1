# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 12:03:31 2018

@author: admin
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import nltk
df=pd.read_csv('Restaurant_Reviews.tsv',delimiter='\t',quoting=3)
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
lines=df.Review

sentences = nltk.sent_tokenize(comment_words) 
tokenized = nltk.word_tokenize(df.Review)
weets_tokens = tokenized(df.Review)


head(df.Review)

corpus=[]

comment_words = ' '
for words in lines:
     comment_words=comment_words+words+' '

tokenized = nltk.word_tokenize(comment_words)

is_noun = lambda pos: pos[:2] == 'JJ'

nouns = [word for (word, pos) in nltk.pos_tag(tokenized) if is_noun(pos)] 

stopwords = set(STOPWORDS)

comment_words=''

for words in nouns:
    comment_words=comment_words+words+' '
    
comment_words

wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = stopwords, 
                min_font_size = 10).generate(comment_words)

plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
plt.show() 


#####################################

nouns


    
pos_ls=[]

df_pos=pd.read_csv('Positive_words.csv')

pos_adj=df_pos['abound'].values.tolist()
    
neg_adj=df_neg['abnormal'].values.tolist()

df_neg=pd.read_csv('neg.csv')

df_neg=pd.read_csv('neg.txt')

words1=''
for nouns in pos_adj:
    words1=words1+nouns+' '
    
words1

wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = stopwords, 
                min_font_size = 10).generate(words1)

plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
plt.show() 


##################for Negative words
words1_neg=''
comm=''
for nouns in neg_adj:
   # print(nouns)
    if nouns==neg_adj:
        print(nouns)
        words1_neg=words1_neg+nouns+' '
    
    if nouns in lines:
        comm=comm+lines+'  '
        
        
if text == 'This is correct':
    print("Correct")        
        
    
words1_neg

wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = stopwords, 
                min_font_size = 10).generate(words1_neg)

plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
plt.show() 

from nltk.corpus import sentiwordnet as swn


############++++++++++++++++Working+++++++++++++++++##############
#############segreagte topics wise comments

topic=''
name = input("What u want to analyse :")
    
for val in df.Review:
  val=str(val)
  if name.lower() in val.lower():
      topic=topic+val+' ' 
      
  

tokenized = nltk.word_tokenize(topic)

is_noun = lambda pos: pos[:2] == 'JJ'

nouns = [word for (word, pos) in nltk.pos_tag(tokenized) if is_noun(pos)] 

stopwords = set(STOPWORDS)

comment_words=''

for words in nouns:
    comment_words=comment_words+words+' '
    
comment_words

wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = stopwords, 
                min_font_size = 10).generate(comment_words)


plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
  
plt.show() 
##################################

#+++++++++++++++####################another sentimental script++++++++++++
##http://www.nltk.org/howto/sentiment.html
https://text-processing.com/demo/sentiment/
http://www.cs.cornell.edu/people/pabo/movie-review-data/

 from nltk.classify import NaiveBayesClassifier
 from nltk.corpus import subjectivity
 from nltk.sentiment import SentimentAnalyzer
 from nltk.sentiment.util import *
 n_instances = 100
 subj_docs = [(sent, 'subj') for sent in subjectivity.sents(categories='subj')[:n_instances]]
 obj_docs = [(sent, 'obj') for sent in subjectivity.sents(categories='obj')[:n_instances]]
 len(subj_docs), len(obj_docs)
(100, 100)
subj_docs[0]
(['smart', 'and', 'alert', ',', 'thirteen', 'conversations', 'about', 'one',
'thing', 'is', 'a', 'small', 'gem', '.'], 'subj')

 train_subj_docs = subj_docs[:80]
 test_subj_docs = subj_docs[80:100]
 train_obj_docs = obj_docs[:80]
 test_obj_docs = obj_docs[80:100]
 training_docs = train_subj_docs+train_obj_docs
 testing_docs = test_subj_docs+test_obj_docs
 sentim_analyzer = SentimentAnalyzer()
 all_words_neg = sentim_analyzer.all_words([mark_negation(doc) for doc in training_docs])
 
  unigram_feats = sentim_analyzer.unigram_word_feats(all_words_neg, min_freq=4)
 len(unigram_feats)
83
 sentim_analyzer.add_feat_extractor(extract_unigram_feats, unigrams=unigram_feats)

 training_set = sentim_analyzer.apply_features(training_docs)
 test_set = sentim_analyzer.apply_features(testing_docs) 
 trainer = NaiveBayesClassifier.train
 classifier = sentim_analyzer.train(trainer, training_set)
Training classifier
for key,value in sorted(sentim_analyzer.evaluate(test_set).items()):
     print('{0}: {1}'.format(key, value))
     
     
     
Evaluating NaiveBayesClassifier results...
Accuracy: 0.8
F-measure [obj]: 0.8
F-measure [subj]: 0.8
Precision [obj]: 0.8
Precision [subj]: 0.8
Recall [obj]: 0.8
Recall [subj]: 0.8 
 


from nltk.sentiment.vader import SentimentIntensityAnalyzer

 sentences = ["VADER is smart, handsome, and funny.", # positive sentence example
...    "VADER is smart, handsome, and funny!", # punctuation emphasis handled correctly (sentiment intensity adjusted)
...    "VADER is very smart, handsome, and funny.",  # booster words handled correctly (sentiment intensity adjusted)
...    "VADER is VERY SMART, handsome, and FUNNY.",  # emphasis for ALLCAPS handled
...    "VADER is VERY SMART, handsome, and FUNNY!!!",# combination of signals - VADER appropriately adjusts intensity
...    "VADER is VERY SMART, really handsome, and INCREDIBLY FUNNY!!!",# booster words & punctuation make this close to ceiling for score
...    "The book was good.",         # positive sentence
...    "The book was kind of good.", # qualified positive sentence is handled correctly (intensity adjusted)
...    "The plot was good, but the characters are uncompelling and the dialog is not great.", # mixed negation sentence
...    "A really bad, horrible book.",       # negative sentence with booster words
...    "At least it isn't a horrible book.", # negated negative sentence with contraction
...    ":) and :D",     # emoticons handled
...    "",              # an empty string is correctly handled
...    "Today sux",     #  negative slang handled
...    "Today sux!",    #  negative slang with punctuation emphasis handled
...    "Today SUX!",    #  negative slang with capitalization emphasis
...    "Today kinda sux! But I'll get by, lol" # mixed sentiment example with slang and constrastive conjunction "but"
... ]

paragraph = "It was one of the worst movies I've seen, despite good reviews. \
... Unbelievably bad acting!! Poor direction. VERY poor production. \
... The movie was bad. Very bad movie. VERY bad movie. VERY BAD movie. VERY BAD movie!"

from nltk import tokenize
lines_list = tokenize.sent_tokenize(paragraph)

sentences.extend(lines_list)

 tricky_sentences = [
...    "Most automated sentiment analysis tools are shit.",
...    "VADER sentiment analysis is the shit.",
...    "Sentiment analysis has never been good.",
...    "Sentiment analysis with VADER has never been this good.",
...    "Warren Beatty has never been so entertaining.",
...    "I won't say that the movie is astounding and I wouldn't claim that \
...    the movie is too banal either.",
...    "I like to hate Michael Bay films, but I couldn't fault this one",
...    "It's one thing to watch an Uwe Boll film, but another thing entirely \
...    to pay for it",
...    "The movie was too good",
...    "This movie was actually neither that funny, nor super witty.",
...    "This movie doesn't care about cleverness, wit or any other kind of \
...    intelligent humor.",
...    "Those who find ugly meanings in beautiful things are corrupt without \
...    being charming.",
...    "There are slow and repetitive parts, BUT it has just enough spice to \
...    keep it interesting.",
...    "The script is not fantastic, but the acting is decent and the cinematography \
...    is EXCELLENT!",
...    "Roger Dodger is one of the most compelling variations on this theme.",
...    "Roger Dodger is one of the least compelling variations on this theme.",
...    "Roger Dodger is at least compelling as a variation on the theme.",
...    "they fall in love with the product",
...    "but then it breaks",
...    "usually around the time the 90 day warranty expires",
...    "the twin towers collapsed today",
...    "However, Mr. Carter solemnly argues, his client carried out the kidnapping \
...    under orders and in the ''least offensive way possible.''"
... ]
 
 sentences.extend(tricky_sentences)
 sid = SentimentIntensityAnalyzer()
for sentence in sentences:
    print(sentence)
    ss = sid.polarity_scores(sentence)
    for k in sorted(ss):
        print('{0}: {1}, '.format(k, ss[k]), end='')
    print()
        
        
    
        
            
 
 for sentence in sentences:
  print(sentence)
     ss = sid.polarity_scores(sentence)
     for k in sorted(ss):
         print('{0}: {1}, '.format(k, ss[k]), end='')
     print()
 
 
 
 
 
#++++++++++++++#################################

for val in df.Review:
  val=str(val)
  #print(val)
  tokens=val.split()
  for i in range(len(tokens)):
    tokens[i]=tokens[i].lower()
    for words in tokens:
      comment_words=comment_words+words+' '
      #print(comment_words)
   for i in range(len(tokens)):
    tokens[i]=tokens[i].lower()
    for words in tokens:
      comment_words=comment_words+words+' '
      print(comment_words)
      
      

      
