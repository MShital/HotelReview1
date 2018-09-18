# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 12:03:31 2018

@author: admin
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
df=pd.read_csv('Restaurant_Reviews.tsv',delimiter='\t',quoting=3)
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

corpus=[]

comment_words = ' '
stopwords = set(STOPWORDS)

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
      
      
      wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = stopwords, 
                min_font_size = 10).generate(comment_words)
      
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
  
plt.show() 