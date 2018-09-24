library("tm")

library(udpipe)

rw=read.csv('HotelRw_csv.csv')

rw_d=data.frame(rw)
library(tm)
myCorpus <- Corpus(VectorSource(rw_d$Review))
#myCorpus <- tm_map(myCorpus, content_transformer(tolower))
removeURL <- function(x) gsub("http[^[:space:]]*", "", x)

# remove URLs

removeURL <- function(x) gsub("http[^[:space:]]*", "", x)



myCorpus <- tm_map(myCorpus, content_transformer(removeURL))

# remove anything other than English letters or space

removeNumPunct <- function(x) gsub("[^[:alpha:][:space:]]*", "", x)

myCorpus <- tm_map(myCorpus, content_transformer(removeNumPunct))



myCorpus <- tm_map(myCorpus, content_transformer(tolower))

# remove stopwords



stopwordlist<-read.csv(file="D:/sales force/tweeter analysis/stopWords.csv",header = FALSE,sep = ",")

myStopwords <- c(setdiff(stopwords('english'), c("r", "big")),"use", "see", "used", "via","https", "amp","will","this","also","first","modi","rahul","jhasanjay")


myStopwords1<-unlist(myStopwords)

stopwordlist<-unlist(stopwordlist)

myCorpus <- tm_map(myCorpus, removeWords, myStopwords1)



#myStopwords<-unlist(myStopwords)
myCorpus <- tm_map(myCorpus, removeWords, myStopwords)
#myCorpus <- tm_map(myCorpus, removeWords, "khairatabad")

myCorpus <- tm_map(myCorpus, removeWords, c("this","will"))
myCorpus <- tm_map(myCorpus, removeWords, c("must","Must","MUST","bjp","Bjp","BJP"))

# remove extra whitespace


myCorpus <- tm_map(myCorpus, stripWhitespace)


# keep a copy for stem completion later

myCorpusCopy <- myCorpus

#myCorpus <- tm_map(myCorpus, stemDocument)

head(myCorpus)

#install.packages("SnowballC")
library(SnowballC)
##Minimum word length is 3 charactes
tdm <- TermDocumentMatrix(myCorpus,control = list(wordLengths = c(4, Inf)))

#head(tdm)
#install.packages("wordcloud")

library(wordcloud)

m <- as.matrix(tdm)

# calculate the frequency of words and sort it by frequency


word.freq <- sort(rowSums(m), decreasing = T)

#wordcloud(words = names(word.freq), freq = word.freq,scale=c(4,.5), min.freq = 5,  random.order = F)

wordcloud(words = names(word.freq), freq = word.freq,scale=c(4,.5), min.freq = 5,  random.order = F,colors=brewer.pal(6, "Dark2"))

head(rw_d)



i=1
all_view=""
for (i in 1:nrow(rw_d)){
  all_view=paste(all_view, rw_d[i,1])
}


udmodel <- udpipe_load_model(file = "E://GitHub//HotelReview//english-ud-2.0-170801.udpipe")
#udmodel <- udpipe_load_model(file = udmodel$file_model)
x <- udpipe_annotate(udmodel,  x = all_view)
x <- as.data.frame(x)
stats <- subset(x, upos %in% "NOUN")
stats
x
stats <- subset(x, upos %in% "NOUN")
stats <- txt_freq(x = stats$lemma)

library(lattice)
stats$key <- factor(stats$key, levels = rev(stats$key))
barchart(key ~ freq, data = head(stats, 30), col = "cadetblue", main = "Most occurring nouns", xlab = "Freq")



stats <- subset(x, upos %in% "ADJ")
stats
x
stats <- subset(x, upos %in% "ADJ")
stats <- txt_freq(x = stats$lemma)

library(lattice)
stats$key <- factor(stats$key, levels = rev(stats$key))
barchart(key ~ freq, data = head(stats, 30), col = "cadetblue", main = "Most occurring nouns", xlab = "Freq")

###################################
#install.packages("ggraph")
library(igraph)
library(ggraph)
library(ggplot2)
wordnetwork <- head(stats, 30)
wordnetwork <- graph_from_data_frame(wordnetwork)
ggraph(wordnetwork, layout = "fr") +
  geom_edge_link(aes(width = cooc, edge_alpha = cooc), edge_colour = "pink") +
  geom_node_text(aes(label = name), col = "darkgreen", size = 5) +
  theme_graph(base_family = "Arial Narrow") +
  theme(legend.position = "none") +
  labs(title = "Cooccurrences within 3 words distance", subtitle = "Nouns & Adjective")


##############
install.packages("textrank")

library(textrank)
i=1
all_view=""
for (i in 1:nrow(rw_d)){
  all_view=paste(all_view, rw_d[i,1])
}


udmodel <- udpipe_load_model(file = "E://GitHub//HotelReview//english-ud-2.0-170801.udpipe")
#udmodel <- udpipe_load_model(file = udmodel$file_model)
x <- udpipe_annotate(udmodel,  x = all_view)
x <- as.data.frame(x)

stats <- textrank_keywords(x$lemma, relevant = x$upos %in% c("NOUN", "ADJ"),    ngram_max = 20, sep = " ")
stats <- subset(stats$keywords, ngram > 1 & freq >= 2)
library(wordcloud)
wordcloud(words = stats$keyword, freq = stats$freq)

#################
rw=read.csv('HotelRw_csv.csv')

rw_d=data.frame(rw)

udmodel <- udpipe_load_model(file = "E://GitHub//HotelReview//english-ud-2.0-170801.udpipe")
x <- udpipe_annotate(udmodel, x = rw_d$Review)
x <- as.data.frame(x)


stats <- merge(x, x, 
               by.x = c("doc_id", "paragraph_id", "sentence_id", "head_token_id"),
               by.y = c("doc_id", "paragraph_id", "sentence_id", "token_id"),
               all.x = TRUE, all.y = FALSE, 
               suffixes = c("", "_parent"), sort = FALSE)
stats <- subset(stats, dep_rel %in% "nsubj" & upos %in% c("NOUN") & upos_parent %in% c("ADJ"))
stats$term <- paste(stats$lemma_parent, stats$lemma, sep = " ")
stats <- txt_freq(stats$term)
library(wordcloud)
wordcloud(words = stats$key, freq = stats$freq, min.freq = 2, max.words = 500,
          random.order = FALSE, colors = brewer.pal(6, "Dark2"))


























































###Most Occuring Adjective
stats <- subset(x, upos %in% "NOUN")
stats
x
stats <- subset(x, upos %in% "ADJ")
stats <- txt_freq(x = stats$lemma)

library(lattice)
stats$key <- factor(stats$key, levels = rev(stats$key))
barchart(key ~ freq, data = head(stats, 30), col = "cadetblue", main = "Most occurring Adjective", xlab = "Freq")



stats <- cooccurrence(x = x$lemma, 
                      relevant = x$upos %in% c("NOUN", "ADJ"), skipgram = 2)
