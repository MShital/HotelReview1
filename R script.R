#tools-->global option-->
#Extract tweets from twitter
search.string <- readline(prompt="Enter an Text: ")
#install.packages("tm",dependencies=TRUE)
#install.packages("NLP",dependencies=TRUE)
library("tm")



rw=read.csv('HotelRw_csv.csv')

rw_d=data.frame(rw)
library(tm)
myCorpus <- Corpus(VectorSource(rw_d$Review))
#myCorpus <- tm_map(myCorpus, content_transformer(tolower))
removeURL <- function(x) gsub("http[^[:space:]]*", "", x)

# remove URLs

removeURL <- function(x) gsub("http[^[:space:]]*", "", x)

#myCorpus<-Corpus(VectorSource(tolower(myCorpus)))

myCorpus <- tm_map(myCorpus, content_transformer(removeURL))

# remove anything other than English letters or space

removeNumPunct <- function(x) gsub("[^[:alpha:][:space:]]*", "", x)

myCorpus <- tm_map(myCorpus, content_transformer(removeNumPunct))


myCorpus <- tm_map(myCorpus, content_transformer(tolower))

# remove stopwords


#myCorpus <- tm_map(myCorpus, content_transformer(tolower))
#stopwordlist<-c("r", "big","use", "see", "used", "via","https", "amp","will","this","also","first","call","till","cant","like","what","This","THIS")
line=gsub("#","",search.string)
write(line,file="D:/sales force/tweeter analysis/stopWords.csv",append=TRUE)

#stopwordlist<-read.csv(choose.files(),header = FALSE,sep = ",")

stopwordlist<-read.csv(file="D:/sales force/tweeter analysis/stopWords.csv",header = FALSE,sep = ",")

myStopwords <- c(setdiff(stopwords('english'), c("r", "big")),"use", "see", "used", "via","https", "amp","will","this","also","first","modi","rahul","jhasanjay")

#########
########
myStopwords <- c(setdiff(stopwords('english'),search_text))

myStopwords1<-unlist(myStopwords)

stopwordlist<-unlist(stopwordlist)

myCorpus <- tm_map(myCorpus, removeWords, myStopwords1)

myCorpus <- tm_map(myCorpus, removeWords, stopwordlist)


myCorpus <- tm_map(myCorpus, removeWords, c("modi","rahul","jhasanjay","speech","rahulgandhi","narendramodi","congress","india","gandhi","vote","sanjaynirupam"))
#myStopwords <- c(stopwords('english'), stopwordlist)
#myStopwords<-unlist(myStopwords)


#myStopwords<-unlist(myStopwords)
myCorpus <- tm_map(myCorpus, removeWords, myStopwords)
#myCorpus <- tm_map(myCorpus, removeWords, "khairatabad")

myCorpus <- tm_map(myCorpus, removeWords, c("this","will"))
myCorpus <- tm_map(myCorpus, removeWords, c("must","Must","MUST","bjp","Bjp","BJP"))

# remove extra whitespace


myCorpus <- tm_map(myCorpus, stripWhitespace)


# keep a copy for stem completion later

myCorpusCopy <- myCorpus

##To Avoid incomplete words dnt run following command

#docs <- tm_map(docs, PlainTextDocument)  # needs to come before stemming
#docs <- tm_map(docs, stemDocument, "english")



#myCorpus <- tm_map(myCorpus, stemDocument)

head(myCorpus)

#install.packages("SnowballC")
library(SnowballC)
##Minimum word length is 3 charactes
tdm <- TermDocumentMatrix(myCorpus,control = list(wordLengths = c(4, Inf)))

head(tdm)
#install.packages("wordcloud")

library(wordcloud)

m <- as.matrix(tdm)

# calculate the frequency of words and sort it by frequency


word.freq <- sort(rowSums(m), decreasing = T)

#wordcloud(words = names(word.freq), freq = word.freq,scale=c(4,.5), min.freq = 5,  random.order = F)

wordcloud(words = names(word.freq), freq = word.freq,scale=c(4,.5), min.freq = 5,  random.order = F,colors=brewer.pal(6, "Dark2"))

##To find association between words
findAssocs(tdm, "anitha", corlimit=0.3)
###########################################################graph

install.packages("graph")
install.packages("Rgraphviz",dependencies = TRUE)
library(graph)
library(Rgraphviz)
source("http://bioconductor.org/biocLite.R")
biocLite("Rgraphviz")
biocLite("Rgraphviz")
tdm <- TermDocumentMatrix(myCorpus,control = list(wordLengths = c(4, Inf)))
tdm
idx <- which(dimnames(tdm)$Terms %in% c("r", "data", "mining"))
as.matrix(tdm[idx, 21:30])
# inspect frequent words
(freq.terms <- findFreqTerms(tdm, lowfreq = 45))
plot(tdm, term = freq.terms, corThreshold = 0.1, weighting = T,type="l")

plot(tdm, term = freq.terms, corThreshold = 0.1,type="l")

#freq.terms <- findFreqTerms(tdm, lowfreq = 20))

plot(tdm, term = freq.terms, corThreshold = 0.7, weighting = T)
###########

require(devtools)
install_github("sentiment140", "okugami79")

library(sentiment)
sentiments <- sentiment(tweets.df$text)
table(sentiments$polarity)




###
dtm <- as.DocumentTermMatrix(tdm)
install.packages("topicmodels")
library(topicmodels)
lda <- LDA(dtm, k = 8) # find 8 topics
term <- terms(lda, 7) # first 7 terms of every topic
(term <- apply(term, MARGIN = 2, paste, collapse = ", "))


table(tweets.df$retweetCount)
selected <- which(tweets.df$retweetCount >= 9)
dates <- strptime(tweets.df$created, format="%k")
dates
tweets.df$created

plot(x=dates, y=tweets.df$retweetCount, type="l", col="grey",
     xlab="Date", ylab="Times retweeted")
colors <- rainbow(10)[1:length(selected)]
points(dates[selected], tweets.df$retweetCount[selected],
       pch=19, col=colors)
text(dates[selected], tweets.df$retweetCount[selected],
     tweets.df$text[selected], col=colors, cex=.9)


###remove the user names


############extraxt users from tweets content about whom people are talking
x <- tweets.df$text

users1 <- function(x){
  xx <- strsplit(x, " ")
  lapply(xx, function(xx)xx[grepl("@[[:alnum:]]", xx)])
}

users1(x)

cc<-users1(x)

require(reshape2)
cc$id <- rownames(cc) 
dd<-melt(cc)
head(dd$value)


#Remove @ and : in text
ff<-gsub("@","",dd$value)
ee<-gsub(":","",ff)
gg<-data.frame(ee)
head(gg$ee)
write.csv( gg , file="user15.csv")



###get rownames in matrix
xx<-rownames(m)
head(xx)
xx.dataframe<-data.frame(xx)
colnames(xx.dataframe)<-c("text")
head(xx.dataframe)
head(gg$ee)
##usernames@

yy.dataframe<-data.frame(gg$ee)
head(yy.dataframe)
colnames(yy.dataframe)<-c("text")
zz<-inner_join(xx.dataframe, yy.dataframe)
un_zz<-unique(zz$text)

len<-length(un_zz)

for(i in 1:len)
{myCorpus <- tm_map(myCorpus, removeWords, un_zz[i]);
}

myStopwords <- c(stopwords('english'), stopwordlist)
myStopwords<-unlist(myStopwords)

myCorpus <- tm_map(myCorpus, removeWords, myStopwords)

myCorpus <- tm_map(myCorpus, stripWhitespace)


myCorpusCopy <- myCorpus

##To Avoid incomplete words dnt run following command

myCorpus <- tm_map(myCorpus, stemDocument)

#install.packages("SnowballC")
library(SnowballC)
##Minimum word length is 4 charactes
tdm <- TermDocumentMatrix(myCorpus,control = list(wordLengths = c(4, Inf)))

install.packages("wordcloud")

library(wordcloud)

m <- as.matrix(tdm)

# calculate the frequency of words and sort it by frequency

word.freq <- sort(rowSums(m), decreasing = T)

wordcloud(words = names(word.freq), freq = word.freq,scale=c(4,.5), min.freq = 5,  random.order = F)

###to write data in csv

m <- as.matrix(tdm)
dim(m)
write.csv(m, file="dtm4.csv")
getwd()

tdm1 <- DocumentTermMatrix(myCorpus,control = list(wordLengths = c(4, Inf)))

mm <- as.matrix(tdm1)



####to display maximum 50 words
wordcloud(words = names(word.freq), freq = word.freq,scale=c(4,.5), min.freq = 3,  random.order = F, max.words=50)
##to add colors to words
wordcloud(words = names(word.freq), freq = word.freq,scale=c(4,.5), min.freq = 3,  random.order = F,colors=brewer.pal(6, "Dark2"))

#3rotate words
wordcloud(words = names(word.freq), freq = word.freq,scale=c(4,.5), min.freq = 3,  random.order = F,colors=brewer.pal(6, "Dark2"),rot.per=0.4)
##To find association between words
findAssocs(tdm, "anitha", corlimit=0.6)

##############################

m <- as.matrix(tdm)
FreqMat <- data.frame(ST = rownames(m), 
                      Freq = rowSums(m), 
                      row.names = NULL)
###sort data on freq of word     



require(devtools)
install_github("sentiment140", "okugami79")

FreqMat1 <- FreqMat[order(-FreqMat$Freq),] 

n<-NROW(FreqMat2)

sel_n<-n*.50
sel_n
summary(FreqMat1)

FreqMat2 <- subset(FreqMat1, FreqMat1$Freq >10)

mat3<-subset(FreqMat2,nrows=20)
NROW(mat3)
library(sentiment)
sentiments <- sentiment(tweets.df$text)
table(sentiments$polarity)
mat3

barplot(mat3$Freq,
        names.arg=FreqMat2$ST)
plot(mat3$ST,mat3$Freq)
infj