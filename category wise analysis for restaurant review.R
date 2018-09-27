head(dataset_original)

row_cnt=nrow(dataset_original)
row_cnt

dataset_original[969,1]

rw_d=data.frame(rw)


i=1
j=1
sel_rw=list()
like=list()
for(i in 1:row_cnt){
  if( length(grep("service",toString(tolower(dataset_original[i,1])),value=FALSE))==0){sel_rw[i]=0}
  else
   sel_rw[i]=1

}


i=1
j=1
food_rw=list()
for(i in 1:row_cnt){
  if( length(grep("food",toString(tolower(dataset_original[i,1])),value=FALSE))==0){food_rw[i]=0}
  else
    food_rw[i]=1
  
}

i=1
j=1
staff_rw=list()
for(i in 1:row_cnt){
  if( length(grep("staff",toString(tolower(dataset_original[i,1])),value=FALSE))==0){staff_rw[i]=0}
  else
    staff_rw[i]=1
  
}



i=1
j=1
place_rw=list()
for(i in 1:row_cnt){
  if( length(grep("place",toString(tolower(dataset_original[i,1])),value=FALSE))==0){place_rw[i]=0}
  else
    place_rw[i]=1
  
}

i=1
j=1
time_rw=list()
for(i in 1:row_cnt){
  if( length(grep("time",toString(tolower(dataset_original[i,1])),value=FALSE))==0){time_rw[i]=0}
  else
    time_rw[i]=1
  
}


cat=data.frame()
dataset_original$Review
cat=data.frame(cbind("time"=time_rw,"place"=place_rw,"sevice"=sel_rw,"food"=food_rw,"staff"=staff_rw,"review"=dataset_original$Review,"Liked"=dataset_original$Liked))
write.csv(data.frame(cat),file="category_wise_analysis",append = TRUE,row.names = FALSE)
help("write.csv")

sapply(cat, class)


class(cat)
write.csv(cat,file = "test")

head(cat)
###########################################
dataset_original
library(dplyr)
f=filter(dataset_original,Liked<1)
head(f)
?filter

cl_data=read.csv(file.choose())
f=filter(cl_data,time>0|place>0|sevice>0|food>0|staff>0)



select(cl_data,)
?select
