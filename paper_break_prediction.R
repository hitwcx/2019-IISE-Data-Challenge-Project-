library(stats)
library(corrplot)
library(tseries)
library(qcc)
library(jmotif)
library(glmnet)

wd<-"C:/Users/.../Conference/IISE 2019/QCRE data challenge 2019"

data.file<-"processminer-rare-event-mts - data.csv"

### parameters
wlen<-c(100, 500, 1000) # window length
alpha.size<-20 
paa.size<-10
Thresh<-c(20, 200, 2000) # number of selected features
train.size<-8000 # to adjust the size of training set
cut<-0.5

### import data
setwd(wd)
data<-read.csv(data.file)

Time<-data$time
Y<-data$y
X<-data[, -match(c("time", "y"), names(data))]

### correlation and correlogram
Cor<-cor(X)

pdf("Correlogram.pdf", width=10, height=10)
corrplot(Cor)
dev.off()

### PCA
PCA<-prcomp(X)

pdf("Pareto Chart for all PCA.pdf", width=10, height=10)
pareto.chart(PCA$sdev, ylab="Variances",xlab="Principal Component") 
dev.off()

PC1<-matrix(unlist(X), , ncol(PCA$rotation))%*%matrix(PCA$rotation[, "PC1"], , 1)

pdf("PC1.pdf", width=50, height=10)
par(mar=c(10, 6, 5, 5))
plot(1:length(PC1), PC1, xlab=" ", ylab="Response", xaxt="n")
axis(1, at=which(Y==1), label=Time[Y==1], las=2)
abline(v=which(Y==1), col=2, lty=2)
dev.off()

### add derivatives & separating training and testing data
X1<-X
for(i in 1:ncol(X))
  X1<-cbind(X1, c(0, abs(diff(X[, i]))))

X.train<-X1[1:train.size, ]
X.test<-X1[(train.size+1):(nrow(X)-wlen[1]), ]
Y.train<-Y[1:train.size]
Y.test<-Y[(train.size+1):nrow(X)]


###### feature selection
### number to words
Bag0<-c() # bags for 0
Bag1<-c() # bags for 1
Bag<-c()
count0<-c()
count1<-c()

Y1<-list() # response for logistic regression
Y.t<-list()
Matrix<-list()
Matrix.t<-list()
for(j in 1:length(wlen))
{
  YY<-c()
  YY.t<-c()
  MM<-matrix(, (nrow(X.train)-wlen[j]+1), ncol(X.train)) # record training set
  MM.t<-matrix(, (nrow(X.test)-wlen[j]+1), ncol(X.test)) # record training set
  
  for(p in 1:length(paa.size))
  {
    for(i in 1:ncol(X1))
    {
      for(a in 1:(nrow(X.train)-wlen[j]+1)) # training
      {
        if(F){
        temp<-series_to_string(paa(X.train[a:(a+wlen[j]-1), i], paa.size[p]), alpha.size)
       
        if(length(which(Bag==temp))==0)
        {
          Bag<-c(Bag, temp)
          
          if(length(which(Y[(a+1):(a+wlen[j])]==1))==0) # shift Yt to Y(t-1) for prediction
          {
            Bag0<-c(Bag0, temp)
            count0<-c(count0, 1)
          }else
          {
            Bag1<-c(Bag1, temp)
            count1<-c(count1, 1)
          }
        }else
        {
          if(length(which(Y[(a+1):(a+wlen[j])]==1))==0) # shift Yt to Y(t-1) for prediction
          {
            count0[which(Bag0==temp)]<-count0[which(Bag0==temp)]+1
          }else
            count1[which(Bag1==temp)]<-count1[which(Bag1==temp)]+1
        }
        }
        if(p==1)
        {
          MM[a, i]<-series_to_string(paa(X.train[a:(a+wlen[j]-1), i], paa.size[p]), alpha.size)
          YY[a]<-Y[a+wlen[j]]
        }
      }
      
      if(p==1)
      {
        for(a in 1:(nrow(X.test)-wlen[j]+1))
        {
           MM.t[a, i]<-series_to_string(paa(X.test[a:(a+wlen[j]-1), i], paa.size[p]), alpha.size)
           YY.t[a]<-Y.test[a+wlen[j]]
        }
      }
    }
  }
  
  Y1[[j]]<-YY
  Y.t[[j]]<-YY.t
  Matrix[[j]]<-MM
  Matrix.t[[j]]<-MM.t
}

for(j in 1:length(wlen))
{
  write.table(Matrix[[j]], paste("Matrix window", wlen[j], ".csv", sep=""), row.names=F, sep=",")
  write.table(Matrix.t[[j]], paste("Matrix.t window", wlen[j], ".csv", sep=""), row.names=F, sep=",")
} 

###### feature selection by chi2 test
P<-sum(count1)
N<-sum(count0)+sum(count1)

chi2<-c()
for(i in 1:length(Bag))
{
  A<-ifelse(length(which(Bag1==Bag[i]))>0, count1[which(Bag1==Bag[i])], 0)
  M<-ifelse(length(which(Bag0==Bag[i]))>0, count0[which(Bag0==Bag[i])], 0)+A
  
  chi2[i]<-N*(A*N-M*P)^2/(P*M*(N-P)*(N-M))
}


for(f in 1:length(Thresh))
{
thresh<-Thresh[f]
  
Order<-order(chi2, decreasing = T)
feature<-Bag[Order][1:thresh] # take the first several words
write.table(feature, paste("feature", thresh, ".csv", sep=""), row.names=F, col.names="feature", sep=",")

# tfidf calculation
bag0<-data.frame("words" = Bag0, "counts" = count0, stringsAsFactors = FALSE)
bag1<-data.frame("words" = Bag1, "counts" = count1, stringsAsFactors = FALSE)

ll<-list("bag0" = bag0, "bag1" = bag1)
tfidf<-bags_to_tfidf(ll)


###### classification
FPR<-c() 
FNR<-c()
accuracy<-c()
recall<-c()
precision<-c()
F1<-c()

FPR.t<-c()
FNR.t<-c()
accuracy.t<-c()
recall.t<-c()
precision.t<-c()
F1.t<-c()

FPR_1<-c()
FNR_1<-c()
accuracy_1<-c()
recall_1<-c()
precision_1<-c()
F1_1<-c()

FPR_1.t<-c()
FNR_1.t<-c()
accuracy_1.t<-c()
recall_1.t<-c()
precision_1.t<-c()
F1_1.t<-c()

for(j in 1:length(wlen))
{
  ### construct count data
  Matrix1<-matrix(, nrow(Matrix[[j]]), thresh)
  Matrix.t1<-matrix(, nrow(Matrix.t[[j]]), thresh)
  for(k in 1:thresh)
  {
    for(i in 1:nrow(Matrix[[j]]))
      Matrix1[i, k]<-length(which(Matrix[[j]][i, ]==feature[k]))
    
    for(i in 1:nrow(Matrix.t[[j]]))
      Matrix.t1[i, k]<-length(which(Matrix.t[[j]][i, ]==feature[k]))
  }
  
  write.table(Matrix1, paste("count of Matrix window", wlen[j], "thresh", thresh, ".csv", sep=""), row.names=F, col.names=feature, sep=",")
  write.table(Matrix.t1, paste("count of Matrix.t window", wlen[j], "thresh", thresh, ".csv", sep=""), row.names=F, col.names=feature, sep=",")
  
  
  ### logistic regression
  fit<-glm(Y1[[j]]~., data=data.frame(Matrix1), family="binomial")

  # prediction
  Y2<-predict(fit, newdata=data.frame(Matrix1), type="response") # training accuracy
  Y3<-rep(0, length(Y2))
  Y3[which(Y2>=cut)]<-1
  
  FP<-length(which(Y1[[j]]==0 & Y3==1))
  TP<-length(which(Y1[[j]]==1 & Y3==1))
  FN<-length(which(Y1[[j]]==1 & Y3==0))
  TN<-length(which(Y1[[j]]==0 & Y3==0))
  
  FPR[j]<-ifelse(FP+TN>0, FP/(FP+TN), NA)
  FNR[j]<-FN/(FN+TP)
  accuracy[j]<-(TP+TN)/length(Y3)
  
  recall[j]<-ifelse(TP+FP>0, TP/(TP+FP), NA)
  precision[j]<-TP/(TP+FN)
  F1[j]<-ifelse(precision[j]+recall[j]>0, 2*precision[j]*recall[j]/(precision[j]+recall[j]), NA)
  
  Y.t1<-predict(fit, newdata=data.frame(Matrix.t1), type="response") # testing accuracy
  Y.t2<-rep(0, length(Y.t1))
  Y.t2[which(Y.t1>=cut)]<-1
  
  FP.t<-length(which(Y.t[[j]]==0 & Y.t2==1))
  TP.t<-length(which(Y.t[[j]]==1 & Y.t2==1))
  FN.t<-length(which(Y.t[[j]]==1 & Y.t2==0))
  TN.t<-length(which(Y.t[[j]]==0 & Y.t2==0))
  
  FPR.t[j]<-ifelse(FP.t+TN.t>0, FP.t/(FP.t+TN.t), NA)
  FNR.t[j]<-FN.t/(FN.t+TP.t)
  accuracy.t[j]<-(TP.t+TN.t)/length(Y.t2)
  
  recall.t[j]<-ifelse(TP.t+FP.t>0, TP.t/(TP.t+FP.t), NA)
  precision.t[j]<-TP.t/(TP.t+FN.t)
  F1.t[j]<-ifelse(precision.t[j]+recall.t[j]>0, 2*precision.t[j]*recall.t[j]/(precision.t[j]+recall.t[j]), NA)


  ############ bags to tfidf
  class0<-c()
  class1<-c()

  for(i in 1:nrow(Matrix[[j]]))
  {
    temp<-unique(Matrix[[j]][i, ])
    
    count<-c()
    for(t in 1:length(temp))
      count[t]<-length(which(Matrix[[j]][i, ]==temp[t]))
    
    class0[i]<-tfidf[match(temp, tfidf[, 1]), 2]%*%count
    class1[i]<-tfidf[match(temp, tfidf[, 1]), 3]%*%count
  }
 
  res<-rep(0, length(Y1[[j]]))
  res[which(class1-class0>0)]<-1
  
  FP<-length(which(Y1[[j]]==0 & res==1))
  TP<-length(which(Y1[[j]]==1 & res==1))
  FN<-length(which(Y1[[j]]==1 & res==0))
  TN<-length(which(Y1[[j]]==0 & res==0))
  
  FPR_1[j]<-ifelse(FP+TN>0, FP/(FP+TN), NA)
  FNR_1[j]<-FN/(FN+TP)
  accuracy_1[j]<-(TP+TN)/length(res)
  
  recall_1[j]<-ifelse(TP+FP>0, TP/(TP+FP), NA)
  precision_1[j]<-TP/(TP+FN)
  F1_1[j]<-ifelse(precision_1[j]+recall_1[j]>0, 2*precision_1[j]*recall_1[j]/(precision_1[j]+recall_1[j]), NA)
  
  # testing accuracy
  class0<-c()
  class1<-c()
  
  for(i in 1:nrow(Matrix.t[[j]]))
  {
    temp<-unique(Matrix.t[[j]][i, ])
    
    count<-c()
    for(t in 1:length(temp))
      count[t]<-length(which(Matrix.t[[j]][i, ]==temp[t]))
    
    class0[i]<-tfidf[match(temp, tfidf[, 1]), 2]%*%count
    class1[i]<-tfidf[match(temp, tfidf[, 1]), 3]%*%count
  }
  
  res.t<-rep(0, length(Y.t[[j]]))
  res.t[which(class1-class0>0)]<-1
  
  FP.t<-length(which(Y.t[[j]]==0 & res.t==1))
  TP.t<-length(which(Y.t[[j]]==1 & res.t==1))
  FN.t<-length(which(Y.t[[j]]==1 & res.t==0))
  TN.t<-length(which(Y.t[[j]]==0 & res.t==0))
  
  FPR_1.t[j]<-ifelse(FP.t+TN.t>0, FP.t/(FP.t+TN.t), NA)
  FNR_1.t[j]<-FN.t/(FN.t+TP.t)
  accuracy_1.t[j]<-(TP.t+TN.t)/length(res.t)
  
  recall_1.t[j]<-ifelse(TP.t+FP.t>0, TP.t/(TP.t+FP.t), NA)
  precision_1.t[j]<-TP.t/(TP.t+FN.t)
  F1_1.t[j]<-ifelse(precision_1.t[j]+recall_1.t[j]>0, 2*precision_1.t[j]*recall_1.t[j]/(precision_1.t[j]+recall_1.t[j]), NA)
}

write.table(rbind(FPR, FNR, accuracy, recall, precision, F1), paste("train_logistic thresh", thresh, ".csv", sep=""), 
            row.names=c("FPR", "FNR", "accuracy", "recall", "precision", "F1 score"), 
            col.names=paste("window size: ", wlen, sep=""), sep=",")
write.table(rbind(FPR.t, FNR.t, accuracy.t, recall.t, precision.t, F1.t), paste("test_logistic thresh", thresh, ".csv", sep=""), 
            row.names=c("FPR", "FNR", "accuracy", "recall", "precision", "F1 score"), 
            col.names=paste("window size: ", wlen, sep=""), sep=",")
write.table(rbind(FPR_1, FNR_1, accuracy_1, recall_1, precision_1, F1_1), paste("train_tfidf.csv", sep=""),
            row.names=c("FPR", "FNR", "accuracy", "recall", "precision", "F1 score"), 
            col.names=paste("window size: ", wlen, sep=""), sep=",")
write.table(rbind(FPR_1.t, FNR_1.t, accuracy_1.t, recall_1.t, precision_1.t, F1_1.t), paste("test_tfidf.csv", sep=""),
            row.names=c("FPR", "FNR", "accuracy", "recall", "precision", "F1 score"), 
            col.names=paste("window size: ", wlen, sep=""), sep=",")
}


####### visualize performance
files1<-list.files(pattern="train_logistic thresh")
files2<-list.files(pattern="test_logistic thresh")

# against Thresh
for(j in 1:length(wlen))
{
  FPR1<-c()
  FNR1<-c()
  recall1<-c()
  precision1<-c()
  
  FPR2<-c()
  FNR2<-c()
  recall2<-c()
  precision2<-c()
  
  for(i in 1:length(Thresh))
  {
    data.train<-read.csv(files1[which(substr(files1, 22, nchar(files1)-4)==Thresh[i])])
    data.test<-read.csv(files2[which(substr(files2, 21, nchar(files2)-4)==Thresh[i])])
    
    FPR1[i]<-data.train[1, j]
    FNR1[i]<-data.train[2, j]
    recall1[i]<-data.train[4, j]
    precision1[i]<-data.train[5, j]
    
    FPR2[i]<-data.test[1, j]
    FNR2[i]<-data.test[2, j]
    recall2[i]<-data.test[4, j]
    precision2[i]<-data.test[5, j]
  }
  
  pdf(paste("against thresh_window", wlen[j], ".pdf", sep=""), width=10, height=8)
  matplot(1:length(Thresh), cbind(FPR1, FNR1, recall1, precision1, FPR2, FNR2, recall2, precision2), type="b", 
          lty=c(rep(1, 4), rep(2, 4)), pch=c(rep(1, 4), rep(5, 4)), lwd=3, col=rep(rainbow(4), 2), cex=2,
          xlab="Feature Set Size", xaxt="n", ylab="Rate")
  axis(1, at=1:length(Thresh), labels=Thresh)
  legend("left", legend=c("FPR: train", "FNR: train", "recall: train", "precision: trian", 
                             "FPR: test", "FNR: test", "recall: test", "precision: test"), 
         lty=c(rep(1, 4), rep(2, 4)), pch=c(rep(1, 4), rep(5, 4)), lwd=3, col=rep(rainbow(4), 2), cex=1.5)
  dev.off()
}

# against window size
for(i in 1:length(Thresh))
{
  data.train<-read.csv(files1[which(substr(files1, 22, nchar(files1)-4)==Thresh[i])])
  data.test<-read.csv(files2[which(substr(files2, 21, nchar(files2)-4)==Thresh[i])])
    
  FPR1<-data.train[1, ]
  FNR1<-data.train[2, ]
  recall1<-data.train[4, ]
  precision1<-data.train[5, ]
    
  FPR2<-data.test[1, ]
  FNR2<-data.test[2, ]
  recall2<-data.test[4, ]
  precision2<-data.test[5, ]
  
  pdf(paste("against window_thresh", wlen[j], ".pdf", sep=""), width=10, height=8)
  matplot(1:length(wlen), cbind(FPR1, FNR1, recall1, precision1, FPR2, FNR2, recall2, precision2), type="b", 
          lty=c(rep(1, 4), rep(2, 4)), pch=c(rep(1, 4), rep(5, 4)), lwd=3, col=rep(1:4, 2), cex=2,
          xlab="Window Size", xaxt="n", ylab="Rate")
  axis(1, at=1:length(wlen), labels=wlen)
  legend("left", legend=c("FPR: train", "FNR: train", "recall: train", "precision: trian", 
                          "FPR: test", "FNR: test", "recall: test", "precision: test"), 
         lty=c(rep(1, 4), rep(2, 4)), pch=c(rep(1, 4), rep(5, 4)), lwd=3, col=rep(1:4, 2), cex=1.5)
  dev.off()
}
