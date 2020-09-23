options(max.print = .Machine$integer.max)
#Create (training) data frame with 2^16 rows and 514 columns(numbers, mu, lagmu from -2^8 to 2^8)
install.packages("numbers")
library(numbers)
numbers<- 2^16:(2^17-1) 
extened_n<- (2^16-2^8): (2^17-1+2^8)
mu_v<- sapply(extened_n, moebius)
mu<- sapply(numbers, moebius)
dat<- data.frame(numbers=numbers, mu=as.factor(mu))
#dat contains numbers and corresponding mu values.


#Create lagged values
mu_backward<- matrix(rep(0, (2^16)*2^8), nrow=2^16, ncol = 2^8)
for (i in 1: 2^8){
  mu_backward[, i]<- mu_v[(2^8-(i-1)): (2^8-(i-1)+2^16-1)]
}
mu_forward<- matrix(rep(0, (2^16)*2^8), nrow=2^16, ncol = 2^8)
for (i in 1:2^8){
  mu_forward[, i]<- mu_v[(2^8+(i+1)): (2^8+(i+1)+2^16-1)]
}
#X1-X256 for backward, X257-X512 for forward
lagmu<- cbind(mu_backward, mu_forward)
lagmu_factor<- apply(lagmu, 2, as.factor)#all mu values have been changed to factors
dat_1<- data.frame(dat, lagmu_factor)
#dat_1 has 514 variables: numbers, mu values, backward lagged mu, forward lagged mu.


#Creating test data set
test_numbers<- seq(numbers[1], numbers[65536], by=2^8+1)
test_mu<- sapply(test_numbers, moebius)
test_dat<- data.frame(numbers=test_numbers, mu=as.factor(test_mu))
extended_test_n<- list()
for (i in 1: length(test_numbers)){
  extended_test_n[[i]]<- (test_numbers[i]-2^8) : (test_numbers[i] + 2^8)
}
extended_test_mu<- sapply(unlist(extended_test_n), moebius)

mu_matrix<- matrix(extended_test_mu, nrow = 513, ncol = 256)
a<- matrix(rep(0, 513*256), nrow=513, ncol=256)
for (j in 1:256){
for (i in 1:256){
  a[i, j]<- mu_matrix[257-i, j]
}
}
for (j in 1:256){
for (i in 257: 513){
  a[i, j]<- mu_matrix[i, j]
}
}
lagged_mu_test<- t(a)[, -257]
f_mu<- apply(lagged_mu_test, 2, as.factor)
#The first entry is 2^16, which is in the training set as well.
test_dat_1<- data.frame(test_dat, f_mu)

#NaiveBayes with categorical (without encoding) inputs.
#library(e1071)
#Naive_Bayes_Model<- naiveBayes(mu~., data=dat_1)
#Naive_Bayes_Model
#NB_Predictions<- predict(Naive_Bayes_Model, test_dat_1)
#table(NB_Predictions, test_dat_1$mu)


#Preprocess: encoding one-hot vector
install.packages("onehot")
library(onehot)
encoder<-onehot(dat_1)
x<- predict(encoder, dat_1)
colnames(x)<- NULL
train_dat<- as.data.frame(x)
#training data with one-hot transfered
names(train_dat)[1]<- "numbers"
encoder_test<- onehot(test_dat_1)
y<- predict(encoder_test, test_dat_1)
colnames(y)<- NULL
test_dat_2<- as.data.frame(y)
#test data with one-hot transfered
names(test_dat_2)[1]<- "numbers"

#Adding one column of single prime factor to training and testing data
p_list<- lapply(train_dat$numbers, primeFactors)
s<- lapply(p_list, function(x) max(x))
s1<- unlist(s)
train_dat_1<- data.frame(train_dat, s1)
#ready-to-go training data

p_list_test<- lapply(test_dat_2$numbers, primeFactors)
s_t<- lapply(p_list_test, function(x) max(x))
s1_t<- unlist(s_t)
test_dat_3<- data.frame(test_dat_2, s1_t)
#ready-to-go testing data 

#Neural Network by neuralnet package
install.packages("neralnet")
install.packages("nnet")
library(neuralnet)
library(nnet)
set.seed(10)
n<- names(train_dat_1)
f<- as.formula(paste("V2+V3+V4~", paste(n[!n %in% c("V2", "V3", "V4")], collapse = "+")))
nn<- neuralnet(f, data=train_dat_1, hidden=c(771, 300, 50), act.fct = "logistic", linear.output = FALSE, lifesign = "minimal")

#Check the accuracy on the training set
pr.nn<- compute(nn, train_dat[, -c(2:4)])
pr.nn_<- pr.nn$net.result
head(pr.nn_)
original_values<- max.col(train_dat[, 2:4])
pr.nn_2<- max.col(pr.nn_)
mean(pr.nn_2 == original_values)

#5 fold cross validation
set.seed(500)
k<- 5
outs<- NULL
proportion<- 0.8
for(i in 1:k)
{
  index <- sample(1:nrow(train_dat), round(proportion*nrow(train_dat)))
  train_cv <- train_dat[index, ]
  test_cv <- train_dat[-index, ]
  nn_cv <- neuralnet(f,
                     data = train_dat,
                     hidden = c(771, 500, 100),
                     act.fct = "logistic",
                     linear.output = FALSE)
 # Compute predictions
  pr.nn <- compute(nn_cv, test_cv[, -c(2:4)])
  # Extract results
  pr.nn_ <- pr.nn$net.result
  # Accuracy (test set)
  original_values <- max.col(test_cv[, 2:4])
  pr.nn_2 <- max.col(pr.nn_)
  outs[i] <- mean(pr.nn_2 == original_values)
}

mean(outs)
  
  
  
  
  
pr.nn_test <- compute(nn, test_dat_2[, -c(2:4)])
# Extract results 
pr.nn_test_ <- pr.nn_test$net.result
# Accuracy (test set)
original_values_test <- max.col(test_dat_2[, 2:4])
pr.nn_2_test <- max.col(pr.nn_test_)
mean(pr.nn_2_test == original_values_test)


#SVM(need to be fixed)
library(e1071)
set.seed(1)
svmfit=svm(V2+V3+V4~., data=train_dat, kernel="radial", gamma=50, cost=1)
tune.out=tune(svm , f, data=train_dat, kernel="radial", ranges=list(cost=c(0.1,1,10,100,1000), gamma=c(0.5,1,2,3,4))) 
summary (tune.out) 
#Prediction Performance
table(true=dat[-train ,"y"], pred=predict (tune.out$best.model, newdata=dat[-train,]))

