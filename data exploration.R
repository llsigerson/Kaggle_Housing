#import data
train<- read.csv("train.csv")
test<- read.csv("test.csv")

#run simple linear regressions for each predictor to see how much variance
#in the outcome they predict
r_squared_vals= numeric(80)
for(i in 1:80){
  print(i)
  r_squared_vals[i]= summary(lm(train$SalePrice~train[,i]))$r.squared
}

#print out variance explained for each variable, ordered, so we can see which variables matter most
results=cbind(round(r_squared_vals,2), names(train)[1:80], sapply(train, class)[1:80])
results[order(results[,1], decreasing = T),]
