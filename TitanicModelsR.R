
dataset <- read.csv("train.csv")

dataset <- dataset[-c(1,4,9,11)]

dataset$Age = ifelse(is.na(dataset$Age),
                     ave(dataset$Age, FUN = function(x) mean(x, na.rm = TRUE)),
                     dataset$Age)

dataset$Sex <- factor(dataset$Sex,levels = c("female","male"),labels =c(0,1) )
dataset$Embarked <- factor(dataset$Embarked,levels = c("C","Q","S"),labels = c(1,2,3))

dataset$Sex <- as.numeric(dataset$Sex)
dataset$Embarked <-as.numeric(dataset$Embarked)
library(caTools)
set.seed(12)
#split=sample.split(dataset$Survived,SplitRatio = 2/3)
#training <- subset(dataset,split==TRUE)
#test <- subset(dataset,split==FALSE)
test <- read.csv("test.csv")
x<- read.csv("test.csv")
test <- test[-c(1,3,8,10)]
test$Age = ifelse(is.na(test$Age),
                     ave(test$Age, FUN = function(x) mean(x, na.rm = TRUE)),
                     test$Age)

test$Sex <- factor(test$Sex,levels = c("female","male"),labels =c(0,1) )
test$Embarked <- factor(test$Embarked,levels = c("C","Q","S"),labels = c(1,2,3))
test$Sex <- as.numeric(test$Sex)
test$Embarked <-as.numeric(test$Embarked)


#feature scaling
dataset[-1]<- scale(dataset[-1])
testset <- scale(testset)

classifier <- glm(formula = Survived ~ ., family = "binomial",data = dataset)
testset <- as.data.frame(testset)

y_pred <- predict(classifier,type="response",newdata=testset)
y <- ifelse(y_pred>0.5,1,0)
y <- as.data.frame(y)
new <- cbind(x$PassengerId,y)

write.csv(new,"result.csv", row.names = FALSE,col.names = c("PassengerID","Survived"))

