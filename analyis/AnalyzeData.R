runCART <-function () {
  set.seed(2016)
  
  all_data = read.csv ("all_data.csv", header = FALSE)
  all_data$V1 = NULL
  all_data$V2 = NULL
  
  library(caret)
  tc <- trainControl("cv",5)
  train.rpart <- train(V3 ~ V4 + V5 + V6+ V7 + V8 + V9 + V10 + V11 + V12 + V14 + V15 + V16 + V17+ V18 + V19 + V20 + V21 + V22 + V23 + V24, data = all_data, method="rpart",trControl=tc)
  print (train.rpart)
  p.rpart = predict(train.rpart, newdata = all_data)
  print (multiclass.roc(as.ordered(all_data$V3), as.ordered(p.rpart)))
}

runRegression <- function () {
  set.seed(2015)
  all_data = read.csv ("all_data.csv", header = FALSE)
  all_data$V1 = NULL
  all_data$V2 = NULL
  require(caTools)
  sample = sample.split(all_data$V3, SplitRatio = 0.8)
  
  train = subset (all_data, sample==TRUE)
  test = subset (all_data, sample == FALSE)
  
  train$V3 <- as.numeric (as.character (factor (train$V3, labels = c(4,3,6,5,2,1))))
  test$V3 <- as.numeric (as.character (factor (test$V3, labels = c(4,3,6,5,2,1))))
  lm1 <- lm(V3 ~ V4 + V5 + V6+ V7 + V8 + V9 + V10 + V11 + V12 + V14 + V15 + V16 + V17+ V18 + V19 + V20 + V21 + V22 + V23 + V24, data = train)
  predictLM <- predict(lm1, newdata = test)
  predictLM <- round (predictLM)
  predictLM = replace(predictLM, predictLM >= 7, 6)
  predictLM = replace(predictLM, predictLM < 1, 1)
  table1 <- table (test$V3, predictLM)
  print ("Confusion matrix")
  print (table1)
  print (paste("Accuracy of linear regression is:", sum (diag(table1)) / sum (table1)))
}

runMultinominalLogisticRegression = function ()
{
  
  library(caret)
  library (nnet)
  set.seed(2015)
  all_data = read.csv ("all_data.csv", header = FALSE)
  all_data$V1 = NULL
  all_data$V2 = NULL
  tc <- trainControl("cv",5)
  
  train.multinom = train(V3 ~ V4 + V5 + V6+ V7 + V8 + V9 + V10 + V11 + V12 + V14 + V15 + V16 + V17+ V18 + V19 + V20 + V21 + V22 + V23 + V24, data = all_data, method="multinom",trControl=tc)
  
  print (train.multinom)
  
  p.multinom = predict(train.multinom, newdata = all_data)
  library (pROC)
  print (multiclass.roc(as.ordered(all_data$V3), as.ordered(p.multinom)))
}

runSVM = function ()
{
  set.seed(2015)
  all_data = read.csv ("all_data.csv", header = FALSE)
  all_data$V1 = NULL
  all_data$V2 = NULL
  all_data$V13 = NULL
  library (e1071)
  library (caret)
  tc <- trainControl("cv",5)
  
  train.svm = train(V3 ~ V4 + V5 + V6+ V7 + V8 + V9 + V10 + V11 + V12 + V14 + V15 + V16 + V17+ V18 + V19 + V20 + V21 + V22 + V23 + V24, data = all_data, method="svmLinear",trControl=tc)
  
  print (train.svm)
  
  p.svm = predict (train.svm, newdata = all_data)
  print (multiclass.roc(as.ordered(all_data$V3), as.ordered(p.svm)))
}

runRFModel <- function ()
{
  if (!require(h2o)) {
    install.packages("h2o")
  }
  library(h2o)
  
  set.seed(2015)
  all_data <- read.csv("all_data.csv", header = FALSE)
  all_data$V1 = NULL
  all_data$V2 = NULL
  all_data$V13 = NULL
  
  localH2O <- h2o.init()
  
  data = as.h2o (all_data)
  rf_h2o = h2o.randomForest(x = 2:21, y = 1, training_frame = data, ntrees = 450, nfolds = 5)
  print (rf_h2o)
  
  h2o.shutdown(prompt = FALSE)
}

runRFModel_withoutReadabilityScore = function()
{
  if (!require(h2o)) {
    install.packages("h2o")
  }
  library(h2o)
  
  localH20 = h2o.init ()
  
  all_data <- read.csv("all_data.csv", header = FALSE)
  all_data$V1 = NULL
  all_data$V2 = NULL
  all_data$V13 = NULL
  
  localH2O <- h2o.init(ip = "localhost", port = 54321, startH2O = TRUE)
  
  data = as.h2o (all_data)
  rf_h2o = h2o.randomForest(x = 2:12, y = 1, training_frame = data, ntrees = 501, nfolds = 5)
  print (rf_h2o)
  
  h2o.shutdown(prompt = FALSE)
  
  # HandTill2001::auc(multcap(response = test$V3, predicted = as.matrix(p_sepa[,2:7])))
}

runKNNModel <- function () {
  library (class)
  
  set.seed(2015)
  all_data <- read.csv("all_data.csv", header =  FALSE)
  all_data$V1 = NULL
  all_data$V2 = NULL
  all_data$V13 = NULL
  all_data$V22 = as.numeric (all_data$V22)
  library(class)
  knn.cv = knn.cv(train = all_data[,2:20], cl = all_data$V3, k = 101)
  t = table (all_data$V3, knn.cv)
  print (sum (diag(t))/sum(t))
  print (multiclass.roc(as.ordered(all_data$V3), as.ordered(knn.cv)))
}

tuning <- function (train_data)
{
  rf_model <- train (user_rating ~ flesch_reading_ease + flesch_kincaid_grade + smog_index + coleman_liau_index + automated_readability_index + dale_chall_readability_score + difficult_words + linsear_write_formula + gunning_fog + infonoisescore + logcontentlength + logreferences + logpagelinks + numimageslength + num_citetemplates + lognoncitetemplates + num_categories + hasinfobox + lvl2headings + lvl3heading, data = train_data, method = "rf", trControl = trainControl(method="cv", number = 5), prox = TRUE, allowParallel = TRUE)
}

calc_rmse <- function (test_data, predict_data) 
{
  sqrt (mean ((test_data - predict_data) ^ 2))
}


ndcg <- function(x) {
  # x is a vector of relevance scores
  ideal_x <- rev(sort(x))
  DCG <- function(y) y[1] + sum(y[-1]/log(2:length(y), base = 2))
  DCG(x)/DCG(ideal_x)
}

ndcg_matrix <- function (predict_dataframe) {
  require (gbm)
  res <- c()
  scores <- c(6,4,3,2,1,0)
  
  for (i in 1:6) {
    col <- predict_dataframe[,i]
    for (j in 1:6) {
      res <- c(res, rep(scores[j], col[j]))
    }
  }
  print (ndcg(res))
}

calc_ndcg_random_forest <- function(rf_model, test_data) {
  predict1 <- predict(rf_model, newdata = test_data, type = "vote")
  colnames(predict1) <- c(3,2,6,4,1,0)
  actual_classes <- c(3,2,6,4,1,0)[test_data$user_rating]
  
  predict_data <- data.frame("predict_class" = as.numeric(),
                             "predict_score" = as.numeric(),
                             "actual_class" = as.numeric())
  
  for (i in 1:nrow(test_data)) {
    predict_class = as.numeric (names (which.max (predict1[i,])))
    predict_score = max(predict1[i,])
    actual_class = actual_classes [i]
    
    predict_data [nrow(predict_data) + 1,] <- c (predict_class, predict_score, actual_class)
  }
  sorted_predict_data <- predict_data[with(predict_data, order(-predict_class, -predict_score)),]
  ndcg (sorted_predict_data$actual_class)
}

feature_select_rf <- function (){
  set.seed(2015)
  train_data <- read.csv ("all_score_train.csv")
  test_data <- read.csv ("all_score_test.csv")
  all_data <- rbind (train_data, test_data)
  require(caTools)
  sample = sample.split(all_data$revid, SplitRatio = 0.8)
  train = subset (all_data, sample==TRUE)
  test = subset (all_data, sample == FALSE)
  
  #Importance of feature
  train$readability_consensus <- NULL
  test$readability_consensus <- NULL
  
  train$pageid <- NULL
  test$pageid <- NULL
  
  train$revid <- NULL
  test$revid <- NULL
  
  model = train (user_rating ~ flesch_reading_ease + flesch_kincaid_grade + smog_index + coleman_liau_index + automated_readability_index + dale_chall_readability_score + difficult_words + linsear_write_formula + gunning_fog + infonoisescore + logcontentlength + logreferences + logpagelinks + numimageslength + num_citetemplates + lognoncitetemplates + num_categories + hasinfobox + lvl2headings + lvl3heading, data = train, method = "lvq", preProcess="scale")
  
  importance <- varImp(model, scale=FALSE)
  
  plot (importance)
}

# convert factor to integer
# df$rating_score = c(1,2,3,4,6,7)[as.numeric(df$user_rating)]

# p1_ores = as.numeric(as.character(factor(as.vector(p.ores[,1]), labels = c(4,3,6,5,2,1))))
# p2_ores = as.numeric(as.character(factor(as.vector(test_ores$V25), labels = c(4,3,6,5,2,1))))
# p1_h2o = as.numeric(as.character(factor(as.vector(p.h2o[,1]), labels = c(4,3,6,5,2,1))))
# p2_h2o = as.numeric(as.character(factor(as.vector(as.data.frame(test_h2o)$V3), labels = c(4,3,6,5,2,1))))

# h2_true = 0
# hw_true = 0
# hh_true = 0
# h2_false = 0
# for (i in 1:4098) {
#   if (p3_h2o[i] == correct[i] & p3_w[i] == correct[i]) {
#     h2_true = h2_true + 1
#   }
#   else if (p3_h2o[i] != correct[i] & p3_w[i] == correct[i]) {
#     hw_true = hw_true + 1
#   }
#   else if (p3_h2o[i] == correct[i] & p3_w[i] != correct[i]) {
#     hh_true = hh_true + 1
#   }
#   else if (p3_h2o[i] != correct[i] & p3_w[i] != correct[i]) {
#     h2_false = h2_false + 1
#   }
# }