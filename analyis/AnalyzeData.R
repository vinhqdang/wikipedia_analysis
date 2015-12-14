runRpart <-function (train, test) {
  require(rpart)
  r1 <- rpart(user_rating ~ flesch_reading_ease + flesch_kincaid_grade + smog_index + coleman_liau_index + automated_readability_index + dale_chall_readability_score + difficult_words + linsear_write_formula + gunning_fog + infonoisescore + logcontentlength + logreferences + logpagelinks + numimageslength + num_citetemplates + lognoncitetemplates + num_categories + hasinfobox + lvl2headings + lvl3heading, data = train, method = "class")
  predictR <- predict(r1, newdata = test, type = "class")
  table1 <- table(test$user_rating, predictR)
  print (table1)
  print (sum (diag(table1)) / sum (table1))
  
}

runRegression <- function (train, test) {
  train$user_rating <- as.numeric(train$user_rating, levels(c(3,4,1,2,5,6)))
  test$user_rating <- as.numeric (test$user_rating, levels(c(3,4,1,2,5,6)))
  lm1 <- lm(user_rating ~ flesch_reading_ease + flesch_kincaid_grade + smog_index + coleman_liau_index + automated_readability_index + dale_chall_readability_score + difficult_words + linsear_write_formula + gunning_fog + infonoisescore + logcontentlength + logreferences + logpagelinks + numimageslength + num_citetemplates + lognoncitetemplates + num_categories + hasinfobox + lvl2headings + lvl3heading, data = train)
  predictLM <- predict(lm1, newdata = test)
  predictLM <- round (predictLM)
  replace(predictLM, predictLM == 7, 6)
  table1 <- table (test$user_rating, predictLM)
  print (table1)
  print (sum (diag(table1)) / sum (table1))
}

runSingleRandomForest <- function (train_data, test_data, ntree, nodesize) {
  qualityForest3 = randomForest(user_rating ~ flesch_reading_ease + flesch_kincaid_grade + smog_index + coleman_liau_index + automated_readability_index + dale_chall_readability_score + difficult_words + linsear_write_formula + gunning_fog + infonoisescore + logcontentlength + logreferences + logpagelinks + numimageslength + num_citetemplates + lognoncitetemplates + num_categories + hasinfobox + lvl2headings + lvl3heading, data = train_data, ntree = ntree, nodesize = nodesize, mtry = 2)
  predictForest3 = predict(qualityForest3, newdata = test_data)
  table3 = table (test_data$user_rating, predictForest3)
  sum (diag(table3)) / sum (table3)
}

runRFModel <- function ()
{
  set.seed(2015)
  train_data <- read.csv ("all_score_train.csv")
  test_data <- read.csv ("all_score_test.csv")
  all_data <- rbind (train_data, test_data)
  require(caTools)
  sample = sample.split(all_data$revid, SplitRatio = 0.8)
  train = subset (all_data, sample==TRUE)
  test = subset (all_data, sample == FALSE)
  runMultiRandomForest (train, test)
}

runKNNModel <- function () {
  set.seed(2015)
  train_data <- read.csv ("all_score_train.csv")
  test_data <- read.csv ("all_score_test.csv")
  all_data <- rbind (train_data, test_data)
  require(caTools)
  sample = sample.split(all_data$revid, SplitRatio = 0.8)
  
  train = subset (all_data, sample==TRUE)
  test = subset (all_data, sample == FALSE)
  
  train_rate <- train$user_rating
  test_rate <- test$user_rating
  
  train$user_rating <- NULL
  train$pageid <- NULL
  train$revid <- NULL
  train$readability_consensus <- NULL
  train$hasinfobox <- as.numeric(train$hasinfobox)
  train$sample <- NULL
  
  test$user_rating <- NULL
  test$pageid <- NULL
  test$revid <- NULL
  test$readability_consensus <- NULL
  test$hasinfobox <- as.numeric(test$hasinfobox)
  test$sample <- NULL
  
  max_correction = 0
  for (k in 1:100) {
    model = knn(train = train, test = test, k = k, cl = train_rate)
    table1 <- table (model, test_rate)
    
    correct = sum (diag(table1))/sum(table1)
    
    if (correct > max_correction) {
      print (paste("Correction of k = ", k, " is: ", correct))
      print (table1)
      max_correction = correct
    }
  }
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
