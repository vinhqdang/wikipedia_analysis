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

runRFModel = function (feature_list)
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
  rf_h2o = h2o.randomForest(x = feature_list, y = 1, training_frame = data, ntrees = 450, nfolds = 5)
  print (rf_h2o)
  rf_h2o
  # h2o.shutdown(prompt = FALSE)
}

runRFModel_withReadabilityScores <- function ()
{
  runRFModel (2:21)
}

runRFModel_EliminateFeature = function ()
{
  for (i in 1:21) {
    print (paste ("Run RF model with eliminating variable ",i))
    feature_list = c(2:21)
    feature_list = feature_list[!feature_list == i]
    rf = runRFModel (feature_list = feature_list)
    write (paste ("Eliminate variable number",i, ":", 
                  rf@model$cross_validation_metrics@metrics$hit_ratio_table$hit_ratio[1],"\n"), 
           file = "RFModel_elminateFeature.txt",append = TRUE)
  }
}

runRFModel_addReadScoreOneByOne = function ()
{
  for (i in 3:9) {
    print (paste ("Run RF model with adding readability score number ",i))
    feature_list = c(2:(12+i))
    rf = runRFModel (feature_list = feature_list)
    write (paste ("adding readability score",i, ":", 
                  rf@model$cross_validation_metrics@metrics$hit_ratio_table$hit_ratio[1],"\n"), 
           file = "RFModel_elminateFeature.txt",append = TRUE)
  }
}

runRFModel_addSingleReadScore = function ()
{
  for (i in 8:9) 
  {
    print (paste ("Run RF model with adding only readability score number ",i))
    feature_list = c(2:12,(12+i))
    rf = runRFModel (feature_list = feature_list)
    write (paste ("adding only readability score",i, ":", 
                  rf@model$cross_validation_metrics@metrics$hit_ratio_table$hit_ratio[1],"\n"), 
           file = "RFModel_elminateFeature.txt",append = TRUE)
  }
}

runRFModel_withoutReadabilityScore = function()
{
  runRFModel (2:12)
}

plot_addingReadScoreOneByOne = function () {
  par(mar=c(12,5,1,1))
  y=c(0.578,0.58629717,0.5903492,0.59410947,0.59857476,0.6057016,0.62106816,0.6289228,0.63416795,0.63746294)
  x_labels = c("Base feature set","flesch_reading_ease","flesch_kincaid_grade","smog_index","coleman_liau_index","automated_readability_index",
            "difficult_words","dale_chall_readability_score","linsear_write_formula","gunning_fog")
  plot (y, xaxt = "n",type = "b", xlab = "",ylab="Accuracy",las=2, ylim = c(0.57,0.64))
  # axis(1, at=1:10, labels=x_labels,las=2)
  
  axis(1, at=seq(1, 10, by=1), labels = FALSE)
  text(seq(1, 10, by=1), par("usr")[3] - 0.2, labels = x_labels, srt = -45, pos = 1, xpd = TRUE)
}

plot_addingSingleReadScore = function () {
  par(mar=c(12,5,1,1))
  y=c(0.578,0.5826866,0.58890134,0.58743715,0.58729076,0.58773,0.58909445,0.588679,0.5867051,0.58616817)
  x_labels = c("Base feature set","flesch_reading_ease","flesch_kincaid_grade","smog_index","coleman_liau_index","automated_readability_index",
               "difficult_words","dale_chall_readability_score","linsear_write_formula","gunning_fog")
  plot (y, xaxt = "n",type = "p", xlab = "",ylab="Accuracy",las=2, ylim = c(0.57,0.60))
  axis(1, at=1:10, labels=x_labels,las=2)
  # axis(2, at=c(1,3,5,7),labels = c(0.57,0.58,0.59,0.60),las=2)
  
  for (i in 1:11) {
    epsilon = 0.2
    segments(x0 = i - epsilon, x1 = i + epsilon, y0 = y[i], y1 = y[i])
    segments(x0 = i-epsilon,x1=i-epsilon, y0 = 0.57, y1=y[i])
    segments(x0 = i+epsilon,x1=i+epsilon, y0 = 0.57, y1=y[i])
  }
  
}

runKNNModel <- function () {
  library (class)
  
  set.seed(2015)
  all_data <- read.csv("all_data.csv", header =  FALSE)
  all_data$V1 = NULL
  all_data$V2 = NULL
  all_data$V13 = NULL
  all_data$V22 = as.numeric (all_data$V22)
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

# rf_model: need to be a h2o randomForest model
# test_data: a h2o dataframe
calc_ndcg_random_forest <- function(rf_model, test_data) {
  predict1 <- predict(rf_model, newdata = as.h2o(test_data), type = "vote")[,2:7]
  colnames(predict1) <- c(3,2,6,4,1,0)
  actual_classes <- c(3,2,6,4,1,0)[test_data$V3]
  
  predict_data <- data.frame("predict_class" = numeric(),
                             "predict_score" = numeric(),
                             "actual_class" = numeric())
  
  for (i in 1:nrow(test_data)) {
    print(paste("Processing element",i))
    v = as.vector (predict1[i,])
    predict_class = as.numeric (c(3,2,6,4,1,0)[which.max (as.vector (predict1[i,]))])
    predict_score = max(predict1[i,])
    actual_class = actual_classes [i]
    
    predict_data [nrow(predict_data) + 1,] <- c (predict_class, predict_score, actual_class)
  }
  sorted_predict_data <- predict_data[with(predict_data, order(-predict_class, -predict_score)),]
  ndcg (sorted_predict_data$actual_class)
  
  ndcg_scores = c()
  for (i in 1:nrow(test_data)) {
    print(paste("Processing element",i))
    ndcg_score = ndcg (sorted_predict_data$actual_class[1:i])
    ndcg_scores = c(ndcg_scores,ndcg_score)
  }
  ndcg_scores
}

feature_select_rf <- function (){
  set.seed(2015)
  all_data <-  read.csv ("all_data.csv", header = FALSE)
  
  all_data$V1 = NULL
  all_data$V2 = NULL
  all_data$V13 = NULL
  require(caTools)
  sample = sample.split(all_data$V3, SplitRatio = 0.8)
  
  
  colnames(all_data) = c("quality_class","content_length","num_references","num_page_links","num_cite_templates",
                         "num_non_cite_templates","num_categories","num_images_length","info_noise_score",
                         "has_infobox","num_lv2_headings","num_lv3_headings",
                         "flesch_reading_ease","flesch_kincaid_grade","smog_index",
                         "coleman_liau_index","automated_readability_index",
                         "difficult_words","dale_chall_readability_score","linsear_write_formula",
                         "gunning_fog")
  
  train = subset (all_data, sample==TRUE)
  test = subset (all_data, sample == FALSE)
  
  library (caret)
  model = train (quality_class ~ content_length + num_references + num_page_links 
                 + num_cite_templates + num_non_cite_templates
                 + num_categories + num_images_length + info_noise_score + has_infobox + num_lv2_headings
                 + num_lv3_headings
                 + flesch_reading_ease + flesch_kincaid_grade + smog_index
                 + coleman_liau_index + automated_readability_index + difficult_words
                 + dale_chall_readability_score + linsear_write_formula + gunning_fog, 
                 data = train, method = "lvq", preProcess="scale")
  
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