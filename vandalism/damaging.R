damage = read.table("enwiki.features_damaging.20k_2015.tsv")

# remove revision ID
damage$V1 = NULL

# library (caTools)
# 
# sample = sample.split(damage$V2, SplitRatio = 0.8)
# 
# train = subset (damage, sample == TRUE)
# test  = subset (damage, sample == FALSE)

# before SMOTE
library (h2o)
h2o.init ()

damage_h2o = as.h2o (damage)

# set column names
response <- "V68"
predictors <- setdiff(names(damage_h2o),c(response, "name"))

# split train/valid/test sets
splits <- h2o.splitFrame(
  data = damage_h2o, 
  ratios = c(0.6,0.2),   ## only need to specify 2 fractions, the 3rd is implied
  destination_frames = c("train.hex", "valid.hex", "test.hex"), seed = 1234
)
train <- splits[[1]]
valid <- splits[[2]]
test  <- splits[[3]]

# baseline model
gbm1 <- h2o.gbm(x = predictors, y = response, training_frame = train)

# AUC 
# 0.8545395
h2o.auc(h2o.performance(gbm1, newdata = valid)) 

# AUC on test
# 0.8785541
h2o.auc(h2o.performance(gbm1, newdata = test))

# 2nd model with cross-validation
gbm2 <- h2o.gbm(x = predictors, y = response, training_frame = h2o.rbind(train, valid), nfolds = 4, seed = 0xDECAF)

# details of 2nd model
gbm2@model$cross_validation_metrics_summary

# AUC of 2nd model
# 0.8699413 > 1st model
h2o.auc(h2o.performance(gbm2, xval = TRUE))

# AUC on test of 2nd mode
# 0.886495
h2o.auc(h2o.performance(gbm2, newdata = test))

# 3rd model
# early stopping
gbm3 <- h2o.gbm(
  ## standard model parameters
  x = predictors, 
  y = response, 
  training_frame = train, 
  validation_frame = valid,
  
  ## more trees is better if the learning rate is small enough 
  ## here, use "more than enough" trees - we have early stopping
  ntrees = 10000,                                                            
  
  ## smaller learning rate is better (this is a good value for most datasets, but see below for annealing)
  learn_rate=0.01,                                                         
  
  ## early stopping once the validation AUC doesn't improve by at least 0.01% for 5 consecutive scoring events
  stopping_rounds = 5, stopping_tolerance = 1e-4, stopping_metric = "AUC", 
  
  ## sample 80% of rows per tree
  sample_rate = 0.8,                                                       
  
  ## sample 80% of columns per split
  col_sample_rate = 0.8,                                                   
  
  ## fix a random number generator seed for reproducibility
  seed = 1234,                                                             
  
  ## score every 10 trees to make early stopping reproducible (it depends on the scoring interval)
  score_tree_interval = 10                                                 
)

## Get the AUC on the validation set
# 0.8630611
h2o.auc(h2o.performance(gbm3, valid = TRUE))


h2o.shutdown(prompt = FALSE)

# for SMOTE
library (DMwR)

smote_train = SMOTE (V68 ~ ., train, perc.over = 10000, perc.under = 20000)