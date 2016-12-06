if (!require("pacman")) install.packages("pacman")
pacman::p_load(caret, caretEnsemble)

all_data = read.table ("enwiki.features_wp10.30k.tsv")

inTrain <- createDataPartition(y = all_data$V25, p = .8, list = FALSE)

training <- all_data[inTrain,]
testing <- all_data [-inTrain,]

my_control <- trainControl(
  method="boot",
  number=25,
  savePredictions="final",
  classProbs=TRUE,
  index=createResample(training$V25, 25),
  summaryFunction=multiClassSummary
)

model_list <- caretList(
  V25~., data=training,
  trControl=my_control,
  methodList=c("rpart","knn")
)

models <- caretList(
  x = training[1:nrow(training),1:24],
  y = training[1:nrow(training),25],
  trControl = trainControl(method = "boot", number=25,savePredictions = "final", classProbs = TRUE),
  methodList = c("rpart","knn")
)
caretStack(models, methods = "rpart")