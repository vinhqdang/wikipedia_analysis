# wikipedia_analysis

This repository contains my code to predict quality class of Wikipedia articles.

You should find the code in R file in **analysis** directory.

Please note that we used the random seed as 2015 in our code.

## Linear regression

The linear regression is done by calling the function *runRegression*. 

## rpart

The rpart model is done by calling the function *runRpart*. 

## kNN

The function for kNN model is *runKNNModel*.

## random forest

We provided two functions for *randomForest* model.

The first function is ``runRFModel``, which will load and run the data with readability scores using k-fold (with k = 5)

The second function is ``runRFModel_withoutReadabilityScore``, which will run without using readability scores, as in [1].

You should observe that the first function provide a better prediction, as we claimed in our submitted paper to ESWC 2016.

## Utilities

We provided some other utility functions such as calculate RMSE or NDCG.

[1] Warncke-Wang, M., Ayukaev, V.R., Hecht, B. and Terveen, L.G., 2015, February. The Success and Failure of Quality Improvement Projects in Peer Production Communities. In Proceedings of the 18th ACM Conference on Computer Supported Cooperative Work & Social Computing (pp. 743-756). ACM.


