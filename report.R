
###read data
telecom=read.csv(file.choose(),header=T)
attach(telecom)

###skim the data
library("skimr")
skim(telecom)

###data visulization

##numerical data
DataExplorer::plot_boxplot(telecom, by = "Churn", ncol = 3)
##categorical data
DataExplorer::plot_bar(telecom, by = "Churn", ncol = 2)
##pairs
library("GGally")
ggpairs(telecom %>% select(SeniorCitizen, tenure, MonthlyCharges, TotalCharges),
        aes(color = Churn))

###pre-processing the data
telecom_data <- telecom %>% 
  select(-gender,-PhoneService, -MultipleLines,-TotalCharges)

a=which(telecom_data$StreamingTV=='No internet service')
telecom_data$OnlineSecurity[a]='No'
telecom_data$OnlineBackup[a]='No'
telecom_data$DeviceProtection[a]='No'
telecom_data$TechSupport[a]='No'
telecom_data$StreamingTV[a]='No'
telecom_data$StreamingMovies[a]='No'

###MLR 3 - select model

##preprocessing data for MLR 3
factor_name<-c("Partner","Dependents","PhoneService","MultipleLines",
               "InternetService","OnlineSecurity","OnlineBackup", "DeviceProtection","TechSupport",
               "StreamingTV","StreamingMovies","Contract","PaperlessBilling","PaymentMethod","Churn")
idx <- which(names(telecom_data)   %in% factor_name)
for(i in idx ){
  telecom_data[,i]  <-  as.factor(telecom_data[,i])
}

##Load package
library("mlr3learners")
library("mlr3proba")
library("data.table")
library("mlr3verse")

##5-folds resampling
set.seed(212) # set seed for reproducibility
tele_task=TaskClassif$new(id = "telecom",
                          backend = telecom_data, 
                          target = "Churn",
                          positive = "Yes")

cv5 <- rsmp("cv", folds = 5)
cv5$instantiate(tele_task)

##determine the value of cp
lrn_cart_cv <- lrn("classif.rpart", predict_type = "prob", xval = 10)

res_cart_cv <- resample(tele_task, lrn_cart_cv, cv5, store_models = TRUE)
rpart::plotcp(res_cart_cv$learners[[4]]$model)

##different learners
lrn_baseline <- lrn("classif.featureless", predict_type = "prob")
lrn_cart <- lrn("classif.rpart", predict_type = "prob")
lrn_cart_cp  <- lrn("classif.rpart", predict_type = "prob", cp = 0.025, id = "cartcp")
lrn_ranger   <- lrn("classif.ranger", predict_type = "prob")
lrn_xgboost  <- lrn("classif.xgboost", predict_type = "prob")

## Dealing with missingness and factors
pl_missing <- po("fixfactors") %>>%
  po("removeconstants") %>>%
  po("imputesample", affect_columns = selector_type(c("ordered", "factor"))) %>>%
  po("imputemean")

lrn_xgboost <- lrn("classif.xgboost", predict_type = "prob")
pl_xgb <- po("encode") %>>%
  po(lrn_xgboost)

lrn_log_reg <- lrn("classif.log_reg", predict_type = "prob")
pl_log_reg <- pl_missing %>>%
  po(lrn_log_reg)

##comparison among different models
tele_res <- benchmark(data.table(
  task       = list(tele_task),
  learner    = list(lrn_baseline,
                    lrn_cart,
                    lrn_cart_cp,
                    lrn_ranger,
                    pl_xgb,
                    pl_log_reg),
  resampling = list(cv5)
), store_models = TRUE)

tele_res$aggregate(list(msr("classif.ce"),
                        msr("classif.acc"),
                        msr("classif.fpr"),
                        msr("classif.fnr")))


###improve the model-logistic regression
set.seed(212)
search_space = ps(
  epsilon = p_dbl(lower = 1e-20, upper = 1e-5),
  maxit = p_int(lower = 1, upper = 50)
)
search_space
measures=msr("classif.acc")
library("mlr3tuning")

evals30 = trm("evals", n_evals = 30) 

instance = TuningInstanceMultiCrit$new(
  task = tele_task,
  learner = lrn_log_reg ,
  resampling = cv5,
  measures = measures,
  search_space = search_space,
  terminator = evals30
)
instance

tuner = tnr("grid_search", resolution = 5)
tuner$optimize(instance)
instance$result_y
instance$result_learner_param_vals


###improve the model-random forest
set.seed(212)
search_space1 = ps(
  mtry = p_int(lower = 1, upper = 12),
  num.trees = p_int(lower = 1, upper = 1000)
  #  min.node.size = p_int(lower = 1, upper = 10)
)
search_space1
#measures = msrs(c("classif.acc", "time_train"))
measures=msr("classif.acc")
library("mlr3tuning")

evals30 = trm("evals", n_evals = 30) 

instance1 = TuningInstanceMultiCrit$new(
  task = tele_task,
  learner = lrn_ranger ,
  resampling = cv5,
  measures = measures,
  search_space = search_space1,
  terminator = evals30
)
instance1


tuner1 = tnr("grid_search", resolution = 5)
tuner1$optimize(instance1)
instance1$result_y
instance1$result_learner_param_vals


####we choose random forest as our final model

###improved random forest-finalized model
##split the data
set.seed(212)
train_set = sample(tele_task$row_ids, 0.8 * tele_task$nrow)
test_set = setdiff(tele_task$row_ids, train_set)

learner_rf = lrn("classif.ranger", importance = "permutation",mtry=2,num.trees=808,predict_type="prob")
learner_rf$train(tele_task,row_ids = train_set)
learner_rf$importance()

##plot the importance
importance = as.data.table(learner_rf$importance(), keep.rownames = TRUE)
colnames(importance) = c("Feature", "Importance")

ggplot(data=importance,
       aes(x = reorder(Feature, Importance), y = Importance)) + 
  geom_col() + coord_flip() + xlab("")

##prediction and confusion matrix
pred_rf = learner_rf$predict(tele_task, row_ids = test_set)
pred_rf$confusion
pred_rf$confusion/rowSums(pred_rf$confusion)*100

#set.seed(212)
resampling = rsmp("holdout", ratio = 2/3)
print(resampling)
res = resample(tele_task, learner = learner_rf, resampling = resampling)
res$aggregate(msr("classif.acc"))

##roc plot
install.packages("precrec")
pred_rf_prob = learner_rf$predict(tele_task, row_ids = test_set)
library(ggplot2)
autoplot(pred_rf_prob,type="roc")



