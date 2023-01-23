library(xgboost)
library(ranger)
library(xgboost)
library(torch)
library(iml)
library(cito)
library(glmnet)
library(glmnetUtils)
library(MASS)
library(tree)


Sys.setenv(OMP_NUM_THREADS=5)

source("code/AME.R")
source("code/Scenarios.R")


## Is boosting unbiased?

task =   # r , X1, X2
  matrix(c(
    0.0, 1.0, 0.5,
    0.5, 1.0, 0.5,    
    0.9, 1.0, 0.5,
    0.9, 1.0, 0.0
  ), ncol = 3, byrow = TRUE)

results = 
  lapply(1:nrow(task), function(i) {
    pars = task[i,]
    cl = parallel::makeCluster(50L)
    
    parallel::clusterExport(cl, varlist = ls(envir = .GlobalEnv))
    parallel::clusterEvalQ(cl, {library(ranger);library(xgboost);library(cito);library(glmnet);library(glmnetUtils);library(tree)})
    
    results = parallel::parLapply(cl, 1:100, function(KK) {
      
      source("code/AME.R")
      
      sim = function() simulate(r = pars[1], effs = c(1, pars[3], 0, 0, 1, seq(-0.2, 0.2, length.out = 5)),n = 1000)
      data = sim()
      train = data
      
      # Boosting linear
      m = get_boosting_model(x = train[,-1], y = train[,1], n_trees = 300,eta = 0.5, booster = "linear")
      eff1 = diag(marginalEffects(m, data = data.frame(train)[,-1], interactions = FALSE, max_indices = 5)$mean)
      
      # Regression tree / no boosting / low complexity
      m = get_boosting_model(x = train[,-1], y = train[,1], n_trees = 1, minsize = 200,mincut = 100, mindev = 0.01, eta = 1.0, booster = "tree")
      eff2 = diag(marginalEffects(m, data = data.frame(train)[,-1], interactions = FALSE, max_indices = 5)$mean)
      
      # Regression tree / no boosting / high complexity
      m = get_boosting_model(x = train[,-1], y = train[,1], n_trees = 1, minsize = 2,mincut = 1, mindev = 0.000001, eta = 1.0, booster = "tree")
      eff3 = diag(marginalEffects(m, data = data.frame(train)[,-1], interactions = FALSE, max_indices = 5)$mean)
      
      
      # Boosting tree / low complexity
      m = get_boosting_model(x = train[,-1], y = train[,1], n_trees = 300, minsize = 200,mincut = 100, mindev = 0.01, eta = 0.5, booster = "tree")
      eff4 = diag(marginalEffects(m, data = data.frame(train)[,-1], interactions = FALSE, max_indices = 5)$mean)
      
      # Boosting tree / no boosting / high complexity
      m = get_boosting_model(x = train[,-1], y = train[,1], n_trees = 300, minsize = 2,mincut = 1, mindev = 0.01, eta = 0.5, booster = "tree")
      eff5 = diag(marginalEffects(m, data = data.frame(train)[,-1], interactions = FALSE, max_indices = 5)$mean)
      
      eff6 = diag(marginalEffects(lm(Y~., data = data.frame(train)), interactions = FALSE, max_indices = 5)$mean)
      
      return(rbind(eff1, eff2, eff3, eff4, eff5, eff6))
    })
    results = abind::abind(results, along = 0L)
    parallel::stopCluster(cl)
    return( apply(results, 2:3, mean) )
  })

saveRDS(results, "results/boosting_regression_trees.RDS")

# 
# 
# task =   # r , X1, X2
# matrix(c(0.0, 1.0, 1.0,
#          0.0, 1.0, 0.5,
#          0.9, 1.0, 1.0,
#          0.9, 1.0, 0.5,
#          0.9, 1.0, 0.0,
#          0.9, 1.0, -1.0,
#          0.9, 1.0, -0.5,
#          -0.9, 1.0, 1.0,
#          -0.9, 1.0, 0.5,
#          -0.9, 1.0, 0.0,
#          -0.9, 1.0, -1.0,
#          -0.9, 1.0, -0.5     # 11    
#          ), ncol = 3, byrow = TRUE)
# 
# res = 
#   lapply(1:nrow(task), function(i) {
#     pars = task[i,]
#     cl = parallel::makeCluster(50L)
#     
#     parallel::clusterExport(cl, varlist = ls(envir = .GlobalEnv))
#     parallel::clusterEvalQ(cl, {library(ranger);library(xgboost);library(cito);library(glmnet);library(glmnetUtils);library(tree)})
#       
#     results = parallel::parLapplyLB(cl, 1:100, function(KK) {
#       
#       predict.naiveBRT = function(model, newdata) {
#         eta = model$eta
#         return(rowSums(matrix(c(1, rep(eta, length(model$model)-1)), nrow(newdata), length(model$model), byrow = TRUE) * sapply(model$model, function(k) predict(k, newdata = data.frame(x = newdata)))))
#       }
#       source("code/AME.R")
#       
#       
#       
#       sim = function() simulate(r = pars[1], effs = c(1, pars[3], 0, 0, 1, seq(-0.2, 0.2, length.out = 10)),n = 5000)
#       data = sim()
#       train = data
#       train[,-1] = scale(train[,-1])
#       
#       n_trees = 250
#       m = xgboost::xgboost(data=xgboost::xgb.DMatrix(as.matrix(train)[,-1],
#                                                      label = (as.matrix(train)[,1, drop=FALSE])),
#                            nrounds = 1,
#                            lambda = 0,
#                            eta = 1,
#                            min_child_weight = 1,
#                            max_depth = 4,
#                            gamma = 0,
#                            objective="reg:squarederror", nthread = 1, verbose = 0)
#       
#       (eff1 = diag(marginalEffects(m, data = data.frame(train)[,-1], interactions = FALSE, max_indices = 5)$mean))
#       m = get_boosting_model(x = train[,-1], y = train[,1], n_trees = n_trees, minsize = 10, bootstrap = NULL, eta = 1.0)
#       eff2 = diag(marginalEffects(m, data = data.frame(train)[,-1], interactions = FALSE, max_indices = 5)$mean)
#       m = get_boosting_model(x = train[,-1], y = train[,1], n_trees = n_trees, minsize = 10, bootstrap = NULL, eta = 0.1)
#       eff3 = diag(marginalEffects(m, data = data.frame(train)[,-1], interactions = FALSE, max_indices = 5)$mean)
#       
#       m = get_boosting_model(x = train[,-1], y = train[,1], n_trees = n_trees, minsize = 3, bootstrap = NULL, eta = 1.0)
#       eff4 = diag(marginalEffects(m, data = data.frame(train)[,-1], interactions = FALSE, max_indices = 5)$mean)
#       m = get_boosting_model(x = train[,-1], y = train[,1], n_trees = n_trees, minsize = 3, bootstrap = NULL, eta = 0.1)
#       eff5 = diag(marginalEffects(m, data = data.frame(train)[,-1], interactions = FALSE, max_indices = 5)$mean)      
#       
#       m = get_boosting_model(x = train[,-1], y = train[,1], n_trees = n_trees, minsize = 10, bootstrap = 0.8, eta = 1.0)
#       eff6 = diag(marginalEffects(m, data = data.frame(train)[,-1], interactions = FALSE, max_indices = 5)$mean)
#       m = get_boosting_model(x = train[,-1], y = train[,1], n_trees = n_trees, minsize = 10, bootstrap = 0.8, eta = 0.1)
#       eff7 = diag(marginalEffects(m, data = data.frame(train)[,-1], interactions = FALSE, max_indices = 5)$mean)
#       m = get_boosting_model(x = train[,-1], y = train[,1], n_trees = n_trees, minsize = 10, bootstrap = 0.5, eta = 0.1)
#       eff8 = diag(marginalEffects(m, data = data.frame(train)[,-1], interactions = FALSE, max_indices = 5)$mean)
#       m = ranger(Y ~., data = data.frame(train), 
#                  num.threads = 3L, num.trees = 500)
#       eff9 = diag(marginalEffects(m, data = data.frame(train), interactions = FALSE, max_indices = 5)$mean)
#       
#       return(rbind(eff1, eff2, eff3, eff4, eff5, eff6, eff7, eff8, eff9))
#     })
#     results = abind::abind(results, along = 0L)
#     parallel::stopCluster(cl)
#     return( list(effs = apply(results, 2:3, mean), 
#                  var = apply(results, 2:3, var)) )
#   })
# 
# saveRDS(res, "results/brt_greedy.RDS")
# 
# 
# 
# 
# sim = function() simulate(r = 0.9, effs = c(1, pars[3], 0, 0, 1, seq(-0.2, 0.2, length.out = 10)),n = 500)
# data = sim()
# train = data
# train[,-1] = scale(train[,-1])
# 
# m = get_boosting_model(x = train[,-1], y = train[,1], n_trees = 100, minsize = 2,mincut =1, mindev = 0.01, bootstrap = NULL, eta = 1, colsample = NULL, booster = "linear")
# m2 = m
# m2$model = m$model[[11]]
# 
# 
# 
# results1 = 
#   sapply(1:100, function(i) {
#     m$N = i
#     print(i)
#     eff = diag(marginalEffects(m, data = data.frame(train)[,-1], interactions = FALSE, max_indices = 5)$mean)
#     return(eff)
#   })
# 
# m = get_boosting_model(x = train[,-1], y = train[,1], n_trees = 100, minsize = 2,mincut =1, mindev = 0.01, bootstrap = NULL, eta = 0.1, colsample = NULL)
# results2 = 
#   sapply(1:100, function(i) {
#     m$N = i
#     print(i)
#     eff = diag(marginalEffects(m, data = data.frame(train)[,-1], interactions = FALSE, max_indices = 5)$mean)
#     return(eff)
#   })
# 
# m = get_boosting_model(x = train[,-1], y = train[,1], n_trees = 30, minsize = 2,mincut =1, mindev = 0.001, bootstrap = NULL, eta = 0.1, colsample = NULL)
# results3 = 
#   sapply(1:30, function(i) {
#     m$N = i
#     print(i)
#     eff = diag(marginalEffects(m, data = data.frame(train)[,-1], interactions = FALSE, max_indices = 5)$mean)
#     return(eff)
#   })
# 
# matplot(t((results1))[1:100,], type = "b")
# 
# 
# m2 = m
# m2$model = list(m$model[[8]])
# m2$N = 1
# diag(marginalEffects(m2, data = data.frame(train)[,-1], interactions = FALSE, max_indices = 5)$mean)
# 
# 
# 
# m = xgboost::xgboost(data=xgboost::xgb.DMatrix(as.matrix(train)[,-1],
#                                                label = (as.matrix(train)[,1, drop=FALSE])),
#                      nrounds = 1,
#                      lambda = 0,
#                      eta = 1,
#                      min_child_weight = 1,
#                      max_depth = 4,
#                      gamma = 0,
#                      objective="reg:squarederror", nthread = 1, verbose = 0)
# 
# 
# 
# sim = function() simulate(r = 0.9, effs = c(1, 0.5, 0, 0, 1, seq(-0.2, 0.2, length.out = 10)),n = 5000)
# data = sim()
# train = data
# train[,-1] = scale(train[,-1])
# 
# results4 = 
#   sapply(1:70, function(i) {
#     m = xgboost::xgboost(data=xgboost::xgb.DMatrix(as.matrix(train)[,-1],
#                                                    label = (as.matrix(train)[,1, drop=FALSE])),
#                          nrounds = i,
#                          lambda = 1,
#                          #eta = 1,
#                          min_child_weight = 1,
#                          max_depth = 4,
#                          gamma = 0,
#                          objective="reg:squarederror", nthread = 1, verbose = 0)
#     eff = diag(marginalEffects(m, data = data.frame(train)[,-1], interactions = FALSE, max_indices = 5)$mean)
#     return(eff)
#   })
# matplot(t((results4))[1:70,], type = "b")
# 
# 
# 
# 
# 
# 
# sim = function() simulate(r = pars[1], effs = c(1, pars[3], 0, 0, 1, seq(-0.2, 0.2, length.out = 10)),n = 1000)
# data = sim()
# train = data
# train[,-1] = scale(train[,-1])
# 
# 
# m = get_boosting_model(x = train[,-1], y = train[,1], n_trees = 450, minsize = 2,mincut =1, mindev = 0.01, bootstrap = NULL, eta = 0.5, colsample = NULL, booster = "linear")
# eff1 = diag(marginalEffects(m, data = data.frame(train)[,-1], interactions = FALSE, max_indices = 5)$mean)
# 
# 
# 
# 
# 
# m = xgboost::xgboost(data=xgboost::xgb.DMatrix(as.matrix(train)[,-1],
#                                                label = (as.matrix(train)[,1, drop=FALSE])),
#                      nrounds = 250,
#                      lambda = 0,
#                      eta = 0.1,
#                      min_child_weight = 100,
#                      max_depth = 20,
#                      gamma = 0,
#                      objective="reg:squarederror", nthread = 1, verbose = 0)
# (eff1 = diag(marginalEffects(m, data = data.frame(train)[,-1], interactions = FALSE, max_indices = 5)$mean))
# 
# 
# 
# task =   # r , X1, X2
#   matrix(c(
#            0.0, 1.0, 0.5,
#            0.9, 1.0, 0.5#,
#            # 0.9, 1.0, 0.0,
#            # 0.9, 1.0, -0.5
#   ), ncol = 3, byrow = TRUE)
# 
# res2 = 
#   lapply(1:nrow(task), function(i) {
#     pars = task[i,]
#     cl = parallel::makeCluster(50L)
#     
#     parallel::clusterExport(cl, varlist = ls(envir = .GlobalEnv))
#     parallel::clusterEvalQ(cl, {library(ranger);library(xgboost);library(cito);library(glmnet);library(glmnetUtils);library(tree)})
#     
#     results = parallel::parLapply(cl, 1:100, function(KK) {
#       
#       source("code/AME.R")
#       
#       sim = function() simulate(r = pars[1], effs = c(1, pars[3], 0, 0, 1, seq(-0.2, 0.2, length.out = 5)),n = 1000)
#       data = sim()
#       train = data
#       train[,-1] = scale(train[,-1])
#         
#       
#       m = get_boosting_model(x = train[,-1], y = train[,1], n_trees = 100, minsize = 2,mincut = 1, mindev = 0.01, bootstrap = NULL, eta = 1.0, colsample = NULL, booster = "linear")
#       (eff1 = diag(marginalEffects(m, data = data.frame(train)[,-1], interactions = FALSE, max_indices = 5)$mean))
#       
#       eff2 = coef(lm(train[,1]~train[,-1]))[2:6]
#       
#       return(rbind(eff1, eff2))
#     })
#     results = abind::abind(results, along = 0L)
#     parallel::stopCluster(cl)
#     return( list(effs = apply(results, 2:3, mean), 
#                  var = apply(results, 2:3, var)) )
#   })
# 
# 
# 
# task =   # r , X1, X2
#   matrix(c(
#     0.0, 1.0, 0.5,
#     0.9, 1.0, 0.5
# #    0.9, 1.0, 0.0,
# #    0.9, 1.0, -0.5
#   ), ncol = 3, byrow = TRUE)
# 
# 
# res = 
#   lapply(1:nrow(task), function(i) {
#     pars = task[i,]
#     
#     cl = parallel::makeCluster(50L)
#     
#     parallel::clusterExport(cl, varlist = ls(envir = .GlobalEnv))
#     parallel::clusterEvalQ(cl, {library(ranger);library(xgboost);library(cito);library(glmnet);library(glmnetUtils);library(tree)})
#     
#     results = parallel::parLapply(cl, 1:100, function(KK) {
#       
#       source("code/AME.R")
#       
#       sim = function() simulate(r = pars[1], effs = c(1, pars[3], 0, 0, 1, seq(-0.2, 0.2, length.out = 5)),n = 10000)
#       data = sim()
#       train = data
#       train[,-1] = scale(train[,-1])
#       
#       
#       m = get_boosting_model(x = train[,-1], y = train[,1], n_trees = 1, minsize = 2,mincut =1, mindev = 0.0000000001, bootstrap = NULL, eta = 1, colsample = NULL, booster = "tree")
#       (eff1 = diag(marginalEffects(m, data = data.frame(train)[,-1], interactions = FALSE, max_indices = 5)$mean))[1:5]
#       
#       eff2 = coef(lm(train[,1]~train[,-1]))[2:6]
#       
#       return(rbind(eff1, eff2))
#     })
#     results = abind::abind(results, along = 0L)
#     parallel::stopCluster(cl)
#     return( list(effs = apply(results, 2:3, mean), 
#                  var = apply(results, 2:3, var)) )
#   })
# 
# 
