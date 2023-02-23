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
    0.9, 1.0, 0.5,
    0.9, 1.0, 0.0,
    0.0, 1.0, 0.0
  ), ncol = 3, byrow = TRUE)

results = 
  lapply(1:nrow(task), function(i) {
    pars = task[i,]
    cl = parallel::makeCluster(50L)
    
    parallel::clusterExport(cl, varlist = ls(envir = .GlobalEnv))
    parallel::clusterEvalQ(cl, {library(ranger);library(xgboost);library(cito);library(glmnet);library(glmnetUtils);library(tree)})
    
    results = parallel::parLapply(cl, 1:500, function(KK) {
      
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
