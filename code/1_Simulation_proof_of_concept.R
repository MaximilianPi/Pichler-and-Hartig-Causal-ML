library(ranger)
library(xgboost)
library(torch)
library(iml)
library(cito)
library(glmnet)
library(glmnetUtils)
set.seed(42)
Sys.setenv(OMP_NUM_THREADS=3)

source("code/AME.R")
source("code/Scenarios.R")


get_result = function(sim ) {
  samples = 500L
  result_list = vector("list", samples)
  cl = parallel::makeCluster(50L)
  nodes = unlist(parallel::clusterEvalQ(cl, paste(Sys.info()[['nodename']], Sys.getpid(), sep='-')))
  parallel::clusterExport(cl, varlist = ls(envir = .GlobalEnv))
  parallel::clusterEvalQ(cl, {library(ranger);library(xgboost);library(cito);library(glmnet);library(glmnetUtils);Sys.setenv(OMP_NUM_THREADS=3)
})
  result_list = 
    parallel::parLapply(cl, 1:samples, function(i) {
      # who am I
      myself = paste(Sys.info()[['nodename']], Sys.getpid(), sep='-')
      dist = cbind(nodes,0:3)
      dev = as.integer(as.numeric(dist[which(dist[,1] %in% myself, arr.ind = TRUE), 2]))
      
      
      Sys.setenv(CUDA_VISIBLE_DEVICES=dev)
      device = "cuda"      
      
      
      data = sim()
      
      train = data[1:1000,]
      test = data[1001:2000,]
      
      result = vector("list", 8L)
      
      mse = function(y, y_hat) (mean((y-y_hat)**2))
      
      ## LM
      m = lm(Y~., data = data.frame(train))
      eff = diag(marginalEffects(m, interactions = FALSE)$mean)
      pred = predict(m, newdata = data.frame(test))
      result[[1]] = list(eff, mse(test[,1], pred))
      
      
      ## RF
      m = ranger(Y ~., data = data.frame(train), 
                 num.threads = 3L)
      eff = diag(marginalEffects(m, data = data.frame(train), interactions = FALSE)$mean)
      pred = predict(m, data = data.frame(test))$predictions
      result[[2]] = list(eff, mse(test[,1], pred))

      ## BRT
      m = xgboost::xgboost(data=xgboost::xgb.DMatrix(as.matrix(train)[,-1],
                                                     label = (as.matrix(train)[,1, drop=FALSE])),
                           nrounds = 100,
                           objective="reg:squarederror", nthread = 1, verbose = 0)
      (eff = diag(marginalEffects(m, data = data.frame(train)[,-1], interactions = FALSE)$mean))
      pred = predict(m, newdata = xgboost::xgb.DMatrix(test[,-1]))
      result[[3]] = list(eff, mse(test[,1], pred))

      ## DNN
      m = cito::dnn(Y~., data = as.data.frame(train), 
                    hidden = rep(50L, 3),
                    verbose = FALSE, 
                    plot=FALSE, lambda = 0.00, alpha = 1., 
                    epochs = 500,
                    lr = 0.01,
                    device = device,
                    lr_scheduler = config_lr_scheduler("reduce_on_plateau", factor = 0.80, patience = 7), early_stopping = 20)
      m$use_model_epoch = length(m$weights)
      eff = diag(marginalEffects(m, interactions = FALSE)$mean)
      pred = predict(m, newdata = data.frame(test))
      result[[4]] = list(eff, mse(test[,1], pred))
      
      
      ## DNN with dropout
      m = cito::dnn(Y~., data = as.data.frame(train), 
                    hidden = rep(50L, 3),
                    verbose = FALSE, 
                    dropout = 0.3,
                    plot=FALSE, lambda = 0.00, alpha = 1., 
                    epochs = 500,
                    lr = 0.01,
                    device = device,
                    lr_scheduler = config_lr_scheduler("reduce_on_plateau", factor = 0.80, patience = 7), early_stopping = 20)
      m$use_model_epoch = length(m$weights)
      eff = diag(marginalEffects(m, interactions = FALSE)$mean)
      pred = predict(m, newdata = data.frame(test))
      result[[5]] = list(eff, mse(test[,1], pred))
      
      
      
      ## L1
      m = cva.glmnet(train[,-1], train[,1], alpha = 1.0)
      eff = diag(marginalEffects(m, data = train[,-1],alpha = 1.0, interactions = FALSE)$mean)
      pred = predict(m, newx = test[,-1], alpha = 1.0)
      result[[6]] = list(eff, mse(test[,1], pred))
      
      
      ## L2 
      m = cva.glmnet(train[,-1], train[,1], alpha = 0.0)
      eff = diag(marginalEffects(m, data = train[,-1],alpha = 0.0, interactions = FALSE)$mean)
      pred = predict(m, newx = test[,-1], alpha = 0.0)
      result[[7]] = list(eff, mse(test[,1], pred))
      
      
      ## L1 + L2 
      m = cva.glmnet(train[,-1], train[,1], alpha = 0.2)
      eff = diag(marginalEffects(m, data = train[,-1],alpha = 0.2, interactions = FALSE)$mean)
      pred = predict(m, newx = test[,-1], alpha = 0.2)
      result[[8]] = list(eff, mse(test[,1], pred))
      
    
      return(result)
    })
  parallel::stopCluster(cl)
  return(result_list)
}

sim = function() simulate(r = 0.9, effs = c(1, 0.5, 0, 0, 1),n = 2000)
results = get_result(sim)
saveRDS(results, "results/confounder.RDS")

sim = function() simulate(r = 0.9, effs = c(1, 0, 0, 0, 1),n = 2000)
results = get_result(sim)
saveRDS(results, "results/collinearity_0.90.RDS")

sim = function() simulate(effs = c(1.0, 0.0, 1.0, 0.0, 1.0), r = 0.0,n = 2000)
results = get_result(sim)
saveRDS(results, "results/effects.RDS")


sim = function() simulate(r = 0.9, effs = c(1, -0.5, 0, 0, 1),n = 2000)
results = get_result(sim)
saveRDS(results, "results/confounder_unequal.RDS")


sim = function() simulate(r = 0.5, effs = c(1, 0, 0, 0, 1),n = 2000)
results = get_result(sim)
saveRDS(results, "results/collinearity_0.5.RDS")

sim = function() simulate(r = 0.99, effs = c(1, 0, 0, 0, 1),n = 2000)
results = get_result(sim)
saveRDS(results, "results/collinearity_0.99.RDS")


sim = function() simulate( effs = c(0.0, 0.0, 0.0, 0.0, 0.0), r = 0.0,n = 2000)
results = get_result(sim)
saveRDS(results, "results/no_effects.RDS")

