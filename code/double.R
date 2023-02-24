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

Sys.setenv(CUDA_VISIBLE_DEVICES=0)
device = "cuda"      

N_pred = 40
effs = c(1, seq(0, 1, length.out = 39))

sim = function(Sigma) {
  return(
    simulate(r = Sigma ,
             effs = effs,
             n = 100))
}
Sigma = trialr::rlkjcorr(1, N_pred, 2)
train = simulate(r = Sigma ,
                effs = effs,
                n = 100)
test = simulate(r = diag(1.0, N_pred) ,
                effs = effs,
                n = 500)
cl = parallel::makeCluster(25L)
parallel::clusterExport(cl, varlist = ls(envir = .GlobalEnv))
parallel::clusterEvalQ(cl, {library(ranger);library(xgboost);library(cito);library(glmnet);library(glmnetUtils);Sys.setenv(OMP_NUM_THREADS=3)
})
result_list = 
  parallel::parSapplyLB(cl, 1:50, function(i) {
    HH = ceiling(seq(2, 5000, length.out = 50))[i]
  res = replicate(10, {
    
    mse = function(y, y_hat) (mean((y-y_hat)**2))
    tmp = rep(NA, 3)
    try({
      lr = 0.009
      if(HH > 2900) lr = 0.002
    m = cito::dnn(Y~., data = as.data.frame(train), 
                  hidden = rep(HH, 2),
                  activation = rep("selu", 2),
                  verbose = F, 
                  plot=F, lambda = 0.00, alpha = 1., 
                  epochs = 300,
                  lr = lr,
                  batchsize = 20L,
                  device = "cpu",
                  lr_scheduler = config_lr_scheduler("reduce_on_plateau", factor = 0.80, patience = 7), early_stopping = 20)
    m$use_model_epoch = length(m$weights)
    eff = diag(marginalEffects(m, interactions = FALSE)$mean)
    pred = predict(m, newdata = data.frame(test), device = "cpu")
    tmp = c(eff[1:2], mse(test[,1], pred))
    }, silent = TRUE)
    tmp
  })
  return(t(apply(res, 1, mean)))
})

