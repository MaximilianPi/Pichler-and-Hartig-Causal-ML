library(ranger)
library(xgboost)
library(torch)
library(iml)
library(mgcv)


predict_rf = function(model, newdata) predict(model, data=newdata)$predictions
predict_xg = function(model, newdata) as.vector(predict(model, as.matrix(newdata)))
predict_torch = function(model, newdata) as.numeric(model( torch_tensor(as.matrix(newdata))))

var_names = c("Y", "A", "B", "C","D", "E")

get_importance2 = function(model, predict_func, data) {
  predictor <- Predictor$new(model, data = data.frame(data[,-1]), y = data[,1], predict.function = predict_func)
  imp <- FeatureEffects$new(predictor, method = "ale")
  AL_IMP = sapply(imp$results, function(i) sd(i$.value[i$.value < quantile(i$.value, 0.975) & i$.value > quantile(i$.value, 0.025)]))
  return(AL_IMP)
}


get_importance = function(model, predict_func, data) {
  predictor <- Predictor$new(model, data = data.frame(data[,-1]), y = data[,1], predict.function = predict_func)
  imp <- FeatureImp$new(predictor, loss = "mae", compare = "ratio")
  imp2 = imp$results$importance
  imp2 = imp2[order(imp$results$feature)]
  return(imp2)
}


AME_direct = function(data, predict_f, model, epsilon = 0.1) {
  data = data[,-1]
  res=
    sapply(1:ncol(data), function(p) {
      eps = epsilon*sd(data[,p])
      grads = lapply(1:nrow(data), function(i) {
        tmp1 = data[i,,drop=FALSE]
        tmp2 = tmp1
        tmp2[1,p] = tmp1[1,p] + eps
        return(list(tmp1, tmp2))
      })
      tmp1 = t(sapply(grads, function(k) k[[1]]))
      tmp2 = t(sapply(grads, function(k) k[[2]])) 
      y1 = predict_f(model, newdata = tmp1)
      y2 = predict_f(model, newdata = tmp2)
      grads = (y2-y1)/(tmp2[,p]-tmp1[,p])
      return(c(mean(grads), mean(abs(grads)), sd(grads)))
    })
  return(t(res))
}

AME_ALE = function(data, predict_f, model, epsilon = NULL) {
  predictor <- Predictor$new(model, data = data.frame(data[,-1]), y = data[,1], predict.function = predict_f)
  ALE <- FeatureEffects$new(predictor, method = "ale")
  results = ALE$results
  if(is.null(epsilon)){
    res= 
      sapply(results, function(feature) {
        len = nrow(feature)
        grads = ((feature$.value[2:len]-feature$.value[1:(len-1)])/(feature$.borders[2:len] - feature$.borders[1:(len-1)]))
        return(c(mean(grads), mean(abs(grads)), sd(grads)))
      })
  } else {
    res= 
      sapply(results, function(feature) {
        model = gam(.value~s(.borders), data = feature)
        y1 = predict(model, newdata= data.frame(.borders= feature$.borders))
        y2 = predict(model, newdata= data.frame(.borders= feature$.borders+epsilon))
        grads = ((y2-y1)/(feature$.borders+epsilon - feature$.borders))
        return(c(mean(grads), mean(abs(grads)), sd(grads)))
      })
  }
  return(t(res))
}


simulate = function(n = 1000, c = 0.97, effs = c(0, 0, 0)) {
  c = 0.97
  AB = mvtnorm::rmvnorm(n, sigma = matrix(c(1, c, c, 1), 2, 2))
  x1 = AB[,1]
  x2 = AB[,2]
  x3 = rnorm(n)
  x4 = rnorm(n)
  x5 = rnorm(n)
  x6 = rnorm(n)
  y = effs[1]*x1 + effs[2]*x2 + effs[3]*x3 + rnorm(n)
  dat = scale(data.frame(y = y, x1 = x1, x2 = x2, x3 = x3, x4 = x4, x5 = x5, x6 = x6))
  return(dat)
}

library(parallel)
RhpcBLASctl::omp_set_num_threads(2L)
RhpcBLASctl::blas_set_num_threads(2L)
cl = makeCluster(20)
clusterExport(cl, varlist = ls(), envir = environment())
clusterEvalQ(cl, {library(ranger);library(xgboost);library(torch); library(iml);library(mgcv)})
clusterEvalQ(cl, {RhpcBLASctl::omp_set_num_threads(2L); RhpcBLASctl::blas_set_num_threads(2L)})
results = 
  parLapply(cl, 1:100, function(outer) {
    data = simulate(n = 1000, c = 0.0)
    ## Bootstrap
    boot = 
      lapply(1:300, function(b) {
        df = data[sample.int(nrow(data), nrow(data), replace = TRUE),]  
        brt = xgboost::xgboost(data=xgboost::xgb.DMatrix(as.matrix(df)[,-1], label = (as.matrix(df)[,1, drop=FALSE])), 
                               nrounds = 140L, objective="reg:linear",params = list( verbosity = 0), nthread = 1)
        effs = 
          cbind(
            AME_direct(df, predict_xg, model = brt)[,1],
            AME_ALE(df, predict_xg, model = brt)[,1],
            AME_ALE(df, predict_xg, model = brt, epsilon = 0.1)[,1],
            coef(lm(y~., data = data.frame(df)))[2:7]
          )
        return(effs)
      })
    gc()
    effs = apply(abind::abind(boot, along = 0), 2:3, mean)
    sds = apply(abind::abind(boot, along = 0), 2:3, sd)
    return(2*pnorm(abs(effs/sds), lower.tail = FALSE))
  })
saveRDS(results, file = "bootstrap_AME.rds")
parallel::stopCluster(cl)


cl = makeCluster(20)
clusterExport(cl, varlist = ls(), envir = environment())
clusterEvalQ(cl, {library(ranger);library(xgboost);library(torch); library(iml);library(mgcv)})
clusterEvalQ(cl, {RhpcBLASctl::omp_set_num_threads(2L); RhpcBLASctl::blas_set_num_threads(2L)})
results = 
  parLapply(cl, 1:100, function(outer) {
    data = simulate(n = 1000, c = 0.0, effs = c(1, 0, 0))
    ## Bootstrap
    boot = 
      lapply(1:300, function(b) {
        df = data[sample.int(nrow(data), nrow(data), replace = TRUE),]  
        brt = xgboost::xgboost(data=xgboost::xgb.DMatrix(as.matrix(df)[,-1], label = (as.matrix(df)[,1, drop=FALSE])), 
                               nrounds = 140L, objective="reg:linear",params = list( verbosity = 0), nthread = 1)
        effs = 
          cbind(
            AME_direct(df, predict_xg, model = brt)[,1],
            AME_ALE(df, predict_xg, model = brt)[,1],
            AME_ALE(df, predict_xg, model = brt, epsilon = 0.1)[,1],
            coef(lm(y~., data = data.frame(df)))[2:7]
          )
        return(effs)
      })
    gc()
    effs = apply(abind::abind(boot, along = 0), 2:3, mean)
    sds = apply(abind::abind(boot, along = 0), 2:3, sd)
    return(2*pnorm(abs(effs/sds), lower.tail = FALSE))
  })
saveRDS(results, file = "bootstrap_AME_w_effect.rds")
parallel::stopCluster(cl)

res_with = readRDS(file = "bootstrap_AME_w_effect.rds")
sapply(1:4, function(i) mean(abind::abind(res_with, along = 0)[,1,i] < 0.05))

res_wo = readRDS(file = "bootstrap_AME.rds")

df = 
  cbind(sapply(1:4, function(i) mean(abind::abind(res_with, along = 0)[,1,i] < 0.05)),
        sapply(1:4, function(i) mean(abind::abind(res_wo, along = 0)[,1,i] < 0.05)))
rownames(df) = c("AME", "AME_ALE", "AME_ALE2", "LM")
colnames(df) = c("p-value w effect", "p-value w/o effect")
df
