get_importance2 = function(model, predict_func, data) {
  predictor <- Predictor$new(model, data = data.frame(data[,-1]), y = data[,1], predict.function = predict_func)
  imp <- FeatureEffects$new(predictor, method = "ale")
  AL_IMP = sapply(imp$results, function(i) sd(i$.value[i$.value < quantile(i$.value, 0.975) & i$.value > quantile(i$.value, 0.025)]))
  #plot(imp)
  return(list(imp, AL_IMP))
}


get_importance = function(model, predict_func, data) {
  predictor <- Predictor$new(model, data = data.frame(data[,-1]), y = data[,1], predict.function = predict_func)
  imp <- FeatureImp$new(predictor, loss = "mae", compare = "ratio")
  imp2 = imp$results$importance
  imp2 = imp2[order(imp$results$feature)]
  return(imp2)
  #return(imp)
}


n = 1000
c = 0.7
x1 = runif(n)
x2 = c * x1 + (1-c)* runif(n)
x3 = runif(n)
y = 0.5*x1 + x2 + 0*x3 + rnorm(n)
dat = data.frame(y = y, x1 = x1, x2 = x2, x3 = x3)
plot(dat)

cor(dat$x1, dat$x2)
summary(lm(y~., data = dat))


n = 1000
c = 0.97
AB = mvtnorm::rmvnorm(n, sigma = matrix(c(1, c, c, 1), 2, 2))
x1 = AB[,1]
x2 = AB[,2]
x3 = rnorm(n)
x4 = rnorm(n)
x5 = rnorm(n)
x6 = rnorm(n)
y = x1**2 + x2 + 0*x3 + rnorm(n)

dat = data.frame(y = y, x1 = x1, x2 = x2, x3 = x3, x4 = x4, x5 = x5, x6 = x6)
#nn = data.frame(matrix(runif(n*20), n, 20))
#colnames(nn) = paste0("A", 1:20)
#dat = cbind(dat, nn)
cor(dat$x1, dat$x2)
summary(lm(y~., data = dat))

library(ranger)

out = ranger(y ~ ., data = dat, importance = "permutation", mtry = ceiling( sqrt(ncol(dat)) ))
plot(get_importance2(out, predict_rf, dat))


dat2 = as.matrix(dat)
dat2 = scale(dat2)
brt = xgboost::xgboost(data=xgboost::xgb.DMatrix(dat2[,-1], label = (dat2[,1, drop=FALSE])), 
                       nrounds = 200L, objective="reg:linear",params = list(nthread = 3))

mean_d = apply(dat2, 2, mean)
mean_d = matrix(mean_d, nrow = 1)
mean_d = mean_d[,-1, drop=FALSE]
AME = function(newdata) {
  mean_d[1,1] = newdata
  return(predict_xg(brt, newdata=mean_d))
}
p1 = dat$x1
p1 = p1[order(p1,decreasing=FALSE)]



AME_direct = function(data, predict_f, model, epsilon = 0.1) {
  data = data[,-1]
  res=
    sapply(1:ncol(data), function(p) {
      eps = epsilon*sd(data[,p])
      grads = sapply(1:nrow(data), function(i) {
        tmp1 = data[i,,drop=FALSE]
        y1 = predict_f(model, newdata = tmp1)
        tmp2 = tmp1
        tmp2[1,p] = tmp1[1,p] + eps
        y2 = predict_f(model, newdata = tmp2)
        return((y2-y1)/(tmp2[1,p]-tmp1[1,p]))
      })
      return(c(mean(grads), mean(abs(grads)), sd(grads)))
    })
  return(t(res))
}
res_AME_direct = AME_direct(dat2, predict_xg, brt)


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
((res_AME_ALE = AME_ALE(dat2, predict_xg, brt)))
((res_AME_ALE = AME_ALE(dat2, predict_xg, brt, epsilon = 0.1)))

df = dat2[order(dat2[,2], decreasing=FALSE), ]
eps = 0.1*sd(dat2[,2])
grads = 
  sapply(2:1000, function(i) {

    y1 = predict_xg(brt, newdata= (df[i-1,-1,drop=FALSE]))
    tmp = (df[i-1,-1,drop=FALSE])
   # tmp[,1] = (df[i,-1,drop=FALSE])[,2]
    tmp[,1] = tmp[,1]+eps
    y2 = predict_xg(brt, newdata= tmp)
    
    return((y2-y1)/(tmp[,1]-df[i-1, 2]))
  })
mean(grads)

eps = 0.1*sd(dat2[,3])
df = dat2[order(dat2[,3], decreasing=FALSE), ]
grads2 = 
  sapply(2:1000, function(i) {
    
    y1 = predict_xg(brt, newdata= (df[i-1,-1,drop=FALSE]))
    tmp = (df[i-1,-1,drop=FALSE])
    #tmp[,2] = (df[i,-1,drop=FALSE])[,3]
    tmp[,2] = tmp[,2]+eps
    y2 = predict_xg(brt, newdata= tmp)
    
    return((y2-y1)/(tmp[,2]-df[i-1, 3]))
  })
mean(grads2)

rr = (get_importance2(brt, predict_xg, dat))
df2 = rr[[1]]$results$x2[,2:3]
library(mgcv)
model = gam(.value~.borders, data = df2)
y1 = predict(model, newdata= data.frame(.borders= df2$.borders))
y2 = predict(model, newdata= data.frame(.borders= df2$.borders+0.01))
mean((y2-y1)/(df2$.borders+0.01 - df2$.borders))



df2 = rr[[1]]$results$x2[,2:3]
grads_ALE1 = ((df2$.value[2:21]-df2$.value[1:20])/(df2$.borders[2:21] - df2$.borders[1:20]))

df2 = rr[[1]]$results$x1[,2:3]
grads_ALE2 = ((df2$.value[2:21]-df2$.value[1:20])/(df2$.borders[2:21] - df2$.borders[1:20]))
mean(abs(grads_ALE1))
mean(abs(grads_ALE2))











eps = 0.01*sd(dat2[,4])
df = dat2[order(dat2[,4], decreasing=FALSE), ]
grads3 = 
  sapply(2:1000, function(i) {
    
    y1 = predict_xg(brt, newdata= (df[i-1,-1,drop=FALSE]))
    tmp = (df[i-1,-1,drop=FALSE])
    #tmp[,2] = (df[i,-1,drop=FALSE])[,3]
    tmp[,3] = tmp[,3]+eps
    y2 = predict_xg(brt, newdata= tmp)
    
    return((y2-y1)/(tmp[,3]-df[i-1, 4]))
  })
mean(grads3)




effs = sapply(2:1000, function(i) diag(rootSolve::gradient(AME, x = p1[i-1], pert = p1[i])))
mean(effs)
effs = sapply(2:1000, function(i) diag(rootSolve::gradient(AME, x = p1[i], pert = 0.01*sd(dat$x1))))
mean(effs)


p2 = dat$x2
p2 = p2[order(p2,decreasing=FALSE)]
AME = function(newdata) {
  mean_d[1,2] = newdata
  p1 = predict_xg(brt, newdata=mean_d)
  mean_d[1,2] = newdata+0.01
  p2 = predict_xg(brt, newdata=mean_d)
  return(predict_xg(brt, newdata=mean_d))
}
effs2 = sapply(2:1000, function(i) diag(rootSolve::gradient(AME, x = p2[i-1], pert = p2[i])))
mean(effs2)


effs2 = sapply(2:1000, function(i) diag(rootSolve::gradient(AME, x = p2[i], pert = 0.01*sd(dat$x2))))
mean(effs)


dat = scale(dat)
mm = (lm(y~I(x1^2)+x2+x3+x4+x5+x6, data = as.data.frame(dat)))
margins::margins(mm)
rr = (get_importance2(brt, predict_xg, dat2))
(get_importance(mm, predict, dat))

car::Anova(mm)


gg = (gam(y~x1+x2+x3+x4+x5+x6, data = as.data.frame(dat)))
margins(gg)
