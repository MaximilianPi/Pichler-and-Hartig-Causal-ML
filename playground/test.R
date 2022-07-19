library(ranger)
library(xgboost)
library(torch)
library(iml)
var_names = c("Y", "A", "B", "C","D", "E")

get_importance = function(model, predict_func, data) {
  predictor <- Predictor$new(model, data = data.frame(data[,-1]), y = data[,1], predict.function = predict_func)
  imp <- FeatureImp$new(predictor, loss = "mae", compare = "ratio")
  return(imp$results[order(imp$results$feature),])
}


get_importance2 = function(model, predict_func, data) {
  predictor <- Predictor$new(model, data = data.frame(data[,-1]), y = data[,1], predict.function = predict_func)
  imp <- FeatureEffects$new(predictor, method = "ale")
  AL_IMP = sapply(imp$results, function(i) sd(i$.value[i$.value < quantile(i$.value, 0.975) & i$.value > quantile(i$.value, 0.025)]))
  return((AL_IMP-min(AL_IMP))/max(AL_IMP))
}

predict_rf = function(model, newdata) predict(model, data=newdata)$predictions
predict_xg = function(model, newdata) as.vector(predict(model, as.matrix(newdata)))
predict_torch = function(model, newdata) as.numeric(model( torch_tensor(as.matrix(newdata))))

sapply(imp$results, function(i) sd(i$.value))

simulate_collinearity = function(r=0.9, n = 1000) {
  ABCDE = MASS::mvrnorm(n, mu=rep(0, 5), Sigma=matrix(c(1, r, 0,0,0,
                                                        r, 1, 0,0,0,
                                                        0, 0, 1,0,0,
                                                        0, 0, 0,1,0,
                                                        0, 0, 0, 0,1), nrow=5, byrow = TRUE), empirical=TRUE)
  A = ABCDE[,1] 
  B = ABCDE[,2]
  C = ABCDE[,3]
  D = ABCDE[,4]
  E = ABCDE[,5]
  
  Y = A*1.0 + 2*E+ rnorm(n, sd = 0.3)
  data = cbind(Y, A, B,C, D, E)
  colnames(data) = var_names
  return(data)
}


simulate_confounder = function(n = 1000) {
  B = rnorm(n) # confounder
  A = B+ rnorm(n, sd = 0.3)
  C = rnorm(n)
  D = rnorm(n)
  E = rnorm(n)
  
  Y = 0.5*A + 1*B + 0*E + rnorm(n, sd = 0.3)
  data = cbind(Y, A, B,C, D, E)
  colnames(data) = var_names
  return(scale(data))
}
data = simulate_collinearity(n = 1000, r = 0.01)

lm = lm(Y ~ A+B+C+D+E, data = data.frame(data))
coef(lm)[6]/coef(lm)[2]
IMP = get_importance(lm, predict_func = predict, data = data)
IMP$importance[5]/IMP$importance[1]

data = simulate_collinearity(n = 1000, r = 0.99)
rf = ranger(Y ~ A+B+C+D+E, data = data.frame(data), importance = "impurity")
IMP = get_importance(rf, predict_func = predict_rf, data = data)
IMP$importance[5]/IMP$importance[1]

al = get_importance2(rf, predict_func = predict_rf, data = data)
plot(al)

resX =   
  lapply(1:100, function(i) {
  data = simulate_confounder(n = 1000)
  brt = xgboost::xgboost(data=xgboost::xgb.DMatrix(data[,-1], label = (data[,1, drop=FALSE])), 
                         nrounds = 200L, objective="reg:linear")
  IMP = get_importance(brt, predict_func = predict_xg, data = data)
  d1= (IMP$importance-min(IMP$importance))/max(IMP$importance)
  
  al = get_importance2(brt, predict_func = predict_xg, data = data)
  AL_IMP = sapply(al$results, function(i) sd(i$.value[i$.value < quantile(i$.value, 0.95) & i$.value > quantile(i$.value, 0.05)]))
  d2=(AL_IMP-min(AL_IMP))/max(AL_IMP)
  AL_IMP = sapply(al$results, function(i) diff(quantile(i$.value, probs = c(0.5, 0.95)))[1])
  d3=(AL_IMP-min(AL_IMP))/max(AL_IMP)
  
  lm = lm(Y ~ A+B+C+D+E, data = data.frame(data))
  d4= coef(lm)[2:6]/max(coef(lm)[2:6])
  return(rbind(d1, d2, d3, d4))
  })

df2 = apply(abind::abind(resX, along = 0L), 2:3, mean)
df[,1]/df[,2]
df[,1]/df[,5]

df2[,1]/df2[,2]
df2[,1]/df2[,5]

apply(t(res), 2, mean)

get_importance(lm, stats::predict, data =data)
 
pred_f =  predict(ranger(Y ~ A+B+E, data = data.frame(data), importance = "permutation"), data = data.frame(data))
pred_A = predict(ranger(Y ~ A, data = data.frame(data), importance = "permutation"), data = data.frame(data))
pred_B = predict(ranger(Y ~ B, data = data.frame(data), importance = "permutation"), data = data.frame(data))
pred_E = predict(ranger(Y ~ E, data = data.frame(data), importance = "permutation"), data = data.frame(data))


L = sum((mean(data[,1])-data[,1])**2)

sum((pred_f$predictions-data[,1])**2)
sum((pred_A$predictions-data[,1])**2)
sum((pred_B$predictions-data[,1])**2)
sum((pred_E$predictions-data[,1])**2)

cor(pred_f$predictions, data[,1])**2
cor(pred_A$predictions, data[,1])**2
cor(pred_B$predictions, data[,1])**2




ame::computeAME(brt, predict.fun = function(object, newdata) as.vector(predict(object, as.matrix(newdata))), 
                data = data.frame(data[,-1]), features = "B",
                aggregate.fun = mean)



nn = nn_sequential(
  nn_linear(ncol(data[,-1]), out_features = 20L),
  nn_relu(),
  nn_linear(20L, out_features = 20L),
  nn_relu(),
  nn_linear(20L, out_features = 20L),
  nn_relu(),
  nn_linear(20L, out_features = 1L)
)
train_torch(nn, data)
opt = optim_adam( nn$parameters , lr = 0.01)

res = 
  sapply(1:200, function(i) {
  opt$zero_grad()
  p = nn(torch_tensor(data[i,-1,drop=FALSE]))
  loss = nnf_mse_loss(p, torch_tensor(data[i,1,drop=FALSE]))
  loss$backward()
  return(apply(abs(as.matrix(nn$parameters$`0.weight`$grad)), 2, mean))
  })





one_by_one = function(n_in = 5L, n_out = 1L, hidden = (3L)) {
  
  ele = torch_ones(c(1L, n_in), requires_grad = TRUE)
  
  pars1 = torch_randn(list(5L, 20L), requires_grad = TRUE)
  bias1 = torch_randn(20L, requires_grad = TRUE)
  
  pars2 = torch_randn(list(20L, 20L), requires_grad = TRUE)
  bias2 = torch_randn(20L, requires_grad = TRUE)
  
  pars3 = torch_randn(list(20L, 1L), requires_grad = TRUE)
  bias3 = torch_randn(1L, requires_grad = TRUE)
  
  out = list()
  out$pars1 = list(ele)
  out$pars2 = list(pars1, bias1, pars2, bias2, pars3, bias3)
  out$forward = function(d) {
    f = torch_mul(ele, d)
    f = nnf_relu(torch_matmul(f, pars1) + bias1)
    f = nnf_relu(torch_matmul(f, pars2) + bias2)
    f = nnf_relu(torch_matmul(f, pars3) + bias3)
    return(f)
  }
  return(out)
}


train_torch = function(nn, data, l1 = FALSE, l2 = FALSE, epochs = 200){
  opt = optim_adam( c(nn$pars2) , lr = 0.01)
  for(i in 1:epochs){
    ind = sample.int(nrow(data), 20L)
    X_b = torch_tensor(data[ind,-1])
    Y_b = torch_tensor(data[ind,1,drop=FALSE])
    opt$zero_grad()
    pred = nn$forward(X_b)
    loss = nnf_mse_loss(pred, Y_b)
    loss$backward()
    opt$step()
  }
  
  opt = optim_adam( c(nn$pars1) , lr = 0.01)
  for(i in 1:epochs){
    ind = sample.int(nrow(data), 20L)
    X_b = torch_tensor(data[ind,-1])
    Y_b = torch_tensor(data[ind,1,drop=FALSE])
    opt$zero_grad()
    pred = nn$forward(X_b)
    loss = nnf_mse_loss(pred, Y_b)
    
    if(l1) {
      reg_loss = torch_zeros(1)
      for(p in nn$pars1){
         reg_loss = reg_loss + torch_norm(p,p = 1)
      }
      loss = loss + reg_loss*0.01
    }
    loss$backward()
    opt$step()
    
  }
  
  
}

nn = one_by_one()
nn$pars1
train_torch(nn, data, l1 = TRUE, epochs = 100)
