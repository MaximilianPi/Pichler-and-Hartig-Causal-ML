

simulate_SDM_corr = function(effs = c(1.0, -0.5), r = 0.9) {
  
  AB = mvtnorm::rmvnorm(1000, mean = c(0,0), sigma = matrix(c(1.0, r, r, 1.0),2,2))
  Temperatur = AB[,1] + rnorm(1000, sd = 0.3)
  Humidity = AB[,2] + rnorm(1000, sd = 0.3)
  Elephant = Temperatur*effs[2] + Humidity*effs[1]  + rnorm(1000, sd = 0.3)
  return(data.frame(Elephant = Elephant, Temperatur = Temperatur, Humidity = Humidity))
}

get_R = function(pred, test) {
  return(cor(test[,1], pred)**2)
}

train_test_brt = function(train, test, MCE = FALSE) {
  brt = xgboost::xgboost(data=xgboost::xgb.DMatrix(as.matrix(train)[,-1,drop=FALSE],
                                                    label = (as.matrix(train)[,1, drop=FALSE])),
                          nrounds = 40L,
                          objective="reg:squarederror", nthread = 1, verbose = 0)
  if(MCE) {
    ME = diag(marginalEffects(brt, as.matrix(train[,-1, drop = FALSE]))$mean)
    return(list(MCE = ME, R2 = get_R(predict(brt, newdata = as.matrix(test[,-1,drop=FALSE])), test)))
  } else {
    return(list(R2=get_R(predict(brt, newdata = as.matrix(test[,-1,drop=FALSE])), test)))
  }
}


train_test_rf = function(train, test, MCE = FALSE) {
  rf = ranger(Elephant ~., data = data.frame(train), num.trees = 100L, num.threads = 3L)
  if(MCE) {
    ME = diag(marginalEffects(rf, as.matrix(train))$mean)
    return(list(MCE = ME, R2 = get_R(predict(rf, data = test)$predictions, test)))
  } else { return(list(R2=get_R(predict(rf, data = test)$predictions, test))) }
}

train_test_cito = function(train, test, MCE = FALSE) {
  dnn = cito::dnn(Elephant~., data = as.data.frame(train), 
                 activation = rep("relu", 3),
                 hidden = rep(20L, 3),
                 verbose = FALSE, 
                 batchsize = 100L, 
                 plot=FALSE, lambda = 0.00, alpha = 1.)
  if(MCE) {
    ME = diag(marginalEffects(dnn)$mean)
    return(list(MCE = ME, R2 = get_R(predict(dnn, newdata = test)[,1], test)))
  } else {return(list(R2=get_R(predict(dnn, newdata = test)[,1], test))) }
}


train_test_lm = function(train, test, MCE = FALSE) {
  lm = lm(Elephant~., data = as.data.frame(train))
  if(MCE) {
    ME = diag(marginalEffects(lm)$mean)
    return(list(MCE = ME, R2 = get_R(predict(lm, newdata = test), test)))
  } else { return(list(R2=get_R(predict(lm, newdata = test), test))) }
}


train_test_glmnet= function(train, test, MCE = FALSE) {
  glmnet = cva.glmnet(as.matrix(train[,-1,drop=FALSE]), train[,1], alpha = 0.2)
  if(MCE) {
  ME = diag(marginalEffects(glmnet, data = as.matrix(train[,-1,drop=FALSE]), alpha = 0.2)$mean)
  return(list(MCE = ME, R2 = get_R(predict(glmnet, newx = as.matrix(test[,-1,drop=FALSE]), alpha = 0.2)[,1], test)))
  } else {return(list(R2=get_R(predict(glmnet, newx = as.matrix(test[,-1,drop=FALSE]), alpha = 0.2)[,1], test)))}
}



##### 6.9.2022
simulate_SDM = function(effs = c(-1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0)) {
  
  Klima = (rnorm(1000, sd = 1))
  Temperatur = effs[5]*Klima + rnorm(1000,sd = ifelse( abs(effs[5])==0, 1, 0.2 ))
  Humidity = effs[6]*Klima + rnorm(1000,sd = ifelse( abs(effs[6])==0, 1, 0.2 ))
  Elephant = Temperatur*effs[1] + Humidity*effs[2]  + rnorm(1000, sd = 0.3)
  Openeness = Humidity*effs[3] + Elephant*effs[4]  + effs[7]*1 + rnorm(1000, sd = ifelse( abs(effs[7])==1, 1, 0.3 ))
  
  return(data.frame(Elephant = Elephant, Temperatur = Temperatur, Openeness = Openeness, Humidity = Humidity))
}


res = array(NA, dim = list(100, 5, 4))

for(i in 1:10) {
  train = simulate_SDM(  c(1, -1.5, 1, 1, 1, 1, 0))
  test = simulate_SDM(   c(1, -1.5, 1, 1, 0, 0, 0))
  
  res[i,1,1] = train_test_brt(train[,c(-3, -2)], test[,c(-3, -2)])$R2
  res[i,1,2] = train_test_brt(train[,c(-3)], test[,c(-3)])$R2
  res[i,1,3] = train_test_brt(train[,c(-2)], test[,c(-2)])$R2
  res[i,1,4] = train_test_brt(train[,], test[,])$R2
  
  res[i,2,1] = train_test_rf(train[,c(-3, -2)], test[,c(-3, -2)])$R2
  res[i,2,2] = train_test_rf(train[,c(-3)], test[,c(-3)])$R2
  res[i,2,3] = train_test_rf(train[,c(-2)], test[,c(-2)])$R2
  res[i,2,4] = train_test_rf(train[,], test[,])$R2
  
  res[i,3,1] = train_test_cito(train[,c(-3, -2)], test[,c(-3, -2)])$R2
  res[i,3,2] = train_test_cito(train[,c(-3)], test[,c(-3)])$R2
  res[i,3,3] = train_test_cito(train[,c(-2)], test[,c(-2)])$R2
  res[i,3,4] = train_test_cito(train[,], test[,])$R2
  
  (res[i,4,1] = train_test_lm(train[,c(-3, -2)], test[,c(-3, -2)])$R2) # H
  (res[i,4,2] = train_test_lm(train[,c(-3)], test[,c(-3)])$R2) # H T
  (res[i,4,3] = train_test_lm(train[,c(-2)], test[,c(-2)])$R2) # H O
  (res[i,4,4] = train_test_lm(train[,], test[,])$R2) # H O T
  
}
saveRDS(res, "results/res_changed_coll.RDS")




res = array(NA, dim = list(100, 5, 4))

for(i in 1:10) {
  
  train = simulate_SDM(  c(1, -1.5, 1, 1, 1, 1, 0))
  test = simulate_SDM(   c(1, -1.5, 0, 0, 1, 1, 1))
  
  res[i,1,1] = train_test_brt(train[,c(-3, -2)], test[,c(-3, -2)])$R2
  res[i,1,2] = train_test_brt(train[,c(-3)], test[,c(-3)])$R2
  res[i,1,3] = train_test_brt(train[,c(-2)], test[,c(-2)])$R2
  res[i,1,4] = train_test_brt(train[,], test[,])$R2
  
  res[i,2,1] = train_test_rf(train[,c(-3, -2)], test[,c(-3, -2)])$R2
  res[i,2,2] = train_test_rf(train[,c(-3)], test[,c(-3)])$R2
  res[i,2,3] = train_test_rf(train[,c(-2)], test[,c(-2)])$R2
  res[i,2,4] = train_test_rf(train[,], test[,])$R2
  
  res[i,3,1] = train_test_cito(train[,c(-3, -2)], test[,c(-3, -2)])$R2
  res[i,3,2] = train_test_cito(train[,c(-3)], test[,c(-3)])$R2
  res[i,3,3] = train_test_cito(train[,c(-2)], test[,c(-2)])$R2
  res[i,3,4] = train_test_cito(train[,], test[,])$R2
  
  (res[i,4,1] = train_test_lm(train[,c(-3, -2)], test[,c(-3, -2)])$R2) # H
  (res[i,4,2] = train_test_lm(train[,c(-3)], test[,c(-3)])$R2) # H T
  (res[i,4,3] = train_test_lm(train[,c(-2)], test[,c(-2)])$R2) # H O
  (res[i,4,4] = train_test_lm(train[,], test[,])$R2) # H O T
  
}
saveRDS(res, "results/res_changed_intervention.RDS")


res = array(NA, dim = list(100, 5, 4))

for(i in 1:10) {
  train = simulate_SDM(  c(1, -1.5, 1, 1, 1, 1, 0))
  test = simulate_SDM(   c(1, -1.5, 1, 1, 1, 1, 0))
  
  res[i,1,1] = train_test_brt(train[,c(-3, -2)], test[,c(-3, -2)])$R2
  res[i,1,2] = train_test_brt(train[,c(-3)], test[,c(-3)])$R2
  res[i,1,3] = train_test_brt(train[,c(-2)], test[,c(-2)])$R2
  res[i,1,4] = train_test_brt(train[,], test[,])$R2
  
  res[i,2,1] = train_test_rf(train[,c(-3, -2)], test[,c(-3, -2)])$R2
  res[i,2,2] = train_test_rf(train[,c(-3)], test[,c(-3)])$R2
  res[i,2,3] = train_test_rf(train[,c(-2)], test[,c(-2)])$R2
  res[i,2,4] = train_test_rf(train[,], test[,])$R2
  
  res[i,3,1] = train_test_cito(train[,c(-3, -2)], test[,c(-3, -2)])$R2
  res[i,3,2] = train_test_cito(train[,c(-3)], test[,c(-3)])$R2
  res[i,3,3] = train_test_cito(train[,c(-2)], test[,c(-2)])$R2
  res[i,3,4] = train_test_cito(train[,], test[,])$R2
  
  res[i,4,1] = train_test_lm(train[,c(-3, -2)], test[,c(-3, -2)])$R2
  res[i,4,2] = train_test_lm(train[,c(-3)], test[,c(-3)])$R2
  res[i,4,3] = train_test_lm(train[,c(-2)], test[,c(-2)])$R2
  res[i,4,4] = train_test_lm(train[,], test[,])$R2
  
}
saveRDS(res, "results/res_changed_no.RDS")









########
