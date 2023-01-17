DNN_tuner = function(data, cv = 10, hyper_samples = 50, true_eff = c(1, 0, 0, 0, 1), device = "cpu") {
  library(caret)
  folds = createFolds(data[,1], k = cv, list = TRUE, returnTrain = FALSE)
  
  NN = hyper_samples
  activations = sample(c("relu", "leaky_relu", "tanh", "selu", "elu", "celu", "gelu"), size = NN, replace = TRUE)
  sgd = runif(NN, 0, 1)
  depth = sample(1:7, NN, replace = TRUE)
  width = sample(2:50, NN, replace = TRUE)
  dropout = runif(NN, 0, 0.3)
  alpha = runif(NN, 0, 1.0)
  lambda = runif(NN, 0.005, 0.4)**2
  pars = data.frame(activations, sgd, depth, width, dropout, alpha, lambda)
  pars$mse_X1 = NA
  pars$mse_X2 = NA
  pars$mse_P = NA
  
  mse = function(y, y_hat) (mean((y-y_hat)**2))
  
  for(i in 1:nrow(pars)) {
      print(i)      
      parameter = pars[i,]
    
      results = 
        sapply(1:length(folds), function(j) {
          
          train = data[-folds[[j]],]
          test = data[folds[[j]],]
          
          m = cito::dnn(Y~., data = as.data.frame(train), 
                        activation = rep(parameter$activations, parameter$depth),
                        hidden = rep(parameter$width, parameter$depth),
                        verbose = FALSE, 
                        epochs = 300,
                        lambda = parameter$lambda,
                        alpha = parameter$alpha,
                        batchsize = max(1, floor(nrow(train)*parameter$sgd)), 
                        lr = 0.05,
                        plot=FALSE, 
                        device = device,
                        lr_scheduler = config_lr_scheduler("reduce_on_plateau", factor = 0.90, patience = 4), early_stopping = 8)
          m$use_model_epoch = length(m$weights)
          eff = diag(marginalEffects(m, data = test, interactions = FALSE, device = device, max_indices = length(true_eff))$mean)[1:length(true_eff)]
          pred = predict(m, newdata = test, device = device)
          return(c(mse(eff[1], true_eff[1]) ,mse(eff[2], true_eff[2]), mse(pred, test[,1])))
        })
      tmp = apply(results, 1, mean)
      pars$mse_X1[i] = tmp[1]
      pars$mse_X2[i] = tmp[2]
      pars$mse_P[i] =  tmp[3]
  }
  
  #### MSE_eff model
  
  parameter_eff = (pars %>% arrange(mse_X1))[1,]
  m_eff = cito::dnn(Y~., data = as.data.frame(data), 
                         activation = rep(parameter_eff$activations, parameter_eff$depth),
                         hidden = rep(parameter_eff$width, parameter_eff$depth),
                         verbose = FALSE, 
                         epochs = 300,
                         lambda = parameter_eff$lambda,
                         alpha = parameter_eff$alpha,
                         batchsize = max(1, floor(nrow(train)*parameter_eff$sgd)), 
                         lr = 0.05,
                         plot=FALSE, 
                         device = device,
                         lr_scheduler = config_lr_scheduler("reduce_on_plateau", factor = 0.90, patience = 4), early_stopping = 8)
  
  parameter_pred = (pars %>% arrange(mse_P))[1,]
  m_pred = cito::dnn(Y~., data = as.data.frame(data), 
                    activation = rep(parameter_pred$activations, parameter_pred$depth),
                    hidden = rep(parameter_pred$width, parameter_pred$depth),
                    verbose = FALSE, 
                    epochs = 300,
                    lambda = parameter_pred$lambda,
                    alpha = parameter_pred$alpha,
                    batchsize = max(1, floor(nrow(train)*parameter_pred$sgd)), 
                    lr = 0.05,
                    plot=FALSE, 
                    device = device,
                    lr_scheduler = config_lr_scheduler("reduce_on_plateau", factor = 0.90, patience = 4), early_stopping = 8)
  
  return(list(models = list(m_eff = m_eff, m_pred = m_pred), hyper = list(parameter_eff = parameter_eff, parameter_pred = parameter_pred)))
  
}