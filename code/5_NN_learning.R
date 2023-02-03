library(torch)
set.seed(42)
Sys.setenv(OMP_NUM_THREADS=3)

source("code/AME.R")
source("code/Scenarios.R")


set.seed(42)
sim = function() simulate(r = 0.9, effs = c(1, 0.0, 0, 0, 0),inter = c(1),n = 1000)
data = sim()

dataT = torch::tensor_dataset(torch_tensor(data[,1,drop=FALSE]), torch_tensor(data[,-1]))
dataL = torch::dataloader(dataT, batch_size = 32L, shuffle = TRUE, pin_memory = TRUE, drop_last = FALSE)

### no hidden layers

res = replicate(50, {

  model = torch::nn_sequential(nn_linear(5L, 1L))
  
  model$cuda("cuda:0")
  opt = optim_adam(params = model$parameters, lr = 0.01)
  lossF = torch::nn_mse_loss()
  
  results = matrix(NA, 6500, 2)
  counter = 1
  for(e in 1:200) {
    print(e)
    coro::loop(for (b in dataL) {
      opt$zero_grad()
      output =  model(b[[2]]$to(device = "cuda:0"))
      loss = lossF(output, b[[1]]$to(device = "cuda:0"))$mean()
      loss$backward()
      opt$step()
      eff_tmp = 
        marginalEffectsGeneric(model, data[,-1], 
                               predict_func = function(model, newdata) as.matrix(model(torch_tensor(newdata)$to(device = "cuda:0"))$data()$cpu()), 
                               interactions = FALSE, max_indices = 2L)
      results[counter, ] = diag(eff_tmp$mean)
      counter = counter + 1
    })
  }
  results
})
res_0 = res
saveRDS(res_0, file = "results/NN_learning_0.RDS")

### 50, 3

res = replicate(50, {
  model = torch::nn_sequential(nn_linear(5L, 50L), nn_relu(), 
                               nn_linear(50L, 50L), nn_relu(), 
                               nn_linear(50L, 50L),  nn_relu(), 
                               nn_linear(50L, 1L))
  model$cuda("cuda:0")
  opt = optim_adam(params = model$parameters, lr = 0.01)
  lossF = torch::nn_mse_loss()
  
  results = matrix(NA, 6500, 2)
  counter = 1
  for(e in 1:200) {
    print(e)
    coro::loop(for (b in dataL) {
      opt$zero_grad()
      output =  model(b[[2]]$to(device = "cuda:0"))
      loss = lossF(output, b[[1]]$to(device = "cuda:0"))$mean()
      loss$backward()
      opt$step()
      eff_tmp = 
        marginalEffectsGeneric(model, data[,-1], 
                               predict_func = function(model, newdata) as.matrix(model(torch_tensor(newdata)$to(device = "cuda:0"))$data()$cpu()), 
                               interactions = FALSE, max_indices = 2L)
      results[counter, ] = diag(eff_tmp$mean)
      counter = counter + 1
    })
  }
  results
})

res_50 = res
saveRDS(res_50, file = "results/NN_learning_50.RDS")


### 500, 3
res = replicate(50, {
  model = torch::nn_sequential(nn_linear(5L, 500L), nn_relu(), 
                               nn_linear(500L, 500L), nn_relu(), 
                               nn_linear(500L, 500L),  nn_relu(), 
                               nn_linear(500L, 1L))
  model$cuda("cuda:0")
  opt = optim_adam(params = model$parameters, lr = 0.01)
  lossF = torch::nn_mse_loss()
  
  results = matrix(NA, 6500, 2)
  counter = 1
  for(e in 1:200) {
    print(e)
    coro::loop(for (b in dataL) {
      opt$zero_grad()
      output =  model(b[[2]]$to(device = "cuda:0"))
      loss = lossF(output, b[[1]]$to(device = "cuda:0"))$mean()
      loss$backward()
      opt$step()
      eff_tmp = 
        marginalEffectsGeneric(model, data[,-1], 
                               predict_func = function(model, newdata) as.matrix(model(torch_tensor(newdata)$to(device = "cuda:0"))$data()$cpu()), 
                               interactions = FALSE, max_indices = 2L)
      results[counter, ] = diag(eff_tmp$mean)
      counter = counter + 1
    })
  }
  results
})
res_500 = res
saveRDS(res_500, file = "results/NN_learning_500.RDS")
