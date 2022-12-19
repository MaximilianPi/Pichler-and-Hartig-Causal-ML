### File which creates the hyper-parameter config file for data-poor scenarios

library(tidyverse)

scenarios = c(
  "_pars_100_10.RDS", 
  "_pars_100_100.RDS",
  "_pars_600_100.RDS",
  "_pars_1000_10.RDS",
  "_pars_1000_100.RDS",
  "_pars_2000_100.RDS"
  )
names(scenarios) = c(
  "N = 100, P = 10", 
  "N = 100, P = 100", 
  "N = 600, P = 100", 
  "N = 1000, P = 10", 
  "N = 1000, P = 100",
  "N = 2000, P = 100"
)

## NN
 scenarios2 = c(
  "_pars_100_10.RDS", 
  "_pars_100_100_SS.RDS",
  "_pars_600_100_SS.RDS",
  "_pars_1000_10.RDS",
  "_pars_1000_100.RDS",
  "_pars_2000_100_SS.RDS"
)
results = lapply(scenarios2, function(p) readRDS(paste0("results/NN", p)) )
results = lapply(results, function(r) do.call(rbind, r))
results = lapply(results, function(r) r[abs(r$bias_1) < 0.98,])
results = lapply(results, function(r) r[abs(r$bias_5) < 0.98,])
res = 
  do.call(rbind,
          lapply(1:length(results), function(i) {
            results[[i]]$scenario = names(scenarios)[i]
            return(results[[i]])
          })
  )


NN_rmse = (res %>% arrange(rmse))
NN_bias = (res %>% arrange(abs(bias_1)))


## BRT
results = lapply(scenarios, function(p) readRDS(paste0("results/BRT", p)) )
results = lapply(results, function(r) do.call(rbind, r))
results = lapply(results, function(r) r[abs(r$bias_1) < 0.98,])
results = lapply(results, function(r) r[abs(r$bias_5) < 0.98,])

res = 
  do.call(rbind,
          lapply(1:length(results), function(i) {
            results[[i]]$scenario = names(scenarios)[i]
            return(results[[i]])
          })
  )

BRT_rmse = (res  %>% arrange(rmse))
BRT_bias = (res  %>% arrange(abs(bias_1)))


## RF
results = lapply(scenarios, function(p) readRDS(paste0("results/RF", p)) )
results = lapply(results, function(r) do.call(rbind, r))
results = lapply(results, function(r) r[abs(r$bias_1) < 0.98,])
results = lapply(results, function(r) r[abs(r$bias_5) < 0.98,])
res = 
  do.call(rbind,
          lapply(1:length(results), function(i) {
            results[[i]]$scenario = names(scenarios)[i]
            return(results[[i]])
          })
  )

RF_rmse = (res %>% arrange(rmse))
RF_bias = (res %>%  arrange(abs(bias_1)))


## Elasticnet
results = lapply(scenarios, function(p) readRDS(paste0("results/Elastic_net", p)) )
results = lapply(results, function(r) do.call(rbind, r))
results = lapply(results, function(r) r[abs(r$bias_1) < 0.98,])
results = lapply(results, function(r) r[abs(r$bias_5) < 0.98,])

res = 
  do.call(rbind,
          lapply(1:length(results), function(i) {
            results[[i]]$scenario = names(scenarios)[i]
            return(results[[i]])
          })
  )

EN_rmse = (res  %>% arrange(rmse))
EN_bias = (res  %>% arrange(abs(bias_1)))





hyper_to_text = function(x, pre="", NN = FALSE) {x
  x = unlist(x)
  if(NN) x[[1]] = paste0("'", x[[1]], "'")
  return(paste0(paste0(paste0(pre, names(x)), "=", x, collapse = "\n"), "\n") )
}


settings = matrix(c(100, 100, 600, 100, 2000, 100), ncol = 2, byrow = TRUE)

sapply(1:3, function(k) {
  tmp = as.integer(settings[k,])
  conn = file(paste0("code/hyper_param_config_",tmp[1],"_",tmp[2],".R"))
  writeLines(
    paste0(
      paste0("## Hyper-parameters for data-poor scenarios created by 'create_hyper_config_.R' file\n"),
       hyper_to_text((RF_rmse %>% filter(scenario == paste0("N = ", tmp[1],", P = ", tmp[2])))[2,-(ncol(RF_rmse): (ncol(RF_rmse) -4 ) )], "RF_"),
      hyper_to_text((BRT_rmse %>% filter(scenario == paste0("N = ", tmp[1],", P = ", tmp[2])))[2,-(ncol(BRT_rmse): (ncol(BRT_rmse) -4 ) )], "BRT_"),
       hyper_to_text((NN_rmse %>% filter(scenario == paste0("N = ", tmp[1],", P = ", tmp[2])))[2,-(ncol(NN_rmse): (ncol(NN_rmse) -4 ) )], "NN_", TRUE),
       hyper_to_text((EN_rmse %>% filter(scenario == paste0("N = ", tmp[1],", P = ", tmp[2])))[2,-(ncol(EN_rmse): (ncol(EN_rmse) -4 ) )], "EN_")), 
    conn )
  close(conn)
})


      