set.seed(42)
library(xgboost)
library(ranger)
library(qgam)
library(mgcv)
library(mgcViz)
library(ggplot2)
library(tidyverse)

source("utils.R")

for(n in c(50, 100, 600, 2000)) {
  paths = c(
    paste0("results/NN_pars_",n,"_100_replicate.RDS"),
    paste0("results/BRT_pars_",n,"_100_replicate.RDS"),
    paste0("results/RF_pars_",n,"_100_replicate.RDS"),
    paste0("results/Elastic_net_pars_",n,"_100_replicate.RDS")
  )
  methods = c("NN", "BRT", "RF", "Elastic_net")
  results = lapply(1:4, function(i) get_coefficients(i, path = paths[i], method = methods[i]))
  data = do.call(rbind, lapply(1:4, function(i) results[[i]]$data ))
  saveRDS(results, paste0("results/hyper_parameter_aggregation_",n,".RDS"))
}