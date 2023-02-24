---
title: "Results"
format: 
  docx:
    toc: true
    number-sections: true
    keep-md: true
    fig-format: svg
crossref:
  fig-title: '**Figure**'
  fig-labels: arabic
  tbl-title: '**Table**'
  tbl-labels: arabic
  title-delim: ":"
---




::: {.cell}

:::


## Results

### Proof of concept


::: {.cell}

:::

::: {.cell fig.format='svg'}
::: {.cell-output .cell-output-stdout}
```
              [,1]      [,2]
[1,]  1.000000e+00 0.7071068
[2,]  0.000000e+00 0.7071068
[3,] -7.071068e-01 0.0000000
[4,] -1.836970e-16 1.7071068
```
:::

::: {.cell-output-display}
![Bias on effect estimates for different ML algorithms in three different simulated causal simulations (a and b). Sample sizes are so large that stochastic effects can be excluded (1000 observations and 500 repetitions). Effects of the ML models were inferred using average conditional effects. Row a) shows results for simulations with uncorrelated predictors with effect sizes ($\beta_1$=1.0,  $\beta_2$=0.0, and $\beta_3$=1.0). Row b) shows the results for simulations with X~1~ and X~2~ being strongly correlated (Pearson correlation factor = 0.9) but only X~1~ affects y.](figures/fig-Fig_2-1.svg){#fig-Fig_2}
:::
:::


### Data-poor simulation


::: {.cell}

:::

::: {.cell fig.format='svg'}
::: {.cell-output-display}
![Bias and variance of estimated effects in data-poor situations. N = 50, 100, and 600 observations of 100 weakly correlated predictors were simulated. True effects in the data generating model were $\beta_1$=1.0, $\beta_2$=0.0, and the other 98 effects were equally spaced between 0 and 1. Models were fitted to the simulated data (1000 replicates) with the optimal hyperparameters (except for LM, which doesn’t have hyperparameters). Hyperparameters were selected based on the minimum MSE of ($\hat{\beta}_1$) (green) or the prediction error (based on $\hat{y}$  ) (red). Bias and variance were calculated for $\hat{\beta}_1$ and $\hat{\beta}_2$. Effects $\hat{\beta}_i$ for $i=1,…,100$) were approximated using ACE.](figures/fig-Fig_3-1.svg){#fig-Fig_3}
:::
:::


#### Hyper-parameter sensitivity analysis


::: {.cell}

:::

::: {.cell fig.format='svg'}
::: {.cell-output-display}
![Results of hyperparameter tuning for Neural Networks (NN), Boosted Regression Trees (BRT), Random Forests (RF), and Elastic Net (EN) for 100 observations with 100 predictors. The influence of the hyperparameters on effect ($\hat{\beta}_1$) (bias, variance, and MSE), and the predictions of the model, ($\hat{y}$), (bias, variance, and MSE) were estimated by a multivariate generalized additive model (GAM). Categorical hyperparameters (activation function in NN) were estimated as fixed effects. The responses (bias, variance, MSE) were centered so that the categorical hyperparameters correspond to the intercepts. The variable importance of the hyperparameters was estimated by a random forest with the MSE of the effect $\hat{\beta}_1$ (first plot) or the prediction (second plot) as the response. Red dots correspond to the best predicted set of hyperparameters (based on a random forest), in the first plot for the minimum MSE of the effect for $\hat{\beta}_1$ and in the second plot for the minimum MSE of the predictions $\hat{y}$.](figures/fig-Fig_4-1.svg){#fig-Fig_4}
:::
:::

::: {.cell}
::: {.cell-output .cell-output-stdout}
```
[1] 0 0 0
```
:::

::: {.cell-output .cell-output-stdout}
```
[1] 0 0 0
```
:::
:::


### Case Study



::: {.cell fig.format='svg'}
::: {.cell-output-display}
![Difference between causal and conventional ML models for in-distribution and out-of-distribution predictions. In a simulated setting, the task is to predict Crop yield based on Plant growth (data-generating model is shown in the figure). Climate is an unobservable confounder and has effects on Plant growth and Pest (growth). In the first scenario, i.e. in-distribution predictions, Climate did not change, i.e. patients were exposed to the same climatic conditions; here the difference in predictive performance for the model with and without Pest growth is marginal (predictive performance was measured by R^2^). In the second theoretical setting, the climatic conditions changed (the effects of Climate on Plant growth and Pest are now zero). Using the previously trained models, the model without Pest deficit performed significantly worse than the model with Pest (plot with out-of-distribution predictions).](figures/fig-Fig_5-1.svg){#fig-Fig_5}
:::
:::
