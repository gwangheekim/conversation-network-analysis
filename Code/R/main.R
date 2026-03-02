
rm(list=ls()); gc()

library(ggplot2)
library(gridExtra)
library(dplyr)
library(plotly)
library(igraph)
library(ggrepel)
library(RColorBrewer)
library(knitr)
library(purrr)
library(tidyverse)
library(MCMCpack)
library(coda)
library(future)
library(furrr)
library(progressr)
library(purrr)
library(future.apply)
library(ggplot2)
library(tidyr)
library(nbamen) # install.packages("./nbamen_0.1.0.tar.gz", repos = NULL, type="source")

source("Code/R/utils.R")
scores_AL = read.csv("Data/scores.csv")

###########################################################################
# Data preprocess
delete_nm = scores_AL$s_l_name[(scores_AL$csts == 0)|(scores_AL$test_total2 == 0)]
scores_AL_d = scores_AL[!unique(scores_AL$s_l_name) %in% delete_nm,]
AL_exp_d = read.csv("Data/Network_EXP.csv")
AL_eoi_d = read.csv("Data/Network_EOI.csv")

AL_exp_d = as.matrix(AL_exp_d[,-1])
AL_eoi_d = as.matrix(AL_eoi_d[,-1])

row.names(AL_exp_d) = colnames(AL_exp_d)
row.names(AL_eoi_d) = colnames(AL_eoi_d)

N = nrow(AL_exp_d)
X = array(0, c(N,N,1))

## Network visualization
draw_network(AL_exp_d, option = "pre", path = NA, seed = 55)
draw_network(AL_exp_d, option = "post", path = NA, seed = 55)
draw_network(AL_eoi_d, option = "pre", path = NA, seed = 46)
draw_network(AL_eoi_d, option = "post", path = NA, seed = 46)

###########################################################################

###########################################################################
# AMEN model estimate
num_cores_to_use = 8
plan(multisession, workers = num_cores_to_use)
opts <- furrr_options(seed = TRUE)
num_runs = 10
run_amen <- function(data, X, ndim, r_sd) {
  amen_count_nb(data=data, X = X, niter=100000, nburn=10000,
                 nthin = 5, overdispersion = T, zeroinflate = F,
                 jump_r = 0.5, r_init = 1,
                 verbose = F, direct = T, fix = T, fixsd = T,
                 jump_gamma = 0.5, jump_beta = 2.0, jump_alpha = 2.0,
                 jump_z = 0.6, jump_w = 0.6, pr_sd_z = 1, pr_sd_w = 1,
                 pr_sd_beta = 3, pr_sd_alpha = 3,
                 pr_mean_r = log(1), pr_sd_r = r_sd,
                 eta_init = 0.5, pr_a_eta = 1, pr_b_eta = 1, jump_eta = 0.3,
                 pr_mean_gamma = 0.5, pr_sd_gamma = 1, ndim = ndim,
                 singledist = F, single = F, hierarchical_r = F,
                 vector = T, missing = -99, covariate = F, fix_r = F)
  
}

set.seed(1234)
results_list1 <- future_map(1:num_runs, ~ run_amen(AL_exp_d, X, ndim=2, r_sd=0.15),
                            .options = opts)

set.seed(1234)
results_list2 <- future_map(1:num_runs, ~ run_amen(AL_eoi_d, X, ndim=3, r_sd=0.1),
                            .options = opts)

plan(sequential)

results_m1 = results_list1
ref_idx_exp = which.min(lapply(results_list1, function(x){x$dic}))
for(i in c(1:length(results_list1))[-ref_idx_exp]){
  re_proc = procrustes_mat(results_m1[[i]], results_m1[[ref_idx_exp]])
  results_m1[[i]]$z_estimate = re_proc$z_estimate
  results_m1[[i]]$w_estimate = re_proc$w_estimate
  results_m1[[i]]$z = re_proc$z
  results_m1[[i]]$w = re_proc$w
}

results_m2 = results_list2
ref_idx_eoi = which.min(lapply(results_list2, function(x){x$dic}))
for(i in c(1:length(results_list1))[-ref_idx_eoi]){
  re_proc = procrustes_mat(results_m2[[i]], results_m2[[ref_idx_eoi]])
  results_m2[[i]]$z_estimate = re_proc$z_estimate
  results_m2[[i]]$w_estimate = re_proc$w_estimate
  results_m2[[i]]$z = re_proc$z
  results_m2[[i]]$w = re_proc$w
}

model_exp_neg = results_m1[[ref_idx_exp]]
model_eoi_neg = results_m2[[ref_idx_eoi]]

###########################################################################

###########################################################################
# Network mediation
make_multi_chain_med(results_m1) %>% knitr::kable(format = "latex")
make_multi_chain_med(results_m2) %>% knitr::kable(format = "latex")

###########################################################################

###########################################################################
# Sensitivity analysis
model = model_exp_neg
plan(multisession, workers = 8)
nsamp = length(model_eoi_neg$lik)
n_rows = 3  
n_cols = 4  

handlers(global = TRUE)
results_list_m1 <- with_progress({
  p <- progressor(steps = nsamp)
  future_map(1:nsamp, function(i) {
    p(sprintf("Processing sample %d", i))
    result <- make_med_res(data = model$data, object = scores_AL_d,
                           U = model$z[i,,], V = model$w[i,,],
                           interval = c(0.025, 0.975), print = F,
                           interaction = T, rotation = F)
    as.matrix(result)
  }, .options = furrr_options(seed = TRUE))
})



model = model_eoi_neg
plan(multisession, workers = 8)
nsamp = length(model_eoi_neg$lik)
n_rows = 3  
n_cols = 4  

handlers(global = TRUE)
results_list_m2 <- with_progress({
  p <- progressor(steps = nsamp)
  future_map(1:nsamp, function(i) {
    p(sprintf("Processing sample %d", i))
    result <- make_med_res(data = model$data, object = scores_AL_d,
                           U = model$z[i,,], V = model$w[i,,],
                           interval = c(0.025, 0.975), print = F,
                           interaction = T, rotation = F)
    as.matrix(result)
  }, .options = furrr_options(seed = TRUE))
})

plan(sequential)

results_array_m1 <- array(NA,dim = c(nsamp, n_rows, n_cols))
for(i in 1:nsamp) {
  results_array_m1[i,,] <- results_list_m1[[i]]
}

results_array_m2 <- array(NA,dim = c(nsamp, n_rows, n_cols))
for(i in 1:nsamp) {
  results_array_m2[i,,] <- results_list_m2[[i]]
}

df_exp = data.frame(
  NIE = results_array_m1[,1,1],
  NDE = results_array_m1[,2,1],
  TE = results_array_m1[,3,1]
)

df_eoi = data.frame(
  NIE = results_array_m2[,1,1],
  NDE = results_array_m2[,2,1],
  TE = results_array_m2[,3,1]
)

make_post_plot = function(df){
  df_long = pivot_longer(df, everything(), names_to = "Effect", values_to = "Value")
  df_long_f = df_long %>% filter(Effect != "TE")
  df_long_f$Effect = factor(df_long_f$Effect, levels = c("NIE", "NDE", "TE"))
  ci_vals = df_long_f %>%
    group_by(Effect) %>%
    summarise(
      lower = quantile(Value, 0.025),
      upper = quantile(Value, 0.975),
      .groups = "drop"
    )
  post_plot = ggplot(df_long_f, aes(x = Value)) +
    geom_density(color = "black", fill = "darkgrey", alpha = 0.7, size = 1) +
    geom_vline(data = ci_vals, aes(xintercept = lower),
               linetype = "dashed", color = "red", size = 0.8) +
    geom_vline(data = ci_vals, aes(xintercept = upper),
               linetype = "dashed", color = "red", size = 0.8) +
    facet_grid(Effect ~ .) +
    theme_bw(base_size = 14) +
    labs(x = "Posterior mean",
         y = "Density") +
    theme(
      strip.text = element_text(face = "bold", size = 20),
      axis.title = element_text(size = 18),
      axis.text = element_text(size = 16)
    ) +
    scale_x_continuous(breaks = seq(-4, 4, by = 1), limits = c(-4, 4))+
    scale_y_continuous(breaks = seq(0, 0.8, by = 0.2), limits = c(0, 0.8))
  
  result = list(figures = post_plot,
                data = df_long)
  return(result)
}

make_post_plot(df_exp)$figures
make_post_plot(df_eoi)$figures

make_post_plot(df_exp)$data %>%
  group_by(Effect) %>%
  summarise(
    mean = mean(Value),
    lower = quantile(Value, 0.025),
    upper = quantile(Value, 0.975),
    .groups = "drop"
  ) %>% 
  mutate(across(where(is.numeric), ~ format(round(., 4), nsmall = 4))) %>% 
  slice(c(2,1,3)) %>% 
  knitr::kable(format = "latex")

make_post_plot(df_eoi)$data %>%
  group_by(Effect) %>%
  summarise(
    mean = mean(Value),
    lower = quantile(Value, 0.025),
    upper = quantile(Value, 0.975),
    .groups = "drop"
  ) %>% 
  mutate(across(where(is.numeric), ~ format(round(., 4), nsmall = 4))) %>% 
  slice(c(2,1,3)) %>% 
  knitr::kable(format = "latex")



