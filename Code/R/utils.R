make_multi_chain_med = function(model_list, q = c(0.025,0.975), rotation = F){
  med_res = list()
  for(i in 1:num_runs){
    med_res[[i]] = make_med_res(model_list[[i]]$data, scores_AL_d,
                                model_list[[i]]$z_estimate,
                                model_list[[i]]$w_estimate,
                                q, print=F,interaction=T,
                                rotation=rotation)
  }
  
  res_d = dim(med_res[[1]])
  med_res_a = matrix("",res_d[1],res_d[2] + 1)
  for(i in 1:res_d[1]){
    for(j in 1:res_d[2]){
      mv = round(mean(map_dbl(med_res,function(x) x[i,j])),4)
      sdv = round(sd(map_dbl(med_res,function(x) x[i,j])),4)
      med_res_a[i,j] = paste0(mv, " (", sdv, ")")
    }
  }
  
  row.names(med_res_a) = c("NIE","NDE","TE")
  colnames(med_res_a) = c("Post.M","Post.SD","2.5%","97.5%","")
  
  for (i in 1:nrow(med_res_a)) {
    val1 = as.numeric(sub(" \\(.*\\)", "", med_res_a[i, 3]))
    val2 = as.numeric(sub(" \\(.*\\)", "", med_res_a[i, 4]))
    if ((val1 * val2) > 0) {
      med_res_a[i, 5] = "*"
    }
  }
  return(med_res_a)
}

format_row_names <- function(df, prefix = "alpha") {
  rownames(df) <- paste0("$\\", prefix, "_{\\text{", rownames(df), "}}$")
  return(df)
}

create_latex_table_split <- function(df, prefix,caption_text, label_text) {
  df_formatted <- format_row_names(df,prefix)
  n <- nrow(df_formatted)
  df1 <- df_formatted[1:(n / 2), ]
  df2 <- df_formatted[(n / 2 + 1):n, ]
  table1_content <- kable(df1,
                          format = "latex",
                          booktabs = TRUE,
                          escape = FALSE, # LaTeX 코드가 그대로 출력되도록 설정
                          digits = 4,
                          col.names = c("Estimate", "Post.SD", "2.5\\%", "97.5\\%")
  ) %>%
    gsub("\\\\toprule", "\\\\hline", .) %>% # \toprule을 \hline로 변경
    gsub("\\\\midrule", "", .) %>% # \midrule 제거
    gsub("\\\\bottomrule", "\\\\hline", .) # \bottomrule을 \hline로 변경
  
  table2_content <- kable(df2,
                          format = "latex",
                          booktabs = TRUE,
                          escape = FALSE,
                          digits = 4,
                          col.names = c("Estimate", "Post.SD", "2.5\\%", "97.5\\%")
  ) %>%
    gsub("\\\\toprule", "\\\\hline", .) %>%
    gsub("\\\\midrule", "", .) %>%
    gsub("\\\\bottomrule", "\\\\hline", .)
  
  latex_code <- paste0(
    "\\begin{table}[!ht]\n",
    "\\centering\n",
    "\\small{\n",
    "\\begin{minipage}{0.45\\textwidth}\n",
    gsub("tabular", "tabular", table1_content), 
    "\n\\end{minipage}\n",
    "\\hspace{0.4cm}\n",
    "\\begin{minipage}{0.45\\textwidth}\n",
    gsub("tabular", "tabular", table2_content), 
    "\n\\end{minipage}\n",
    "}\n",
    "\\caption{", caption_text, "}\n",
    "\\label{", label_text, "}\n",
    "\\end{table}\n"
  )
  
  return(latex_code)
}

preprocess_data = function(data_exp, data_eoi, scores_AL){
  A_exp_AL = make_adj_matrix(data_exp, type = "exp")
  common_names = scores_AL$s_l_name[scores_AL$s_l_name %in% rownames(A_exp_AL)]
  sorted_common_names = sort(common_names)
  AL_exp_d_idx = match(sorted_common_names, rownames(A_exp_AL))
  
  AL_exp_d = A_exp_AL[AL_exp_d_idx,AL_exp_d_idx]
  diag(AL_exp_d) = 0
  
  AL_eoi_d_l = make_level_adj(data_eoi, c("low"), scores_AL) * 1
  AL_eoi_d_m = make_level_adj(data_eoi, c("medium"), scores_AL) * 2
  AL_eoi_d_h = make_level_adj(data_eoi, c("high"), scores_AL) * 3
  AL_eoi_d = round(AL_eoi_d_l + AL_eoi_d_m + AL_eoi_d_h)
  diag(AL_eoi_d) = 0
  
  return(list(EXP = AL_exp_d,
              EOI = AL_eoi_d))
}

run_sd_r_experiment <- function(data_list, X,
                                dim_fixed,
                                sd_r_candidates = c(0.1, 0.15, 0.2, 0.25, 0.3),
                                num_repeats = 3,
                                data_names = NULL,
                                seed = 100,
                                mcmc_settings = list(),
                                parallel_workers = NULL,
                                ALc = ALc) {
  
  if (!is.null(parallel_workers)) {
    max_cores <- parallel::detectCores()
    actual_workers <- min(parallel_workers, max_cores)
    cat(sprintf("Setting up parallel processing with %d workers (max available: %d)\n",
                actual_workers, max_cores))
    plan(multisession, workers = actual_workers)
  } else {
    plan(sequential)
  }
  
  default_settings <- list(
    niter = 100000, nburn = 10000, nthin = 5,
    overdispersion = TRUE, zeroinflate = FALSE,
    jump_r = 0.5, verbose = FALSE,
    direct = TRUE, fix = TRUE, fixsd = TRUE,
    jump_gamma = 0.05, jump_beta = 2, jump_alpha = 2,
    jump_z = 0.5, jump_w = 0.5, pr_sd_z = 1, pr_sd_w = 1,
    pr_sd_beta = 1, pr_sd_alpha = 1,
    pr_sd_r = 0.2,
    pr_a_eta = 1, pr_b_eta = 1, jump_eta = 1, eta_init = 0.01,
    pr_mean_gamma = 0.5, pr_sd_gamma = 1, fix_r = F,
    singledist = FALSE, single = FALSE, hierarchical_r = FALSE,
    vector = TRUE, missing = -99, covariate = FALSE
  )
  
  settings <- modifyList(default_settings, mcmc_settings)
  if (is.null(data_names)) {
    data_names <- paste0("Dataset_", seq_along(data_list))
  }
  
  experiment_grid <- expand.grid(
    dataset_idx = seq_along(data_list),
    sd_r = sd_r_candidates,
    repeat_id = seq_len(num_repeats),
    stringsAsFactors = FALSE
  )
  
  cat(sprintf("Total experiments to run: %d (fixed dim = %d)\n",
              nrow(experiment_grid), dim_fixed))
  
  run_single_experiment <- function(dataset_idx, sd_r, repeat_id) {
    mu <- mean(data_list[[dataset_idx]][ALc > 0])
    v  <- var(data_list[[dataset_idx]][ALc > 0])
    r_init <- mu^2 / (v - mu)
    
    per_run_settings <- modifyList(settings, list(pr_sd_r = sd_r))
    args <- c(
      list(
        data = data_list[[dataset_idx]],
        X = X,
        ndim = dim_fixed,
        pr_mean_r = log(1),
        r_init = 1
      ),
      per_run_settings
    )
    
    tryCatch({
      model_output <- do.call(lsirm1pl_count, args)
      list(
        model_output = model_output,
        bic = model_output$bic,
        waic = model_output$waic,
        dic = model_output$dic,
        log_likelihood = max(model_output$lik, na.rm = TRUE),
        status = "success",
        error_message = NA
      )
    }, error = function(e) {
      warning(sprintf("Error in %s - sd_r: %.3f, Repeat: %d: %s",
                      data_names[dataset_idx], sd_r, repeat_id, e$message))
      list(
        model_output = NULL,
        bic = NA, waic = NA, dic = NA, log_likelihood = NA,
        status = "error",
        error_message = e$message
      )
    })
  }
  
  start_time <- Sys.time()
  set.seed(seed)
  results_raw <- future_pmap(experiment_grid, run_single_experiment,
                             .options = furrr_options(seed = TRUE))
  end_time <- Sys.time()
  runtime <- end_time - start_time
  
  cat(sprintf("\nTotal runtime: %s\n", format(runtime)))
  plan(sequential)
  
  processed_metrics_list <- list()
  
  for (d_idx in unique(experiment_grid$dataset_idx)) {
    for (curr_sd in unique(experiment_grid$sd_r)) {
      group_idx <- which(experiment_grid$dataset_idx == d_idx &
                           experiment_grid$sd_r == curr_sd)
      sorted_rows <- experiment_grid[group_idx, ] %>% arrange(repeat_id)
      
      group_outputs <- lapply(seq_len(nrow(sorted_rows)), function(i) {
        original_idx <- which(
          experiment_grid$dataset_idx == sorted_rows$dataset_idx[i] &
            experiment_grid$sd_r == sorted_rows$sd_r[i] &
            experiment_grid$repeat_id == sorted_rows$repeat_id[i]
        )
        results_raw[[original_idx]]
      })
      
      reference_model_output <- NULL
      reference_run_idx_in_group <- NA
      
      successful_runs <- which(sapply(group_outputs, function(x) x$status == "success"))
      if (length(successful_runs) > 0) {
        dic_values <- sapply(successful_runs, function(i) group_outputs[[i]]$dic)
        best_run_idx <- successful_runs[which.min(dic_values)]
        reference_model_output <- group_outputs[[best_run_idx]]$model_output
        reference_run_idx_in_group <- best_run_idx
        cat(sprintf("Selected chain %d as reference (DIC: %.2f) for dataset %s, sd_r %.3f, dim %d\n",
                    best_run_idx, min(dic_values), data_names[d_idx], curr_sd, dim_fixed))
      }
      
      for (i in seq_along(group_outputs)) {
        current_info <- sorted_rows[i, ]
        current_raw <- group_outputs[[i]]
        
        final_metrics <- list(
          dataset = data_names[current_info$dataset_idx],
          sd_r = current_info$sd_r,
          dimension = dim_fixed,
          repeat_id = current_info$repeat_id,
          bic = NA, waic = NA, dic = NA, log_likelihood = NA,
          error_message = NA
        )
        
        if (current_raw$status == "success") {
          if (!is.null(reference_model_output) && i != reference_run_idx_in_group) {
            tryCatch({
              re_proc <- procrustes_mat(current_raw$model_output, reference_model_output)
              model_data_for_waic <- if (!is.null(current_raw$model_output$data)) {
                current_raw$model_output$data
              } else {
                data_list[[current_info$dataset_idx]]
              }
              
              waic_dic_result <- calculate_waic_dic_procrustes_cpp(
                data = as.matrix(model_data_for_waic),
                beta_samples = current_raw$model_output$beta,
                alpha_samples = current_raw$model_output$alpha,
                gamma_samples = current_raw$model_output$gamma,
                r_samples = current_raw$model_output$r,
                z_proc = re_proc$z,
                w_proc = re_proc$w,
                ndim = ncol(re_proc$z_estimate),
                overdispersion = settings$overdispersion,
                zeroinflate = settings$zeroinflate,
                missing = settings$missing,
                vector = settings$vector
              )
              
              final_metrics$bic <- current_raw$bic
              final_metrics$waic <- waic_dic_result$waic
              final_metrics$dic <- waic_dic_result$dic
              final_metrics$log_likelihood <- current_raw$log_likelihood
              final_metrics$error_message <- NA
            }, error = function(e) {
              warning(sprintf("Procrustes/WAIC/DIC recalculation error for %s - sd_r: %.3f, Repeat: %d: %s",
                              data_names[current_info$dataset_idx],
                              current_info$sd_r,
                              current_info$repeat_id, e$message))
              final_metrics$bic <- current_raw$bic
              final_metrics$waic <- NA
              final_metrics$dic <- NA
              final_metrics$log_likelihood <- current_raw$log_likelihood
              final_metrics$error_message <- paste("Procrustes processing error:", e$message)
            })
          } else {
            final_metrics$bic <- current_raw$bic
            final_metrics$waic <- current_raw$waic
            final_metrics$dic <- current_raw$dic
            final_metrics$log_likelihood <- current_raw$log_likelihood
            final_metrics$error_message <- current_raw$error_message
          }
        } else {
          final_metrics$bic <- NA
          final_metrics$waic <- NA
          final_metrics$dic <- NA
          final_metrics$log_likelihood <- NA
          final_metrics$error_message <- current_raw$error_message
        }
        
        processed_metrics_list[[length(processed_metrics_list) + 1]] <- final_metrics
      }
    }
  }
  
  metrics <- purrr::map_dfr(processed_metrics_list, ~.)
  
  summary_stats <- metrics %>%
    group_by(dataset, sd_r) %>%
    summarise(
      across(c(bic, waic, dic, log_likelihood),
             list(mean = ~mean(.x, na.rm = TRUE),
                  sd = ~sd(.x, na.rm = TRUE))),
      n_runs = n(),
      n_errors = sum(!is.na(error_message)),
      .groups = 'drop'
    )
  
  list(
    metrics = metrics,
    summary = summary_stats,
    experiment_conditions = experiment_grid,
    runtime = runtime
  )
}

run_dimension_experiment <- function(data_list, X, 
                                     dim_range = 1:8, 
                                     num_repeats = 3,
                                     data_names = NULL,
                                     seed = 100,
                                     mcmc_settings = list(),
                                     parallel_workers = NULL,
                                     ALc = ALc) {  
  
  if(!is.null(parallel_workers)) {
    max_cores <- parallel::detectCores()
    actual_workers <- min(parallel_workers, max_cores)
    
    cat(sprintf("Setting up parallel processing with %d workers (max available: %d)\n", 
                actual_workers, max_cores))
    plan(multisession, workers = actual_workers)
  } else {
    plan(sequential) 
  }
  
  default_settings <- list(
    niter = 100000, nburn = 10000, nthin = 5,
    overdispersion = TRUE, zeroinflate = FALSE,
    jump_r = 0.5, verbose = FALSE,
    direct = TRUE, fix = TRUE, fixsd = TRUE,
    jump_gamma = 0.05, jump_beta = 2, jump_alpha = 2,
    jump_z = 0.5, jump_w = 0.5, pr_sd_z = 1, pr_sd_w = 1,
    pr_sd_beta = 1, pr_sd_alpha = 1,
    pr_sd_r = 0.2,
    pr_a_eta = 1, pr_b_eta = 1, jump_eta = 1, eta_init = 0.01,
    pr_mean_gamma = 0.5, pr_sd_gamma = 1, fix_r = F,
    singledist = FALSE, single = FALSE, hierarchical_r = FALSE,
    vector = TRUE, missing = -99, covariate = FALSE
  )
  
  settings <- modifyList(default_settings, mcmc_settings)
  if(is.null(data_names)) {
    data_names <- paste0("Dataset_", 1:length(data_list))
  }
  
  experiment_grid <- expand.grid(
    dataset_idx = 1:length(data_list),
    dim = dim_range,
    repeat_id = 1:num_repeats,
    stringsAsFactors = FALSE
  )
  
  cat(sprintf("Total experiments to run: %d\n", nrow(experiment_grid)))
  
  run_single_experiment <- function(dataset_idx, dim, repeat_id) {
    mu = mean(data_list[[dataset_idx]][ALc>0])
    v = var(data_list[[dataset_idx]][ALc>0])
    r_init = mu^2 / (v - mu)
    
    args <- c(list(data = data_list[[dataset_idx]], X = X, ndim = dim, pr_mean_r = log(1), r_init = 1), settings)
    result_list <- tryCatch({
      model_output <- do.call(lsirm1pl_count, args)
      list(
        model_output = model_output, 
        bic = model_output$bic,      
        waic = model_output$waic,    
        dic = model_output$dic,      
        log_likelihood = max(model_output$lik, na.rm = TRUE), 
        error_message = NA,
        status = "success"
      )
    }, error = function(e) {
      warning(sprintf("Error in %s - Dimension: %d, Repeat: %d: %s", 
                      data_names[dataset_idx], dim, repeat_id, e$message))
      list(
        model_output = NULL, # 오류 발생 시 모델 출력은 NULL
        bic = NA, waic = NA, dic = NA, log_likelihood = NA,
        error_message = e$message,
        status = "error"
      )
    })
    
    return(result_list)
  }
  
  start_time <- Sys.time()
  
  set.seed(seed)
  results_raw <- future_pmap(experiment_grid, run_single_experiment,
                             .options = furrr_options(seed = TRUE))
  
  end_time <- Sys.time()
  runtime <- end_time - start_time
  
  cat(sprintf("\nTotal runtime: %s\n", format(runtime)))
  plan(sequential)
  
  processed_metrics_list <- list()
  for (d_idx in unique(experiment_grid$dataset_idx)) {
    for (curr_dim in unique(experiment_grid$dim)) {
      group_exp_grid_indices <- which(experiment_grid$dataset_idx == d_idx & 
                                        experiment_grid$dim == curr_dim)
      sorted_exp_grid_rows <- experiment_grid[group_exp_grid_indices, ] %>%
        arrange(repeat_id)
      
      group_raw_outputs_sorted <- lapply(1:nrow(sorted_exp_grid_rows), function(i) {
        original_idx_in_results_raw <- which(experiment_grid$dataset_idx == sorted_exp_grid_rows$dataset_idx[i] &
                                               experiment_grid$dim == sorted_exp_grid_rows$dim[i] &
                                               experiment_grid$repeat_id == sorted_exp_grid_rows$repeat_id[i])
        results_raw[[original_idx_in_results_raw]]
      })
      
      reference_model_output <- NULL
      reference_run_idx_in_group <- NA
      
      successful_runs <- which(sapply(group_raw_outputs_sorted, function(x) x$status == "success"))
      
      if (length(successful_runs) > 0) {
        dic_values <- sapply(successful_runs, function(i) {
          group_raw_outputs_sorted[[i]]$dic
        })
        
        best_run_idx <- successful_runs[which.min(dic_values)]
        reference_model_output <- group_raw_outputs_sorted[[best_run_idx]]$model_output
        reference_run_idx_in_group <- best_run_idx
        
        cat(sprintf("Selected chain %d as reference (DIC: %.2f) for dataset %s, dimension %d\n", 
                   best_run_idx, min(dic_values), data_names[d_idx], curr_dim))
      }
      
      for (i in 1:length(group_raw_outputs_sorted)) {
        current_run_info <- sorted_exp_grid_rows[i, ]
        current_raw_output <- group_raw_outputs_sorted[[i]]
        
        final_metrics_for_this_run <- list(
          dataset = data_names[current_run_info$dataset_idx],
          dimension = current_run_info$dim,
          repeat_id = current_run_info$repeat_id,
          bic = NA, waic = NA, dic = NA, log_likelihood = NA,
          error_message = NA
        )
        
        if (current_raw_output$status == "success") {
          if (!is.null(reference_model_output) && i != reference_run_idx_in_group) {
            
            proc_result_try <- tryCatch({
              re_proc <- procrustes_mat(current_raw_output$model_output, reference_model_output)
              model_data_for_waic <- if (!is.null(current_raw_output$model_output$data)) {
                current_raw_output$model_output$data
              } else {
                data_list[[current_run_info$dataset_idx]]
              }
              
              waic_dic_result <- calculate_waic_dic_procrustes_cpp(
                data = as.matrix(model_data_for_waic), 
                beta_samples = current_raw_output$model_output$beta,
                alpha_samples = current_raw_output$model_output$alpha,
                gamma_samples = current_raw_output$model_output$gamma,
                r_samples = current_raw_output$model_output$r,
                z_proc = re_proc$z, # Procrustes aligned z (posterior samples)
                w_proc = re_proc$w, # Procrustes aligned w (posterior samples)
                ndim = ncol(re_proc$z_estimate), 
                overdispersion = settings$overdispersion, 
                zeroinflate = settings$zeroinflate,       
                missing = settings$missing,
                vector = settings$vector
              )
              
              final_metrics_for_this_run$bic <- current_raw_output$bic
              final_metrics_for_this_run$waic <- waic_dic_result$waic
              final_metrics_for_this_run$dic <- waic_dic_result$dic
              final_metrics_for_this_run$log_likelihood <- current_raw_output$log_likelihood
              final_metrics_for_this_run$error_message <- NA 
              
            }, error = function(e) {
              warning(sprintf("Procrustes/WAIC/DIC recalculation error for %s - Dimension: %d, Repeat: %d: %s", 
                              data_names[current_run_info$dataset_idx], 
                              current_run_info$dim, 
                              current_run_info$repeat_id, e$message))
              final_metrics_for_this_run$bic <- current_raw_output$bic 
              final_metrics_for_this_run$waic <- NA 
              final_metrics_for_this_run$dic <- NA
              final_metrics_for_this_run$log_likelihood <- current_raw_output$log_likelihood
              final_metrics_for_this_run$error_message <- paste("Procrustes processing error:", e$message)
            })
          } else {
            final_metrics_for_this_run$bic <- current_raw_output$bic
            final_metrics_for_this_run$waic <- current_raw_output$waic
            final_metrics_for_this_run$dic <- current_raw_output$dic
            final_metrics_for_this_run$log_likelihood <- current_raw_output$log_likelihood
            final_metrics_for_this_run$error_message <- current_raw_output$error_message
          }
        } else {
          final_metrics_for_this_run$bic <- NA
          final_metrics_for_this_run$waic <- NA
          final_metrics_for_this_run$dic <- NA
          final_metrics_for_this_run$log_likelihood <- NA
          final_metrics_for_this_run$error_message <- current_raw_output$error_message
        }
        processed_metrics_list[[length(processed_metrics_list) + 1]] <- final_metrics_for_this_run
      }
    }
  }
  
  metrics <- purrr::map_dfr(processed_metrics_list, ~.) 
  
  summary_stats <- metrics %>%
    group_by(dataset, dimension) %>%
    summarise(
      across(c(bic, waic, dic, log_likelihood), 
             list(mean = ~mean(.x, na.rm = TRUE), 
                  sd = ~sd(.x, na.rm = TRUE))),
      n_runs = n(),
      n_errors = sum(!is.na(error_message)), 
      .groups = 'drop'
    )
  
  list(
    metrics = metrics,
    summary = summary_stats,
    experiment_conditions = experiment_grid,
    runtime = runtime
  )
}

procrustes_mat = function(output, reference){
  
  w.star = reference$w_raw[reference$map_inf$iter,,]
  z.star = reference$z_raw[reference$map_inf$iter,,]
  # w.star = reference$w_estimate
  # z.star = reference$z_estimate
  
  nmcmc = length(reference$lik)
  nsample = nitem = nrow(reference$data)
  ndim = ncol(z.star)
  
  w.proc = array(0,dim=c(nmcmc,nitem,ndim))
  z.proc = array(0,dim=c(nmcmc,nsample,ndim))
  
  pb <- txtProgressBar(title = "progress bar", min = 0, max = nmcmc,
                       style = 3, width = 50)
  
  for(iter in 1:nmcmc){
    z.iter = output$z_raw[iter,,]
    z.proc[iter,,] = procrustes(z.iter,z.star)$X.new
    
    w.iter = output$w_raw[iter,,]
    w.proc[iter,,] = procrustes(w.iter,w.star)$X.new
    
    setTxtProgressBar(pb, iter, label = paste(round(iter/nmcmc * 100, 0), "% done"))
  }
  
  w.est = colMeans(w.proc, dims = 1)
  z.est = colMeans(z.proc, dims = 1)
  
  return(list(w_estimate = w.est,
              z_estimate = z.est,
              w = w.proc,
              z = z.proc))
  
}

apply_procrustes <- function(results_list, 
                             reference_index = 1, 
                             use_parallel = FALSE, 
                             n_cores = 2) {
  
  if(use_parallel) {
    plan(multisession, workers = n_cores)
    on.exit(plan(sequential))
    cat(sprintf("Processing %d results in parallel with %d cores...\n", 
                length(results_list)-1, n_cores))
  } else {
    plan(sequential)
    cat(sprintf("Processing %d results sequentially...\n", length(results_list)-1))
  }
  
  results_m = results_list
  indices_to_process <- 2:length(results_list)
  total_results <- length(results_list) - 1
  
  if(use_parallel) {
    future_walk(indices_to_process, function(i) {
      re_proc = procrustes_mat(results_m[[i]], results_m[[reference_index]])
      results_m[[i]]$z_estimate <<- re_proc$z_estimate
      results_m[[i]]$w_estimate <<- re_proc$w_estimate
      results_m[[i]]$z <<- re_proc$z
      results_m[[i]]$w <<- re_proc$w
    }, .options = furrr_options(seed = TRUE))
    cat("All results processed!\n")
  } else {
    for(j in seq_along(indices_to_process)){
      i <- indices_to_process[j]
      cat(sprintf("\n=== Processing Result %d/%d ===\n", j, total_results))
      re_proc = procrustes_mat(results_m[[i]], results_m[[reference_index]], 
                               result_index = j, total_results = total_results)
      results_m[[i]]$z_estimate = re_proc$z_estimate
      results_m[[i]]$w_estimate = re_proc$w_estimate
      results_m[[i]]$z = re_proc$z
      results_m[[i]]$w = re_proc$w
    }
    cat("\nAll results processed!\n")
  }
  
  return(results_m)
}

draw_network = function(adj, option = "pre", path = NA, seed = 9){
  g = igraph::graph_from_adjacency_matrix(adj, mode = "directed", weighted = TRUE)
  
  if(option == "pre"){
    V(g)$score = scale(scores_AL_d$csts)[,1]
  }else if(option == "post"){
    V(g)$score = scale(scores_AL_d$test_total2)[,1] 
  }
  
  color_palette = (colorRampPalette(c("royalblue1", "white", "indianred2"))(100))
  score_normalized = (V(g)$score - min(V(g)$score)) / (max(V(g)$score) - min(V(g)$score))
  node_colors = color_palette[round(score_normalized * 99) + 1]
  
  set.seed(seed)
  layout = igraph::layout_with_fr(g)
  
  if(!is.na(path)){
    png(path, width = 800, height = 800, res = 150)
    
    edge_widths = igraph::E(g)$weight
    quantiles = round(quantile(edge_widths, c(0.25,0.5,0.75)),0)
    edge_widths[edge_widths > 0 & edge_widths <= quantiles[1]] <- 1
    edge_widths[edge_widths > quantiles[1] & edge_widths <= quantiles[2]] <- 3
    edge_widths[edge_widths > quantiles[2] & edge_widths <= quantiles[3]] <- 5
    edge_widths[edge_widths > quantiles[3]] <- 7
    
    par(mar = c(.0, .0, .0, .0))
    plot(g, edge.curved = 0.6,
         edge.width = edge_widths,
         edge.color = "darkblue",
         vertex.color = node_colors,
         vertex.size = 25,
         vertex.label.color = "black",
         vertex.label.cex = 0.7,
         vertex.label.font = 4,
         layout = layout)
    
    x_start = -.75
    y_start = -1
    legend_width = 0.1
    legend_height = 0.3
    
    legend_gradient(
      x = x_start, y = y_start, title = "Standardized math score",
      width = legend_width, height = legend_height,
      grad_colors = color_palette,
      labels = round(seq(min(V(g)$score), max(V(g)$score), length.out = 5), 2)
    )
    
    dev.off()
  }else{
    edge_widths = igraph::E(g)$weight
    quantiles = round(quantile(edge_widths, c(0.25,0.5,0.75)),0)
    edge_widths[edge_widths > 0 & edge_widths <= quantiles[1]] <- 1
    edge_widths[edge_widths > quantiles[1] & edge_widths <= quantiles[2]] <- 3
    edge_widths[edge_widths > quantiles[2] & edge_widths <= quantiles[3]] <- 5
    edge_widths[edge_widths > quantiles[3]] <- 7
    
    par(mar = c(.0, .0, .0, .0))
    plot(g, edge.curved = 0.6,
         edge.width = edge_widths,
         edge.color = "darkblue",
         vertex.color = node_colors,
         vertex.size = 25,
         vertex.label.color = "black",
         vertex.label.cex = 0.7,
         vertex.label.font = 4,
         layout = layout)
    
    x_start = -.75
    y_start = -1
    legend_width = 0.1
    legend_height = 0.3
    
    legend_gradient(
      x = x_start, y = y_start, title = "Standardized math score",
      width = legend_width, height = legend_height,
      grad_colors = color_palette,
      labels = round(seq(min(V(g)$score), max(V(g)$score), length.out = 5), 2)
    )
  }
}

make_med_res = function(data, object, U, V, interval = c(0.025,0.975),
                        print = FALSE, seed = 10,
                        interaction = TRUE, rotation = FALSE){
  
  Umat = if(is.vector(U)) matrix(U, ncol=1) else as.matrix(U)
  Vmat = if(is.vector(V)) matrix(V, ncol=1) else as.matrix(V)
  
  if(rotation){
    if (ncol(Umat) >= 2) {
      rot = GPArotation::Varimax(Umat)
      Q = rot$Th
      Umat = rot$loadings
      if (ncol(Vmat) >= 2) {
        Vmat = Vmat %*% t(Q)  
      }
    }  
  }
  
  row.names(Umat) = row.names(Vmat) = row.names(data)
  Umat = Umat[order(row.names(Umat)), , drop=FALSE]
  Vmat = Vmat[order(row.names(Vmat)), , drop=FALSE]
  
  if (exists("delete_nm") && length(delete_nm) > 0) {
    Umat = Umat[!rownames(Umat) %in% delete_nm, , drop=FALSE]
    Vmat = Vmat[!rownames(Vmat) %in% delete_nm, , drop=FALSE]
  }
  
  # object = scores_AL_d
  object = object %>% dplyr::mutate(A = ifelse(csts >= 350, 0, 1))
  
  object_filtered = object[!object$s_l_name %in% delete_nm,]
  object_filtered = object_filtered[order(object_filtered$s_l_name),]
  
  data_list = list(
    Y = scale(object_filtered$test_total2),
    X = object_filtered$gender,
    C1 = object_filtered$A
  )
  
  for(i in 1:ncol(Umat)) {
    data_list[[paste0("M", i)]] = Umat[,i]
  }
  
  for(i in 1:ncol(Vmat)) {
    data_list[[paste0("M", ncol(Umat) + i)]] = Vmat[,i]
  }
  
  data <- data.frame(data_list)
  
  num_M = sum(!is.na(stringr::str_match(colnames(data), "M")))
  num_C = sum(!is.na(stringr::str_match(colnames(data), "C")))
  
  form_Y = if (interaction) {
    formula(paste0("Y ~ X + ",
                   paste0("C",c(1:num_C),collapse = " + "),
                   " + ", paste0("M",c(1:num_M),collapse = " + "),
                   " + ", paste0("X:M",c(1:num_M),collapse = " + ")))
  } else {
    formula(paste0("Y ~ X + ",
                   paste0("C",c(1:num_C),collapse = " + "),
                   " + ", paste0("M",c(1:num_M),collapse = " + ")))
  }
  
  tot = as.matrix(MCMCpack::MCMCregress(formula = form_Y, data = data,
                                        b0 = 0, B0 = 1,  # Precision 1 = variance 1
                                        c0 = 0.001, d0 = 0.001,
                                        mcmc = 10000, burnin = 5000, thin = 1, seed = seed))
  
  med_models <- list()
  for(i in 1:num_M){
    form_M = formula(paste0("M",i," ~ X + ",
                            paste0("C",c(1:num_C), collapse = " + ")))
    med_models[[i]] = as.matrix(MCMCpack::MCMCregress(formula = form_M, data = data,
                                                      b0 = 0, B0 = 1,
                                                      c0 = 0.001, d0 = 0.001,
                                                      mcmc = 10000, burnin = 5000, thin = 1, seed = seed))
  }
  
  n_samples <- nrow(tot)
  Inds_matrix = matrix(NA, nrow = n_samples, ncol = num_M)
  NDE_part2_matrix = matrix(0, nrow = n_samples, ncol = num_M)
  NDE_part3_matrix = matrix(0, nrow = n_samples, ncol = num_M)
  
  for(i in 1:num_M){
    a0k = med_models[[i]][,"(Intercept)"]
    a1k = med_models[[i]][,"X"]
    a2k = med_models[[i]][,"C1"]
    b2k = tot[,paste0("M",i)]
    
    if (interaction) {
      b3k = tot[,paste0("X:M",i)]
      Inds_matrix[,i] = a1k * (b2k + b3k)
      NDE_part2_matrix[,i] = b3k * a0k
      NDE_part3_matrix[,i] = b3k * a2k
    } else {
      Inds_matrix[,i] = a1k * b2k
    }
  }
  
  NIE_samples = rowSums(Inds_matrix)
  b1_samples = tot[,"X"]
  
  prob_C1 = mean(data$C1, na.rm = TRUE)
  
  if (interaction) {
    NDE_samples = b1_samples + rowSums(NDE_part2_matrix) + rowSums(NDE_part3_matrix)
  } else {
    NDE_samples = b1_samples
  }
  
  TE_samples = NIE_samples + NDE_samples
  med_res = t(apply(cbind(NIE_samples, NDE_samples, TE_samples), 2,
                    function(x){round(c(mean(x),sd(x),quantile(x,interval)), 4)}))
  
  if(print){
    colnames(med_res) = c("Estimate","Post.SD",
                          paste0(interval[1]*100,"%"), paste0(interval[2]*100,"%"))
    rownames(med_res) = c("NIE", "NDE", "TE")
    med_res = cbind(med_res,
                    ifelse(med_res[,paste0(interval[1]*100,"%")]*med_res[,paste0(interval[2]*100,"%")] > 0, "*", ""))
  }
  
  return(med_res)
}

make_boxplot = function(model, param, N, data){
  call_p = ifelse(param == "alpha", "alpha", param)
  df <- as.data.frame(model[[call_p]][, c(N:1)]) %>%
    setNames(paste0(param, "[", rev(colnames(data)), "]")) %>% 
    pivot_longer(everything(), names_to = "Column", values_to = "Value") %>%
    mutate(Column = factor(Column, levels = paste0(param, "[", rev(colnames(data)), "]")))
  
  summary_df <- df %>%
    group_by(Column) %>%
    summarise(
      Mean = mean(Value),
      SD = sd(Value),
      Lower = quantile(Value, 0.025),
      Upper = quantile(Value, 0.975)
    ) %>% 
    arrange(desc(Column))
  
  p1 = ggplot(df, aes(x = Column, y = Value)) +
    geom_boxplot() + 
    scale_y_continuous(limits = c(-6.5, 3.5), breaks = seq(-6, 4, by = 1)) +
    scale_x_discrete(labels = sapply(paste0(param,"[", rev(colnames(AL_eoi_d)), "]"), function(x) parse(text = x))) +
    geom_point(data = summary_df, aes(x = Column, y = Mean), color = "blue", size = 3) + 
    geom_errorbar(data = summary_df, aes(x = Column, y = Mean, ymin = Lower, ymax = Upper), width = 0.2, color = "red") +
    labs(x = "Parameters", y = parse(text=param)) + 
    theme(axis.text.y = element_text(size = 15, color = "black", face = "bold"),
          axis.text.x = element_text(size = 15, color = "black"),
          axis.title = element_text(size = 18, color = "black")) +
    coord_flip()
  
  return(p1)
}

make_adj_matrix = function(data, type = "exp", level = c("low", "medium", "high")) {
  if (type == "exp") {
    data_exp = data[!data$name %in% c("aud", "t"),]
    data_exp_g2 = data_exp %>% filter(group_size == 2)
    data_exp_g2 = data_exp_g2[!grepl(" ", data_exp_g2$ref, fixed = TRUE),]
    data_exp_g3 = data_exp %>% filter(group_size == 3)
    data_exp_g4 = data_exp %>% filter(group_size == 4)
    
    g3_refs = strsplit(data_exp_g3$ref, " ")
    g4_refs = strsplit(data_exp_g4$ref, " ")
    
    edgelist1 = data.frame(
      source = data_exp_g2$name,
      target = data_exp_g2$ref
    )
    
    edgelist2 = data.frame(
      source = rep(data_exp_g3$name, 2),
      target = c(sapply(g3_refs, `[`, 1), sapply(g3_refs, `[`, 2))
    )
    
    edgelist3 = data.frame(
      source = rep(data_exp_g4$name, 3),
      target = c(
        sapply(g4_refs, `[`, 1),
        sapply(g4_refs, `[`, 2),
        sapply(g4_refs, `[`, 3)
      )
    )
    
    E = rbind(edgelist1, edgelist2, edgelist3)
    G_exp = igraph::graph_from_edgelist(as.matrix(E), directed = T)
    A = as.matrix(igraph::as_adjacency_matrix(G_exp, sparse = FALSE, type = "both"))
    
  } else if (type == "eoi") {
    data_eoi = data[!data$name %in% c("aud", "t"),]
    data_eoi = data_eoi[data_eoi$level %in% level,]
    data_eoi = data_eoi[!is.na(data_eoi$ref),]
    
    edgelist_list = list()
    
    for (i in seq_len(nrow(data_eoi))) {
      source = data_eoi$name[i]
      refs = unlist(strsplit(data_eoi$ref[i], " "))
      refs = refs[refs != ""]
      if (length(refs) > 0) {
        edgelist_list[[i]] = data.frame(
          source = rep(source, length(refs)),
          target = refs,
          stringsAsFactors = FALSE
        )
      }
    }
    
    if (length(edgelist_list) == 0) {
      warning("No valid edges found.")
      return(matrix(0, nrow = 0, ncol = 0))
    }
    
    E = do.call(rbind, edgelist_list)
    G_eoi = igraph::graph_from_edgelist(as.matrix(E), directed = T)
    A = as.matrix(igraph::as_adjacency_matrix(G_eoi, sparse = FALSE, type = "both"))
  }
  
  return(A)
}

make_level_adj = function(data, level, scores){
  A_eoi_AL_dt = make_adj_matrix(data, type = "eoi", level = c(level))
  alidx_eoi = order(row.names(A_eoi_AL_dt))
  A_eoi_AL_dt = A_eoi_AL_dt[alidx_eoi,alidx_eoi]
  
  all_names = scores$s_l_name
  current_names = rownames(A_eoi_AL_dt)
  
  missing_names = setdiff(all_names, current_names)
  
  if(length(missing_names) > 0) {
    current_size = nrow(A_eoi_AL_dt)
    missing_size = length(missing_names)
    new_size = current_size + missing_size
    new_matrix = matrix(0, nrow = new_size, ncol = new_size)
    
    new_matrix[1:current_size, 1:current_size] = A_eoi_AL_dt
    new_rownames = c(current_names, missing_names)
    rownames(new_matrix) = new_rownames
    colnames(new_matrix) = new_rownames
    
    A_eoi_AL_dt = new_matrix
  }
  AL_eoi_d_idx = rownames(A_eoi_AL_dt) %in% all_names
  data_level = A_eoi_AL_dt[AL_eoi_d_idx, AL_eoi_d_idx]
  diag(data_level) = 0
  
  return(data_level)
}


make_skeleton_g = function(data){
  names <- unique(unlist(strsplit(data, " ")))
  adj_matrix <- matrix(0, nrow = length(names), ncol = length(names), dimnames = list(names, names))
  for (pair in data) {
    individuals <- strsplit(pair, " ")[[1]]
    for (i in 1:(length(individuals) - 1)) {
      for (j in (i + 1):length(individuals)) {
        adj_matrix[individuals[i], individuals[j]] <- adj_matrix[individuals[i], individuals[j]] + 1
        adj_matrix[individuals[j], individuals[i]] <- adj_matrix[individuals[j], individuals[i]] + 1
      }
    }
  }
  return(adj_matrix)
}

legend_gradient <- function(x, y, width, height, grad_colors, labels, title=NULL) {
  gradient <- as.raster((rev(matrix(grad_colors, nrow = 1))))
  rasterImage(gradient, x, y, x + width, y + height)
  text(x + width + 0.01, seq(y, y + height, length.out = length(labels)), labels, pos = 4, cex = 0.8)
  if (!is.null(title)) {
    text(x + width / 2, y + height + 0.02, title, pos = 3, cex = 0.9, font = 2) 
  }
}


calc_bic = function(model, X, vector = T, fix = T, covariate = F, missing = -99){
  
  data = model$data
  diag(data) = missing
  N = nrow(data)
  
  p = 2*N + 2*N*2 + ifelse(model$overdispersion,1,0) + ifelse(model$gamma_estimate != 1,1,0) + ifelse(covariate, ncol(model$delta),0)
  alpha.est = model$alpha_estimate
  beta.est = model$beta_estimate
  gamma.est = model$gamma_estimate
  z.est = model$z_estimate
  w.est = model$w_estimate
  delta.est = colMeans(model$delta)
  
  # Vector = TRUE only
  lik = 0
  mu.mat = matrix(0,N,N)
  for(i in 1:N){
    for(j in 1:N){
      if((i != j) & (data[i,j] != missing)){
        if(vector){
          if(covariate){
            if(fix){
              mu = exp(X[i,j,] %*% delta.est + alpha.est[i] + gamma.est*(z.est[i,] %*% w.est[j,])[1,1])  
            }else{
              mu = exp(X[i,j,] %*% delta.est + alpha.est[i] + beta.est[j] + gamma.est*(z.est[i,] %*% w.est[j,])[1,1])  
            }
          }else{
            if(fix){
              mu = exp(alpha.est[i] + gamma.est*(z.est[i,] %*% w.est[j,])[1,1])  
            }else{
              mu = exp(alpha.est[i] + beta.est[j] + gamma.est*(z.est[i,] %*% w.est[j,])[1,1])    
            }  
          }
        }else{
          mu = exp(alpha.est[i] + beta.est[j] - sqrt(sum((z.est[i,] - w.est[j,])^2)))
        }
        
        if(model$overdispersion){
          lik = lik + dnbinom(data[i,j], mu = mu, size = model$r_estimate, log = T)
        }else{
          lik = lik + dpois(data[i,j], lambda = mu, log=T)
        }
        mu.mat[i,j] = mu
      }
    }
  }
  
  Nm = length(data[data != missing])
  bic = -2 * lik + p * log(Nm)
  return(list(bic = bic,
              mu = mu.mat))
}

plot_model = function(model, b = 0.0, vector = T, fix = F, rotation = F, xrange = c(-2,2), yrange = c(-2,2), pos="z"){
  
  data = model$data
  
  if(ncol(model$z_estimate) == 2){
    z_estimate = model$z_estimate
    w_estimate = model$w_estimate
    
    if(rotation){
      rot = GPArotation::Varimax(z_estimate)
      z_estimate = rot$loadings
      w_estimate = w_estimate %*% t(solve(rot$Th))
    }
    
    z_estimate = as.data.frame(z_estimate)
    w_estimate = as.data.frame(w_estimate)
    
    colnames(z_estimate) = c("x", "y")
    colnames(w_estimate) = c("x", "y")
    
    z_estimate$name = rownames(data)
    w_estimate$name = rownames(data)
    
    
    if(fix){
      xran = xrange
      yran = yrange
    }else{
      xran = range(c(z_estimate$x - b, w_estimate$x + b))
      yran = range(c(z_estimate$y - b, w_estimate$y + b))
    }
    
    
    p1 <- ggplot() +
      geom_point(data = z_estimate, aes(x = x, y = y), color = "red") +
      geom_text_repel(data = z_estimate, aes(x = x, y = y, label = name, fontface = "bold"), size = 5) +
      xlim(xran) +
      ylim(yran) +
      ggtitle("Sender position") +
      theme_minimal() + 
      theme(
        plot.title = element_text(hjust = 0.0, size = 20),
        axis.title.x = element_text(size = 14),  # X-axis title size
        axis.title.y = element_text(size = 14),  # Y-axis title size
        axis.text.x = element_text(size = 12),   # X-axis text size
        axis.text.y = element_text(size = 12)    # Y-axis text size
      )
    
    p2 <- ggplot() +
      geom_point(data = w_estimate, aes(x = x, y = y), color = "red") +
      geom_text_repel(data = w_estimate, aes(x = x, y = y, label = name, fontface = "bold"), size = 5) +
      xlim(xran) +
      ylim(yran) +
      ggtitle("Receiver position") +
      theme_minimal() +
      theme(
        plot.title = element_text(hjust = 0.0, size = 20),
        axis.title.x = element_text(size = 14),  # X-axis title size
        axis.title.y = element_text(size = 14),  # Y-axis title size
        axis.text.x = element_text(size = 12),   # X-axis text size
        axis.text.y = element_text(size = 12)    # Y-axis text size
      )
    
    if(vector){
      p1 = p1 + geom_segment(data = z_estimate, aes(x = 0, y = 0, xend = x, yend = y), 
                             arrow = arrow(length = unit(0.2, "cm")))
      p2 = p2 + geom_segment(data = w_estimate, aes(x = 0, y = 0, xend = x, yend = y), 
                             arrow = arrow(length = unit(0.2, "cm")))
    }
    return(grid.arrange(p1, p2, ncol = 2))
  }else if(ncol(model$z_estimate) == 3){
    if(pos == "z"){
      data_matrix = model$z_estimate  
    }else{
      data_matrix = model$w_estimate  
    }
    
    x <- data_matrix[, 1]
    y <- data_matrix[, 2]
    z <- data_matrix[, 3]
    labels <- rownames(model$data)
    
    x_range <- xrange
    y_range <- yrange
    z_range <- xrange
    
    # Create an interactive 3D scatter plot with labels and individual arrows
    fig <- plot_ly() %>%
      # Add points with labels
      add_trace(
        x = x, y = y, z = z,
        type = 'scatter3d',
        mode = 'markers+text',            # Display markers and labels
        text = labels,                    # Add labels for each point
        textposition = "top right",       # Position labels
        marker = list(color = 'blue', size = 5),
        textfont = list(size = 20, color = 'black', family = "Arial Black"),
        showlegend = F
      )
    
    # Add arrows from (0,0,0) to each point
    for (i in 1:length(x)) {
      fig <- fig %>%
        add_trace(
          x = c(0, x[i]),                 # Start at origin, end at each point
          y = c(0, y[i]),
          z = c(0, z[i]),
          type = 'scatter3d',
          mode = 'lines',
          line = list(color = 'red', width = 2),
          showlegend = F
        )
    }
    
    # Set axis ranges to include origin
    fig <- fig %>%
      layout(
        title = list(
          text = ifelse(pos == "z","Sender","Receiver"),
          font = list(size = 30, color = "black"),
          x = 0.1,
          y = 0.95,
          xanchor = "center"
        ),
        scene = list(
          xaxis = list(title = '', range = x_range, showgrid = FALSE, showticklabels = FALSE, zeroline = FALSE),
          yaxis = list(title = '', range = y_range, showgrid = FALSE, showticklabels = FALSE, zeroline = FALSE),
          zaxis = list(title = '', range = z_range, showgrid = FALSE, showticklabels = FALSE, zeroline = FALSE)
        )
      )
    return(fig)
  }else{
    print("NA")
  }
}


acceptance_tab = function(model){
  D1 = data.frame(accept_alpha = model$accept_alpha,
                  accept_beta = model$accept_beta,
                  accept_z = model$accept_z,
                  accept_w = model$accept_w)
  D2 = data.frame(accept_gamma = model$accept_gamma,
                  accept_r = model$accept_r)  
  return(list(D1 = D1,
              D2 = D2))
}

