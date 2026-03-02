
#' AMEN Count Model with Negative Binomial Distribution
#'
#' Fits an Additive and Multiplicative Effects Network (AMEN) model for count data 
#' using negative binomial distribution with MCMC algorithm
#'
#' @param data A matrix or data frame containing count response data
#' @param X Design matrix for covariates 
#' @param ndim Number of latent dimensions (default: 2)
#' @param niter Number of MCMC iterations (default: 15000)
#' @param nburn Number of burn-in iterations (default: 2500)
#' @param nthin Thinning interval (default: 5)
#' @param nprint Print interval for progress (default: 500)
#' @param pr_sd_z Prior standard deviation for z parameters (default: 0.5)
#' @param pr_sd_w Prior standard deviation for w parameters (default: 0.5)
#' @param jump_beta Jump size for beta parameters (default: 0.4)
#' @param jump_alpha Jump size for alpha parameters (default: 1)
#' @param jump_gamma Jump size for gamma parameter (multiplicative weight term) (default: 0.025)
#' @param jump_z Jump size for z parameters (default: 0.5)
#' @param jump_w Jump size for w parameters (default: 0.5)
#' @param jump_eta Jump size for eta parameters (default: 1)
#' @param jump_delta Jump size for delta parameters (default: 1)
#' @param eta_init Initial value for eta (default: 0.0)
#' @param r_init Initial value for r parameter (default: 1)
#' @param pr_mean_beta Prior mean for beta parameters (default: 0)
#' @param pr_sd_beta Prior standard deviation for beta parameters (default: 1)
#' @param pr_mean_alpha Prior mean for alpha parameters (default: 0)
#' @param pr_sd_alpha Prior standard deviation for alpha parameters (default: 1)
#' @param jump_r Jump size for r parameter (default: 0.025)
#' @param pr_mean_r Prior mean for r parameter (default: 0.5)
#' @param pr_sd_r Prior standard deviation for r parameter (default: 1)
#' @param pr_mean_delta Prior mean for delta parameters (default: 0)
#' @param pr_sd_delta Prior standard deviation for delta parameters (default: 1)
#' @param pr_mean_gamma Prior mean for gamma parameter (multiplicative weight term) (default: 0.5)
#' @param pr_sd_gamma Prior standard deviation for gamma parameter (multiplicative weight term) (default: 1)
#' @param pr_a_beta Prior shape parameter for beta (default: 0.001)
#' @param pr_b_beta Prior rate parameter for beta (default: 0.001)
#' @param pr_a_alpha Prior shape parameter for alpha (default: 0.001)
#' @param pr_b_alpha Prior rate parameter for alpha (default: 0.001)
#' @param pr_a_eta Prior shape parameter for eta (default: 1)
#' @param pr_b_eta Prior rate parameter for eta (default: 1)
#' @param direct Logical for direct sampling (default: TRUE)
#' @param overdispersion Logical for overdispersion (default: FALSE)
#' @param zeroinflate Logical for zero inflation (default: FALSE)
#' @param fix Logical for fixed parameters (default: FALSE)
#' @param fixsd Logical for fixed standard deviation (default: FALSE)
#' @param missing Missing value indicator (default: -99)
#' @param verbose Logical for verbose output (default: FALSE)
#' @param single Logical for single chain (default: FALSE)
#' @param singledist Logical for single distribution (default: TRUE)
#' @param vector Logical for vector operations (default: FALSE)
#' @param test Logical for test mode (default: FALSE)
#' @param fix_r Logical for fixed r parameter (default: FALSE)
#' @param covariate Logical for covariate inclusion (default: FALSE)
#' @param hierarchical_r Logical for hierarchical r (default: FALSE)
#' @param pr_a_r_sd Prior shape for r standard deviation (default: 0.001)
#' @param pr_b_r_sd Prior rate for r standard deviation (default: 0.001)
#' @param skip Logical to skip certain computations (default: FALSE)
#'
#' @return A list containing:
#' \item{data}{Original data matrix}
#' \item{bic}{Bayesian Information Criterion}
#' \item{waic}{Widely Applicable Information Criterion}
#' \item{dic}{Deviance Information Criterion}
#' \item{beta_estimate}{Posterior mean estimates for additive row effects}
#' \item{alpha_estimate}{Posterior mean estimates for additive column effects}
#' \item{gamma_estimate}{Posterior mean estimate for multiplicative weight parameter}
#' \item{r_estimate}{Posterior mean estimate for overdispersion parameter}
#' \item{z_estimate}{Posterior mean estimates for row latent positions}
#' \item{w_estimate}{Posterior mean estimates for column latent positions}
#' \item{mcmc_inf}{MCMC information including burn-in, iterations, and thinning}
#' \item{map_inf}{Maximum a posteriori information}
#' \item{beta}{Full posterior samples for additive row effects}
#' \item{alpha}{Full posterior samples for additive column effects}
#' \item{gamma}{Full posterior samples for multiplicative weight parameter}
#' \item{r}{Full posterior samples for overdispersion parameter}
#' \item{z}{Procrustes-aligned posterior samples for row latent positions}
#' \item{w}{Procrustes-aligned posterior samples for column latent positions}
#' \item{accept_*}{Acceptance rates for various parameters}
#'
#' @examples
#' \dontrun{
#' # Example usage with real data
#' # AL_exp_d should be your count data matrix
#' # X should be your design matrix
#' 
#' model_exp_neg <- amen_count_nb(data=AL_exp_d, X = X, niter=100000, nburn=10000,
#'                                nthin = 5, overdispersion = TRUE, zeroinflate = FALSE,
#'                                jump_r = 0.5, r_init = 1,
#'                                verbose = FALSE, direct = TRUE, fix = TRUE, fixsd = TRUE,
#'                                jump_gamma = 0.5, jump_beta = 2.5, jump_alpha = 2.5,
#'                                jump_z = 0.6, jump_w = 0.6, pr_sd_z = 1, pr_sd_w = 1,
#'                                pr_sd_beta = 3, pr_sd_alpha = 3,
#'                                pr_mean_r = log(1), pr_sd_r = 0.15,
#'                                eta_init = 0.5, pr_a_eta = 1, pr_b_eta = 1, jump_eta = 0.3,
#'                                pr_mean_gamma = 0.5, pr_sd_gamma = 1, ndim = 2,
#'                                singledist = FALSE, single = FALSE, hierarchical_r = FALSE,
#'                                vector = TRUE, missing = -99, covariate = FALSE, fix_r = FALSE)
#'
#' }
#'
#' @export
#' @importFrom MCMCpack procrustes
#' @importFrom utils txtProgressBar setTxtProgressBar
#' @importFrom stats quantile
#' @useDynLib nbamen, .registration = TRUE
#' @importFrom Rcpp sourceCpp
amen_count_nb = function(data, X, ndim = 2, niter = 15000, nburn = 2500, nthin = 5, nprint = 500, pr_sd_z = 0.5, pr_sd_w = 0.5,
                    jump_beta = 0.4, jump_alpha = 1, jump_gamma = 0.025, jump_z = 0.5, jump_w = 0.5, jump_eta = 1, jump_delta = 1, eta_init = 0.0, r_init = 1,
                    pr_mean_beta = 0, pr_sd_beta = 1, pr_mean_alpha = 0, pr_sd_alpha = 1, jump_r = 0.025, pr_mean_r = 0.5, pr_sd_r = 1,
                    pr_mean_delta = 0, pr_sd_delta = 1, pr_mean_gamma = 0.5, pr_sd_gamma = 1,
                    pr_a_beta = 0.001, pr_b_beta = 0.001, pr_a_alpha = 0.001, pr_b_alpha = 0.001, pr_a_eta = 1, pr_b_eta = 1,
                    direct = TRUE, overdispersion = FALSE, zeroinflate = FALSE, fix = FALSE, fixsd = FALSE, missing = -99,
                    verbose = FALSE, single = FALSE, singledist = TRUE, vector = FALSE, test = FALSE, fix_r = FALSE,
                    covariate = FALSE, hierarchical_r = FALSE, pr_a_r_sd = 0.001, pr_b_r_sd = 0.001, skip = FALSE) {

    if(is.data.frame(data)){
        cname = colnames(data)
    }else{
        cname = paste("Node", 1:ncol(data), sep=" ")
    }

    cat("\n\nFitting with MCMC algorithm with count\n")
    output = amen_count_nb_cpp(data=as.matrix(data), X = X, ndim=ndim, niter=niter, nburn=nburn, nthin=nthin, nprint=nprint,
                                    jump_beta=jump_beta, jump_alpha=jump_alpha, jump_gamma = jump_gamma,
                                    jump_z=jump_z, jump_w=jump_w, r_init=r_init, jump_delta = jump_delta,
                                    pr_mean_beta=pr_mean_beta, pr_sd_beta=pr_sd_beta, pr_mean_alpha=pr_mean_alpha, pr_sd_alpha=pr_sd_alpha,
                                    pr_sd_z = pr_sd_z, pr_sd_w = pr_sd_w, pr_mean_delta = pr_mean_delta, pr_sd_delta = pr_sd_delta,
                                    jump_r=jump_r, pr_mean_r=pr_mean_r, pr_sd_r = pr_sd_r,
                                    pr_mean_gamma = pr_mean_gamma, pr_sd_gamma = pr_sd_gamma,
                                    pr_a_beta=pr_a_beta, pr_b_beta=pr_b_beta, pr_a_alpha=pr_a_alpha,
                                    pr_b_alpha=pr_b_alpha, pr_a_r_sd=pr_a_r_sd, pr_b_r_sd=pr_b_r_sd,
                                    fix = fix, fixsd = fixsd, missing = missing,
                                    singledist = singledist, hierarchical_r = hierarchical_r, fix_r = fix_r,
                                    vector = vector, covariate = covariate, verbose=verbose)

    mcmc.inf = list(nburn=nburn, niter=niter, nthin=nthin)
    nsample = nrow(data)
    nitem = ncol(data)

    nmcmc = as.integer((niter - nburn) / nthin)
    max.address = min(which.max(output$map))
    map.inf = data.frame(value = output$map[which.max(output$map)], iter = which.max(output$map))
    w.star = output$w[max.address,,]
    z.star = output$z[max.address,,]
    w.proc = array(0,dim=c(nmcmc,nitem,ndim))
    z.proc = array(0,dim=c(nmcmc,nsample,ndim))

    cat("\n\nProcrustes Matching Analysis\n")
    pb = txtProgressBar(title = "progress bar", min = 0, max = nmcmc,
                        style = 3, width = 50)

    for(iter in 1:nmcmc){
        z.iter = matrix(output$z[iter,,], ncol = dim(output$z)[3])
        w.iter = matrix(output$w[iter,,], ncol = dim(output$w)[3])

        if(iter != max.address){
            z.proc[iter,,] = MCMCpack::procrustes(z.iter, z.star)$X.new
            w.proc[iter,,] = MCMCpack::procrustes(w.iter, w.star)$X.new
        } else {
            z.proc[iter,,] = z.iter
            w.proc[iter,,] = w.iter
        }
        setTxtProgressBar(pb, iter)
    }

    w.est = colMeans(w.proc, dims = 1)
    z.est = colMeans(z.proc, dims = 1)

    beta.estimate = apply(output$beta, 2, mean)
    alpha.estimate = apply(output$alpha, 2, mean)
    sigma_alpha.estimate = mean(output$sigma_alpha)
    gamma.estimate = mean(output$gamma)
    r.estimate = mean(output$r)
    eta.estimate = 0.0  # Default value as specified in function parameters
    beta.summary = data.frame(cbind(apply(output$beta, 2, mean), t(apply(output$beta, 2, function(x) quantile(x, probs = c(0.025, 0.975))))))
    colnames(beta.summary) <- c("Estimate", "2.5%", "97.5%")
    rownames(beta.summary) <- cname

    # Calculate BIC
    cat("\n\nCalculate BIC\n")
    log_like = log_likelihood_count_cpp(data = as.matrix(data), ndim = ndim, beta_est = as.matrix(beta.estimate), alpha_est = as.matrix(alpha.estimate),
                                        gamma_est = gamma.estimate, z_est = z.est, w_est = w.est, r_est = r.estimate, eta_est = eta.estimate,
                                        zeroinflate = zeroinflate, overdispersion = overdispersion, vector = vector)

    if(fix){
        p = nitem + nsample + 1 + ndim * nitem + ndim * nsample # r, betas, alphas, d*nrow, d*ncol  
    }else{
        p = nitem + nsample + 1 + 1 + ndim * nitem + ndim * nsample
    }
    bic = -2 * log_like[[1]] + p * log(nsample * (nsample-1)) 
    
    # Calculate wAIC and DIC
    cat("\n\nCalculate wAIC and DIC\n")
    waic_dic_result <- calculate_waic_dic_procrustes_cpp(
        data = as.matrix(data),
        beta_samples = output$beta,
        alpha_samples = output$alpha,
        gamma_samples = output$gamma,
        r_samples = output$r,
        z_proc = z.proc,
        w_proc = w.proc,
        ndim = ndim,
        overdispersion = overdispersion,
        zeroinflate = zeroinflate,
        missing = missing,
        vector = vector
    )

    waic <- waic_dic_result$waic
    dic <- waic_dic_result$dic
    lppd <- waic_dic_result$lppd
    p_waic <- waic_dic_result$p_waic
    p_d <- waic_dic_result$p_d

    result <- list(data = data,
            bic = bic,
            waic = waic,
            dic = dic,
            dic_pd = p_d,
            lik = output$map,
            mcmc_inf = mcmc.inf,
            map_inf = map.inf,
            beta_estimate  = beta.estimate,
            beta_summary = beta.summary,
            alpha_estimate = alpha.estimate,
            sigma_alpha_estimate    = sigma_alpha.estimate,
            gamma_estimate = gamma.estimate,
            r_estimate     = r.estimate,
            z_estimate     = z.est,
            w_estimate     = w.est,
            beta           = output$beta,
            beta_sd        = output$sigma_beta,
            alpha          = output$alpha,
            alpha_sd       = output$sigma_alpha,
            delta          = output$delta,
            delta_sd       = output$sigma_delta,                     
            gamma          = output$gamma,
            r              = output$r,
            r_sd           = output$sigma_r,
            z              = z.proc,
            w              = w.proc,
            z_raw          = output$z,
            w_raw          = output$w,
            accept_beta    = output$accept_beta,
            accept_alpha   = output$accept_alpha,
            accept_delta   = output$accept_delta,
            accept_w       = output$accept_w,
            accept_z       = output$accept_z,
            accept_gamma   = output$accept_gamma,
            accept_r       = output$accept_r)
    
    return(result)
}
