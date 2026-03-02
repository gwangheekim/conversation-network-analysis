#' @keywords internal
"_PACKAGE"

#' nbamen: AMEN Model for Count Data with Negative Binomial Distribution
#'
#' This package implements Additive and Multiplicative Effects Network (AMEN) models 
#' for count data using negative binomial distribution with MCMC algorithms. The package 
#' provides functions for fitting network models with additive row/column effects and 
#' multiplicative latent factors, including support for overdispersion.
#'
#' @section Main Functions:
#' \describe{
#'   \item{amen_count_nb}{Fits an AMEN model for count data using negative binomial distribution with MCMC}
#' }
#'
#' @section Key Features:
#' \itemize{
#'   \item MCMC sampling with customizable priors and jump sizes
#'   \item Additive row and column effects modeling
#'   \item Multiplicative latent factor modeling
#'   \item Procrustes matching for post-processing MCMC samples
#'   \item Comprehensive model diagnostics (BIC, WAIC, DIC)
#'   \item Support for overdispersion and zero-inflation
#'   \item Hierarchical modeling capabilities
#' }
#'
#' @name nbamen-package
#' @aliases nbamen
#' @useDynLib nbamen, .registration = TRUE
#' @importFrom Rcpp sourceCpp
NULL
