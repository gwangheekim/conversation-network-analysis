
#include <RcppArmadillo.h>
#include <iostream>
#include <cstdio>
#include <chrono>
#include "progress.h"
using namespace arma;

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(cpp11)]]

arma::vec get_slices2(const arma::cube& A, int row_index, int col_index) {
  int num_slices = A.n_slices;
  arma::vec result(num_slices);
  
  int r = row_index;
  int c = col_index;
  
  for (int s = 0; s < num_slices; s++) {
    result(s) = A(r, c, s);
  }
  
  return result;
}

// [[Rcpp::export]]
Rcpp::List amen_count_nb_cpp(arma::mat data, arma::cube X, const int ndim, const int niter, const int nburn, const int nthin, const int nprint,
                                 const double jump_beta, const double jump_alpha, const double jump_z,const double jump_w, 
                                 const double jump_delta, const double r_init,
                                 const double pr_mean_beta, double pr_sd_beta, double pr_a_beta, double pr_b_beta, const double pr_a_alpha, const double pr_b_alpha,
                                 const double pr_mean_alpha, double pr_sd_alpha, const double pr_mean_delta, double pr_sd_delta,
                                 const double pr_sd_z, const double pr_sd_w,
                                 const double pr_mean_gamma, const double pr_sd_gamma, const double jump_r, const double pr_mean_r, double pr_sd_r, const double jump_gamma,
                                 const double pr_a_r_sd, const double pr_b_r_sd, const bool hierarchical_r, const bool fix_r,
                                 const bool fix, const double missing, const bool fixsd, const bool singledist, const bool vector, const bool covariate, const bool verbose){
  
  
  int i, j, k, i2, j2, k2, count = 0, accept = 0;
  double num, den, old_like_beta, new_like_beta, old_like_alpha, new_like_alpha, old_like_delta, new_like_delta;
  // double old_like_z, new_like_z, old_like_w, new_like_w, old_like_gamma, new_like_gamma ;
  // double ratio, un, post_a, post_b, dist_temp, dist_old_temp, dist_new_temp;
  double old_like_gamma, new_like_gamma;
  double ratio, un, post_a, post_b, dist_temp;
  double rold, rnew; 
  double pr_mean_z = 0.0, pr_mean_w = 0.0, post;
  
  const int N = data.n_rows;
  const int P = data.n_cols;
  const int K = X.n_slices;
  
  // row effect   
  arma::dvec oldalpha(N, fill::randn);
  arma::dvec newalpha = oldalpha;
  
  // column effect
  arma::dvec oldbeta(P, fill::randn);
  arma::dvec newbeta = oldbeta;  
  
  // fixed effect
  arma::dvec olddelta(K, fill::randn);
  arma::dvec newdelta = olddelta;
  
  // row latent
  arma::dmat oldz(N,ndim,fill::randn);
  arma::dmat newz = oldz;
  
  // col latent
  arma::dmat oldw(P,ndim,fill::randn);
  arma::dmat neww = oldw;
  
  double oldgamma=1, newgamma=1;
  double pointwise_loglik = 0.0;
  
  arma::dmat samp_alpha((niter-nburn)/nthin, N, fill::zeros);
  arma::dmat samp_beta((niter-nburn)/nthin, P, fill::zeros);
  arma::dmat samp_delta((niter-nburn)/nthin, K, fill::zeros);
  arma::dcube samp_z(((niter-nburn)/nthin), N, ndim, fill::zeros);
  arma::dcube samp_w(((niter-nburn)/nthin), P, ndim, fill::zeros);
  arma::dvec samp_sd_alpha((niter-nburn)/nthin, fill::zeros);
  arma::dvec samp_sd_beta((niter-nburn)/nthin, fill::zeros);
  arma::dvec samp_sd_delta((niter-nburn)/nthin, fill::zeros);
  arma::dvec sample_post((niter-nburn)/nthin, fill::zeros);
  arma::dvec samp_gamma(((niter-nburn)/nthin), fill::zeros);
  arma::dvec samp_r(((niter-nburn)/nthin), fill::zeros);
  arma::dvec samp_sd_r(((niter-nburn)/nthin), fill::zeros);
  
  // Add pointwise log-likelihood storage for WAIC calculation
  int n_obs = 0;
  for(i = 0; i < N; i++){
    for(k = 0; k < P; k++){
      if((i != k) & (data(i,k) != missing)){
        n_obs++;
      }
    }
  }
  arma::dmat samp_pointwise_loglik(((niter-nburn)/nthin), n_obs, fill::zeros);
  
  arma::dvec accept_alpha(N, fill::zeros);
  arma::dvec accept_beta(P, fill::zeros);  
  arma::dvec accept_delta(K, fill::zeros);
  arma::dvec accept_z(N, fill::zeros);
  arma::dvec accept_w(P, fill::zeros);
  double accept_gamma=0;
  
  accept = count = 0;
  
  arma::dmat dist(N,P,fill::zeros);
  arma::dvec old_dist_k(N,fill::zeros);
  arma::dvec new_dist_k(N,fill::zeros);
  arma::dvec old_dist_i(P,fill::zeros);
  arma::dvec new_dist_i(P,fill::zeros);
  
  rold = r_init;
  double new_like_r, old_like_r, accept_r = 0;
  
  for(int iter = 0; iter < niter; iter++){
    if (iter % 5 == 0){
      Rcpp::checkUserInterrupt();
    }
    
    if(vector){
      if(singledist){
        dist = oldz * oldz.t();
      }else{
        dist = oldz * oldw.t();
      }
      dist.diag().zeros();
    }else{
      dist.fill(0.0);
      for(i = 0; i < N; i++){
        for(k = 0; k < P; k++){
          dist_temp = 0.0;
          if((i != k) & (data(i,k) != missing)){
            if(singledist){              
              for(j = 0; j < ndim; j++) dist_temp += std::pow((oldz(i,j)-oldz(k,j)), 2.0);                            
            }else{
              for(j = 0; j < ndim; j++) dist_temp += std::pow((oldz(i,j)-oldw(k,j)), 2.0);
            }
          }        
          dist(i,k) = std::sqrt(dist_temp);
        }
      }
    }
    
    // alpha update    
    for(i = 0; i < N; i++){
      newalpha(i) = oldalpha(i) + R::rnorm(0, jump_alpha);
      old_like_alpha = new_like_alpha = 0.0;
      
      for(k = 0; k < P; k++){
        if((i != k)  & (data(i,k) != missing)){
          new_like_alpha += dnbinom_mu(data(i,k), rold, exp(arma::dot(get_slices2(X,i,k), olddelta) + newalpha(i) + oldbeta(k) + (vector ? 1 : -1) * oldgamma * dist(i,k)), 1);
          old_like_alpha += dnbinom_mu(data(i,k), rold, exp(arma::dot(get_slices2(X,i,k), olddelta) + oldalpha(i) + oldbeta(k) + (vector ? 1 : -1) * oldgamma * dist(i,k)), 1);
        }        
      }
      
      num = new_like_alpha + R::dnorm4(newalpha(i), pr_mean_alpha, pr_sd_alpha, 1);
      den = old_like_alpha + R::dnorm4(oldalpha(i), pr_mean_alpha, pr_sd_alpha, 1);
      ratio = num - den;
      
      if(ratio > 0.0) accept = 1;
      else{
        un = R::runif(0,1);
        if(std::log(un) < ratio) accept = 1;
        else accept = 0;
      }
      
      if(accept == 1){
        oldalpha(i) = newalpha(i);
        accept_alpha(i) += 1.0 / (niter * 1.0);
      }
    }
    
    
    // beta update
    for(k = 0; k < P; k++){
      newbeta(k) = oldbeta(k) + R::rnorm(0, jump_beta);
      old_like_beta = new_like_beta = 0.0;
      
      for(i = 0; i < N; i++){
        if((i != k)  & (data(i,k) != missing)){
          new_like_beta += dnbinom_mu(data(i,k), rold, exp(arma::dot(get_slices2(X,i,k), olddelta) + oldalpha(i) + newbeta(k) + (vector ? 1 : -1) * oldgamma * dist(i,k)), 1);
          old_like_beta += dnbinom_mu(data(i,k), rold, exp(arma::dot(get_slices2(X,i,k), olddelta) + oldalpha(i) + oldbeta(k) + (vector ? 1 : -1) * oldgamma * dist(i,k)), 1);
        }        
      }
      
      num = new_like_beta + R::dnorm4(newbeta(k), pr_mean_beta, pr_sd_beta, 1);
      den = old_like_beta + R::dnorm4(oldbeta(k), pr_mean_beta, pr_sd_beta, 1);
      ratio = num - den;
      
      if(ratio > 0.0) accept = 1;
      else{
        un = R::runif(0,1);
        if(std::log(un) < ratio) accept = 1;
        else accept = 0;
      }
      
      if(accept == 1){
        oldbeta(k) = newbeta(k);
        accept_beta(k) += 1.0 / (niter * 1.0);
      }
    }   
    
    
    // gamma update
    if(fix){
      oldgamma = 1.0;
    }else{
      newgamma = R::rlnorm(std::log(oldgamma), jump_gamma);
      old_like_gamma = new_like_gamma = 0.0;
      for(i = 0; i < N; i++){
        for(k = 0; k < P; k++){
          if((i != k) & (data(i,k) != missing)){
            new_like_gamma += dnbinom_mu(data(i,k),rold,exp(arma::dot(get_slices2(X,i,k), olddelta) + oldalpha(i) + oldbeta(k) + (vector ? 1 : -1) * newgamma * dist(i,k)),1);
            old_like_gamma += dnbinom_mu(data(i,k),rold,exp(arma::dot(get_slices2(X,i,k), olddelta) + oldalpha(i) + oldbeta(k) + (vector ? 1 : -1) * oldgamma * dist(i,k)),1);
          }
        }
      }
      
      num = new_like_gamma + R::dlnorm(oldgamma, std::log(newgamma), jump_gamma, 1) + R::dlnorm(newgamma, pr_mean_gamma, pr_sd_gamma, 1);
      den = old_like_gamma + R::dlnorm(newgamma, std::log(oldgamma), jump_gamma, 1) + R::dlnorm(oldgamma, pr_mean_gamma, pr_sd_gamma, 1);
      ratio = num - den;
      
      if(ratio > 0.0) accept = 1;
      else{
        un = R::runif(0,1);
        if(std::log(un) < ratio) accept = 1;
        else accept = 0;
      }
      
      if(accept == 1){
        oldgamma = newgamma;
        accept_gamma += 1.0 / (niter * 1.0);
      }
    }
    
    
    // ----------------------------------------------
    // Precompute constants reused in MALA updates
    // ----------------------------------------------
    const bool is_vector = vector;
    const bool is_single = singledist;
    const double sgn = is_vector ? 1.0 : -1.0;           // vector: +1, distance: -1
    const double weight = is_single ? 0.5 : 1.0;         // single-dist mixture weight
    const double gamma_c = oldgamma;                     // γ
    const double rsize = rold;                           // NB size r
    const int D = ndim, I = N, J = P;
    const double inv_sigma2_z = 1.0 / (pr_sd_z * pr_sd_z);
    const double inv_sigma2_w = 1.0 / (pr_sd_w * pr_sd_w);
    const double eps_z = jump_z;
    const double half_eps2_z = 0.5 * eps_z * eps_z;
    const double inv_eps2_z = 1.0 / (eps_z * eps_z);
    const double eps_w = jump_w;
    const double half_eps2_w = 0.5 * eps_w * eps_w;
    const double inv_eps2_w = 1.0 / (eps_w * eps_w);
    
    // ----------------------------------------------
    // z_i update (MALA)
    // ----------------------------------------------
    for (i = 0; i < I; i++) {
      arma::rowvec zi = oldz.row(i);
      
      // eval_z(z): log-likelihood and gradient wrt z_i
      auto eval_z = [&](const arma::rowvec& zcur, double& loglik, arma::rowvec& grad) {
        // NB(mean=μ, size=r), vector mode: η_ik = x_ik^T δ + θ_i + β_k + γ z_i^T w_k, μ_ik = exp(η_ik)
        // dℓ/dη_ik = r (y_ik − μ_ik) / (r + μ_ik)  := g_ik
        // ∇_{z_i} log π = Σ_k [ g_ik · γ · w_k ] − (z_i − μ_z)/σ_z^2
        loglik = 0.0;
        grad.set_size(D);
        grad.zeros();
        
        for (k2 = 0; k2 < J; k2++) {
          if ((i != k2) && (data(i, k2) != missing)) {
            arma::rowvec base(D, arma::fill::zeros);
            double dist_ik = 0.0;
            
            if (is_vector) {
              // vector(inner product): base = w_k, dist = z_i^T w_k
              base = oldw.row(k2);
              dist_ik = arma::dot(zcur, base);
            } else {
              // distance mode (unchanged)
              arma::rowvec cp = is_single ? oldz.row(k2) : oldw.row(k2);
              arma::rowvec diff = zcur - cp;
              dist_ik = std::sqrt(arma::dot(diff, diff));
              if (dist_ik > 1e-12) base = diff / dist_ik;
            }
            
            const double L = arma::dot(get_slices2(X, i, k2), olddelta) + oldalpha(i) + oldbeta(k2) + sgn * gamma_c * dist_ik;
            const double mu = std::exp(L);
            const int y = (int)data(i, k2);
            
            // log-likelihood contribution
            loglik += weight * R::dnbinom_mu(y, rsize, mu, 1);
            
            // g_ik = dℓ/dη = r (y − μ)/(r + μ); since L = η, dℓ/dL = g_ik
            const double dlog_dL = mu * ((double)y / mu - ((double)y + rsize) / (rsize + mu)); // == r(y−μ)/(r+μ)
            grad += weight * (dlog_dL * sgn * gamma_c) * base; // vector mode: base = w_k
          }
        }
        
        // Gaussian prior gradient on z_i: −(z_i − μ_z)/σ_z^2
        for (j2 = 0; j2 < D; j2++) {
          grad(j2) += - (zcur(j2) - pr_mean_z) * inv_sigma2_z;
        }
      };
      
      // Step 1. Evaluate at current z_i
      double like_old_i = 0.0;
      arma::rowvec grad_old_i(D, arma::fill::zeros);
      eval_z(zi, like_old_i, grad_old_i);
      
      double prior_old_i = 0.0;
      for (j2 = 0; j2 < D; j2++) prior_old_i += R::dnorm4(zi(j2), pr_mean_z, pr_sd_z, 1);
      
      // Step 2. MALA proposal: z* = z + (ε^2/2) ∇log π(z) + ε · N(0, I)
      arma::rowvec mean_old = zi + half_eps2_z * grad_old_i;
      
      arma::rowvec noise(D);
      for (j2 = 0; j2 < D; j2++) noise(j2) = R::rnorm(0.0, 1.0);
      arma::rowvec zi_prop = mean_old + eps_z * noise;
      
      // Step 3. Evaluate at proposal
      double like_new_i = 0.0;
      arma::rowvec grad_new_i(D, arma::fill::zeros);
      eval_z(zi_prop, like_new_i, grad_new_i);
      
      double prior_new_i = 0.0;
      for (j2 = 0; j2 < D; j2++) prior_new_i += R::dnorm4(zi_prop(j2), pr_mean_z, pr_sd_z, 1);
      
      // Step 4. MALA proposal densities (q)
      // log q(z'|z) = − 1/(2 ε^2) || z' − z − (ε^2/2) ∇logπ(z) ||^2 + const
      double log_q_new_given_old = -0.5 * inv_eps2_z * arma::dot(zi_prop - mean_old, zi_prop - mean_old);
      arma::rowvec mean_new = zi_prop + half_eps2_z * grad_new_i;
      double log_q_old_given_new = -0.5 * inv_eps2_z * arma::dot(zi - mean_new, zi - mean_new);
      
      // Step 5. MH accept ratio:
      // log a = [log π(z*) − log π(z)] + [log q(z|z*) − log q(z*|z)]
      num = like_new_i + prior_new_i + log_q_old_given_new;
      den = like_old_i + prior_old_i + log_q_new_given_old;
      ratio = num - den;
      
      if (ratio > 0.0) accept = 1;
      else {
        un = R::runif(0, 1);
        accept = (std::log(un) < ratio) ? 1 : 0;
      }
      
      if (accept == 1) {
        oldz.row(i) = zi_prop;
        accept_z(i) += 1.0 / (niter * 1.0);
      }
    }
    
    // ----------------------------------------------
    // w_k update (MALA)
    // ----------------------------------------------
    if (singledist) {
      oldw.fill(0);
    } else {
      for (k = 0; k < J; k++) {
        arma::rowvec wk = oldw.row(k);
        
        // eval_w(w): log-likelihood and gradient wrt w_k
        auto eval_w = [&](const arma::rowvec& wcur, double& loglik, arma::rowvec& grad) {
          // NB(mean=μ, size=r), vector mode: η_ik = x_ik^T δ + θ_i + β_k + γ z_i^T w_k, μ_ik = exp(η_ik)
          // dℓ/dη_ik = r (y_ik − μ_ik) / (r + μ_ik)  := g_ik
          // ∇_{w_k} log π = Σ_i [ g_ik · γ · z_i ] − (w_k − μ_w)/σ_w^2
          loglik = 0.0;
          grad.set_size(D);
          grad.zeros();
          
          for (i2 = 0; i2 < I; i2++) {
            if ((i2 != k) && (data(i2, k) != missing)) {
              arma::rowvec base(D, arma::fill::zeros);
              double dist_ik = 0.0;
              
              if (is_vector) {
                // vector(inner product): base = z_i, dist = z_i^T w_k
                base = oldz.row(i2);
                dist_ik = arma::dot(base, wcur);
              } else {
                // distance mode (unchanged)
                arma::rowvec diff = wcur - oldz.row(i2); // d dist / d w
                dist_ik = std::sqrt(arma::dot(diff, diff));
                if (dist_ik > 1e-12) base = diff / dist_ik;
              }
              
              const double L = arma::dot(get_slices2(X, i2, k), olddelta) + oldalpha(i2) + oldbeta(k) + sgn * gamma_c * dist_ik;
              const double mu = std::exp(L);
              const int y = (int)data(i2, k);
              
              // log-likelihood contribution
              loglik += R::dnbinom_mu(y, rsize, mu, 1);
              
              // g_ik and gradient
              const double dlog_dL = mu * ((double)y / mu - ((double)y + rsize) / (rsize + mu)); // == r(y−μ)/(r+μ)
              grad += (dlog_dL * sgn * gamma_c) * base; // vector mode: base = z_i
            }
          }
          
          // Gaussian prior gradient on w_k: −(w_k − μ_w)/σ_w^2
          for (j2 = 0; j2 < D; j2++) {
            grad(j2) += - (wcur(j2) - pr_mean_w) * inv_sigma2_w;
          }
        };
        
        // Step 1. Evaluate at current w_k
        double like_old_k = 0.0;
        arma::rowvec grad_old_k(D, arma::fill::zeros);
        eval_w(wk, like_old_k, grad_old_k);
        
        double prior_old_k = 0.0;
        for (j2 = 0; j2 < D; j2++) prior_old_k += R::dnorm4(wk(j2), pr_mean_w, pr_sd_w, 1);
        
        // Step 2. MALA proposal: w* = w + (ε^2/2) ∇log π(w) + ε · N(0, I)
        arma::rowvec mean_old_w = wk + half_eps2_w * grad_old_k;
        
        arma::rowvec noise_w(D);
        for (j2 = 0; j2 < D; j2++) noise_w(j2) = R::rnorm(0.0, 1.0);
        arma::rowvec wk_prop = mean_old_w + eps_w * noise_w;
        
        // Step 3. Evaluate at proposal
        double like_new_k = 0.0;
        arma::rowvec grad_new_k(D, arma::fill::zeros);
        eval_w(wk_prop, like_new_k, grad_new_k);
        
        double prior_new_k = 0.0;
        for (j2 = 0; j2 < D; j2++) prior_new_k += R::dnorm4(wk_prop(j2), pr_mean_w, pr_sd_w, 1);
        
        // Step 4. MALA proposal densities (q)
        double log_q_new_given_old_w = -0.5 * inv_eps2_w * arma::dot(wk_prop - mean_old_w, wk_prop - mean_old_w);
        arma::rowvec mean_new_w = wk_prop + half_eps2_w * grad_new_k;
        double log_q_old_given_new_w = -0.5 * inv_eps2_w * arma::dot(wk - mean_new_w, wk - mean_new_w);
        
        // Step 5. MH accept ratio
        num = like_new_k + prior_new_k + log_q_old_given_new_w;
        den = like_old_k + prior_old_k + log_q_new_given_old_w;
        ratio = num - den;
        
        if (ratio > 0.0) accept = 1;
        else {
          un = R::runif(0, 1);
          accept = (std::log(un) < ratio) ? 1 : 0;
        }
        
        if (accept == 1) {
          oldw.row(k) = wk_prop;
          accept_w(k) += 1.0 / (niter * 1.0);
        }
      }
    }
    
    if(fix_r){
      rold = r_init;
    }else{
      // overdispersion parameter r update
      rnew = R::rlnorm(std::log(rold), jump_r);
      old_like_r = new_like_r = 0.0;
      for(i = 0; i < N; i++){
        for(k = 0; k < P; k++){
          if((i != k) & (data(i,k) != missing)){
            new_like_r += dnbinom_mu(data(i,k),rnew,exp(arma::dot(get_slices2(X,i,k), olddelta) + oldalpha(i) + oldbeta(k) + (vector ? 1 : -1) * oldgamma * dist(i,k)),1);
            old_like_r += dnbinom_mu(data(i,k),rold,exp(arma::dot(get_slices2(X,i,k), olddelta) + oldalpha(i) + oldbeta(k) + (vector ? 1 : -1) * oldgamma * dist(i,k)),1);
          }
        }
      }
      
      num = new_like_r + R::dlnorm(rold, std::log(rnew), jump_r, 1) + R::dlnorm(rnew, pr_mean_r, pr_sd_r, 1); // not symmetric 
      den = old_like_r + R::dlnorm(rnew, std::log(rold), jump_r, 1) + R::dlnorm(rold, pr_mean_r, pr_sd_r, 1);
      ratio = num - den;
      
      if(ratio > 0.0) accept = 1;
      else{
        un = R::runif(0,1);
        if(std::log(un) < ratio) accept = 1;
        else accept = 0;
      }
      
      if(accept == 1){
        rold = rnew;
        accept_r += 1.0 / (niter * 1.0);
      }
      
      // pr_sd_r hierarchical update
      if(hierarchical_r){
        // pr_sd_r^2 ~ Inverse-Gamma(pr_a_r_sd, pr_b_r_sd)
        // i.e., precision = 1/pr_sd_r^2 ~ Gamma(pr_a_r_sd, pr_b_r_sd)
        double post_a_r_sd = pr_a_r_sd + 0.5;
        double post_b_r_sd = pr_b_r_sd + 0.5 * std::pow((std::log(rold) - pr_mean_r), 2.0);
        double precision = R::rgamma(post_a_r_sd, 1.0 / post_b_r_sd);
        pr_sd_r = std::sqrt(1.0 / precision);
      }
    }
    
    // delta update    
    if(covariate){
      for(j = 0; j < K; j++){
        old_like_delta = new_like_delta = 0.0;
        newdelta(j) = olddelta(j) + R::rnorm(0, jump_delta);
        
        for(i = 0; i < N; i++){
          for(k = 0; k < P; k++){
            if((i != k) & (data(i,k) != missing)){
              new_like_delta += dnbinom_mu(data(i,k),rnew,exp(arma::dot(get_slices2(X,i,k), newdelta) + oldalpha(i) + oldbeta(k) + (vector ? 1 : -1) * oldgamma * dist(i,k)),1);
              old_like_delta += dnbinom_mu(data(i,k),rold,exp(arma::dot(get_slices2(X,i,k), olddelta) + oldalpha(i) + oldbeta(k) + (vector ? 1 : -1) * oldgamma * dist(i,k)),1);
            }
          }
        }
        
        num = new_like_delta + R::dnorm4(newdelta(j), pr_mean_delta, pr_sd_delta, 1);
        den = old_like_delta + R::dnorm4(olddelta(j), pr_mean_delta, pr_sd_delta, 1);
        ratio = num - den;
        
        if(ratio > 0.0) accept = 1;
        else{
          un = R::runif(0,1);
          if(std::log(un) < ratio) accept = 1;
          else accept = 0;
        }
        
        if(accept == 1){
          olddelta(j) = newdelta(j);
          accept_delta(j) += 1.0 / (niter * 1.0);
        }
        
      }
    }else{
      olddelta.fill(0);
    }
    
    if(fixsd){
      //sigma_delta update with gibbs
      post_a = 0.001 * 2.0 + K;
      post_b = 0.001;
      for(k = 0; k < P; k++) post_b += arma::dot(olddelta, olddelta) / 2;
      pr_sd_delta = std::sqrt(2 * post_b *(1.0 /  R::rchisq(post_a)));
    }else{
      //sigma_alpha update with gibbs
      post_a = 2 * pr_a_alpha  + N;
      post_b = pr_b_alpha;
      for(i = 0; i < N; i++) post_b += std::pow((oldalpha(i) - pr_mean_alpha), 2.0) / 2;
      pr_sd_alpha = std::sqrt(2 * post_b *(1.0 /  R::rchisq(post_a)));
      
      
      //sigma_beta update with gibbs
      post_a = 2 * pr_a_beta  + P;
      post_b = pr_b_beta;
      for(k = 0; k < P; k++) post_b += std::pow((oldbeta(k) - pr_mean_beta), 2.0) / 2;
      pr_sd_beta = std::sqrt(2 * post_b *(1.0 /  R::rchisq(post_a)));
      
      //sigma_delta update with gibbs
      post_a = 0.001 * 2.0 + K;
      post_b = 0.001;
      for(k = 0; k < P; k++) post_b += arma::dot(olddelta, olddelta) / 2;
      pr_sd_delta = std::sqrt(2 * post_b *(1.0 /  R::rchisq(post_a)));
    }
    
    if(iter >= nburn && iter % nthin == 0){
      for(i = 0; i < N; i++) samp_alpha(count,i) = oldalpha(i);
      for(k = 0; k < P; k++) samp_beta(count,k) = oldbeta(k);
      for(j = 0; j < K; j++) samp_delta(count,j) = olddelta(j);
      for(i = 0; i < N; i++){
        for(j = 0; j < ndim; j++){
          samp_z(count,i,j) = oldz(i,j);
        }
      }
      for(k = 0; k < P; k++){
        for(j = 0; j < ndim; j++){          
          samp_w(count,k,j) = oldw(k,j);
        }
      }
      
      samp_r(count) = rold;
      samp_gamma(count) = oldgamma;
      samp_sd_beta(count) = pr_sd_beta;
      samp_sd_alpha(count) = pr_sd_alpha;
      samp_sd_delta(count) = pr_sd_delta;
      samp_sd_r(count) = pr_sd_r;
      
      // distance
      if(vector){
        if(singledist){
          dist = oldz * oldz.t();
        }else{
          dist = oldz * oldw.t();
        }
        dist.diag().zeros();
      }else{
        dist.fill(0.0);
        for(i = 0; i < N; i++){
          for(k = 0; k < P; k++){
            dist_temp = 0.0;
            if((i != k) & (data(i,k) != missing)){
              if(singledist){              
                for(j = 0; j < ndim; j++) dist_temp += std::pow((oldz(i,j)-oldz(k,j)), 2.0);                            
              }else{
                for(j = 0; j < ndim; j++) dist_temp += std::pow((oldz(i,j)-oldw(k,j)), 2.0);
              }
            }        
            dist(i,k) = std::sqrt(dist_temp);
          }
        }
      }
      
      post = 0.0;
      for(i = 0; i < N; i++) post += R::dnorm4(oldalpha(i), pr_mean_alpha, pr_sd_alpha, 1);
      for(k = 0; k < P; k++) post += R::dnorm4(oldbeta(k), pr_mean_beta, pr_sd_beta, 1);
      if(covariate){
        for(j = 0; j < K; j++) post += R::dnorm4(olddelta(j), pr_mean_delta, pr_sd_delta, 1);  
      }
      if(fix){
        post += 0.0; 
      }else{
        post += R::dlnorm(oldgamma, pr_mean_gamma, pr_sd_gamma, 1);
      }
      post += R::dlnorm(rold, pr_mean_r, pr_sd_r, 1);
      
      for(i = 0; i < N; i++)
        for(j = 0; j < ndim; j++) post += R::dnorm4(oldz(i,j),pr_mean_z,pr_sd_z,1);
      for(k = 0; k < P; k++)
        for(j = 0; j < ndim; j++) post += R::dnorm4(oldw(k,j),pr_mean_w,pr_sd_w,1);
      
      // Store pointwise log-likelihood for WAIC calculation
      int obs_idx = 0;
      for(i = 0; i < N; i++){
        for(k = 0; k < P; k++){
          if((i != k) & (data(i,k) != missing)){
            pointwise_loglik = dnbinom_mu(data(i,k),rold,exp(arma::dot(get_slices2(X,i,k), olddelta) + oldalpha(i) + oldbeta(k) + (vector ? 1 : -1) * oldgamma * dist(i,k)),1);
            post += pointwise_loglik;
            samp_pointwise_loglik(count, obs_idx) = pointwise_loglik;
            obs_idx++;
          }
        }
      }
      sample_post(count) = post;
      count++;
    } // burn, thin
    
    if(verbose){
      int percent = 0;
      if(iter % nprint == 0){
        percent = (iter*100)/niter;
        Rprintf("Iteration: %.5u %3d%% ", iter, percent);
      }
    }else{
      // progress bar
      progressbar(iter+1,niter);
    }
    
  } //for end
  
  Rcpp::List output;
  output["beta"] = samp_beta;
  output["alpha"] = samp_alpha;
  output["delta"] = samp_delta;
  output["z"] = samp_z;
  output["w"] = samp_w;
  output["r"] = samp_r;
  output["gamma"] = samp_gamma;
  output["sigma_beta"] = samp_sd_beta;
  output["sigma_alpha"] = samp_sd_alpha;
  output["sigma_delta"] = samp_sd_delta;
  output["sigma_r"] = samp_sd_r;
  output["map"] = sample_post;
  output["accept_beta"] = accept_beta;
  output["accept_alpha"] = accept_alpha;
  output["accept_delta"] = accept_delta;
  output["accept_z"] = accept_z;
  output["accept_w"] = accept_w;
  output["accept_gamma"] = accept_gamma;
  output["accept_r"] = accept_r;
  output["pointwise_loglik"] = samp_pointwise_loglik;
  
  return(output);
  
} // function end

// [[Rcpp::export]]
Rcpp::List log_likelihood_count_cpp(arma::mat data, const int ndim, arma::mat beta_est, arma::mat alpha_est, const double gamma_est, arma::mat z_est, arma::mat w_est,
                                    const double r_est, const double eta_est, const bool zeroinflate, const bool overdispersion, const bool vector = true){
  double log_likelihood = 0, interaction_temp = 0;
  const int nsample = data.n_rows;
  const int nitem = data.n_cols;
  int i, j, k;
  arma::dmat interaction(nsample,nitem,fill::zeros);
  
  interaction.fill(0.0);
  for(i = 0; i < nitem; i++){
    for(k = 0; k < nsample; k++){
      interaction_temp = 0.0;
      if(vector){
        // Inner product
        for(j = 0; j < ndim; j++) interaction_temp += z_est(k,j) * w_est(i,j);
      } else {
        // Distance (negative for consistency with inner product direction)
        for(j = 0; j < ndim; j++) interaction_temp += std::pow((z_est(k,j)-w_est(i,j)), 2.0);
        interaction_temp = -std::sqrt(interaction_temp);
      }
      interaction(k,i) = interaction_temp;
    }
  }
  
  for(i = 0; i < nitem; i++){
    for(k = 0; k < nsample; k++){
      log_likelihood += dnbinom_mu(data(k,i),r_est,exp(beta_est(i)+alpha_est(k)+gamma_est*interaction(k,i)),1);
    }
  }

  Rcpp::List output;
  output["log_likelihood"] = log_likelihood;
  return(output);
} // function end

// [[Rcpp::export]]
Rcpp::List calculate_waic_dic_procrustes_cpp(
  arma::mat data,
  arma::mat beta_samples,
  arma::mat alpha_samples,
  arma::vec gamma_samples,
  arma::vec r_samples,
  arma::cube z_proc,        // [n_samples, n_persons, ndim]
  arma::cube w_proc,        // [n_samples, n_items, ndim]
  int ndim,
  bool overdispersion = false,
  bool zeroinflate = false,
  int missing = -99,
  bool vector = false
) {
  int n_samples = beta_samples.n_rows;
  int n_persons = data.n_rows;
  int n_items = data.n_cols;

  // Count valid observations
  int n_obs = 0;
  for(int i = 0; i < n_persons; i++) {
    for(int j = 0; j < n_items; j++) {
      if((i != j) && (data(i,j) != missing)) {
        n_obs++;
      }
    }
  }

  // Initialize pointwise log-likelihood matrix
  arma::mat log_lik_matrix(n_samples, n_obs);

  // Calculate pointwise log-likelihood for each MCMC sample
  int obs_idx = 0;
  for(int i = 0; i < n_persons; i++) {
    for(int j = 0; j < n_items; j++) {
      if((i != j) && (data(i,j) != missing)) {
        
        for(int iter = 0; iter < n_samples; iter++) {
          double beta_curr = beta_samples(iter, j);
          double alpha_curr = alpha_samples(iter, i);
          double gamma_curr = gamma_samples(iter);
          double r_curr = r_samples(iter);
          // Calculate inner product using Procrustes matched samples
          double inner_prod = 0.0;
          for(int d = 0; d < ndim; d++) {
            inner_prod += z_proc(iter, i, d) * w_proc(iter, j, d);
          }
          
          // Calculate log-likelihood with proper parameterization
          if(overdispersion) {
            double mu = std::exp(beta_curr + alpha_curr + gamma_curr * inner_prod);
            log_lik_matrix(iter, obs_idx) = R::dnbinom_mu(data(i,j), r_curr, mu, 1);          
          } else {
            double lambda = std::exp(beta_curr + alpha_curr + gamma_curr * inner_prod);
            log_lik_matrix(iter, obs_idx) = R::dpois(data(i,j), lambda, 1);
          }
        }
        obs_idx++;
      }
    }
  }

  // Calculate WAIC with numerical stability
  double lppd = 0.0;
  double p_waic = 0.0;

  for(int obs = 0; obs < n_obs; obs++) {
    arma::vec log_lik_obs = log_lik_matrix.col(obs);
    arma::uvec finite_idx = arma::find_finite(log_lik_obs);
    
    if(finite_idx.n_elem > 0) {
      arma::vec valid_log_lik = log_lik_obs(finite_idx);
      double max_log_lik = arma::max(valid_log_lik);
      arma::vec exp_centered = arma::exp(valid_log_lik - max_log_lik);
      double mean_exp = arma::mean(exp_centered);
      
      if(mean_exp > 1e-15) {
        lppd += log(mean_exp) + max_log_lik;
      } else {
        lppd += max_log_lik - 15.0;
      }
      
      if(valid_log_lik.n_elem > 1) {
        double var_val = arma::var(valid_log_lik, 1);
        if(std::isfinite(var_val)) {
          p_waic += var_val;
        }
      }
    }
  }

  double waic = -2.0 * lppd + 2.0 * p_waic;

  // Calculate DIC using Procrustes matched posterior means
  arma::vec beta_mean = arma::mean(beta_samples, 0).t();
  arma::vec alpha_mean = arma::mean(alpha_samples, 0).t();
  double gamma_mean = arma::mean(gamma_samples);
  double r_mean = arma::mean(r_samples);
  arma::mat z_mean = arma::mean(z_proc, 0);  // [n_persons, ndim]
  arma::mat w_mean = arma::mean(w_proc, 0);  // [n_items, ndim]

  // Calculate deviance at posterior means
  double log_like_mean = 0.0;
  for(int i = 0; i < n_persons; i++) {
    for(int j = 0; j < n_items; j++) {
      if((i != j) && (data(i,j) != missing)) {
        double inner_prod = 0.0;
        for(int d = 0; d < ndim; d++) {
          inner_prod += z_mean(i, d) * w_mean(j, d);
        }
        
        if (overdispersion) {
          double mu = std::exp(beta_mean(j) + alpha_mean(i) + gamma_mean * inner_prod);
          log_like_mean += R::dnbinom_mu(data(i,j), r_mean, mu, 1);
        } else {
          double lambda = std::exp(beta_mean(j) + alpha_mean(i) + gamma_mean * inner_prod);
          log_like_mean += R::dpois(data(i,j), lambda, 1);
        }
      }
    }
  }

  double d_hat = -2.0 * log_like_mean;

  // Calculate mean deviance
  double d_bar = 0.0;
  for(int iter = 0; iter < n_samples; iter++) {
    double log_like_iter = 0.0;
    for(int i = 0; i < n_persons; i++) {
      for(int j = 0; j < n_items; j++) {
        if((i != j) && (data(i,j) != missing)) {
          double inner_prod = 0.0;
          for(int d = 0; d < ndim; d++) {
            inner_prod += z_proc(iter, i, d) * w_proc(iter, j, d);
          }
          
          if (overdispersion) {
            double mu = std::exp(beta_samples(iter, j) + alpha_samples(iter, i) + gamma_samples(iter) * inner_prod);
            log_like_iter += R::dnbinom_mu(data(i,j), r_samples(iter), mu, 1);
          } else {
            double lambda = std::exp(beta_samples(iter, j) + alpha_samples(iter, i) + gamma_samples(iter) * inner_prod);
            log_like_iter += R::dpois(data(i,j), lambda, 1);
          }
        }
      }
    }
    d_bar += (-2.0 * log_like_iter);
  }
  d_bar /= n_samples;

  double p_d = d_bar - d_hat;
  double dic = d_hat + 2.0 * p_d;

  Rcpp::List output;
  output["waic"] = waic;
  output["dic"] = dic;
  output["lppd"] = lppd;
  output["p_waic"] = p_waic;
  output["p_d"] = p_d;
  output["pointwise_loglik"] = log_lik_matrix;

  return output;
}
