// Routines for:
//      1. admm_ovglasso
//      2. admm_ovglasso_cov
//
//      ADMM for linear regression models with
//          overlap group-LASSO penalty with non penalized covariates

// Authors:
//           Bernardi Mauro, University of Padova
//           Last update: July 12, 2022

// List of implemented methods:



// [[Rcpp::depends(RcppArmadillo)]]
#include "linreg_ovglasso_ADMM.h"
#include "linreg_lasso_ADMM.h"

/*
 Attention: the dimension of the vector x is p, where p is the number of columns
            of W (penalised covariates) and q is the dimension of the vector of
            regression parameters that are not penalized which correspond to
            the design matrix Z (non-penalised covariates),
            while the dimension of the vectors u and z are p                */

/*
    Overlap Group-LASSO via ADMM: wrap functions, to be used
                                  to fit the model once (single lambda)     */

/* ADMM with OVGLASSO                                         */
//' @name  fdaSP.internal
//' @keywords internal
// [[Rcpp::export(.admm_ovglasso)]]
Rcpp::List admm_ovglasso(const arma::mat& A,
                         const arma::colvec& b,
                         const arma::mat& groups,
                         const arma::vec& group_weights,
                         const arma::vec& var_weights,
                         arma::colvec& u,
                         arma::colvec& z,
                         const double lambda,
                         const bool rho_adaptation,
                         double rho,
                         const double tau,
                         const double mu,
                         const double reltol,
                         const double abstol,
                         const int maxiter,
                         const int ping) {
  
  /* Get dimensions             */
  const int m = A.n_rows;
  const int n = A.n_cols;
  
  /* Variable delcaration           */
  List out;
    
  /* Run ADMM            */
  if (m >= n) {
    /* large n problems            */
    out = admm_ovglasso_large_m(A, b, groups, group_weights, var_weights,
                                u, z, lambda, rho_adaptation, rho, tau, mu,
                                reltol, abstol, maxiter, ping);
  } else {
    /* large p problems            */
    out = admm_ovglasso_large_n(A, b, groups, group_weights, var_weights,
                                u, z, lambda, rho_adaptation, rho, tau, mu,
                                reltol, abstol, maxiter, ping);
  }

  /* Return output      */
  return(out);
}

/* ADMM with OVGLASSO and smoothing effects                                       */
//' @name  fdaSP.internal
//' @keywords internal
// [[Rcpp::export(.admm_ovglasso_smo)]]
Rcpp::List admm_ovglasso_smo(const arma::mat& A,
                             const arma::colvec& b,
                             const arma::mat& groups,
                             const arma::vec& group_weights,
                             const arma::vec& var_weights,
                             arma::colvec& u,
                             arma::colvec& z,
                             const double lambda,
                             const double lambda2,
                             const int diff_order,
                             const bool rho_adaptation,
                             double rho,
                             const double tau,
                             const double mu,
                             const double reltol,
                             const double abstol,
                             const int maxiter,
                             const int ping) {
  
  /* Get dimensions             */
  const int m = A.n_rows;
  const int n = A.n_cols;
  
  /* Variable delcaration           */
  List out;
    
  /* Run ADMM            */
  if (m >= n) {
    /* large n problems            */
    out = admm_ovglasso_smo_large_m(A, b, groups, group_weights, var_weights,
                                    u, z, lambda, lambda2, diff_order,
                                    rho_adaptation, rho, tau, mu,
                                    reltol, abstol, maxiter, ping);
  } else {
    /* large p problems            */
    out = admm_ovglasso_smo_large_n(A, b, groups, group_weights, var_weights,
                                    u, z, lambda, lambda2, diff_order,
                                    rho_adaptation, rho, tau, mu,
                                    reltol, abstol, maxiter, ping);
  }

  /* Return output      */
  return(out);
}

/* ADMM with OVGLASSO with covariates           */
//' @name  fdaSP.internal
//' @keywords internal
// [[Rcpp::export(.admm_ovglasso_cov)]]
Rcpp::List admm_ovglasso_cov(const arma::mat& W,
                             const arma::mat& Z,
                             const arma::vec& y,
                             arma::colvec& u,
                             arma::colvec& z,
                             const arma::mat& groups,
                             const arma::vec& group_weights,
                             const arma::vec& var_weights,
                             const double lambda,
                             const bool rho_adaptation,
                             double rho,
                             const double tau,
                             const double mu,
                             const double reltol,
                             const double abstol,
                             const int maxiter,
                             const int ping) {
  
  /* Get dimensions             */
  const int m       = W.n_rows;
  const int n_spcov = W.n_cols;
  
  /* Variable delcaration           */
  List out;
    
  /* Run ADMM            */
  if (m >= n_spcov) {
    /* large n problems            */
    out = admm_ovglasso_cov_large_m(W, Z, y, u, z,
                                    groups, group_weights, var_weights,
                                    lambda, rho_adaptation, rho, tau, mu,
                                    reltol, abstol, maxiter, ping);
  } else {
    /* large p problems            */
    out = admm_ovglasso_cov_large_n(W, Z, y, u, z,
                                    groups, group_weights, var_weights,
                                    lambda, rho_adaptation, rho, tau, mu,
                                    reltol, abstol, maxiter, ping);
  }

  /* Return output      */
  return(out);
}

/* ADMM with OVGLASSO with covariates and smoothing effects           */
//' @name  fdaSP.internal
//' @keywords internal
// [[Rcpp::export(.admm_ovglasso_cov_smo)]]
Rcpp::List admm_ovglasso_cov_smo(const arma::mat& W,
                                 const arma::mat& Z,
                                 const arma::vec& y,
                                 arma::colvec& u,
                                 arma::colvec& z,
                                 const arma::mat& groups,
                                 const arma::vec& group_weights,
                                 const arma::vec& var_weights,
                                 const double lambda,
                                 const double lambda2,
                                 const int diff_order,
                                 const bool rho_adaptation,
                                 double rho,
                                 const double tau,
                                 const double mu,
                                 const double reltol,
                                 const double abstol,
                                 const int maxiter,
                                 const int ping) {
  
  /* Get dimensions             */
  const int m       = W.n_rows;
  const int n_spcov = W.n_cols;
  
  /* Variable delcaration           */
  List out;
    
  /* Run ADMM            */
  if (m >= n_spcov) {
    /* large n problems            */
    out = admm_ovglasso_cov_smo_large_m(W, Z, y, u, z,
                                        groups, group_weights, var_weights,
                                        lambda, lambda2, diff_order, rho_adaptation, rho, tau, mu,
                                        reltol, abstol, maxiter, ping);
  } else {
    /* large p problems            */
    out = admm_ovglasso_cov_smo_large_n(W, Z, y, u, z,
                                        groups, group_weights, var_weights,
                                        lambda, lambda2, diff_order, rho_adaptation, rho, tau, mu,
                                        reltol, abstol, maxiter, ping);
  }

  /* Return output      */
  return(out);
}

/*
    Overlap Group-LASSO via ADMM: wrap functions, fast routines
                                  to be used to fit several times
                                  (multiple lambdas)                  */

/* ADMM with OVGLASSO, fast wrap                                         */
//' @name  fdaSP.internal
//' @keywords internal
// [[Rcpp::export(.admm_ovglasso_fast)]]
Rcpp::List admm_ovglasso_fast(const arma::mat& A,
                              const arma::vec& b,
                              const arma::mat& groups,
                              const arma::vec& group_weights,
                              const arma::vec& var_weights,
                              const arma::vec& lambda,
                              const bool rho_adaptation,
                              double rho,
                              const double tau,
                              const double mu,
                              const double reltol,
                              const double abstol,
                              const int maxiter,
                              const int ping) {

  /* Get dimensions             */
  const int m = A.n_rows;
  const int n = A.n_cols;

  /* Variable delcaration           */
  List out;

  /* Run ADMM            */
  if (m >= n) {
    /* large n problems            */
    out = admm_ovglasso_large_m_fast(A, b, groups, group_weights, var_weights,
                                     lambda, rho_adaptation, rho, tau, mu,
                                     reltol, abstol, maxiter, ping);
  } else {
    /* large p problems            */
    out = admm_ovglasso_large_n_fast(A, b, groups, group_weights, var_weights,
                                     lambda, rho_adaptation, rho, tau, mu,
                                     reltol, abstol, maxiter, ping);
  }

  /* Return output      */
  return(out);
}

/* ADMM with OVGLASSO, fast wrap with smoothing effects            */
//' @name  fdaSP.internal
//' @keywords internal
// [[Rcpp::export(.admm_ovglasso_smo_fast)]]
Rcpp::List admm_ovglasso_smo_fast(const arma::mat& A,
                                  const arma::vec& b,
                                  const arma::mat& groups,
                                  const arma::vec& group_weights,
                                  const arma::vec& var_weights,
                                  const arma::vec& lambda,
                                  const arma::vec& lambda2,
                                  const int diff_order,
                                  const bool rho_adaptation,
                                  double rho,
                                  const double tau,
                                  const double mu,
                                  const double reltol,
                                  const double abstol,
                                  const int maxiter,
                                  const int ping) {

  /* Get dimensions             */
  const int m = A.n_rows;
  const int n = A.n_cols;

  /* Variable delcaration           */
  List out;

  /* Run ADMM            */
  if (m >= n) {
    /* large n problems            */
    out = admm_ovglasso_smo_large_m_fast(A, b, groups, group_weights, var_weights,
                                         lambda, lambda2, diff_order, rho_adaptation, rho, tau, mu,
                                         reltol, abstol, maxiter, ping);
  } else {
    /* large p problems            */
    // ancora da modificare con la nuova versione
    out = admm_ovglasso_smo_large_n_fast(A, b, groups, group_weights, var_weights,
                                         lambda, lambda2, diff_order, rho_adaptation, rho, tau, mu,
                                         reltol, abstol, maxiter, ping);
  }

  /* Return output      */
  return(out);
}

/* ADMM with OVGLASSO, fast wrap with covariates      */
//' @name  fdaSP.internal
//' @keywords internal
// [[Rcpp::export(.admm_ovglasso_cov_fast)]]
Rcpp::List admm_ovglasso_cov_fast(const arma::mat& W,
                                  const arma::mat& Z,
                                  const arma::vec& y,
                                  const arma::mat& groups,
                                  const arma::vec& group_weights,
                                  const arma::vec& var_weights,
                                  const arma::vec& lambda,
                                  const bool rho_adaptation,
                                  double rho,
                                  const double tau,
                                  const double mu,
                                  const double reltol,
                                  const double abstol,
                                  const int maxiter,
                                  const int ping) {

  /* Get dimensions             */
  const int m       = W.n_rows;
  const int n_spcov = W.n_cols;

  /* Variable delcaration           */
  List out;

  /* Run ADMM            */
  if (m >= n_spcov) {
    /* large n problems            */
    out = admm_ovglasso_cov_large_m_fast(W, Z, y,
                                         groups, group_weights, var_weights,
                                         lambda, rho_adaptation, rho, tau, mu,
                                         reltol, abstol, maxiter, ping);
  } else {
    /* large p problems            */
    out = admm_ovglasso_cov_large_n_fast(W, Z, y,
                                         groups, group_weights, var_weights,
                                         lambda, rho_adaptation, rho, tau, mu,
                                         reltol, abstol, maxiter, ping);
  }

  /* Return output      */
  return(out);
}

/* ADMM with OVGLASSO, fast wrap  with covariates and smoothing effects    */
//' @name  fdaSP.internal
//' @keywords internal
// [[Rcpp::export(.admm_ovglasso_cov_smo_fast)]]
Rcpp::List admm_ovglasso_cov_smo_fast(const arma::mat& W,
                                      const arma::mat& Z,
                                      const arma::vec& y,
                                      const arma::mat& groups,
                                      const arma::vec& group_weights,
                                      const arma::vec& var_weights,
                                      const arma::vec& lambda,
                                      const arma::vec& lambda2,
                                      const int diff_order,
                                      const bool rho_adaptation,
                                      double rho,
                                      const double tau,
                                      const double mu,
                                      const double reltol,
                                      const double abstol,
                                      const int maxiter,
                                      const int ping) {
  
  /* Get dimensions             */
  const int m       = W.n_rows;
  const int n_spcov = W.n_cols;

  /* Variable delcaration           */
  List out;

  /* Run ADMM            */
  if (m >= n_spcov) {
    /* large n problems            */
    out = admm_ovglasso_cov_smo_large_m_fast(W, Z, y,
                                             groups, group_weights, var_weights,
                                             lambda, lambda2, diff_order,
                                             rho_adaptation, rho, tau, mu,
                                             reltol, abstol, maxiter, ping);
  } else {
    /* large p problems            */
    out = admm_ovglasso_cov_smo_large_n_fast(W, Z, y,
                                             groups, group_weights, var_weights,
                                             lambda, lambda2, diff_order,
                                             rho_adaptation, rho, tau, mu,
                                             reltol, abstol, maxiter, ping);
    
    
  }

  /* Return output      */
  return(out);
}

/*
    Sparse Overlap Group-LASSO via ADMM: wrap functions, to be used
                                  to fit the model once (single lambda)     */

/* ADMM with sparse OVGLASSO                                         */
//' @name  fdaSP.internal
//' @keywords internal
// [[Rcpp::export(.admm_spovglasso)]]
Rcpp::List admm_spovglasso(const arma::mat& A,
                           const arma::colvec& b,
                           const arma::mat& groups,
                           const arma::vec& group_weights,
                           const arma::vec& var_weights,
                           const arma::vec& var_weights_L1,
                           arma::colvec& u,
                           arma::colvec& z,
                           const double lambda,
                           const double alpha,
                           const bool rho_adaptation,
                           double rho,
                           const double tau,
                           const double mu,
                           const double reltol,
                           const double abstol,
                           const int maxiter,
                           const int ping) {
  
  /* Get dimensions             */
  const int m = A.n_rows;
  const int n = A.n_cols;
  
  /* Variable delcaration           */
  List out;
    
  /* Run ADMM            */
  if (m >= n) {
    /* large n problems            */
    out = admm_spovglasso_large_m(A, b, groups, group_weights, var_weights, var_weights_L1,
                                  u, z, lambda, alpha, rho_adaptation, rho, tau, mu,
                                  reltol, abstol, maxiter, ping);
  } else {
    /* large p problems            */
    out = admm_spovglasso_large_n(A, b, groups, group_weights, var_weights, var_weights_L1,
                                  u, z, lambda, alpha, rho_adaptation, rho, tau, mu,
                                  reltol, abstol, maxiter, ping);
  }

  /* Return output      */
  return(out);
}

/* ADMM with OVGLASSO with covariates           */
//' @name  fdaSP.internal
//' @keywords internal
// [[Rcpp::export(.admm_spovglasso_cov)]]
Rcpp::List admm_spovglasso_cov(const arma::mat& W,
                               const arma::mat& Z,
                               const arma::vec& y,
                               arma::colvec& u,
                               arma::colvec& z,
                               const arma::mat& groups,
                               const arma::vec& group_weights,
                               const arma::vec& var_weights,
                               const arma::vec& var_weights_L1,
                               const double lambda,
                               const double alpha,
                               const bool rho_adaptation,
                               double rho,
                               const double tau,
                               const double mu,
                               const double reltol,
                               const double abstol,
                               const int maxiter,
                               const int ping) {
  
  /* Get dimensions             */
  const int m       = W.n_rows;
  const int n_spcov = W.n_cols;
  
  /* Variable delcaration           */
  List out;
    
  /* Run ADMM            */
  if (m >= n_spcov) {
    /* large n problems            */
    out = admm_spovglasso_cov_large_m(W, Z, y, u, z,
                                      groups, group_weights, var_weights, var_weights_L1,
                                      lambda, alpha, rho_adaptation, rho, tau, mu,
                                      reltol, abstol, maxiter, ping);
  } else {
    /* large p problems            */
    out = admm_spovglasso_cov_large_n(W, Z, y, u, z,
                                      groups, group_weights, var_weights, var_weights_L1,
                                      lambda, alpha, rho_adaptation, rho, tau, mu,
                                      reltol, abstol, maxiter, ping);
  }

  /* Return output      */
  return(out);
}

/*
    Sparse Overlap Group-LASSO via ADMM: wrap functions, fast routines
                                  to be used to fit several times
                                  (multiple lambdas)                  */

/* ADMM with OVGLASSO, fast wrap                                         */
//' @name  fdaSP.internal
//' @keywords internal
// [[Rcpp::export(.admm_spovglasso_fast)]]
Rcpp::List admm_spovglasso_fast(const arma::mat& A,
                                const arma::vec& b,
                                const arma::mat& groups,
                                const arma::vec& group_weights,
                                const arma::vec& var_weights,
                                const arma::vec& var_weights_L1,
                                const arma::vec& lambda,
                                const double alpha,
                                const bool rho_adaptation,
                                double rho,
                                const double tau,
                                const double mu,
                                const double reltol,
                                const double abstol,
                                const int maxiter,
                                const int ping) {

  /* Get dimensions             */
  const int m = A.n_rows;
  const int n = A.n_cols;

  /* Variable delcaration           */
  List out;

  /* Run ADMM            */
  if (m >= n) {
    /* large n problems            */
    out = admm_spovglasso_large_m_fast(A, b, groups, group_weights, var_weights, var_weights_L1,
                                       lambda, alpha, rho_adaptation, rho, tau, mu,
                                       reltol, abstol, maxiter, ping);
  } else {
    /* large p problems            */
    out = admm_spovglasso_large_n_fast(A, b, groups, group_weights, var_weights, var_weights_L1,
                                       lambda, alpha, rho_adaptation, rho, tau, mu,
                                       reltol, abstol, maxiter, ping);
  }

  /* Return output      */
  return(out);
}

/* ADMM with OVGLASSO, fast wrap with covariates      */
//' @name  fdaSP.internal
//' @keywords internal
// [[Rcpp::export(.admm_spovglasso_cov_fast)]]
Rcpp::List admm_spovglasso_cov_fast(const arma::mat& W,
                                    const arma::mat& Z,
                                    const arma::vec& y,
                                    const arma::mat& groups,
                                    const arma::vec& group_weights,
                                    const arma::vec& var_weights,
                                    const arma::vec& var_weights_L1,
                                    const arma::vec& lambda,
                                    const double alpha,
                                    const bool rho_adaptation,
                                    double rho,
                                    const double tau,
                                    const double mu,
                                    const double reltol,
                                    const double abstol,
                                    const int maxiter,
                                    const int ping) {

  /* Get dimensions             */
  const int m       = W.n_rows;
  const int n_spcov = W.n_cols;

  /* Variable delcaration           */
  List out;

  /* Run ADMM            */
  if (m >= n_spcov) {
    /* large n problems            */
    out = admm_spovglasso_cov_large_m_fast(W, Z, y,
                                           groups, group_weights, var_weights, var_weights_L1,
                                           lambda, alpha, rho_adaptation, rho, tau, mu,
                                           reltol, abstol, maxiter, ping);
  } else {
    /* large p problems            */
    out = admm_spovglasso_cov_large_n_fast(W, Z, y,
                                           groups, group_weights, var_weights, var_weights_L1,
                                           lambda, alpha, rho_adaptation, rho, tau, mu,
                                           reltol, abstol, maxiter, ping);
  }

  /* Return output      */
  return(out);
}

/*
    Overlap Group-LASSO via ADMM: main functions, to be used
                                  to fit the model once (single lambda)     */

Rcpp::List admm_ovglasso_large_m(const arma::mat& A,
                                 const arma::colvec& b,
                                 const arma::mat& groups,
                                 const arma::vec& group_weights,
                                 const arma::vec& var_weights,
                                 arma::colvec& u,
                                 arma::colvec& z,
                                 const double lambda,
                                 const bool rho_adaptation,
                                 double rho,
                                 const double tau,
                                 const double mu,
                                 const double reltol,
                                 const double abstol,
                                 const int maxiter,
                                 const int ping) {
    
  /* Variable delcaration           */
  int k = 0;
  uword g_id_init, g_id_end;
  double rho_old = 0.0, elTime = 0.0, rss = 0.0, pen = 0.0;
  
  /* Get dimensions             */
  const int m     = A.n_rows;
  const int n     = A.n_cols;
  const int G     = groups.n_rows;
  const int dim_z = z.n_elem;
  
  /* Definition of vectors and matrices     */
  arma::colvec x(n, fill::zeros);
  arma::colvec q(n, fill::zeros);
  arma::colvec z_old(dim_z, fill::zeros);
  arma::mat ATA(n, n, fill::zeros);
  arma::mat FTF(n, n, fill::zeros);
  arma::vec ATb(n, fill::zeros);
  arma::mat U(n, n, fill::zeros);
  arma::mat L(n, n, fill::zeros);
  arma::colvec Glen(G, fill::zeros);
  arma::vec h_objval(maxiter, fill::zeros);
  arma::vec h_r_norm(maxiter, fill::zeros);
  arma::vec h_s_norm(maxiter, fill::zeros);
  arma::vec h_eps_pri(maxiter, fill::zeros);
  arma::vec h_eps_dual(maxiter, fill::zeros);
  arma::vec rho_store(maxiter+1, fill::zeros);
  
  /* ::::::::::::::::::::::::::::::::::::::::::::::
   Set the computational time                                 */
  wall_clock timer;
  timer.tic();
  
  /* store rho (only for adaptation)         */
  rho_store(0) = rho;
  rho_old      = rho;

  /* Precompute relevant quantities         */
  ATA            = A.t() * A / static_cast<float>(m);
  ATb            = A.t() * b / static_cast<float>(m);
  Glen           = sum(groups, 1);
  arma::sp_mat F = ovglasso_Gmat2Fmat_sparse(groups, var_weights);
  FTF            = F.t() * F;
  U              = glasso_factor_fast_large_m(ATA, FTF, rho); // returns upper
  L              = U.t();
  
  /* main loop  */
  for (k=0; k<maxiter; k++) {
    // 4-1. update 'x'
    if (rho != rho_old) {
      U = glasso_factor_fast_large_m(ATA, FTF, rho); // returns upper
      L = U.t();
    }
    q = ATb + rho * spmat2vec_mult(F.t(), z - u); // temporary value
    x = solve(trimatu(U), solve(trimatl(L), q));

    // 4-2. update 'z' with adaptation
    // see Boyd et al (2011) page 21
    z_old     = z;
    g_id_init = 0;
    for (int g=0; g<G; g++) {
      // precompute the quantities for updating z
      g_id_end         = g_id_init + Glen(g) - 1;
      arma::vec u_g    = u.subvec(g_id_init, g_id_end);
      arma::sp_mat F_g = F.submat(g_id_init, 0, g_id_end, n-1);

      // update z
      arma::vec z_g                 = glasso_prox(F_g * x + u_g, (lambda * group_weights[g]) / rho);
      z.subvec(g_id_init, g_id_end) = z_g;
      g_id_init                     = g_id_end + 1;
    }
    
    // 4-3. update 'u'
    u += spmat2vec_mult(F, x) - z;
  
    // 4-3. dianostics, reporting
    rss         = lm_rss(A, b, x);
    pen         = ovglasso_penalty(x, groups, group_weights, var_weights);
    h_objval(k) = 0.5 *  rss / static_cast<float>(m) + lambda * pen;
    h_r_norm(k) = norm(glasso_residual_sparse(F, x, z), 2);
    h_s_norm(k) = norm(glasso_dual_residual_sparse(F, z, z_old, rho), 2);
    if (norm(F * x, 2) > norm(-z, 2)) {
      h_eps_pri(k) = std::sqrt(z.n_elem) * abstol + reltol * norm(F * x, 2);
    } else {
      h_eps_pri(k) = std::sqrt(z.n_elem) * abstol + reltol * norm(-z, 2);
    }
    h_eps_dual(k) = std::sqrt(z.n_elem) * abstol + reltol * norm(F.t() * rho * u, 2);
    
    /* :::::::::::::::::::::::::::::::::::::::::::::::::::::::: */
    /* Relaxation                                               */
    rho_old = rho;
    if (rho_adaptation) {
      if (h_r_norm(k) > mu * h_s_norm(k)) {
        rho = rho * tau;
        u   = u / tau;
      } else if (h_s_norm(k) > mu * h_r_norm(k)) {
        rho = rho / tau;
        u   = u * tau;
      }
    }
    rho_store(k+1) = rho;
    
    /* ::::::::::::::::::::::::::::::::::::::::::::::
     Compute elapsed time                                */
    elTime = timer.toc();
    
    /* :::::::::::::::::::::::::::::::::::::::::::::::::::::::: */
    /* Print to screen                                          */
    if ((ping > 0) && (k < maxiter)) {
      if (rho_adaptation) {
        if (((k+1) % ping) == 0) {
          Rcpp::Rcout << "\n\n\n" << std::endl;
          Rcpp::Rcout << "ADMM with adaptation for overlap group-LASSO is running!" << std::endl;
          Rcpp::Rcout << "Iteration n.: " << k+1 << " of " << maxiter << "\n" << std::endl;
        }
      } else {
        if (((k+1) % ping) == 0) {
          Rcpp::Rcout << "\n\n\n" << std::endl;
          Rcpp::Rcout << "ADMM for overlap group-LASSO is running!" << std::endl;
          Rcpp::Rcout << "Iteration n.: " << k+1 << " of " << maxiter << "\n" << std::endl;
        }
      }
    }

    // 4-4. termination
    if ((h_r_norm(k) < h_eps_pri(k)) && (h_s_norm(k) < h_eps_dual(k))) {
      break;
    }
  }

  /* Get output         */
  List output;
  output["x"] = x;               
  output["u"] = u;
  output["z"] = z;
  if (k < maxiter) {
    output["convergence"] = 1.0;
    output["niter"]       = k+1;               
    output["objval"]      = h_objval.subvec(0, k);        
    output["r_norm"]      = h_r_norm.subvec(0, k);
    output["s_norm"]      = h_s_norm.subvec(0, k);
    output["eps_pri"]     = h_eps_pri.subvec(0, k);
    output["eps_dual"]    = h_eps_dual.subvec(0, k);
    output["rho"]         = rho_store.subvec(0, k+1);
  } else {
    output["convergence"] = 0.0;
    output["niter"]       = maxiter;
    output["objval"]      = h_objval;
    output["r_norm"]      = h_r_norm;
    output["s_norm"]      = h_s_norm;
    output["eps_pri"]     = h_eps_pri;
    output["eps_dual"]    = h_eps_dual;
    output["rho"]         = rho_store;
  }
  output["eltime"]   = elTime;
  if ((k+1) < maxiter) {
    output["convergence"] = 1.0;
  } else {
    output["convergence"] = 0.0;
  }

  /* Return output      */
  return(output);
}

Rcpp::List admm_ovglasso_smo_large_m(const arma::mat& A,
                                     const arma::colvec& b,
                                     const arma::mat& groups,
                                     const arma::vec& group_weights,
                                     const arma::vec& var_weights,
                                     arma::colvec& u,
                                     arma::colvec& z,
                                     const double lambda,
                                     const double lambda2,
                                     const int diff_order,
                                     const bool rho_adaptation,
                                     double rho,
                                     const double tau,
                                     const double mu,
                                     const double reltol,
                                     const double abstol,
                                     const int maxiter,
                                     const int ping) {
    
  /* Variable delcaration           */
  int k = 0;
  double rho_old = 0.0, elTime = 0.0, rss = 0.0, pen = 0.0, pen2 = 0.0;
  uword g_id_init, g_id_end;

  /* Get dimensions             */
  const int m     = A.n_rows;
  const int n     = A.n_cols;
  const int G     = groups.n_rows;
  const int dim_z = z.n_elem;
  
  /* Definition of vectors and matrices     */
  arma::colvec x(n, fill::zeros);
  arma::colvec q(n, fill::zeros);
  arma::colvec z_old(dim_z, fill::zeros);
  arma::mat ATA(n, n, fill::zeros);
  arma::mat FTF(n, n, fill::zeros);
  arma::mat DTD(n, n, fill::zeros);
  arma::vec ATb(n, fill::zeros);
  arma::mat U(n, n, fill::zeros);
  arma::mat L(n, n, fill::zeros);
  arma::colvec Glen(G, fill::zeros);
  arma::mat XtTXt(n, n, fill::zeros);
  arma::vec h_objval(maxiter, fill::zeros);
  arma::vec h_r_norm(maxiter, fill::zeros);
  arma::vec h_s_norm(maxiter, fill::zeros);
  arma::vec h_eps_pri(maxiter, fill::zeros);
  arma::vec h_eps_dual(maxiter, fill::zeros);
  arma::vec rho_store(maxiter+1, fill::zeros);
  
  /* ::::::::::::::::::::::::::::::::::::::::::::::
   Set the computational time                                 */
  wall_clock timer;
  timer.tic();
  
  /* store rho (only for adaptation)         */
  rho_store(0) = rho;
  rho_old      = rho;

  /* Precompute relevant quantities         */
  ATA            = A.t() * A / static_cast<float>(m);
  ATb            = A.t() * b / static_cast<float>(m);
  Glen           = sum(groups, 1);
  arma::sp_mat F = ovglasso_Gmat2Fmat_sparse(groups, var_weights);
  FTF            = F.t() * F;
  DTD            = forward_diff_penalty_matrix(n, 1, diff_order);
  XtTXt          = ATA + lambda2 * DTD;
  U              = glasso_factor_smo_fast_large_m2(XtTXt, FTF, rho); // returns upper
  L              = U.t();

  /* main loop  */
  for (k=0; k<maxiter; k++) {
    // 4-1. update 'x'
    if (rho != rho_old) {
      U = glasso_factor_smo_fast_large_m2(XtTXt, FTF, rho); // returns upper
      L = U.t();
    }
    q = ATb + rho * spmat2vec_mult(F.t(), z - u); // temporary value
    x = solve(trimatu(U), solve(trimatl(L), q));

    // 4-2. update 'z' with adaptation
    // see Boyd et al (2011) page 21
    z_old     = z;
    g_id_init = 0;
    for (int g=0; g<G; g++) {
      // precompute the quantities for updating z
      g_id_end         = g_id_init + Glen(g) - 1;
      arma::vec u_g    = u.subvec(g_id_init, g_id_end);
      arma::sp_mat F_g = F.submat(g_id_init, 0, g_id_end, n-1);

      // update z
      arma::vec z_g                 = glasso_prox(F_g * x + u_g, (lambda * group_weights[g]) / rho);
      z.subvec(g_id_init, g_id_end) = z_g;
      g_id_init                     = g_id_end + 1;
    }
    
    // 4-3. update 'u'
    u += spmat2vec_mult(F, x) - z;
  
    // 4-3. dianostics, reporting
    rss         = lm_rss(A, b, x);
    pen         = ovglasso_penalty(x, groups, group_weights, var_weights);
    pen2        = 0.5 * as_scalar(x.t() * DTD * x);
    h_objval(k) = 0.5 *  rss / static_cast<float>(m) + lambda * pen + lambda2 * pen2;
    h_r_norm(k) = norm(glasso_residual_sparse(F, x, z), 2);
    h_s_norm(k) = norm(glasso_dual_residual_sparse(F, z, z_old, rho), 2);
    if (norm(F * x, 2) > norm(-z, 2)) {
      h_eps_pri(k) = std::sqrt(z.n_elem) * abstol + reltol * norm(F * x, 2);
    } else {
      h_eps_pri(k) = std::sqrt(z.n_elem) * abstol + reltol * norm(-z, 2);
    }
    h_eps_dual(k) = std::sqrt(z.n_elem) * abstol + reltol * norm(F.t() * rho * u, 2);
    
    /* :::::::::::::::::::::::::::::::::::::::::::::::::::::::: */
    /* Relaxation                                               */
    rho_old = rho;
    if (rho_adaptation) {
      if (h_r_norm(k) > mu * h_s_norm(k)) {
        rho = rho * tau;
        u   = u / tau;
      } else if (h_s_norm(k) > mu * h_r_norm(k)) {
        rho = rho / tau;
        u   = u * tau;
      }
    }
    rho_store(k+1) = rho;
    
    /* ::::::::::::::::::::::::::::::::::::::::::::::
     Compute elapsed time                                */
    elTime = timer.toc();
    
    /* :::::::::::::::::::::::::::::::::::::::::::::::::::::::: */
    /* Print to screen                                          */
    if ((ping > 0) && (k < maxiter)) {
      if (rho_adaptation) {
        if (((k+1) % ping) == 0) {
          Rcpp::Rcout << "\n\n\n" << std::endl;
          Rcpp::Rcout << "ADMM with adaptation for overlap group-LASSO is running!" << std::endl;
          Rcpp::Rcout << "Iteration n.: " << k+1 << " of " << maxiter << "\n" << std::endl;
        }
      } else {
        if (((k+1) % ping) == 0) {
          Rcpp::Rcout << "\n\n\n" << std::endl;
          Rcpp::Rcout << "ADMM for overlap group-LASSO is running!" << std::endl;
          Rcpp::Rcout << "Iteration n.: " << k+1 << " of " << maxiter << "\n" << std::endl;
        }
      }
    }

    // 4-4. termination
    if ((h_r_norm(k) < h_eps_pri(k)) && (h_s_norm(k) < h_eps_dual(k))) {
      break;
    }
  }

  /* Get output         */
  List output;
  output["x"] = x;               
  output["u"] = u;
  output["z"] = z;
  if (k < maxiter) {
    output["convergence"] = 1.0;
    output["niter"]       = k+1;               
    output["objval"]      = h_objval.subvec(0, k);        
    output["r_norm"]      = h_r_norm.subvec(0, k);
    output["s_norm"]      = h_s_norm.subvec(0, k);
    output["eps_pri"]     = h_eps_pri.subvec(0, k);
    output["eps_dual"]    = h_eps_dual.subvec(0, k);
    output["rho"]         = rho_store.subvec(0, k+1);
  } else {
    output["convergence"] = 0.0;
    output["niter"]       = maxiter;               
    output["objval"]      = h_objval;        
    output["r_norm"]      = h_r_norm;
    output["s_norm"]      = h_s_norm;
    output["eps_pri"]     = h_eps_pri;
    output["eps_dual"]    = h_eps_dual;
    output["rho"]         = rho_store;
  }
  output["eltime"]   = elTime;
  if ((k+1) < maxiter) {
    output["convergence"] = 1.0;
  } else {
    output["convergence"] = 0.0;
  }

  /* Return output      */
  return(output);
}

Rcpp::List admm_ovglasso_large_n(const arma::mat& A,
                                 const arma::colvec& b,
                                 const arma::mat& groups,
                                 const arma::vec& group_weights,
                                 const arma::vec& var_weights,
                                 arma::colvec& u,
                                 arma::colvec& z,
                                 const double lambda,
                                 const bool rho_adaptation,
                                 double rho,
                                 const double tau,
                                 const double mu,
                                 const double reltol,
                                 const double abstol,
                                 const int maxiter,
                                 const int ping) {
    
  /* Variable delcaration           */
  int k = 0;
  double rho_old = 0.0, elTime = 0.0, rss = 0.0, pen = 0.0;
  uword g_id_init, g_id_end;
  
  /* Get dimensions             */
  const int m     = A.n_rows;
  const int n     = A.n_cols;
  const int G     = groups.n_rows;
  const int dim_z = z.n_elem;
  
  /* Definition of vectors and matrices     */
  arma::colvec x(n, fill::zeros);
  arma::colvec x_star(n, fill::zeros);
  arma::colvec q(n, fill::zeros);
  arma::colvec z_old(dim_z, fill::zeros);
  arma::mat FTF(n, n, fill::zeros);
  arma::mat FTF_INV(n, n, fill::zeros);
  arma::mat AFTF_INVA(m, m, fill::zeros);
  arma::vec ATb(n, fill::zeros);
  arma::mat U(m, m, fill::zeros);
  arma::mat L(m, m, fill::zeros);
  arma::colvec Glen(G, fill::zeros);
  arma::vec h_objval(maxiter, fill::zeros);
  arma::vec h_r_norm(maxiter, fill::zeros);
  arma::vec h_s_norm(maxiter, fill::zeros);
  arma::vec h_eps_pri(maxiter, fill::zeros);
  arma::vec h_eps_dual(maxiter, fill::zeros);
  arma::vec rho_store(maxiter+1, fill::zeros);
  
  /* ::::::::::::::::::::::::::::::::::::::::::::::
   Set the computational time                                 */
  wall_clock timer;
  timer.tic();
  
  /* store rho (only for adaptation)         */
  rho_store(0) = rho;
  rho_old      = rho;

  /* Precompute relevant quantities         */
  ATb            = A.t() * b / static_cast<float>(m);
  Glen           = sum(groups, 1);
  arma::sp_mat F = ovglasso_Gmat2Fmat_sparse(groups, var_weights);
  FTF            = F.t() * F;
  FTF_INV        = inv(diagmat(FTF));
  AFTF_INVA      = A * FTF_INV * A.t();
  U              = glasso_factor_fast_large_n(AFTF_INVA, rho, m);
  L              = U.t();
  
  /* main loop  */
  for (k=0; k<maxiter; k++) {
    // 4-1. update 'x'
    if (rho != rho_old) {
      U = glasso_factor_fast_large_n(AFTF_INVA, rho, m); // returns upper
      L = U.t();
    }
    q      = ATb + rho * spmat2vec_mult(F.t(), z - u); // temporary value
    x_star = (diagmat(FTF_INV) * q) / rho;
    x      = x_star - FTF_INV * A.t() * solve(trimatu(U), solve(trimatl(L), A * x_star));

    // 4-2. update 'z' with adaptation
    // see Boyd et al (2011) page 21
    z_old     = z;
    g_id_init = 0;
    for (int g=0; g<G; g++) {
      // precompute the quantities for updating z
      g_id_end         = g_id_init + Glen(g) - 1;
      arma::vec u_g    = u.subvec(g_id_init, g_id_end);
      arma::sp_mat F_g = F.submat(g_id_init, 0, g_id_end, n-1);

      // update z
      arma::vec z_g                 = glasso_prox(F_g * x + u_g, (lambda * group_weights[g]) / rho);
      z.subvec(g_id_init, g_id_end) = z_g;
      g_id_init                     = g_id_end + 1;
    }
    
    // 4-3. update 'u'
    u += spmat2vec_mult(F, x) - z;
  
    // 4-3. dianostics, reporting
    rss         = lm_rss(A, b, x);
    pen         = ovglasso_penalty(x, groups, group_weights, var_weights);
    h_objval(k) = 0.5 *  rss / static_cast<float>(m) + lambda * pen;
    h_r_norm(k) = norm(glasso_residual_sparse(F, x, z), 2);
    h_s_norm(k) = norm(glasso_dual_residual_sparse(F, z, z_old, rho), 2);
    if (norm(F * x, 2) > norm(-z, 2)) {
      h_eps_pri(k) = std::sqrt(z.n_elem) * abstol + reltol * norm(F * x, 2);
    } else {
      h_eps_pri(k) = std::sqrt(z.n_elem) * abstol + reltol * norm(-z, 2);
    }
    h_eps_dual(k) = std::sqrt(z.n_elem) * abstol + reltol * norm(F.t() * rho * u, 2);
    
    /* :::::::::::::::::::::::::::::::::::::::::::::::::::::::: */
    /* Relaxation                                               */
    rho_old = rho;
    if (rho_adaptation) {
      if (h_r_norm(k) > mu * h_s_norm(k)) {
        rho = rho * tau;
        u   = u / tau;
      } else if (h_s_norm(k) > mu * h_r_norm(k)) {
        rho = rho / tau;
        u   = u * tau;
      }
    }
    rho_store(k+1) = rho;
    
    /* ::::::::::::::::::::::::::::::::::::::::::::::
     Compute elapsed time                                */
    elTime = timer.toc();
    
    /* :::::::::::::::::::::::::::::::::::::::::::::::::::::::: */
    /* Print to screen                                          */
    if ((ping > 0) && (k < maxiter)) {
      if (rho_adaptation) {
        if (((k+1) % ping) == 0) {
          Rcpp::Rcout << "\n\n\n" << std::endl;
          Rcpp::Rcout << "ADMM with adaptation for overlap group-LASSO is running!" << std::endl;
          Rcpp::Rcout << "Iteration n.: " << k+1 << " of " << maxiter << "\n" << std::endl;
        }
      } else {
        if (((k+1) % ping) == 0) {
          Rcpp::Rcout << "\n\n\n" << std::endl;
          Rcpp::Rcout << "ADMM for overlap group-LASSO is running!" << std::endl;
          Rcpp::Rcout << "Iteration n.: " << k+1 << " of " << maxiter << "\n" << std::endl;
        }
      }
    }

    // 4-4. termination
    if ((h_r_norm(k) < h_eps_pri(k)) && (h_s_norm(k) < h_eps_dual(k))) {
      break;
    }
  }

  /* Get output         */
  List output;
  output["x"] = x;               
  output["u"] = u;
  output["z"] = z;
  if (k < maxiter) {
    output["convergence"] = 1.0;
    output["niter"]       = k+1;               
    output["objval"]      = h_objval.subvec(0, k);        
    output["r_norm"]      = h_r_norm.subvec(0, k);
    output["s_norm"]      = h_s_norm.subvec(0, k);
    output["eps_pri"]     = h_eps_pri.subvec(0, k);
    output["eps_dual"]    = h_eps_dual.subvec(0, k);
    output["rho"]         = rho_store.subvec(0, k+1);
  } else {
    output["convergence"] = 0.0;
    output["niter"]       = maxiter;               
    output["objval"]      = h_objval;        
    output["r_norm"]      = h_r_norm;
    output["s_norm"]      = h_s_norm;
    output["eps_pri"]     = h_eps_pri;
    output["eps_dual"]    = h_eps_dual;
    output["rho"]         = rho_store;
  }
  output["eltime"]   = elTime;
  if ((k+1) < maxiter) {
    output["convergence"] = 1.0;
  } else {
    output["convergence"] = 0.0;
  }

  /* Return output      */
  return(output);
}

Rcpp::List admm_ovglasso_smo_large_n(const arma::mat& A,
                                     const arma::colvec& b,
                                     const arma::mat& groups,
                                     const arma::vec& group_weights,
                                     const arma::vec& var_weights,
                                     arma::colvec& u,
                                     arma::colvec& z,
                                     const double lambda,
                                     const double lambda2,
                                     const int diff_order,
                                     const bool rho_adaptation,
                                     double rho,
                                     const double tau,
                                     const double mu,
                                     const double reltol,
                                     const double abstol,
                                     const int maxiter,
                                     const int ping) {
    
  /* Variable delcaration           */
  int k = 0;
  double rho_old = 0.0, elTime = 0.0, rss = 0.0, pen = 0.0, pen2 = 0.0;
  uword g_id_init, g_id_end;
  
  /* Get dimensions             */
  const int m     = A.n_rows;
  const int n     = A.n_cols;
  const int G     = groups.n_rows;
  const int dim_z = z.n_elem;
  
  /* Definition of vectors and matrices     */
  arma::colvec x(n, fill::zeros);
  arma::colvec x_star(n, fill::zeros);
  arma::colvec q(n, fill::zeros);
  arma::colvec z_old(dim_z, fill::zeros);
  arma::mat FTF(n, n, fill::zeros);
  arma::mat FTF_INV(n, n, fill::zeros);
  arma::mat D(n-diff_order, n, fill::zeros);
  arma::mat DTD(n, n, fill::zeros);
  arma::mat Atilde(n+m, n, fill::zeros);
  arma::mat AtFTF_INVAt(n+m, n+m, fill::zeros);
  arma::vec ATb(n, fill::zeros);
  arma::mat U(m, m, fill::zeros);
  arma::mat L(m, m, fill::zeros);
  arma::colvec Glen(G, fill::zeros);
  arma::vec h_objval(maxiter, fill::zeros);
  arma::vec h_r_norm(maxiter, fill::zeros);
  arma::vec h_s_norm(maxiter, fill::zeros);
  arma::vec h_eps_pri(maxiter, fill::zeros);
  arma::vec h_eps_dual(maxiter, fill::zeros);
  arma::vec rho_store(maxiter+1, fill::zeros);
  
  /* ::::::::::::::::::::::::::::::::::::::::::::::
   Set the computational time                                 */
  wall_clock timer;
  timer.tic();
  
  /* store rho (only for adaptation)         */
  rho_store(0) = rho;
  rho_old      = rho;

  /* Precompute relevant quantities         */
  ATb                 = A.t() * b / static_cast<float>(m);
  Glen                = sum(groups, 1);
  arma::sp_mat F      = ovglasso_Gmat2Fmat_sparse(groups, var_weights);
  FTF                 = F.t() * F;
  FTF_INV             = diagmat(1.0 / diagvec(FTF));
  D                   = forward_diff_difference_matrix(n, 1, diff_order);
  DTD                 = D.t() * D;
  Atilde.head_rows(m) = A / sqrt(static_cast<float>(m));    // compute relevant quantities: these quantities change with lambda2
  Atilde.tail_rows(n) = std::sqrt(lambda2) * D;
  AtFTF_INVAt         = Atilde * FTF_INV * Atilde.t();
  U                   = glasso_factor_smo_fast_large_n2(AtFTF_INVAt, rho, m); // returns upper
  L                   = U.t();

  /* main loop  */
  for (k=0; k<maxiter; k++) {
    // 4-1. update 'x'
    if (rho != rho_old) {
      U = glasso_factor_smo_fast_large_n2(AtFTF_INVAt, rho, m); // returns upper
      L = U.t();
    }
    q      = ATb + rho * spmat2vec_mult(F.t(), z - u); // temporary value
    x_star = (diagmat(FTF_INV) * q) / rho;
    x      = x_star - diagmat(FTF_INV) * Atilde.t() * solve(trimatu(U), solve(trimatl(L), Atilde * x_star));
        
    // 4-2. update 'z' with adaptation
    // see Boyd et al (2011) page 21
    z_old     = z;
    g_id_init = 0;
    for (int g=0; g<G; g++) {
      // precompute the quantities for updating z
      g_id_end         = g_id_init + Glen(g) - 1;
      arma::vec u_g    = u.subvec(g_id_init, g_id_end);
      arma::sp_mat F_g = F.submat(g_id_init, 0, g_id_end, n-1);

      // update z
      arma::vec z_g                 = glasso_prox(F_g * x + u_g, (lambda * group_weights[g]) / rho);
      z.subvec(g_id_init, g_id_end) = z_g;
      g_id_init                     = g_id_end + 1;
    }
    
    // 4-3. update 'u'
    u += spmat2vec_mult(F, x) - z;
  
    // 4-3. dianostics, reporting
    //h_objval(k) = glasso_objfun(A, b, Glen, lambda, x, z, G);
    rss         = lm_rss(A, b, x);
    pen         = ovglasso_penalty(x, groups, group_weights, var_weights);
    pen2        = 0.5 * as_scalar(x.t() * DTD * x);
    h_objval(k) = 0.5 *  rss / static_cast<float>(m) + lambda * pen + lambda2 * pen2;
    h_r_norm(k) = norm(glasso_residual_sparse(F, x, z), 2);
    h_s_norm(k) = norm(glasso_dual_residual_sparse(F, z, z_old, rho), 2);
    if (norm(F * x, 2) > norm(-z, 2)) {
      h_eps_pri(k) = std::sqrt(z.n_elem) * abstol + reltol * norm(F * x, 2);
    } else {
      h_eps_pri(k) = std::sqrt(z.n_elem) * abstol + reltol * norm(-z, 2);
    }
    h_eps_dual(k) = std::sqrt(z.n_elem) * abstol + reltol * norm(F.t() * rho * u, 2);
    
    /* :::::::::::::::::::::::::::::::::::::::::::::::::::::::: */
    /* Relaxation                                               */
    rho_old = rho;
    if (rho_adaptation) {
      if (h_r_norm(k) > mu * h_s_norm(k)) {
        rho = rho * tau;
        u   = u / tau;
      } else if (h_s_norm(k) > mu * h_r_norm(k)) {
        rho = rho / tau;
        u   = u * tau;
      }
    }
    rho_store(k+1) = rho;
    
    /* ::::::::::::::::::::::::::::::::::::::::::::::
     Compute elapsed time                                */
    elTime = timer.toc();
    
    /* :::::::::::::::::::::::::::::::::::::::::::::::::::::::: */
    /* Print to screen                                          */
    if ((ping > 0) && (k < maxiter)) {
      if (rho_adaptation) {
        if (((k+1) % ping) == 0) {
          Rcpp::Rcout << "\n\n\n" << std::endl;
          Rcpp::Rcout << "ADMM with adaptation for overlap group-LASSO is running!" << std::endl;
          Rcpp::Rcout << "Iteration n.: " << k+1 << " of " << maxiter << "\n" << std::endl;
        }
      } else {
        if (((k+1) % ping) == 0) {
          Rcpp::Rcout << "\n\n\n" << std::endl;
          Rcpp::Rcout << "ADMM for overlap group-LASSO is running!" << std::endl;
          Rcpp::Rcout << "Iteration n.: " << k+1 << " of " << maxiter << "\n" << std::endl;
        }
      }
    }

    // 4-4. termination
    if ((h_r_norm(k) < h_eps_pri(k)) && (h_s_norm(k) < h_eps_dual(k))) {
      break;
    }
  }

  /* Get output         */
  List output;
  output["x"] = x;               
  output["u"] = u;
  output["z"] = z;
  if (k < maxiter) {
    output["convergence"] = 1.0;
    output["niter"]       = k+1;               
    output["objval"]      = h_objval.subvec(0, k);        
    output["r_norm"]      = h_r_norm.subvec(0, k);
    output["s_norm"]      = h_s_norm.subvec(0, k);
    output["eps_pri"]     = h_eps_pri.subvec(0, k);
    output["eps_dual"]    = h_eps_dual.subvec(0, k);
    output["rho"]         = rho_store.subvec(0, k+1);
  } else {
    output["convergence"] = 0.0;
    output["niter"]       = maxiter;               
    output["objval"]      = h_objval;        
    output["r_norm"]      = h_r_norm;
    output["s_norm"]      = h_s_norm;
    output["eps_pri"]     = h_eps_pri;
    output["eps_dual"]    = h_eps_dual;
    output["rho"]         = rho_store;
  }
  output["eltime"]   = elTime;
  if ((k+1) < maxiter) {
    output["convergence"] = 1.0;
  } else {
    output["convergence"] = 0.0;
  }

  /* Return output      */
  return(output);
}

Rcpp::List admm_ovglasso_cov_large_m(const arma::mat& W,
                                     const arma::mat& Z,
                                     const arma::vec& y,
                                     arma::colvec& u,
                                     arma::colvec& z,
                                     const arma::mat& groups,
                                     const arma::vec& group_weights,
                                     const arma::vec& var_weights,
                                     const double lambda,
                                     const bool rho_adaptation,
                                     double rho,
                                     const double tau,
                                     const double mu,
                                     const double reltol,
                                     const double abstol,
                                     const int maxiter,
                                     const int ping) {

  /* Variable delcaration           */
  double y2 = 0.0, sqrtm = 0.0, rho_old = 0.0, elTime = 0.0, rss = 0.0, pen = 0.0;
  int n1 = 0, dim_z = 0, k = 0;
  uword g_id_init, g_id_end;

  /* Get dimensions             */
  const int m       = W.n_rows;
  const int n_spcov = W.n_cols;
  const int n_cov   = Z.n_cols;
  const int G       = groups.n_rows;
  
  /* Variable definition            */
  //sqrtn = std::sqrt(static_cast<float>(n_spcov));
  sqrtm = std::sqrt(static_cast<float>(m));

  /* Get output lists          */
  List out, output;

  /* Variable definition            */
  if (m >= n_spcov) {
    n1 = n_spcov;
  } else {
    n1 = m;
  }

  /* Definition of vectors and matrices     */
  arma::colvec x(n_spcov, fill::zeros);
  arma::colvec v(n_cov, fill::zeros);
  arma::colvec q(n_spcov, fill::zeros);
  arma::colvec x_star(n_spcov, fill::zeros);
  arma::mat W_(m, n_spcov, fill::zeros);
  arma::mat Z_(m, n_cov, fill::zeros);
  arma::mat U(n_spcov, n_spcov, fill::zeros);
  arma::mat L(n_spcov, n_spcov, fill::zeros);
  arma::mat WW_CHOL_U(n1, n_spcov, fill::zeros);
  arma::mat ZZ_CHOL_U(n_cov, n_cov, fill::zeros);
  arma::vec Wy(n_spcov, fill::zeros);
  arma::vec ones_m(m, fill::ones);
  arma::mat eye_m(m, m, fill::eye);
  arma::mat ZZ(n_cov, n_cov, fill::zeros);
  arma::mat L_ZZ(n_cov, n_cov, fill::zeros);
  arma::mat L_ZZ_INV(n_cov, n_cov, fill::zeros);
  arma::mat ZZ_INV(n_cov, n_cov, fill::zeros);
  arma::mat P_ZZ(m, m, fill::zeros);
  arma::mat M_ZZ(m, m, fill::zeros);
  arma::mat WMW(n_spcov, n_spcov, fill::zeros);
  arma::vec WMy(n_spcov, fill::zeros);
  arma::vec Zy(n_cov, fill::zeros);
  arma::vec v_LS(n_cov, fill::zeros);
  arma::mat ZW(n_cov, n_spcov, fill::zeros);
  arma::mat P_ZW(n_cov, n_spcov, fill::zeros);
  arma::mat FTF(n_spcov, n_spcov, fill::zeros);
  arma::colvec Glen(G, fill::zeros);
  arma::vec rho_store(maxiter+1, fill::zeros);
  arma::vec h_objval(maxiter, fill::zeros);
  arma::vec h_r_norm(maxiter, fill::zeros);
  arma::vec h_s_norm(maxiter, fill::zeros);
  arma::vec h_eps_pri(maxiter, fill::zeros);
  arma::vec h_eps_dual(maxiter, fill::zeros);
  
  /* Precompute relevant quantities         */
  Z_             = Z.each_col() % (ones_m * (1.0 / sqrtm));                       // 19Nov25: divisione per n controllata e risulta corretta
  W_             = W.each_col() % (ones_m * (1.0 / sqrtm));
  ZZ             = Z_.t() * Z_;
  L_ZZ           = (chol(ZZ)).t();
  L_ZZ_INV       = inv(trimatl(L_ZZ));
  ZZ_INV         = L_ZZ_INV.t() * L_ZZ_INV;
  P_ZZ           = Z_ * ZZ_INV * Z_.t();
  M_ZZ           = eye_m - P_ZZ;
  WMW            = W_.t() * M_ZZ * W_;
  WMy            = W_.t() * M_ZZ * y / static_cast<float>(sqrtm);
  Zy             = Z_.t() * y / static_cast<float>(sqrtm);
  v_LS           = L_ZZ_INV.t() * L_ZZ_INV * Zy;
  ZW             = Z_.t() * W_;
  P_ZW           = L_ZZ_INV.t() * L_ZZ_INV * ZW;
  WW_CHOL_U      = chol_qr_fact(W_, m, n_spcov);
  ZZ_CHOL_U      = chol_qr_fact(Z_, m, n_cov);
  y2             = as_scalar(y.t() * y) / static_cast<float>(m);
  Wy             = W_.t() * y / static_cast<float>(sqrtm);
  Glen           = sum(groups, 1);
  arma::sp_mat F = ovglasso_Gmat2Fmat_sparse(groups, var_weights);
  FTF            = F.t() * F;
  
  /* Initialize vector z_old         */
  dim_z = sum(Glen);
  arma::colvec z_old(dim_z, fill::zeros);

  /* ::::::::::::::::::::::::::::::::::::::::::::::
   Set the computational time                                 */
  wall_clock timer;
  timer.tic();
  
  /* store rho (only for adaptation)         */
  rho_store(0) = rho;
  rho_old      = rho;

  /* Precompute relevant quantities         */
  U = glasso_factor_cov_fast_large_m(WMW, FTF, rho);                      // returns upper
  L = U.t();
  
  /* main loop  */
  for (k=0; k<maxiter; k++) {

    // 4-1. update 'x'
    if (rho != rho_old) {
      U = glasso_factor_cov_fast_large_m(WMW, FTF, rho);                  // returns upper
      L = U.t();
    }
    q = WMy + rho * spmat2vec_mult(F.t(), z - u);        // temporary value
    x = solve(trimatu(U), solve(trimatl(L), q));         // updated of regression parameters associated to covariates W (sparsified covariates, x)
    v = v_LS - P_ZW * x;                                 // updated of regression parameters associated to covariates Z (covariates not sparsified, v)
    
    // 4-2. update 'z' with adaptation
    // see Boyd et al (2011) page 21
    z_old     = z;
    g_id_init = 0;
    for (int g=0; g<G; g++) {
      // precompute the quantities for updating z
      g_id_end         = g_id_init + Glen(g) - 1;
      arma::vec u_g    = u.subvec(g_id_init, g_id_end);
      arma::sp_mat F_g = F.submat(g_id_init, 0, g_id_end, n_spcov-1);

      // update z
      arma::vec z_g                 = glasso_prox(F_g * x + u_g, (lambda * group_weights[g]) / rho);
      z.subvec(g_id_init, g_id_end) = z_g;
      g_id_init                     = g_id_end + 1;
    }

    // 4-3. update 'u'
    u += spmat2vec_mult(F, x) - z;
  
    // 4-3. dianostics, reporting
    rss         = lm_cov_rss_fast(WW_CHOL_U, ZZ_CHOL_U, ZW, Wy, Zy, y2, x, v, m);
    pen         = ovglasso_penalty(x, groups, group_weights, var_weights);
    h_objval(k) = 0.5 *  rss / static_cast<float>(m) + lambda * pen;
    h_r_norm(k) = norm(glasso_residual_sparse(F, x, z), 2);
    h_s_norm(k) = norm(glasso_dual_residual_sparse(F, z, z_old, rho), 2);
    if (norm(F * x, 2) > norm(-z, 2)) {
      h_eps_pri(k) = std::sqrt(z.n_elem) * abstol + reltol * norm(F * x, 2);
    } else {
      h_eps_pri(k) = std::sqrt(z.n_elem) * abstol + reltol * norm(-z, 2);
    }
    h_eps_dual(k) = std::sqrt(z.n_elem) * abstol + reltol * norm(F.t() * rho * u, 2);
    
    /* :::::::::::::::::::::::::::::::::::::::::::::::::::::::: */
    /* Relaxation                                               */
    rho_old = rho;
    if (rho_adaptation) {
      if (h_r_norm(k) > mu * h_s_norm(k)) {
        rho = rho * tau;
        u   = u / tau;
      } else if (h_s_norm(k) > mu * h_r_norm(k)) {
        rho = rho / tau;
        u   = u * tau;
      }
    }
    rho_store(k+1) = rho;
        
    /* :::::::::::::::::::::::::::::::::::::::::::::::::::::::: */
    /* Print to screen                                          */
    if ((ping > 0) && (k < maxiter)) {
      if (rho_adaptation) {
        if (((k+1) % ping) == 0) {
          Rcpp::Rcout << "\n\n\n" << std::endl;
          Rcpp::Rcout << "ADMM with adaptation for overlap group-LASSO is running!" << std::endl;
          Rcpp::Rcout << "Iteration n.: " << k+1 << " of " << maxiter << "\n" << std::endl;
        }
      } else {
        if (((k+1) % ping) == 0) {
          Rcpp::Rcout << "\n\n\n" << std::endl;
          Rcpp::Rcout << "ADMM for overlap group-LASSO is running!" << std::endl;
          Rcpp::Rcout << "Iteration n.: " << k+1 << " of " << maxiter << "\n" << std::endl;
        }
      }
    }

    // 4-4. termination
    if ((h_r_norm(k) < h_eps_pri(k)) && (h_s_norm(k) < h_eps_dual(k))) {
      break;
    }
  }

  /* ::::::::::::::::::::::::::::::::::::::::::::::
   Compute elapsed time                                */
  elTime = timer.toc();
  
  /* Get output         */
  output["x"]      = x;               
  output["v"]      = v;               
  output["u"]      = u;
  output["z"]      = z;
  output["eltime"] = elTime;
  if (k < maxiter) {
    output["convergence"] = 1.0;
    output["niter"]       = k+1;               
    output["objval"]      = h_objval.subvec(0, k);        
    output["r_norm"]      = h_r_norm.subvec(0, k);
    output["s_norm"]      = h_s_norm.subvec(0, k);
    output["eps_pri"]     = h_eps_pri.subvec(0, k);
    output["eps_dual"]    = h_eps_dual.subvec(0, k);
    output["rho"]         = rho_store.subvec(0, k+1);
  } else {
    output["convergence"] = 0.0;
    output["niter"]       = maxiter;               
    output["objval"]      = h_objval;        
    output["r_norm"]      = h_r_norm;
    output["s_norm"]      = h_s_norm;
    output["eps_pri"]     = h_eps_pri;
    output["eps_dual"]    = h_eps_dual;
    output["rho"]         = rho_store;
  }
  
  /* Return output      */
  return(output);
}

Rcpp::List admm_ovglasso_cov_smo_large_m(const arma::mat& W,
                                         const arma::mat& Z,
                                         const arma::vec& y,
                                         arma::colvec& u,
                                         arma::colvec& z,
                                         const arma::mat& groups,
                                         const arma::vec& group_weights,
                                         const arma::vec& var_weights,
                                         const double lambda,
                                         const double lambda2,
                                         const int diff_order,
                                         const bool rho_adaptation,
                                         double rho,
                                         const double tau,
                                         const double mu,
                                         const double reltol,
                                         const double abstol,
                                         const int maxiter,
                                         const int ping) {

  /* Variable delcaration           */
  double y2 = 0.0, sqrtm = 0.0, rho_old = 0.0, elTime = 0.0, rss = 0.0, pen = 0.0, pen2 = 0.0;
  int n1 = 0, dim_z = 0, k = 0;
  uword g_id_init, g_id_end;

  /* Get dimensions             */
  const int m       = W.n_rows;
  const int n_spcov = W.n_cols;
  const int n_cov   = Z.n_cols;
  const int G       = groups.n_rows;
  
  /* Variable definition            */
  sqrtm = std::sqrt(static_cast<float>(m));

  /* Get output lists          */
  List out, output;

  /* Variable definition            */
  if (m >= n_spcov) {
    n1 = n_spcov;
  } else {
    n1 = m;
  }

  /* Definition of vectors and matrices     */
  arma::colvec x(n_spcov, fill::zeros);
  arma::colvec v(n_cov, fill::zeros);
  arma::colvec q(n_spcov, fill::zeros);
  arma::colvec x_star(n_spcov, fill::zeros);
  arma::mat W_(m, n_spcov, fill::zeros);
  arma::mat Z_(m, n_cov, fill::zeros);
  arma::mat U(n_spcov, n_spcov, fill::zeros);
  arma::mat L(n_spcov, n_spcov, fill::zeros);
  arma::mat WW_CHOL_U(n1, n_spcov, fill::zeros);
  arma::mat ZZ_CHOL_U(n_cov, n_cov, fill::zeros);
  arma::vec Wy(n_spcov, fill::zeros);
  arma::vec ones_m(m, fill::ones);
  arma::mat eye_m(m, m, fill::eye);
  arma::mat ZZ(n_cov, n_cov, fill::zeros);
  arma::mat L_ZZ(n_cov, n_cov, fill::zeros);
  arma::mat L_ZZ_INV(n_cov, n_cov, fill::zeros);
  arma::mat ZZ_INV(n_cov, n_cov, fill::zeros);
  arma::mat P_ZZ(m, m, fill::zeros);
  arma::mat M_ZZ(m, m, fill::zeros);
  arma::mat WMW(n_spcov, n_spcov, fill::zeros);
  arma::vec WMy(n_spcov, fill::zeros);
  arma::vec Zy(n_cov, fill::zeros);
  arma::vec v_LS(n_cov, fill::zeros);
  arma::mat ZW(n_cov, n_spcov, fill::zeros);
  arma::mat P_ZW(n_cov, n_spcov, fill::zeros);
  arma::mat FTF(n_spcov, n_spcov, fill::zeros);
  arma::mat DTD(n_spcov, n_spcov, fill::zeros);
  arma::colvec Glen(G, fill::zeros);
  arma::vec rho_store(maxiter+1, fill::zeros);
  arma::vec h_objval(maxiter, fill::zeros);
  arma::vec h_r_norm(maxiter, fill::zeros);
  arma::vec h_s_norm(maxiter, fill::zeros);
  arma::vec h_eps_pri(maxiter, fill::zeros);
  arma::vec h_eps_dual(maxiter, fill::zeros);
  
  /* Precompute relevant quantities         */
  Z_             = Z.each_col() % (ones_m * (1.0 / sqrtm));                   // 19Nov25: divisione per n controllata e risulta corretta
  W_             = W.each_col() % (ones_m * (1.0 / sqrtm));
  ZZ             = Z_.t() * Z_;
  L_ZZ           = (chol(ZZ)).t();
  L_ZZ_INV       = inv(trimatl(L_ZZ));
  ZZ_INV         = L_ZZ_INV.t() * L_ZZ_INV;
  P_ZZ           = Z_ * ZZ_INV * Z_.t();
  M_ZZ           = eye_m - P_ZZ;
  WMW            = W_.t() * M_ZZ * W_;
  WMy            = W_.t() * M_ZZ * y / static_cast<float>(sqrtm);
  Zy             = Z_.t() * y / static_cast<float>(sqrtm);
  v_LS           = L_ZZ_INV.t() * L_ZZ_INV * Zy;
  ZW             = Z_.t() * W_;
  P_ZW           = L_ZZ_INV.t() * L_ZZ_INV * ZW;
  WW_CHOL_U      = chol_qr_fact(W_, m, n_spcov);
  ZZ_CHOL_U      = chol_qr_fact(Z_, m, n_cov);
  y2             = as_scalar(y.t() * y) / static_cast<float>(m);
  Wy             = W_.t() * y / static_cast<float>(sqrtm);
  Glen           = sum(groups, 1);
  arma::sp_mat F = ovglasso_Gmat2Fmat_sparse(groups, var_weights);
  FTF            = F.t() * F;
  DTD            = forward_diff_penalty_matrix(n_spcov, 1, diff_order);
  
  /* Initialize vector z_old         */
  dim_z = sum(Glen);
  arma::colvec z_old(dim_z, fill::zeros);

  /* ::::::::::::::::::::::::::::::::::::::::::::::
   Set the computational time                                 */
  wall_clock timer;
  timer.tic();
  
  /* store rho (only for adaptation)         */
  rho_store(0) = rho;
  rho_old      = rho;

  /* Precompute relevant quantities         */
  //U = glasso_factor_fast_large_m(ATA, FTF, rho); // returns upper
  U = glasso_factor_cov_smo_fast_large_m(WMW, FTF, DTD, lambda2, rho);
  L = U.t();

  /* main loop  */
  for (k=0; k<maxiter; k++) {

    // 4-1. update 'x'
    if (rho != rho_old) {
      //U = glasso_factor_fast_large_m(ATA, FTF, rho); // returns upper
      U = glasso_factor_cov_smo_fast_large_m(WMW, FTF, DTD, lambda2, rho);
      L = U.t();
    }
    q = WMy + rho * spmat2vec_mult(F.t(), z - u);        // temporary value
    x = solve(trimatu(U), solve(trimatl(L), q));         // updated of regression parameters associated to covariates W (sparsified covariates, x)
    v = v_LS - P_ZW * x;
    
    // 4-2. update 'z' with adaptation
    // see Boyd et al (2011) page 21
    z_old     = z;
    g_id_init = 0;
    for (int g=0; g<G; g++) {
      // precompute the quantities for updating z
      g_id_end         = g_id_init + Glen(g) - 1;
      arma::vec u_g    = u.subvec(g_id_init, g_id_end);
      arma::sp_mat F_g = F.submat(g_id_init, 0, g_id_end, n_spcov-1);

      // update z
      arma::vec z_g                 = glasso_prox(F_g * x + u_g, (lambda * group_weights[g]) / rho);
      z.subvec(g_id_init, g_id_end) = z_g;
      g_id_init                     = g_id_end + 1;
    }

    // 4-3. update 'u'
    u += spmat2vec_mult(F, x) - z;
  
    // 4-3. dianostics, reporting
    rss         = lm_cov_rss_fast(WW_CHOL_U, ZZ_CHOL_U, ZW, Wy, Zy, y2, x, v, m);
    pen         = ovglasso_penalty(x, groups, group_weights, var_weights);
    pen2        = 0.5 * as_scalar(x.t() * DTD * x);
    h_objval(k) = 0.5 *  rss / static_cast<float>(m) + lambda * pen + lambda2 * pen2;
    h_r_norm(k) = norm(glasso_residual_sparse(F, x, z), 2);
    h_s_norm(k) = norm(glasso_dual_residual_sparse(F, z, z_old, rho), 2);
    if (norm(F * x, 2) > norm(-z, 2)) {
      h_eps_pri(k) = std::sqrt(z.n_elem) * abstol + reltol * norm(F * x, 2);
    } else {
      h_eps_pri(k) = std::sqrt(z.n_elem) * abstol + reltol * norm(-z, 2);
    }
    h_eps_dual(k) = std::sqrt(z.n_elem) * abstol + reltol * norm(F.t() * rho * u, 2);
    
    /* :::::::::::::::::::::::::::::::::::::::::::::::::::::::: */
    /* Relaxation                                               */
    rho_old = rho;
    if (rho_adaptation) {
      if (h_r_norm(k) > mu * h_s_norm(k)) {
        rho = rho * tau;
        u   = u / tau;
      } else if (h_s_norm(k) > mu * h_r_norm(k)) {
        rho = rho / tau;
        u   = u * tau;
      }
    }
    rho_store(k+1) = rho;
        
    /* :::::::::::::::::::::::::::::::::::::::::::::::::::::::: */
    /* Print to screen                                          */
    if ((ping > 0) && (k < maxiter)) {
      if (rho_adaptation) {
        if (((k+1) % ping) == 0) {
          Rcpp::Rcout << "\n\n\n" << std::endl;
          Rcpp::Rcout << "ADMM with adaptation for overlap group-LASSO is running!" << std::endl;
          Rcpp::Rcout << "Iteration n.: " << k+1 << " of " << maxiter << "\n" << std::endl;
        }
      } else {
        if (((k+1) % ping) == 0) {
          Rcpp::Rcout << "\n\n\n" << std::endl;
          Rcpp::Rcout << "ADMM for overlap group-LASSO is running!" << std::endl;
          Rcpp::Rcout << "Iteration n.: " << k+1 << " of " << maxiter << "\n" << std::endl;
        }
      }
    }

    // 4-4. termination
    if ((h_r_norm(k) < h_eps_pri(k)) && (h_s_norm(k) < h_eps_dual(k))) {
      break;
    }
  }

  /* ::::::::::::::::::::::::::::::::::::::::::::::
   Compute elapsed time                                */
  elTime = timer.toc();
  
  /* Get output         */
  output["x"]      = x;               
  output["v"]      = v;               
  output["u"]      = u;
  output["z"]      = z;
  output["eltime"] = elTime;
  if (k < maxiter) {
    output["convergence"] = 1.0;
    output["niter"]       = k+1;               
    output["objval"]      = h_objval.subvec(0, k);        
    output["r_norm"]      = h_r_norm.subvec(0, k);
    output["s_norm"]      = h_s_norm.subvec(0, k);
    output["eps_pri"]     = h_eps_pri.subvec(0, k);
    output["eps_dual"]    = h_eps_dual.subvec(0, k);
    output["rho"]         = rho_store.subvec(0, k+1);
  } else {
    output["convergence"] = 0.0;
    output["niter"]       = maxiter;               
    output["objval"]      = h_objval;        
    output["r_norm"]      = h_r_norm;
    output["s_norm"]      = h_s_norm;
    output["eps_pri"]     = h_eps_pri;
    output["eps_dual"]    = h_eps_dual;
    output["rho"]         = rho_store;
  }
  
  /* Return output      */
  return(output);
}

Rcpp::List admm_ovglasso_cov_large_n(const arma::mat& W,
                                     const arma::mat& Z,
                                     const arma::vec& y,
                                     arma::colvec& u,
                                     arma::colvec& z,
                                     const arma::mat& groups,
                                     const arma::vec& group_weights,
                                     const arma::vec& var_weights,
                                     const double lambda,
                                     const bool rho_adaptation,
                                     double rho,
                                     const double tau,
                                     const double mu,
                                     const double reltol,
                                     const double abstol,
                                     const int maxiter,
                                     const int ping) {

  /* Variable delcaration           */
  double y2 = 0.0, sqrtm = 0.0, rho_old = 0.0, elTime = 0.0, rss = 0.0, pen = 0.0;
  int n1 = 0, dim_z = 0, k = 0;
  uword g_id_init, g_id_end;

  /* Get dimensions             */
  const int m       = W.n_rows;
  const int n_spcov = W.n_cols;
  const int n_cov   = Z.n_cols;
  const int G = groups.n_rows;
  
  /* Variable definition            */
  //sqrtn = std::sqrt(static_cast<float>(n_spcov));
  sqrtm = std::sqrt(static_cast<float>(m));

  /* Get output lists          */
  List out, output;

  /* Variable definition            */
  if (m >= n_spcov) {
    n1 = n_spcov;
  } else {
    n1 = m;
  }

  /* Definition of vectors and matrices     */
  arma::colvec x(n_spcov, fill::zeros);
  arma::colvec v(n_cov, fill::zeros);
  arma::colvec q(n_spcov, fill::zeros);
  arma::colvec x_star(n_spcov, fill::zeros);
  arma::mat Z_(m, n_cov, fill::zeros);
  arma::mat U(m, m, fill::zeros);
  arma::mat L(m, m, fill::zeros);
  arma::mat WW_CHOL_U(n1, n_spcov, fill::zeros);
  arma::mat ZZ_CHOL_U(n_cov, n_cov, fill::zeros);
  arma::vec Wy(n_spcov, fill::zeros);
  arma::vec ones_m(m, fill::ones);
  arma::mat eye_m(m, m, fill::eye);
  arma::mat ZZ(n_cov, n_cov, fill::zeros);
  arma::mat L_ZZ(n_cov, n_cov, fill::zeros);
  arma::mat L_ZZ_INV(n_cov, n_cov, fill::zeros);
  arma::mat ZZ_INV(n_cov, n_cov, fill::zeros);
  arma::mat P_ZZ(m, m, fill::zeros);
  arma::mat M_ZZ(m, m, fill::zeros);
  arma::mat WMW(n_spcov, n_spcov, fill::zeros);
  arma::vec WMy(n_spcov, fill::zeros);
  arma::vec Zy(n_cov, fill::zeros);
  arma::vec v_LS(n_cov, fill::zeros);
  arma::mat ZW(n_cov, n_spcov, fill::zeros);
  arma::mat P_ZW(n_cov, n_spcov, fill::zeros);
  //arma::mat WW(m, m, fill::zeros);
  arma::mat M_ZZW(m, n_spcov, fill::zeros);
  arma::mat M_ZZW2(m, m, fill::zeros);
  arma::mat FTF(n_spcov, n_spcov, fill::zeros);
  arma::mat FTF_INV(n_spcov, n_spcov, fill::zeros);
  arma::colvec Glen(G, fill::zeros);
  arma::vec rho_store(maxiter+1, fill::zeros);
  arma::vec h_objval(maxiter, fill::zeros);
  arma::vec h_r_norm(maxiter, fill::zeros);
  arma::vec h_s_norm(maxiter, fill::zeros);
  arma::vec h_eps_pri(maxiter, fill::zeros);
  arma::vec h_eps_dual(maxiter, fill::zeros);
  
  /* Precompute relevant quantities         */
  Z_        = Z.each_col() % (ones_m * (1.0 / sqrtm));
  //W         = W.each_col() % (ones_m * (1.0 / sqrtm));          // è corretto non dividere W per la numerosità campionaria m
  ZZ        = Z_.t() * Z_;
  L_ZZ      = (chol(ZZ)).t();
  L_ZZ_INV  = inv(trimatl(L_ZZ));
  ZZ_INV    = L_ZZ_INV.t() * L_ZZ_INV;
  P_ZZ      = Z_ * ZZ_INV * Z_.t();
  M_ZZ      = eye_m - P_ZZ;
  WMW       = W.t() * M_ZZ * W;                                 // 19Nov25: divisione per n corretta così come è ora
  WMy       = W.t() * M_ZZ * y;
  Zy        = Z_.t() * y / static_cast<float>(sqrtm);
  v_LS      = L_ZZ_INV.t() * L_ZZ_INV * Zy;
  ZW        = Z_.t() * W / static_cast<float>(sqrtm);
  P_ZW      = L_ZZ_INV.t() * L_ZZ_INV * ZW;
  WW_CHOL_U = chol_qr_fact(W, m, n_spcov);
  ZZ_CHOL_U = chol_qr_fact(Z_, m, n_cov);
  y2        = as_scalar(y.t() * y) / static_cast<float>(m);
  Wy        = W.t() * y;
  M_ZZW     = M_ZZ * W;               // 19Nov25: divisione per n corretta così come è ora
  
  Glen           = sum(groups, 1);
  arma::sp_mat F = ovglasso_Gmat2Fmat_sparse(groups, var_weights);
  FTF            = diagmat(spmat2spmat_mult(F.t(), F));
  FTF_INV        = diagmat(1.0 / diagvec(FTF));
  M_ZZW2         = M_ZZW * diagmat(FTF_INV) * M_ZZW.t();
  
  /* Initialize vectors z_old         */
  dim_z = sum(Glen);
  arma::colvec z_old(dim_z, fill::zeros);
  
  /* ::::::::::::::::::::::::::::::::::::::::::::::
   Set the computational time                                 */
  wall_clock timer;
  timer.tic();
  
  /* store rho (only for adaptation)         */
  rho_store(0) = rho;
  rho_old      = rho;

  /* Precompute relevant quantities         */
  U = glasso_factor_cov_fast_large_n(M_ZZW2, rho, m); // returns upper
  L = U.t();
  
  /* main loop  */
  for (k=0; k<maxiter; k++) {

    // 4-1. update 'x'
    if (rho != rho_old) {
      U = glasso_factor_cov_fast_large_n(M_ZZW2, rho, m); // returns upper
      L = U.t();
    }
    q      = WMy + rho * spmat2vec_mult(F.t(), z - u); // temporary value
    x_star = (diagmat(FTF_INV) * q) / rho;
    x      = x_star - diagmat(FTF_INV) * M_ZZW.t() * solve(trimatu(U), solve(trimatl(L), M_ZZW * x_star)); // updated of regression parameters associated to covariates W
    v      = v_LS - P_ZW * x;                                 // updated of regression parameters associated to covariates Z (covariates not sparsified, v)

    // 4-2. update 'z' with adaptation
    // see Boyd et al (2011) page 21
    z_old     = z;
    g_id_init = 0;
    for (int g=0; g<G; g++) {
      // precompute the quantities for updating z
      g_id_end         = g_id_init + Glen(g) - 1;
      arma::vec u_g    = u.subvec(g_id_init, g_id_end);
      arma::sp_mat F_g = F.submat(g_id_init, 0, g_id_end, n_spcov-1);

      // update z
      arma::vec z_g                 = glasso_prox(F_g * x + u_g, (lambda * group_weights[g]) / rho);
      z.subvec(g_id_init, g_id_end) = z_g;
      g_id_init                     = g_id_end + 1;
    }

    // 4-3. update 'u'
    u += spmat2vec_mult(F, x) - z;
  
    // 4-3. dianostics, reporting
    rss         = lm_cov_rss_fast(WW_CHOL_U, ZZ_CHOL_U, ZW, Wy, Zy, y2, x, v, m);
    pen         = ovglasso_penalty(x, groups, group_weights, var_weights);
    h_objval(k) = 0.5 *  rss / static_cast<float>(m) + lambda * pen;
    h_r_norm(k) = norm(glasso_residual_sparse(F, x, z), 2);
    h_s_norm(k) = norm(glasso_dual_residual_sparse(F, z, z_old, rho), 2);
    if (norm(F * x, 2) > norm(-z, 2)) {
      h_eps_pri(k) = std::sqrt(z.n_elem) * abstol + reltol * norm(F * x, 2);
    } else {
      h_eps_pri(k) = std::sqrt(z.n_elem) * abstol + reltol * norm(-z, 2);
    }
    h_eps_dual(k) = std::sqrt(z.n_elem) * abstol + reltol * norm(F.t() * rho * u, 2);
    
    /* :::::::::::::::::::::::::::::::::::::::::::::::::::::::: */
    /* Relaxation                                               */
    rho_old = rho;
    if (rho_adaptation) {
      if (h_r_norm(k) > mu * h_s_norm(k)) {
        rho = rho * tau;
        u   = u / tau;
      } else if (h_s_norm(k) > mu * h_r_norm(k)) {
        rho = rho / tau;
        u   = u * tau;
      }
    }
    rho_store(k+1) = rho;
        
    /* :::::::::::::::::::::::::::::::::::::::::::::::::::::::: */
    /* Print to screen                                          */
    if ((ping > 0) && (k < maxiter)) {
      if (rho_adaptation) {
        if (((k+1) % ping) == 0) {
          Rcpp::Rcout << "\n\n\n" << std::endl;
          Rcpp::Rcout << "ADMM with adaptation for overlap group-LASSO is running!" << std::endl;
          Rcpp::Rcout << "Iteration n.: " << k+1 << " of " << maxiter << "\n" << std::endl;
        }
      } else {
        if (((k+1) % ping) == 0) {
          Rcpp::Rcout << "\n\n\n" << std::endl;
          Rcpp::Rcout << "ADMM for overlap group-LASSO is running!" << std::endl;
          Rcpp::Rcout << "Iteration n.: " << k+1 << " of " << maxiter << "\n" << std::endl;
        }
      }
    }

    // 4-4. termination
    if ((h_r_norm(k) < h_eps_pri(k)) && (h_s_norm(k) < h_eps_dual(k))) {
      break;
    }
  }

  /* ::::::::::::::::::::::::::::::::::::::::::::::
   Compute elapsed time                                */
  elTime = timer.toc();
  
  /* Get output         */
  output["x"]      = x;               
  output["v"]      = v;               
  output["u"]      = u;
  output["z"]      = z;
  output["eltime"] = elTime;
  if (k < maxiter) {
    output["convergence"] = 1.0;
    output["niter"]       = k+1;               
    output["objval"]      = h_objval.subvec(0, k);        
    output["r_norm"]      = h_r_norm.subvec(0, k);
    output["s_norm"]      = h_s_norm.subvec(0, k);
    output["eps_pri"]     = h_eps_pri.subvec(0, k);
    output["eps_dual"]    = h_eps_dual.subvec(0, k);
    output["rho"]         = rho_store.subvec(0, k+1);
  } else {
    output["convergence"] = 0.0;
    output["niter"]       = maxiter;               
    output["objval"]      = h_objval;        
    output["r_norm"]      = h_r_norm;
    output["s_norm"]      = h_s_norm;
    output["eps_pri"]     = h_eps_pri;
    output["eps_dual"]    = h_eps_dual;
    output["rho"]         = rho_store;
  }

  /* Return output      */
  return(output);
}

Rcpp::List admm_ovglasso_cov_smo_large_n(const arma::mat& W,
                                         const arma::mat& Z,
                                         const arma::vec& y,
                                         arma::colvec& u,
                                         arma::colvec& z,
                                         const arma::mat& groups,
                                         const arma::vec& group_weights,
                                         const arma::vec& var_weights,
                                         const double lambda,
                                         const double lambda2,
                                         const int diff_order,
                                         const bool rho_adaptation,
                                         double rho,
                                         const double tau,
                                         const double mu,
                                         const double reltol,
                                         const double abstol,
                                         const int maxiter,
                                         const int ping) {

  /* Variable delcaration           */
  double y2 = 0.0, sqrtm = 0.0, rho_old = 0.0, elTime = 0.0, rss = 0.0, pen = 0.0, pen2 = 0.0;
  int n1 = 0, dim_z = 0, k = 0;
  uword g_id_init, g_id_end;

  /* Get dimensions             */
  const int m       = W.n_rows;
  const int n_spcov = W.n_cols;
  const int n_cov   = Z.n_cols;
  const int G = groups.n_rows;
  
  /* Variable definition            */
  //sqrtn = std::sqrt(static_cast<float>(n_spcov));
  sqrtm = std::sqrt(static_cast<float>(m));

  /* Get output lists          */
  List out, output;

  /* Variable definition            */
  if (m >= n_spcov) {
    n1 = n_spcov;
  } else {
    n1 = m;
  }

  /* Definition of vectors and matrices     */
  arma::colvec x(n_spcov, fill::zeros);
  arma::colvec v(n_cov, fill::zeros);
  arma::colvec q(n_spcov, fill::zeros);
  arma::colvec x_star(n_spcov, fill::zeros);
  arma::mat Z_(m, n_cov, fill::zeros);
  arma::mat U(m, m, fill::zeros);
  arma::mat L(m, m, fill::zeros);
  arma::mat WW_CHOL_U(n1, n_spcov, fill::zeros);
  arma::mat ZZ_CHOL_U(n_cov, n_cov, fill::zeros);
  arma::vec Wy(n_spcov, fill::zeros);
  arma::vec ones_m(m, fill::ones);
  arma::mat eye_m(m, m, fill::eye);
  arma::mat ZZ(n_cov, n_cov, fill::zeros);
  arma::mat L_ZZ(n_cov, n_cov, fill::zeros);
  arma::mat L_ZZ_INV(n_cov, n_cov, fill::zeros);
  arma::mat ZZ_INV(n_cov, n_cov, fill::zeros);
  arma::mat P_ZZ(m, m, fill::zeros);
  arma::mat M_ZZ(m, m, fill::zeros);
  arma::mat WMW(n_spcov, n_spcov, fill::zeros);
  arma::vec WMy(n_spcov, fill::zeros);
  arma::vec Zy(n_cov, fill::zeros);
  arma::vec v_LS(n_cov, fill::zeros);
  arma::mat ZW(n_cov, n_spcov, fill::zeros);
  arma::mat P_ZW(n_cov, n_spcov, fill::zeros);
  //arma::mat WW(m, m, fill::zeros);
  //arma::mat M_ZZW(m, n_spcov, fill::zeros);
  //arma::mat M_ZZW2(m, m, fill::zeros);
  arma::mat FTF(n_spcov, n_spcov, fill::zeros);
  arma::mat FTF_INV(n_spcov, n_spcov, fill::zeros);
  arma::mat D(n_spcov-diff_order, n_spcov, fill::zeros);
  arma::mat DTD(n_spcov, n_spcov, fill::zeros);
  arma::mat Atilde(n_spcov+m, n_spcov, fill::zeros);
  arma::mat AtFTF_INVAt(n_spcov+m, n_spcov+m, fill::zeros);
  arma::colvec Glen(G, fill::zeros);
  arma::vec rho_store(maxiter+1, fill::zeros);
  arma::vec h_objval(maxiter, fill::zeros);
  arma::vec h_r_norm(maxiter, fill::zeros);
  arma::vec h_s_norm(maxiter, fill::zeros);
  arma::vec h_eps_pri(maxiter, fill::zeros);
  arma::vec h_eps_dual(maxiter, fill::zeros);
  
  /* Precompute relevant quantities         */
  Z_                        = Z.each_col() % (ones_m * (1.0 / sqrtm));
  //W                         = W.each_col() % (ones_m * (1.0 / sqrtm));      // è corretto non dividere W per la numerosità campionaria m
  ZZ                        = Z_.t() * Z_;
  L_ZZ                      = (chol(ZZ)).t();
  L_ZZ_INV                  = inv(trimatl(L_ZZ));
  ZZ_INV                    = L_ZZ_INV.t() * L_ZZ_INV;
  P_ZZ                      = Z_ * ZZ_INV * Z_.t();
  M_ZZ                      = eye_m - P_ZZ;
  WMW                       = W.t() * M_ZZ * W;                               // 19Nov25: divisione per n corretta così come è ora
  WMy                       = W.t() * M_ZZ * y;
  Zy                        = Z_.t() * y / static_cast<float>(sqrtm);
  v_LS                      = L_ZZ_INV.t() * L_ZZ_INV * Zy;
  ZW                        = Z_.t() * W / static_cast<float>(sqrtm);
  P_ZW                      = L_ZZ_INV.t() * L_ZZ_INV * ZW;
  WW_CHOL_U                 = chol_qr_fact(W, m, n_spcov);
  ZZ_CHOL_U                 = chol_qr_fact(Z_, m, n_cov);
  y2                        = as_scalar(y.t() * y) / static_cast<float>(m);
  Wy                        = W.t() * y;
  //M_ZZW                     = M_ZZ * W / static_cast<float>(m);                 // 19Nov25: divisione per n corretta così come è ora
  Glen                      = sum(groups, 1);
  arma::sp_mat F            = ovglasso_Gmat2Fmat_sparse(groups, var_weights);
  FTF                       = diagmat(spmat2spmat_mult(F.t(), F));
  FTF_INV                   = diagmat(1.0 / diagvec(FTF));
  D                         = forward_diff_difference_matrix(n_spcov, 1, diff_order);
  DTD                       = D.t() * D;
  Atilde.head_rows(m)       = W * M_ZZ / sqrt(static_cast<float>(m));         // compute relevant quantities: these quantities change with lambda2
  Atilde.tail_rows(n_spcov) = std::sqrt(lambda2) * D;
  AtFTF_INVAt               = Atilde * FTF_INV * Atilde.t();

  /* Initialize vectors z_old         */
  dim_z = sum(Glen);
  arma::colvec z_old(dim_z, fill::zeros);
  
  /* ::::::::::::::::::::::::::::::::::::::::::::::
   Set the computational time                                 */
  wall_clock timer;
  timer.tic();
  
  /* store rho (only for adaptation)         */
  rho_store(0) = rho;
  rho_old      = rho;

  /* Precompute relevant quantities         */
  U = glasso_factor_cov_smo_fast_large_n(AtFTF_INVAt, rho, m); // returns upper
  L = U.t();
  
  /* main loop  */
  for (k=0; k<maxiter; k++) {

    // 4-1. update 'x'
    if (rho != rho_old) {
      U = glasso_factor_cov_smo_fast_large_n(AtFTF_INVAt, rho, m); // returns upper
      L = U.t();
    }
    q      = WMy + rho * spmat2vec_mult(F.t(), z - u); // temporary value
    x_star = (diagmat(FTF_INV) * q) / rho;
    
    // updated of regression parameters associated to covariates W
    x = x_star - diagmat(FTF_INV) * Atilde.t() * solve(trimatu(U), solve(trimatl(L), Atilde * x_star));
    
    // updated of regression parameters associated to covariates Z (covariates not sparsified, v)
    v = v_LS - P_ZW * x;

    // 4-2. update 'z' with adaptation
    // see Boyd et al (2011) page 21
    z_old     = z;
    g_id_init = 0;
    for (int g=0; g<G; g++) {
      // precompute the quantities for updating z
      g_id_end         = g_id_init + Glen(g) - 1;
      arma::vec u_g    = u.subvec(g_id_init, g_id_end);
      arma::sp_mat F_g = F.submat(g_id_init, 0, g_id_end, n_spcov-1);

      // update z
      arma::vec z_g                 = glasso_prox(F_g * x + u_g, (lambda * group_weights[g]) / rho);
      z.subvec(g_id_init, g_id_end) = z_g;
      g_id_init                     = g_id_end + 1;
    }

    // 4-3. update 'u'
    u += spmat2vec_mult(F, x) - z;
  
    // 4-3. dianostics, reporting
    rss         = lm_cov_rss_fast(WW_CHOL_U, ZZ_CHOL_U, ZW, Wy, Zy, y2, x, v, m);
    pen         = ovglasso_penalty(x, groups, group_weights, var_weights);
    pen2        = 0.5 * as_scalar(x.t() * DTD * x);
    h_objval(k) = 0.5 *  rss / static_cast<float>(m) + lambda * pen + lambda2 * pen2;
    h_r_norm(k) = norm(glasso_residual_sparse(F, x, z), 2);
    h_s_norm(k) = norm(glasso_dual_residual_sparse(F, z, z_old, rho), 2);
    if (norm(F * x, 2) > norm(-z, 2)) {
      h_eps_pri(k) = std::sqrt(z.n_elem) * abstol + reltol * norm(F * x, 2);
    } else {
      h_eps_pri(k) = std::sqrt(z.n_elem) * abstol + reltol * norm(-z, 2);
    }
    h_eps_dual(k) = std::sqrt(z.n_elem) * abstol + reltol * norm(F.t() * rho * u, 2);
    
    /* :::::::::::::::::::::::::::::::::::::::::::::::::::::::: */
    /* Relaxation                                               */
    rho_old = rho;
    if (rho_adaptation) {
      if (h_r_norm(k) > mu * h_s_norm(k)) {
        rho = rho * tau;
        u   = u / tau;
      } else if (h_s_norm(k) > mu * h_r_norm(k)) {
        rho = rho / tau;
        u   = u * tau;
      }
    }
    rho_store(k+1) = rho;
        
    /* :::::::::::::::::::::::::::::::::::::::::::::::::::::::: */
    /* Print to screen                                          */
    if ((ping > 0) && (k < maxiter)) {
      if (rho_adaptation) {
        if (((k+1) % ping) == 0) {
          Rcpp::Rcout << "\n\n\n" << std::endl;
          Rcpp::Rcout << "ADMM with adaptation for overlap group-LASSO is running!" << std::endl;
          Rcpp::Rcout << "Iteration n.: " << k+1 << " of " << maxiter << "\n" << std::endl;
        }
      } else {
        if (((k+1) % ping) == 0) {
          Rcpp::Rcout << "\n\n\n" << std::endl;
          Rcpp::Rcout << "ADMM for overlap group-LASSO is running!" << std::endl;
          Rcpp::Rcout << "Iteration n.: " << k+1 << " of " << maxiter << "\n" << std::endl;
        }
      }
    }

    // 4-4. termination
    if ((h_r_norm(k) < h_eps_pri(k)) && (h_s_norm(k) < h_eps_dual(k))) {
      break;
    }
  }

  /* ::::::::::::::::::::::::::::::::::::::::::::::
   Compute elapsed time                                */
  elTime = timer.toc();
  
  /* Get output         */
  output["x"]      = x;               
  output["v"]      = v;               
  output["u"]      = u;
  output["z"]      = z;
  output["eltime"] = elTime;
  if (k < maxiter) {
    output["convergence"] = 1.0;
    output["niter"]       = k+1;               
    output["objval"]      = h_objval.subvec(0, k);        
    output["r_norm"]      = h_r_norm.subvec(0, k);
    output["s_norm"]      = h_s_norm.subvec(0, k);
    output["eps_pri"]     = h_eps_pri.subvec(0, k);
    output["eps_dual"]    = h_eps_dual.subvec(0, k);
    output["rho"]         = rho_store.subvec(0, k+1);
  } else {
    output["convergence"] = 0.0;
    output["niter"]       = maxiter;               
    output["objval"]      = h_objval;        
    output["r_norm"]      = h_r_norm;
    output["s_norm"]      = h_s_norm;
    output["eps_pri"]     = h_eps_pri;
    output["eps_dual"]    = h_eps_dual;
    output["rho"]         = rho_store;
  }

  /* Return output      */
  return(output);
}

/*
    Overlap Group-LASSO via ADMM: main functions, to be used
                                  to fit the model several times
                                  (multiple lambdas)                  */

Rcpp::List admm_ovglasso_large_m_fast(const arma::mat& A,
                                      const arma::vec& b,
                                      const arma::mat& groups,
                                      const arma::vec& group_weights,
                                      const arma::vec& var_weights,
                                      const arma::vec& lambda,
                                      const bool rho_adaptation,
                                      const double rho,
                                      const double tau,
                                      const double mu,
                                      const double reltol,
                                      const double abstol,
                                      const int maxiter,
                                      const int ping) {

  /* Variable delcaration           */
  double b2 = 0.0, min_mse = 0.0;
  int n1 = 0, dim_z = 0;
  uword idx_min_mse;

  /* Get dimensions             */
  const int m = A.n_rows;
  const int n = A.n_cols;
  const int G = groups.n_rows;

  /* Get number of lambda           */
  const int nlambda = lambda.n_elem;

  /* Get output lists          */
  List out, output;

  /* Variable definition            */
  if (m >= n) {
    n1 = n;
  } else {
    n1 = m;
  }

  /* Definition of vectors and matrices     */
  arma::mat ATA(n, n, fill::zeros);
  arma::vec ATb(n, fill::zeros);
  arma::mat ATA_CHOL_U(n1, n, fill::zeros);
  arma::mat FTF(n, n, fill::zeros);
  arma::colvec x(n, fill::zeros);
  arma::colvec Glen(G, fill::zeros);

  /* Store output    */
  arma::field<vec> h_objval(nlambda);
  arma::field<vec> h_r_norm(nlambda);
  arma::field<vec> h_s_norm(nlambda);
  arma::field<vec> h_eps_pri(nlambda);
  arma::field<vec> h_eps_dual(nlambda);
  arma::field<vec> h_rho(nlambda);
  arma::vec niter(nlambda, fill::zeros);
  arma::vec eltime(nlambda, fill::zeros);
  arma::mat X(nlambda, n, fill::zeros);
  arma::vec conv(nlambda, fill::zeros);
  arma::mat residuals(nlambda, m, fill::zeros);
  arma::mat fitted(nlambda, m, fill::zeros);
  arma::vec mse(nlambda, fill::zeros);
  
  /* Precompute relevant quantities         */
  ATA            = A.t() * A / static_cast<float>(m);
  ATb            = A.t() * b / static_cast<float>(m);
  ATA_CHOL_U     = chol_qr_fact(A, m, n);
  b2             = as_scalar(b.t() * b) / static_cast<float>(m);
  Glen           = sum(groups, 1);
  arma::sp_mat F = ovglasso_Gmat2Fmat_sparse(groups, var_weights);
  FTF            = F.t() * F;
    
  /* Initialize vectors u and z         */
  dim_z = sum(Glen);
  arma::colvec u(dim_z, fill::zeros);
  arma::colvec z(dim_z, fill::zeros);
  arma::mat Umat(nlambda, dim_z, fill::zeros);
  arma::mat Zmat(nlambda, dim_z, fill::zeros);

  /* main loop  */
  for (int j=0; j<nlambda; j++) {
    /* perform linear regression with lasso penalty for the j-th lambda       */
    out = admm_ovglasso_large_m_update_1lambda(A, b, Glen, groups, group_weights, var_weights,
                                               F, FTF, ATA, ATA_CHOL_U, ATb,
                                               b2, m, n, G, u, z, lambda(j),
                                               rho_adaptation, rho, tau, mu,
                                               reltol, abstol, maxiter, ping);

    /* Retrieve output      */
    u = as<arma::colvec>(out["u"]);
    z = as<arma::colvec>(out["z"]);
    x = as<arma::colvec>(out["x"]);

    /* Store output      */
    niter(j)         = out["niter"];
    eltime(j)        = out["eltime"];
    conv(j)          = out["convergence"];
    X.row(j)         = x.t();
    h_objval(j)      = as<arma::vec>(out["objval"]);
    h_r_norm(j)      = as<arma::vec>(out["r_norm"]);
    h_s_norm(j)      = as<arma::vec>(out["s_norm"]);
    h_eps_pri(j)     = as<arma::vec>(out["eps_pri"]);
    h_eps_dual(j)    = as<arma::vec>(out["eps_dual"]);
    h_rho(j)         = (as<arma::vec>(out["rho"])).subvec(0, niter(j)-1);
    residuals.row(j) = (as<arma::vec>(out["residuals"])).t();
    fitted.row(j)    = (as<arma::vec>(out["fitted.values"])).t();
    mse(j)           = out["mse"];
    Umat.row(j)      = u.t();
    Zmat.row(j)      = z.t();
  }

  /* Find min MSE in-sample      */
  idx_min_mse = index_min(mse);
  min_mse     = mse(idx_min_mse);

  /* Get output         */
  output["coef.path"]    = X;
  output["iternum"]      = niter;               
  output["objfun"]       = h_objval;        
  output["r_norm"]       = h_r_norm;
  output["s_norm"]       = h_s_norm;
  output["err_pri"]      = h_eps_pri;
  output["err_dual"]     = h_eps_dual;
  output["convergence"]  = conv;
  output["elapsedTime"]  = eltime;
  output["rho"]          = h_rho;
  output["lambda"]       = lambda;
  output["min.mse"]      = min_mse;
  output["indi.min.mse"] = idx_min_mse;
  output["lambda.min"]   = lambda[idx_min_mse];
  output["coefficients"] = (X.row(idx_min_mse)).t();
  output["mse"]          = mse;
  output["U"]            = Umat;
  output["Z"]            = Zmat;

  /* Return output      */
  return(output);
}

Rcpp::List admm_ovglasso_smo_large_m_fast(const arma::mat& A,
                                          const arma::vec& b,
                                          const arma::mat& groups,
                                          const arma::vec& group_weights,
                                          const arma::vec& var_weights,
                                          const arma::vec& lambda,
                                          const arma::vec& lambda2,
                                          const int diff_order,
                                          const bool rho_adaptation,
                                          double rho,
                                          const double tau,
                                          const double mu,
                                          const double reltol,
                                          const double abstol,
                                          const int maxiter,
                                          const int ping) {

  /* Variable delcaration           */
  double b2 = 0.0, min_mse = 0.0;
  int n1 = 0, dim_z = 0;
  uword idx_min_mse, idx_min_mse_row, idx_min_mse_col;
  
  /* Get dimensions             */
  const int m = A.n_rows;
  const int n = A.n_cols;
  const int G = groups.n_rows;

  /* Get number of lambda           */
  const int nlambda  = lambda.n_elem;
  const int nlambda2 = lambda2.n_elem;

  /* Get output lists          */
  List out, output;

  /* Variable definition            */
  if (m >= n) {
    n1 = n;
  } else {
    n1 = m;
  }

  /* Definition of vectors and matrices     */
  arma::mat ATA(n, n, fill::zeros);
  arma::vec ATb(n, fill::zeros);
  arma::mat ATA_CHOL_U(n1, n, fill::zeros);
  arma::mat FTF(n, n, fill::zeros);
  arma::mat DTD(n, n, fill::zeros);
  arma::colvec x(n, fill::zeros);
  arma::colvec Glen(G, fill::zeros);

  /* Store output    */
  arma::field<vec> h_objval(nlambda, nlambda2);
  arma::field<vec> h_r_norm(nlambda, nlambda2);
  arma::field<vec> h_s_norm(nlambda, nlambda2);
  arma::field<vec> h_eps_pri(nlambda, nlambda2);
  arma::field<vec> h_eps_dual(nlambda, nlambda2);
  arma::field<vec> h_rho(nlambda, nlambda2);
  arma::mat niter(nlambda, nlambda2, fill::zeros);
  arma::mat eltime(nlambda, nlambda2, fill::zeros);
  arma::mat conv(nlambda, nlambda2, fill::zeros);
  arma::cube X(nlambda, nlambda2, n, fill::zeros);
  arma::cube residuals(nlambda, nlambda2, m, fill::zeros);
  arma::cube fitted(nlambda, nlambda2, m, fill::zeros);
  arma::mat mse(nlambda, nlambda2, fill::zeros);
  
  /* Precompute relevant quantities         */
  ATA            = A.t() * A / static_cast<float>(m);
  ATb            = A.t() * b / static_cast<float>(m);
  ATA_CHOL_U     = chol_qr_fact(A, m, n);
  b2             = as_scalar(b.t() * b) / static_cast<float>(m);
  Glen           = sum(groups, 1);
  arma::sp_mat F = ovglasso_Gmat2Fmat_sparse(groups, var_weights);
  FTF            = F.t() * F;
  DTD            = forward_diff_penalty_matrix(n, 1, diff_order);
  
  /* Initialize vectors u and z         */
  dim_z = sum(Glen);
  arma::colvec u(dim_z, fill::zeros);
  arma::colvec z(dim_z, fill::zeros);
  arma::cube Umat(nlambda, nlambda2, dim_z, fill::zeros);
  arma::cube Zmat(nlambda, nlambda2, dim_z, fill::zeros);

  /* main loop  */
  for (int j=0; j<nlambda; j++) {
    for (int k=0; k<nlambda2; k++) {
      /* perform linear regression with lasso penalty for the j-th lambda       */
      out = admm_ovglasso_smo_large_m_update_1lambda(A, b, Glen, groups, group_weights,
                                                     var_weights, F,
                                                     FTF, DTD, ATA, ATA_CHOL_U, ATb,
                                                     b2, m, n, G, u, z, lambda(j), lambda2(k),
                                                     rho_adaptation, rho, tau, mu,
                                                     reltol, abstol, maxiter, ping);

      /* Retrieve output      */
      u = as<arma::colvec>(out["u"]);
      z = as<arma::colvec>(out["z"]);
      x = as<arma::colvec>(out["x"]);
      
      /* Store output      */
      niter(j, k)          = out["niter"];
      eltime(j, k)         = out["eltime"];
      conv(j, k)           = out["convergence"];
      X.tube(j, k)         = x;
      h_objval(j, k)       = as<arma::vec>(out["objval"]);
      h_r_norm(j, k)       = as<arma::vec>(out["r_norm"]);
      h_s_norm(j, k)       = as<arma::vec>(out["s_norm"]);
      h_eps_pri(j, k)      = as<arma::vec>(out["eps_pri"]);
      h_eps_dual(j, k)     = as<arma::vec>(out["eps_dual"]);
      h_rho(j, k)          = (as<arma::vec>(out["rho"])).subvec(0, niter(j, k)-1);
      rho                  = out["rho.updated"];
      residuals.tube(j, k) = as<arma::vec>(out["residuals"]);
      fitted.tube(j, k)    = as<arma::vec>(out["fitted.values"]);
      mse(j, k)            = out["mse"];
      Umat.tube(j, k)      = u;
      Zmat.tube(j, k)      = z;
    }
  }

  /* Find min MSE in-sample      */
  idx_min_mse     = mse.index_min();
  min_mse         = mse(idx_min_mse);
  idx_min_mse_row = idx_min_mse % X.n_rows;
  idx_min_mse_col = idx_min_mse / X.n_rows;
  
  /* Get output         */
  output["coef.path"]    = X;
  output["iternum"]      = niter;
  output["objfun"]       = h_objval;
  output["r_norm"]       = h_r_norm;
  output["s_norm"]       = h_s_norm;
  output["err_pri"]      = h_eps_pri;
  output["err_dual"]     = h_eps_dual;
  output["convergence"]  = conv;
  output["elapsedTime"]  = eltime;
  output["rho"]          = h_rho;
  output["lambda1"]      = lambda;
  output["lambda2"]      = lambda2;
  output["min.mse"]      = min_mse;
  output["indi.min.mse"] = idx_min_mse;
  //output["lambda.min"]   = lambda(idx_min_mse);
  output["lambda.min"]   = lambda(idx_min_mse_row);
  output["lambda2.min"]   = lambda2(idx_min_mse_col);
  output["coefficients"] = X.tube(idx_min_mse_row, idx_min_mse_col);
  output["mse"]          = mse;
  output["U"]            = Umat;
  output["Z"]            = Zmat;

  /* Return output      */
  return(output);
}

Rcpp::List admm_ovglasso_large_n_fast(const arma::mat& A,
                                      const arma::vec& b,
                                      const arma::mat& groups,
                                      const arma::vec& group_weights,
                                      const arma::vec& var_weights,
                                      const arma::vec& lambda,
                                      const bool rho_adaptation,
                                      double rho,
                                      const double tau,
                                      const double mu,
                                      const double reltol,
                                      const double abstol,
                                      const int maxiter,
                                      const int ping) {

  /* Variable delcaration           */
  double b2 = 0.0, min_mse = 0.0;
  int n1 = 0, dim_z = 0;
  uword idx_min_mse;

  /* Get dimensions             */
  const int m = A.n_rows;
  const int n = A.n_cols;
  const int G = groups.n_rows;

  /* Get number of lambda           */
  const int nlambda = lambda.n_elem;

  /* Get output lists          */
  List out, output;

  /* Variable definition            */
  if (m >= n) {
    n1 = n;
  } else {
    n1 = m;
  }

  /* Definition of vectors and matrices     */
  arma::vec ATb(n, fill::zeros);
  arma::mat ATA_CHOL_U(n1, n, fill::zeros);
  arma::mat FTF(n, n, fill::zeros);
  arma::colvec x(n, fill::zeros);
  arma::colvec Glen(G, fill::zeros);
  arma::mat FTF_INV(n, n, fill::zeros);
  arma::mat AFTF_INVA(m, m, fill::zeros);

  /* Store output    */
  arma::field<vec> h_objval(nlambda);
  arma::field<vec> h_r_norm(nlambda);
  arma::field<vec> h_s_norm(nlambda);
  arma::field<vec> h_eps_pri(nlambda);
  arma::field<vec> h_eps_dual(nlambda);
  arma::field<vec> h_rho(nlambda);
  arma::vec niter(nlambda, fill::zeros);
  arma::vec eltime(nlambda, fill::zeros);
  arma::mat X(nlambda, n, fill::zeros);
  arma::vec conv(nlambda, fill::zeros);
  arma::mat residuals(nlambda, m, fill::zeros);
  arma::mat fitted(nlambda, m, fill::zeros);
  arma::vec mse(nlambda, fill::zeros);

  /* Precompute relevant quantities         */
  ATb            = A.t() * b / static_cast<float>(m);
  ATA_CHOL_U     = chol_qr_fact(A, m, n);
  b2             = as_scalar(b.t() * b) / static_cast<float>(m);
  Glen           = sum(groups, 1);
  arma::sp_mat F = ovglasso_Gmat2Fmat_sparse(groups, var_weights);
  FTF            = diagmat(spmat2spmat_mult(F.t(), F));
  FTF_INV        = diagmat(1.0 / diagvec(FTF));
  AFTF_INVA      = A * diagmat(FTF_INV) * A.t();

  /* Initialize vectors u and z         */
  dim_z = sum(Glen);
  arma::colvec u(dim_z, fill::zeros);
  arma::colvec z(dim_z, fill::zeros);
  arma::mat Umat(nlambda, dim_z, fill::zeros);
  arma::mat Zmat(nlambda, dim_z, fill::zeros);

  /* main loop  */
  for (int j=0; j<nlambda; j++) {
    /* perform linear regression with lasso penalty for the j-th lambda       */
    out = admm_ovglasso_large_n_update_1lambda(A, b, Glen, groups,
                                               group_weights, var_weights, F,
                                               FTF_INV, AFTF_INVA,
                                               ATA_CHOL_U, ATb,
                                               b2, m, n, G,
                                               u, z, lambda(j),
                                               rho_adaptation, rho, tau, mu,
                                               reltol, abstol, maxiter, ping);

    /* Retrieve output      */
    u = as<arma::colvec>(out["u"]);
    z = as<arma::colvec>(out["z"]);
    x = as<arma::colvec>(out["x"]);

    /* Store output      */
    niter(j)         = out["niter"];
    eltime(j)        = out["eltime"];
    conv(j)          = out["convergence"];
    X.row(j)         = x.t();
    h_objval(j)      = as<arma::vec>(out["objval"]);
    h_r_norm(j)      = as<arma::vec>(out["r_norm"]);
    h_s_norm(j)      = as<arma::vec>(out["s_norm"]);
    h_eps_pri(j)     = as<arma::vec>(out["eps_pri"]);
    h_eps_dual(j)    = as<arma::vec>(out["eps_dual"]);
    h_rho(j)         = (as<arma::vec>(out["rho"])).subvec(0, niter(j)-1);
    residuals.row(j) = (as<arma::vec>(out["residuals"])).t();
    fitted.row(j)    = (as<arma::vec>(out["fitted.values"])).t();
    mse(j)           = out["mse"];
    Umat.row(j)      = u.t();
    Zmat.row(j)      = z.t();
  }

  /* Find min MSE in-sample      */
  idx_min_mse = index_min(mse);
  min_mse     = mse(idx_min_mse);

  /* Get output         */
  output["coef.path"]    = X;
  output["iternum"]      = niter;               
  output["objfun"]       = h_objval;        
  output["r_norm"]       = h_r_norm;
  output["s_norm"]       = h_s_norm;
  output["err_pri"]      = h_eps_pri;
  output["err_dual"]     = h_eps_dual;
  output["convergence"]  = conv;
  output["elapsedTime"]  = eltime;
  output["rho"]          = h_rho;
  output["lambda"]       = lambda;
  output["min.mse"]      = min_mse;
  output["indi.min.mse"] = idx_min_mse;
  output["lambda.min"]   = lambda[idx_min_mse];
  output["coefficients"] = (X.row(idx_min_mse)).t();
  output["mse"]          = mse;
  output["U"]            = Umat;
  output["Z"]            = Zmat;

  /* Return output      */
  return(output);
}

Rcpp::List admm_ovglasso_smo_large_n_fast(const arma::mat& A,
                                          const arma::vec& b,
                                          const arma::mat& groups,
                                          const arma::vec& group_weights,
                                          const arma::vec& var_weights,
                                          const arma::vec& lambda,
                                          const arma::vec& lambda2,
                                          const int diff_order,
                                          const bool rho_adaptation,
                                          double rho,
                                          const double tau,
                                          const double mu,
                                          const double reltol,
                                          const double abstol,
                                          const int maxiter,
                                          const int ping) {

  /* Variable delcaration           */
  double b2 = 0.0, min_mse = 0.0;
  int n1 = 0, dim_z = 0;
  uword idx_min_mse, idx_min_mse_row, idx_min_mse_col;
  
  /* Get dimensions             */
  const int m = A.n_rows;
  const int n = A.n_cols;
  const int G = groups.n_rows;

  /* Get number of lambda           */
  const int nlambda  = lambda.n_elem;
  const int nlambda2 = lambda2.n_elem;

  /* Get output lists          */
  List out, output;

  /* Variable definition            */
  if (m >= n) {
    n1 = n;
  } else {
    n1 = m;
  }

  /* Definition of vectors and matrices     */
  arma::vec ATb(n, fill::zeros);
  arma::mat ATA_CHOL_U(n1, n, fill::zeros);
  arma::mat FTF(n, n, fill::zeros);
  arma::mat FTF_INV(n, n, fill::zeros);
  arma::mat D(n-diff_order, n, fill::zeros);
  arma::mat DTD(n, n, fill::zeros);
  arma::colvec x(n, fill::zeros);
  arma::colvec Glen(G, fill::zeros);
  arma::mat Atilde(n+m, n, fill::zeros);
  arma::mat AtFTF_INVAt(n+m, n+m, fill::zeros);

  /* Store output    */
  arma::field<vec> h_objval(nlambda, nlambda2);
  arma::field<vec> h_r_norm(nlambda, nlambda2);
  arma::field<vec> h_s_norm(nlambda, nlambda2);
  arma::field<vec> h_eps_pri(nlambda, nlambda2);
  arma::field<vec> h_eps_dual(nlambda, nlambda2);
  arma::field<vec> h_rho(nlambda, nlambda2);
  arma::mat niter(nlambda, nlambda2, fill::zeros);
  arma::mat eltime(nlambda, nlambda2, fill::zeros);
  arma::mat conv(nlambda, nlambda2, fill::zeros);
  arma::cube X(nlambda, nlambda2, n, fill::zeros);
  arma::cube residuals(nlambda, nlambda2, m, fill::zeros);
  arma::cube fitted(nlambda, nlambda2, m, fill::zeros);
  arma::mat mse(nlambda, nlambda2, fill::zeros);

  /* Precompute relevant quantities         */
  ATb            = A.t() * b / static_cast<float>(m);
  ATA_CHOL_U     = chol_qr_fact(A, m, n);
  b2             = as_scalar(b.t() * b) / static_cast<float>(m);
  Glen           = sum(groups, 1);
  arma::sp_mat F = ovglasso_Gmat2Fmat_sparse(groups, var_weights);
  FTF            = diagmat(spmat2spmat_mult(F.t(), F));
  FTF_INV        = diagmat(1.0 / diagvec(FTF));
  D              =  forward_diff_difference_matrix(n, 1, diff_order);
  DTD            = D.t() * D;
  
  /* Initialize vectors u and z         */
  dim_z = sum(Glen);
  arma::colvec u(dim_z, fill::zeros);
  arma::colvec z(dim_z, fill::zeros);
  arma::cube Umat(nlambda, nlambda2, dim_z, fill::zeros);
  arma::cube Zmat(nlambda, nlambda2, dim_z, fill::zeros);

  /* main loop  */
  for (int j=0; j<nlambda; j++) {
    for (int k=0; k<nlambda2; k++) {
      
      /* compute relevant quantities: these quantities change with lambda2    */
      Atilde.head_rows(m) = A / sqrt(static_cast<float>(m));
      Atilde.tail_rows(n) = std::sqrt(lambda2(k)) * D;
      AtFTF_INVAt         = Atilde * FTF_INV * Atilde.t();
      
      /* perform linear regression with lasso penalty for the j-th lambda       */
      out = admm_ovglasso_smo_large_n_update_1lambda(A, b, Glen, groups, group_weights,
                                                     var_weights, F,
                                                     FTF_INV, Atilde, AtFTF_INVAt,
                                                     ATA_CHOL_U, ATb,
                                                     b2, m, n, G,
                                                     u, z, lambda(j), lambda2(k),
                                                     rho_adaptation, rho, tau, mu,
                                                     reltol, abstol, maxiter, ping);
      
      /* Retrieve output      */
      u = as<arma::colvec>(out["u"]);
      z = as<arma::colvec>(out["z"]);
      x = as<arma::colvec>(out["x"]);

      /* Store output      */
      niter(j, k)          = out["niter"];
      eltime(j, k)         = out["eltime"];
      conv(j, k)           = out["convergence"];
      X.tube(j, k)         = x;
      h_objval(j, k)       = as<arma::vec>(out["objval"]);
      h_r_norm(j, k)       = as<arma::vec>(out["r_norm"]);
      h_s_norm(j, k)       = as<arma::vec>(out["s_norm"]);
      h_eps_pri(j, k)      = as<arma::vec>(out["eps_pri"]);
      h_eps_dual(j, k)     = as<arma::vec>(out["eps_dual"]);
      h_rho(j, k)          = (as<arma::vec>(out["rho"])).subvec(0, niter(j, k)-1);
      rho                  = out["rho.updated"];
      residuals.tube(j, k) = as<arma::vec>(out["residuals"]);
      fitted.tube(j, k)    = as<arma::vec>(out["fitted.values"]);
      mse(j, k)            = out["mse"];
      Umat.tube(j, k)      = u;
      Zmat.tube(j, k)      = z;
    }
  }

  /* Find min MSE in-sample      */
  idx_min_mse     = mse.index_min();
  min_mse         = mse(idx_min_mse);
  idx_min_mse_row = idx_min_mse % X.n_rows;
  idx_min_mse_col = idx_min_mse / X.n_rows;

  /* Get output         */
  output["coef.path"]    = X;
  output["iternum"]      = niter;
  output["objfun"]       = h_objval;
  output["r_norm"]       = h_r_norm;
  output["s_norm"]       = h_s_norm;
  output["err_pri"]      = h_eps_pri;
  output["err_dual"]     = h_eps_dual;
  output["convergence"]  = conv;
  output["elapsedTime"]  = eltime;
  output["rho"]          = h_rho;
  output["lambda"]       = lambda;
  output["min.mse"]      = min_mse;
  output["indi.min.mse"] = idx_min_mse;
  output["lambda.min"]   = lambda(idx_min_mse_row);
  output["lambda2.min"]  = lambda2(idx_min_mse_col);
  output["coefficients"] = X.tube(idx_min_mse_row, idx_min_mse_col);
  output["mse"]          = mse;
  output["U"]            = Umat;
  output["Z"]            = Zmat;
  
  /* Return output      */
  return(output);
}

Rcpp::List admm_ovglasso_cov_large_m_fast(const arma::mat& W,
                                          const arma::mat& Z,
                                          const arma::vec& y,
                                          const arma::mat& groups,
                                          const arma::vec& group_weights,
                                          const arma::vec& var_weights,
                                          const arma::vec lambda,
                                          const bool rho_adaptation,
                                          double rho,
                                          const double tau,
                                          const double mu,
                                          const double reltol,
                                          const double abstol,
                                          const int maxiter,
                                          const int ping) {

  /* Variable delcaration           */
  double y2 = 0.0, min_mse = 0.0, sqrtm = 0.0;
  int n1 = 0, dim_z = 0;
  uword idx_min_mse;

  /* Get dimensions             */
  const int m       = W.n_rows;
  const int n_spcov = W.n_cols;
  const int n_cov   = Z.n_cols;
  const int G       = groups.n_rows;
  
  /* Variable definition            */
  sqrtm = std::sqrt(static_cast<float>(m));

  /* Get number of lambda           */
  const int nlambda = lambda.n_elem;

  /* Get output lists          */
  List out, output;

  /* Variable definition            */
  if (m >= n_spcov) {
    n1 = n_spcov;
  } else {
    n1 = m;
  }

  /* Definition of vectors and matrices     */
  arma::colvec x(n_spcov, fill::zeros);
  arma::colvec v(n_cov, fill::zeros);
  arma::mat W_(m, n_spcov, fill::zeros);
  arma::mat Z_(m, n_cov, fill::zeros);
  arma::mat WW_CHOL_U(n1, n_spcov, fill::zeros);
  arma::mat ZZ_CHOL_U(n_cov, n_cov, fill::zeros);
  arma::vec Wy(n_spcov, fill::zeros);
  arma::vec ones_m(m, fill::ones);
  arma::mat eye_m(m, m, fill::eye);
  arma::mat ZZ(n_cov, n_cov, fill::zeros);
  arma::mat L_ZZ(n_cov, n_cov, fill::zeros);
  arma::mat L_ZZ_INV(n_cov, n_cov, fill::zeros);
  arma::mat ZZ_INV(n_cov, n_cov, fill::zeros);
  arma::mat P_ZZ(m, m, fill::zeros);
  arma::mat M_ZZ(m, m, fill::zeros);
  arma::mat WMW(n_spcov, n_spcov, fill::zeros);
  arma::vec WMy(n_spcov, fill::zeros);
  arma::vec Zy(n_cov, fill::zeros);
  arma::vec v_LS(n_cov, fill::zeros);
  arma::mat ZW(n_cov, n_spcov, fill::zeros);
  arma::mat P_ZW(n_cov, n_spcov, fill::zeros);
  arma::mat FTF(n_spcov, n_spcov, fill::zeros);
  arma::colvec Glen(G, fill::zeros);
  arma::colvec fitted_(m, fill::zeros);
  arma::colvec residuals_(m, fill::zeros);
  arma::vec mse(nlambda, fill::zeros);

  /* Store output    */
  arma::field<vec> h_objval(nlambda);
  arma::field<vec> h_r_norm(nlambda);
  arma::field<vec> h_s_norm(nlambda);
  arma::field<vec> h_eps_pri(nlambda);
  arma::field<vec> h_eps_dual(nlambda);
  arma::field<vec> h_rho(nlambda);
  arma::vec niter(nlambda, fill::zeros);
  arma::vec eltime(nlambda, fill::zeros);
  arma::mat X(nlambda, n_spcov, fill::zeros);
  arma::mat V(nlambda, n_cov, fill::zeros);
  arma::vec conv(nlambda, fill::zeros);
  arma::mat residuals(nlambda, m, fill::zeros);
  arma::mat fitted(nlambda, m, fill::zeros);
  
  /* Precompute relevant quantities         */
  Z_             = Z.each_col() % (ones_m * (1.0 / sqrtm));             // 19Nov25: la divisione per n é corretta
  W_             = W.each_col() % (ones_m * (1.0 / sqrtm));
  ZZ             = Z_.t() * Z_;
  L_ZZ           = (chol(ZZ)).t();
  L_ZZ_INV       = inv(trimatl(L_ZZ));
  ZZ_INV         = L_ZZ_INV.t() * L_ZZ_INV;
  P_ZZ           = Z_ * ZZ_INV * Z_.t();
  M_ZZ           = eye_m - P_ZZ;
  WMW            = W_.t() * M_ZZ * W_;
  WMy            = W_.t() * M_ZZ * y / static_cast<float>(sqrtm);
  Zy             = Z_.t() * y / static_cast<float>(sqrtm);
  v_LS           = L_ZZ_INV.t() * L_ZZ_INV * Zy;
  ZW             = Z_.t() * W_;
  P_ZW           = L_ZZ_INV.t() * L_ZZ_INV * ZW;
  WW_CHOL_U      = chol_qr_fact(W_, m, n_spcov);
  ZZ_CHOL_U      = chol_qr_fact(Z_, m, n_cov);
  y2             = as_scalar(y.t() * y) / static_cast<float>(m);
  Wy             = W_.t() * y / static_cast<float>(sqrtm);
  Glen           = sum(groups, 1);
  arma::sp_mat F = ovglasso_Gmat2Fmat_sparse(groups, var_weights);
  FTF            = F.t() * F;
  
  /* Initialize vectors u and z         */
  dim_z = sum(Glen);
  arma::colvec u(dim_z, fill::zeros);
  arma::colvec z(dim_z, fill::zeros);
  arma::mat Umat(nlambda, dim_z, fill::zeros);
  arma::mat Zmat(nlambda, dim_z, fill::zeros);

  /* main loop  */
  for (int j=0; j<nlambda; j++) {
    /* perform linear regression with lasso penalty for the j-th lambda       */
    out = admm_ovglasso_cov_large_m_update_1lambda(W_, Z_, y, Glen, groups, group_weights,
                                                   var_weights, G, F, FTF,
                                                   WMW, WMy, v_LS, P_ZW, WW_CHOL_U, Wy,
                                                   y2, ZZ_CHOL_U, Zy, ZW,
                                                   u, z, lambda(j), rho_adaptation,
                                                   rho, tau, mu, reltol, abstol, maxiter, ping);

    /* Retrieve output      */
    u = as<arma::colvec>(out["u"]);
    z = as<arma::colvec>(out["z"]);
    x = as<arma::colvec>(out["x"]);
    v = as<arma::colvec>(out["v"]);
    
    /* Retrieve output      */
    fitted_    = W * x + Z * v;
    residuals_ = y - fitted_;
    
    /* Store output      */
    niter(j)         = out["niter"];
    eltime(j)        = out["eltime"];
    conv(j)          = out["convergence"];
    X.row(j)         = x.t();
    V.row(j)         = v.t();
    h_objval(j)      = as<arma::vec>(out["objval"]);
    h_r_norm(j)      = as<arma::vec>(out["r_norm"]);
    h_s_norm(j)      = as<arma::vec>(out["s_norm"]);
    h_eps_pri(j)     = as<arma::vec>(out["eps_pri"]);
    h_eps_dual(j)    = as<arma::vec>(out["eps_dual"]);
    h_rho(j)         = (as<arma::vec>(out["rho"])).subvec(0, niter(j)-1);
    rho              = out["rho.updated"];
    //residuals.row(j) = (as<arma::vec>(out["residuals"])).t();
    //fitted.row(j)    = (as<arma::vec>(out["fitted.values"])).t();
    //mse(j)           = out["mse"];
    residuals.row(j) = fitted_.t();
    fitted.row(j)    = residuals_.t();
    mse(j)           = as_scalar(residuals_.t() * residuals_) / (static_cast<float>(m));
    Umat.row(j)      = u.t();
    Zmat.row(j)      = z.t();
  }

  /* Find min MSE in-sample      */
  idx_min_mse = index_min(mse);
  min_mse     = mse(idx_min_mse);

  /* Get output         */
  output["sp.coef.path"]    = X;
  output["coef.path"]       = V;
  output["iternum"]         = niter;               
  output["objfun"]          = h_objval;        
  output["r_norm"]          = h_r_norm;
  output["s_norm"]          = h_s_norm;
  output["err_pri"]         = h_eps_pri;
  output["err_dual"]        = h_eps_dual;
  output["convergence"]     = conv;
  output["elapsedTime"]     = eltime;
  output["rho"]             = h_rho;
  output["lambda"]          = lambda;
  output["min.mse"]         = min_mse;
  output["indi.min.mse"]    = idx_min_mse;
  output["lambda.min"]      = lambda[idx_min_mse];
  output["sp.coefficients"] = (X.row(idx_min_mse)).t();
  output["coefficients"]    = (V.row(idx_min_mse)).t();
  output["mse"]             = mse;
  output["U"]               = Umat;
  output["Z"]               = Zmat;

  /* Return output      */
  return(output);
}

Rcpp::List admm_ovglasso_cov_smo_large_m_fast(const arma::mat& W,
                                              const arma::mat& Z,
                                              const arma::vec& y,
                                              const arma::mat& groups,
                                              const arma::vec& group_weights,
                                              const arma::vec& var_weights,
                                              const arma::vec& lambda,
                                              const arma::vec& lambda2,
                                              const int diff_order,
                                              bool rho_adaptation,
                                              double rho,
                                              const double tau,
                                              const double mu,
                                              const double reltol,
                                              const double abstol,
                                              const int maxiter,
                                              const int ping) {

  /* Variable delcaration           */
  double y2 = 0.0, min_mse = 0.0, sqrtm = 0.0;
  int n1 = 0, dim_z = 0;
  uword idx_min_mse, idx_min_mse_row, idx_min_mse_col;

  /* Get dimensions             */
  const int m       = W.n_rows;
  const int n_spcov = W.n_cols;
  const int n_cov   = Z.n_cols;
  const int G       = groups.n_rows;
  
  /* Variable definition            */
  sqrtm = std::sqrt(static_cast<float>(m));

  /* Get number of lambda           */
  const int nlambda  = lambda.n_elem;
  const int nlambda2 = lambda2.n_elem;

  /* Get output lists          */
  List out, output;

  /* Variable definition            */
  if (m >= n_spcov) {
    n1 = n_spcov;
  } else {
    n1 = m;
  }

  /* Definition of vectors and matrices     */
  arma::colvec x(n_spcov, fill::zeros);
  arma::colvec v(n_cov, fill::zeros);
  arma::mat W_(m, n_spcov, fill::zeros);
  arma::mat Z_(m, n_cov, fill::zeros);
  arma::mat WW_CHOL_U(n1, n_spcov, fill::zeros);
  arma::mat ZZ_CHOL_U(n_cov, n_cov, fill::zeros);
  arma::vec Wy(n_spcov, fill::zeros);
  arma::vec ones_m(m, fill::ones);
  arma::mat eye_m(m, m, fill::eye);
  arma::mat ZZ(n_cov, n_cov, fill::zeros);
  arma::mat L_ZZ(n_cov, n_cov, fill::zeros);
  arma::mat L_ZZ_INV(n_cov, n_cov, fill::zeros);
  arma::mat ZZ_INV(n_cov, n_cov, fill::zeros);
  arma::mat P_ZZ(m, m, fill::zeros);
  arma::mat M_ZZ(m, m, fill::zeros);
  arma::mat WMW(n_spcov, n_spcov, fill::zeros);
  arma::vec WMy(n_spcov, fill::zeros);
  arma::vec Zy(n_cov, fill::zeros);
  arma::vec v_LS(n_cov, fill::zeros);
  arma::mat ZW(n_cov, n_spcov, fill::zeros);
  arma::mat P_ZW(n_cov, n_spcov, fill::zeros);
  arma::mat FTF(n_spcov, n_spcov, fill::zeros);
  arma::mat DTD(n_spcov, n_spcov, fill::zeros);
  arma::colvec Glen(G, fill::zeros);
  arma::colvec fitted_(m, fill::zeros);
  arma::colvec residuals_(m, fill::zeros);
  //arma::vec mse(nlambda, fill::zeros);

  /* Store output    */
  arma::field<vec> h_objval(nlambda, nlambda2);
  arma::field<vec> h_r_norm(nlambda, nlambda2);
  arma::field<vec> h_s_norm(nlambda, nlambda2);
  arma::field<vec> h_eps_pri(nlambda, nlambda2);
  arma::field<vec> h_eps_dual(nlambda, nlambda2);
  arma::field<vec> h_rho(nlambda, nlambda2);
  arma::mat niter(nlambda, nlambda2, fill::zeros);
  arma::mat eltime(nlambda, nlambda2, fill::zeros);
  arma::mat conv(nlambda, nlambda2, fill::zeros);
  arma::cube residuals(nlambda, nlambda2, m, fill::zeros);
  arma::cube fitted(nlambda, nlambda2, m, fill::zeros);
  arma::mat mse(nlambda, nlambda2, fill::zeros);
  arma::cube X(nlambda, nlambda2, n_spcov, fill::zeros);
  arma::cube V(nlambda, nlambda2, n_cov, fill::zeros);

  /* Precompute relevant quantities         */
  Z_             = Z.each_col() % (ones_m * (1.0 / sqrtm));             // 19Nov25: la divisione per n é corretta
  W_             = W.each_col() % (ones_m * (1.0 / sqrtm));
  ZZ             = Z_.t() * Z_;
  L_ZZ           = (chol(ZZ)).t();
  L_ZZ_INV       = inv(trimatl(L_ZZ));
  ZZ_INV         = L_ZZ_INV.t() * L_ZZ_INV;
  P_ZZ           = Z_ * ZZ_INV * Z_.t();
  M_ZZ           = eye_m - P_ZZ;
  WMW            = W_.t() * M_ZZ * W_;
  WMy            = W_.t() * M_ZZ * y / static_cast<float>(sqrtm);
  Zy             = Z_.t() * y / static_cast<float>(sqrtm);
  v_LS           = L_ZZ_INV.t() * L_ZZ_INV * Zy;
  ZW             = Z_.t() * W_;
  P_ZW           = L_ZZ_INV.t() * L_ZZ_INV * ZW;
  WW_CHOL_U      = chol_qr_fact(W_, m, n_spcov);
  ZZ_CHOL_U      = chol_qr_fact(Z_, m, n_cov);
  y2             = as_scalar(y.t() * y) / static_cast<float>(m);
  Wy             = W_.t() * y / static_cast<float>(sqrtm);
  Glen           = sum(groups, 1);
  arma::sp_mat F = ovglasso_Gmat2Fmat_sparse(groups, var_weights);
  FTF            = F.t() * F;
  DTD            = forward_diff_penalty_matrix(n_spcov, 1, diff_order);
  
  /* Initialize vectors u and z         */
  dim_z = sum(Glen);
  arma::colvec u(dim_z, fill::zeros);
  arma::colvec z(dim_z, fill::zeros);
  arma::cube Umat(nlambda, nlambda2, dim_z, fill::zeros);
  arma::cube Zmat(nlambda, nlambda2, dim_z, fill::zeros);
  
  /* main loop  */
  for (int j=0; j<nlambda; j++) {
    for (int k = 0; k<nlambda2; k++) {
      /* perform linear regression with lasso penalty for the j-th lambda       */
      out = admm_ovglasso_cov_smo_large_m_update_1lambda(W_, Z_, y, Glen, groups, group_weights, 
                                                         var_weights, G, F, FTF, DTD,
                                                         WMW, WMy, v_LS, P_ZW, WW_CHOL_U, Wy,
                                                         y2, ZZ_CHOL_U, Zy, ZW,
                                                         u, z, lambda(j), lambda2(k), rho_adaptation,
                                                         rho, tau, mu, reltol, abstol, maxiter, ping);
      
      /* Retrieve output      */
      u = as<arma::colvec>(out["u"]);
      z = as<arma::colvec>(out["z"]);
      x = as<arma::colvec>(out["x"]);
      v = as<arma::colvec>(out["v"]);
      
      /* Retrieve output      */
      fitted_    = W * x + Z * v;
      residuals_ = y - fitted_;
      
      /* Store output      */
      niter(j, k)          = out["niter"];
      eltime(j, k)         = out["eltime"];
      conv(j, k)           = out["convergence"];
      X.tube(j, k)         = x;
      V.tube(j, k)         = v;
      h_objval(j, k)       = as<arma::vec>(out["objval"]);
      h_r_norm(j, k)       = as<arma::vec>(out["r_norm"]);
      h_s_norm(j, k)       = as<arma::vec>(out["s_norm"]);
      h_eps_pri(j, k)      = as<arma::vec>(out["eps_pri"]);
      h_eps_dual(j, k)     = as<arma::vec>(out["eps_dual"]);
      h_rho(j, k)          = (as<arma::vec>(out["rho"])).subvec(0, niter(j, k)-1);
      rho                  = out["rho.updated"];
      //residuals.tube(j, k) = as<arma::vec>(out["residuals"]);
      //fitted.tube(j, k)    = as<arma::vec>(out["fitted.values"]);
      //mse(j, k)            = out["mse"];
      residuals.tube(j, k) = fitted_;
      fitted.tube(j, k)    = residuals_;
      mse(j, k)            = as_scalar(residuals_.t() * residuals_) / (static_cast<float>(m));
      Umat.tube(j, k)      = u;
      Zmat.tube(j, k)      = z;
    }
  }

  /* Find min MSE in-sample      */
  idx_min_mse     = mse.index_min();
  min_mse         = mse(idx_min_mse);
  idx_min_mse_row = idx_min_mse % X.n_rows;
  idx_min_mse_col = idx_min_mse / X.n_rows;

  /* Get output         */
  output["sp.coef.path"]    = X;
  output["coef.path"]       = V;
  output["iternum"]         = niter;               
  output["objfun"]          = h_objval;        
  output["r_norm"]          = h_r_norm;
  output["s_norm"]          = h_s_norm;
  output["err_pri"]         = h_eps_pri;
  output["err_dual"]        = h_eps_dual;
  output["convergence"]     = conv;
  output["elapsedTime"]     = eltime;
  output["rho"]             = h_rho;
  output["lambda"]          = lambda;
  output["min.mse"]         = min_mse;
  output["indi.min.mse"]    = idx_min_mse;
  output["lambda.min"]      = lambda(idx_min_mse_row);
  output["lambda2.min"]     = lambda2(idx_min_mse_col);
  output["sp.coefficients"] = X.tube(idx_min_mse_row, idx_min_mse_col);
  output["coefficients"]    = V.tube(idx_min_mse_row, idx_min_mse_col);
  output["mse"]             = mse;
  output["U"]               = Umat;
  output["Z"]               = Zmat;

  /* Return output      */
  return(output);
}

Rcpp::List admm_ovglasso_cov_large_n_fast(const arma::mat& W,
                                          const arma::mat& Z,
                                          const arma::vec& y,
                                          const arma::mat& groups,
                                          const arma::vec& group_weights,
                                          const arma::vec& var_weights,
                                          const arma::vec& lambda,
                                          const bool rho_adaptation,
                                          double rho,
                                          const double tau,
                                          const double mu,
                                          const double reltol,
                                          const double abstol,
                                          const int maxiter,
                                          const int ping) {

  /* Variable delcaration           */
  double y2 = 0.0, min_mse = 0.0, sqrtm = 0.0;
  int n1 = 0, dim_z = 0;
  uword idx_min_mse;

  /* Get dimensions             */
  const int m       = W.n_rows;
  const int n_spcov = W.n_cols;
  const int n_cov   = Z.n_cols;
  const int G       = groups.n_rows;
  
  /* Variable definition            */
  sqrtm = std::sqrt(static_cast<float>(m));

  /* Get number of lambda           */
  const int nlambda = lambda.n_elem;

  /* Get output lists          */
  List out, output;

  /* Variable definition            */
  if (m >= n_spcov) {
    n1 = n_spcov;
  } else {
    n1 = m;
  }

  /* Definition of vectors and matrices     */
  arma::colvec x(n_spcov, fill::zeros);
  arma::colvec v(n_cov, fill::zeros);
  arma::mat Z_(m, n_cov, fill::zeros);
  arma::mat WW_CHOL_U(n1, n_spcov, fill::zeros);
  arma::mat ZZ_CHOL_U(n_cov, n_cov, fill::zeros);
  arma::vec Wy(n_spcov, fill::zeros);
  arma::vec ones_m(m, fill::ones);
  arma::mat eye_m(m, m, fill::eye);
  arma::mat ZZ(n_cov, n_cov, fill::zeros);
  arma::mat L_ZZ(n_cov, n_cov, fill::zeros);
  arma::mat L_ZZ_INV(n_cov, n_cov, fill::zeros);
  arma::mat ZZ_INV(n_cov, n_cov, fill::zeros);
  arma::mat P_ZZ(m, m, fill::zeros);
  arma::mat M_ZZ(m, m, fill::zeros);
  arma::mat WMW(n_spcov, n_spcov, fill::zeros);
  arma::vec WMy(n_spcov, fill::zeros);
  arma::vec Zy(n_cov, fill::zeros);
  arma::vec v_LS(n_cov, fill::zeros);
  arma::mat ZW(n_cov, n_spcov, fill::zeros);
  arma::mat P_ZW(n_cov, n_spcov, fill::zeros);
  //arma::mat WW(m, m, fill::zeros);
  arma::mat M_ZZW(m, n_spcov, fill::zeros);
  arma::mat M_ZZW2(m, m, fill::zeros);
  arma::mat FTF(n_spcov, n_spcov, fill::zeros);
  arma::mat FTF_INV(n_spcov, n_spcov, fill::zeros);
  arma::colvec Glen(G, fill::zeros);
  arma::colvec fitted_(m, fill::zeros);
  arma::colvec residuals_(m, fill::zeros);
  arma::vec mse(nlambda, fill::zeros);
  
  /* Store output    */
  arma::field<vec> h_objval(nlambda);
  arma::field<vec> h_r_norm(nlambda);
  arma::field<vec> h_s_norm(nlambda);
  arma::field<vec> h_eps_pri(nlambda);
  arma::field<vec> h_eps_dual(nlambda);
  arma::field<vec> h_rho(nlambda);
  arma::vec niter(nlambda, fill::zeros);
  arma::vec eltime(nlambda, fill::zeros);
  arma::mat X(nlambda, n_spcov, fill::zeros);
  arma::mat V(nlambda, n_cov, fill::zeros);
  arma::vec conv(nlambda, fill::zeros);
  arma::mat residuals(nlambda, m, fill::zeros);
  arma::mat fitted(nlambda, m, fill::zeros);
  
  /* Precompute relevant quantities         */
  Z_        = Z.each_col() % (ones_m * (1.0 / sqrtm));
  //W         = W.each_col() % (ones_m * (1.0 / sqrtm));            // è corretto non dividere W per la numerosità campionaria m
  ZZ        = Z_.t() * Z_;
  L_ZZ      = (chol(ZZ)).t();
  L_ZZ_INV  = inv(trimatl(L_ZZ));
  ZZ_INV    = L_ZZ_INV.t() * L_ZZ_INV;
  P_ZZ      = Z_ * ZZ_INV * Z_.t();
  M_ZZ      = eye_m - P_ZZ;
  WMW       = W.t() * M_ZZ * W;                                   // 19Nov25: la divisione per n è corretta così come è ora
  WMy       = W.t() * M_ZZ * y;
  Zy        = Z_.t() * y / static_cast<float>(sqrtm);
  v_LS      = L_ZZ_INV.t() * L_ZZ_INV * Zy;
  ZW        = Z_.t() * W / static_cast<float>(sqrtm);
  P_ZW      = L_ZZ_INV.t() * L_ZZ_INV * ZW;
  WW_CHOL_U = chol_qr_fact(W, m, n_spcov);
  ZZ_CHOL_U = chol_qr_fact(Z_, m, n_cov);
  y2        = as_scalar(y.t() * y) / static_cast<float>(m);
  Wy        = W.t() * y;
  M_ZZW     = M_ZZ * W;                                           // 19Nov25: la divisione per n è corretta così come è ora
  
  Glen           = sum(groups, 1);
  arma::sp_mat F = ovglasso_Gmat2Fmat_sparse(groups, var_weights);
  FTF            = diagmat(spmat2spmat_mult(F.t(), F));
  FTF_INV        = diagmat(1.0 / diagvec(FTF));
  M_ZZW2         = M_ZZW * diagmat(FTF_INV) * M_ZZW.t();
  
  /* Initialize vectors u and z         */
  dim_z = sum(Glen);
  arma::colvec u(dim_z, fill::zeros);
  arma::colvec z(dim_z, fill::zeros);
  arma::mat Umat(nlambda, dim_z, fill::zeros);
  arma::mat Zmat(nlambda, dim_z, fill::zeros);
  
  /* main loop  */
  for (int j=0; j<nlambda; j++) {
    /* perform linear regression with lasso penalty for the j-th lambda       */
    out = admm_ovglasso_cov_large_n_update_1lambda(W, Z_, y, Glen, groups,
                                                   group_weights, var_weights,
                                                   G, F, FTF_INV, M_ZZW, M_ZZW2,
                                                   WMW, WMy, v_LS, P_ZW, WW_CHOL_U, Wy, y2,
                                                   ZZ_CHOL_U, Zy, ZW, u, z, lambda(j),
                                                   rho_adaptation, rho, tau, mu,
                                                   reltol, abstol, maxiter, ping);

    /* Retrieve output: warm start      */
    u = as<arma::colvec>(out["u"]);
    z = as<arma::colvec>(out["z"]);
    x = as<arma::colvec>(out["x"]);
    v = as<arma::colvec>(out["v"]);
    
    /* Retrieve output      */
    fitted_    = W * x + Z * v;
    residuals_ = y - fitted_;
    
    /* Store output      */
    niter(j)         = out["niter"];
    eltime(j)        = out["eltime"];
    conv(j)          = out["convergence"];
    X.row(j)         = x.t();
    V.row(j)         = v.t();
    h_objval(j)      = as<arma::vec>(out["objval"]);
    h_r_norm(j)      = as<arma::vec>(out["r_norm"]);
    h_s_norm(j)      = as<arma::vec>(out["s_norm"]);
    h_eps_pri(j)     = as<arma::vec>(out["eps_pri"]);
    h_eps_dual(j)    = as<arma::vec>(out["eps_dual"]);
    h_rho(j)         = (as<arma::vec>(out["rho"])).subvec(0, niter(j)-1);
    rho              = out["rho.updated"];
    //residuals.row(j) = (as<arma::vec>(out["residuals"])).t();
    //fitted.row(j)    = (as<arma::vec>(out["fitted.values"])).t();
    //mse(j)           = out["mse"];
    residuals.row(j) = fitted_.t();
    fitted.row(j)    = residuals_.t();
    mse(j)           = as_scalar(residuals_.t() * residuals_) / (static_cast<float>(m));
    Umat.row(j)      = u.t();
    Zmat.row(j)      = z.t();
  }

  /* Find min MSE in-sample      */
  idx_min_mse = index_min(mse);
  min_mse     = mse(idx_min_mse);

  /* Get output         */
  output["sp.coef.path"]    = X;
  output["coef.path"]       = V;
  output["iternum"]         = niter;
  output["objfun"]          = h_objval;
  output["r_norm"]          = h_r_norm;
  output["s_norm"]          = h_s_norm;
  output["err_pri"]         = h_eps_pri;
  output["err_dual"]        = h_eps_dual;
  output["convergence"]     = conv;
  output["elapsedTime"]     = eltime;
  output["rho"]             = h_rho;
  output["lambda"]          = lambda;
  output["min.mse"]         = min_mse;
  output["indi.min.mse"]    = idx_min_mse;
  output["lambda.min"]      = lambda[idx_min_mse];
  output["sp.coefficients"] = (X.row(idx_min_mse)).t();
  output["coefficients"]    = (V.row(idx_min_mse)).t();
  output["mse"]             = mse;
  output["U"]               = Umat;
  output["Z"]               = Zmat;
  
  /* Return output      */
  return(output);
}

Rcpp::List admm_ovglasso_cov_smo_large_n_fast(const arma::mat& W,
                                              const arma::mat& Z,
                                              const arma::vec& y,
                                              const arma::mat& groups,
                                              const arma::vec& group_weights,
                                              const arma::vec& var_weights,
                                              const arma::vec& lambda,
                                              const arma::vec& lambda2,
                                              const int diff_order,
                                              const bool rho_adaptation,
                                              double rho,
                                              const double tau,
                                              const double mu,
                                              const double reltol,
                                              const double abstol,
                                              const int maxiter,
                                              const int ping) {

  /* Variable delcaration           */
  double y2 = 0.0, min_mse = 0.0, sqrtm = 0.0;
  int n1 = 0, dim_z = 0;
  uword idx_min_mse, idx_min_mse_row, idx_min_mse_col;

  /* Get dimensions             */
  const int m       = W.n_rows;
  const int n_spcov = W.n_cols;
  const int n_cov   = Z.n_cols;
  const int G       = groups.n_rows;
  
  /* Variable definition            */
  sqrtm = std::sqrt(static_cast<float>(m));

  /* Get number of lambda           */
  const int nlambda  = lambda.n_elem;
  const int nlambda2 = lambda2.n_elem;

  /* Get output lists          */
  List out, output;

  /* Variable definition            */
  if (m >= n_spcov) {
    n1 = n_spcov;
  } else {
    n1 = m;
  }

  /* Definition of vectors and matrices     */
  arma::colvec x(n_spcov, fill::zeros);
  arma::colvec v(n_cov, fill::zeros);
  arma::mat Z_(m, n_cov, fill::zeros);
  arma::mat WW_CHOL_U(n1, n_spcov, fill::zeros);
  arma::mat ZZ_CHOL_U(n_cov, n_cov, fill::zeros);
  arma::vec Wy(n_spcov, fill::zeros);
  arma::vec ones_m(m, fill::ones);
  arma::mat eye_m(m, m, fill::eye);
  arma::mat ZZ(n_cov, n_cov, fill::zeros);
  arma::mat L_ZZ(n_cov, n_cov, fill::zeros);
  arma::mat L_ZZ_INV(n_cov, n_cov, fill::zeros);
  arma::mat ZZ_INV(n_cov, n_cov, fill::zeros);
  arma::mat P_ZZ(m, m, fill::zeros);
  arma::mat M_ZZ(m, m, fill::zeros);
  arma::mat WMW(n_spcov, n_spcov, fill::zeros);
  arma::vec WMy(n_spcov, fill::zeros);
  arma::vec Zy(n_cov, fill::zeros);
  arma::vec v_LS(n_cov, fill::zeros);
  arma::mat ZW(n_cov, n_spcov, fill::zeros);
  arma::mat P_ZW(n_cov, n_spcov, fill::zeros);
  //arma::mat WW(m, m, fill::zeros);
  arma::mat M_ZZW(m, n_spcov, fill::zeros);
  arma::mat FTF(n_spcov, n_spcov, fill::zeros);
  arma::mat FTF_INV(n_spcov, n_spcov, fill::zeros);
  arma::colvec Glen(G, fill::zeros);
  arma::mat D(n_spcov-diff_order, n_spcov, fill::zeros);
  arma::mat DTD(n_spcov, n_spcov, fill::zeros);
  arma::mat Atilde(n_spcov+m, n_spcov, fill::zeros);
  arma::mat AtFTF_INVAt(n_spcov+m, n_spcov+m, fill::zeros);
  
  /* Store output    */
  arma::field<vec> h_objval(nlambda, nlambda2);
  arma::field<vec> h_r_norm(nlambda, nlambda2);
  arma::field<vec> h_s_norm(nlambda, nlambda2);
  arma::field<vec> h_eps_pri(nlambda, nlambda2);
  arma::field<vec> h_eps_dual(nlambda, nlambda2);
  arma::field<vec> h_rho(nlambda, nlambda2);
  arma::mat niter(nlambda, nlambda2, fill::zeros);
  arma::mat eltime(nlambda, nlambda2, fill::zeros);
  arma::mat conv(nlambda, nlambda2, fill::zeros);
  arma::cube X(nlambda, nlambda2, n_spcov, fill::zeros);
  arma::cube V(nlambda, nlambda2, n_cov, fill::zeros);
  arma::cube residuals(nlambda, nlambda2, m, fill::zeros);
  arma::cube fitted(nlambda, nlambda2, m, fill::zeros);
  arma::mat mse(nlambda, nlambda2, fill::zeros);
  
  /* Precompute relevant quantities         */
  Z_        = Z.each_col() % (ones_m * (1.0 / sqrtm));
  //W         = W.each_col() % (ones_m * (1.0 / sqrtm));        // è corretto non dividere W per la numerosità campionaria m
  ZZ        = Z_.t() * Z_;
  L_ZZ      = (chol(ZZ)).t();
  L_ZZ_INV  = inv(trimatl(L_ZZ));
  ZZ_INV    = L_ZZ_INV.t() * L_ZZ_INV;
  P_ZZ      = Z_ * ZZ_INV * Z_.t();
  M_ZZ      = eye_m - P_ZZ;
  WMW       = W.t() * M_ZZ * W;                               // 19Nov25: la divisione per n è corretta così come è ora
  WMy       = W.t() * M_ZZ * y;
  Zy        = Z_.t() * y / static_cast<float>(sqrtm);
  v_LS      = L_ZZ_INV.t() * L_ZZ_INV * Zy;
  ZW        = Z_.t() * W / static_cast<float>(sqrtm);
  P_ZW      = L_ZZ_INV.t() * L_ZZ_INV * ZW;
  WW_CHOL_U = chol_qr_fact(W, m, n_spcov);
  ZZ_CHOL_U = chol_qr_fact(Z_, m, n_cov);
  y2        = as_scalar(y.t() * y) / static_cast<float>(m);
  Wy        = W.t() * y;
  M_ZZW     = M_ZZ * W;                                         // 19Nov25: la divisione per n è corretta così come è ora NON SONO CONVINTO
  
  Glen           = sum(groups, 1);
  arma::sp_mat F = ovglasso_Gmat2Fmat_sparse(groups, var_weights);
  FTF            = diagmat(spmat2spmat_mult(F.t(), F));
  FTF_INV        = diagmat(1.0 / diagvec(FTF));
  D              = forward_diff_difference_matrix(n_spcov, 1, diff_order);
  DTD            = D.t() * D;
  
  /* Initialize vectors u and z         */
  dim_z = sum(Glen);
  arma::colvec u(dim_z, fill::zeros);
  arma::colvec z(dim_z, fill::zeros);
  arma::cube Umat(nlambda, nlambda2, dim_z, fill::zeros);
  arma::cube Zmat(nlambda, nlambda2, dim_z, fill::zeros);
  
  /* main loop  */
  for (int j=0; j<nlambda; j++) {
    for (int k=0; k<nlambda2; k++) {
      
      /* compute relevant quantities: these quantities change with lambda2    */
      Atilde.head_rows(m)       = W * M_ZZ / sqrt(static_cast<float>(m));
      Atilde.tail_rows(n_spcov) = std::sqrt(lambda2(k)) * D;
      AtFTF_INVAt               = Atilde * FTF_INV * Atilde.t();
      
      /* perform linear regression with lasso penalty for the j-th lambda       */
      out = admm_ovglasso_cov_smo_large_n_update_1lambda(W, Z, y, Glen, groups, group_weights, var_weights,
                                                         G, F, FTF_INV, DTD, Atilde, AtFTF_INVAt, M_ZZW,
                                                         WMW, WMy, v_LS, P_ZW, WW_CHOL_U, Wy, y2,
                                                         ZZ_CHOL_U, Zy, ZW, u, z, lambda(j), lambda2(k),
                                                         rho_adaptation, rho, tau, mu, reltol, abstol, maxiter, ping);
      
      /* Retrieve output: warm start      */
      u = as<arma::colvec>(out["u"]);
      z = as<arma::colvec>(out["z"]);
      x = as<arma::colvec>(out["x"]);
      v = as<arma::colvec>(out["v"]);

      /* Store output      */
      niter(j, k)          = out["niter"];
      eltime(j, k)         = out["eltime"];
      conv(j, k)           = out["convergence"];
      h_objval(j, k)       = as<arma::vec>(out["objval"]);
      h_r_norm(j, k)       = as<arma::vec>(out["r_norm"]);
      h_s_norm(j, k)       = as<arma::vec>(out["s_norm"]);
      h_eps_pri(j, k)      = as<arma::vec>(out["eps_pri"]);
      h_eps_dual(j, k)     = as<arma::vec>(out["eps_dual"]);
      h_rho(j, k)          = (as<arma::vec>(out["rho"])).subvec(0, niter(j, k)-1);
      rho                  = out["rho.updated"];
      X.tube(j, k)         = x;
      V.tube(j, k)         = v;
      residuals.tube(j, k) = as<arma::vec>(out["residuals"]);
      fitted.tube(j, k)    = as<arma::vec>(out["fitted.values"]);
      mse(j, k)            = out["mse"];
      Umat.tube(j, k)      = u;
      Zmat.tube(j, k)      = z;
    }
  }

  /* Find min MSE in-sample      */
  idx_min_mse     = mse.index_min();
  min_mse         = mse(idx_min_mse);
  idx_min_mse_row = idx_min_mse % X.n_rows;
  idx_min_mse_col = idx_min_mse / X.n_rows;

  /* Get output         */
  output["sp.coef.path"]    = X;
  output["coef.path"]       = V;
  output["iternum"]         = niter;
  output["objfun"]          = h_objval;
  output["r_norm"]          = h_r_norm;
  output["s_norm"]          = h_s_norm;
  output["err_pri"]         = h_eps_pri;
  output["err_dual"]        = h_eps_dual;
  output["convergence"]     = conv;
  output["elapsedTime"]     = eltime;
  output["rho"]             = h_rho;
  output["lambda"]          = lambda;
  output["min.mse"]         = min_mse;
  output["indi.min.mse"]    = idx_min_mse;
  output["lambda.min"]      = lambda(idx_min_mse_row);
  output["lambda2.min"]     = lambda2(idx_min_mse_col);
  output["sp.coefficients"] = X.tube(idx_min_mse_row, idx_min_mse_col);
  output["coefficients"]    = V.tube(idx_min_mse_row, idx_min_mse_col);
  output["mse"]             = mse;
  output["U"]               = Umat;
  output["Z"]               = Zmat;
  
  /* Return output      */
  return(output);
}

/*
    Overlap Group-LASSO via ADMM: functions to fit a single lambda
*/

Rcpp::List admm_ovglasso_large_m_update_1lambda(const arma::mat& A,
                                                const arma::vec& b,
                                                const arma::colvec& Glen,
                                                const arma::mat& groups,
                                                const arma::vec& group_weights,
                                                const arma::vec& var_weights,
                                                const arma::sp_mat& F,
                                                const arma::mat& FTF,
                                                const arma::mat& ATA,
                                                const arma::mat& ATA_CHOL_U,
                                                const arma::colvec& ATb,
                                                const double b2,
                                                const int m,
                                                const int n,
                                                const int G,
                                                arma::colvec& u,
                                                arma::colvec& z,
                                                const double lambda,
                                                const bool rho_adaptation,
                                                double rho,
                                                const double tau,
                                                const double mu,
                                                const double reltol,
                                                const double abstol,
                                                const int maxiter,
                                                const int ping) {
  
  /* Variable delcaration           */
  int k = 0, dim_z = 0;
  double rho_old = 0.0, elTime = 0.0, mse = 0.0, rss = 0.0, pen = 0.0;
  uword g_id_init, g_id_end;
  
  /* Get dimensions                 */
  dim_z = z.n_elem;
  
  /* Definition of vectors and matrices     */
  arma::colvec x(n, fill::zeros);
  arma::colvec q(n, fill::zeros);
  arma::colvec z_old(dim_z, fill::zeros);
  arma::mat U(n, n, fill::zeros);
  arma::mat L(n, n, fill::zeros);
  arma::vec h_objval(maxiter, fill::zeros);
  arma::vec h_r_norm(maxiter, fill::zeros);
  arma::vec h_s_norm(maxiter, fill::zeros);
  arma::vec h_eps_pri(maxiter, fill::zeros);
  arma::vec h_eps_dual(maxiter, fill::zeros);
  arma::vec rho_store(maxiter+1, fill::zeros);
  arma::colvec fitted(m, fill::zeros);
  arma::colvec residuals(m, fill::zeros);

  /* ::::::::::::::::::::::::::::::::::::::::::::::
   Set the computational time                                 */
  wall_clock timer;
  timer.tic();
  
  /* store rho (only for adaptation)         */
  rho_store(0) = rho;
  rho_old      = rho;

  /* Precompute relevant quantities         */
  U = glasso_factor_fast_large_m(ATA, FTF, rho); // returns upper
  L = U.t();
  
  /* main loop  */
  for (k=0; k<maxiter; k++) {

    // 4-1. update 'x'
    if (rho != rho_old) {
      U = glasso_factor_fast_large_m(ATA, FTF, rho); // returns upper
      L = U.t();
    }
    //q = ATb + rho * F.t() * (z - u); // temporary value
    q = ATb + rho * spmat2vec_mult(F.t(), z - u);
    x = solve(trimatu(U), solve(trimatl(L), q));
    
    // 4-2. update 'z' with adaptation
    // see Boyd et al (2011) page 21
    z_old     = z;
    g_id_init = 0;
    for (int g=0; g<G; g++) {
      // precompute the quantities for updating z
      g_id_end         = g_id_init + Glen(g) - 1;
      arma::vec u_g    = u.subvec(g_id_init, g_id_end);
      arma::sp_mat F_g = F.submat(g_id_init, 0, g_id_end, n-1);
      
      // update z
      arma::vec z_g                 = glasso_prox(F_g * x + u_g, (lambda * group_weights[g]) / rho);
      z.subvec(g_id_init, g_id_end) = z_g;
      g_id_init                     = g_id_end + 1;
    }

    // 4-3. update 'u'
    u += spmat2vec_mult(F, x) - z;
  
    // 4-3. dianostics, reporting
    rss         = lm_rss(A, b, x);
    pen         = ovglasso_penalty(x, groups, group_weights, var_weights);
    h_objval(k) = 0.5 *  rss / static_cast<float>(m) + lambda * pen;
    h_r_norm(k) = norm(glasso_residual_sparse(F, x, z), 2);
    h_s_norm(k) = norm(glasso_dual_residual_sparse(F, z, z_old, rho), 2);
    if (norm(F * x, 2) > norm(-z, 2)) {
      h_eps_pri(k) = std::sqrt(z.n_elem) * abstol + reltol * norm(F * x, 2);
    } else {
      h_eps_pri(k) = std::sqrt(z.n_elem) * abstol + reltol * norm(-z, 2);
    }
    h_eps_dual(k) = std::sqrt(z.n_elem) * abstol + reltol * norm(F.t() * rho * u, 2);
    
    /* :::::::::::::::::::::::::::::::::::::::::::::::::::::::: */
    /* Relaxation                                               */
    rho_old = rho;
    if (rho_adaptation) {
      if (h_r_norm(k) > mu * h_s_norm(k)) {
        rho = rho * tau;
        u   = u / tau;
      } else if (h_s_norm(k) > mu * h_r_norm(k)) {
        rho = rho / tau;
        u   = u * tau;
      }
    }
    rho_store(k+1) = rho;
        
    /* :::::::::::::::::::::::::::::::::::::::::::::::::::::::: */
    /* Print to screen                                          */
    if ((ping > 0) && (k < maxiter)) {
      if (rho_adaptation) {
        if (((k+1) % ping) == 0) {
          Rcpp::Rcout << "\n\n\n" << std::endl;
          Rcpp::Rcout << "ADMM with adaptation for overlap group-LASSO is running!" << std::endl;
          Rcpp::Rcout << "Iteration n.: " << k+1 << " of " << maxiter << "\n" << std::endl;
        }
      } else {
        if (((k+1) % ping) == 0) {
          Rcpp::Rcout << "\n\n\n" << std::endl;
          Rcpp::Rcout << "ADMM for overlap group-LASSO is running!" << std::endl;
          Rcpp::Rcout << "Iteration n.: " << k+1 << " of " << maxiter << "\n" << std::endl;
        }
      }
    }

    // 4-4. termination
    if ((h_r_norm(k) < h_eps_pri(k)) && (h_s_norm(k) < h_eps_dual(k))) {
      break;
    }
  }

  /* ::::::::::::::::::::::::::::::::::::::::::::::
   Compute elapsed time                                */
  elTime = timer.toc();
  
  /* ::::::::::::::::::::::::::::::::::::::::::::::
   Compute post-estimation quantities                   */
  fitted    = A * x;
  residuals = b - fitted;
  mse       = as_scalar(residuals.t() * residuals) / (static_cast<float>(m));
  
  /* Get output         */
  List output;
  output["x"]      = x;               
  output["u"]      = u;
  output["z"]      = z;
  output["eltime"] = elTime;
  if (k < maxiter) {
    output["convergence"] = 1.0;
    output["niter"]       = k+1;               
    output["objval"]      = h_objval.subvec(0, k);        
    output["r_norm"]      = h_r_norm.subvec(0, k);
    output["s_norm"]      = h_s_norm.subvec(0, k);
    output["eps_pri"]     = h_eps_pri.subvec(0, k);
    output["eps_dual"]    = h_eps_dual.subvec(0, k);
    output["rho"]         = rho_store.subvec(0, k+1);
  } else {
    output["convergence"] = 0.0;
    output["niter"]       = maxiter;               
    output["objval"]      = h_objval;        
    output["r_norm"]      = h_r_norm;
    output["s_norm"]      = h_s_norm;
    output["eps_pri"]     = h_eps_pri;
    output["eps_dual"]    = h_eps_dual;
    output["rho"]         = rho_store;
  }
  output["residuals"]     = residuals;
  output["fitted.values"] = fitted;
  output["mse"]           = mse;
  output["rho.updated"]   = rho;
  
  /* Return output      */
  return(output);
}

Rcpp::List admm_ovglasso_smo_large_m_update_1lambda(const arma::mat& A,
                                                    const arma::vec& b,
                                                    const arma::colvec& Glen,
                                                    const arma::mat& groups,
                                                    const arma::vec& group_weights,
                                                    const arma::vec& var_weights,
                                                    const arma::sp_mat& F,
                                                    const arma::mat& FTF,
                                                    const arma::mat& DTD,
                                                    const arma::mat& ATA,
                                                    const arma::mat& ATA_CHOL_U,
                                                    const arma::colvec& ATb,
                                                    const double b2,
                                                    const int m,
                                                    const int n,
                                                    const int G,
                                                    arma::colvec& u,
                                                    arma::colvec& z,
                                                    const double lambda,
                                                    const double lambda2,
                                                    bool rho_adaptation,
                                                    double rho,
                                                    const double tau,
                                                    const double mu,
                                                    const double reltol,
                                                    const double abstol,
                                                    const int maxiter,
                                                    const int ping) {
  
  /* Variable delcaration           */
  int k = 0, dim_z = 0;
  double rho_old = 0.0, elTime = 0.0, mse = 0.0, rss = 0.0, pen = 0.0, pen2 = 0.0;
  uword g_id_init = 0, g_id_end = 0;
  
  /* Get dimensions                 */
  dim_z = z.n_elem;
  
  /* Definition of vectors and matrices     */
  arma::colvec x(n, fill::zeros);
  arma::colvec q(n, fill::zeros);
  arma::colvec z_old(dim_z, fill::zeros);
  arma::mat U(n, n, fill::zeros);
  arma::mat L(n, n, fill::zeros);
  arma::mat XtTXt(n, n, fill::zeros);
  arma::vec h_objval(maxiter, fill::zeros);
  arma::vec h_r_norm(maxiter, fill::zeros);
  arma::vec h_s_norm(maxiter, fill::zeros);
  arma::vec h_eps_pri(maxiter, fill::zeros);
  arma::vec h_eps_dual(maxiter, fill::zeros);
  arma::vec rho_store(maxiter+1, fill::zeros);
  arma::colvec fitted(m, fill::zeros);
  arma::colvec residuals(m, fill::zeros);
  
  /* ::::::::::::::::::::::::::::::::::::::::::::::
   Set the computational time                                 */
  wall_clock timer;
  timer.tic();
  
  /* store rho (only for adaptation)         */
  rho_store(0) = rho;
  rho_old      = rho;

  /* Precompute relevant quantities         */
  XtTXt = ATA + lambda2 * DTD;
  U     = glasso_factor_smo_fast_large_m2(XtTXt, FTF, rho); // returns upper
  L     = U.t();

  /* main loop  */
  for (k=0; k<maxiter; k++) {
    
    // 4-1. update 'x'
    if (rho != rho_old) {
      U = glasso_factor_smo_fast_large_m2(XtTXt, FTF, rho); // returns upper
      L = U.t();
    }
    //q = ATb + rho * F.t() * (z - u); // temporary value
    q = ATb + rho * spmat2vec_mult(F.t(), z - u);
    x = solve(trimatu(U), solve(trimatl(L), q));
    
    // 4-2. update 'z' with adaptation
    // see Boyd et al (2011) page 21
    z_old     = z;
    g_id_init = 0;
    for (int g=0; g<G; g++) {
      // precompute the quantities for updating z
      g_id_end         = g_id_init + Glen(g) - 1;
      arma::vec u_g    = u.subvec(g_id_init, g_id_end);
      arma::sp_mat F_g = F.submat(g_id_init, 0, g_id_end, n-1);
      
      // update z
      arma::vec z_g                 = glasso_prox(F_g * x + u_g, (lambda * group_weights[g]) / rho);
      z.subvec(g_id_init, g_id_end) = z_g;
      g_id_init                     = g_id_end + 1;
    }
  
    // 4-3. update 'u'
    u += spmat2vec_mult(F, x) - z;
  
    // 4-3. dianostics, reporting
    rss         = lm_rss(A, b, x);
    pen2        = 0.5 * as_scalar(x.t() * DTD * x);
    h_objval(k) = 0.5 *  rss / static_cast<float>(m) + lambda * pen + lambda2 * pen2;
    h_r_norm(k) = norm(glasso_residual_sparse(F, x, z), 2);
    h_s_norm(k) = norm(glasso_dual_residual_sparse(F, z, z_old, rho), 2);
    if (norm(F * x, 2) > norm(-z, 2)) {
      h_eps_pri(k) = std::sqrt(z.n_elem) * abstol + reltol * norm(F * x, 2);
    } else {
      h_eps_pri(k) = std::sqrt(z.n_elem) * abstol + reltol * norm(-z, 2);
    }
    h_eps_dual(k) = std::sqrt(z.n_elem) * abstol + reltol * norm(F.t() * rho * u, 2);
    
    /* :::::::::::::::::::::::::::::::::::::::::::::::::::::::: */
    /* Relaxation                                               */
    rho_old = rho;
    if (rho_adaptation) {
      if (h_r_norm(k) > mu * h_s_norm(k)) {
        rho = rho * tau;
        u   = u / tau;
      } else if (h_s_norm(k) > mu * h_r_norm(k)) {
        rho = rho / tau;
        u   = u * tau;
      }
    }
    rho_store(k+1) = rho;
          
    /* :::::::::::::::::::::::::::::::::::::::::::::::::::::::: */
    /* Print to screen                                          */
    if ((ping > 0) && (k < maxiter)) {
      if (rho_adaptation) {
        if (((k+1) % ping) == 0) {
          Rcpp::Rcout << "\n\n\n" << std::endl;
          Rcpp::Rcout << "ADMM with adaptation for overlap group-LASSO is running!" << std::endl;
          Rcpp::Rcout << "Iteration n.: " << k+1 << " of " << maxiter << "\n" << std::endl;
        }
      } else {
        if (((k+1) % ping) == 0) {
          Rcpp::Rcout << "\n\n\n" << std::endl;
          Rcpp::Rcout << "ADMM for overlap group-LASSO is running!" << std::endl;
          Rcpp::Rcout << "Iteration n.: " << k+1 << " of " << maxiter << "\n" << std::endl;
        }
      }
    }

    // 4-4. termination
    if ((h_r_norm(k) < h_eps_pri(k)) && (h_s_norm(k) < h_eps_dual(k))) {
      break;
    }
  }

  /* ::::::::::::::::::::::::::::::::::::::::::::::
   Compute elapsed time                                */
  elTime = timer.toc();
  
  /* ::::::::::::::::::::::::::::::::::::::::::::::
   Compute post-estimation quantities                   */
  fitted    = A * x;
  residuals = b - fitted;
  mse       = as_scalar(residuals.t() * residuals) / (static_cast<float>(m));

  /* Get output         */
  List output;
  output["x"]      = x;               
  output["u"]      = u;
  output["z"]      = z;
  output["eltime"] = elTime;
  if (k < maxiter) {
    output["convergence"] = 1.0;
    output["niter"]       = k+1;               
    output["objval"]      = h_objval.subvec(0, k);        
    output["r_norm"]      = h_r_norm.subvec(0, k);
    output["s_norm"]      = h_s_norm.subvec(0, k);
    output["eps_pri"]     = h_eps_pri.subvec(0, k);
    output["eps_dual"]    = h_eps_dual.subvec(0, k);
    output["rho"]         = rho_store.subvec(0, k+1);
  } else {
    output["convergence"] = 0.0;
    output["niter"]       = maxiter;               
    output["objval"]      = h_objval;        
    output["r_norm"]      = h_r_norm;
    output["s_norm"]      = h_s_norm;
    output["eps_pri"]     = h_eps_pri;
    output["eps_dual"]    = h_eps_dual;
    output["rho"]         = rho_store;
  }
  output["residuals"]     = residuals;
  output["fitted.values"] = fitted;
  output["mse"]           = mse;
  output["rho.updated"]   = rho;
  
  /* Return output      */
  return(output);
}

Rcpp::List admm_ovglasso_large_n_update_1lambda(const arma::mat& A,
                                                const arma::vec& b,
                                                const arma::colvec& Glen,
                                                const arma::mat& groups,
                                                const arma::vec& group_weights,
                                                const arma::vec& var_weights,
                                                const arma::sp_mat& F,
                                                const arma::mat& FTF_INV,
                                                const arma::mat& AFTF_INVA,
                                                const arma::mat& ATA_CHOL_U,
                                                const arma::colvec& ATb,
                                                const double b2,
                                                const int m,
                                                const int n,
                                                const int G,
                                                arma::colvec& u,
                                                arma::colvec& z,
                                                const double lambda,
                                                const bool rho_adaptation,
                                                double rho,
                                                const double tau,
                                                const double mu,
                                                const double reltol,
                                                const double abstol,
                                                const int maxiter,
                                                const int ping) {
  
  /* Variable delcaration           */
  int k = 0, dim_z = 0;
  double rho_old = 0.0, elTime = 0.0, mse = 0.0, rss = 0.0, pen = 0.0;
  uword g_id_init, g_id_end;
  
  /* Get dimensions                 */
  dim_z = z.n_elem;
  
  /* Definition of vectors and matrices     */
  arma::colvec x(n, fill::zeros);
  arma::colvec x_star(n, fill::zeros);
  arma::colvec q(n, fill::zeros);
  arma::colvec z_old(dim_z, fill::zeros);
  arma::mat U(m, m, fill::zeros);
  arma::mat L(m, m, fill::zeros);
  arma::vec h_objval(maxiter, fill::zeros);
  arma::vec h_r_norm(maxiter, fill::zeros);
  arma::vec h_s_norm(maxiter, fill::zeros);
  arma::vec h_eps_pri(maxiter, fill::zeros);
  arma::vec h_eps_dual(maxiter, fill::zeros);
  arma::vec rho_store(maxiter+1, fill::zeros);
  arma::colvec fitted(m, fill::zeros);
  arma::colvec residuals(m, fill::zeros);

  /* ::::::::::::::::::::::::::::::::::::::::::::::
   Set the computational time                                 */
  wall_clock timer;
  timer.tic();
  
  /* store rho (only for adaptation)         */
  rho_store(0) = rho;
  rho_old      = rho;

  /* Precompute relevant quantities         */
  U = glasso_factor_fast_large_n(AFTF_INVA, rho, m); // returns upper
  L = U.t();
  
  /* main loop  */
  for (k=0; k<maxiter; k++) {

    // 4-1. update 'x'
    if (rho != rho_old) {
      U = glasso_factor_fast_large_n(AFTF_INVA, rho, m); // returns upper
      L = U.t();
    }
    //q      = ATb + rho * F.t() * (z - u); // temporary value
    q      = ATb + rho * spmat2vec_mult(F.t(), z - u);
    x_star = (diagmat(FTF_INV) * q) / rho;
    x      = x_star - diagmat(FTF_INV) * A.t() * solve(trimatu(U), solve(trimatl(L), A * x_star));
    
    // 4-2. update 'z' with adaptation
    // see Boyd et al (2011) page 21
    z_old     = z;
    g_id_init = 0;
    for (int g=0; g<G; g++) {
      // precompute the quantities for updating z
      g_id_end         = g_id_init + Glen(g) - 1;
      arma::vec u_g    = u.subvec(g_id_init, g_id_end);
      arma::sp_mat F_g = F.submat(g_id_init, 0, g_id_end, n-1);

      // update z
      arma::vec z_g                 = glasso_prox(F_g * x + u_g, (lambda * group_weights[g]) / rho);
      z.subvec(g_id_init, g_id_end) = z_g;
      g_id_init                     = g_id_end + 1;
    }

    // 4-3. update 'u'
    //u = u + F * x - z;
    u += spmat2vec_mult(F, x) - z;
  
    // 4-3. dianostics, reporting
    rss         = lm_rss(A, b, x);
    pen         = ovglasso_penalty(x, groups, group_weights, var_weights);
    h_objval(k) = 0.5 *  rss / static_cast<float>(m) + lambda * pen;
    h_r_norm(k) = norm(glasso_residual_sparse(F, x, z), 2);
    h_s_norm(k) = norm(glasso_dual_residual_sparse(F, z, z_old, rho), 2);
    if (norm(F * x, 2) > norm(-z, 2)) {
      h_eps_pri(k) = std::sqrt(z.n_elem) * abstol + reltol * norm(F * x, 2);
    } else {
      h_eps_pri(k) = std::sqrt(z.n_elem) * abstol + reltol * norm(-z, 2);
    }
    h_eps_dual(k) = std::sqrt(z.n_elem) * abstol + reltol * norm(F.t() * rho * u, 2);
    
    /* :::::::::::::::::::::::::::::::::::::::::::::::::::::::: */
    /* Relaxation                                               */
    rho_old = rho;
    if (rho_adaptation) {
      if (h_r_norm(k) > mu * h_s_norm(k)) {
        rho = rho * tau;
        u   = u / tau;
      } else if (h_s_norm(k) > mu * h_r_norm(k)) {
        rho = rho / tau;
        u   = u * tau;
      }
    }
    rho_store(k+1) = rho;
        
    /* :::::::::::::::::::::::::::::::::::::::::::::::::::::::: */
    /* Print to screen                                          */
    if ((ping > 0) && (k < maxiter)) {
      if (rho_adaptation) {
        if (((k+1) % ping) == 0) {
          Rcpp::Rcout << "\n\n\n" << std::endl;
          Rcpp::Rcout << "ADMM with adaptation for overlap group-LASSO is running!" << std::endl;
          Rcpp::Rcout << "Iteration n.: " << k+1 << " of " << maxiter << "\n" << std::endl;
        }
      } else {
        if (((k+1) % ping) == 0) {
          Rcpp::Rcout << "\n\n\n" << std::endl;
          Rcpp::Rcout << "ADMM for overlap group-LASSO is running!" << std::endl;
          Rcpp::Rcout << "Iteration n.: " << k+1 << " of " << maxiter << "\n" << std::endl;
        }
      }
    }

    // 4-4. termination
    if ((h_r_norm(k) < h_eps_pri(k)) && (h_s_norm(k) < h_eps_dual(k))) {
      break;
    }
  }

  /* ::::::::::::::::::::::::::::::::::::::::::::::
   Compute elapsed time                                */
  elTime = timer.toc();
  
  /* ::::::::::::::::::::::::::::::::::::::::::::::
   Compute post-estimation quantities                   */
  fitted    = A * x;
  residuals = b - fitted;
  mse       = as_scalar(residuals.t() * residuals) / (static_cast<float>(m));
  
  /* Get output         */
  List output;
  output["x"]      = x;               
  output["u"]      = u;
  output["z"]      = z;
  output["eltime"] = elTime;
  if (k < maxiter) {
    output["convergence"] = 1.0;
    output["niter"]       = k+1;               
    output["objval"]      = h_objval.subvec(0, k);        
    output["r_norm"]      = h_r_norm.subvec(0, k);
    output["s_norm"]      = h_s_norm.subvec(0, k);
    output["eps_pri"]     = h_eps_pri.subvec(0, k);
    output["eps_dual"]    = h_eps_dual.subvec(0, k);
    output["rho"]         = rho_store.subvec(0, k+1);
  } else {
    output["convergence"] = 0.0;
    output["niter"]       = maxiter;               
    output["objval"]      = h_objval;        
    output["r_norm"]      = h_r_norm;
    output["s_norm"]      = h_s_norm;
    output["eps_pri"]     = h_eps_pri;
    output["eps_dual"]    = h_eps_dual;
    output["rho"]         = rho_store;
  }
  output["residuals"]     = residuals;
  output["fitted.values"] = fitted;
  output["mse"]           = mse;
  output["rho.updated"]   = rho;
  
  /* Return output      */
  return(output);
}

Rcpp::List admm_ovglasso_smo_large_n_update_1lambda(const arma::mat& A,
                                                    const arma::vec& b,
                                                    const arma::colvec& Glen,
                                                    const arma::mat& groups,
                                                    const arma::vec& group_weights,
                                                    const arma::vec& var_weights,
                                                    const arma::sp_mat& F,
                                                    const arma::mat& FTF_INV,
                                                    const arma::mat& At,
                                                    const arma::mat& AtFTF_INVAt,
                                                    const arma::mat& ATA_CHOL_U,
                                                    const arma::colvec& ATb,
                                                    const double b2,
                                                    const int m,
                                                    const int n,
                                                    const int G,
                                                    arma::colvec& u,
                                                    arma::colvec& z,
                                                    const double lambda,
                                                    const double lambda2,
                                                    const bool rho_adaptation,
                                                    double rho,
                                                    const double tau,
                                                    const double mu,
                                                    const double reltol,
                                                    const double abstol,
                                                    const int maxiter,
                                                    const int ping) {
  
  /* Variable delcaration           */
  int k, dim_z;
  double rho_old, elTime = 0.0, mse, rss = 0.0, pen = 0.0;
  uword g_id_init, g_id_end;
  
  /* Get dimensions                 */
  dim_z = z.n_elem;
  
  /* Definition of vectors and matrices     */
  arma::colvec x(n, fill::zeros);
  arma::colvec x_star(n, fill::zeros);
  arma::colvec q(n, fill::zeros);
  arma::colvec z_old(dim_z, fill::zeros);
  arma::mat U(m, m, fill::zeros);
  arma::mat L(m, m, fill::zeros);
  arma::vec h_objval(maxiter, fill::zeros);
  arma::vec h_r_norm(maxiter, fill::zeros);
  arma::vec h_s_norm(maxiter, fill::zeros);
  arma::vec h_eps_pri(maxiter, fill::zeros);
  arma::vec h_eps_dual(maxiter, fill::zeros);
  arma::vec rho_store(maxiter+1, fill::zeros);
  arma::colvec fitted(m, fill::zeros);
  arma::colvec residuals(m, fill::zeros);

  /* ::::::::::::::::::::::::::::::::::::::::::::::
   Set the computational time                                 */
  wall_clock timer;
  timer.tic();
  
  /* store rho (only for adaptation)         */
  rho_store(0) = rho;
  rho_old      = rho;

  /* Precompute relevant quantities         */
  //U = glasso_factor_smo_fast_large_n(AFTF_INVA, m); // returns upper
  U = glasso_factor_smo_fast_large_n2(AtFTF_INVAt, rho, m); // returns upper
  L = U.t();
  
  /* main loop  */
  for (k=0; k<maxiter; k++) {

    // 4-1. update 'x'
    if (rho != rho_old) {
      //U = glasso_factor_smo_fast_large_n(AFTF_INVA, m); // returns upper
      U = glasso_factor_smo_fast_large_n2(AtFTF_INVAt, rho, m); // returns upper
      L = U.t();
    }
    //q      = ATb + rho * F.t() * (z - u); // temporary value
    q      = ATb + rho * spmat2vec_mult(F.t(), z - u);   // PROBLEMA: non capisco perché non dividiamo per m F * (z-u)
    x_star = (diagmat(FTF_INV) * q) / rho;
    x      = x_star - diagmat(FTF_INV) * At.t() * solve(trimatu(U), solve(trimatl(L), At * x_star));
    
    // 4-2. update 'z' with adaptation
    // see Boyd et al (2011) page 21
    z_old     = z;
    g_id_init = 0;
    for (int g=0; g<G; g++) {
      // precompute the quantities for updating z
      g_id_end         = g_id_init + Glen(g) - 1;
      arma::vec u_g    = u.subvec(g_id_init, g_id_end);
      arma::sp_mat F_g = F.submat(g_id_init, 0, g_id_end, n-1);

      // update z
      arma::vec z_g                 = glasso_prox(F_g * x + u_g, (lambda * group_weights[g]) / rho);
      z.subvec(g_id_init, g_id_end) = z_g;
      g_id_init                     = g_id_end + 1;
    }

    // 4-3. update 'u'
    //u = u + F * x - z;
    u += spmat2vec_mult(F, x) - z;
  
    // 4-3. dianostics, reporting
    rss         = lm_rss(A, b, x);
    pen         = ovglasso_penalty(x, groups, group_weights, var_weights);
    h_objval(k) = 0.5 *  rss / static_cast<float>(m) + lambda * pen;
    h_r_norm(k) = norm(glasso_residual_sparse(F, x, z), 2);
    h_s_norm(k) = norm(glasso_dual_residual_sparse(F, z, z_old, rho), 2);
    if (norm(F * x, 2) > norm(-z, 2)) {
      h_eps_pri(k) = std::sqrt(z.n_elem) * abstol + reltol * norm(F * x, 2);
    } else {
      h_eps_pri(k) = std::sqrt(z.n_elem) * abstol + reltol * norm(-z, 2);
    }
    h_eps_dual(k) = std::sqrt(z.n_elem) * abstol + reltol * norm(F.t() * rho * u, 2);
    
    /* :::::::::::::::::::::::::::::::::::::::::::::::::::::::: */
    /* Relaxation                                               */
    rho_old = rho;
    if (rho_adaptation) {
      if (h_r_norm(k) > mu * h_s_norm(k)) {
        rho = rho * tau;
        u   = u / tau;
      } else if (h_s_norm(k) > mu * h_r_norm(k)) {
        rho = rho / tau;
        u   = u * tau;
      }
    }
    rho_store(k+1) = rho;
        
    /* :::::::::::::::::::::::::::::::::::::::::::::::::::::::: */
    /* Print to screen                                          */
    if ((ping > 0) && (k < maxiter)) {
      if (rho_adaptation) {
        if (((k+1) % ping) == 0) {
          Rcpp::Rcout << "\n\n\n" << std::endl;
          Rcpp::Rcout << "ADMM with adaptation for overlap group-LASSO is running!" << std::endl;
          Rcpp::Rcout << "Iteration n.: " << k+1 << " of " << maxiter << "\n" << std::endl;
        }
      } else {
        if (((k+1) % ping) == 0) {
          Rcpp::Rcout << "\n\n\n" << std::endl;
          Rcpp::Rcout << "ADMM for overlap group-LASSO is running!" << std::endl;
          Rcpp::Rcout << "Iteration n.: " << k+1 << " of " << maxiter << "\n" << std::endl;
        }
      }
    }

    // 4-4. termination
    if ((h_r_norm(k) < h_eps_pri(k)) && (h_s_norm(k) < h_eps_dual(k))) {
      break;
    }
  }

  /* ::::::::::::::::::::::::::::::::::::::::::::::
   Compute elapsed time                                */
  elTime = timer.toc();
  
  /* ::::::::::::::::::::::::::::::::::::::::::::::
   Compute post-estimation quantities                   */
  fitted    = A * x;
  residuals = b - fitted;
  mse       = as_scalar(residuals.t() * residuals) / (static_cast<float>(m));
  
  /* Get output         */
  List output;
  output["x"]      = x;               
  output["u"]      = u;
  output["z"]      = z;
  output["eltime"] = elTime;
  if (k < maxiter) {
    output["convergence"] = 1.0;
    output["niter"]       = k+1;               
    output["objval"]      = h_objval.subvec(0, k);        
    output["r_norm"]      = h_r_norm.subvec(0, k);
    output["s_norm"]      = h_s_norm.subvec(0, k);
    output["eps_pri"]     = h_eps_pri.subvec(0, k);
    output["eps_dual"]    = h_eps_dual.subvec(0, k);
    output["rho"]         = rho_store.subvec(0, k+1);
  } else {
    output["convergence"] = 0.0;
    output["niter"]       = maxiter;               
    output["objval"]      = h_objval;        
    output["r_norm"]      = h_r_norm;
    output["s_norm"]      = h_s_norm;
    output["eps_pri"]     = h_eps_pri;
    output["eps_dual"]    = h_eps_dual;
    output["rho"]         = rho_store;
  }
  output["residuals"]     = residuals;
  output["fitted.values"] = fitted;
  output["mse"]           = mse;
  output["rho.updated"]   = rho;
  
  /* Return output      */
  return(output);
}

Rcpp::List admm_ovglasso_cov_large_m_update_1lambda(const arma::mat& W,
                                                    const arma::mat& Z,
                                                    const arma::vec& y,
                                                    const arma::colvec& Glen,
                                                    const arma::mat& groups,
                                                    const arma::vec& group_weights,
                                                    const arma::vec& var_weights,
                                                    const int G,
                                                    const arma::sp_mat& F,
                                                    const arma::mat& FTF,
                                                    const arma::mat& WMW,
                                                    const arma::vec& WMy,
                                                    const arma::vec& v_LS,
                                                    const arma::mat& P_ZW,
                                                    const arma::mat& WW_CHOL_U,
                                                    const arma::colvec& Wy,
                                                    const double y2,
                                                    const arma::mat& ZZ_CHOL_U,
                                                    const arma::colvec& Zy,
                                                    const arma::mat& ZW,
                                                    arma::colvec& u,
                                                    arma::colvec& z,
                                                    const double lambda,
                                                    const bool rho_adaptation,
                                                    double rho,
                                                    const double tau,
                                                    const double mu,
                                                    const double reltol,
                                                    const double abstol,
                                                    const int maxiter,
                                                    const int ping) {
  
  /* Variable delcaration           */
  int k = 0;
  double rho_old = 0.0, elTime = 0.0, rss = 0.0, pen = 0.0;
  uword g_id_init, g_id_end;
  
  /* Get dimensions             */
  const int m       = W.n_rows;
  const int n_spcov = W.n_cols;
  const int n_cov   = Z.n_cols;
  const int dim_z   = z.n_elem;
    
  /* Definition of vectors and matrices     */
  arma::colvec x(n_spcov, fill::zeros);
  arma::colvec v(n_cov, fill::zeros);
  arma::colvec q(n_spcov, fill::zeros);
  arma::colvec z_old(dim_z, fill::zeros);
  arma::mat U(n_spcov, n_spcov, fill::zeros);
  arma::mat L(n_spcov, n_spcov, fill::zeros);
  arma::vec h_objval(maxiter, fill::zeros);
  arma::vec h_r_norm(maxiter, fill::zeros);
  arma::vec h_s_norm(maxiter, fill::zeros);
  arma::vec h_eps_pri(maxiter, fill::zeros);
  arma::vec h_eps_dual(maxiter, fill::zeros);
  arma::vec rho_store(maxiter+1, fill::zeros);
  //arma::colvec fitted(m, fill::zeros);
  //arma::colvec residuals(m, fill::zeros);
  
  /* ::::::::::::::::::::::::::::::::::::::::::::::
   Set the computational time                                 */
  wall_clock timer;
  timer.tic();
  
  /* store rho (only for adaptation)         */
  rho_store(0) = rho;
  rho_old      = rho;

  /* Precompute relevant quantities         */
  //U = glasso_factor_fast_large_m(ATA, FTF, rho); // returns upper
  U = glasso_factor_cov_fast_large_m(WMW, FTF, rho);
  L = U.t();
  
  /* main loop  */
  for (k=0; k<maxiter; k++) {

    // 4-1. update 'x'
    if (rho != rho_old) {
      //U = glasso_factor_fast_large_m(ATA, FTF, rho); // returns upper
      U = glasso_factor_cov_fast_large_m(WMW, FTF, rho);
      L = U.t();
    }
    q = WMy + rho * spmat2vec_mult(F.t(), z - u);        // temporary value
    x = solve(trimatu(U), solve(trimatl(L), q));         // updated of regression parameters associated to covariates W (sparsified covariates, x)
    v = v_LS - P_ZW * x;                                 // updated of regression parameters associated to covariates Z (covariates not sparsified, v)
    
    // 4-2. update 'z' with adaptation
    // see Boyd et al (2011) page 21
    z_old     = z;
    g_id_init = 0;
    for (int g=0; g<G; g++) {
      // precompute the quantities for updating z
      g_id_end         = g_id_init + Glen(g) - 1;
      arma::vec u_g    = u.subvec(g_id_init, g_id_end);
      arma::sp_mat F_g = F.submat(g_id_init, 0, g_id_end, n_spcov-1);

      // update z
      arma::vec z_g                 = glasso_prox(F_g * x + u_g, (lambda * group_weights[g]) / rho);
      z.subvec(g_id_init, g_id_end) = z_g;
      g_id_init                     = g_id_end + 1;
    }

    // 4-3. update 'u'
    u += spmat2vec_mult(F, x) - z;
  
    // 4-3. dianostics, reporting
    rss         = lm_cov_rss_fast(WW_CHOL_U, ZZ_CHOL_U, ZW, Wy, Zy, y2, x, v, m);
    pen         = ovglasso_penalty(x, groups, group_weights, var_weights);
    h_objval(k) = 0.5 *  rss / static_cast<float>(m) + lambda * pen;
    h_r_norm(k) = norm(glasso_residual_sparse(F, x, z), 2);
    h_s_norm(k) = norm(glasso_dual_residual_sparse(F, z, z_old, rho), 2);
    if (norm(F * x, 2) > norm(-z, 2)) {
      h_eps_pri(k) = std::sqrt(z.n_elem) * abstol + reltol * norm(F * x, 2);
    } else {
      h_eps_pri(k) = std::sqrt(z.n_elem) * abstol + reltol * norm(-z, 2);
    }
    h_eps_dual(k) = std::sqrt(z.n_elem) * abstol + reltol * norm(F.t() * rho * u, 2);
    
    /* :::::::::::::::::::::::::::::::::::::::::::::::::::::::: */
    /* Relaxation                                               */
    rho_old = rho;
    if (rho_adaptation) {
      if (h_r_norm(k) > mu * h_s_norm(k)) {
        rho = rho * tau;
        u   = u / tau;
      } else if (h_s_norm(k) > mu * h_r_norm(k)) {
        rho = rho / tau;
        u   = u * tau;
      }
    }
    rho_store(k+1) = rho;
        
    /* :::::::::::::::::::::::::::::::::::::::::::::::::::::::: */
    /* Print to screen                                          */
    if ((ping > 0) && (k < maxiter)) {
      if (rho_adaptation) {
        if (((k+1) % ping) == 0) {
          Rcpp::Rcout << "\n\n\n" << std::endl;
          Rcpp::Rcout << "ADMM with adaptation for overlap group-LASSO is running!" << std::endl;
          Rcpp::Rcout << "Iteration n.: " << k+1 << " of " << maxiter << "\n" << std::endl;
        }
      } else {
        if (((k+1) % ping) == 0) {
          Rcpp::Rcout << "\n\n\n" << std::endl;
          Rcpp::Rcout << "ADMM for overlap group-LASSO is running!" << std::endl;
          Rcpp::Rcout << "Iteration n.: " << k+1 << " of " << maxiter << "\n" << std::endl;
        }
      }
    }

    // 4-4. termination
    if ((h_r_norm(k) < h_eps_pri(k)) && (h_s_norm(k) < h_eps_dual(k))) {
      break;
    }
  }

  /* ::::::::::::::::::::::::::::::::::::::::::::::
   Compute elapsed time                                */
  elTime = timer.toc();
  
  /* Get output         */
  List output;
  output["x"]      = x;               
  output["v"]      = v;               
  output["u"]      = u;
  output["z"]      = z;
  output["eltime"] = elTime;
  if (k < maxiter) {
    output["convergence"] = 1.0;
    output["niter"]       = k+1;               
    output["objval"]      = h_objval.subvec(0, k);        
    output["r_norm"]      = h_r_norm.subvec(0, k);
    output["s_norm"]      = h_s_norm.subvec(0, k);
    output["eps_pri"]     = h_eps_pri.subvec(0, k);
    output["eps_dual"]    = h_eps_dual.subvec(0, k);
    output["rho"]         = rho_store.subvec(0, k+1);
  } else {
    output["convergence"] = 0.0;
    output["niter"]       = maxiter;               
    output["objval"]      = h_objval;        
    output["r_norm"]      = h_r_norm;
    output["s_norm"]      = h_s_norm;
    output["eps_pri"]     = h_eps_pri;
    output["eps_dual"]    = h_eps_dual;
    output["rho"]         = rho_store;
  }
  //output["residuals"]     = residuals;
  //output["fitted.values"] = fitted;
  //output["mse"]           = mse;
  output["rho.updated"]   = rho;
  
  /* Return output      */
  return(output);
}

Rcpp::List admm_ovglasso_cov_smo_large_m_update_1lambda(const arma::mat& W,
                                                        const arma::mat& Z,
                                                        const arma::vec& y,
                                                        const arma::colvec& Glen,
                                                        const arma::mat& groups,
                                                        const arma::vec& group_weights,
                                                        const arma::vec& var_weights,
                                                        const int G,
                                                        const arma::sp_mat& F,
                                                        const arma::mat& FTF,
                                                        const arma::mat& DTD,
                                                        const arma::mat& WMW,
                                                        const arma::vec& WMy,
                                                        const arma::vec& v_LS,
                                                        const arma::mat& P_ZW,
                                                        const arma::mat& WW_CHOL_U,
                                                        const arma::colvec& Wy,
                                                        const double y2,
                                                        const arma::mat& ZZ_CHOL_U,
                                                        const arma::colvec& Zy,
                                                        const arma::mat& ZW,
                                                        arma::colvec& u,
                                                        arma::colvec& z,
                                                        const double lambda,
                                                        const double lambda2,
                                                        bool rho_adaptation,
                                                        double rho,
                                                        const double tau,
                                                        const double mu,
                                                        const double reltol,
                                                        const double abstol,
                                                        const int maxiter,
                                                        const int ping) {
  
  /* Variable delcaration           */
  int k = 0;
  double rho_old = 0.0, elTime = 0.0, rss = 0.0, pen = 0.0, pen2 = 0.0;
  uword g_id_init, g_id_end;
  
  /* Get dimensions             */
  const int m       = W.n_rows;
  const int n_spcov = W.n_cols;
  const int n_cov   = Z.n_cols;
  const int dim_z   = z.n_elem;
    
  /* Definition of vectors and matrices     */
  arma::colvec x(n_spcov, fill::zeros);
  arma::colvec v(n_cov, fill::zeros);
  arma::colvec q(n_spcov, fill::zeros);
  arma::colvec z_old(dim_z, fill::zeros);
  arma::mat U(n_spcov, n_spcov, fill::zeros);
  arma::mat L(n_spcov, n_spcov, fill::zeros);
  arma::vec h_objval(maxiter, fill::zeros);
  arma::vec h_r_norm(maxiter, fill::zeros);
  arma::vec h_s_norm(maxiter, fill::zeros);
  arma::vec h_eps_pri(maxiter, fill::zeros);
  arma::vec h_eps_dual(maxiter, fill::zeros);
  arma::vec rho_store(maxiter+1, fill::zeros);
  arma::colvec fitted(m, fill::zeros);
  arma::colvec residuals(m, fill::zeros);
  
  /* ::::::::::::::::::::::::::::::::::::::::::::::
   Set the computational time                                 */
  wall_clock timer;
  timer.tic();
  
  /* store rho (only for adaptation)         */
  rho_store(0) = rho;
  rho_old      = rho;

  /* Precompute relevant quantities         */
  //U = glasso_factor_fast_large_m(ATA, FTF, rho); // returns upper
  //U = glasso_factor_cov_fast_large_m(WMW, FTF, rho);
  U = glasso_factor_cov_smo_fast_large_m(WMW, FTF, DTD, lambda2, rho);
  L = U.t();
  
  /* main loop  */
  for (k=0; k<maxiter; k++) {

    // 4-1. update 'x'
    if (rho != rho_old) {
      //U = glasso_factor_fast_large_m(ATA, FTF, rho); // returns upper
      //U = glasso_factor_cov_fast_large_m(WMW, FTF, rho);
      U = glasso_factor_cov_smo_fast_large_m(WMW, FTF, DTD, lambda2, rho);
      L = U.t();
    }
    q = WMy + rho * spmat2vec_mult(F.t(), z - u);        // temporary value
    x = solve(trimatu(U), solve(trimatl(L), q));         // updated of regression parameters associated to covariates W (sparsified covariates, x)
    v = v_LS - P_ZW * x;                                 // updated of regression parameters associated to covariates Z (covariates not sparsified, v)
    
    // 4-2. update 'z' with adaptation
    // see Boyd et al (2011) page 21
    z_old     = z;
    g_id_init = 0;
    for (int g=0; g<G; g++) {
      // precompute the quantities for updating z
      g_id_end         = g_id_init + Glen(g) - 1;
      arma::vec u_g    = u.subvec(g_id_init, g_id_end);
      arma::sp_mat F_g = F.submat(g_id_init, 0, g_id_end, n_spcov-1);

      // update z
      arma::vec z_g                 = glasso_prox(F_g * x + u_g, (lambda * group_weights[g]) / rho);
      z.subvec(g_id_init, g_id_end) = z_g;
      g_id_init                     = g_id_end + 1;
    }

    // 4-3. update 'u'
    u += spmat2vec_mult(F, x) - z;
  
    // 4-3. dianostics, reporting
    rss         = lm_cov_rss_fast(WW_CHOL_U, ZZ_CHOL_U, ZW, Wy, Zy, y2, x, v, m);
    pen         = ovglasso_penalty(x, groups, group_weights, var_weights);
    pen2        = 0.5 * as_scalar(x.t() * DTD * x);
    h_objval(k) = 0.5 *  rss / static_cast<float>(m) + lambda * pen + lambda2 * pen2;
    h_r_norm(k) = norm(glasso_residual_sparse(F, x, z), 2);
    h_s_norm(k) = norm(glasso_dual_residual_sparse(F, z, z_old, rho), 2);
    if (norm(F * x, 2) > norm(-z, 2)) {
      h_eps_pri(k) = std::sqrt(z.n_elem) * abstol + reltol * norm(F * x, 2);
    } else {
      h_eps_pri(k) = std::sqrt(z.n_elem) * abstol + reltol * norm(-z, 2);
    }
    h_eps_dual(k) = std::sqrt(z.n_elem) * abstol + reltol * norm(F.t() * rho * u, 2);
    
    /* :::::::::::::::::::::::::::::::::::::::::::::::::::::::: */
    /* Relaxation                                               */
    rho_old = rho;
    if (rho_adaptation) {
      if (h_r_norm(k) > mu * h_s_norm(k)) {
        rho = rho * tau;
        u   = u / tau;
      } else if (h_s_norm(k) > mu * h_r_norm(k)) {
        rho = rho / tau;
        u   = u * tau;
      }
    }
    rho_store(k+1) = rho;
        
    /* :::::::::::::::::::::::::::::::::::::::::::::::::::::::: */
    /* Print to screen                                          */
    if ((ping > 0) && (k < maxiter)) {
      if (rho_adaptation) {
        if (((k+1) % ping) == 0) {
          Rcpp::Rcout << "\n\n\n" << std::endl;
          Rcpp::Rcout << "ADMM with adaptation for overlap group-LASSO is running!" << std::endl;
          Rcpp::Rcout << "Iteration n.: " << k+1 << " of " << maxiter << "\n" << std::endl;
        }
      } else {
        if (((k+1) % ping) == 0) {
          Rcpp::Rcout << "\n\n\n" << std::endl;
          Rcpp::Rcout << "ADMM for overlap group-LASSO is running!" << std::endl;
          Rcpp::Rcout << "Iteration n.: " << k+1 << " of " << maxiter << "\n" << std::endl;
        }
      }
    }

    // 4-4. termination
    if ((h_r_norm(k) < h_eps_pri(k)) && (h_s_norm(k) < h_eps_dual(k))) {
      break;
    }
  }

  /* ::::::::::::::::::::::::::::::::::::::::::::::
   Compute elapsed time                                */
  elTime = timer.toc();
  
  /* Get output         */
  List output;
  output["x"]      = x;               
  output["v"]      = v;               
  output["u"]      = u;
  output["z"]      = z;
  output["eltime"] = elTime;
  if (k < maxiter) {
    output["convergence"] = 1.0;
    output["niter"]       = k+1;               
    output["objval"]      = h_objval.subvec(0, k);        
    output["r_norm"]      = h_r_norm.subvec(0, k);
    output["s_norm"]      = h_s_norm.subvec(0, k);
    output["eps_pri"]     = h_eps_pri.subvec(0, k);
    output["eps_dual"]    = h_eps_dual.subvec(0, k);
    output["rho"]         = rho_store.subvec(0, k+1);
  } else {
    output["convergence"] = 0.0;
    output["niter"]       = maxiter;               
    output["objval"]      = h_objval;        
    output["r_norm"]      = h_r_norm;
    output["s_norm"]      = h_s_norm;
    output["eps_pri"]     = h_eps_pri;
    output["eps_dual"]    = h_eps_dual;
    output["rho"]         = rho_store;
  }
  //output["residuals"]     = residuals;
  //output["fitted.values"] = fitted;
  //output["mse"]           = mse;
  output["rho.updated"]   = rho;
  
  /* Return output      */
  return(output);
}

Rcpp::List admm_ovglasso_cov_large_n_update_1lambda(const arma::mat& W,
                                                    const arma::mat& Z,
                                                    const arma::vec& y,
                                                    const arma::colvec& Glen,
                                                    const arma::mat& groups,
                                                    const arma::vec& group_weights,
                                                    const arma::vec& var_weights,
                                                    const int G,
                                                    const arma::sp_mat& F,
                                                    const arma::mat& FF_INV,
                                                    const arma::mat& M_ZZW,
                                                    const arma::mat& M_ZZW2,
                                                    const arma::mat& WMW,
                                                    const arma::vec& WMy,
                                                    const arma::vec& v_LS,
                                                    const arma::mat& P_ZW,
                                                    const arma::mat& WW_CHOL_U,
                                                    const arma::colvec& Wy,
                                                    const double y2,
                                                    const arma::mat& ZZ_CHOL_U,
                                                    const arma::colvec& Zy,
                                                    const arma::mat& ZW,
                                                    arma::colvec& u,
                                                    arma::colvec& z,
                                                    const double lambda,
                                                    const bool rho_adaptation,
                                                    double rho,
                                                    const double tau,
                                                    const double mu,
                                                    const double reltol,
                                                    const double abstol,
                                                    const int maxiter,
                                                    const int ping) {
  
  /* Variable delcaration           */
  int k = 0;
  double rho_old = 0.0, elTime = 0.0, rss = 0.0, pen = 0.0;
  uword g_id_init, g_id_end;

  /* Get dimensions             */
  const int m       = W.n_rows;
  const int n_spcov = W.n_cols;
  const int n_cov   = Z.n_cols;
  const int dim_z   = z.n_elem;
    
  /* Definition of vectors and matrices     */
  arma::colvec x(n_spcov, fill::zeros);
  arma::colvec x_star(n_spcov, fill::zeros);
  arma::colvec v(n_cov, fill::zeros);
  arma::colvec q(n_spcov, fill::zeros);
  arma::colvec z_old(dim_z, fill::zeros);
  arma::mat U(m, m, fill::zeros);
  arma::mat L(m, m, fill::zeros);
  arma::vec h_objval(maxiter, fill::zeros);
  arma::vec h_r_norm(maxiter, fill::zeros);
  arma::vec h_s_norm(maxiter, fill::zeros);
  arma::vec h_eps_pri(maxiter, fill::zeros);
  arma::vec h_eps_dual(maxiter, fill::zeros);
  arma::vec rho_store(maxiter+1, fill::zeros);
  arma::colvec fitted(m, fill::zeros);
  arma::colvec residuals(m, fill::zeros);

  /* ::::::::::::::::::::::::::::::::::::::::::::::
   Set the computational time                                 */
  wall_clock timer;
  timer.tic();
  
  /* store rho (only for adaptation)         */
  rho_store(0) = rho;
  rho_old      = rho;

  /* Precompute relevant quantities         */
  U = glasso_factor_cov_fast_large_n(M_ZZW2, rho, m); // returns upper
  L = U.t();
  
  /* main loop  */
  for (k=0; k<maxiter; k++) {

    // 4-1. update 'x'
    if (rho != rho_old) {
      U = glasso_factor_cov_fast_large_n(M_ZZW2, rho, m); // returns upper
      L = U.t();
    }
    q      = WMy + rho * spmat2vec_mult(F.t(), z - u); // temporary value
    x_star = (diagmat(FF_INV) * q) / rho;
    x      = x_star - diagmat(FF_INV) * M_ZZW.t() * solve(trimatu(U), solve(trimatl(L), M_ZZW * x_star)); // updated of regression parameters associated to covariates W
    v      = v_LS - P_ZW * x;         // updated of regression parameters associated to covariates Z (covariates not sparsified, v)

    // 4-2. update 'z' with adaptation
    // see Boyd et al (2011) page 21
    z_old     = z;
    g_id_init = 0;
    for (int g=0; g<G; g++) {
      // precompute the quantities for updating z
      g_id_end         = g_id_init + Glen(g) - 1;
      arma::vec u_g    = u.subvec(g_id_init, g_id_end);
      arma::sp_mat F_g = F.submat(g_id_init, 0, g_id_end, n_spcov-1);

      // update z
      arma::vec z_g                 = glasso_prox(F_g * x + u_g, (lambda * group_weights[g]) / rho);
      z.subvec(g_id_init, g_id_end) = z_g;
      g_id_init                     = g_id_end + 1;
    }

    // 4-3. update 'u'
    u += spmat2vec_mult(F, x) - z;
  
    // 4-3. dianostics, reporting
    rss         = lm_cov_rss_fast(WW_CHOL_U, ZZ_CHOL_U, ZW, Wy, Zy, y2, x, v, m);
    pen         = ovglasso_penalty(x, groups, group_weights, var_weights);
    h_objval(k) = 0.5 *  rss / static_cast<float>(m) + lambda * pen;
    h_r_norm(k) = norm(glasso_residual_sparse(F, x, z), 2);
    h_s_norm(k) = norm(glasso_dual_residual_sparse(F, z, z_old, rho), 2);
    if (norm(F * x, 2) > norm(-z, 2)) {
      h_eps_pri(k) = std::sqrt(z.n_elem) * abstol + reltol * norm(F * x, 2);
    } else {
      h_eps_pri(k) = std::sqrt(z.n_elem) * abstol + reltol * norm(-z, 2);
    }
    h_eps_dual(k) = std::sqrt(z.n_elem) * abstol + reltol * norm(F.t() * rho * u, 2);
    
    /* :::::::::::::::::::::::::::::::::::::::::::::::::::::::: */
    /* Relaxation                                               */
    rho_old = rho;
    if (rho_adaptation) {
      if (h_r_norm(k) > mu * h_s_norm(k)) {
        rho = rho * tau;
        u   = u / tau;
      } else if (h_s_norm(k) > mu * h_r_norm(k)) {
        rho = rho / tau;
        u   = u * tau;
      }
    }
    rho_store(k+1) = rho;
        
    /* :::::::::::::::::::::::::::::::::::::::::::::::::::::::: */
    /* Print to screen                                          */
    if ((ping > 0) && (k < maxiter)) {
      if (rho_adaptation) {
        if (((k+1) % ping) == 0) {
          Rcpp::Rcout << "\n\n\n" << std::endl;
          Rcpp::Rcout << "ADMM with adaptation for overlap group-LASSO is running!" << std::endl;
          Rcpp::Rcout << "Iteration n.: " << k+1 << " of " << maxiter << "\n" << std::endl;
        }
      } else {
        if (((k+1) % ping) == 0) {
          Rcpp::Rcout << "\n\n\n" << std::endl;
          Rcpp::Rcout << "ADMM for overlap group-LASSO is running!" << std::endl;
          Rcpp::Rcout << "Iteration n.: " << k+1 << " of " << maxiter << "\n" << std::endl;
        }
      }
    }

    // 4-4. termination
    if ((h_r_norm(k) < h_eps_pri(k)) && (h_s_norm(k) < h_eps_dual(k))) {
      break;
    }
  }

  /* ::::::::::::::::::::::::::::::::::::::::::::::
   Compute elapsed time                                */
  elTime = timer.toc();
  
  /* Get output         */
  List output;
  output["x"]      = x;               
  output["v"]      = v;               
  output["u"]      = u;
  output["z"]      = z;
  output["eltime"] = elTime;
  if (k < maxiter) {
    output["convergence"] = 1.0;
    output["niter"]       = k+1;               
    output["objval"]      = h_objval.subvec(0, k);        
    output["r_norm"]      = h_r_norm.subvec(0, k);
    output["s_norm"]      = h_s_norm.subvec(0, k);
    output["eps_pri"]     = h_eps_pri.subvec(0, k);
    output["eps_dual"]    = h_eps_dual.subvec(0, k);
    output["rho"]         = rho_store.subvec(0, k+1);
  } else {
    output["convergence"] = 0.0;
    output["niter"]       = maxiter;               
    output["objval"]      = h_objval;        
    output["r_norm"]      = h_r_norm;
    output["s_norm"]      = h_s_norm;
    output["eps_pri"]     = h_eps_pri;
    output["eps_dual"]    = h_eps_dual;
    output["rho"]         = rho_store;
  }
  //output["residuals"]     = residuals;
  //output["fitted.values"] = fitted;
  //output["mse"]           = mse;
  output["rho.updated"]   = rho;
  
  /* Return output      */
  return(output);
}

Rcpp::List admm_ovglasso_cov_smo_large_n_update_1lambda(const arma::mat& W,
                                                        const arma::mat& Z,
                                                        const arma::vec& y,
                                                        const arma::colvec& Glen,
                                                        const arma::mat& groups,
                                                        const arma::vec& group_weights,
                                                        const arma::vec& var_weights,
                                                        const int G,
                                                        const arma::sp_mat& F,
                                                        const arma::mat& FTF_INV,
                                                        const arma::mat& DTD,
                                                        const arma::mat& At,
                                                        const arma::mat& AtFTF_INVAt,
                                                        const arma::mat& M_ZZW,
                                                        const arma::mat& WMW,
                                                        const arma::vec& WMy,
                                                        const arma::vec& v_LS,
                                                        const arma::mat& P_ZW,
                                                        const arma::mat& WW_CHOL_U,
                                                        const arma::colvec& Wy,
                                                        const double y2,
                                                        const arma::mat& ZZ_CHOL_U,
                                                        const arma::colvec& Zy,
                                                        const arma::mat& ZW,
                                                        arma::colvec& u,
                                                        arma::colvec& z,
                                                        const double lambda,
                                                        const double lambda2,
                                                        const bool rho_adaptation,
                                                        double rho,
                                                        const double tau,
                                                        const double mu,
                                                        const double reltol,
                                                        const double abstol,
                                                        const int maxiter,
                                                        const int ping) {
      

  /* Variable delcaration           */
  int k = 0;
  double rho_old = 0.0, elTime = 0.0, mse = 0.0, rss = 0.0, pen = 0.0, pen2 = 0.0;
  uword g_id_init, g_id_end;

  /* Get dimensions             */
  const int m       = W.n_rows;
  const int n_spcov = W.n_cols;
  const int n_cov   = Z.n_cols;
  const int dim_z   = z.n_elem;
    
  /* Definition of vectors and matrices     */
  arma::colvec x(n_spcov, fill::zeros);
  arma::colvec x_star(n_spcov, fill::zeros);
  arma::colvec v(n_cov, fill::zeros);
  arma::colvec q(n_spcov, fill::zeros);
  arma::colvec z_old(dim_z, fill::zeros);
  arma::mat U(m, m, fill::zeros);
  arma::mat L(m, m, fill::zeros);
  arma::vec h_objval(maxiter, fill::zeros);
  arma::vec h_r_norm(maxiter, fill::zeros);
  arma::vec h_s_norm(maxiter, fill::zeros);
  arma::vec h_eps_pri(maxiter, fill::zeros);
  arma::vec h_eps_dual(maxiter, fill::zeros);
  arma::vec rho_store(maxiter+1, fill::zeros);
  arma::colvec fitted(m, fill::zeros);
  arma::colvec residuals(m, fill::zeros);

  /* ::::::::::::::::::::::::::::::::::::::::::::::
   Set the computational time                                 */
  wall_clock timer;
  timer.tic();
  
  /* store rho (only for adaptation)         */
  rho_store(0) = rho;
  rho_old      = rho;

  /* Precompute relevant quantities         */
  U = glasso_factor_cov_smo_fast_large_n(AtFTF_INVAt, rho, m); // returns upper
  L = U.t();
  
  /* main loop  */
  for (k=0; k<maxiter; k++) {

    // 4-1. update 'x'
    if (rho != rho_old) {
      U = glasso_factor_cov_smo_fast_large_n(AtFTF_INVAt, rho, m); // returns upper
      L = U.t();
    }
    q      = WMy + rho * spmat2vec_mult(F.t(), z - u); // temporary value
    x_star = (diagmat(FTF_INV) * q) / rho;
    
    // updated of regression parameters associated to covariates W
    x = x_star - diagmat(FTF_INV) * At.t() * solve(trimatu(U), solve(trimatl(L), At * x_star));
    
    // updated of regression parameters associated to covariates Z (covariates not sparsified, v)
    v = v_LS - P_ZW * x;
    
    // 4-2. update 'z' with adaptation
    // see Boyd et al (2011) page 21
    z_old     = z;
    g_id_init = 0;
    for (int g=0; g<G; g++) {
      // precompute the quantities for updating z
      g_id_end         = g_id_init + Glen(g) - 1;
      arma::vec u_g    = u.subvec(g_id_init, g_id_end);
      arma::sp_mat F_g = F.submat(g_id_init, 0, g_id_end, n_spcov-1);

      // update z
      arma::vec z_g                 = glasso_prox(F_g * x + u_g, (lambda * group_weights[g]) / rho);
      z.subvec(g_id_init, g_id_end) = z_g;
      g_id_init                     = g_id_end + 1;
    }

    // 4-3. update 'u'
    u += spmat2vec_mult(F, x) - z;
  
    // 4-3. dianostics, reporting
    rss         = lm_cov_rss_fast(WW_CHOL_U, ZZ_CHOL_U, ZW, Wy, Zy, y2, x, v, m);
    pen         = ovglasso_penalty(x, groups, group_weights, var_weights);
    pen2        = 0.5 * as_scalar(x.t() * DTD * x);
    h_objval(k) = 0.5 *  rss / static_cast<float>(m) + lambda * pen + lambda2 * pen2;
    h_r_norm(k) = norm(glasso_residual_sparse(F, x, z), 2);
    h_s_norm(k) = norm(glasso_dual_residual_sparse(F, z, z_old, rho), 2);
    if (norm(F * x, 2) > norm(-z, 2)) {
      h_eps_pri(k) = std::sqrt(z.n_elem) * abstol + reltol * norm(F * x, 2);
    } else {
      h_eps_pri(k) = std::sqrt(z.n_elem) * abstol + reltol * norm(-z, 2);
    }
    h_eps_dual(k) = std::sqrt(z.n_elem) * abstol + reltol * norm(F.t() * rho * u, 2);
    
    /* :::::::::::::::::::::::::::::::::::::::::::::::::::::::: */
    /* Relaxation                                               */
    rho_old = rho;
    if (rho_adaptation) {
      if (h_r_norm(k) > mu * h_s_norm(k)) {
        rho = rho * tau;
        u   = u / tau;
      } else if (h_s_norm(k) > mu * h_r_norm(k)) {
        rho = rho / tau;
        u   = u * tau;
      }
    }
    rho_store(k+1) = rho;
        
    /* :::::::::::::::::::::::::::::::::::::::::::::::::::::::: */
    /* Print to screen                                          */
    if ((ping > 0) && (k < maxiter)) {
      if (rho_adaptation) {
        if (((k+1) % ping) == 0) {
          Rcpp::Rcout << "\n\n\n" << std::endl;
          Rcpp::Rcout << "ADMM with adaptation for overlap group-LASSO is running!" << std::endl;
          Rcpp::Rcout << "Iteration n.: " << k+1 << " of " << maxiter << "\n" << std::endl;
        }
      } else {
        if (((k+1) % ping) == 0) {
          Rcpp::Rcout << "\n\n\n" << std::endl;
          Rcpp::Rcout << "ADMM for overlap group-LASSO is running!" << std::endl;
          Rcpp::Rcout << "Iteration n.: " << k+1 << " of " << maxiter << "\n" << std::endl;
        }
      }
    }

    // 4-4. termination
    if ((h_r_norm(k) < h_eps_pri(k)) && (h_s_norm(k) < h_eps_dual(k))) {
      break;
    }
  }

  /* ::::::::::::::::::::::::::::::::::::::::::::::
   Compute elapsed time                                */
  elTime = timer.toc();
  
  /* ::::::::::::::::::::::::::::::::::::::::::::::
   Compute post-estimation quantities                   */
  fitted    = W * x + Z * v;
  residuals = y - fitted;
  mse       = as_scalar(residuals.t() * residuals) / (static_cast<float>(m));
  
  /* Get output         */
  List output;
  output["x"]      = x;               
  output["v"]      = v;               
  output["u"]      = u;
  output["z"]      = z;
  output["eltime"] = elTime;
  if (k < maxiter) {
    output["convergence"] = 1.0;
    output["niter"]       = k+1;               
    output["objval"]      = h_objval.subvec(0, k);        
    output["r_norm"]      = h_r_norm.subvec(0, k);
    output["s_norm"]      = h_s_norm.subvec(0, k);
    output["eps_pri"]     = h_eps_pri.subvec(0, k);
    output["eps_dual"]    = h_eps_dual.subvec(0, k);
    output["rho"]         = rho_store.subvec(0, k+1);
  } else {
    output["convergence"] = 0.0;
    output["niter"]       = maxiter;               
    output["objval"]      = h_objval;        
    output["r_norm"]      = h_r_norm;
    output["s_norm"]      = h_s_norm;
    output["eps_pri"]     = h_eps_pri;
    output["eps_dual"]    = h_eps_dual;
    output["rho"]         = rho_store;
  }
  output["residuals"]     = residuals;
  output["fitted.values"] = fitted;
  output["mse"]           = mse;
  output["rho.updated"]   = rho;
  
  /* Return output      */
  return(output);
}

/*
    Sparse Overlap Group-LASSO via ADMM: main functions, to be used
                                  to fit the model once (single lambda)     */

Rcpp::List admm_spovglasso_large_m(const arma::mat& A,
                                   const arma::vec& b,
                                   const arma::mat& groups,
                                   const arma::vec& group_weights,
                                   const arma::vec& var_weights,
                                   const arma::vec& var_weights_L1,
                                   arma::colvec& u,
                                   arma::colvec& z,
                                   const double lambda1,
                                   const double lambda2,
                                   const bool rho_adaptation,
                                   double rho,
                                   const double tau,
                                   const double mu,
                                   const double reltol,
                                   const double abstol,
                                   const int maxiter,
                                   const int ping) {
    
  /* Variable delcaration           */
  int k = 0;
  uword g_id_init, g_id_end;
  double spglasso_alpha = 0.0, spglasso_lambda = 0.0, rho_old = 0.0, elTime = 0.0, rss = 0.0, pen = 0.0;
  
  /* Get dimensions             */
  const int m     = A.n_rows;
  const int n     = A.n_cols;
  const int G     = groups.n_rows;
  const int dim_z = z.n_elem;
      
  /* Definition of vectors and matrices     */
  arma::colvec x(n, fill::zeros);
  arma::colvec q(n, fill::zeros);
  arma::colvec z_old(dim_z, fill::zeros);
  arma::mat ATA(n, n, fill::zeros);
  arma::mat FTF(n, n, fill::zeros);
  arma::vec ATb(n, fill::zeros);
  arma::mat U(n, n, fill::zeros);
  arma::mat L(n, n, fill::zeros);
  arma::colvec Glen(G, fill::zeros);
  arma::vec h_objval(maxiter, fill::zeros);
  arma::vec h_r_norm(maxiter, fill::zeros);
  arma::vec h_s_norm(maxiter, fill::zeros);
  arma::vec h_eps_pri(maxiter, fill::zeros);
  arma::vec h_eps_dual(maxiter, fill::zeros);
  arma::vec rho_store(maxiter+1, fill::zeros);
  
  /* ::::::::::::::::::::::::::::::::::::::::::::::
   Set the computational time                                 */
  wall_clock timer;
  timer.tic();
  
  /* store rho (only for adaptation)         */
  rho_store(0) = rho;
  rho_old      = rho;
  
  /* Precompute relevant quantities         */
  spglasso_lambda = lambda1;
  spglasso_alpha  = lambda2;
  ATA             = A.t() * A / static_cast<float>(m);
  ATb             = A.t() * b / static_cast<float>(m);
  Glen            = sum(groups, 1);
  arma::sp_mat F  = spovglasso_Gmat2Fmat_sparse(groups, var_weights, var_weights_L1);
  FTF             = F.t() * F;
  U               = glasso_factor_fast_large_m(ATA, FTF, rho); // returns upper
  L               = U.t();
  
  /* main loop  */
  for (k=0; k<maxiter; k++) {
    // 4-1. update 'x'
    if (rho != rho_old) {
      U = glasso_factor_fast_large_m(ATA, FTF, rho); // returns upper
      L = U.t();
    }
    q = ATb + rho * spmat2vec_mult(F.t(), z - u); // temporary value
    x = solve(trimatu(U), solve(trimatl(L), q));
    
    // 4-2. update 'z' with adaptation
    // see Boyd et al (2011) page 21
    z_old     = z;
    g_id_init = 0;
    for (int g=0; g<G; g++) {
      // precompute the quantities for updating z
      g_id_end         = g_id_init + Glen(g) - 1;
      arma::vec u_g    = u.subvec(g_id_init, g_id_end);
      arma::sp_mat F_g = F.submat(g_id_init, 0, g_id_end, n-1);

      // update z
      arma::vec z_g                 = glasso_prox(F_g * x + u_g, (spglasso_lambda * (1.0 - spglasso_alpha) * group_weights[g]) / rho); // questa è cambiata
      z.subvec(g_id_init, g_id_end) = z_g;
      g_id_init                     = g_id_end + 1;
    }
    // update z for the last group
    g_id_end                      = g_id_init + n - 1;
    arma::vec u_g                 = u.subvec(g_id_init, g_id_end);
    arma::sp_mat F_g              = F.submat(g_id_init, 0, g_id_end, n-1);
    arma::vec z_g                 = lasso_prox(F_g * x + u_g, spglasso_lambda * spglasso_alpha / rho);
    z.subvec(g_id_init, g_id_end) = z_g;

    // 4-3. update 'u'
    u += spmat2vec_mult(F, x) - z;
  
    // 4-3. dianostics, reporting
    rss         = lm_rss(A, b, x);
    pen         = spovglasso_penalty(x, groups, group_weights, var_weights, var_weights_L1, lambda2);
    h_objval(k) = 0.5 *  rss / static_cast<float>(m) + lambda1 * pen;
    h_r_norm(k) = norm(glasso_residual_sparse(F, x, z), 2);
    h_s_norm(k) = norm(glasso_dual_residual_sparse(F, z, z_old, rho), 2);
    if (norm(F * x, 2) > norm(-z, 2)) {
      h_eps_pri(k) = std::sqrt(z.n_elem) * abstol + reltol * norm(F * x, 2);
    } else {
      h_eps_pri(k) = std::sqrt(z.n_elem) * abstol + reltol * norm(-z, 2);
    }
    h_eps_dual(k) = std::sqrt(z.n_elem) * abstol + reltol * norm(F.t() * rho * u, 2);
    
    /* :::::::::::::::::::::::::::::::::::::::::::::::::::::::: */
    /* Relaxation                                               */
    rho_old = rho;
    if (rho_adaptation) {
      if (h_r_norm(k) > mu * h_s_norm(k)) {
        rho = rho * tau;
        u   = u / tau;
      } else if (h_s_norm(k) > mu * h_r_norm(k)) {
        rho = rho / tau;
        u   = u * tau;
      }
    }
    rho_store(k+1) = rho;
    
    /* ::::::::::::::::::::::::::::::::::::::::::::::
     Compute elapsed time                                */
    elTime = timer.toc();
    
    /* :::::::::::::::::::::::::::::::::::::::::::::::::::::::: */
    /* Print to screen                                          */
    if ((ping > 0) && (k < maxiter)) {
      if (rho_adaptation) {
        if (((k+1) % ping) == 0) {
          Rcpp::Rcout << "\n\n\n" << std::endl;
          Rcpp::Rcout << "ADMM with adaptation for sparse overlap group-LASSO is running!" << std::endl;
          Rcpp::Rcout << "Iteration n.: " << k+1 << " of " << maxiter << "\n" << std::endl;
        }
      } else {
        if (((k+1) % ping) == 0) {
          Rcpp::Rcout << "\n\n\n" << std::endl;
          Rcpp::Rcout << "ADMM for sparse overlap group-LASSO is running!" << std::endl;
          Rcpp::Rcout << "Iteration n.: " << k+1 << " of " << maxiter << "\n" << std::endl;
        }
      }
    }

    // 4-4. termination
    if ((h_r_norm(k) < h_eps_pri(k)) && (h_s_norm(k) < h_eps_dual(k))) {
      break;
    }
  }
  
  /* :::::::::::::::::::::::::::::::::::::::::::::::::::::::: */
  /* Print to screen                                          */
  if (ping > 0) {
    Rcpp::Rcout << "ADMM for sparse overlap group-LASSO" << std::endl;
    Rcpp::Rcout << "Iterations n.: " << k << std::endl;
    if ((k+1) < maxiter) {
      Rcpp::Rcout << "Convergence achieved!" << std::endl;
    }
  }

  /* Get output         */
  List output;
  output["x"] = x;
  output["u"] = u;
  output["z"] = z;
  if (k < maxiter) {
    output["convergence"] = 1.0;
    output["niter"]       = k+1;
    output["objval"]      = h_objval.subvec(0, k);
    output["r_norm"]      = h_r_norm.subvec(0, k);
    output["s_norm"]      = h_s_norm.subvec(0, k);
    output["eps_pri"]     = h_eps_pri.subvec(0, k);
    output["eps_dual"]    = h_eps_dual.subvec(0, k);
    output["rho"]         = rho_store.subvec(0, k+1);
  } else {
    output["convergence"] = 0.0;
    output["niter"]       = maxiter;
    output["objval"]      = h_objval;
    output["r_norm"]      = h_r_norm;
    output["s_norm"]      = h_s_norm;
    output["eps_pri"]     = h_eps_pri;
    output["eps_dual"]    = h_eps_dual;
    output["rho"]         = rho_store;
  }
  output["eltime"]   = elTime;
  if ((k+1) < maxiter) {
    output["convergence"] = 1.0;
  } else {
    output["convergence"] = 0.0;
  }

  /* Return output      */
  return(output);
}

Rcpp::List admm_spovglasso_large_n(const arma::mat& A,
                                   const arma::vec& b,
                                   const arma::mat& groups,
                                   const arma::vec& group_weights,
                                   const arma::vec& var_weights,
                                   const arma::vec& var_weights_L1,
                                   arma::colvec& u,
                                   arma::colvec& z,
                                   const double lambda1,
                                   const double lambda2,
                                   const bool rho_adaptation,
                                   double rho,
                                   const double tau,
                                   const double mu,
                                   const double reltol,
                                   const double abstol,
                                   const int maxiter,
                                   const int ping) {
    
  /* Variable delcaration           */
  //double rho2;
  int k = 0;
  uword g_id_init, g_id_end;
  double spglasso_alpha = 0.0, spglasso_lambda = 0.0, rho_old = 0.0, elTime = 0.0, rss = 0.0, pen = 0.0;
  
  /* Get dimensions             */
  const int m     = A.n_rows;
  const int n     = A.n_cols;
  const int G     = groups.n_rows;
  const int dim_z = z.n_elem;
      
  /* Definition of vectors and matrices     */
  arma::colvec x(n, fill::zeros);
  arma::colvec x_star(n, fill::zeros);
  arma::colvec q(n, fill::zeros);
  arma::colvec z_old(dim_z, fill::zeros);
  arma::mat FTF(n, n, fill::zeros);
  arma::mat FTF_INV(n, n, fill::zeros);
  arma::mat AFTF_INVA(m, m, fill::zeros);
  arma::vec ATb(n, fill::zeros);
  arma::mat U(m, m, fill::zeros);
  arma::mat L(m, m, fill::zeros);
  arma::colvec Glen(G, fill::zeros);
  arma::vec h_objval(maxiter, fill::zeros);
  arma::vec h_r_norm(maxiter, fill::zeros);
  arma::vec h_s_norm(maxiter, fill::zeros);
  arma::vec h_eps_pri(maxiter, fill::zeros);
  arma::vec h_eps_dual(maxiter, fill::zeros);
  arma::vec rho_store(maxiter+1, fill::zeros);

  /* ::::::::::::::::::::::::::::::::::::::::::::::
   Set the computational time                                 */
  wall_clock timer;
  timer.tic();
  
  /* store rho (only for adaptation)         */
  rho_store(0) = rho;
  rho_old      = rho;
  
  /* Precompute relevant quantities         */
  spglasso_lambda = lambda1;
  spglasso_alpha  = lambda2;
  ATb             = A.t() * b / static_cast<float>(m);
  Glen            = sum(groups, 1);
  arma::sp_mat F  = spovglasso_Gmat2Fmat_sparse(groups, var_weights, var_weights_L1);
  FTF             = F.t() * F;
  FTF_INV         = inv(diagmat(FTF));
  AFTF_INVA       = A * FTF_INV * A.t();
  U               = glasso_factor_fast_large_n(AFTF_INVA, rho, m);
  L               = U.t();
  
  /* main loop  */
  for (k=0; k<maxiter; k++) {
    // 4-1. update 'x'
    if (rho != rho_old) {
      U = glasso_factor_fast_large_n(AFTF_INVA, rho, m); // returns upper
      L = U.t();
    }
    q      = ATb + rho * spmat2vec_mult(F.t(), z - u); // temporary value
    x_star = (diagmat(FTF_INV) * q) / rho;
    x      = x_star - FTF_INV * A.t() * solve(trimatu(U), solve(trimatl(L), A * x_star));

    // 4-2. update 'z' with adaptation
    // see Boyd et al (2011) page 21
    z_old     = z;
    g_id_init = 0;
    for (int g=0; g<G; g++) {
      // precompute the quantities for updating z
      g_id_end         = g_id_init + Glen(g) - 1;
      arma::vec u_g    = u.subvec(g_id_init, g_id_end);
      arma::sp_mat F_g = F.submat(g_id_init, 0, g_id_end, n-1);

      // update z
      //arma::vec z_g                 = glasso_prox(F_g * x + u_g, spglasso_lambda * (1.0 - spglasso_alpha) / rho);
      arma::vec z_g                 = glasso_prox(F_g * x + u_g, (spglasso_lambda * (1.0 - spglasso_alpha) * group_weights[g]) / rho); // questa è cambiata
      z.subvec(g_id_init, g_id_end) = z_g;
      g_id_init                     = g_id_end + 1;
    }
    // update z for the last group
    g_id_end                      = g_id_init + n - 1;
    arma::vec u_g                 = u.subvec(g_id_init, g_id_end);
    arma::sp_mat F_g              = F.submat(g_id_init, 0, g_id_end, n-1);
    arma::vec z_g                 = lasso_prox(F_g * x + u_g, spglasso_lambda * spglasso_alpha / rho);
    z.subvec(g_id_init, g_id_end) = z_g;

    // 4-3. update 'u'
    u += spmat2vec_mult(F, x) - z;
  
    // 4-3. dianostics, reporting
    rss         = lm_rss(A, b, x);
    pen         = spovglasso_penalty(x, groups, group_weights, var_weights, var_weights_L1, spglasso_alpha);
    h_objval(k) = 0.5 *  rss / static_cast<float>(m) + spglasso_lambda * pen;
    h_r_norm(k) = norm(glasso_residual_sparse(F, x, z), 2);
    h_s_norm(k) = norm(glasso_dual_residual_sparse(F, z, z_old, rho), 2);
    if (norm(F * x, 2) > norm(-z, 2)) {
      h_eps_pri(k) = std::sqrt(z.n_elem) * abstol + reltol * norm(F * x, 2);
    } else {
      h_eps_pri(k) = std::sqrt(z.n_elem) * abstol + reltol * norm(-z, 2);
    }
    h_eps_dual(k) = std::sqrt(z.n_elem) * abstol + reltol * norm(F.t() * rho * u, 2);
    
    /* :::::::::::::::::::::::::::::::::::::::::::::::::::::::: */
    /* Relaxation                                               */
    rho_old = rho;
    if (rho_adaptation) {
      if (h_r_norm(k) > mu * h_s_norm(k)) {
        rho = rho * tau;
        u   = u / tau;
      } else if (h_s_norm(k) > mu * h_r_norm(k)) {
        rho = rho / tau;
        u   = u * tau;
      }
    }
    rho_store(k+1) = rho;
    
    /* ::::::::::::::::::::::::::::::::::::::::::::::
     Compute elapsed time                                */
    elTime = timer.toc();
    
    /* :::::::::::::::::::::::::::::::::::::::::::::::::::::::: */
    /* Print to screen                                          */
    if ((ping > 0) && (k < maxiter)) {
      if (rho_adaptation) {
        if (((k+1) % ping) == 0) {
          Rcpp::Rcout << "\n\n\n" << std::endl;
          Rcpp::Rcout << "ADMM with adaptation for sparse overlap group-LASSO is running!" << std::endl;
          Rcpp::Rcout << "Iteration n.: " << k+1 << " of " << maxiter << "\n" << std::endl;
        }
      } else {
        if (((k+1) % ping) == 0) {
          Rcpp::Rcout << "\n\n\n" << std::endl;
          Rcpp::Rcout << "ADMM for sparse overlap group-LASSO is running!" << std::endl;
          Rcpp::Rcout << "Iteration n.: " << k+1 << " of " << maxiter << "\n" << std::endl;
        }
      }
    }

    // 4-4. termination
    if ((h_r_norm(k) < h_eps_pri(k)) && (h_s_norm(k) < h_eps_dual(k))) {
      break;
    }
  }
  
  /* :::::::::::::::::::::::::::::::::::::::::::::::::::::::: */
  /* Print to screen                                          */
  if (ping > 0) {
    Rcpp::Rcout << "ADMM for sparse overlap group-LASSO" << std::endl;
    Rcpp::Rcout << "Iterations n.: " << k << std::endl;
    if ((k+1) < maxiter) {
      Rcpp::Rcout << "Convergence achieved!" << std::endl;
    }
  }

  /* Get output         */
  List output;
  output["x"] = x;
  output["u"] = u;
  output["z"] = z;
  if (k < maxiter) {
    output["convergence"] = 1.0;
    output["niter"]       = k+1;
    output["objval"]      = h_objval.subvec(0, k);
    output["r_norm"]      = h_r_norm.subvec(0, k);
    output["s_norm"]      = h_s_norm.subvec(0, k);
    output["eps_pri"]     = h_eps_pri.subvec(0, k);
    output["eps_dual"]    = h_eps_dual.subvec(0, k);
    output["rho"]         = rho_store.subvec(0, k+1);
  } else {
    output["convergence"] = 0.0;
    output["niter"]       = maxiter;
    output["objval"]      = h_objval;
    output["r_norm"]      = h_r_norm;
    output["s_norm"]      = h_s_norm;
    output["eps_pri"]     = h_eps_pri;
    output["eps_dual"]    = h_eps_dual;
    output["rho"]         = rho_store;
  }
  output["eltime"]   = elTime;
  if ((k+1) < maxiter) {
    output["convergence"] = 1.0;
  } else {
    output["convergence"] = 0.0;
  }

  /* Return output      */
  return(output);
}

Rcpp::List admm_spovglasso_cov_large_m(const arma::mat& W,
                                       const arma::mat& Z,
                                       const arma::vec& y,
                                       arma::colvec& u,
                                       arma::colvec& z,
                                       const arma::mat& groups,
                                       const arma::vec& group_weights,
                                       const arma::vec& var_weights,
                                       const arma::vec& var_weights_L1,
                                       const double lambda,
                                       const double alpha,
                                       const bool rho_adaptation,
                                       double rho,
                                       const double tau,
                                       const double mu,
                                       const double reltol,
                                       const double abstol,
                                       const int maxiter,
                                       const int ping) {

  /* Variable delcaration           */
  double y2 = 0.0, sqrtm = 0.0, rho_old = 0.0, elTime = 0.0, rss = 0.0, pen = 0.0;
  int n1 = 0, dim_z = 0, k = 0;
  uword g_id_init, g_id_end;

  /* Get dimensions             */
  const int m       = W.n_rows;
  const int n_spcov = W.n_cols;
  const int n_cov   = Z.n_cols;
  const int G       = groups.n_rows;
  
  /* Variable definition            */
  //sqrtn = std::sqrt(static_cast<float>(n_spcov));
  sqrtm = std::sqrt(static_cast<float>(m));

  /* Get output lists          */
  List out, output;

  /* Variable definition            */
  if (m >= n_spcov) {
    n1 = n_spcov;
  } else {
    n1 = m;
  }

  /* Definition of vectors and matrices     */
  arma::colvec x(n_spcov, fill::zeros);
  arma::colvec v(n_cov, fill::zeros);
  arma::mat W_(m, n_spcov, fill::zeros);
  arma::mat Z_(m, n_cov, fill::zeros);
  arma::colvec q(n_spcov, fill::zeros);
  arma::mat U(n_spcov, n_spcov, fill::zeros);
  arma::mat L(n_spcov, n_spcov, fill::zeros);
  arma::mat WW_CHOL_U(n1, n_spcov, fill::zeros);
  arma::mat ZZ_CHOL_U(n_cov, n_cov, fill::zeros);
  arma::vec Wy(n_spcov, fill::zeros);
  arma::vec ones_m(m, fill::ones);
  arma::mat eye_m(m, m, fill::eye);
  arma::mat ZZ(n_cov, n_cov, fill::zeros);
  arma::mat L_ZZ(n_cov, n_cov, fill::zeros);
  arma::mat L_ZZ_INV(n_cov, n_cov, fill::zeros);
  arma::mat ZZ_INV(n_cov, n_cov, fill::zeros);
  arma::mat P_ZZ(m, m, fill::zeros);
  arma::mat M_ZZ(m, m, fill::zeros);
  arma::mat WMW(n_spcov, n_spcov, fill::zeros);
  arma::vec WMy(n_spcov, fill::zeros);
  arma::vec Zy(n_cov, fill::zeros);
  arma::vec v_LS(n_cov, fill::zeros);
  arma::mat ZW(n_cov, n_spcov, fill::zeros);
  arma::mat P_ZW(n_cov, n_spcov, fill::zeros);
  arma::mat FTF(n_spcov, n_spcov, fill::zeros);
  arma::colvec Glen(G, fill::zeros);
  arma::vec rho_store(maxiter+1, fill::zeros);
  arma::vec h_objval(maxiter, fill::zeros);
  arma::vec h_r_norm(maxiter, fill::zeros);
  arma::vec h_s_norm(maxiter, fill::zeros);
  arma::vec h_eps_pri(maxiter, fill::zeros);
  arma::vec h_eps_dual(maxiter, fill::zeros);
  
  /* Precompute relevant quantities         */
  Z_             = Z.each_col() % (ones_m * (1.0 / sqrtm));                     // 19Nov15: divisione per n controllata e corretta
  W_             = W.each_col() % (ones_m * (1.0 / sqrtm));
  ZZ             = Z_.t() * Z_;
  L_ZZ           = (chol(ZZ)).t();
  L_ZZ_INV       = inv(trimatl(L_ZZ));
  ZZ_INV         = L_ZZ_INV.t() * L_ZZ_INV;
  P_ZZ           = Z_ * ZZ_INV * Z_.t();
  M_ZZ           = eye_m - P_ZZ;
  WMW            = W_.t() * M_ZZ * W_;
  WMy            = W_.t() * M_ZZ * y / static_cast<float>(sqrtm);
  Zy             = Z_.t() * y / static_cast<float>(sqrtm);
  v_LS           = L_ZZ_INV.t() * L_ZZ_INV * Zy;
  ZW             = Z_.t() * W_;
  P_ZW           = L_ZZ_INV.t() * L_ZZ_INV * ZW;
  WW_CHOL_U      = chol_qr_fact(W_, m, n_spcov);
  ZZ_CHOL_U      = chol_qr_fact(Z_, m, n_cov);
  y2             = as_scalar(y.t() * y) / static_cast<float>(m);
  Wy             = W_.t() * y / static_cast<float>(sqrtm);
  Glen           = sum(groups, 1);
  arma::sp_mat F = spovglasso_Gmat2Fmat_sparse(groups, var_weights, var_weights_L1);
  FTF            = diagmat(spmat2spmat_mult(F.t(), F));
  
  /* Initialize vectors z_old         */
  dim_z = n_spcov + sum(Glen);
  arma::colvec z_old(dim_z, fill::zeros);

  /* ::::::::::::::::::::::::::::::::::::::::::::::
   Set the computational time                                 */
  wall_clock timer;
  timer.tic();
  
  /* store rho (only for adaptation)         */
  rho_store(0) = rho;
  rho_old      = rho;

  /* Precompute relevant quantities         */
  //U = glasso_factor_fast_large_m(ATA, FTF, rho); // returns upper
  U = glasso_factor_cov_fast_large_m(WMW, FTF, rho);
  L = U.t();
  
  /* main loop  */
  for (k=0; k<maxiter; k++) {

    // 4-1. update 'x'
    if (rho != rho_old) {
      //U = glasso_factor_fast_large_m(ATA, FTF, rho); // returns upper
      U = glasso_factor_cov_fast_large_m(WMW, FTF, rho);
      L = U.t();
    }
    q = WMy + rho * spmat2vec_mult(F.t(), z - u);        // temporary value
    x = solve(trimatu(U), solve(trimatl(L), q));         // updated of regression parameters associated to covariates W (sparsified covariates, x)
    v = v_LS - P_ZW * x;                                 // updated of regression parameters associated to covariates Z (covariates not sparsified, v)
    
    // 4-2. update 'z' with adaptation
    // see Boyd et al (2011) page 21
    z_old     = z;
    g_id_init = 0;
    for (int g=0; g<G; g++) {
      // precompute the quantities for updating z
      g_id_end         = g_id_init + Glen(g) - 1;
      arma::vec u_g    = u.subvec(g_id_init, g_id_end);
      arma::sp_mat F_g = F.submat(g_id_init, 0, g_id_end, n_spcov-1);

      // update z
      arma::vec z_g                 = glasso_prox(F_g * x + u_g, (lambda * group_weights[g]) / rho);
      z.subvec(g_id_init, g_id_end) = z_g;
      g_id_init                     = g_id_end + 1;
    }
    // update z for the last group
    g_id_end                      = g_id_init + n_spcov - 1;
    arma::vec u_g                 = u.subvec(g_id_init, g_id_end);
    arma::sp_mat F_g              = F.submat(g_id_init, 0, g_id_end, n_spcov-1);
    arma::vec z_g                 = lasso_prox(F_g * x + u_g, lambda * alpha / rho);
    z.subvec(g_id_init, g_id_end) = z_g;

    // 4-3. update 'u'
    u += spmat2vec_mult(F, x) - z;
  
    // 4-3. dianostics, reporting
    rss         = lm_cov_rss_fast(WW_CHOL_U, ZZ_CHOL_U, ZW, Wy, Zy, y2, x, v, m);
    pen         = spovglasso_penalty(x, groups, group_weights, var_weights, var_weights_L1, alpha);
    h_objval(k) = 0.5 * rss / static_cast<float>(m) + lambda * pen;
    h_r_norm(k) = norm(glasso_residual_sparse(F, x, z), 2);
    h_s_norm(k) = norm(glasso_dual_residual_sparse(F, z, z_old, rho), 2);
    if (norm(F * x, 2) > norm(-z, 2)) {
      h_eps_pri(k) = std::sqrt(z.n_elem) * abstol + reltol * norm(F * x, 2);
    } else {
      h_eps_pri(k) = std::sqrt(z.n_elem) * abstol + reltol * norm(-z, 2);
    }
    h_eps_dual(k) = std::sqrt(z.n_elem) * abstol + reltol * norm(F.t() * rho * u, 2);
    
    /* :::::::::::::::::::::::::::::::::::::::::::::::::::::::: */
    /* Relaxation                                               */
    rho_old = rho;
    if (rho_adaptation) {
      if (h_r_norm(k) > mu * h_s_norm(k)) {
        rho = rho * tau;
        u   = u / tau;
      } else if (h_s_norm(k) > mu * h_r_norm(k)) {
        rho = rho / tau;
        u   = u * tau;
      }
    }
    rho_store(k+1) = rho;
        
    /* :::::::::::::::::::::::::::::::::::::::::::::::::::::::: */
    /* Print to screen                                          */
    if ((ping > 0) && (k < maxiter)) {
      if (rho_adaptation) {
        if (((k+1) % ping) == 0) {
          Rcpp::Rcout << "\n\n\n" << std::endl;
          Rcpp::Rcout << "ADMM with adaptation for sparse overlap group-LASSO is running!" << std::endl;
          Rcpp::Rcout << "Iteration n.: " << k+1 << " of " << maxiter << "\n" << std::endl;
        }
      } else {
        if (((k+1) % ping) == 0) {
          Rcpp::Rcout << "\n\n\n" << std::endl;
          Rcpp::Rcout << "ADMM for sparse overlap group-LASSO is running!" << std::endl;
          Rcpp::Rcout << "Iteration n.: " << k+1 << " of " << maxiter << "\n" << std::endl;
        }
      }
    }

    // 4-4. termination
    if ((h_r_norm(k) < h_eps_pri(k)) && (h_s_norm(k) < h_eps_dual(k))) {
      break;
    }
  }

  /* ::::::::::::::::::::::::::::::::::::::::::::::
   Compute elapsed time                                */
  elTime = timer.toc();
  
  /* Get output         */
  output["x"]      = x;
  output["v"]      = v;
  output["u"]      = u;
  output["z"]      = z;
  output["eltime"] = elTime;
  if (k < maxiter) {
    output["convergence"] = 1.0;
    output["niter"]       = k+1;
    output["objval"]      = h_objval.subvec(0, k);
    output["r_norm"]      = h_r_norm.subvec(0, k);
    output["s_norm"]      = h_s_norm.subvec(0, k);
    output["eps_pri"]     = h_eps_pri.subvec(0, k);
    output["eps_dual"]    = h_eps_dual.subvec(0, k);
    output["rho"]         = rho_store.subvec(0, k+1);
  } else {
    output["convergence"] = 0.0;
    output["niter"]       = maxiter;
    output["objval"]      = h_objval;
    output["r_norm"]      = h_r_norm;
    output["s_norm"]      = h_s_norm;
    output["eps_pri"]     = h_eps_pri;
    output["eps_dual"]    = h_eps_dual;
    output["rho"]         = rho_store;
  }
  
  /* Return output      */
  return(output);
}

Rcpp::List admm_spovglasso_cov_large_n(const arma::mat& W,
                                       const arma::mat& Z,
                                       const arma::vec& y,
                                       arma::colvec& u,
                                       arma::colvec& z,
                                       const arma::mat& groups,
                                       const arma::vec& group_weights,
                                       const arma::vec& var_weights,
                                       const arma::vec& var_weights_L1,
                                       const double lambda,
                                       const double alpha,
                                       const bool rho_adaptation,
                                       double rho,
                                       const double tau,
                                       const double mu,
                                       const double reltol,
                                       const double abstol,
                                       const int maxiter,
                                       const int ping) {

  /* Variable delcaration           */
  double y2 = 0.0, sqrtm = 0.0, rho_old = 0.0, elTime = 0.0, rss = 0.0, pen = 0.0;
  int n1 = 0, dim_z = 0, k = 0;
  uword g_id_init, g_id_end;

  /* Get dimensions             */
  const int m       = W.n_rows;
  const int n_spcov = W.n_cols;
  const int n_cov   = Z.n_cols;
  const int G = groups.n_rows;
  
  /* Variable definition            */
  //sqrtn = std::sqrt(static_cast<float>(n_spcov));
  sqrtm = std::sqrt(static_cast<float>(m));

  /* Get output lists          */
  List out, output;

  /* Variable definition            */
  if (m >= n_spcov) {
    n1 = n_spcov;
  } else {
    n1 = m;
  }

  /* Definition of vectors and matrices     */
  arma::colvec x(n_spcov, fill::zeros);
  arma::colvec x_star(n_spcov, fill::zeros);
  arma::mat Z_(m, n_cov, fill::zeros);
  arma::colvec v(n_cov, fill::zeros);
  arma::colvec q(n_spcov, fill::zeros);
  arma::mat U(m, m, fill::zeros);
  arma::mat L(m, m, fill::zeros);
  arma::mat WW_CHOL_U(n1, n_spcov, fill::zeros);
  arma::mat ZZ_CHOL_U(n_cov, n_cov, fill::zeros);
  arma::vec Wy(n_spcov, fill::zeros);
  arma::vec ones_m(m, fill::ones);
  arma::mat eye_m(m, m, fill::eye);
  arma::mat ZZ(n_cov, n_cov, fill::zeros);
  arma::mat L_ZZ(n_cov, n_cov, fill::zeros);
  arma::mat L_ZZ_INV(n_cov, n_cov, fill::zeros);
  arma::mat ZZ_INV(n_cov, n_cov, fill::zeros);
  arma::mat P_ZZ(m, m, fill::zeros);
  arma::mat M_ZZ(m, m, fill::zeros);
  arma::mat WMW(n_spcov, n_spcov, fill::zeros);
  arma::vec WMy(n_spcov, fill::zeros);
  arma::vec Zy(n_cov, fill::zeros);
  arma::vec v_LS(n_cov, fill::zeros);
  arma::mat ZW(n_cov, n_spcov, fill::zeros);
  arma::mat P_ZW(n_cov, n_spcov, fill::zeros);
  arma::mat M_ZZW(m, n_spcov, fill::zeros);
  arma::mat M_ZZW2(m, m, fill::zeros);
  arma::mat FTF(n_spcov, n_spcov, fill::zeros);
  arma::mat FTF_INV(n_spcov, n_spcov, fill::zeros);
  arma::colvec Glen(G, fill::zeros);
  arma::vec rho_store(maxiter+1, fill::zeros);
  arma::vec h_objval(maxiter, fill::zeros);
  arma::vec h_r_norm(maxiter, fill::zeros);
  arma::vec h_s_norm(maxiter, fill::zeros);
  arma::vec h_eps_pri(maxiter, fill::zeros);
  arma::vec h_eps_dual(maxiter, fill::zeros);
  
  /* Precompute relevant quantities         */
  Z_        = Z.each_col() % (ones_m * (1.0 / sqrtm));
  //W         = W.each_col() % (ones_m * (1.0 / sqrtm));        // è corretto non dividere W per la numerosità campionaria m
  ZZ        = Z_.t() * Z_;
  L_ZZ      = (chol(ZZ)).t();
  L_ZZ_INV  = inv(trimatl(L_ZZ));
  ZZ_INV    = L_ZZ_INV.t() * L_ZZ_INV;
  P_ZZ      = Z_ * ZZ_INV * Z_.t();
  M_ZZ      = eye_m - P_ZZ;
  WMW       = W.t() * M_ZZ * W;                               // 19Nov15: divisione per n controllata e corretta così come è ora
  WMy       = W.t() * M_ZZ * y;
  Zy        = Z_.t() * y / static_cast<float>(sqrtm);
  v_LS      = L_ZZ_INV.t() * L_ZZ_INV * Zy;
  ZW        = Z_.t() * W / static_cast<float>(sqrtm);
  P_ZW      = L_ZZ_INV.t() * L_ZZ_INV * ZW;
  WW_CHOL_U = chol_qr_fact(W, m, n_spcov);
  ZZ_CHOL_U = chol_qr_fact(Z_, m, n_cov);
  y2        = as_scalar(y.t() * y) / static_cast<float>(m);
  Wy        = W.t() * y;
  M_ZZW     = M_ZZ * W;                                       // 19Nov15: divisione per n controllata e corretta così come è ora
  
  Glen           = sum(groups, 1);
  arma::sp_mat F = spovglasso_Gmat2Fmat_sparse(groups, var_weights, var_weights_L1);
  FTF            = diagmat(spmat2spmat_mult(F.t(), F));
  FTF_INV        = diagmat(1.0 / diagvec(FTF));
  M_ZZW2         = M_ZZW * diagmat(FTF_INV) * M_ZZW.t();
  
  /* Initialize vectors z_old and x_star         */
  dim_z = n_spcov + sum(Glen);
  arma::colvec z_old(dim_z, fill::zeros);

  /* ::::::::::::::::::::::::::::::::::::::::::::::
   Set the computational time                                 */
  wall_clock timer;
  timer.tic();
  
  /* store rho (only for adaptation)         */
  rho_store(0) = rho;
  rho_old      = rho;

  /* Precompute relevant quantities         */
  U = glasso_factor_cov_fast_large_n(M_ZZW2, rho, m); // returns upper
  L = U.t();
  
  /* main loop  */
  for (k=0; k<maxiter; k++) {

    // 4-1. update 'x'
    if (rho != rho_old) {
      U = glasso_factor_cov_fast_large_n(M_ZZW2, rho, m); // returns upper
      L = U.t();
    }
    q      = WMy + rho * spmat2vec_mult(F.t(), z - u); // temporary value
    x_star = (diagmat(FTF_INV) * q) / rho;
    x      = x_star - diagmat(FTF_INV) * M_ZZW.t() * solve(trimatu(U), solve(trimatl(L), M_ZZW * x_star)); // updated of regression parameters associated to covariates W
    v      = v_LS - P_ZW * x;                                 // updated of regression parameters associated to covariates Z (covariates not sparsified, v)

    // 4-2. update 'z' with adaptation
    // see Boyd et al (2011) page 21
    z_old     = z;
    g_id_init = 0;
    for (int g=0; g<G; g++) {
      // precompute the quantities for updating z
      g_id_end         = g_id_init + Glen(g) - 1;
      arma::vec u_g    = u.subvec(g_id_init, g_id_end);
      arma::sp_mat F_g = F.submat(g_id_init, 0, g_id_end, n_spcov-1);

      // update z
      arma::vec z_g                 = glasso_prox(F_g * x + u_g, (lambda * group_weights[g]) / rho);
      z.subvec(g_id_init, g_id_end) = z_g;
      g_id_init                     = g_id_end + 1;
    }

    // 4-3. update 'u'
    u += spmat2vec_mult(F, x) - z;
  
    // 4-3. dianostics, reporting
    rss         = lm_cov_rss_fast(WW_CHOL_U, ZZ_CHOL_U, ZW, Wy, Zy, y2, x, v, m);
    pen         = spovglasso_penalty(x, groups, group_weights, var_weights, var_weights_L1, alpha);
    h_objval(k) = 0.5 * rss / static_cast<float>(m) + lambda * pen;
    h_r_norm(k) = norm(glasso_residual_sparse(F, x, z), 2);
    h_s_norm(k) = norm(glasso_dual_residual_sparse(F, z, z_old, rho), 2);
    if (norm(F * x, 2) > norm(-z, 2)) {
      h_eps_pri(k) = std::sqrt(z.n_elem) * abstol + reltol * norm(F * x, 2);
    } else {
      h_eps_pri(k) = std::sqrt(z.n_elem) * abstol + reltol * norm(-z, 2);
    }
    h_eps_dual(k) = std::sqrt(z.n_elem) * abstol + reltol * norm(F.t() * rho * u, 2);
    
    /* :::::::::::::::::::::::::::::::::::::::::::::::::::::::: */
    /* Relaxation                                               */
    rho_old = rho;
    if (rho_adaptation) {
      if (h_r_norm(k) > mu * h_s_norm(k)) {
        rho = rho * tau;
        u   = u / tau;
      } else if (h_s_norm(k) > mu * h_r_norm(k)) {
        rho = rho / tau;
        u   = u * tau;
      }
    }
    rho_store(k+1) = rho;
        
    /* :::::::::::::::::::::::::::::::::::::::::::::::::::::::: */
    /* Print to screen                                          */
    if ((ping > 0) && (k < maxiter)) {
      if (rho_adaptation) {
        if (((k+1) % ping) == 0) {
          Rcpp::Rcout << "\n\n\n" << std::endl;
          Rcpp::Rcout << "ADMM with adaptation for sparse overlap group-LASSO is running!" << std::endl;
          Rcpp::Rcout << "Iteration n.: " << k+1 << " of " << maxiter << "\n" << std::endl;
        }
      } else {
        if (((k+1) % ping) == 0) {
          Rcpp::Rcout << "\n\n\n" << std::endl;
          Rcpp::Rcout << "ADMM for sparse overlap group-LASSO is running!" << std::endl;
          Rcpp::Rcout << "Iteration n.: " << k+1 << " of " << maxiter << "\n" << std::endl;
        }
      }
    }

    // 4-4. termination
    if ((h_r_norm(k) < h_eps_pri(k)) && (h_s_norm(k) < h_eps_dual(k))) {
      break;
    }
  }

  /* ::::::::::::::::::::::::::::::::::::::::::::::
   Compute elapsed time                                */
  elTime = timer.toc();
  
  /* Get output         */
  output["x"]      = x;
  output["v"]      = v;
  output["u"]      = u;
  output["z"]      = z;
  output["eltime"] = elTime;
  if (k < maxiter) {
    output["convergence"] = 1.0;
    output["niter"]       = k+1;
    output["objval"]      = h_objval.subvec(0, k);
    output["r_norm"]      = h_r_norm.subvec(0, k);
    output["s_norm"]      = h_s_norm.subvec(0, k);
    output["eps_pri"]     = h_eps_pri.subvec(0, k);
    output["eps_dual"]    = h_eps_dual.subvec(0, k);
    output["rho"]         = rho_store.subvec(0, k+1);
  } else {
    output["convergence"] = 0.0;
    output["niter"]       = maxiter;
    output["objval"]      = h_objval;
    output["r_norm"]      = h_r_norm;
    output["s_norm"]      = h_s_norm;
    output["eps_pri"]     = h_eps_pri;
    output["eps_dual"]    = h_eps_dual;
    output["rho"]         = rho_store;
  }

  /* Return output      */
  return(output);
}

/*
    Sparse Overlap Group-LASSO via ADMM: main functions, to be used
                                  to fit the model several times
                                  (multiple lambdas)                  */

Rcpp::List admm_spovglasso_large_m_fast(const arma::mat& A,
                                        const arma::vec& b,
                                        const arma::mat& groups,
                                        const arma::vec& group_weights,
                                        const arma::vec& var_weights,
                                        const arma::vec& var_weights_L1,
                                        const arma::vec& lambda,
                                        const double alpha,
                                        const bool rho_adaptation,
                                        double rho,
                                        const double tau,
                                        const double mu,
                                        const double reltol,
                                        const double abstol,
                                        const int maxiter,
                                        const int ping) {
    
  /* Variable delcaration           */
  //double rho2;
  double b2 = 0.0, min_mse = 0.0;
  int n1 = 0, dim_z = 0;
  uword idx_min_mse;
  
  /* Get dimensions             */
  const int m = A.n_rows;
  const int n = A.n_cols;
  const int G = groups.n_rows;
  
  /* Get number of lambda           */
  const int nlambda = lambda.n_elem;

  /* Get output lists          */
  List out, output;
  
  /* Variable definition            */
  if (m >= n) {
    n1 = n;
  } else {
    n1 = m;
  }

  /* Definition of vectors and matrices     */
  arma::mat ATA(n, n, fill::zeros);
  arma::vec ATb(n, fill::zeros);
  arma::mat ATA_CHOL_U(n1, n, fill::zeros);
  arma::mat FTF(n, n, fill::zeros);
  arma::colvec x(n, fill::zeros);
  arma::colvec Glen(G, fill::zeros);

  /* Store output    */
  arma::field<vec> h_objval(nlambda);
  arma::field<vec> h_r_norm(nlambda);
  arma::field<vec> h_s_norm(nlambda);
  arma::field<vec> h_eps_pri(nlambda);
  arma::field<vec> h_eps_dual(nlambda);
  arma::field<vec> h_rho(nlambda);
  arma::vec niter(nlambda, fill::zeros);
  arma::vec eltime(nlambda, fill::zeros);
  arma::mat X(nlambda, n, fill::zeros);
  arma::vec conv(nlambda, fill::zeros);
  arma::mat residuals(nlambda, m, fill::zeros);
  arma::mat fitted(nlambda, m, fill::zeros);
  arma::vec mse(nlambda, fill::zeros);

  /* Precompute relevant quantities         */
  ATA            = A.t() * A / static_cast<float>(m);
  ATb            = A.t() * b / static_cast<float>(m);
  ATA_CHOL_U     = chol_qr_fact(A, m, n);
  b2             = as_scalar(b.t() * b) / static_cast<float>(m);
  Glen           = sum(groups, 1);
  arma::sp_mat F = spglasso_Gmat2Fmat_sparse(groups, var_weights, var_weights_L1);
  FTF            = diagmat(spmat2spmat_mult(F.t(), F));
  
  /* Initialize vectors u and z         */
  dim_z = n + sum(Glen);
  arma::colvec u(dim_z, fill::zeros);
  arma::colvec z(dim_z, fill::zeros);
  arma::mat Umat(nlambda, dim_z, fill::zeros);
  arma::mat Zmat(nlambda, dim_z, fill::zeros);

  /* main loop  */
  for (int j=0; j<nlambda; j++) {
    /* perform linear regression with lasso penalty for the j-th lambda       */
    out = admm_spovglasso_large_m_update_1lambda(A, b, groups, group_weights, var_weights, var_weights_L1,
                                                 F, FTF, ATA, ATA_CHOL_U, ATb, b2, m, n, G,
                                                 u, z, lambda(j), alpha, rho_adaptation, rho, tau, mu,
                                                 reltol, abstol, maxiter, ping);

    /* Retrieve output      */
    u = as<arma::colvec>(out["u"]);
    z = as<arma::colvec>(out["z"]);
    x = as<arma::colvec>(out["x"]);

    /* Store output      */
    niter(j)         = out["niter"];
    eltime(j)        = out["eltime"];
    conv(j)          = out["convergence"];
    X.row(j)         = x.t();
    h_objval(j)      = as<arma::vec>(out["objval"]);
    h_r_norm(j)      = as<arma::vec>(out["r_norm"]);
    h_s_norm(j)      = as<arma::vec>(out["s_norm"]);
    h_eps_pri(j)     = as<arma::vec>(out["eps_pri"]);
    h_eps_dual(j)    = as<arma::vec>(out["eps_dual"]);
    h_rho(j)         = (as<arma::vec>(out["rho"])).subvec(0, niter(j)-1);
    rho              = out["rho.updated"];
    residuals.row(j) = (as<arma::vec>(out["residuals"])).t();
    fitted.row(j)    = (as<arma::vec>(out["fitted.values"])).t();
    mse(j)           = out["mse"];
    Umat.row(j)      = u.t();
    Zmat.row(j)      = z.t();
  }

  /* Find min MSE in-sample      */
  idx_min_mse = index_min(mse);
  min_mse     = mse(idx_min_mse);

  /* Get output         */
  output["coef.path"]    = X;
  output["iternum"]      = niter;
  output["objfun"]       = h_objval;
  output["r_norm"]       = h_r_norm;
  output["s_norm"]       = h_s_norm;
  output["err_pri"]      = h_eps_pri;
  output["err_dual"]     = h_eps_dual;
  output["convergence"]  = conv;
  output["elapsedTime"]  = eltime;
  output["rho"]          = h_rho;
  output["lambda"]       = lambda;
  output["min.mse"]      = min_mse;
  output["indi.min.mse"] = idx_min_mse;
  output["lambda.min"]   = lambda[idx_min_mse];
  output["coefficients"] = (X.row(idx_min_mse)).t();
  output["mse"]          = mse;
  output["U"]            = Umat;
  output["Z"]            = Zmat;

  /* Return output      */
  return(output);
}

Rcpp::List admm_spovglasso_large_n_fast(const arma::mat& A,
                                        const arma::vec& b,
                                        const arma::mat& groups,
                                        const arma::vec& group_weights,
                                        const arma::vec& var_weights,
                                        const arma::vec& var_weights_L1,
                                        const arma::vec& lambda,
                                        const double alpha,
                                        const bool rho_adaptation,
                                        double rho,
                                        const double tau,
                                        const double mu,
                                        const double reltol,
                                        const double abstol,
                                        const int maxiter,
                                        const int ping) {

  /* Variable delcaration           */
  double b2 = 0.0, min_mse = 0.0;
  int n1 = 0, dim_z = 0;
  uword idx_min_mse;

  /* Get dimensions             */
  const int m = A.n_rows;
  const int n = A.n_cols;
  const int G = groups.n_rows;

  /* Get number of lambda           */
  const int nlambda = lambda.n_elem;

  /* Get output lists          */
  List out, output;

  /* Variable definition            */
  if (m >= n) {
    n1 = n;
  } else {
    n1 = m;
  }

  /* Definition of vectors and matrices     */
  arma::vec ATb(n, fill::zeros);
  arma::mat ATA_CHOL_U(n1, n, fill::zeros);
  arma::mat FTF(n, n, fill::zeros);
  arma::colvec x(n, fill::zeros);
  arma::colvec Glen(G, fill::zeros);
  arma::mat FTF_INV(n, n, fill::zeros);
  arma::mat AFTF_INVA(m, m, fill::zeros);

  /* Store output    */
  arma::field<vec> h_objval(nlambda);
  arma::field<vec> h_r_norm(nlambda);
  arma::field<vec> h_s_norm(nlambda);
  arma::field<vec> h_eps_pri(nlambda);
  arma::field<vec> h_eps_dual(nlambda);
  arma::field<vec> h_rho(nlambda);
  arma::vec niter(nlambda, fill::zeros);
  arma::vec eltime(nlambda, fill::zeros);
  arma::mat X(nlambda, n, fill::zeros);
  arma::vec conv(nlambda, fill::zeros);
  arma::mat residuals(nlambda, m, fill::zeros);
  arma::mat fitted(nlambda, m, fill::zeros);
  arma::vec mse(nlambda, fill::zeros);

  /* Precompute relevant quantities         */
  ATb            = A.t() * b / static_cast<float>(m);
  ATA_CHOL_U     = chol_qr_fact(A, m, n);
  b2             = as_scalar(b.t() * b) / static_cast<float>(m);
  Glen           = sum(groups, 1);
  arma::sp_mat F = spglasso_Gmat2Fmat_sparse(groups, var_weights, var_weights_L1);
  FTF            = diagmat(spmat2spmat_mult(F.t(), F));
  FTF_INV        = diagmat(1.0 / diagvec(FTF));
  AFTF_INVA      = A * diagmat(FTF_INV) * A.t();
  
  /* Initialize vectors u and z         */
  dim_z = n + sum(Glen);
  arma::colvec u(dim_z, fill::zeros);
  arma::colvec z(dim_z, fill::zeros);
  arma::mat Umat(nlambda, dim_z, fill::zeros);
  arma::mat Zmat(nlambda, dim_z, fill::zeros);

  /* main loop  */
  for (int j=0; j<nlambda; j++) {
    /* perform linear regression with lasso penalty for the j-th lambda       */
    out = admm_spovglasso_large_n_update_1lambda(A, b, groups, group_weights,
                                                 var_weights, var_weights_L1,
                                                 F, FTF_INV, AFTF_INVA,
                                                 ATA_CHOL_U, ATb, b2, m, n, G, u, z,
                                                 lambda(j), alpha,
                                                 rho_adaptation,  rho,  tau,  mu,
                                                 reltol,  abstol,  maxiter,  ping);

    /* Retrieve output      */
    u = as<arma::colvec>(out["u"]);
    z = as<arma::colvec>(out["z"]);
    x = as<arma::colvec>(out["x"]);

    /* Store output      */
    niter(j)         = out["niter"];
    eltime(j)        = out["eltime"];
    conv(j)          = out["convergence"];
    X.row(j)         = x.t();
    h_objval(j)      = as<arma::vec>(out["objval"]);
    h_r_norm(j)      = as<arma::vec>(out["r_norm"]);
    h_s_norm(j)      = as<arma::vec>(out["s_norm"]);
    h_eps_pri(j)     = as<arma::vec>(out["eps_pri"]);
    h_eps_dual(j)    = as<arma::vec>(out["eps_dual"]);
    h_rho(j)         = (as<arma::vec>(out["rho"])).subvec(0, niter(j)-1);
    rho              = out["rho.updated"];
    residuals.row(j) = (as<arma::vec>(out["residuals"])).t();
    fitted.row(j)    = (as<arma::vec>(out["fitted.values"])).t();
    mse(j)           = out["mse"];
    Umat.row(j)      = u.t();
    Zmat.row(j)      = z.t();
  }

  /* Find min MSE in-sample      */
  idx_min_mse = index_min(mse);
  min_mse     = mse(idx_min_mse);

  /* Get output         */
  output["coef.path"]    = X;
  output["iternum"]      = niter;
  output["objfun"]       = h_objval;
  output["r_norm"]       = h_r_norm;
  output["s_norm"]       = h_s_norm;
  output["err_pri"]      = h_eps_pri;
  output["err_dual"]     = h_eps_dual;
  output["convergence"]  = conv;
  output["elapsedTime"]  = eltime;
  output["rho"]          = h_rho;
  output["lambda"]       = lambda;
  output["min.mse"]      = min_mse;
  output["indi.min.mse"] = idx_min_mse;
  output["lambda.min"]   = lambda[idx_min_mse];
  output["coefficients"] = (X.row(idx_min_mse)).t();
  output["mse"]          = mse;
  output["U"]            = Umat;
  output["Z"]            = Zmat;

  /* Return output      */
  return(output);
}

Rcpp::List admm_spovglasso_cov_large_m_fast(const arma::mat& W,
                                            const arma::mat& Z,
                                            const arma::vec& y,
                                            const arma::mat& groups,
                                            const arma::vec& group_weights,
                                            const arma::vec& var_weights,
                                            const arma::vec& var_weights_L1,
                                            const arma::vec& lambda,
                                            const double alpha,
                                            const bool rho_adaptation,
                                            double rho,
                                            const double tau,
                                            const double mu,
                                            const double reltol,
                                            const double abstol,
                                            const int maxiter,
                                            const int ping) {

  /* Variable delcaration           */
  double y2 = 0.0, min_mse = 0.0, sqrtm = 0.0;
  int n1 = 0, dim_z = 0;
  uword idx_min_mse;

  /* Get dimensions             */
  const int m       = W.n_rows;
  const int n_spcov = W.n_cols;
  const int n_cov   = Z.n_cols;
  const int G       = groups.n_rows;
  
  /* Variable definition            */
  sqrtm = std::sqrt(static_cast<float>(m));

  /* Get number of lambda           */
  const int nlambda = lambda.n_elem;

  /* Get output lists          */
  List out, output;

  /* Variable definition            */
  if (m >= n_spcov) {
    n1 = n_spcov;
  } else {
    n1 = m;
  }

  /* Definition of vectors and matrices     */
  arma::colvec x(n_spcov, fill::zeros);
  arma::colvec v(n_cov, fill::zeros);
  arma::mat W_(m, n_spcov, fill::zeros);
  arma::mat Z_(m, n_cov, fill::zeros);
  arma::mat WW_CHOL_U(n1, n_spcov, fill::zeros);
  arma::mat ZZ_CHOL_U(n_cov, n_cov, fill::zeros);
  arma::vec Wy(n_spcov, fill::zeros);
  arma::vec ones_m(m, fill::ones);
  arma::mat eye_m(m, m, fill::eye);
  arma::mat ZZ(n_cov, n_cov, fill::zeros);
  arma::mat L_ZZ(n_cov, n_cov, fill::zeros);
  arma::mat L_ZZ_INV(n_cov, n_cov, fill::zeros);
  arma::mat ZZ_INV(n_cov, n_cov, fill::zeros);
  arma::mat P_ZZ(m, m, fill::zeros);
  arma::mat M_ZZ(m, m, fill::zeros);
  arma::mat WMW(n_spcov, n_spcov, fill::zeros);
  arma::vec WMy(n_spcov, fill::zeros);
  arma::vec Zy(n_cov, fill::zeros);
  arma::vec v_LS(n_cov, fill::zeros);
  arma::mat ZW(n_cov, n_spcov, fill::zeros);
  arma::mat P_ZW(n_cov, n_spcov, fill::zeros);
  arma::mat FTF(n_spcov, n_spcov, fill::zeros);
  arma::colvec Glen(G, fill::zeros);
  arma::colvec fitted_(m, fill::zeros);
  arma::colvec residuals_(m, fill::zeros);
  arma::vec mse(nlambda, fill::zeros);

  /* Store output    */
  arma::field<vec> h_objval(nlambda);
  arma::field<vec> h_r_norm(nlambda);
  arma::field<vec> h_s_norm(nlambda);
  arma::field<vec> h_eps_pri(nlambda);
  arma::field<vec> h_eps_dual(nlambda);
  arma::field<vec> h_rho(nlambda);
  arma::vec niter(nlambda, fill::zeros);
  arma::vec eltime(nlambda, fill::zeros);
  arma::mat X(nlambda, n_spcov, fill::zeros);
  arma::mat V(nlambda, n_cov, fill::zeros);
  arma::vec conv(nlambda, fill::zeros);
  arma::mat residuals(nlambda, m, fill::zeros);
  arma::mat fitted(nlambda, m, fill::zeros);
  
  /* Precompute relevant quantities         */
  Z_             = Z.each_col() % (ones_m * (1.0 / sqrtm));                       // 19Nov25: divisione per n controllata e corretta
  W_             = W.each_col() % (ones_m * (1.0 / sqrtm));
  ZZ             = Z_.t() * Z_;
  L_ZZ           = (chol(ZZ)).t();
  L_ZZ_INV       = inv(trimatl(L_ZZ));
  ZZ_INV         = L_ZZ_INV.t() * L_ZZ_INV;
  P_ZZ           = Z_ * ZZ_INV * Z_.t();
  M_ZZ           = eye_m - P_ZZ;
  WMW            = W_.t() * M_ZZ * W_;
  WMy            = W_.t() * M_ZZ * y / static_cast<float>(sqrtm);
  Zy             = Z_.t() * y / static_cast<float>(sqrtm);
  v_LS           = L_ZZ_INV.t() * L_ZZ_INV * Zy;
  ZW             = Z_.t() * W_;
  P_ZW           = L_ZZ_INV.t() * L_ZZ_INV * ZW;
  WW_CHOL_U      = chol_qr_fact(W_, m, n_spcov);
  ZZ_CHOL_U      = chol_qr_fact(Z_, m, n_cov);
  y2             = as_scalar(y.t() * y) / static_cast<float>(m);
  Wy             = W_.t() * y / static_cast<float>(sqrtm);
  Glen           = sum(groups, 1);
  arma::sp_mat F = spglasso_Gmat2Fmat_sparse(groups, var_weights, var_weights_L1);
  FTF            = diagmat(spmat2spmat_mult(F.t(), F));
  
  /* Initialize vectors u and z         */
  dim_z = n_spcov + sum(Glen);
  arma::colvec u(dim_z, fill::zeros);
  arma::colvec z(dim_z, fill::zeros);
  arma::mat Umat(nlambda, dim_z, fill::zeros);
  arma::mat Zmat(nlambda, dim_z, fill::zeros);

  /* main loop  */
  for (int j=0; j<nlambda; j++) {
    /* perform linear regression with lasso penalty for the j-th lambda       */
    out = admm_spovglasso_cov_large_m_update_1lambda(W_, Z_, y, groups,
                                                     group_weights, var_weights, var_weights_L1,
                                                     G, F, FTF,
                                                     WMW, WMy, v_LS, P_ZW, WW_CHOL_U,
                                                     Wy, y2, ZZ_CHOL_U, Zy, ZW,
                                                     u, z, lambda(j), alpha, rho_adaptation,
                                                     rho, tau, mu, reltol, abstol, maxiter, ping);

    /* Retrieve output      */
    u = as<arma::colvec>(out["u"]);
    z = as<arma::colvec>(out["z"]);
    x = as<arma::colvec>(out["x"]);
    v = as<arma::colvec>(out["v"]);
    
    /* Retrieve output      */
    fitted_    = W * x + Z * v;
    residuals_ = y - fitted_;
    
    /* Store output      */
    niter(j)         = out["niter"];
    eltime(j)        = out["eltime"];
    conv(j)          = out["convergence"];
    X.row(j)         = x.t();
    V.row(j)         = v.t();
    h_objval(j)      = as<arma::vec>(out["objval"]);
    h_r_norm(j)      = as<arma::vec>(out["r_norm"]);
    h_s_norm(j)      = as<arma::vec>(out["s_norm"]);
    h_eps_pri(j)     = as<arma::vec>(out["eps_pri"]);
    h_eps_dual(j)    = as<arma::vec>(out["eps_dual"]);
    h_rho(j)         = (as<arma::vec>(out["rho"])).subvec(0, niter(j)-1);
    rho              = out["rho.updated"];
    residuals.row(j) = fitted_.t();
    fitted.row(j)    = residuals_.t();
    mse(j)           = as_scalar(residuals_.t() * residuals_) / (static_cast<float>(m));
    Umat.row(j)      = u.t();
    Zmat.row(j)      = z.t();
  }

  /* Find min MSE in-sample      */
  idx_min_mse = index_min(mse);
  min_mse     = mse(idx_min_mse);

  /* Get output         */
  output["sp.coef.path"]    = X;
  output["coef.path"]       = V;
  output["iternum"]         = niter;
  output["objfun"]          = h_objval;
  output["r_norm"]          = h_r_norm;
  output["s_norm"]          = h_s_norm;
  output["err_pri"]         = h_eps_pri;
  output["err_dual"]        = h_eps_dual;
  output["convergence"]     = conv;
  output["elapsedTime"]     = eltime;
  output["rho"]             = h_rho;
  output["lambda"]          = lambda;
  output["min.mse"]         = min_mse;
  output["indi.min.mse"]    = idx_min_mse;
  output["lambda.min"]      = lambda[idx_min_mse];
  output["sp.coefficients"] = (X.row(idx_min_mse)).t();
  output["coefficients"]    = (V.row(idx_min_mse)).t();
  output["mse"]             = mse;
  output["U"]               = Umat;
  output["Z"]               = Zmat;

  /* Return output      */
  return(output);
}

Rcpp::List admm_spovglasso_cov_large_n_fast(const arma::mat& W,
                                            const arma::mat& Z,
                                            const arma::vec& y,
                                            const arma::mat& groups,
                                            const arma::vec& group_weights,
                                            const arma::vec& var_weights,
                                            const arma::vec& var_weights_L1,
                                            const arma::vec& lambda,
                                            const double alpha,
                                            const bool rho_adaptation,
                                            double rho,
                                            const double tau,
                                            const double mu,
                                            const double reltol,
                                            const double abstol,
                                            const int maxiter,
                                            const int ping) {

  /* Variable delcaration           */
  double y2 = 0.0, min_mse = 0.0, sqrtm = 0.0;
  int n1 = 0, dim_z = 0;
  uword idx_min_mse;

  /* Get dimensions             */
  const int m       = W.n_rows;
  const int n_spcov = W.n_cols;
  const int n_cov   = Z.n_cols;
  const int G       = groups.n_rows;
  
  /* Variable definition            */
  sqrtm = std::sqrt(static_cast<float>(m));

  /* Get number of lambda           */
  const int nlambda = lambda.n_elem;

  /* Get output lists          */
  List out, output;

  /* Variable definition            */
  if (m >= n_spcov) {
    n1 = n_spcov;
  } else {
    n1 = m;
  }

  /* Definition of vectors and matrices     */
  arma::colvec x(n_spcov, fill::zeros);
  arma::colvec v(n_cov, fill::zeros);
  arma::mat Z_(m, n_cov, fill::zeros);
  arma::mat WW_CHOL_U(n1, n_spcov, fill::zeros);
  arma::mat ZZ_CHOL_U(n_cov, n_cov, fill::zeros);
  arma::vec Wy(n_spcov, fill::zeros);
  arma::vec ones_m(m, fill::ones);
  arma::mat eye_m(m, m, fill::eye);
  arma::mat ZZ(n_cov, n_cov, fill::zeros);
  arma::mat L_ZZ(n_cov, n_cov, fill::zeros);
  arma::mat L_ZZ_INV(n_cov, n_cov, fill::zeros);
  arma::mat ZZ_INV(n_cov, n_cov, fill::zeros);
  arma::mat P_ZZ(m, m, fill::zeros);
  arma::mat M_ZZ(m, m, fill::zeros);
  arma::mat WMW(n_spcov, n_spcov, fill::zeros);
  arma::vec WMy(n_spcov, fill::zeros);
  arma::vec Zy(n_cov, fill::zeros);
  arma::vec v_LS(n_cov, fill::zeros);
  arma::mat ZW(n_cov, n_spcov, fill::zeros);
  arma::mat P_ZW(n_cov, n_spcov, fill::zeros);
  arma::mat M_ZZW(m, n_spcov, fill::zeros);
  arma::mat M_ZZW2(m, m, fill::zeros);
  arma::mat FTF(n_spcov, n_spcov, fill::zeros);
  arma::mat FTF_INV(n_spcov, n_spcov, fill::zeros);
  arma::colvec Glen(G, fill::zeros);
  arma::colvec fitted_(m, fill::zeros);
  arma::colvec residuals_(m, fill::zeros);
  arma::vec mse(nlambda, fill::zeros);
  
  /* Store output    */
  arma::field<vec> h_objval(nlambda);
  arma::field<vec> h_r_norm(nlambda);
  arma::field<vec> h_s_norm(nlambda);
  arma::field<vec> h_eps_pri(nlambda);
  arma::field<vec> h_eps_dual(nlambda);
  arma::field<vec> h_rho(nlambda);
  arma::vec niter(nlambda, fill::zeros);
  arma::vec eltime(nlambda, fill::zeros);
  arma::mat X(nlambda, n_spcov, fill::zeros);
  arma::mat V(nlambda, n_cov, fill::zeros);
  arma::vec conv(nlambda, fill::zeros);
  arma::mat residuals(nlambda, m, fill::zeros);
  arma::mat fitted(nlambda, m, fill::zeros);
  
  /* Precompute relevant quantities         */
  Z_        = Z.each_col() % (ones_m * (1.0 / sqrtm));
  //W         = W.each_col() % (ones_m * (1.0 / sqrtm));          // è corretto non dividere W per la numerosità
  ZZ        = Z_.t() * Z_;
  L_ZZ      = (chol(ZZ)).t();
  L_ZZ_INV  = inv(trimatl(L_ZZ));
  ZZ_INV    = L_ZZ_INV.t() * L_ZZ_INV;
  P_ZZ      = Z_ * ZZ_INV * Z_.t();
  M_ZZ      = eye_m - P_ZZ;
  WMW       = W.t() * M_ZZ * W;                                 // 19Nov25: divisione per n controllata e corretta, così come è ora
  WMy       = W.t() * M_ZZ * y;
  Zy        = Z_.t() * y / static_cast<float>(sqrtm);
  v_LS      = L_ZZ_INV.t() * L_ZZ_INV * Zy;
  ZW        = Z_.t() * W / static_cast<float>(sqrtm);
  P_ZW      = L_ZZ_INV.t() * L_ZZ_INV * ZW;
  WW_CHOL_U = chol_qr_fact(W, m, n_spcov);
  ZZ_CHOL_U = chol_qr_fact(Z_, m, n_cov);
  y2        = as_scalar(y.t() * y) / static_cast<float>(m);
  Wy        = W.t() * y;
  M_ZZW     = M_ZZ * W;                                       // 19Nov25: divisione per n controllata e corretta, così come è ora
  //M_ZZW2    = M_ZZW * M_ZZW.t();
  
  Glen           = sum(groups, 1);
  arma::sp_mat F = spglasso_Gmat2Fmat_sparse(groups, var_weights, var_weights_L1);
  FTF            = diagmat(spmat2spmat_mult(F.t(), F));
  FTF_INV        = diagmat(1.0 / diagvec(FTF));
  M_ZZW2         = M_ZZW * diagmat(FTF_INV) * M_ZZW.t();
  
  /* Initialize vectors u and z         */
  dim_z = n_spcov + sum(Glen);
  arma::colvec u(dim_z, fill::zeros);
  arma::colvec z(dim_z, fill::zeros);
  arma::mat Umat(nlambda, dim_z, fill::zeros);
  arma::mat Zmat(nlambda, dim_z, fill::zeros);
  
  /* main loop  */
  for (int j=0; j<nlambda; j++) {
    /* perform linear regression with lasso penalty for the j-th lambda       */
    out = admm_spovglasso_cov_large_n_update_1lambda(W, Z_, y, groups, group_weights, var_weights, var_weights_L1,
                                                     G, F, FTF_INV, M_ZZW, M_ZZW2,
                                                     WMW, WMy, v_LS, P_ZW, WW_CHOL_U, Wy, y2,
                                                     ZZ_CHOL_U, Zy, ZW, u, z, lambda(j), alpha,
                                                     rho_adaptation, rho, tau, mu, reltol, abstol, maxiter, ping);

    /* Retrieve output      */
    u = as<arma::colvec>(out["u"]);
    z = as<arma::colvec>(out["z"]);
    x = as<arma::colvec>(out["x"]);
    v = as<arma::colvec>(out["v"]);
    
    /* Retrieve output      */
    fitted_    = W * x + Z * v;
    residuals_ = y - fitted_;

    /* Store output      */
    niter(j)         = out["niter"];
    eltime(j)        = out["eltime"];
    conv(j)          = out["convergence"];
    X.row(j)         = x.t();
    V.row(j)         = v.t();
    h_objval(j)      = as<arma::vec>(out["objval"]);
    h_r_norm(j)      = as<arma::vec>(out["r_norm"]);
    h_s_norm(j)      = as<arma::vec>(out["s_norm"]);
    h_eps_pri(j)     = as<arma::vec>(out["eps_pri"]);
    h_eps_dual(j)    = as<arma::vec>(out["eps_dual"]);
    h_rho(j)         = (as<arma::vec>(out["rho"])).subvec(0, niter(j)-1);
    rho              = out["rho.updated"];
    residuals.row(j) = fitted_.t();
    fitted.row(j)    = residuals_.t();
    mse(j)           = as_scalar(residuals_.t() * residuals_) / (static_cast<float>(m));
    Umat.row(j)      = u.t();
    Zmat.row(j)      = z.t();
  }

  /* Find min MSE in-sample      */
  idx_min_mse = index_min(mse);
  min_mse     = mse(idx_min_mse);

  /* Get output         */
  output["sp.coef.path"]    = X;
  output["coef.path"]       = V;
  output["iternum"]         = niter;
  output["objfun"]          = h_objval;
  output["r_norm"]          = h_r_norm;
  output["s_norm"]          = h_s_norm;
  output["err_pri"]         = h_eps_pri;
  output["err_dual"]        = h_eps_dual;
  output["convergence"]     = conv;
  output["elapsedTime"]     = eltime;
  output["rho"]             = h_rho;
  output["lambda"]          = lambda;
  output["min.mse"]         = min_mse;
  output["indi.min.mse"]    = idx_min_mse;
  output["lambda.min"]      = lambda[idx_min_mse];
  output["sp.coefficients"] = (X.row(idx_min_mse)).t();
  output["coefficients"]    = (V.row(idx_min_mse)).t();
  output["mse"]             = mse;
  output["U"]               = Umat;
  output["Z"]               = Zmat;
  
  /* Return output      */
  return(output);
}

/*
    Overlap Group-LASSO via ADMM: functions to fit a single lambda
*/

Rcpp::List admm_spovglasso_large_m_update_1lambda(const arma::mat& A,
                                                  const arma::vec& b,
                                                  const arma::mat& groups,
                                                  const arma::vec& group_weights,
                                                  const arma::vec& var_weights,
                                                  const arma::vec& var_weights_L1,
                                                  const arma::sp_mat& F,
                                                  const arma::mat& FTF,
                                                  const arma::mat& ATA,
                                                  const arma::mat& ATA_CHOL_U,
                                                  const arma::colvec& ATb,
                                                  const double b2,
                                                  const int m,
                                                  const int n,
                                                  const int G,
                                                  arma::colvec& u,
                                                  arma::colvec& z,
                                                  const double lambda,
                                                  const double alpha,
                                                  const bool rho_adaptation,
                                                  double rho,
                                                  const double tau,
                                                  const double mu,
                                                  const double reltol,
                                                  const double abstol,
                                                  const int maxiter,
                                                  const int ping) {
  
  /* Variable delcaration           */
  int k = 0, dim_z = 0;
  double rho_old = 0.0, elTime = 0.0, mse = 0.0, rss = 0.0, pen = 0.0;
  uword g_id_init, g_id_end;
  
  /* Get dimensions                 */
  dim_z = z.n_elem;
  
  /* Definition of vectors and matrices     */
  arma::colvec x(n, fill::zeros);
  arma::colvec q(n, fill::zeros);
  arma::colvec z_old(dim_z, fill::zeros);
  arma::mat U(n, n, fill::zeros);
  arma::mat L(n, n, fill::zeros);
  arma::vec h_objval(maxiter, fill::zeros);
  arma::vec h_r_norm(maxiter, fill::zeros);
  arma::vec h_s_norm(maxiter, fill::zeros);
  arma::vec h_eps_pri(maxiter, fill::zeros);
  arma::vec h_eps_dual(maxiter, fill::zeros);
  arma::vec rho_store(maxiter+1, fill::zeros);
  arma::colvec fitted(m, fill::zeros);
  arma::colvec residuals(m, fill::zeros);
  arma::colvec Glen(G, fill::zeros);
  
  /* Get groups length                */
  Glen = sum(groups, 1);

  /* ::::::::::::::::::::::::::::::::::::::::::::::
   Set the computational time                                 */
  wall_clock timer;
  timer.tic();
  
  /* store rho (only for adaptation)         */
  rho_store(0) = rho;
  rho_old      = rho;

  /* Precompute relevant quantities         */
  U = glasso_factor_fast_large_m(ATA, FTF, rho); // returns upper
  L = U.t();
  
  /* main loop  */
  for (k=0; k<maxiter; k++) {

    // 4-1. update 'x'
    if (rho != rho_old) {
      U = glasso_factor_fast_large_m(ATA, FTF, rho); // returns upper
      L = U.t();
    }
    q = ATb + rho * spmat2vec_mult(F.t(), z - u); // temporary value
    x = solve(trimatu(U), solve(trimatl(L), q));
    
    // 4-2. update 'z' with adaptation
    // see Boyd et al (2011) page 21
    z_old     = z;
    g_id_init = 0;
    for (int g=0; g<G; g++) {
      // precompute the quantities for updating z
      g_id_end         = g_id_init + Glen(g) - 1;
      arma::vec u_g    = u.subvec(g_id_init, g_id_end);
      arma::sp_mat F_g = F.submat(g_id_init, 0, g_id_end, n-1);

      // update z
      arma::vec z_g                 = glasso_prox(F_g * x + u_g, (lambda * (1.0 - alpha) * group_weights[g]) / rho);
      z.subvec(g_id_init, g_id_end) = z_g;
      g_id_init                     = g_id_end + 1;
    }
    // update z for the last group
    g_id_end                      = g_id_init + n - 1;
    arma::vec u_g                 = u.subvec(g_id_init, g_id_end);
    arma::sp_mat F_g              = F.submat(g_id_init, 0, g_id_end, n-1);
    arma::vec z_g                 = lasso_prox(F_g * x + u_g, lambda * alpha / rho);
    z.subvec(g_id_init, g_id_end) = z_g;

    // 4-3. update 'u'
    u += spmat2vec_mult(F, x) - z;
      
    // 4-3. dianostics, reporting
    rss         = lm_rss_fast(ATA_CHOL_U, ATb, b2, x, m, n);
    pen         = spovglasso_penalty(x, groups, group_weights, var_weights, var_weights_L1, alpha);
    h_objval(k) = 0.5 *  rss / static_cast<float>(m) + lambda * pen;
    h_r_norm(k) = norm(glasso_residual_sparse(F, x, z), 2);
    h_s_norm(k) = norm(glasso_dual_residual_sparse(F, z, z_old, rho), 2);
    if (norm(F * x, 2) > norm(-z, 2)) {
      h_eps_pri(k) = std::sqrt(z.n_elem) * abstol + reltol * norm(F * x, 2);
    } else {
      h_eps_pri(k) = std::sqrt(z.n_elem) * abstol + reltol * norm(-z, 2);
    }
    h_eps_dual(k) = std::sqrt(z.n_elem) * abstol + reltol * norm(F.t() * rho * u, 2);
    
    /* :::::::::::::::::::::::::::::::::::::::::::::::::::::::: */
    /* Relaxation                                               */
    rho_old = rho;
    if (rho_adaptation) {
      if (h_r_norm(k) > mu * h_s_norm(k)) {
        rho = rho * tau;
        u   = u / tau;
      } else if (h_s_norm(k) > mu * h_r_norm(k)) {
        rho = rho / tau;
        u   = u * tau;
      }
    }
    rho_store(k+1) = rho;
        
    /* :::::::::::::::::::::::::::::::::::::::::::::::::::::::: */
    /* Print to screen                                          */
    if ((ping > 0) && (k < maxiter)) {
      if (rho_adaptation) {
        if (((k+1) % ping) == 0) {
          Rcpp::Rcout << "\n\n\n" << std::endl;
          Rcpp::Rcout << "ADMM with adaptation for sparse overlap group-LASSO is running!" << std::endl;
          Rcpp::Rcout << "Iteration n.: " << k+1 << " of " << maxiter << "\n" << std::endl;
        }
      } else {
        if (((k+1) % ping) == 0) {
          Rcpp::Rcout << "\n\n\n" << std::endl;
          Rcpp::Rcout << "ADMM for sparse overlap group-LASSO is running!" << std::endl;
          Rcpp::Rcout << "Iteration n.: " << k+1 << " of " << maxiter << "\n" << std::endl;
        }
      }
    }

    // 4-4. termination
    if ((h_r_norm(k) < h_eps_pri(k)) && (h_s_norm(k) < h_eps_dual(k))) {
      break;
    }
  }

  /* ::::::::::::::::::::::::::::::::::::::::::::::
   Compute elapsed time                                */
  elTime = timer.toc();
  
  /* ::::::::::::::::::::::::::::::::::::::::::::::
   Compute post-estimation quantities                   */
  fitted    = A * x;
  residuals = b - fitted;
  mse       = as_scalar(residuals.t() * residuals) / (static_cast<float>(m));
  
  /* Get output         */
  List output;
  output["x"]      = x;
  output["u"]      = u;
  output["z"]      = z;
  output["eltime"] = elTime;
  if (k < maxiter) {
    output["convergence"] = 1.0;
    output["niter"]       = k+1;
    output["objval"]      = h_objval.subvec(0, k);
    output["r_norm"]      = h_r_norm.subvec(0, k);
    output["s_norm"]      = h_s_norm.subvec(0, k);
    output["eps_pri"]     = h_eps_pri.subvec(0, k);
    output["eps_dual"]    = h_eps_dual.subvec(0, k);
    output["rho"]         = rho_store.subvec(0, k+1);
  } else {
    output["convergence"] = 0.0;
    output["niter"]       = maxiter;
    output["objval"]      = h_objval;
    output["r_norm"]      = h_r_norm;
    output["s_norm"]      = h_s_norm;
    output["eps_pri"]     = h_eps_pri;
    output["eps_dual"]    = h_eps_dual;
    output["rho"]         = rho_store;
  }
  output["residuals"]     = residuals;
  output["fitted.values"] = fitted;
  output["mse"]           = mse;
  output["rho.updated"]   = rho;
  
  /* Return output      */
  return(output);
}

Rcpp::List admm_spovglasso_large_n_update_1lambda(const arma::mat& A,
                                                  const arma::vec& b,
                                                  const arma::mat& groups,
                                                  const arma::vec& group_weights,
                                                  const arma::vec& var_weights,
                                                  const arma::vec& var_weights_L1,
                                                  const arma::sp_mat& F,
                                                  const arma::mat& FTF_INV,
                                                  const arma::mat& AFTF_INVA,
                                                  const arma::mat& ATA_CHOL_U,
                                                  const arma::colvec& ATb,
                                                  const double b2,
                                                  const int m,
                                                  const int n,
                                                  const int G,
                                                  arma::colvec& u,
                                                  arma::colvec& z,
                                                  const double lambda,
                                                  const double alpha,
                                                  const bool rho_adaptation,
                                                  double rho,
                                                  const double tau,
                                                  const double mu,
                                                  const double reltol,
                                                  const double abstol,
                                                  const int maxiter,
                                                  const int ping) {
  
  /* Variable delcaration           */
  int k = 0, dim_z = 0;
  double rho_old = 0.0, elTime = 0.0, mse = 0.0, rss = 0.0, pen = 0.0;
  uword g_id_init, g_id_end;
  
  /* Get dimensions                 */
  dim_z = z.n_elem;
  
  /* Definition of vectors and matrices     */
  arma::colvec x(n, fill::zeros);
  arma::colvec x_star(n, fill::zeros);
  arma::colvec q(n, fill::zeros);
  arma::colvec z_old(dim_z, fill::zeros);
  arma::mat U(m, m, fill::zeros);
  arma::mat L(m, m, fill::zeros);
  arma::vec h_objval(maxiter, fill::zeros);
  arma::vec h_r_norm(maxiter, fill::zeros);
  arma::vec h_s_norm(maxiter, fill::zeros);
  arma::vec h_eps_pri(maxiter, fill::zeros);
  arma::vec h_eps_dual(maxiter, fill::zeros);
  arma::vec rho_store(maxiter+1, fill::zeros);
  arma::colvec fitted(m, fill::zeros);
  arma::colvec residuals(m, fill::zeros);
  arma::colvec Glen(G, fill::zeros);
  
  /* Get groups length                */
  Glen = sum(groups, 1);

  /* ::::::::::::::::::::::::::::::::::::::::::::::
   Set the computational time                                 */
  wall_clock timer;
  timer.tic();
  
  /* store rho (only for adaptation)         */
  rho_store(0) = rho;
  rho_old      = rho;

  /* Precompute relevant quantities         */
  U = glasso_factor_fast_large_n(AFTF_INVA, rho, m); // returns upper
  L = U.t();
  
  /* main loop  */
  for (k=0; k<maxiter; k++) {

    // 4-1. update 'x'
    if (rho != rho_old) {
      U = glasso_factor_fast_large_n(AFTF_INVA, rho, m); // returns upper
      L = U.t();
    }
    //q      = ATb + rho * F.t() * (z - u); // temporary value
    q      = ATb + rho * spmat2vec_mult(F.t(), z - u);
    x_star = (diagmat(FTF_INV) * q) / rho;
    x      = x_star - diagmat(FTF_INV) * A.t() * solve(trimatu(U), solve(trimatl(L), A * x_star));
    
    // 4-2. update 'z' with adaptation
    // see Boyd et al (2011) page 21
    z_old     = z;
    g_id_init = 0;
    for (int g=0; g<G; g++) {
      // precompute the quantities for updating z
      g_id_end         = g_id_init + Glen(g) - 1;
      arma::vec u_g    = u.subvec(g_id_init, g_id_end);
      arma::sp_mat F_g = F.submat(g_id_init, 0, g_id_end, n-1);

      // update z
      arma::vec z_g                 = glasso_prox(F_g * x + u_g, (lambda * (1.0 - alpha) * group_weights[g]) / rho);     // qui ho cambiato
      z.subvec(g_id_init, g_id_end) = z_g;
      g_id_init                     = g_id_end + 1;
    }
    // update z for the last group
    g_id_end                      = g_id_init + n - 1;
    arma::vec u_g                 = u.subvec(g_id_init, g_id_end);
    arma::sp_mat F_g              = F.submat(g_id_init, 0, g_id_end, n-1);
    arma::vec z_g                 = lasso_prox(F_g * x + u_g, lambda * alpha / rho);
    z.subvec(g_id_init, g_id_end) = z_g;

    // 4-3. update 'u'
    //u = u + F * x - z;
    u += spmat2vec_mult(F, x) - z;
  
    // 4-3. dianostics, reporting
    rss         = lm_rss_fast(ATA_CHOL_U, ATb, b2, x, m, n);
    pen         = spovglasso_penalty(x, groups, group_weights, var_weights, var_weights_L1, alpha);
    h_objval(k) = 0.5 *  rss / static_cast<float>(m) + lambda * pen;
    h_r_norm(k) = norm(glasso_residual_sparse(F, x, z), 2);
    h_s_norm(k) = norm(glasso_dual_residual_sparse(F, z, z_old, rho), 2);
    if (norm(F * x, 2) > norm(-z, 2)) {
      h_eps_pri(k) = std::sqrt(z.n_elem) * abstol + reltol * norm(F * x, 2);
    } else {
      h_eps_pri(k) = std::sqrt(z.n_elem) * abstol + reltol * norm(-z, 2);
    }
    h_eps_dual(k) = std::sqrt(z.n_elem) * abstol + reltol * norm(F.t() * rho * u, 2);
    
    /* :::::::::::::::::::::::::::::::::::::::::::::::::::::::: */
    /* Relaxation                                               */
    rho_old = rho;
    if (rho_adaptation) {
      if (h_r_norm(k) > mu * h_s_norm(k)) {
        rho = rho * tau;
        u   = u / tau;
      } else if (h_s_norm(k) > mu * h_r_norm(k)) {
        rho = rho / tau;
        u   = u * tau;
      }
    }
    rho_store(k+1) = rho;
        
    /* :::::::::::::::::::::::::::::::::::::::::::::::::::::::: */
    /* Print to screen                                          */
    if ((ping > 0) && (k < maxiter)) {
      if (rho_adaptation) {
        if (((k+1) % ping) == 0) {
          Rcpp::Rcout << "\n\n\n" << std::endl;
          Rcpp::Rcout << "ADMM with adaptation for sparse overlap group-LASSO is running!" << std::endl;
          Rcpp::Rcout << "Iteration n.: " << k+1 << " of " << maxiter << "\n" << std::endl;
        }
      } else {
        if (((k+1) % ping) == 0) {
          Rcpp::Rcout << "\n\n\n" << std::endl;
          Rcpp::Rcout << "ADMM for sparse overlap group-LASSO is running!" << std::endl;
          Rcpp::Rcout << "Iteration n.: " << k+1 << " of " << maxiter << "\n" << std::endl;
        }
      }
    }

    // 4-4. termination
    if ((h_r_norm(k) < h_eps_pri(k)) && (h_s_norm(k) < h_eps_dual(k))) {
      break;
    }
  }

  /* ::::::::::::::::::::::::::::::::::::::::::::::
   Compute elapsed time                                */
  elTime = timer.toc();
  
  /* ::::::::::::::::::::::::::::::::::::::::::::::
   Compute post-estimation quantities                   */
  fitted    = A * x;
  residuals = b - fitted;
  mse       = as_scalar(residuals.t() * residuals) / (static_cast<float>(m));
  
  /* Get output         */
  List output;
  output["x"]      = x;
  output["u"]      = u;
  output["z"]      = z;
  output["eltime"] = elTime;
  if (k < maxiter) {
    output["convergence"] = 1.0;
    output["niter"]       = k+1;
    output["objval"]      = h_objval.subvec(0, k);
    output["r_norm"]      = h_r_norm.subvec(0, k);
    output["s_norm"]      = h_s_norm.subvec(0, k);
    output["eps_pri"]     = h_eps_pri.subvec(0, k);
    output["eps_dual"]    = h_eps_dual.subvec(0, k);
    output["rho"]         = rho_store.subvec(0, k+1);
  } else {
    output["convergence"] = 0.0;
    output["niter"]       = maxiter;
    output["objval"]      = h_objval;
    output["r_norm"]      = h_r_norm;
    output["s_norm"]      = h_s_norm;
    output["eps_pri"]     = h_eps_pri;
    output["eps_dual"]    = h_eps_dual;
    output["rho"]         = rho_store;
  }
  output["residuals"]     = residuals;
  output["fitted.values"] = fitted;
  output["mse"]           = mse;
  output["rho.updated"]   = rho;
  
  /* Return output      */
  return(output);
}

Rcpp::List admm_spovglasso_cov_large_m_update_1lambda(const arma::mat& W,
                                                      const arma::mat& Z,
                                                      const arma::vec& y,
                                                      const arma::mat& groups,
                                                      const arma::vec& group_weights,
                                                      const arma::vec& var_weights,
                                                      const arma::vec& var_weights_L1,
                                                      const int G,
                                                      const arma::sp_mat& F,
                                                      const arma::mat& FTF,
                                                      const arma::mat& WMW,
                                                      const arma::vec& WMy,
                                                      const arma::vec& v_LS,
                                                      const arma::mat& P_ZW,
                                                      const arma::mat& WW_CHOL_U,
                                                      const arma::colvec& Wy,
                                                      const double y2,
                                                      const arma::mat& ZZ_CHOL_U,
                                                      const arma::colvec& Zy,
                                                      const arma::mat& ZW,
                                                      arma::colvec& u,
                                                      arma::colvec& z,
                                                      const double lambda,
                                                      const double alpha,
                                                      const bool rho_adaptation,
                                                      double rho,
                                                      const double tau,
                                                      const double mu,
                                                      const double reltol,
                                                      const double abstol,
                                                      const int maxiter,
                                                      const int ping) {
  
  /* Variable delcaration           */
  int k = 0;
  double rho_old = 0.0, elTime = 0.0, rss = 0.0, pen = 0.0;
  uword g_id_init, g_id_end;
  
  /* Get dimensions             */
  const int m       = W.n_rows;
  const int n_spcov = W.n_cols;
  const int n_cov   = Z.n_cols;
  const int dim_z   = z.n_elem;
    
  /* Definition of vectors and matrices     */
  arma::colvec x(n_spcov, fill::zeros);
  arma::colvec v(n_cov, fill::zeros);
  arma::colvec q(n_spcov, fill::zeros);
  arma::colvec z_old(dim_z, fill::zeros);
  arma::mat U(n_spcov, n_spcov, fill::zeros);
  arma::mat L(n_spcov, n_spcov, fill::zeros);
  arma::vec h_objval(maxiter, fill::zeros);
  arma::vec h_r_norm(maxiter, fill::zeros);
  arma::vec h_s_norm(maxiter, fill::zeros);
  arma::vec h_eps_pri(maxiter, fill::zeros);
  arma::vec h_eps_dual(maxiter, fill::zeros);
  arma::vec rho_store(maxiter+1, fill::zeros);
  arma::colvec Glen(G, fill::zeros);
  
  /* Get groups length                */
  Glen = sum(groups, 1);
  
  /* ::::::::::::::::::::::::::::::::::::::::::::::
   Set the computational time                                 */
  wall_clock timer;
  timer.tic();
  
  /* store rho (only for adaptation)         */
  rho_store(0) = rho;
  rho_old      = rho;

  /* Precompute relevant quantities         */
  //U = glasso_factor_fast_large_m(ATA, FTF, rho); // returns upper
  U = glasso_factor_cov_fast_large_m(WMW, FTF, rho);
  L = U.t();
  
  /* main loop  */
  for (k=0; k<maxiter; k++) {

    // 4-1. update 'x'
    if (rho != rho_old) {
      //U = glasso_factor_fast_large_m(ATA, FTF, rho); // returns upper
      U = glasso_factor_cov_fast_large_m(WMW, FTF, rho);
      L = U.t();
    }
    q = WMy + rho * spmat2vec_mult(F.t(), z - u);        // temporary value
    x = solve(trimatu(U), solve(trimatl(L), q));         // updated of regression parameters associated to covariates W (sparsified covariates, x)
    v = v_LS - P_ZW * x;                                 // updated of regression parameters associated to covariates Z (covariates not sparsified, v)
    
    // 4-2. update 'z' with adaptation
    // see Boyd et al (2011) page 21
    z_old     = z;
    g_id_init = 0;
    for (int g=0; g<G; g++) {
      // precompute the quantities for updating z
      g_id_end         = g_id_init + Glen(g) - 1;
      arma::vec u_g    = u.subvec(g_id_init, g_id_end);
      arma::sp_mat F_g = F.submat(g_id_init, 0, g_id_end, n_spcov-1);

      // update z
      arma::vec z_g                 = glasso_prox(F_g * x + u_g, (lambda * group_weights[g]) / rho);
      z.subvec(g_id_init, g_id_end) = z_g;
      g_id_init                     = g_id_end + 1;
    }
    // update z for the last group
    g_id_end                      = g_id_init + n_spcov - 1;
    arma::vec u_g                 = u.subvec(g_id_init, g_id_end);
    arma::sp_mat F_g              = F.submat(g_id_init, 0, g_id_end, n_spcov-1);
    arma::vec z_g                 = lasso_prox(F_g * x + u_g, lambda * alpha / rho);
    z.subvec(g_id_init, g_id_end) = z_g;

    // 4-3. update 'u'
    u += spmat2vec_mult(F, x) - z;
  
    // 4-3. dianostics, reporting
    rss         = lm_cov_rss_fast(WW_CHOL_U, ZZ_CHOL_U, ZW, Wy, Zy, y2, x, v, m);
    pen         = spovglasso_penalty(x, groups, group_weights, var_weights, var_weights_L1, alpha);
    h_objval(k) = 0.5 *  rss / static_cast<float>(m) + lambda * pen;
    h_r_norm(k) = norm(glasso_residual_sparse(F, x, z), 2);
    h_s_norm(k) = norm(glasso_dual_residual_sparse(F, z, z_old, rho), 2);
    if (norm(F * x, 2) > norm(-z, 2)) {
      h_eps_pri(k) = std::sqrt(z.n_elem) * abstol + reltol * norm(F * x, 2);
    } else {
      h_eps_pri(k) = std::sqrt(z.n_elem) * abstol + reltol * norm(-z, 2);
    }
    h_eps_dual(k) = std::sqrt(z.n_elem) * abstol + reltol * norm(F.t() * rho * u, 2);
    
    /* :::::::::::::::::::::::::::::::::::::::::::::::::::::::: */
    /* Relaxation                                               */
    rho_old = rho;
    if (rho_adaptation) {
      if (h_r_norm(k) > mu * h_s_norm(k)) {
        rho = rho * tau;
        u   = u / tau;
      } else if (h_s_norm(k) > mu * h_r_norm(k)) {
        rho = rho / tau;
        u   = u * tau;
      }
    }
    rho_store(k+1) = rho;
        
    /* :::::::::::::::::::::::::::::::::::::::::::::::::::::::: */
    /* Print to screen                                          */
    if ((ping > 0) && (k < maxiter)) {
      if (rho_adaptation) {
        if (((k+1) % ping) == 0) {
          Rcpp::Rcout << "\n\n\n" << std::endl;
          Rcpp::Rcout << "ADMM with adaptation for sparse overlap group-LASSO is running!" << std::endl;
          Rcpp::Rcout << "Iteration n.: " << k+1 << " of " << maxiter << "\n" << std::endl;
        }
      } else {
        if (((k+1) % ping) == 0) {
          Rcpp::Rcout << "\n\n\n" << std::endl;
          Rcpp::Rcout << "ADMM for sparse overlap group-LASSO is running!" << std::endl;
          Rcpp::Rcout << "Iteration n.: " << k+1 << " of " << maxiter << "\n" << std::endl;
        }
      }
    }

    // 4-4. termination
    if ((h_r_norm(k) < h_eps_pri(k)) && (h_s_norm(k) < h_eps_dual(k))) {
      break;
    }
  }

  /* ::::::::::::::::::::::::::::::::::::::::::::::
   Compute elapsed time                                */
  elTime = timer.toc();
  
  /* Get output         */
  List output;
  output["x"]      = x;
  output["v"]      = v;
  output["u"]      = u;
  output["z"]      = z;
  output["eltime"] = elTime;
  if (k < maxiter) {
    output["convergence"] = 1.0;
    output["niter"]       = k+1;
    output["objval"]      = h_objval.subvec(0, k);
    output["r_norm"]      = h_r_norm.subvec(0, k);
    output["s_norm"]      = h_s_norm.subvec(0, k);
    output["eps_pri"]     = h_eps_pri.subvec(0, k);
    output["eps_dual"]    = h_eps_dual.subvec(0, k);
    output["rho"]         = rho_store.subvec(0, k+1);
  } else {
    output["convergence"] = 0.0;
    output["niter"]       = maxiter;
    output["objval"]      = h_objval;
    output["r_norm"]      = h_r_norm;
    output["s_norm"]      = h_s_norm;
    output["eps_pri"]     = h_eps_pri;
    output["eps_dual"]    = h_eps_dual;
    output["rho"]         = rho_store;
  }
  output["rho.updated"]   = rho;
  
  /* Return output      */
  return(output);
}

Rcpp::List admm_spovglasso_cov_large_n_update_1lambda(const arma::mat& W,
                                                      const arma::mat& Z,
                                                      const arma::vec& y,
                                                      const arma::mat& groups,
                                                      const arma::vec& group_weights,
                                                      const arma::vec& var_weights,
                                                      const arma::vec& var_weights_L1,
                                                      const int G,
                                                      const arma::sp_mat& F,
                                                      const arma::mat& FF_INV,
                                                      const arma::mat& M_ZZW,
                                                      const arma::mat& M_ZZW2,
                                                      const arma::mat& WMW,
                                                      const arma::vec& WMy,
                                                      const arma::vec& v_LS,
                                                      const arma::mat& P_ZW,
                                                      const arma::mat& WW_CHOL_U,
                                                      const arma::colvec& Wy,
                                                      const double y2,
                                                      const arma::mat& ZZ_CHOL_U,
                                                      const arma::colvec& Zy,
                                                      const arma::mat& ZW,
                                                      arma::colvec& u,
                                                      arma::colvec& z,
                                                      const double lambda,
                                                      const double alpha,
                                                      const bool rho_adaptation,
                                                      double rho,
                                                      const double tau,
                                                      const double mu,
                                                      const double reltol,
                                                      const double abstol,
                                                      const int maxiter,
                                                      const int ping) {
  
  /* Variable delcaration           */
  int k = 0;
  double rho_old = 0.0, elTime = 0.0, rss = 0.0, pen = 0.0;
  uword g_id_init, g_id_end;

  /* Get dimensions             */
  const int m       = W.n_rows;
  const int n_spcov = W.n_cols;
  const int n_cov   = Z.n_cols;
  const int dim_z   = z.n_elem;
    
  /* Definition of vectors and matrices     */
  arma::colvec x(n_spcov, fill::zeros);
  arma::colvec x_star(n_spcov, fill::zeros);
  arma::colvec v(n_cov, fill::zeros);
  arma::colvec q(n_spcov, fill::zeros);
  arma::colvec z_old(dim_z, fill::zeros);
  arma::mat U(m, m, fill::zeros);
  arma::mat L(m, m, fill::zeros);
  arma::vec h_objval(maxiter, fill::zeros);
  arma::vec h_r_norm(maxiter, fill::zeros);
  arma::vec h_s_norm(maxiter, fill::zeros);
  arma::vec h_eps_pri(maxiter, fill::zeros);
  arma::vec h_eps_dual(maxiter, fill::zeros);
  arma::vec rho_store(maxiter+1, fill::zeros);
  arma::colvec Glen(G, fill::zeros);
  
  /* Get groups length                */
  Glen = sum(groups, 1);
  
  /* ::::::::::::::::::::::::::::::::::::::::::::::
   Set the computational time                                 */
  wall_clock timer;
  timer.tic();
  
  /* store rho (only for adaptation)         */
  rho_store(0) = rho;
  rho_old      = rho;

  /* Precompute relevant quantities         */
  U = glasso_factor_cov_fast_large_n(M_ZZW2, rho, m); // returns upper
  L = U.t();
  
  /* main loop  */
  for (k=0; k<maxiter; k++) {

    // 4-1. update 'x'
    if (rho != rho_old) {
      U = glasso_factor_cov_fast_large_n(M_ZZW2, rho, m); // returns upper
      L = U.t();
    }
    q      = WMy + rho * spmat2vec_mult(F.t(), z - u); // temporary value
    x_star = (diagmat(FF_INV) * q) / rho;
    x      = x_star - diagmat(FF_INV) * M_ZZW.t() * solve(trimatu(U), solve(trimatl(L), M_ZZW * x_star)); // updated of regression parameters associated to covariates W
    v      = v_LS - P_ZW * x;                                 // updated of regression parameters associated to covariates Z (covariates not sparsified, v)

    // 4-2. update 'z' with adaptation
    // see Boyd et al (2011) page 21
    z_old     = z;
    g_id_init = 0;
    for (int g=0; g<G; g++) {
      // precompute the quantities for updating z
      g_id_end         = g_id_init + Glen(g) - 1;
      arma::vec u_g    = u.subvec(g_id_init, g_id_end);
      arma::sp_mat F_g = F.submat(g_id_init, 0, g_id_end, n_spcov-1);

      // update z
      arma::vec z_g                 = glasso_prox(F_g * x + u_g, (lambda * group_weights[g]) / rho);
      z.subvec(g_id_init, g_id_end) = z_g;
      g_id_init                     = g_id_end + 1;
    }
    // update z for the last group
    g_id_end                      = g_id_init + n_spcov - 1;
    arma::vec u_g                 = u.subvec(g_id_init, g_id_end);
    arma::sp_mat F_g              = F.submat(g_id_init, 0, g_id_end, n_spcov-1);
    arma::vec z_g                 = lasso_prox(F_g * x + u_g, lambda * alpha / rho);
    z.subvec(g_id_init, g_id_end) = z_g;

    // 4-3. update 'u'
    u += spmat2vec_mult(F, x) - z;
  
    // 4-3. dianostics, reporting
    rss         = lm_cov_rss_fast(WW_CHOL_U, ZZ_CHOL_U, ZW, Wy, Zy, y2, x, v, m);
    pen         = spovglasso_penalty(x, groups, group_weights, var_weights, var_weights_L1, alpha);
    h_objval(k) = 0.5 *  rss / static_cast<float>(m) + lambda * pen;
    h_r_norm(k) = norm(glasso_residual_sparse(F, x, z), 2);
    h_s_norm(k) = norm(glasso_dual_residual_sparse(F, z, z_old, rho), 2);
    if (norm(F * x, 2) > norm(-z, 2)) {
      h_eps_pri(k) = std::sqrt(z.n_elem) * abstol + reltol * norm(F * x, 2);
    } else {
      h_eps_pri(k) = std::sqrt(z.n_elem) * abstol + reltol * norm(-z, 2);
    }
    h_eps_dual(k) = std::sqrt(z.n_elem) * abstol + reltol * norm(F.t() * rho * u, 2);
    
    /* :::::::::::::::::::::::::::::::::::::::::::::::::::::::: */
    /* Relaxation                                               */
    rho_old = rho;
    if (rho_adaptation) {
      if (h_r_norm(k) > mu * h_s_norm(k)) {
        rho = rho * tau;
        u   = u / tau;
      } else if (h_s_norm(k) > mu * h_r_norm(k)) {
        rho = rho / tau;
        u   = u * tau;
      }
    }
    rho_store(k+1) = rho;
        
    /* :::::::::::::::::::::::::::::::::::::::::::::::::::::::: */
    /* Print to screen                                          */
    if ((ping > 0) && (k < maxiter)) {
      if (rho_adaptation) {
        if (((k+1) % ping) == 0) {
          Rcpp::Rcout << "\n\n\n" << std::endl;
          Rcpp::Rcout << "ADMM with adaptation for sparse overlap group-LASSO is running!" << std::endl;
          Rcpp::Rcout << "Iteration n.: " << k+1 << " of " << maxiter << "\n" << std::endl;
        }
      } else {
        if (((k+1) % ping) == 0) {
          Rcpp::Rcout << "\n\n\n" << std::endl;
          Rcpp::Rcout << "ADMM for sparse overlap group-LASSO is running!" << std::endl;
          Rcpp::Rcout << "Iteration n.: " << k+1 << " of " << maxiter << "\n" << std::endl;
        }
      }
    }

    // 4-4. termination
    if ((h_r_norm(k) < h_eps_pri(k)) && (h_s_norm(k) < h_eps_dual(k))) {
      break;
    }
  }

  /* ::::::::::::::::::::::::::::::::::::::::::::::
   Compute elapsed time                                */
  elTime = timer.toc();
  
  /* Get output         */
  List output;
  output["x"]      = x;
  output["v"]      = v;
  output["u"]      = u;
  output["z"]      = z;
  output["eltime"] = elTime;
  if (k < maxiter) {
    output["convergence"] = 1.0;
    output["niter"]       = k+1;
    output["objval"]      = h_objval.subvec(0, k);
    output["r_norm"]      = h_r_norm.subvec(0, k);
    output["s_norm"]      = h_s_norm.subvec(0, k);
    output["eps_pri"]     = h_eps_pri.subvec(0, k);
    output["eps_dual"]    = h_eps_dual.subvec(0, k);
    output["rho"]         = rho_store.subvec(0, k+1);
  } else {
    output["convergence"] = 0.0;
    output["niter"]       = maxiter;
    output["objval"]      = h_objval;
    output["r_norm"]      = h_r_norm;
    output["s_norm"]      = h_s_norm;
    output["eps_pri"]     = h_eps_pri;
    output["eps_dual"]    = h_eps_dual;
    output["rho"]         = rho_store;
  }
  output["rho.updated"]   = rho;
  
  /* Return output      */
  return(output);
}
