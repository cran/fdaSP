#ifndef LINREG_ADMM_H
#define LINREG_ADMM_H

#include <RcppArmadillo.h>
#include "ADMMutilsLM.h"

#define DOUBLE_EPS 2.220446e-16
#define ARMA_64BIT_WORD 1
#define SAFE_LOG(a) (((a) <= 0.0) ? log(DOUBLE_EPS) : log(a))
#define SAFE_ZERO(a) ((a) == 0 ? DOUBLE_EPS : (a))
#define SQRT_DOUBLE_EPS sqrt(DOUBLE_EPS)

using std::pow;
using std::exp;
using std::sqrt;
using std::log;

using namespace Rcpp;
using namespace arma;

// ==========================================================
// admm_enet: Linear regression model with ENET penalty
// Linear regression model with ENET penalty: evaluate the objective function
Rcpp::List admm_enet(const arma::mat& A, const arma::colvec& b,  const double lambda, const double alpha,
                     const double reltol, const double abstol, const int maxiter, const double rho);

// ======================================================
// admm_genlasso: Linear regression model with generalized LASSO penalty
// Linear regression model with generalized LASSO penalty: evaluate the objective function
Rcpp::List admm_genlasso(const arma::mat& A, const arma::colvec& b, const arma::mat &D,
                         const double lambda, const double reltol, const double abstol,
                         const int maxiter, const double rho);

// =========================================================
// admm_lasso
// ADMM for linear regression models with LASSO penalty
// elementwise soft thresholding operator
/*
* LASSO via ADMM (from Stanford)
* http://stanford.edu/~boyd/papers/pdf/admm_distr_stats.pdf
* page 19 : section 3.3.1 : stopping criteria part (3.12).
*/
Rcpp::List admm_lasso(const arma::mat& A, const arma::colvec& b, arma::colvec& u, arma::colvec& z, const double lambda,
                      bool rho_adaptation, double rho = 1, const double tau = 2, const double mu = 10,
                      const double reltol = 1e-5, const double abstol = 1e-5, const int maxiter = 100, const int ping = 0);

/*
* Adaptive LASSO via ADMM (from Stanford)
* http://stanford.edu/~boyd/papers/pdf/admm_distr_stats.pdf
* page 19 : section 3.3.1 : stopping criteria part (3.12).
*/
Rcpp::List admm_adalasso(const arma::mat& A, const arma::colvec& b, arma::vec& var_weights,
                         arma::colvec& u, arma::colvec& z, const double lambda,
                         bool rho_adaptation, double rho = 1, const double tau = 2, const double mu = 10,
                         const double reltol = 1e-5, const double abstol = 1e-5, const int maxiter = 100, const int ping = 0);

Rcpp::List admm_adalasso_large_m(const arma::mat& A, const arma::colvec& b, arma::vec& var_weights,
                                 arma::colvec& u, arma::colvec& z, const double lambda,
                                 bool rho_adaptation, double rho, const double tau, const double mu,
                                 const double reltol, const double abstol, const int maxiter, const int ping);

Rcpp::List admm_adalasso_large_n(const arma::mat& A, const arma::colvec& b, arma::vec& var_weights,
                                 arma::colvec& u, arma::colvec& z, const double lambda,
                                 bool rho_adaptation, double rho, const double tau, const double mu,
                                 const double reltol, const double abstol, const int maxiter, const int ping);

// =========================================================
// admm_glasso
// ADMM for linear regression models with group-LASSO penalty

/* Group Lasso vector (or block) soft thresholding operator        */
/*
* GLASSO via ADMM (from Stanford)
*/
Rcpp::List admm_glasso(const arma::mat& A, const arma::colvec& b,
                       arma::mat& groups, arma::vec& group_weights, arma::vec& var_weights,
                       arma::colvec& u, arma::colvec& z, const double lambda,
                       bool rho_adaptation, double rho = 1, const double tau = 2, const double mu = 10,
                       const double reltol = 1e-5, const double abstol = 1e-5, const int maxiter = 100, const int ping = 0);

Rcpp::List admm_glasso_large_m(const arma::mat& A, const arma::colvec& b,
                               arma::mat& groups, arma::vec& group_weights, arma::vec& var_weights,
                               arma::colvec& u, arma::colvec& z, const double lambda,
                               bool rho_adaptation, double rho, const double tau, const double mu,
                               const double reltol, const double abstol, const int maxiter, const int ping);

Rcpp::List admm_glasso_large_n(const arma::mat& A, const arma::colvec& b,
                               arma::mat& groups, arma::vec& group_weights, arma::vec& var_weights,
                               arma::colvec& u, arma::colvec& z, const double lambda,
                               bool rho_adaptation, double rho, const double tau, const double mu,
                               const double reltol, const double abstol, const int maxiter, const int ping);

// =========================================================
// admm_spglasso
// ADMM for linear regression models with sparse group-LASSO penalty
/*
* SPGLASSO via ADMM (from Stanford)
*/
Rcpp::List admm_spglasso(const arma::mat& A, const arma::colvec& b,
                         arma::mat& groups, arma::vec& group_weights,
                         arma::vec& var_weights, arma::vec& var_weights_L1,
                         arma::colvec& u, arma::colvec& z, const double lambda1, const double lambda2,
                         bool rho_adaptation, double rho = 1, const double tau = 2, const double mu = 10,
                         const double reltol = 1e-5, const double abstol = 1e-5, const int maxiter = 100, const int ping = 0);

Rcpp::List admm_spglasso_large_m(const arma::mat& A, const arma::colvec& b,
                                 arma::mat& groups, arma::vec& group_weights,
                                 arma::vec& var_weights, arma::vec& var_weights_L1,
                                 arma::colvec& u, arma::colvec& z, const double lambda1, const double lambda2,
                                 bool rho_adaptation, double rho, const double tau, const double mu,
                                 const double reltol, const double abstol, const int maxiter, const int ping);

Rcpp::List admm_spglasso_large_n(const arma::mat& A, const arma::colvec& b,
                                 arma::mat& groups, arma::vec& group_weights,
                                 arma::vec& var_weights, arma::vec& var_weights_L1,
                                 arma::colvec& u, arma::colvec& z, const double lambda1, const double lambda2,
                                 bool rho_adaptation, double rho, const double tau, const double mu,
                                 const double reltol, const double abstol, const int maxiter, const int ping);

// =========================================================
// admm_ovglasso
// ADMM for linear regression models with overlap group-LASSO penalty

/*
* Overlap Group-LASSO via ADMM (from Stanford)
*/
Rcpp::List admm_ovglasso(const arma::mat& A, const arma::colvec& b,
                         arma::mat& groups, arma::vec& group_weights, arma::vec& var_weights,
                         arma::colvec& u, arma::colvec& z, const double lambda,
                         bool rho_adaptation, double rho = 1, const double tau = 2, const double mu = 10,
                         const double reltol = 1e-5, const double abstol = 1e-5, const int maxiter = 100, const int ping = 0);

Rcpp::List admm_ovglasso_large_m(const arma::mat& A, const arma::colvec& b,
                                 arma::mat& groups, arma::vec& group_weights, arma::vec& var_weights,
                                 arma::colvec& u, arma::colvec& z, const double lambda,
                                 bool rho_adaptation, double rho, const double tau, const double mu,
                                 const double reltol, const double abstol, const int maxiter, const int ping);


Rcpp::List admm_ovglasso_large_n(const arma::mat& A, const arma::colvec& b,
                                 arma::mat& groups, arma::vec& group_weights, arma::vec& var_weights,
                                 arma::colvec& u, arma::colvec& z, const double lambda,
                                 bool rho_adaptation, double rho, const double tau, const double mu,
                                 const double reltol, const double abstol, const int maxiter, const int ping);

// =========================================================
// Fast algorithms for efficiently computing the whole path
// ADMM for linear regression models with LASSO penalty
Rcpp::List admm_lasso_fast(const arma::mat& A, arma::vec& b, arma::vec lambda,
                           bool rho_adaptation, double rho = 1, const double tau = 2, const double mu = 10,
                           const double reltol = 1e-5, const double abstol = 1e-5, const int maxiter = 100, const int ping = 0);

// =========================================================
// Fast algorithms for efficiently computing the whole path
// ADMM for linear regression models with Group-LASSO penalty
Rcpp::List admm_glasso_fast(const arma::mat& A, arma::vec& b,
                            arma::mat& groups, arma::vec& group_weights, arma::vec& var_weights,
                            const arma::vec lambda, bool rho_adaptation, double rho = 1, const double tau = 2, const double mu = 10,
                            const double reltol = 1e-5, const double abstol = 1e-5, const int maxiter = 100, const int ping = 0);


Rcpp::List admm_glasso_large_m_fast(const arma::mat& A, arma::vec& b,
                                    arma::mat& groups, arma::vec& group_weights, arma::vec& var_weights,
                                    const arma::vec lambda, bool rho_adaptation, double rho, const double tau, const double mu,
                                    const double reltol, const double abstol, const int maxiter, const int ping);

Rcpp::List admm_glasso_large_n_fast(const arma::mat& A, arma::vec& b,
                                    arma::mat& groups, arma::vec& group_weights, arma::vec& var_weights,
                                    const arma::vec lambda, bool rho_adaptation, double rho, const double tau, const double mu,
                                    const double reltol, const double abstol, const int maxiter, const int ping);

// =========================================================
// Fast algorithms for efficiently computing the whole path
// ADMM for linear regression models with sparse Group-LASSO penalty
Rcpp::List admm_spglasso_fast(const arma::mat& A, arma::vec& b,
                              arma::mat& groups, arma::vec& group_weights,
                              arma::vec& var_weights, arma::vec& var_weights_L1,
                              const arma::vec lambda, const double alpha,
                              bool rho_adaptation, double rho = 1, const double tau = 2, const double mu = 10,
                              const double reltol = 1e-5, const double abstol = 1e-5, const int maxiter = 100, const int ping = 0);

Rcpp::List admm_spglasso_large_m_fast(const arma::mat& A, arma::vec& b,
                                      arma::mat& groups, arma::vec& group_weights,
                                      arma::vec& var_weights, arma::vec& var_weights_L1,
                                      const arma::vec lambda, const double alpha,
                                      bool rho_adaptation, double rho, const double tau, const double mu,
                                      const double reltol, const double abstol, const int maxiter, const int ping);

Rcpp::List admm_spglasso_large_n_fast(const arma::mat& A, arma::vec& b,
                                      arma::mat& groups, arma::vec& group_weights, arma::vec& var_weights, arma::vec& var_weights_L1,
                                      const arma::vec lambda, const double alpha, bool rho_adaptation, double rho, const double tau, const double mu,
                                      const double reltol, const double abstol, const int maxiter, const int ping);


// =========================================================
// Fast algorithms for efficiently computing the whole path
// ADMM for linear regression models with overlap Group-LASSO penalty
Rcpp::List admm_ovglasso_fast(const arma::mat& A, arma::vec& b,
                              arma::mat& groups, arma::vec& group_weights, arma::vec& var_weights,
                              const arma::vec lambda, bool rho_adaptation, double rho, const double tau, const double mu,
                              const double reltol, const double abstol, const int maxiter, const int ping);

/*
 Overlap Group-LASSO via ADMM: fast routine for large m
 where m denotes the number of observations
*/
Rcpp::List admm_ovglasso_large_m_fast(const arma::mat& A, arma::vec& b,
                                      arma::mat& groups, arma::vec& group_weights, arma::vec& var_weights,
                                      const arma::vec lambda, bool rho_adaptation, double rho, const double tau, const double mu,
                                      const double reltol, const double abstol, const int maxiter, const int ping);

/*
 Overlap Group-LASSO via ADMM: fast routine for large n
 where n denotes the number of parameters
*/
Rcpp::List admm_ovglasso_large_n_fast(const arma::mat& A, arma::vec& b,
                                      arma::mat& groups, arma::vec& group_weights, arma::vec& var_weights,
                                      const arma::vec lambda, bool rho_adaptation, double rho, const double tau, const double mu,
                                      const double reltol, const double abstol, const int maxiter, const int ping);





Rcpp::List admm_adalasso_fast(const arma::mat& A, arma::vec& b,
                              arma::vec& var_weights, const arma::vec lambda,
                              bool rho_adaptation, double rho, const double tau, const double mu,
                              const double reltol, const double abstol, const int maxiter, const int ping);

  

/*
 Adaptive-LASSO via ADMM: fast routine for large m
 where m denotes the number of observations
*/
Rcpp::List admm_adalasso_large_m_fast(const arma::mat& A, arma::vec& b, arma::vec& var_weights, const arma::vec lambda,
                                      bool rho_adaptation, double rho, const double tau, const double mu,
                                      const double reltol, const double abstol, const int maxiter, const int ping);

  

/*
 Adaptive-LASSO via ADMM: fast routine for large n
 where n denotes the number of parameters
*/
Rcpp::List admm_adalasso_large_n_fast(const arma::mat& A, arma::vec& b, arma::vec& var_weights, const arma::vec lambda,
                                      bool rho_adaptation, double rho, const double tau, const double mu,
                                      const double reltol, const double abstol, const int maxiter, const int ping);






#endif
