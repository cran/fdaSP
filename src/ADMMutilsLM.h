#ifndef ADMMUTILSLM_H
#define ADMMUTILSLM_H

#include <RcppArmadillo.h>
#include "utils.h"

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


// admm_enet ==========================================================
arma::colvec enet_prox(arma::colvec a, const double kappa);

double enet_objfun(const arma::mat &A, const arma::colvec &b, const double lambda,
                   const double admm_glasso, const arma::colvec &x, const arma::colvec &z);

arma::mat enet_factor(arma::mat A, double rho);

// admm_genlasso ======================================================
arma::colvec genlasso_shrinkage(arma::colvec a, const double kappa);

double genlasso_objective(const arma::mat &A,const arma::colvec &b, const arma::mat &D,
                          const double lambda, const arma::colvec &x, const arma::colvec &z);

arma::mat genlasso_factor(const arma::mat &A, double rho,const arma::mat &D);

// =========================================================
// admm_lasso
// ADMM for linear regression models with LASSO penalty
// elementwise soft thresholding operator
arma::colvec lasso_prox(arma::colvec a, const double kappa);

// Linear regression model with LASSO penalty: evaluate the objective function
double lasso_objfun(arma::mat A, arma::colvec b, const double lambda,
                    arma::colvec x, arma::colvec z);

double lasso_objfun_fast(arma::mat ATA_CHOL_U, arma::colvec ATb, const double b2,
                         const double lambda, arma::colvec x, arma::colvec z, const int m, const int n);
void lasso_factor(arma::mat& U, arma::mat A, double rho);
void lasso_factor_fast(arma::mat& U, arma::mat AA, double rho, const int m, const int n);

// Update x for linreg with lasso penalty
arma::vec admm_lasso_update_x(arma::mat A, arma::mat U, arma::mat L,
                              arma::colvec ATb, arma::vec z, arma::vec u, const double rho);

/*
 LASSO via ADMM, just to compute the whole path fast.
*/
Rcpp::List admm_lasso_update_1lambda(const arma::mat& A, const arma::vec& b, const arma::mat& ATA_CHOL_U, const arma::mat& AA,
                                     const arma::colvec& ATb, const double b2, const int m, const int n,
                                     arma::colvec u, arma::colvec z, const double lambda,
                                     bool rho_adaptation, double rho = 1, const double tau = 2, const double mu = 10,
                                     const double reltol = 1e-5, const double abstol= 1e-5, const int maxiter = 100, const int ping = 0);

/*
* Adaptive LASSO via ADMM (from Stanford)
* http://stanford.edu/~boyd/papers/pdf/admm_distr_stats.pdf
* page 19 : section 3.3.1 : stopping criteria part (3.12).
*/
arma::mat adalasso_Fmat(arma::vec var_weights);

// Linear regression model with adaptive-LASSO penalty: evaluate the objective function
double adalasso_objfun(arma::mat A, arma::colvec b, const double lambda,
                       arma::colvec x, arma::colvec z, arma::colvec var_weights);

double adalasso_objfun_fast(arma::mat ATA_CHOL_U, arma::colvec ATb, const double b2,
                            const double lambda, arma::colvec x, arma::colvec z,
                            const int m, const int n, arma::colvec var_weights);

arma::vec adalasso_residual(arma::mat F, arma::colvec x, arma::colvec z);
arma::vec adalasso_dual_residual(arma::mat F, arma::colvec z, arma::colvec z_old, double rho);

// Evaluate the RIDGE matrix: returns the Cholesky (Upper triangular matrix)
arma::mat adalasso_factor_fast_large_m(arma::mat ATA, arma::mat FTF, double rho);
arma::mat adalasso_factor_fast_large_n(arma::mat X, double rho, const int m);

// =========================================================
// admm_glasso
// ADMM for linear regression models with group-LASSO penalty
arma::colvec glasso_prox(arma::colvec a, const double kappa);

// Linear regression model with GLASSO penalty: evaluate the objective function
double glasso_objfun(arma::mat A, arma::colvec b, arma::vec Glen,
                     const double lambda, arma::colvec x, arma::colvec z, const int G);

// Linear regression model with sparse GLASSO penalty: evaluate the objective function
double spglasso_objfun(arma::mat A, arma::colvec b, arma::vec Glen, 
                       const double lambda, const double alpha,
                       arma::colvec x, arma::colvec z, const int G);

double spglasso_objfun_fast(arma::mat ATA_CHOL_U, arma::colvec ATb, const double b2,
                            arma::vec Glen, const double lambda, const double alpha,
                            arma::colvec x, arma::colvec z, const int m, const int n, const int G);

double glasso_objfun_fast(arma::mat ATA_CHOL_U, arma::colvec ATb, const double b2,
                          arma::vec Glen, const double lambda, arma::colvec x, arma::colvec z,
                          const int m, const int n, const int G);

// Evaluate the RIDGE matrix: returns the Cholesky (Upper triangular matrix)
arma::mat glasso_factor(arma::mat A, arma::mat FTF, double rho);
arma::mat glasso_factor_fast_large_m(arma::mat ATA, arma::mat FTF, double rho);
arma::mat glasso_factor_fast_large_n(arma::mat X, double rho, const int m);
arma::vec glasso_residual(arma::mat F, arma::colvec x, arma::colvec z);
arma::vec glasso_dual_residual(arma::mat F, arma::colvec z, arma::colvec z_old, double rho);
arma::mat glasso_Gvec2F1mat(arma::rowvec Gvec, arma::vec weights);
arma::mat glasso_Gmat2Fmat(arma::mat Gmat,
                           arma::vec group_weights,
                           arma::vec var_weights);

arma::sp_mat glasso_Gvec2F1mat_spmat(arma::rowvec Gvec);
arma::sp_mat glasso_Gmat2Fmat_sparse(arma::mat Gmat,
                                     arma::vec group_weights,
                                     arma::vec var_weights);


arma::vec glasso_dual_residual_sparse(arma::sp_mat F, arma::colvec z, arma::colvec z_old, double rho);
arma::vec glasso_residual_sparse(arma::sp_mat F, arma::colvec x, arma::colvec z);

arma::mat spglasso_Gmat2Fmat(arma::mat Gmat,
                             arma::vec group_weights,
                             arma::vec var_weights,
                             arma::vec var_weights_L1);

arma::sp_mat spglasso_Gmat2Fmat_sparse(arma::mat Gmat,
                                       arma::vec group_weights,
                                       arma::vec var_weights,
                                       arma::vec var_weights_L1);

Rcpp::List admm_glasso_large_m_update_1lambda(const arma::mat& A, const arma::vec& b,
                                              const arma::colvec Glen, const arma::sp_mat F, const arma::mat FTF,
                                              const arma::mat ATA, const arma::mat& ATA_CHOL_U, const arma::colvec& ATb,
                                              const double b2, const int m, const int n, const int G,
                                              arma::colvec u, arma::colvec z, const double lambda,
                                              bool rho_adaptation, double rho = 1, const double tau = 2, const double mu = 10,
                                              const double reltol = 1e-5, const double abstol = 1e-5, const int maxiter = 100, const int ping = 0);


Rcpp::List admm_glasso_large_n_update_1lambda(const arma::mat& A, const arma::vec& b, const arma::colvec Glen, const arma::sp_mat F,
                                              const arma::mat FTF_INV, const arma::mat AFTF_INVA,
                                              const arma::mat& ATA_CHOL_U, const arma::colvec& ATb,
                                              const double b2, const int m, const int n, const int G,
                                              arma::colvec u, arma::colvec z, const double lambda,
                                              bool rho_adaptation, double rho = 1, const double tau = 2, const double mu = 10,
                                              const double reltol = 1e-5, const double abstol = 1e-5, const int maxiter = 100, const int ping = 0);

Rcpp::List admm_spglasso_large_m_update_1lambda(const arma::mat& A, const arma::vec& b,
                                                const arma::colvec Glen, const arma::sp_mat F, const arma::mat FTF,
                                                const arma::mat ATA, const arma::mat& ATA_CHOL_U, const arma::colvec& ATb,
                                                const double b2, const int m, const int n, const int G,
                                                arma::colvec u, arma::colvec z, const double lambda, const double alpha,
                                                bool rho_adaptation, double rho = 1, const double tau = 2, const double mu = 10,
                                                const double reltol = 1e-5, const double abstol = 1e-5, const int maxiter = 100, const int ping = 0);


Rcpp::List admm_spglasso_large_n_update_1lambda(const arma::mat& A, arma::vec& b, const arma::colvec Glen, const arma::sp_mat F,
                                                const arma::mat FTF_INV, const arma::mat AFTF_INVA,
                                                const arma::mat& ATA_CHOL_U, const arma::colvec& ATb,
                                                const double b2, const int m, const int n, const int G,
                                                arma::colvec u, arma::colvec z, const double lambda, const double alpha,
                                                bool rho_adaptation, double rho = 1, const double tau = 2, const double mu = 10,
                                                const double reltol = 1e-5, const double abstol = 1e-5, const int maxiter = 100, const int ping = 0);


Rcpp::List admm_ovglasso_large_m_update_1lambda(const arma::mat& A, const arma::vec& b,
                                                const arma::colvec Glen, const arma::sp_mat F, const arma::mat FTF,
                                                const arma::mat ATA, const arma::mat& ATA_CHOL_U, const arma::colvec& ATb,
                                                const double b2, const int m, const int n, const int G,
                                                arma::colvec u, arma::colvec z, const double lambda,
                                                bool rho_adaptation, double rho = 1, const double tau = 2, const double mu = 10,
                                                const double reltol = 1e-5, const double abstol = 1e-5, const int maxiter = 100, const int ping = 0);
  
Rcpp::List admm_ovglasso_large_n_update_1lambda(const arma::mat& A, const arma::vec& b, const arma::colvec Glen, const arma::sp_mat F,
                                                const arma::mat FTF_INV, const arma::mat AFTF_INVA,
                                                const arma::mat& ATA_CHOL_U, const arma::colvec& ATb,
                                                const double b2, const int m, const int n, const int G,
                                                arma::colvec u, arma::colvec z, const double lambda,
                                                bool rho_adaptation, double rho = 1, const double tau = 2, const double mu = 10,
                                                const double reltol = 1e-5, const double abstol = 1e-5, const int maxiter = 100, const int ping = 0);

Rcpp::List admm_adalasso_large_m_update_1lambda(const arma::mat& A, const arma::vec& b,
                                                const arma::mat F, const arma::mat FTF,
                                                const arma::mat ATA, const arma::mat& ATA_CHOL_U, const arma::colvec& ATb,
                                                const double b2, const int m, const int n,
                                                arma::colvec u, arma::colvec z, const double lambda,
                                                bool rho_adaptation, double rho, const double tau, const double mu,
                                                const double reltol, const double abstol, const int maxiter, const int ping);

Rcpp::List admm_adalasso_large_n_update_1lambda(const arma::mat& A, const arma::vec& b, const arma::mat F,
                                                const arma::mat FTF_INV, const arma::mat AFTF_INVA,
                                                const arma::mat& ATA_CHOL_U, const arma::colvec& ATb,
                                                const double b2, const int m, const int n, 
                                                arma::colvec u, arma::colvec z, const double lambda,
                                                bool rho_adaptation, double rho, const double tau, const double mu,
                                                const double reltol, const double abstol, const int maxiter, const int ping);



#endif
