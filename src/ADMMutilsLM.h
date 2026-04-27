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

// =========================================================
// admm_enet
arma::colvec enet_prox(const arma::colvec& a,
                       const double kappa);
double enet_objfun(const arma::mat& A,
                   const arma::colvec& b,
                   const double lambda,
                   const double admm_glasso,
                   const arma::colvec& x,
                   const arma::colvec& z);
arma::mat enet_factor(const arma::mat& A,
                      const double rho);

// =========================================================
// admm_genlasso
arma::colvec genlasso_shrinkage(const arma::colvec& a, 
                                const double kappa);
double genlasso_objective(const arma::mat& A,
                          const arma::colvec& b,
                          const arma::mat& D,
                          const double lambda,
                          const arma::colvec& x,
                          const arma::colvec& z);
arma::mat genlasso_factor(const arma::mat& A,
                          const double rho,
                          const arma::mat& D);

// =========================================================
// admm_lasso
// ADMM for linear regression models with LASSO penalty
// elementwise soft thresholding operator
arma::colvec lasso_prox(const arma::colvec& a,
                        const double kappa);

// Linear regression model with LASSO penalty: evaluate the objective function
double lasso_objfun(const arma::mat& A,
                    const arma::colvec& b,
                    const double lambda,
                    const arma::colvec& x,
                    double& rss,
                    double& pen);
void lasso_factor(const arma::mat& U,
                  const arma::mat A,
                  const double rho);
void lasso_factor_fast(arma::mat& U,
                       const arma::mat& AA,
                       const double rho,
                       const int m,
                       const int n);
double lasso_objfun_fast(const arma::mat& ATA_CHOL_U,
                         const arma::colvec& ATb,
                         const double b2,
                         const double lambda, 
                         const arma::colvec& x,
                         const arma::colvec& z,
                         const int m,
                         const int n);
// Update x for linreg with lasso penalty
arma::vec admm_lasso_update_x(const arma::mat& A,
                              const arma::mat& U,
                              const arma::mat& L,
                              const arma::colvec& ATb,
                              const arma::vec& z,
                              const arma::vec& u,
                              const double rho);

/*   Adaptive LASSO via ADMM                    */
arma::colvec adalasso_prox(const arma::colvec& a,
                           const arma::vec kappa);
arma::mat adalasso_Fmat(const arma::vec& var_weights);

// Linear regression model with adaptive-LASSO penalty: evaluate the objective function
double adalasso_objfun(const arma::mat& A,
                       const arma::colvec& b,
                       const double lambda,
                       const arma::colvec& x,
                       const arma::vec& var_weights,
                       double& rss,
                       double& pen);
arma::vec adalasso_residual(const arma::colvec& x,
                            const arma::colvec& z);
arma::vec adalasso_dual_residual(const arma::colvec& z,
                                 const arma::colvec& z_old,
                                 double rho);
// Evaluate the RIDGE matrix: returns the Cholesky (Upper triangular matrix)
arma::mat adalasso_factor_fast_large_m(const arma::mat& ATA,
                                       const double rho);
arma::mat adalasso_factor_fast_large_n(const arma::mat& X,
                                       const double rho,
                                       const int m);

// =========================================================
// admm_glasso
// ADMM for linear regression models with group-LASSO penalty
arma::colvec glasso_prox(const arma::colvec& a,
                         const double kappa);

// Evaluate the RIDGE matrix: returns the Cholesky (Upper triangular matrix)
arma::mat glasso_factor(const arma::mat& A,
                        const arma::mat& FTF,
                        const double rho);
arma::mat glasso_factor_fast_large_m(const arma::mat& ATA,
                                     const arma::mat& FTF,
                                     const double rho);
arma::mat glasso_factor_smo_fast_large_m(const arma::mat& ATA,
                                         const arma::mat& FTF,
                                         const arma::mat& DTD,
                                         const double rho,
                                         const double lambda);
arma::mat glasso_factor_smo_fast_large_m2(const arma::mat& XtTXt,
                                          const arma::mat& FTF,
                                          const double rho);
arma::mat glasso_factor_fast_large_n(const arma::mat& X,
                                     const double rho,
                                     const int m);
arma::mat glasso_factor_smo_fast_large_n(const arma::mat& X,
                                         const int m);
arma::mat glasso_factor_smo_fast_large_n2(const arma::mat& X,
                                          const double rho,
                                          const int m);
arma::vec glasso_residual(const arma::mat& F,
                          const arma::colvec& x,
                          const arma::colvec& z);
arma::vec glasso_dual_residual(const arma::mat& F,
                               const arma::colvec& z,
                               const arma::colvec& z_old,
                               const double rho);
arma::vec glasso_residual_sparse(const arma::sp_mat& F,
                                 const arma::colvec& x,
                                 const arma::colvec& z);
arma::vec glasso_dual_residual_sparse(const arma::sp_mat& F,
                                      const arma::colvec& z,
                                      const arma::colvec& z_old,
                                      const double rho);
arma::mat glasso_Gvec2F1mat(const arma::rowvec& Gvec);
arma::sp_mat glasso_Gvec2F1mat_spmat(const arma::rowvec& Gvec);
arma::mat glasso_Gmat2Fmat(const arma::mat& Gmat,
                           const arma::vec& group_weights,
                           const arma::vec& var_weights);
arma::sp_mat glasso_Gmat2Fmat_sparse(const arma::mat& Gmat,
                                     const arma::vec& var_weights);

arma::sp_mat ovglasso_Gmat2Fmat_sparse(const arma::mat& Gmat,
                                       const arma::vec& var_weights);

// admm_spglasso
arma::mat spglasso_Gmat2Fmat(const arma::mat& Gmat,
                             const arma::vec& group_weights,
                             const arma::vec& var_weights,
                             const arma::vec& var_weights_L1);


arma::sp_mat spglasso_Gmat2Fmat_sparse(const arma::mat& Gmat,
                                       const arma::vec& var_weights,
                                       const arma::vec& var_weights_L1);

arma::mat glasso_Gmat2Dmat(const arma::mat& Gmat);

// LASSO penalty: evaluate the RIDGE matrix, returns the Cholesky (Upper triangular matrix)
arma::mat lasso_factor_cov_fast_large_m(const arma::mat& XMX,
                                        double rho);

arma::mat lasso_factor_cov_fast_large_n(const arma::mat& XX,
                                        double rho,
                                        const int m);

// Group-LASSO penalty: evaluate the RIDGE matrix, returns the Cholesky (Upper triangular matrix)
arma::mat glasso_factor_cov_fast_large_m(const arma::mat& XMX,
                                         const arma::mat& FTF,
                                         double rho);

// previous code with smoothing
arma::mat glasso_factor_cov_smo_fast_large_m(const arma::mat& XMX,
                                             const arma::mat& FTF,
                                             const arma::mat& DTD,
                                             const double lambda,
                                             double rho);

arma::mat glasso_factor_cov_fast_large_n(const arma::mat& XMX,
                                         double rho,
                                         const int m);

// stesso codice precedente ma con smoothing effects
arma::mat glasso_factor_cov_smo_fast_large_n(const arma::mat& X,
                                             double rho,
                                             const int m);

// Linear regression model with LASSO and adaptive-LASSO penalty:
// evaluate the penalty function
double lm_rss(const arma::mat& A,
              const arma::vec& b,
              const arma::vec& x);
double lm_rss_fast(const arma::mat& ATA_CHOL_U,
                   const arma::vec& ATb,
                   const double b2,
                   const arma::vec& x,
                   const int m,
                   const int n);
double lm_cov_rss_fast(const arma::mat& XX_CHOL_U,
                       const arma::mat& ZZ_CHOL_U,
                       const arma::mat& ZX,
                       const arma::vec& Xy,
                       const arma::vec& Zy,
                       const double y2,
                       const arma::colvec& x,
                       const arma::colvec& v,
                       const int m);
double lasso_penalty(const arma::vec& x);
double adalasso_penalty(const arma::vec& x,
                        const arma::vec& var_weights);

// Linear regression model with group-LASSO penalty:
// evaluate the penalty function
double glasso_penalty(const arma::vec& x,
                      const arma::mat& groups,
                      const arma::vec& group_weights,
                      const arma::vec& var_weights);

// Linear regression model with sparse group-LASSO penalty:
// evaluate the penalty function
double spglasso_penalty(const arma::vec& x,
                        const arma::mat& groups,
                        const arma::vec& group_weights,
                        const arma::vec& var_weights,
                        const arma::vec& var_weights_L1,
                        const double alpha);

// Linear regression model with overlap group-LASSO penalty:
// evaluate the penalty function
double ovglasso_penalty(const arma::vec& x,
                        const arma::mat& groups,
                        const arma::vec& group_weights,
                        const arma::vec& var_weights);

arma::sp_mat spovglasso_Gmat2Fmat_sparse(const arma::mat& Gmat,
                                         const arma::vec& var_weights,
                                         const arma::vec& var_weights_L1);

double spovglasso_penalty(const arma::vec& x,
                          const arma::mat& groups,
                          const arma::vec& group_weights,
                          const arma::vec& var_weights,
                          const arma::vec& var_weights_L1,
                          const double alpha);








#endif
