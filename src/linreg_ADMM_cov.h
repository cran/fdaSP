#ifndef LINREG_ADMM_COV_H
#define LINREG_ADMM_COV_H

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

// Evaluate the RIDGE matrix: returns the Cholesky (Upper triangular matrix)
arma::mat lasso_factor_cov_fast_large_m(arma::mat XMX, double rho);
arma::mat lasso_factor_cov_fast_large_n(arma::mat XX, double rho, const int m);
arma::mat glasso_factor_cov_fast_large_m(arma::mat XMX, arma::mat FTF, double rho);
arma::mat glasso_factor_cov_fast_large_n(arma::mat XMX, double rho, const int m);

// Objective functions
double lasso_cov_objfun_fast(arma::mat XX_CHOL_U, arma::mat ZZ_CHOL_U, arma::mat ZX, arma::colvec Xy, arma::colvec Zy, const double y2,
                             const double lambda, arma::colvec x, arma::colvec v, arma::colvec z, const int m);

double linreg_cov_logl_fast(arma::mat XX_CHOL_U, arma::mat ZZ_CHOL_U, arma::mat ZX,
                            arma::colvec Xy, arma::colvec Zy, const double y2,
                            arma::colvec x, arma::colvec v, const int m);

double lasso_cov_penfun(const arma::colvec z, const double lambda);
double glasso_cov_penfun(const arma::vec Glen, const int G, const arma::colvec z, const double lambda);
double spglasso_cov_penfun(const arma::vec Glen, const int G, const int n, const arma::colvec z,
                           const double lambda, const double alpha);
double adalasso_cov_penfun(const arma::colvec z, const double lambda, const arma::colvec var_weights);

// Linear regression model with LASSO penalty and covarates:
Rcpp::List admm_lasso_cov_large_m_update_1lambda(arma::mat& W, arma::mat Z, arma::vec& y,
                                                 const arma::mat WMW, const arma::vec WMy, const arma::vec v_LS, const arma::mat P_ZW,
                                                 const arma::mat& WW_CHOL_U, const arma::colvec& Wy, const double y2,
                                                 const arma::mat& ZZ_CHOL_U, const arma::colvec& Zy, const arma::mat ZW,
                                                 arma::colvec u, arma::colvec z, const double lambda,
                                                 bool rho_adaptation, double rho, const double tau, const double mu,
                                                 const double reltol, const double abstol, const int maxiter, const int ping);

Rcpp::List admm_lasso_cov_large_n_update_1lambda(arma::mat& W, arma::mat Z, arma::vec& y, const arma::mat WW, const arma::mat M_ZZW, const arma::mat M_ZZW2,
                                                 const arma::mat WMW, const arma::vec WMy, const arma::vec v_LS, const arma::mat P_ZW,
                                                 const arma::mat& WW_CHOL_U, const arma::colvec& Wy, const double y2,
                                                 const arma::mat& ZZ_CHOL_U, const arma::colvec& Zy, const arma::mat ZW,
                                                 arma::colvec u, arma::colvec z, const double lambda,
                                                 bool rho_adaptation, double rho, const double tau, const double mu,
                                                 const double reltol, const double abstol, const int maxiter, const int ping);

Rcpp::List admm_lasso_cov_large_m_fast(arma::mat& W, arma::mat Z, arma::vec& y,
                                       const arma::vec lambda, bool rho_adaptation, double rho, const double tau, const double mu,
                                       const double reltol, const double abstol, const int maxiter, const int ping);

Rcpp::List admm_lasso_cov_large_n_fast(arma::mat& W, arma::mat Z, arma::vec& y,
                                       const arma::vec lambda, bool rho_adaptation, double rho, const double tau, const double mu,
                                       const double reltol, const double abstol, const int maxiter, const int ping);

Rcpp::List admm_lasso_cov_fast(arma::mat& W, arma::mat Z, arma::vec& y,
                               const arma::vec lambda, bool rho_adaptation, double rho, const double tau, const double mu,
                               const double reltol, const double abstol, const int maxiter, const int ping);

// Linear regression model with group-LASSO penalty and covarates:
Rcpp::List admm_glasso_cov_large_m_update_1lambda(arma::mat W, arma::mat Z, arma::vec& y,
                                                  const arma::colvec Glen, const int G, const arma::sp_mat F, const arma::mat FTF,
                                                  const arma::mat WMW, const arma::vec WMy, const arma::vec v_LS, const arma::mat P_ZW,
                                                  const arma::mat& WW_CHOL_U, const arma::colvec& Wy, const double y2,
                                                  const arma::mat& ZZ_CHOL_U, const arma::colvec& Zy, const arma::mat ZW,
                                                  arma::colvec u, arma::colvec z, const double lambda,
                                                  bool rho_adaptation, double rho, const double tau, const double mu,
                                                  const double reltol, const double abstol, const int maxiter, const int ping);

Rcpp::List admm_glasso_cov_large_n_update_1lambda(arma::mat& W, arma::mat& Z, arma::vec& y, const arma::colvec Glen, const int G,
                                                  const arma::sp_mat F, const arma::mat FF_INV,
                                                  const arma::mat WW, const arma::mat M_ZZW, const arma::mat M_ZZW2,
                                                  const arma::mat WMW, const arma::vec WMy, const arma::vec v_LS, const arma::mat P_ZW,
                                                  const arma::mat& WW_CHOL_U, const arma::colvec& Wy, const double y2,
                                                  const arma::mat& ZZ_CHOL_U, const arma::colvec& Zy, const arma::mat ZW,
                                                  arma::colvec u, arma::colvec z, const double lambda,
                                                  bool rho_adaptation, double rho, const double tau, const double mu,
                                                  const double reltol, const double abstol, const int maxiter, const int ping);

Rcpp::List admm_glasso_cov_large_m_fast(arma::mat& W, arma::mat &Z, arma::vec& y,
                                        arma::mat& groups, arma::vec& group_weights, arma::vec& var_weights,
                                        const arma::vec lambda, bool rho_adaptation, double rho, const double tau, const double mu,
                                        const double reltol, const double abstol, const int maxiter, const int ping);

Rcpp::List admm_glasso_cov_large_n_fast(arma::mat& W, arma::mat Z, arma::vec& y,
                                        arma::mat& groups, arma::vec& group_weights, arma::vec& var_weights,
                                        const arma::vec lambda, bool rho_adaptation, double rho, const double tau, const double mu,
                                        const double reltol, const double abstol, const int maxiter, const int ping);

Rcpp::List admm_glasso_cov_fast(arma::mat& W, arma::mat Z, arma::vec& y,
                                arma::mat& groups, arma::vec& group_weights, arma::vec& var_weights,
                                const arma::vec lambda, bool rho_adaptation, double rho, const double tau, const double mu,
                                const double reltol, const double abstol, const int maxiter, const int ping);

// Linear regression model with overlap group-LASSO penalty and covarates:
Rcpp::List admm_ovglasso_cov_large_m_update_1lambda(arma::mat& W, arma::mat& Z, arma::vec& y,
                                                    const arma::colvec Glen, const int G, const arma::sp_mat F, const arma::mat FTF,
                                                    const arma::mat WMW, const arma::vec WMy, const arma::vec v_LS, const arma::mat P_ZW,
                                                    const arma::mat& WW_CHOL_U, const arma::colvec& Wy, const double y2,
                                                    const arma::mat& ZZ_CHOL_U, const arma::colvec& Zy, const arma::mat ZW,
                                                    arma::colvec u, arma::colvec z, const double lambda,
                                                    bool rho_adaptation, double rho, const double tau, const double mu,
                                                    const double reltol, const double abstol, const int maxiter, const int ping);

Rcpp::List admm_ovglasso_cov_large_n_update_1lambda(arma::mat& W, arma::mat& Z, arma::vec& y, const arma::colvec Glen, const int G,
                                                    const arma::sp_mat F, const arma::mat FF_INV,
                                                    const arma::mat WW, const arma::mat M_ZZW, const arma::mat M_ZZW2,
                                                    const arma::mat WMW, const arma::vec WMy, const arma::vec v_LS, const arma::mat P_ZW,
                                                    const arma::mat& WW_CHOL_U, const arma::colvec& Wy, const double y2,
                                                    const arma::mat& ZZ_CHOL_U, const arma::colvec& Zy, const arma::mat ZW,
                                                    arma::colvec u, arma::colvec z, const double lambda,
                                                    bool rho_adaptation, double rho, const double tau, const double mu,
                                                    const double reltol, const double abstol, const int maxiter, const int ping);

Rcpp::List admm_ovglasso_cov_large_m_fast(arma::mat& W, arma::mat& Z, arma::vec& y,
                                          arma::mat& groups, arma::vec& group_weights, arma::vec& var_weights,
                                          const arma::vec lambda, bool rho_adaptation, double rho, const double tau, const double mu,
                                          const double reltol, const double abstol, const int maxiter, const int ping);

Rcpp::List admm_ovglasso_cov_large_n_fast(arma::mat& W, arma::mat& Z, arma::vec& y,
                                          arma::mat& groups, arma::vec& group_weights, arma::vec& var_weights,
                                          const arma::vec lambda, bool rho_adaptation, double rho, const double tau, const double mu,
                                          const double reltol, const double abstol, const int maxiter, const int ping);

Rcpp::List admm_ovglasso_cov_fast(arma::mat& W, arma::mat& Z, arma::vec& y,
                                  arma::mat& groups, arma::vec& group_weights, arma::vec& var_weights,
                                  const arma::vec lambda, bool rho_adaptation, double rho, const double tau, const double mu,
                                  const double reltol, const double abstol, const int maxiter, const int ping);

// Linear regression model with adaptive-LASSO penalty and covarates:
Rcpp::List admm_adalasso_cov_large_n_update_1lambda(arma::mat& W, arma::mat& Z, arma::vec& y,
                                                    const arma::mat F, const arma::mat FF_INV,
                                                    const arma::mat WW, const arma::mat M_ZZW, const arma::mat M_ZZW2,
                                                    const arma::mat WMW, const arma::vec WMy, const arma::vec v_LS, const arma::mat P_ZW,
                                                    const arma::mat& WW_CHOL_U, const arma::colvec& Wy, const double y2,
                                                    const arma::mat& ZZ_CHOL_U, const arma::colvec& Zy, const arma::mat ZW,
                                                    arma::colvec u, arma::colvec z, const double lambda,
                                                    bool rho_adaptation, double rho, const double tau, const double mu,
                                                    const double reltol, const double abstol, const int maxiter, const int ping);

Rcpp::List admm_adalasso_cov_large_m_update_1lambda(arma::mat& W, arma::mat& Z, arma::vec& y,
                                                    const arma::mat F, const arma::mat FTF,
                                                    const arma::mat WMW, const arma::vec WMy, const arma::vec v_LS, const arma::mat P_ZW,
                                                    const arma::mat& WW_CHOL_U, const arma::colvec& Wy, const double y2,
                                                    const arma::mat& ZZ_CHOL_U, const arma::colvec& Zy, const arma::mat ZW,
                                                    arma::colvec u, arma::colvec z, const double lambda,
                                                    bool rho_adaptation, double rho, const double tau, const double mu,
                                                    const double reltol, const double abstol, const int maxiter, const int ping);

Rcpp::List admm_adalasso_cov_large_m_fast(arma::mat& W, arma::mat& Z, arma::vec& y, arma::vec& var_weights,
                                          const arma::vec lambda, bool rho_adaptation, double rho, const double tau, const double mu,
                                          const double reltol, const double abstol, const int maxiter, const int ping);

Rcpp::List admm_adalasso_cov_large_n_fast(arma::mat& W, arma::mat& Z, arma::vec& y, arma::vec& var_weights,
                                          const arma::vec lambda, bool rho_adaptation, double rho, const double tau, const double mu,
                                          const double reltol, const double abstol, const int maxiter, const int ping);

Rcpp::List admm_adalasso_cov_fast(arma::mat& W, arma::mat& Z, arma::vec& y, arma::vec& var_weights,
                                  const arma::vec lambda, bool rho_adaptation, double rho, const double tau, const double mu,
                                  const double reltol, const double abstol, const int maxiter, const int ping);

// Linear regression model with sparse group-LASSO penalty and covarates:
Rcpp::List admm_spglasso_cov_large_m_update_1lambda(arma::mat& W, arma::mat& Z, arma::vec& y,
                                                    const arma::colvec Glen, const int G, const arma::sp_mat F, const arma::mat FTF,
                                                    const arma::mat WMW, const arma::vec WMy, const arma::vec v_LS, const arma::mat P_ZW,
                                                    const arma::mat& WW_CHOL_U, const arma::colvec& Wy, const double y2,
                                                    const arma::mat& ZZ_CHOL_U, const arma::colvec& Zy, const arma::mat ZW,
                                                    arma::colvec u, arma::colvec z, const double lambda, const double alpha,
                                                    bool rho_adaptation, double rho, const double tau, const double mu,
                                                    const double reltol, const double abstol, const int maxiter, const int ping);

Rcpp::List admm_spglasso_cov_large_m_fast(arma::mat& W, arma::mat& Z, arma::vec& y,
                                          arma::mat& groups, arma::vec& group_weights, arma::vec& var_weights, arma::vec& var_weights_L1,
                                          const arma::vec lambda, const double alpha, bool rho_adaptation, double rho, const double tau, const double mu,
                                          const double reltol, const double abstol, const int maxiter, const int ping);

Rcpp::List admm_spglasso_cov_large_n_update_1lambda(arma::mat& W, arma::mat& Z, arma::vec& y, const arma::colvec Glen, const int G,
                                                    const arma::sp_mat F, const arma::mat FF_INV,
                                                    const arma::mat WW, const arma::mat M_ZZW, const arma::mat M_ZZW2,
                                                    const arma::mat WMW, const arma::vec WMy, const arma::vec v_LS, const arma::mat P_ZW,
                                                    const arma::mat& WW_CHOL_U, const arma::colvec& Wy, const double y2,
                                                    const arma::mat& ZZ_CHOL_U, const arma::colvec& Zy, const arma::mat ZW,
                                                    arma::colvec u, arma::colvec z, const double lambda, const double alpha,
                                                    bool rho_adaptation, double rho, const double tau, const double mu,
                                                    const double reltol, const double abstol, const int maxiter, const int ping);

Rcpp::List admm_spglasso_cov_large_n_fast(arma::mat& W, arma::mat& Z, arma::vec& y,
                                          arma::mat& groups, arma::vec& group_weights, arma::vec& var_weights,
                                          const arma::vec lambda, const double alpha, bool rho_adaptation, double rho, const double tau, const double mu,
                                          const double reltol, const double abstol, const int maxiter, const int ping);

Rcpp::List admm_spglasso_cov_fast(arma::mat& W, arma::mat& Z, arma::vec& y,
                                  arma::mat& groups, arma::vec& group_weights, arma::vec& var_weights,
                                  const arma::vec lambda, const double alpha, bool rho_adaptation, double rho, const double tau, const double mu,
                                  const double reltol, const double abstol, const int maxiter, const int ping);


// =========================================================
// Linear regression model with (LASSO, GLASSO, spGLASSO and OVGLASSO penalty)
// These function estimate the model for a single lambda (fast routines are implemented)
Rcpp::List admm_lasso_cov_large_m(arma::mat& W, arma::mat& Z, const arma::colvec& y,
                                  arma::colvec& u, arma::colvec& z, const double lambda,
                                  bool rho_adaptation, double rho, const double tau, const double mu,
                                  const double reltol, const double abstol, const int maxiter, const int ping);

Rcpp::List admm_lasso_cov_large_n(arma::mat& W, arma::mat& Z, const arma::colvec& y,
                                  arma::colvec& u, arma::colvec& z, const double lambda,
                                  bool rho_adaptation, double rho, const double tau, const double mu,
                                  const double reltol, const double abstol, const int maxiter, const int ping);

Rcpp::List admm_glasso_cov_large_m(arma::mat& W, arma::mat& Z, arma::vec& y, arma::colvec& u, arma::colvec& z,
                                   arma::mat& groups, arma::vec& group_weights, arma::vec& var_weights,
                                   double lambda, bool rho_adaptation, double rho, const double tau, const double mu,
                                   const double reltol, const double abstol, const int maxiter, const int ping);

Rcpp::List admm_glasso_cov_large_n(arma::mat& W, arma::mat Z, arma::vec& y, arma::colvec& u, arma::colvec& z,
                                   arma::mat& groups, arma::vec& group_weights, arma::vec& var_weights,
                                   double lambda, bool rho_adaptation, double rho, const double tau, const double mu,
                                   const double reltol, const double abstol, const int maxiter, const int ping);


Rcpp::List admm_ovglasso_cov_large_m(arma::mat& W, arma::mat& Z, arma::vec& y, arma::colvec& u, arma::colvec& z,
                                     arma::mat& groups, arma::vec& group_weights, arma::vec& var_weights,
                                     double lambda, bool rho_adaptation, double rho, const double tau, const double mu,
                                     const double reltol, const double abstol, const int maxiter, const int ping);

Rcpp::List admm_ovglasso_cov_large_n(arma::mat& W, arma::mat& Z, arma::vec& y, arma::colvec& u, arma::colvec& z,
                                     arma::mat& groups, arma::vec& group_weights, arma::vec& var_weights,
                                     double, bool rho_adaptation, double rho, const double tau, const double mu,
                                     const double reltol, const double abstol, const int maxiter, const int ping);

Rcpp::List admm_adalasso_cov_large_m(arma::mat& W, arma::mat& Z, arma::vec& y, arma::vec& var_weights, arma::colvec& u, arma::colvec& z,
                                     double lambda, bool rho_adaptation, double rho, const double tau, const double mu,
                                     const double reltol, const double abstol, const int maxiter, const int ping);

Rcpp::List admm_adalasso_cov_large_n(arma::mat& W, arma::mat& Z, arma::vec& y, arma::vec& var_weights, arma::colvec& u, arma::colvec& z,
                                     double lambda, bool rho_adaptation, double rho, const double tau, const double mu,
                                     const double reltol, const double abstol, const int maxiter, const int ping);

Rcpp::List admm_spglasso_cov_large_m(arma::mat& W, arma::mat& Z, arma::vec& y, arma::colvec& u, arma::colvec& z,
                                     arma::mat& groups, arma::vec& group_weights, arma::vec& var_weights, arma::vec& var_weights_L1,
                                     double lambda, const double alpha, bool rho_adaptation, double rho, const double tau, const double mu,
                                     const double reltol, const double abstol, const int maxiter, const int ping);

Rcpp::List admm_spglasso_cov_large_n(arma::mat& W, arma::mat& Z, arma::vec& y, arma::colvec& u, arma::colvec& z,
                                     arma::mat& groups, arma::vec& group_weights, arma::vec& var_weights, arma::vec& var_weights_L1,
                                     double, const double alpha, bool rho_adaptation, double rho, const double tau, const double mu,
                                     const double reltol, const double abstol, const int maxiter, const int ping);


// =========================================================
// Linear regression model with (LASSO, GLASSO, spGLASSO and OVGLASSO penalty)
// WRAP FUNCTIONS
Rcpp::List admm_lasso_cov(arma::mat& W, arma::mat& Z, const arma::colvec& y,
                          arma::colvec& u, arma::colvec& z, const double lambda,
                          bool rho_adaptation, double rho, const double tau, const double mu,
                          const double reltol, const double abstol, const int maxiter, const int ping);

Rcpp::List admm_glasso_cov(arma::mat& W, arma::mat& Z, arma::vec& y, arma::colvec& u, arma::colvec& z,
                           arma::mat& groups, arma::vec& group_weights, arma::vec& var_weights,
                           double lambda, bool rho_adaptation, double rho, const double tau, const double mu,
                           const double reltol, const double abstol, const int maxiter, const int ping);

Rcpp::List admm_ovglasso_cov(arma::mat& W, arma::mat& Z, arma::vec& y, arma::colvec& u, arma::colvec& z,
                             arma::mat& groups, arma::vec& group_weights, arma::vec& var_weights,
                             double lambda, bool rho_adaptation, double rho, const double tau, const double mu,
                             const double reltol, const double abstol, const int maxiter, const int ping);

Rcpp::List admm_adalasso_cov(arma::mat W, arma::mat& Z, arma::colvec& y, arma::vec& var_weights,
                             arma::colvec& u, arma::colvec& z, const double lambda,
                             bool rho_adaptation, double rho, const double tau, const double mu,
                             const double reltol, const double abstol, const int maxiter, const int ping);

Rcpp::List admm_spglasso_cov(arma::mat& W, arma::mat& Z, arma::vec& y,
                             arma::mat& groups, arma::vec& group_weights,
                             arma::vec& var_weights, arma::vec& var_weights_L1,
                             arma::colvec& u, arma::colvec& z, const double lambda1, const double lambda2,
                             bool rho_adaptation, double rho, const double tau, const double mu,
                             const double reltol, const double abstol, const int maxiter, const int ping);

#endif
