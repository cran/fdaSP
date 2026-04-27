#ifndef DOF_UTILS
#define DOF_UTILS

#define ARMA_USE_SUPERLU 1

#include <RcppArmadillo.h>

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(cpp11)]]

//#include "ADMMutilsLM.h"
#include "linreg_lasso_ADMM.h"

using std::pow;
using std::exp;
using std::sqrt;
using std::log;

using namespace Rcpp;
using namespace arma;

#define DOUBLE_EPS 2.220446e-16
#define SAFE_LOG(a) (((a) <= 0.0) ? log(DOUBLE_EPS) : log(a))
#define SAFE_ZERO(a) ((a) == 0 ? DOUBLE_EPS : (a))
#define SQRT_DOUBLE_EPS sqrt(DOUBLE_EPS)

/* ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    utilities
   ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::     */
arma::umat ovglasso_group_selmat(const arma::uvec& y,
                                 const arma::uvec& x);

Rcpp::List lm_OVGLASSO_admm_active_sets(const arma::vec& parms,
                                        const double lambda,
                                        const arma::mat& GRmat,
                                        const arma::vec& group_weights,
                                        const arma::vec& var_weights,
                                        const arma::vec& Uvec,
                                        const double err_primal,
                                        const double err_dual,
                                        const double rho,
                                        const double toler_c,
                                        const double toler_d);

/* ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    utility functions for the degrees of freedom
   ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::     */

/* linear regression model with lasso penalty           */
Rcpp::List lm_dof_LASSO_1lambda(const arma::mat& X,
                                const arma::vec& coeff,
                                const double lambda,
                                const arma::vec& Uvec,
                                const double err_primal,
                                const double err_dual,
                                const double rho,
                                const double toler_c,
                                const double toler_d);

/* linear regression model with overlap group-lasso penalty           */
Rcpp::List lm_dof_OVGLASSO_1lambda(const arma::mat& X,
                                   const arma::vec& coeff,
                                   const double lambda,
                                   const arma::mat& GRmat,
                                   const arma::vec& group_weights,
                                   const arma::vec& var_weights,
                                   const arma::vec& Uvec,
                                   const double err_primal,
                                   const double err_dual,
                                   const double rho,
                                   const double toler_c,
                                   const double toler_d);

/* linear regression model and non penalized covariates
    with overlap group-lasso penalty                          */
Rcpp::List lm_cov_dof_OVGLASSO_1lambda(const arma::mat& X,
                                       const arma::mat& Z,
                                       const arma::vec& coeff_X,
                                       const double lambda,
                                       const arma::mat& GRmat,
                                       const arma::vec& group_weights,
                                       const arma::vec& var_weights,
                                       const arma::vec& Uvec,
                                       const double err_primal,
                                       const double err_dual,
                                       const double rho,
                                       const double toler_c,
                                       const double toler_d);

/* linear regression model with adaptive overlap group-lasso penalty         */
Rcpp::List lm_adaptive_dof_OVGLASSO_1lambda(const arma::mat& X,
                                            const arma::vec& coeff,
                                            const arma::vec& coeff_LS,
                                            const double lambda,
                                            const arma::mat& GRmat,
                                            const arma::vec& group_weights,
                                            const arma::vec& var_weights,
                                            const arma::vec& Uvec,
                                            const double err_primal,
                                            const double err_dual,
                                            const double rho,
                                            const double toler_c,
                                            const double toler_d);

/* linear regression model and non penalized covariates
    with adaptive overlap group-lasso penalty                          */
Rcpp::List lm_cov_adaptive_dof_OVGLASSO_1lamba(const arma::mat& X,
                                               const arma::mat& Z,
                                               const arma::vec& coeff_X,
                                               const arma::vec& coeff_X_LS,
                                               const double lambda,
                                               const arma::mat& GRmat,
                                               const arma::vec& group_weights,
                                               const arma::vec& var_weights,
                                               const arma::vec& Uvec,
                                               const double err_primal,
                                               const double err_dual,
                                               const double rho,
                                               const double toler_c,
                                               const double toler_d);

/* function on scalar regression model with overlap group-lasso penalty:
        without additional covariates and smoothing effects           */
Rcpp::List f2s_dof_1lambda(const arma::mat& W,
                           const arma::vec& coeff,
                           const double lambda,
                           const arma::mat& GRmat,
                           const arma::vec& group_weights,
                           const arma::vec& var_weights,
                           const arma::vec& Uvec,
                           const double err_primal,
                           const double err_dual,
                           const double rho,
                           const double toler_c,
                           const double toler_d);

/* function on scalar regression model with overlap group-lasso penalty:
        without additional covariates and without smoothing effects           */
Rcpp::List f2s_cov_dof_1lambda(const arma::mat& W,
                               const arma::mat& Z,
                               const arma::vec& coeff_W,
                               const double lambda,
                               const arma::mat& GRmat,
                               const arma::vec& group_weights,
                               const arma::vec& var_weights,
                               const arma::vec& Uvec,
                               const double err_primal,
                               const double err_dual,
                               const double rho,
                               const double toler_c,
                               const double toler_d);

/* function on scalar regression model with overlap group-lasso penalty:
        without additional covariates and with smoothing effects           */
Rcpp::List f2s_smo_dof_1lambda(const arma::mat& W,
                               const arma::vec& coeff,
                               const double lambda,
                               const double lambda2,
                               const int diff_order,
                               const arma::mat& GRmat,
                               const arma::vec& group_weights,
                               const arma::vec& var_weights,
                               const arma::vec& Uvec,
                               const double err_primal,
                               const double err_dual,
                               const double rho,
                               const double toler_c,
                               const double toler_d);

/* function on scalar regression model with overlap group-lasso penalty:
        with additional covariates and with smoothing effects           */
Rcpp::List f2s_cov_smo_dof_1lambda(const arma::mat& W,
                                   const arma::mat& Z,
                                   const arma::vec& coeff_W,
                                   const double lambda,
                                   const double lambda2,
                                   const int diff_order,
                                   const arma::mat& GRmat,
                                   const arma::vec& group_weights,
                                   const arma::vec& var_weights,
                                   const arma::vec& Uvec,
                                   const double err_primal,
                                   const double err_dual,
                                   const double rho,
                                   const double toler_c,
                                   const double toler_d);

/* function on scalar regression model with overlap group-lasso penalty:
        without additional covariates and smoothing effects. Here we consider
        adaptive weights.                                                         */
Rcpp::List f2s_adaptive_dof_1lambda(const arma::mat& W,
                                    const arma::vec& coeff,
                                    const arma::vec& coeff_LS,
                                    const double lambda,
                                    const arma::mat& GRmat,
                                    const arma::vec& group_weights,
                                    const arma::vec& var_weights,
                                    const arma::vec& Uvec,
                                    const double err_primal,
                                    const double err_dual,
                                    const double rho,
                                    const double toler_c,
                                    const double toler_d);

/* function on scalar regression model with overlap group-lasso penalty:
        with additional covariates and without smoothing effects. Here we consider
        adaptive weights.                                                              */
Rcpp::List f2s_cov_adaptive_dof_1lambda(const arma::mat& W,
                                        const arma::mat& Z,
                                        const arma::vec& coeff_W,
                                        const arma::vec& coeff_W_LS,
                                        const double lambda,
                                        const arma::mat& GRmat,
                                        const arma::vec& group_weights,
                                        const arma::vec& var_weights,
                                        const arma::vec& Uvec,
                                        const double err_primal,
                                        const double err_dual,
                                        const double rho,
                                        const double toler_c,
                                        const double toler_d);

/* function on scalar regression model with overlap group-lasso penalty:
        without additional covariates and with smoothing effects. Here we consider
        adaptive weights.                                                            */
Rcpp::List f2s_smo_adaptive_dof_1lambda(const arma::mat& W,
                                        const arma::vec& coeff,
                                        const arma::vec& coeff_LS,
                                        const double lambda,
                                        const double lambda2,
                                        const int diff_order,
                                        const arma::mat& GRmat,
                                        const arma::vec& group_weights,
                                        const arma::vec& var_weights,
                                        const arma::vec& Uvec,
                                        const double err_primal,
                                        const double err_dual,
                                        const double rho,
                                        const double toler_c,
                                        const double toler_d);

/* function on scalar regression model with overlap group-lasso penalty:
        with additional covariates and with smoothing effects. Here we consider
        adaptive weights.                                                            */
Rcpp::List f2s_adaptive_cov_smo_dof_1lambda(const arma::mat& W,
                                            const arma::mat& Z,
                                            const arma::vec& coeff_W,
                                            const arma::vec& coeff_W_LS,
                                            const double lambda,
                                            const double lambda2,
                                            const int diff_order,
                                            const arma::mat& GRmat,
                                            const arma::vec& group_weights,
                                            const arma::vec& var_weights,
                                            const arma::vec& Uvec,
                                            const double err_primal,
                                            const double err_dual,
                                            const double rho,
                                            const double toler_c,
                                            const double toler_d);

/* ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    automatic criteria for model selection
   ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::     */

arma::vec f2s_bic(const arma::vec& y,
                  const arma::mat& X,
                  const arma::mat& coeff_path,
                  const arma::vec& df);

arma::vec f2s_cov_bic(const arma::vec& y,
                      const arma::mat& W,
                      const arma::mat& Z,
                      const arma::mat& W_coeff_path,
                      const arma::mat& Z_coeff_path,
                      const arma::vec& df);

arma::vec f2s_ebic(const arma::vec& y,
                   const arma::mat& X,
                   const arma::mat& coeff_path,
                   const arma::vec& df,
                   const double kappa);

arma::vec f2s_cov_ebic(const arma::vec& y,
                       const arma::mat& W,
                       const arma::mat& Z,
                       const arma::mat& W_coeff_path,
                       const arma::mat& Z_coeff_path,
                       const arma::vec& df,
                       const double kappa);
#endif
