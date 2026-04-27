#ifndef LINREG_LASSO_ADMM_H
#define LINREG_LASSO_ADMM_H

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

/*
    LASSO via ADMM
                                            */

// admm_enet: Linear regression model with ENET penalty
// Linear regression model with ENET penalty: evaluate the objective function
Rcpp::List admm_enet(const arma::mat& A,
                     const arma::colvec& b,
                     const double lambda,
                     const double alpha,
                     const double reltol,
                     const double abstol,
                     const int maxiter,
                     const double rho);

// admm_genlasso: Linear regression model with generalized LASSO penalty
// Linear regression model with generalized LASSO penalty: evaluate the objective function
Rcpp::List admm_genlasso(const arma::mat& A,
                         const arma::colvec& b,
                         const arma::mat& D,
                         const double lambda,
                         const double reltol,
                         const double abstol,
                         const int maxiter,
                         const double rho);

/*
    LASSO and adaptive LASSO via ADMM: wrap functions
                                                                        */

Rcpp::List admm_lasso(const arma::mat& A,
                      const arma::colvec& b,
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
                      const int ping);

Rcpp::List admm_lasso_cov(const arma::mat& W,
                          const arma::mat& Z,
                          const arma::colvec& y,
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
                          const int ping);

Rcpp::List admm_adalasso(const arma::mat& A,
                         const arma::colvec& b,
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
                         const int ping);

Rcpp::List admm_adalasso_cov(const arma::mat& W,
                             const arma::mat& Z,
                             const arma::colvec& y,
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
                             const int ping);

/* LASSO via ADMM: fast routines                  */
Rcpp::List admm_lasso_fast(const arma::mat& A,
                           const arma::vec& b,
                           const arma::vec& lambda,
                           const bool rho_adaptation,
                           double rho,
                           const double tau,
                           const double mu,
                           const double reltol,
                           const double abstol,
                           const int maxiter,
                           const int ping);

Rcpp::List admm_lasso_cov_fast(const arma::mat& W,
                               const arma::mat& Z,
                               const arma::vec& y,
                               const arma::vec& lambda,
                               const bool rho_adaptation,
                               double rho,
                               const double tau,
                               const double mu,
                               const double reltol,
                               const double abstol,
                               const int maxiter,
                               const int ping);

Rcpp::List admm_adalasso_fast(const arma::mat& A,
                              const arma::vec& b,
                              const arma::vec& var_weights,
                              const arma::vec lambda,
                              const bool rho_adaptation,
                              double rho,
                              const double tau,
                              const double mu,
                              const double reltol,
                              const double abstol,
                              const int maxiter,
                              const int ping);

Rcpp::List admm_adalasso_cov_fast(const arma::mat& W,
                                  const arma::mat& Z,
                                  const arma::vec& y,
                                  const arma::vec& var_weights,
                                  const arma::vec& lambda,
                                  const bool rho_adaptation,
                                  double rho,
                                  const double tau,
                                  const double mu,
                                  const double reltol,
                                  const double abstol,
                                  const int maxiter,
                                  const int ping);

/*
    LASSO and adaptive LASSO via ADMM: main functions
      
                                                                        */

Rcpp::List admm_lasso_cov_large_m(const arma::mat& W,
                                  const arma::mat& Z,
                                  const arma::colvec& y,
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
                                  const int ping);

Rcpp::List admm_lasso_cov_large_n(const arma::mat& W,
                                  const arma::mat& Z,
                                  const arma::colvec& y,
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
                                  const int ping);

Rcpp::List admm_adalasso_large_m(const arma::mat& A,
                                 const arma::colvec& b,
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
                                 const int ping);

Rcpp::List admm_adalasso_large_n(const arma::mat& A,
                                 const arma::colvec& b,
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
                                 const int ping);

Rcpp::List admm_adalasso_cov_large_m(const arma::mat& W,
                                     const arma::mat& Z,
                                     const arma::vec& y,
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
                                     const int ping);

Rcpp::List admm_adalasso_cov_large_n(const arma::mat& W,
                                     const arma::mat& Z,
                                     const arma::vec& y,
                                     const arma::vec& var_weights,
                                     arma::colvec& u,
                                     arma::colvec& z,
                                     double lambda,
                                     const bool rho_adaptation,
                                     double rho,
                                     const double tau,
                                     const double mu,
                                     const double reltol,
                                     const double abstol,
                                     const int maxiter,
                                     const int ping);

/*
    LASSO and adaptive LASSO via ADMM: main functions,
      fast routines (multiple lambdas)
      
                                                              */

Rcpp::List admm_lasso_cov_large_m_fast(const arma::mat& W,
                                       const arma::mat& Z,
                                       const arma::vec& y,
                                       const arma::vec& lambda,
                                       const bool rho_adaptation,
                                       double rho,
                                       const double tau,
                                       const double mu,
                                       const double reltol,
                                       const double abstol,
                                       const int maxiter,
                                       const int ping);

Rcpp::List admm_lasso_cov_large_n_fast(const arma::mat& W,
                                       const arma::mat& Z,
                                       const arma::vec& y,
                                       const arma::vec& lambda,
                                       const bool rho_adaptation,
                                       double rho,
                                       const double tau,
                                       const double mu,
                                       const double reltol,
                                       const double abstol,
                                       const int maxiter,
                                       const int ping);

Rcpp::List admm_adalasso_large_m_fast(const arma::mat& A,
                                      const arma::vec& b,
                                      const arma::vec& var_weights,
                                      const arma::vec& lambda,
                                      const bool rho_adaptation,
                                      double rho,
                                      const double tau,
                                      const double mu,
                                      const double reltol,
                                      const double abstol,
                                      const int maxiter,
                                      const int ping);

Rcpp::List admm_adalasso_large_n_fast(const arma::mat& A,
                                      const arma::vec& b,
                                      const arma::vec& var_weights,
                                      const arma::vec& lambda,
                                      const bool rho_adaptation,
                                      double rho,
                                      const double tau,
                                      const double mu,
                                      const double reltol,
                                      const double abstol,
                                      const int maxiter,
                                      const int ping);

Rcpp::List admm_adalasso_cov_large_m_fast(const arma::mat& W,
                                          const arma::mat& Z,
                                          const arma::vec& y,
                                          const arma::vec& var_weights,
                                          const arma::vec& lambda,
                                          const bool rho_adaptation,
                                          double rho,
                                          const double tau,
                                          const double mu,
                                          const double reltol,
                                          const double abstol,
                                          const int maxiter,
                                          const int ping);

Rcpp::List admm_adalasso_cov_large_n_fast(const arma::mat& W,
                                          const arma::mat& Z,
                                          const arma::vec& y,
                                          const arma::vec& var_weights,
                                          const arma::vec& lambda,
                                          const bool rho_adaptation,
                                          double rho,
                                          const double tau,
                                          const double mu,
                                          const double reltol,
                                          const double abstol,
                                          const int maxiter,
                                          const int ping);

/*
    LASSO and adaptive LASSO via ADMM: main functions,
      routines for single lambda
      
                                                              */

Rcpp::List admm_lasso_update_1lambda(const arma::mat& A,
                                     const arma::vec& b,
                                     const arma::mat& ATA_CHOL_U,
                                     const arma::mat& AA,
                                     const arma::vec& ATb,
                                     const double b2,
                                     const int m,
                                     const int n,
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
                                     const int ping);

Rcpp::List admm_adalasso_large_m_update_1lambda(const arma::mat& A,
                                                const arma::vec& b,
                                                const arma::mat& ATA,
                                                const arma::mat& ATA_CHOL_U,
                                                const arma::vec& ATb,
                                                const double b2,
                                                const arma::vec& var_weights,
                                                const int m,
                                                const int n,
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
                                                const int ping);

Rcpp::List admm_adalasso_large_n_update_1lambda(const arma::mat& A,
                                                const arma::vec& b,
                                                const arma::mat& AFTF_INVA,
                                                const arma::mat& ATA_CHOL_U,
                                                const arma::vec& ATb,
                                                const double b2,
                                                const arma::vec& var_weights,
                                                const int m,
                                                const int n,
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
                                                const int ping);

Rcpp::List admm_lasso_cov_large_m_update_1lambda(const arma::mat& W,
                                                 const arma::mat& Z,
                                                 const arma::vec& y,
                                                 const arma::mat& WMW,
                                                 const arma::vec& WMy,
                                                 const arma::vec& v_LS,
                                                 const arma::mat& P_ZW,
                                                 const arma::mat& WW_CHOL_U,
                                                 const arma::vec& Wy,
                                                 const double y2,
                                                 const arma::mat& ZZ_CHOL_U,
                                                 const arma::vec& Zy,
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
                                                 const int ping);

Rcpp::List admm_lasso_cov_large_n_update_1lambda(const arma::mat& W,
                                                 const arma::mat& Z,
                                                 const arma::vec& y,
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
                                                 bool rho_adaptation,
                                                 double rho,
                                                 const double tau,
                                                 const double mu,
                                                 const double reltol,
                                                 const double abstol,
                                                 const int maxiter,
                                                 const int ping);

Rcpp::List admm_adalasso_cov_large_m_update_1lambda(const arma::mat& W,
                                                    const arma::mat& Z,
                                                    const arma::vec& y,
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
                                                    const int ping);

Rcpp::List admm_adalasso_cov_large_n_update_1lambda(const arma::mat& W,
                                                    const arma::mat& Z,
                                                    const arma::vec& y,
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
                                                    const int ping);


#endif
