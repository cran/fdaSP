#ifndef ADMMUTILSOTHER_H
#define ADMMUTILSOTHER_H

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

// ============================================================
// admm_tv
arma::colvec tv_shrinkage(const arma::colvec& a,
                          const double kappa);
double tv_objective(const arma::colvec& b,
                    const double& lambda,
                    const arma::mat& D,
                    const arma::colvec& x,
                    const arma::colvec& z);

/*  Total Variation Minimization via ADMM    */
Rcpp::List admm_tv(const arma::colvec& b,
                   const arma::colvec& xinit,
                   const double lambda,
                   const double reltol, 
                   const double abstol,
                   const int maxiter,
                   const double rho, 
                   const double alpha);

// ============================================================
// admm_bp
arma::colvec bp_shrinkage(const arma::colvec& a,
                          const double kappa);

/*  Basis Pursuit via ADMM      */
Rcpp::List admm_bp(const arma::mat& A,
                   const arma::colvec& b,
                   const arma::colvec& xinit,
                   const double reltol,
                   const double abstol, 
                   const int maxiter,
                   const double rho, 
                   const double alpha);

// ===========================================================
// admm_lad
arma::colvec lad_shrinkage(const arma::colvec& a,
                           const double kappa);

/*    LAD via ADMM          */
Rcpp::List admm_lad(const arma::mat& A,
                    const arma::colvec& b,
                    const arma::colvec& xinit,
                    const double reltol,
                    const double abstol,
                    const int maxiter,
                    const double rho, 
                    const double alpha);

// ===========================================================
// admm_rpca
arma::vec shrink_vec_rpca(const arma::vec& x, 
                          const double tau);
arma::mat shrink_mat_rpca(const arma::mat& A,
                          const double tau);
arma::mat rpca_vectorpadding(const arma::vec& x,
                             const int n,
                             const int p);
Rcpp::List admm_rpca(const arma::mat& M, 
                     const double tol,
                     const int maxiter,
                     const double mu,
                     const double lambda);

// ===========================================================
// admm_spca
arma::vec spca_gamma(const arma::vec& sigma,
                     const double r);
arma::mat spca_shrinkage(const arma::mat& A,
                         const double tau);
Rcpp::List admm_spca(const arma::mat& Sigma, 
                     const double reltol,
                     const double abstol,
                     const int maxiter, 
                     const double mu,
                     const double rho);

// ===========================================================
// admm_sdp
arma::mat sdp_evdplus(const arma::mat& X);
double sdp_gap(const arma::vec& b,
               const arma::vec& y,
               const arma::mat& C,
               const arma::mat& X);
Rcpp::List admm_sdp(const arma::mat& C,
                    const arma::field<arma::mat>& listA,
                    const arma::vec& b,
                    const double mymu,
                    const double myrho,
                    const double mygamma,
                    const int maxiter,
                    const double abstol,
                    const bool printer);



#endif
