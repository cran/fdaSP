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

// gaussian random simulation
arma::vec rnormC(int size, double mean, double sd);
arma::vec vecC(arma::mat X);
arma::mat invvecC(arma::vec x, const int nrow);

// admm_tv ============================================================
arma::colvec tv_shrinkage(arma::colvec a, const double kappa);

double tv_objective(arma::colvec b, const double lambda, arma::mat D,
                    arma::colvec x, arma::colvec z);
/*
* Total Variation Minimization via ADMM (from Stanford)
* http://stanford.edu/~boyd/papers/pdf/admm_distr_stats.pdf
* page 19 : section 3.3.1 : stopping criteria part (3.12).
*/
Rcpp::List admm_tv(const arma::colvec& b, arma::colvec& xinit, const double lambda,
                   const double reltol, const double abstol, const int maxiter,
                   const double rho, const double alpha);


// admm_bp ============================================================
arma::colvec bp_shrinkage(arma::colvec a, const double kappa);

/*
* Basis Pursuit via ADMM (from Stanford)
* URL : https://web.stanford.edu/~boyd/papers/admm/basis_pursuit/basis_pursuit.html
* http://stanford.edu/~boyd/papers/pdf/admm_distr_stats.pdf
* page 19 : section 3.3.1 : stopping criteria part (3.12).
*/
Rcpp::List admm_bp(const arma::mat& A, const arma::colvec& b, arma::colvec& xinit,
                   const double reltol, const double abstol, const int maxiter,
                   const double rho, const double alpha);

// admm_lad ===========================================================
arma::colvec lad_shrinkage(arma::colvec a, const double kappa);

/*
* LAD via ADMM (from Stanford)
* http://stanford.edu/~boyd/papers/pdf/admm_distr_stats.pdf
* page 19 : section 3.3.1 : stopping criteria part (3.12).
*/
Rcpp::List admm_lad(const arma::mat& A, const arma::colvec& b, arma::colvec& xinit,
                    const double reltol, const double abstol, const int maxiter,
                    const double rho, const double alpha);

// admm_rpca ==========================================================
arma::vec shrink_vec_rpca(arma::vec x, double tau);

arma::mat shrink_mat_rpca(arma::mat A, const double tau);

arma::mat rpca_vectorpadding(arma::vec x, const int n, const int p);

Rcpp::List admm_rpca(const arma::mat& M, const double tol, const int maxiter,
                     double mu, double lambda);

// admm_spca ==========================================================
arma::vec spca_gamma(arma::vec sigma, double r);

arma::mat spca_shrinkage(arma::mat A, const double tau);

Rcpp::List admm_spca(const arma::mat& Sigma, const double reltol, const double abstol,
                     const int maxiter, double mu, double rho);

// admm_sdp ===========================================================
arma::mat sdp_evdplus(arma::mat& X);

double sdp_gap(arma::vec& b, arma::vec& y, arma::mat& C, arma::mat& X);

Rcpp::List admm_sdp(arma::mat& C, arma::field<arma::mat>& listA, arma::vec b,
                    double mymu, double myrho, double mygamma, int maxiter, double abstol, bool printer);










#endif
