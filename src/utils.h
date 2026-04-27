#ifndef UTILS_H
#define UTILS_H

#include <RcppArmadillo.h>

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
arma::vec rnormC(const int size,
                 const double mean,
                 const double sd);

// vectorization and matrix operations
arma::vec vecC(const arma::mat& X);
arma::mat invvecC(const arma::vec& x, 
                  const int nrow);
void squaredmat(arma::mat& XX,
                const arma::mat& X,
                const int m,
                const int n);

// Evaluate the RIDGE matrix: returns the Cholesky (Upper triangular matrix)
arma::mat ridge_chol(const arma::mat& Omega,
                     const arma::mat& Sig,
                     const double lambda);

// multiple inveresion
arma::cube multipleinversion(const arma::mat& A,
                             const double rho,
                             const arma::mat& L,
                             const arma::mat& R,
                             const arma::vec& lambda2);

// Cholesky factorization via QR decomposition
arma::mat chol_qr_fact_large_n(const arma::mat& X,
                               const int n,
                               const int p);
arma::mat chol_qr_fact_large_p(const arma::mat& X,
                               const int n,
                               const int p);
arma::mat chol_qr_fact(const arma::mat& X,
                       const int n,
                       const int p);


arma::sp_mat mat2spmat(S4 mat);
arma::sp_mat spmat2vec_mult(const arma::sp_mat& A,
                            const arma::vec& b);
arma::sp_mat spmat2spmat_mult(const arma::sp_mat& A,
                              const arma::sp_mat& B);

arma::vec forward_diff(const arma::vec& f,
                       const double h,
                       const int d);
arma::mat forward_diff_penalty_matrix(const int n,
                                      const double h,
                                      const int d);
arma::mat forward_diff_difference_matrix(const int n,
                                         const double h,
                                         const int d);

int svd_check(const arma::mat& X,
              arma::mat& U,
              arma::mat& V,
              arma::vec& s);
arma::mat pseudoinv(const arma::mat& X);
arma::vec lm_ols(const arma::mat& X,
                 const arma::vec& y);
arma::vec lm_ols_large_n(const arma::mat& X,
                         const arma::vec& y,
                         const int n,
                         const int p);
arma::vec lm_ols_large_p(const arma::mat& X,
                         const arma::vec& y,
                         const int n,
                         const int p);
arma::vec lm_ols_fast(const arma::mat& XTX_CHOL,
                      const arma::vec& XTy);
Rcpp::List lm_ols_FWL(const arma::mat& X,
                      const arma::mat& Z,
                      const arma::vec& y);
Rcpp::List lm_ols_FWL_fast(const arma::mat& L_ZZ,
                           const arma::mat& L_XMX,
                           const arma::mat& P_ZX,
                           const arma::mat& XMy,
                           const arma::vec& Zy);

double gennorm(const arma::vec& x,
               const int type);

#endif
