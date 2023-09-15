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
arma::vec rnormC(int size, double mean, double sd);

// vectorization and matrix operations
arma::vec vecC(arma::mat X);
arma::mat invvecC(arma::vec x, const int nrow);
void squaredmat(arma::mat& XX, arma::mat X, const int m, const int n);

// Evaluate the RIDGE matrix: returns the Cholesky (Upper triangular matrix)
arma::mat ridge_chol(arma::mat Omega, arma::mat Sig, double lambda);

// multiple inveresion
arma::cube multipleinversion(arma::mat A, double rho, arma::mat L, arma::mat R, arma::vec lambda2);

// Cholesky factorization via QR decomposition
arma::mat chol_qr_fact_large_n(arma::mat X, const int n, const int p);
arma::mat chol_qr_fact_large_p(arma::mat X, const int n, const int p);
arma::mat chol_qr_fact(arma::mat X, const int n, const int p);


arma::sp_mat mat2spmat(S4 mat);
arma::sp_mat spmat2vec_mult(const arma::sp_mat& A, const arma::vec& b);
arma::sp_mat spmat2spmat_mult(const arma::sp_mat& A, const arma::sp_mat& B);





#endif
