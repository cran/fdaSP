#ifndef MMUTILS_H
#define MMUTILS_H

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

int mod(int a, int n);

arma::vec fast_XTX_plus_D_update(arma::mat mXX, arma::vec vXy, arma::vec omega);

arma::mat fast_XTX_plus_D_inversion(arma::mat mXX, arma::vec omega);

arma::vec LASSO_MM_Woodbury_update(arma::mat mX, arma::vec vXy, arma::vec omega, double lambda, unsigned int n, unsigned int p);

List lmridge(arma::mat mX, arma::vec vY, arma::vec lambda, unsigned int n, unsigned int p);




#endif
