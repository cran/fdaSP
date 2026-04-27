// Utility routines

// Authors:
//           Bernardi Mauro, University of Padova
//           Last update: July 12, 2022

// List of implemented methods:

// [[Rcpp::depends(RcppArmadillo)]]
#include "utils.h"

/*
 * gaussian random simulation
*/
//' @keywords internal
//' @noRd
arma::vec rnormC(const int size,
                 const double mean,
                 const double sd) {
  arma::vec x = arma::randn(size) * sd + mean;
  return x;
}

/*
 * vectorize a matrix
*/
//' @keywords internal
//' @noRd
arma::vec vecC(const arma::mat& X) {
  
  /* Variable delcaration           */
  const int n = X.n_rows;
  const int p = X.n_cols;

  /* Definition of vectors and matrices     */
  arma::colvec x(n * p, fill::zeros);
  
  /* Vectorize     */
  x = vectorise(X, 0);
  
  /* Return output     */
  return(x);
}

/*
 * inverse of the vectorization operation
*/
//' @keywords internal
//' @noRd
arma::mat invvecC(const arma::vec& x, 
                  const int nrow) {
  
  /* Variable delcaration           */
  const int n = x.n_elem;
  const int p = n / nrow;

  /* Definition of vectors and matrices     */
  arma::mat X(nrow, p, fill::zeros);
  
  /* Vectorize     */
  for (int j=0; j<p; j++) {
    X.col(j) = x.subvec(j * nrow, (j+1)*nrow-1);
  }
  
  /* Return output     */
  return(X);
}

/*
 * square of a matrix
*/
//' @keywords internal
//' @noRd
void squaredmat(arma::mat& XX,
                const arma::mat& X,
                const int m,
                const int n) {
  if (m >= n) {
    XX = X.t() * X;
  } else {
    XX = X * X.t();
  }
}

/*
 * Cholesky factor of the ridge
*/
//' @keywords internal
//' @noRd
arma::mat ridge_chol(const arma::mat& Omega,
                     const arma::mat& Sig,
                     const double lambda) {

  /* Variable delcaration           */
  const int n = Omega.n_cols;
  arma::mat U(n, n, fill::zeros);
  
  /* Compute the Cholesky factor of the RIDGE matrix        */
  U = chol(Omega + lambda * Sig);
  
  /* return output         */
  return(U);
}

/*
 * multiple inversion
*/
//' @keywords internal
//' @noRd
arma::cube multipleinversion(const arma::mat& A,
                             const double rho,
                             const arma::mat& L,
                             const arma::mat& R,
                             const arma::vec& lambda2) {
  // 1. get size information
  const int p = A.n_cols;
  const int d2 = L.n_rows;
  const int nlambda = lambda2.n_elem;

  // 2. initialize and preliminary computations
  arma::cube output(p,p,nlambda,fill::zeros);

  arma::mat AA = A.t()*A;
  arma::mat LL = L.t()*L;
  arma::mat RR = R.t()*R; // equivalent to M = R^T R;

  // 3. computation : the very first one
  output.slice(0) = arma::pinv(AA+(rho*LL)+(2*lambda2(0)*RR));

  // 4. computation : iterative update
  double dlbd = 0.0;                     // difference in lambda2 values
  arma::mat Ainv(p,p,fill::zeros);       // inverse of A for previous step
  arma::mat solveLHS(d2,d2,fill::zeros); // iterative comp::inverse term
  arma::mat solveTMP(d2,p,fill::zeros);  // iterative comp::solveLHS*R
  arma::mat eye_d2(d2,d2,fill::eye);     // diagonal matrix of size (d2-by-d2)
  arma::mat eye_p(p,p,fill::eye);        // diagnoal matrix of size (p -by- p)

  arma::mat invold = output.slice(0);
  arma::mat invnew(p,p,fill::zeros);

  for (int i=1;i<nlambda;i++) {
    // 4-1. update setup
    dlbd = lambda2(i)-lambda2(i-1);

    // 4-2. preliminary computation
    solveLHS = (eye_d2 + (2*dlbd*(R*invold*R.t())));
    solveTMP = arma::solve(solveLHS, R);

    // 4-3. compute and record
    invnew          = (eye_p - 2*dlbd*(invold*R.t()*solveTMP))*invold;
    output.slice(i) = invnew;

    // 4-4. update
    invold = invnew;
  }

  // 5. return output
  return(output);
}

/*
 * QR factorization for large n
*/
//' @keywords internal
//' @noRd
arma::mat chol_qr_fact_large_n(const arma::mat& X,
                               const int n,
                               const int p) {
      
  /* ::::::::::::::::::::::::::::::::::::::::::::::
   vector and matrices declaration                       */
  arma::mat Qmat(n, p, fill::zeros);
  arma::mat Rmat(p, p, fill::zeros);
  
  /* :::::::::::::::::::::::::::::::::::::::::
   Perform QR update                     */
  qr_econ(Qmat, Rmat, X);
  
  /* return output          */
  return(Rmat);
}

/*
 * QR factorization for large p
*/
//' @keywords internal
//' @noRd
arma::mat chol_qr_fact_large_p(const arma::mat& X,
                               const int n,
                               const int p) {
  
  /* ::::::::::::::::::::::::::::::::::::::::::::::
   vector and matrices declaration                       */
  arma::mat Qmat(n, n, fill::zeros);
  arma::mat Rmat(n, p, fill::zeros);
  
  /* :::::::::::::::::::::::::::::::::::::::::
   Perform QR update                      */
  qr_econ(Qmat, Rmat, X);
  
  /* return output          */
  return(Rmat);
}

/*
 * QR factorization
*/
//' @keywords internal
//' @noRd
arma::mat chol_qr_fact(const arma::mat& X,
                       const int n,
                       const int p) {
    
  /* ::::::::::::::::::::::::::::::::::::::::::::::
  variable declaration                                  */
  int nr = 0, nc = 0;
    
  /* ::::::::::::::::::::::::::::::::::::::::::::::
   define the dimension of the R matrix                 */
  if (n >= p) {
    nr = p;
    nc = p;
  } else {
    nr = n;
    nc = p;
  }
  
  /* ::::::::::::::::::::::::::::::::::::::::::::::
   vector and matrices declaration                       */
  arma::mat Rmat(nr, nc, fill::zeros);
  
  /* :::::::::::::::::::::::::::::::::::::::::
   Perform QR update of XTX + D                      */
  if (n >= p) {
    Rmat = chol_qr_fact_large_n(X, n, p);
  } else {
    Rmat = chol_qr_fact_large_p(X, n, p);
  }
  
  /* return output          */
  return(Rmat);
}

/*
 * matrix to sparse matrix conversion
*/
//' @keywords internal
//' @noRd
arma::sp_mat mat2spmat(S4 mat) {
  
  /* variable declaration             */
  int nrow = 0, ncol = 0;

  // obtain dim, i, p. x from S4 object
  IntegerVector dims = mat.slot("Dim");
  nrow               = dims[0];
  ncol               = dims[1];
  
  // define the positions and elements of the sparse matrix
  arma::urowvec i = Rcpp::as<arma::urowvec>(mat.slot("i"));
  arma::urowvec p = Rcpp::as<arma::urowvec>(mat.slot("p"));
  arma::vec x     = Rcpp::as<arma::vec>(mat.slot("x"));
  
  // use Armadillo sparse matrix constructor
  arma::sp_mat res(i, p, x, nrow, ncol);
  //Rcout << "SpMat res:\n" << res << std::endl;
  
  /* return output          */
  return(res);
}

/*
 * sparse matrix to vector multiplication
*/
//' @keywords internal
//' @noRd
arma::sp_mat spmat2vec_mult(const arma::sp_mat& A,
                            const arma::vec& b) {
  arma::sp_mat result(A * b);
  return(result);
}

/*
 * sparse matrix to sparse matrix multiplication
*/
//' @keywords internal
//' @noRd
arma::sp_mat spmat2spmat_mult(const arma::sp_mat& A,
                              const arma::sp_mat& B) {
  arma::sp_mat result(A * B);
  return(result);
}

//' Forward finite difference approximation of order d
//'
//' @param f Numeric vector containing function values.
//' @param h Step size (default = 1.0).
//' @param d Order of the derivative (integer >= 1).
//' @return Numeric vector of forward differences of order d (length n, last d elements = NA).
//' #@examples
//' #f <- c(0, 1, 4, 9, 16)
//' #forward_diff(f = f, h = 1, d = 2)
// [[Rcpp::export]]
arma::vec forward_diff(const arma::vec& f, 
                       const double h,
                       const int d) {
  
  /* variable declaration             */
  int n = f.n_elem;
  double sum = 0.0, c = 0.0;

  /* checks          */
  if (n <= d) {
    stop("Vector length must be greater than derivative order d.");
  }
  if (d < 1) {
    stop("Derivative order d must be >= 1.");
  }
  
  /* vector and matrices declaration             */
  arma::vec df(n - d, fill::zeros);
  arma::vec coeff(d + 1, fill::ones);
    
  // Precompute binomial coefficients
  for (int k = 0; k <= d; ++k) {
    // compute binomial coefficient C(d, k)
    c = 1.0;
    for (int j = 1; j <= k; ++j) {
      c *= (double)(d - j + 1) / j;
    }
    coeff(k) = std::pow(-1.0, k) * c;
  }

  // Compute finite differences
  for (int i = 0; i < n - d; ++i) {
    sum = 0.0;
    for (int k = 0; k <= d; ++k) {
      sum += coeff(k) * f(i + d - k);
    }
    df(i) = sum / std::pow(h, d);
  }

  /* return output          */
  return df;
}

//' Compute the discrete difference penalty matrix Q = D^T D
//'
//' Constructs the d-th order forward-difference operator D (scaled by 1/h^d)
//' and returns Q = D^T D.
//'
//' @param n Length of the vector the operator acts on.
//' @param h Step size (default = 1.0).
//' @param d Derivative order (integer >= 1).
//' @return Symmetric n x n matrix Q = D^T D.
//' @examples
//' #Q1 <- forward_diff_penalty_matrix(5, h = 1, d = 1)
//' #Q2 <- forward_diff_penalty_matrix(5, h = 1, d = 2)
//' #Q1
//' @name  fdaSP.internal
//' @keywords internal
// [[Rcpp::export(.forward_diff_penalty_matrix)]]
arma::mat forward_diff_penalty_matrix(const int n,
                                      const double h,
                                      const int d) {
  
  /* variable declaration             */
  int rows = n - d;
  double c = 0.0;
  
  /* vector and matrices declaration             */
  arma::mat D(rows, n, fill::zeros);
  arma::mat Q(n, n, fill::zeros);
  arma::vec coeff(d + 1, fill::ones);
  
  /* checks          */
  if (n <= d) {
    stop("Vector length must be greater than derivative order d.");
  }
  if (d < 1) {
    stop("Derivative order d must be >= 1.");
  }
  
  // compute binomial coefficients (with alternating sign)
  for (int k = 0; k <= d; ++k) {
    c = 1.0;
    for (int j = 1; j <= k; ++j) {
      c *= (double)(d - j + 1) / j;
    }
    coeff(k) = std::pow(-1.0, d - k) * c / std::pow(h, d);
  }

  // build sparse forward-difference matrix D
  for (int i = 0; i < rows; ++i) {
    for (int k = 0; k <= d; ++k) {
      D(i, i + k) = coeff(k);
    }
  }
  // compute Q = D^T * D
  Q = D.t() * D;

  /* return output          */
  return Q;
}

//' @name  fdaSP.internal
//' @keywords internal
// [[Rcpp::export(.forward_diff_difference_matrix)]]
arma::mat forward_diff_difference_matrix(const int n,
                                         const double h,
                                         const int d) {
  
  /* variable declaration             */
  int rows = n - d;
  double c = 0.0;
  
  /* vector and matrices declaration             */
  arma::mat D(rows, n, fill::zeros);
  arma::vec coeff(d + 1, fill::ones);
  
  /* checks          */
  if (n <= d) {
    stop("Vector length must be greater than derivative order d.");
  }
  if (d < 1) {
    stop("Derivative order d must be >= 1.");
  }
  
  // compute binomial coefficients (with alternating sign)
  for (int k = 0; k <= d; ++k) {
    c = 1.0;
    for (int j = 1; j <= k; ++j) {
      c *= (double)(d - j + 1) / j;
    }
    coeff(k) = std::pow(-1.0, d - k) * c / std::pow(h, d);
  }

  // build sparse forward-difference matrix D
  for (int i = 0; i < rows; ++i) {
    for (int k = 0; k <= d; ++k) {
      D(i, i + k) = coeff(k);
    }
  }

  /* return output          */
  return D;
}

// [[Rcpp::export]]
int svd_check(const arma::mat& X,
              arma::mat& U,
              arma::mat& V,
              arma::vec& s) {
  
  /* Variable definition            */
  int n = X.n_rows, p = X.n_cols, check = 1;
  double tol = 0.0, s_min = 0.0, s_max = 0.0;
    
  // perform SVD decomposition: X = U * diagmat(s) * V.t()
  arma::svd_econ(U, s, V, X, "both");
  
  // Condition number
  s_max = s.max();
  s_min = s.min();
  
  // define tolerance
  tol = std::numeric_limits<double>::epsilon() * std::max(n, p) * s_max;

  // decision rule for inversion method
  if (s_min < tol) {
    check = 0;
  } else {
    check = 1;
  }
    
  /* return output    */
  return check;
}

// [[Rcpp::export]]
arma::mat pseudoinv(const arma::mat& X) {
  
  /* Variable definition            */
  int n = X.n_rows, p = X.n_cols;
  double tol = 0.0, s_min = 0.0, s_max = 0.0, cond_number = 0.0;
  
  /* Definition of vectors and matrices     */
  arma::mat U, V;
  arma::vec s;
  
  // perform SVD decomposition: X = U * diagmat(s) * V.t()
  arma::svd_econ(U, s, V, X, "both");
  
  // Condition number
  s_max       = s.max();
  s_min       = s.min();
  cond_number = s_max / s_min;
  
  // define tolerance
  tol = std::numeric_limits<double>::epsilon() * std::max(n, p) * s_max;

  // decision rule for inversion method
  arma::vec inv_s = arma::zeros(s.n_elem);
  if (cond_number < 1e8) {
    // Matrix is reasonably well-conditioned:
    // invert all singular values normally
    inv_s = 1.0 / s;
  } else {
    // Matrix is ill-conditioned or rank-deficient:
    // zero-out small singular values
    for (size_t i = 0; i < s.n_elem; ++i) {
      if (s(i) > tol) {
        inv_s(i) = 1.0 / s(i);
      } else {
        inv_s(i) = 0.0;
      }
    }
  }

  // Pseudo-inverse = V * diag(inv_s) * U.t()
  arma::mat pinvX = V * arma::diagmat(inv_s) * U.t();
  
  /* return output    */
  return pinvX;
}

// [[Rcpp::export]]
arma::vec lm_ols(const arma::mat& X,
                 const arma::vec& y) {

  /* Variable definition            */
  const int p = X.n_cols, n = X.n_rows;
  
  /* Definition of vectors and matrices     */
  arma::vec coeff(p, fill::zeros);

  /* Definition of vectors and matrices     */
  if (n > p) {
    coeff = lm_ols_large_n(X, y, n, p);
  } else {
    coeff = lm_ols_large_p(X, y, n, p);
  }
  
  /* return output    */
  return(coeff);
}

arma::vec lm_ols_large_n(const arma::mat& X,
                         const arma::vec& y,
                         const int n,
                         const int p) {

  /* Variable definition            */
  double sqrtn = 0.0;

  /* Definition of vectors and matrices     */
  arma::mat X_(n, p, fill::zeros);
  arma::mat XTX(p, p, fill::zeros);
  arma::mat L_XTX(p, p, fill::zeros);
  arma::vec XTy(p, fill::zeros);
  arma::vec coeff(p, fill::zeros);
  arma::vec ones_n(n, fill::ones);
  
  /* Variable definition            */
  sqrtn = std::sqrt(static_cast<float>(n));
  
  /* Definition of vectors and matrices     */
  X_    = X.each_col() % (ones_n * (1.0 / sqrtn));
  XTX   = X_.t() * X_;
  XTy   = X_.t() * y / static_cast<float>(sqrtn);
  L_XTX = (chol(XTX)).t();
  
  /* OLS solution    */
  coeff = solve(trimatu(L_XTX.t()), solve(trimatl(L_XTX), XTy));
  
  /* return output    */
  return(coeff);
}

arma::vec lm_ols_large_p(const arma::mat& X,
                         const arma::vec& y,
                         const int n,
                         const int p) {

  /* Definition of vectors and matrices     */
  arma::vec ones_n(n, fill::ones);
  arma::vec y_(n, fill::zeros);
  arma::mat X_(n, p, fill::zeros);
  arma::vec coeff(p, fill::zeros);

  /* Variable definition            */
  const double sqrtn = std::sqrt(static_cast<float>(n));
  
  /* Definition of vectors and matrices     */
  X_ = X.each_col() % (ones_n * (1.0 / sqrtn));
  y_ = y / static_cast<float>(sqrtn);
  
  /* OLS solution    */
  coeff = pseudoinv(X_) * y_;
  
  /* return output    */
  return(coeff);
}

// [[Rcpp::export]]
arma::vec lm_ols_fast(const arma::mat& XTX_CHOL,
                      const arma::vec& XTy) {

  /* Variable definition            */
  const int p = XTX_CHOL.n_cols;        

  /* Definition of vectors and matrices     */
  arma::vec coeff(p, fill::zeros);
  arma::vec res(p, fill::zeros);
  
  /* OLS solution    */
  arma::solve(res, trimatl(XTX_CHOL.t()), XTy, arma::solve_opts::fast);
  arma::solve(coeff, trimatu(XTX_CHOL), res, arma::solve_opts::fast);
  
  /* return output    */
  return(coeff);
}

// [[Rcpp::export]]
Rcpp::List lm_ols_FWL(const arma::mat& X,
                      const arma::mat& Z,
                      const arma::vec& y) {
  
  /* Variable delcaration           */
  double sqrtn = 0.0;
  int n = 0, p = 0, q = 0;
  
  /* Get dimensions             */
  n = X.n_rows;
  p = X.n_cols;
  q = Z.n_cols;
  
  /* Variable definition            */
  sqrtn = std::sqrt(static_cast<float>(n));
  
  /* Get output lists          */
  List output;
    
  /* Definition of vectors and matrices     */
  arma::mat X_(n, p, fill::zeros);
  arma::mat Z_(n, q, fill::zeros);
  arma::colvec coeff_X(p, fill::zeros);
  arma::colvec coeff_Z(q, fill::zeros);
  arma::colvec coeff_Z_LS(q, fill::zeros);
  arma::mat ZZ(q, q, fill::zeros);
  arma::mat L_ZZ(q, q, fill::zeros);
  arma::mat A(q, q, fill::zeros);
  arma::mat P_ZZ(n, n, fill::zeros);
  arma::mat eye_n(n, n, fill::eye);
  arma::mat M_ZZ(n, n, fill::zeros);
  arma::vec Zy(q, fill::zeros);
  arma::mat XMX(p, p, fill::zeros);
  arma::vec XMy(p, fill::zeros);
  arma::mat L_XMX(p, p, fill::zeros);
  arma::mat ZX(q, p, fill::zeros);
  arma::mat P_ZX(q, p, fill::zeros);
  arma::vec ones_n(n, fill::ones);
  
  /* Precompute relevant quantities         */
  Z_    = Z.each_col() % (ones_n * (1.0 / sqrtn));
  X_    = X.each_col() % (ones_n * (1.0 / sqrtn));
  ZZ    = Z_.t() * Z_;
  L_ZZ  = (chol(ZZ)).t();
  A     = solve(trimatl(L_ZZ), Z_.t());
  P_ZZ  = A.t() * A;
  M_ZZ  = eye_n - P_ZZ;
  Zy    = Z_.t() * y / static_cast<float>(sqrtn);
  XMX   = X_.t() * M_ZZ * X_;
  XMy   = X_.t() * M_ZZ * y / static_cast<float>(sqrtn);
  ZX    = Z_.t() * X_;
  P_ZX  = solve(trimatu(L_ZZ.t()), solve(trimatl(L_ZZ), ZX));
  L_XMX = (chol(XMX)).t();
  
  /* Compute LS estimates using FWL theorem         */
  coeff_Z_LS = solve(trimatu(L_ZZ.t()), solve(trimatl(L_ZZ), Zy));
  coeff_X    = solve(trimatu(L_XMX.t()), solve(trimatl(L_XMX), XMy));
  coeff_Z    = coeff_Z_LS - P_ZX * coeff_X;

  /* Retrieve output      */
  output["coeff_X"] = coeff_X;
  output["coeff_Z"] = coeff_Z;

  /* Return output      */
  return(output);
}

// [[Rcpp::export]]
Rcpp::List lm_ols_FWL_fast(const arma::mat& L_ZZ,
                           const arma::mat& L_XMX,
                           const arma::mat& P_ZX,
                           const arma::mat& XMy,
                           const arma::vec& Zy) {
  
  /* Variable delcaration           */
  int p = 0, q = 0;
  
  /* Get dimensions             */
  p = L_XMX.n_cols;
  q = L_ZZ.n_cols;
    
  /* Get output lists          */
  List output;
    
  /* Definition of vectors and matrices     */
  arma::colvec coeff_X(p, fill::zeros);
  arma::colvec coeff_Z(q, fill::zeros);
  arma::colvec coeff_Z_LS(q, fill::zeros);
  
  /* Compute LS estimates using FWL theorem         */
  coeff_Z_LS = solve(trimatu(L_ZZ.t()), solve(trimatl(L_ZZ), Zy));
  coeff_X    = solve(trimatu(L_XMX.t()), solve(trimatl(L_XMX), XMy));
  coeff_Z    = coeff_Z_LS - P_ZX * coeff_X;
  
  /* Retrieve output      */
  output["coeff_X"] = coeff_X;
  output["coeff_Z"] = coeff_Z;

  /* Return output      */
  return(output);
}

// [[Rcpp::export]]
double gennorm(const arma::vec& x,
               const int type) {
  
  double norm_ = 0.0;
  norm_ = norm(x, type);

  /* Return output      */
  return(norm_);
}


  

