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
arma::vec rnormC(int size, double mean, double sd) {
  arma::vec x = arma::randn(size) * sd + mean;
  return x;
}

/*
 * vectorize a matrix
*/
//' @keywords internal
//' @noRd
arma::vec vecC(arma::mat X) {
  
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
arma::mat invvecC(arma::vec x, const int nrow) {
  
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
void squaredmat(arma::mat& XX, arma::mat X, const int m, const int n) {
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
arma::mat ridge_chol(arma::mat Omega, arma::mat Sig, double lambda) {

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
arma::cube multipleinversion(arma::mat A, double rho, arma::mat L, arma::mat R, arma::vec lambda2) {
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
arma::mat chol_qr_fact_large_n(arma::mat X, const int n, const int p) {
      
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
arma::mat chol_qr_fact_large_p(arma::mat X, const int n, const int p) {
  
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
arma::mat chol_qr_fact(arma::mat X, const int n, const int p) {
    
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
arma::sp_mat spmat2vec_mult(const arma::sp_mat& A, const arma::vec& b) {
  arma::sp_mat result(A * b);
  return(result);
}

/*
 * sparse matrix to sparse matrix multiplication
*/
//' @keywords internal
//' @noRd
arma::sp_mat spmat2spmat_mult(const arma::sp_mat& A, const arma::sp_mat& B) {
  arma::sp_mat result(A * B);
  return(result);
}


