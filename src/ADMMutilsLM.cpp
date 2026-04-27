// Utility routines

// Authors:
//           Bernardi Mauro, University of Padova
//           Last update: July 12, 2022

// List of implemented methods:

// [[Rcpp::depends(RcppArmadillo)]]
#include "ADMMutilsLM.h"

// ==========================================================
// admm_enet: linear regression model with ENET penalty
arma::colvec enet_prox(const arma::colvec& a, 
                       const double kappa) {
  const int n = a.n_elem;
  arma::colvec y(n, fill::zeros);
  for (int i=0; i<n; i++) {
    // first term : max(0, a-kappa)
    if (a(i) - kappa > 0) {
      y(i) = a(i) - kappa;
    }
    // second term : -max(0, -a-kappa)
    if (-a(i) - kappa > 0) {
      y(i) = a(i) + kappa;
    }
  }
  return(y);
}
double enet_objfun(const arma::mat& A,
                   const arma::colvec& b,
                   const double lambda,
                   const double alpha, 
                   const arma::colvec& x,
                   const arma::colvec& z) {
  
  return(0.5 * norm(A * x - b, 2) + lambda * alpha * norm(z, 1) + 0.5 * (1.0 - alpha) * lambda * norm(x, 2));
}
arma::mat enet_factor(const arma::mat& A,
                      const double rho) {
  
  const int n = A.n_cols;
  arma::vec onesN(n, fill::ones);
  arma::mat U = arma::chol(A.t() * A + rho * diagmat(onesN));
  return(U);
}

// ======================================================
// admm_genlasso: linear regression model with generalized LASSO penalty
arma::colvec genlasso_shrinkage(const arma::colvec& a,
                                const double kappa) {
  
  const int n = a.n_elem;
  arma::colvec y(n,fill::zeros);
  for (int i=0;i<n;i++) {
    // first term : max(0, a-kappa)
    if (a(i)-kappa > 0){
      y(i) = a(i)-kappa;
    }
    // second term : -max(0, -a-kappa)
    if (-a(i)-kappa > 0){
      y(i) = y(i) + a(i) + kappa;
    }
  }
  return(y);
}
double genlasso_objective(const arma::mat& A,
                          const arma::colvec& b,
                          const arma::mat& D,
                          const double lambda,
                          const arma::colvec& x,
                          const arma::colvec& z) {
  
  return(pow(norm(A*x-b, 2), 2) / 2 + lambda*norm(D * x, 1));
}
arma::mat genlasso_factor(const arma::mat& A,
                          const double rho,
                          const arma::mat& D) {
  
  arma::mat U = chol(A.t()*A + rho*D.t()*D);
  
  return(U);
}

// =========================================================
// admm_lasso: linear regression model with generalized LASSO penalty
// elementwise soft thresholding operator
arma::colvec lasso_prox(const arma::colvec& a,
                        const double kappa) {
  
  const int n = a.n_elem;
  arma::colvec y(n, fill::zeros);
  
  for (int i=0; i<n; i++) {
    // first term : max(0, a-kappa)
    if (a(i)-kappa > 0.0) {
      y(i) = a(i)-kappa;
    }
    // second term : -max(0, -a-kappa)
    if (-a(i)-kappa > 0.0) {
      y(i) = a(i) + kappa;
    }
  }
  return(y);
}
// Evaluate the objective function
double lasso_objfun(const arma::mat& A,
                    const arma::colvec& b,
                    const double lambda,
                    const arma::colvec& x,
                    double& rss,
                    double& pen) {
  
  const int m = A.n_rows;
  double objfun = 0.0;
  arma::vec resid(m, fill::zeros);
  
  /* compute the objective function   */
  resid  = A * x - b;
  rss    = as_scalar(resid.t() * resid);
  pen    = norm(x, 1);
  objfun = 0.5 * rss / static_cast<float>(m) + lambda * pen;
  
  /* Return output      */
  return(objfun);
}
// Evaluate the RIDGE matrix: returns the Cholesky (Upper triangular matrix)
void lasso_factor(arma::mat& U,
                  const arma::mat& A,
                  const double rho) {
  
  /* Variable delcaration           */
  const int m = A.n_rows;
  const int n = A.n_cols;
  int m1 = 0;
  if (m >= n) {
    m1 = n;
  } else {
    m1 = m;
  }

  /* Definition of vectors and matrices     */
  arma::mat AA(m1, m1, fill::zeros);
  arma::vec eyeM(m1, fill::ones);
  
  /* Compute the Cholesky factor of the RIDGE matrix        */
  if (m >= n) {
    /* Precompute relevant quantities         */
    AA = A.t() * A / static_cast<float>(m);
    
    /* compute the Cholesky       */
    U = chol(AA + rho * diagmat(eyeM));
  } else {
    /* Precompute relevant quantities         */
    AA = A * A.t() / static_cast<float>(m);
    
    //U = chol(diagmat(eyeM * static_cast<float>(m)) + (1.0 / rho) * (A * A.t()));
    U = chol(diagmat(eyeM * rho * static_cast<float>(m)) + AA);
  }
}
// Evaluate the RIDGE matrix: returns the Cholesky (Upper triangular matrix)
void lasso_factor_fast(arma::mat& U,
                       const arma::mat& AA,
                       const double rho,
                       const int m,
                       const int n) {

  /* Compute the Cholesky factor of the RIDGE matrix        */
  if (m >= n) {
    arma::vec eyeN(n, fill::ones);
    U = chol(AA / static_cast<float>(m) + rho * diagmat(eyeN));
  } else {
    arma::vec eyeM(m, fill::ones);
    //U = chol(diagmat(eyeM * static_cast<float>(m)) + (1.0 / rho) * (A * A.t()));
    U = chol(diagmat(eyeM * rho * static_cast<float>(m)) + AA);
  }
}
// Linear regression model with LASSO penalty:
// evaluate the objective function
// this is the fast version
double lasso_objfun_fast(const arma::mat& ATA_CHOL_U,
                         const arma::colvec& ATb,
                         const double b2,
                         const double lambda, 
                         const arma::colvec& x,
                         const arma::colvec& z,
                         const int m,
                         const int n) {
  
  /* Variable delcaration           */
  int n1 = 0;
  double ll = 0.0, objfun = 0.0;
  
  /* define dimensions        */
  if (m >= n) {
    n1 = n;
  } else {
    n1 = m;
  }
  
  /* Definition of vectors and matrices     */
  arma::colvec tmp(n1, fill::zeros);
  
  // evaluate the quadratic form (negative log-likelihood)
  // previously divided by m
  if (m >= n) {
    tmp = trimatu(ATA_CHOL_U) * x;
  } else {
    tmp = trimatu(ATA_CHOL_U.submat(0, 0, m-1, m-1)) * x.subvec(0, m-1) + ATA_CHOL_U.submat(0, m, m-1, n-1) * x.subvec(m, n-1);
  }
  ll = as_scalar(tmp.t() * tmp) + b2 - 2.0 * as_scalar(x.t() * ATb);
  
  // evaluate the objective function
  objfun = 0.5 * ll + lambda * norm(z, 1);
  
  /* Return output      */
  return(objfun);
}
// Update x for linreg with lasso penalty
arma::vec admm_lasso_update_x(const arma::mat& A,
                              const arma::mat& U,
                              const arma::mat& L,
                              const arma::colvec& ATb,
                              const arma::vec& z,
                              const arma::vec& u,
                              const double rho) {
  
  /* Variable delcaration           */
  const int m = A.n_rows, n = A.n_cols;
  
  /* Definition of vectors and matrices     */
  arma::colvec x(n, fill::zeros);
  arma::colvec q(n, fill::zeros);
  
  // 4-1. update 'x'
  q = ATb + rho * (z - u); // temporary value
  if (m >= n) {
    x = solve(trimatu(U), solve(trimatl(L), q));
  } else {
    //x = q / rho - (A.t() * solve(trimatu(U), solve(trimatl(L), A * q))) / rho2;
    x = q / rho - (A.t() * solve(trimatu(U), solve(trimatl(L), A * q))) / rho;
  }
  return(x);
}

/*  Adaptive LASSO via ADMM     */
arma::colvec adalasso_prox(const arma::colvec& a,
                           const arma::vec kappa) {
  
  const int n = a.n_elem;
  arma::colvec y(n, fill::zeros);
  
  for (int i=0; i<n; i++) {
    // first term : max(0, a-kappa)
    if (a(i)-kappa(i) > 0.0) {
      y(i) = a(i)-kappa(i);
    }
    // second term : -max(0, -a-kappa)
    if (-a(i)-kappa(i) > 0.0) {
      y(i) = a(i) + kappa(i);
    }
  }
  return(y);
}
arma::mat adalasso_Fmat(const arma::vec& var_weights) {
    
  /* Get dimensions             */
  const int p = var_weights.n_elem;

  /* Definition of vectors and matrices     */
  arma::mat Fmat(p, p, fill::zeros);
  
  /* Get the F1 matrix         */
  Fmat = diagmat(var_weights);
    
  /* Return output      */
  return(Fmat);
}
// evaluate the objective function
double adalasso_objfun(const arma::mat& A,
                       const arma::colvec& b,
                       const double lambda,
                       const arma::colvec& x,
                       const arma::vec& var_weights,
                       double& rss,
                       double& pen) {
  
  const int m = A.n_rows;
  double objfun = 0.0;
  arma::vec resid(m, fill::zeros);
  
  /* compute the objective function   */
  resid  = A * x - b;
  rss    = as_scalar(resid.t() * resid);
  pen    = norm(diagmat(var_weights) * x, 1);
  objfun = 0.5 * rss / static_cast<float>(m) + lambda * pen;
  
  /* Return output      */
  return(objfun);
}
arma::vec adalasso_residual(const arma::colvec& x,
                            const arma::colvec& z) {
  
  return(x - z);
}
arma::vec adalasso_dual_residual(const arma::colvec& z,
                                 const arma::colvec& z_old,
                                 double rho) {
  
  return(rho * (z - z_old));
}
// Evaluate the RIDGE matrix: returns the Cholesky (Upper triangular matrix)
arma::mat adalasso_factor_fast_large_m(const arma::mat& ATA,
                                       const double rho) {

  /* Variable delcaration           */
  const int n = ATA.n_cols;
  arma::mat U(n, n, fill::zeros);
  arma::mat eyen(n, n, fill::eye);
  
  /* Compute the Cholesky factor of the RIDGE matrix        */
  U = chol(ATA + rho * eyen);
  
  /* return output         */
  return(U);
}
arma::mat adalasso_factor_fast_large_n(const arma::mat& X,
                                       const double rho,
                                       const int m) {
  
  /* Definition of vectors and matrices     */
  arma::mat U(m, m, fill::zeros);
  arma::vec eyeM(m, fill::ones);

  /* Compute the Cholesky factor of the RIDGE matrix        */
  U = chol(diagmat(eyeM * rho * static_cast<float>(m)) + X);

  /* return output         */
  return(U);
}
// =========================================================
// admm_glasso
// ADMM for linear regression models with group-LASSO penalty

/* Group Lasso vector (or block) soft thresholding operator        */
arma::colvec glasso_prox(const arma::colvec& a,
                         const double kappa) {

  const int n = a.n_elem;
  double x;
  arma::colvec y(n, fill::zeros);

  x = 1.0 - kappa / norm(a);
  if (x > 0.0) {
    y = a * x;
  }
  return(y);
}
// Evaluate the RIDGE matrix: returns the Cholesky (Upper triangular matrix)
arma::mat glasso_factor(const arma::mat& A,
                        const arma::mat& FTF,
                        const double rho) {

  /* Variable delcaration           */
  const int m = A.n_rows;
  const int n = A.n_cols;
  arma::mat U;
  
  /* Definition of vectors and matrices     */
  arma::mat ATA(n, n, fill::zeros);
  
  /* Precompute relevant quantities         */
  ATA = A.t() * A / static_cast<float>(m);
    
  /* Compute the Cholesky factor of the RIDGE matrix        */
  U = chol(ATA + rho * FTF);
  
  /* return output         */
  return(U);
}
// Evaluate the RIDGE matrix: returns the Cholesky (Upper triangular matrix)
arma::mat glasso_factor_fast_large_m(const arma::mat& ATA,
                                     const arma::mat& FTF,
                                     const double rho) {

  /* Variable delcaration           */
  const int n = ATA.n_cols;
  arma::mat U(n, n, fill::zeros);
  
  /* Compute the Cholesky factor of the RIDGE matrix        */
  U = chol(ATA + rho * diagmat(FTF));
  
  /* return output         */
  return(U);
}
// Evaluate the RIDGE matrix: returns the Cholesky (Upper triangular matrix)
// previous code with smoothing
arma::mat glasso_factor_smo_fast_large_m(const arma::mat& ATA,
                                         const arma::mat& FTF,
                                         const arma::mat& DTD,
                                         const double rho,
                                         const double lambda) {

  /* Variable delcaration           */
  const int n = ATA.n_cols;
  arma::mat U(n, n, fill::zeros);
  
  /* Compute the Cholesky factor of the RIDGE matrix        */
  U = chol(ATA + rho * diagmat(FTF) + lambda * DTD);
  
  /* return output         */
  return(U);
}

arma::mat glasso_factor_smo_fast_large_m2(const arma::mat& XtTXt,
                                          const arma::mat& FTF,
                                          const double rho) {

  /* Variable delcaration           */
  const int n = XtTXt.n_cols;
  arma::mat U(n, n, fill::zeros);
  
  /* Compute the Cholesky factor of the RIDGE matrix        */
  U = chol(XtTXt + rho * diagmat(FTF));
  
  /* return output         */
  return(U);
}

arma::mat glasso_factor_fast_large_n(const arma::mat& X,
                                     const double rho,
                                     const int m) {
  
  /* Definition of vectors and matrices     */
  arma::mat U(m, m, fill::zeros);
  arma::vec eyeM(m, fill::ones);

  /* Compute the Cholesky factor of the RIDGE matrix        */
  U = chol(diagmat(eyeM * rho * static_cast<float>(m)) + X);

  /* return output         */
  return(U);
}

// previous code with smoothing
arma::mat glasso_factor_smo_fast_large_n(const arma::mat& X,
                                         const int m) {
  
  /* Definition of vectors and matrices     */
  arma::mat U(m, m, fill::zeros);
  arma::vec eyeM(m, fill::ones);

  /* Compute the Cholesky factor of the RIDGE matrix        */
  U = chol(diagmat(eyeM * static_cast<float>(m)) + X);

  /* return output         */
  return(U);
}
arma::mat glasso_factor_smo_fast_large_n2(const arma::mat& X, 
                                          const double rho,
                                          const int m) {
  
  /* Definition of vectors and matrices     */
  arma::mat U(m, m, fill::zeros);
  arma::vec eyeM(m, fill::ones);

  /* Compute the Cholesky factor of the RIDGE matrix        */
  U = chol(diagmat(eyeM * rho) + X);

  /* return output         */
  return(U);
}

arma::vec glasso_residual(const arma::mat& F,
                          const arma::colvec& x,
                          const arma::colvec& z) {
  return(F * x - z);
}

arma::vec glasso_dual_residual(const arma::mat& F,
                               const arma::colvec& z,
                               const arma::colvec& z_old,
                               const double rho) {
  return(rho * F.t() * (z - z_old));
}

arma::vec glasso_residual_sparse(const arma::sp_mat& F,
                                 const arma::colvec& x,
                                 const arma::colvec& z) {
  return(F * x - z);
}

arma::vec glasso_dual_residual_sparse(const arma::sp_mat& F,
                                      const arma::colvec& z,
                                      const arma::colvec& z_old,
                                      const double rho) {
  return(rho * F.t() * (z - z_old));
}

arma::mat glasso_Gvec2F1mat(const arma::rowvec& Gvec) {
    
  /* Get dimensions             */
  const int p = Gvec.n_elem;
  const int n = as_scalar(sum(Gvec));
  
  /* Definition of vectors and matrices     */
  arma::mat F1mat(n, p, fill::zeros);
  arma::uvec idx(p, fill::zeros);

  /* Get the F1 matrix    */
  idx = find(Gvec == 1);
  for (int k=0; k<n; k++) {
    F1mat(k, idx(k)) = 1.0;
  }
  
  /* Return output      */
  return(F1mat);
}

arma::sp_mat glasso_Gvec2F1mat_spmat(const arma::rowvec& Gvec) {
    
  /* Get dimensions             */
  const int p = Gvec.n_elem;
  const int n = as_scalar(sum(Gvec));
  
  /* Definition of vectors and matrices     */
  arma::sp_mat F1mat(n, p);
  arma::uvec idx(p, fill::zeros);

  /* Get the F1 matrix    */
  idx = find(Gvec == 1);
  for (int k=0; k<n; k++) {
    F1mat(k, idx(k)) = 1.0;
  }
  
  /* Return output      */
  return(F1mat);
}

arma::mat glasso_Gmat2Fmat(const arma::mat& Gmat,
                           const arma::vec& group_weights,
                           const arma::vec& var_weights) {
    
  /* Get dimensions             */
  int n1 = 0, g_in = 0, g_end = 0;
  const int G = Gmat.n_rows;
  const int p = Gmat.n_cols;
  const int n = as_scalar(sum(sum(Gmat, 0), 1));
  
  /* Definition of vectors and matrices     */
  arma::mat Fmat(n, p, fill::zeros);
  arma::mat Tmat(p, p, fill::zeros);
  
  /* Precompute relevant quantities         */
  Tmat = diagmat(var_weights);
    
  /* Get the F matrix    */
  g_in = 0;
  for (int g=0; g<G; g++) {
    arma::mat F1mat = glasso_Gvec2F1mat(Gmat.row(g));
    for (int r=0; r<(int)F1mat.n_rows; r++) {
      F1mat.row(r) *= group_weights(g);
    }
    F1mat                            = F1mat * diagmat(Tmat);
    n1                               = F1mat.n_rows;
    g_end                            = g_in + n1 - 1;
    Fmat.submat(g_in, 0, g_end, p-1) = F1mat;
    g_in                             = g_end + 1;
  }

  /* Return output      */
  return(Fmat);
}

// [[Rcpp::export]]
arma::sp_mat glasso_Gmat2Fmat_sparse(const arma::mat& Gmat,
                                     const arma::vec& var_weights) {
    
  /* Get dimensions             */
  int n1 = 0, g_in = 0, g_end = 0;
  const int G = Gmat.n_rows;
  const int p = Gmat.n_cols;
  const int n = as_scalar(sum(sum(Gmat, 0), 1));
  
  /* Definition of vectors and matrices     */
  arma::sp_mat Fmat(n, p);
  arma::mat Tmat(p, p, fill::zeros);
  
  /* Precompute relevant quantities         */
  Tmat = diagmat(var_weights);
    
  /* Get the F matrix    */
  g_in = 0;
  for (int g=0; g<G; g++) {
    arma::sp_mat F1mat               = glasso_Gvec2F1mat_spmat(Gmat.row(g));
    F1mat                            = F1mat * diagmat(Tmat);
    n1                               = F1mat.n_rows;
    g_end                            = g_in + n1 - 1;
    Fmat.submat(g_in, 0, g_end, p-1) = F1mat;
    g_in                             = g_end + 1;
  }

  /* Return output      */
  return(Fmat);
}

// [[Rcpp::export]]
arma::sp_mat ovglasso_Gmat2Fmat_sparse(const arma::mat& Gmat,
                                       const arma::vec& var_weights) {
    
  /* Get dimensions             */
  int n1 = 0, g_in = 0, g_end = 0;
  const int G = Gmat.n_rows;
  const int p = Gmat.n_cols;
  const int n = as_scalar(sum(sum(Gmat, 0), 1));
  
  /* Definition of vectors and matrices     */
  arma::sp_mat Fmat(n, p);
  arma::mat Tmat(p, p, fill::zeros);
  
  /* Precompute relevant quantities         */
  Tmat = diagmat(var_weights);
  //Tmat = diagmat(1.0 / var_weights);
    
  /* Get the F matrix    */
  g_in = 0;
  for (int g=0; g<G; g++) {
    arma::sp_mat F1mat               = glasso_Gvec2F1mat_spmat(Gmat.row(g));
    F1mat                            = F1mat * diagmat(Tmat);
    n1                               = F1mat.n_rows;
    g_end                            = g_in + n1 - 1;
    Fmat.submat(g_in, 0, g_end, p-1) = F1mat;
    g_in                             = g_end + 1;
  }

  /* Return output      */
  return(Fmat);
}

arma::mat spglasso_Gmat2Fmat(const arma::mat& Gmat,
                             const arma::vec& group_weights,
                             const arma::vec& var_weights,
                             const arma::vec& var_weights_L1) {
    
  /* Get dimensions             */
  int n1 = 0, g_in = 0, g_end = 0;
  const int G = Gmat.n_rows;
  const int p = Gmat.n_cols;
  const int n = as_scalar(sum(sum(Gmat, 0), 1));
  
  /* Definition of vectors and matrices     */
  arma::mat Fmat(n, p, fill::zeros);
  arma::mat F1all(p, p, fill::eye);
  arma::mat Tmat(p, p, fill::zeros);
  arma::mat Tmat_L1(p, p, fill::zeros);
  
  /* Precompute relevant quantities         */
  Tmat    = diagmat(var_weights);
  Tmat_L1 = diagmat(var_weights_L1);
    
  /* Get the F matrix    */
  g_in = 0;
  for (int g=0; g<G; g++) {
    arma::mat F1mat = glasso_Gvec2F1mat(Gmat.row(g));
    for (int r=0; r<(int)F1mat.n_rows; r++) {
      F1mat.row(r) *= group_weights(g);
    }
    F1mat                            = F1mat * diagmat(Tmat);
    n1                               = F1mat.n_rows;
    g_end                            = g_in + n1 - 1;
    Fmat.submat(g_in, 0, g_end, p-1) = F1mat;
    g_in                             = g_end + 1;
  }
  arma::mat Fmat_ALL = join_vert(Fmat, F1all * diagmat(Tmat_L1));

  /* Return output      */
  return(Fmat_ALL);
}

arma::sp_mat spglasso_Gmat2Fmat_sparse(const arma::mat& Gmat,
                                       const arma::vec& var_weights,
                                       const arma::vec& var_weights_L1) {
    
  /* Get dimensions             */
  int n1 = 0, g_in = 0, g_end = 0;
  const int G = Gmat.n_rows;
  const int p = Gmat.n_cols;
  const int n = as_scalar(sum(sum(Gmat, 0), 1));
  
  /* Definition of vectors and matrices     */
  arma::sp_mat Fmat(n + p, p);
  arma::mat Tmat(p, p, fill::zeros);
  arma::mat Tmat_L1(p, p, fill::zeros);
  
  /* Precompute relevant quantities         */
  Tmat    = diagmat(var_weights);
  Tmat_L1 = diagmat(var_weights_L1);
    
  /* Get the F matrix    */
  g_in = 0;
  for (int g=0; g<G; g++) {
    arma::sp_mat F1mat               = glasso_Gvec2F1mat_spmat(Gmat.row(g));
    F1mat                            = F1mat * diagmat(Tmat);
    n1                               = F1mat.n_rows;
    g_end                            = g_in + n1 - 1;
    Fmat.submat(g_in, 0, g_end, p-1) = F1mat;
    g_in                             = g_end + 1;
  }
  g_end                            = g_in + p - 1;
  Fmat.submat(g_in, 0, g_end, p-1) = diagmat(Tmat_L1);
  
  /* Return output      */
  return(Fmat);
}

arma::mat glasso_Gmat2Dmat(const arma::mat& Gmat) {
    
  /* Get dimensions             */
  int n1 = 0, g_in = 0, g_end = 0;
  const int G = Gmat.n_rows;
  const int p = Gmat.n_cols;
  const int n = as_scalar(sum(sum(Gmat, 0), 1));
  
  /* Definition of vectors and matrices     */
  arma::mat Fmat(n, p, fill::zeros);
  arma::mat Dmat(p, p, fill::zeros);
  arma::mat Dmat_inv(p, p, fill::zeros);
  arma::mat Dmat_half_inv(p, p, fill::zeros);
  arma::mat Hmat(n, p, fill::zeros);
  
  /* Get the F matrix    */
  g_in  = 0;
  for (int g=0; g<G; g++) {
    arma::mat F1mat                  = glasso_Gvec2F1mat(Gmat.row(g));
    n1                               = F1mat.n_rows;
    g_end                            = g_in + n1 - 1;
    Fmat.submat(g_in, 0, g_end, p-1) = F1mat;
    g_in                             = g_end + 1;
  }
  
  /* Get the normalized F matrix  (H matrix)  */
  Dmat          = Fmat.t() * Fmat;
  Dmat_inv      = inv(diagmat(Dmat));
  Dmat_half_inv = sqrtmat_sympd(Dmat_inv);
  Hmat          = Fmat * Dmat_half_inv;

  /* Return output      */
  return(Hmat);
}

// LASSO penalty: evaluate the RIDGE matrix, returns the Cholesky (Upper triangular matrix)
arma::mat lasso_factor_cov_fast_large_m(const arma::mat& XMX,
                                        double rho) {
  
  /* Variable delcaration           */
  const int n = XMX.n_cols;
  arma::vec eye_n(n, fill::ones);
  arma::mat U(n, n, fill::zeros);
  
  /* Compute the Cholesky factor of the RIDGE matrix        */
  U = chol(diagmat(eye_n) * rho + XMX);
  
  /* return output         */
  return(U);
}

arma::mat lasso_factor_cov_fast_large_n(const arma::mat& XX,
                                        double rho,
                                        const int m) {

  /* Definition of vectors and matrices     */
  arma::mat U(m, m, fill::zeros);
  arma::vec ones_m(m, fill::ones);
  
  /* Compute the Cholesky factor of the RIDGE matrix        */
  U = chol(diagmat(ones_m * rho * static_cast<float>(m)) + XX);

  /* return output         */
  return(U);
}

// Group-LASSO penalty: evaluate the RIDGE matrix, returns the Cholesky (Upper triangular matrix)
arma::mat glasso_factor_cov_fast_large_m(const arma::mat& XMX,
                                         const arma::mat& FTF,
                                         double rho) {

  /* Variable delcaration           */
  const int n = XMX.n_cols;
  arma::mat U(n, n, fill::zeros);
  
  /* Compute the Cholesky factor of the RIDGE matrix        */
  U = chol(XMX + rho * diagmat(FTF));
  
  /* return output         */
  return(U);
}

// previous code with smoothing
arma::mat glasso_factor_cov_smo_fast_large_m(const arma::mat& XMX,
                                             const arma::mat& FTF,
                                             const arma::mat& DTD,
                                             const double lambda,
                                             double rho) {

  /* Variable delcaration           */
  const int n = XMX.n_cols;
  arma::mat U(n, n, fill::zeros);
  
  /* Compute the Cholesky factor of the RIDGE matrix        */
  U = chol(XMX + rho * diagmat(FTF) + lambda * DTD);
  
  /* return output         */
  return(U);
}

arma::mat glasso_factor_cov_fast_large_n(const arma::mat& XMX,
                                         double rho,
                                         const int m) {

  /* Definition of vectors and matrices     */
  arma::mat U(m, m, fill::zeros);
  arma::vec ones_m(m, fill::ones);
  
  /* Compute the Cholesky factor of the RIDGE matrix        */
  U = chol(diagmat(ones_m * rho * static_cast<float>(m)) + XMX);

  /* return output         */
  return(U);
}

// stesso codice precedente ma con smoothing effects
arma::mat glasso_factor_cov_smo_fast_large_n(const arma::mat& X,
                                             double rho,
                                             const int m) {

  /* Definition of vectors and matrices     */
  arma::mat U(m, m, fill::zeros);
  arma::vec ones_m(m, fill::ones);
  
  /* Compute the Cholesky factor of the RIDGE matrix        */
  U = chol(diagmat(ones_m * rho) + X);

  /* return output         */
  return(U);
}

// Linear regression model with LASSO and adaptive-LASSO penalty:
// evaluate the penalty function
// evaluate the objective function
double lm_rss(const arma::mat& A,
              const arma::vec& b,
              const arma::vec& x) {
  
  const int m = A.n_rows;
  double rss = 0.0;
  arma::vec resid(m, fill::zeros);
  
  /* compute the rss   */
  resid = A * x - b;
  rss   = as_scalar(resid.t() * resid);
  
  /* Return output      */
  return(rss);
}

double lm_rss_fast(const arma::mat& ATA_CHOL_U,
                   const arma::vec& ATb,
                   const double b2,
                   const arma::vec& x,
                   const int m,
                   const int n) {
  
  /* Variable delcaration           */
  int n1 = 0;
  double rss = 0.0;
  
  /* define dimensions        */
  if (m >= n) {
    n1 = n;
  } else {
    n1 = m;
  }
  
  /* Definition of vectors and matrices     */
  arma::colvec tmp(n1, fill::zeros);
  
  // evaluate the quadratic form (rss)
  // previously divided by m
  if (m >= n) {
    tmp = trimatu(ATA_CHOL_U) * x;
  } else {
    tmp = trimatu(ATA_CHOL_U.submat(0, 0, m-1, m-1)) * x.subvec(0, m-1) + ATA_CHOL_U.submat(0, m, m-1, n-1) * x.subvec(m, n-1);
  }
  rss = as_scalar(tmp.t() * tmp) + b2 - 2.0 * as_scalar(x.t() * ATb);
    
  /* Return output      */
  return(rss);
}

double lm_cov_rss_fast(const arma::mat& XX_CHOL_U,
                       const arma::mat& ZZ_CHOL_U,
                       const arma::mat& ZX,
                       const arma::vec& Xy,
                       const arma::vec& Zy,
                       const double y2,
                       const arma::colvec& x,
                       const arma::colvec& v,
                       const int m) {
  
  /* Variable delcaration           */
  int n1 = 0;
  double rss = 0.0;
  
  /* Get dimensions             */
  const int n_spcov = Xy.n_elem;
  const int n_cov   = Zy.n_elem;
  
  /* define dimensions        */
  if (m >= n_spcov) {
    n1 = n_spcov;
  } else {
    n1 = m;
  }
  
  /* Definition of vectors and matrices     */
  arma::colvec tmp_x(n1, fill::zeros);
  arma::colvec tmp_z(n_cov, fill::zeros);
  
  // evaluate the quadratic form (rss)
  // previously divided by m
  if (m >= n_spcov) {
    tmp_x = trimatu(XX_CHOL_U) * x;
  } else {
    tmp_x = trimatu(XX_CHOL_U.submat(0, 0, m-1, m-1)) * x.subvec(0, m-1) + XX_CHOL_U.submat(0, m, m-1, n_spcov-1) * x.subvec(m, n_spcov-1);
  }
  tmp_z = trimatu(ZZ_CHOL_U) * v;
  rss   = as_scalar(tmp_x.t() * tmp_x) + as_scalar(tmp_z.t() * tmp_z) + y2 - 2.0 * as_scalar(x.t() * Xy) - 2.0 * as_scalar(v.t() * Zy) - 2.0 * as_scalar(v.t() * ZX * x);
    
  /* return output      */
  return(rss);
}

double lasso_penalty(const arma::vec& x) {
  
  /* Variable delcaration           */
  double pen = 0.0;
  
  // evaluate the penalty function
  pen = as_scalar(norm(x, 1));
  
  /* Return output      */
  return(pen);
}

double adalasso_penalty(const arma::vec& x,
                        const arma::vec& var_weights) {
  
  /* Variable delcaration           */
  double pen = 0.0;
  
  // evaluate the penalty function
  pen = as_scalar(norm(diagmat(var_weights) * x, 1));
  
  /* Return output      */
  return(pen);
}

// Linear regression model with group-LASSO penalty:
// evaluate the penalty function
double glasso_penalty(const arma::vec& x,
                      const arma::mat& groups,
                      const arma::vec& group_weights,
                      const arma::vec& var_weights) {
   
   /* Variable delcaration           */
   int G = groups.n_rows, p = x.n_elem;
   double pen = 0.0;
   arma::uvec g_idx;
   arma::vec z(p, fill::zeros);
   
   // evaluate the group-Lasso penalty function
   z   = var_weights % x;
   pen = 0.0;
   for (int g = 0; g < G; g++) {
     g_idx = arma::find(groups.row(g) == 1);
     if (!g_idx.is_empty()) {
       pen += group_weights(g) * arma::norm(z.elem(g_idx), 2);
     }
   }
   
   /* Return output      */
   return(pen);
 }

// Linear regression model with sparse group-LASSO penalty:
// evaluate the penalty function
double spglasso_penalty(const arma::vec& x,
                        const arma::mat& groups,
                        const arma::vec& group_weights,
                        const arma::vec& var_weights,
                        const arma::vec& var_weights_L1,
                        const double alpha) {
  
  /* Variable delcaration           */
  int G = groups.n_rows, p = x.n_elem;
  double pen = 0.0, pen_GL = 0.0, pen_L1 = 0.0;
  arma::uvec g_idx;
  arma::vec z(p, fill::zeros);
  
  // evaluate the group-Lasso penalty function
  z   = var_weights % x;
  pen = 0.0;
  for (int g = 0; g < G; g++) {
    g_idx = arma::find(groups.row(g) == 1);
    if (!g_idx.is_empty()) {
      pen += group_weights(g) * arma::norm(z.elem(g_idx), 2);
    }
  }
  pen_L1 = as_scalar(norm(diagmat(var_weights_L1) * x, 1));
  pen    = (1.0 - alpha) * pen_GL + alpha  * pen_L1;
  
  /* Return output      */
  return(pen);
}

// Linear regression model with overlap group-LASSO penalty:
// evaluate the penalty function
double ovglasso_penalty(const arma::vec& x,
                        const arma::mat& groups,
                        const arma::vec& group_weights,
                        const arma::vec& var_weights) {
  
  /* Variable delcaration           */
  int G = groups.n_rows, p = x.n_elem;
  double pen = 0.0;
  arma::uvec g_idx;
  arma::vec z(p, fill::zeros);
  
  // evaluate the group-Lasso penalty function
  z   = var_weights % x;
  pen = 0.0;
  for (int g = 0; g < G; g++) {
    g_idx = arma::find(groups.row(g) == 1);
    if (!g_idx.is_empty()) {
      pen += group_weights(g) * arma::norm(z.elem(g_idx), 2);
    }
  }
  
  /* Return output      */
  return(pen);
}

/* sparse overlap Group-Lasso                       */
arma::sp_mat spovglasso_Gmat2Fmat_sparse(const arma::mat& Gmat,
                                         const arma::vec& var_weights,
                                         const arma::vec& var_weights_L1) {
    
  /* Get dimensions             */
  int n1 = 0, g_in = 0, g_end = 0;
  const int G = Gmat.n_rows;
  const int p = Gmat.n_cols;
  const int n = as_scalar(sum(sum(Gmat, 0), 1));
  
  /* Definition of vectors and matrices     */
  arma::sp_mat Fmat(n + p, p);
  arma::mat Tmat(p, p, fill::zeros);
  arma::mat Tmat_L1(p, p, fill::zeros);
  
  /* Precompute relevant quantities         */
  Tmat    = diagmat(var_weights);
  Tmat_L1 = diagmat(var_weights_L1);
    
  /* Get the F matrix    */
  g_in = 0;
  for (int g=0; g<G; g++) {
    arma::sp_mat F1mat               = glasso_Gvec2F1mat_spmat(Gmat.row(g));
    F1mat                            = F1mat * diagmat(Tmat);
    n1                               = F1mat.n_rows;
    g_end                            = g_in + n1 - 1;
    Fmat.submat(g_in, 0, g_end, p-1) = F1mat;
    g_in                             = g_end + 1;
  }
  g_end                            = g_in + p - 1;
  Fmat.submat(g_in, 0, g_end, p-1) = diagmat(Tmat_L1);

  /* Return output      */
  return(Fmat);
}

// Linear regression model with sparse overlap group-LASSO penalty:
// evaluate the penalty function
double spovglasso_penalty(const arma::vec& x,
                          const arma::mat& groups,
                          const arma::vec& group_weights,
                          const arma::vec& var_weights,
                          const arma::vec& var_weights_L1,
                          const double alpha) {
  
  /* Variable delcaration           */
  int G = groups.n_rows, p = x.n_elem;
  double pen = 0.0, pen_GL = 0.0, pen_L1 = 0.0;
  arma::uvec g_idx;
  arma::vec z(p, fill::zeros);
  
  // evaluate the group-Lasso penalty function
  z   = var_weights % x;
  pen = 0.0;
  for (int g = 0; g < G; g++) {
    g_idx = arma::find(groups.row(g) == 1);
    if (!g_idx.is_empty()) {
      pen += group_weights(g) * arma::norm(z.elem(g_idx), 2);
    }
  }
  pen_L1 = as_scalar(norm(diagmat(var_weights_L1) * x, 1));
  pen    = (1.0 - alpha) * pen_GL + alpha  * pen_L1;
  
  /* Return output      */
  return(pen);
}














