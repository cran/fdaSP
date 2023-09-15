// Utility routines

// Authors:
//           Bernardi Mauro, University of Padova
//           Last update: July 12, 2022

// List of implemented methods:

// [[Rcpp::depends(RcppArmadillo)]]
#include "ADMMutilsLM.h"

// ==========================================================
// admm_enet: Linear regression model with ENET penalty
arma::colvec enet_prox(arma::colvec a, const double kappa) {
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

double enet_objfun(const arma::mat &A, const arma::colvec &b, const double lambda,
                   const double alpha, const arma::colvec &x, const arma::colvec &z) {
  return(0.5 * norm(A * x - b, 2) + lambda * alpha * norm(z, 1) + 0.5 * (1.0 - alpha) * lambda * norm(x, 2));
}

arma::mat enet_factor(arma::mat A, double rho) {
  const int n = A.n_cols;
  arma::mat U;
  arma::vec onesN(n, fill::ones);
  U = arma::chol(A.t() * A + rho * diagmat(onesN));
  return(U);
}

// ======================================================
// admm_genlasso: Linear regression model with generalized LASSO penalty
arma::colvec genlasso_shrinkage(arma::colvec a, const double kappa) {
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

double genlasso_objective(const arma::mat &A,const arma::colvec &b, const arma::mat &D,
                          const double lambda, const arma::colvec &x, const arma::colvec &z) {
  return(pow(norm(A*x-b,2),2)/2 + lambda*norm(D*x,1));
}

arma::mat genlasso_factor(const arma::mat &A, double rho,const arma::mat &D) {
  arma::mat U;
  U = chol(A.t()*A + rho*D.t()*D);
  return(U);
}

// =========================================================
// admm_lasso
// ADMM for linear regression models with LASSO penalty
// elementwise soft thresholding operator
arma::colvec lasso_prox(arma::colvec a, const double kappa) {
  
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

// Linear regression model with LASSO penalty: evaluate the objective function
double lasso_objfun(arma::mat A, arma::colvec b, const double lambda, arma::colvec x, arma::colvec z) {
  const int m = A.n_rows;
  return(0.5 * norm(A * x - b, 2) / static_cast<float>(m) + lambda * norm(z, 1));
}

// Evaluate the RIDGE matrix: returns the Cholesky (Upper triangular matrix)
void lasso_factor(arma::mat& U, arma::mat A, double rho) {
  
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
void lasso_factor_fast(arma::mat& U, arma::mat AA, double rho, const int m, const int n) {

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
double lasso_objfun_fast(arma::mat ATA_CHOL_U, arma::colvec ATb, const double b2,
                         const double lambda, arma::colvec x, arma::colvec z, const int m, const int n) {
  
  /* Variable delcaration           */
  int n1;
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
arma::vec admm_lasso_update_x(arma::mat A, arma::mat U, arma::mat L,
                              arma::colvec ATb, arma::vec z, arma::vec u, const double rho) {
  
  /* Variable delcaration           */
  const int m = A.n_rows;
  const int n = A.n_cols;
  //double rho2 = rho * rho;
  
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

/*
* Adaptive LASSO via ADMM (from Stanford)
* http://stanford.edu/~boyd/papers/pdf/admm_distr_stats.pdf
* page 19 : section 3.3.1 : stopping criteria part (3.12).
*/
arma::mat adalasso_Fmat(arma::vec var_weights) {
    
  /* Get dimensions             */
  const int p = var_weights.n_elem;

  /* Definition of vectors and matrices     */
  arma::mat Fmat(p, p, fill::zeros);
  
  /* Get the F1 matrix         */
  Fmat = diagmat(var_weights);
    
  /* Return output      */
  return(Fmat);
}

// Linear regression model with adaptive-LASSO penalty: evaluate the objective function
double adalasso_objfun(arma::mat A, arma::colvec b, const double lambda, arma::colvec x, arma::colvec z, arma::colvec var_weights) {
  const int m = A.n_rows;
  return(0.5 * norm(A * x - b, 2) / static_cast<float>(m) + lambda * norm(diagmat(var_weights) * z, 1));
}

double adalasso_objfun_fast(arma::mat ATA_CHOL_U, arma::colvec ATb, const double b2,
                            const double lambda, arma::colvec x, arma::colvec z, const int m, const int n, arma::colvec var_weights) {
  
  /* Variable delcaration           */
  int n1;
  double ll = 0.0, objfun = 0.0;
  double pen = 0.0;
  
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
  
  // evaluate the penalty function
  pen = norm(diagmat(var_weights) * z, 1);
  
  // evaluate the objective function
  objfun = 0.5 * ll + lambda * pen;
  
  /* Return output      */
  return(objfun);
}

arma::vec adalasso_residual(arma::mat F, arma::colvec x, arma::colvec z) {
  return(diagmat(F) * x - z);
}

arma::vec adalasso_dual_residual(arma::mat F, arma::colvec z, arma::colvec z_old, double rho) {
  return(rho * diagmat(F) * (z - z_old));
}

// Evaluate the RIDGE matrix: returns the Cholesky (Upper triangular matrix)
arma::mat adalasso_factor_fast_large_m(arma::mat ATA, arma::mat FTF, double rho) {

  /* Variable delcaration           */
  const int n = ATA.n_cols;
  arma::mat U(n, n, fill::zeros);
  
  /* Compute the Cholesky factor of the RIDGE matrix        */
  U = chol(ATA + rho * diagmat(FTF));
  
  /* return output         */
  return(U);
}

arma::mat adalasso_factor_fast_large_n(arma::mat X, double rho, const int m) {
  
  /* Definition of vectors and matrices     */
  arma::mat U(m, m, fill::zeros);
  arma::vec eyeM(m, fill::ones);

  /* Compute the Cholesky factor of the RIDGE matrix        */
  U = chol(diagmat(eyeM * rho * static_cast<float>(m)) + X);

  /* return output         */
  return(U);
}

/*
 LASSO via ADMM, just to compute the whole path fast.
*/
Rcpp::List admm_lasso_update_1lambda(const arma::mat& A, const arma::vec& b, const arma::mat& ATA_CHOL_U, const arma::mat& AA,
                                     const arma::colvec& ATb, const double b2, const int m, const int n,
                                     arma::colvec u, arma::colvec z, const double lambda,
                                     bool rho_adaptation, double rho, const double tau, const double mu,
                                     const double reltol, const double abstol, const int maxiter, const int ping) {
  
  /* Variable delcaration           */
  int k, n1;
  double sqrtn, rho_old, elTime = 0.0, mse;
  if (m >= n) {
    n1 = n;
  } else {
    n1 = m;
  }
  
  /* Definition of vectors and matrices     */
  arma::colvec x(n, fill::zeros);
  arma::colvec q(n, fill::zeros);
  arma::colvec z_old(n, fill::zeros);
  arma::mat U(n1, n1, fill::zeros);
  arma::mat L(n1, n1, fill::zeros);
  arma::vec h_objval(maxiter, fill::zeros);
  arma::vec h_r_norm(maxiter, fill::zeros);
  arma::vec h_s_norm(maxiter, fill::zeros);
  arma::vec h_eps_pri(maxiter, fill::zeros);
  arma::vec h_eps_dual(maxiter, fill::zeros);
  arma::vec rho_store(maxiter+1, fill::zeros);
  arma::colvec fitted(m, fill::zeros);
  arma::colvec residuals(m, fill::zeros);

  /* Variable definition            */
  sqrtn = std::sqrt(static_cast<float>(n));

  /* ::::::::::::::::::::::::::::::::::::::::::::::
   Set the computational time                                 */
  wall_clock timer;
  timer.tic();
  
  /* store rho (only for adaptation)         */
  rho_store(0) = rho;
  rho_old      = rho;

  /* Precompute relevant quantities         */
  lasso_factor_fast(U, AA, rho, m, n); // returns upper
  L   = U.t();
  
  /* main loop  */
  for (k=0; k<maxiter; k++) {

    // 4-1. update 'x'
    if (rho != rho_old) {
      lasso_factor_fast(U, AA, rho, m, n); // returns upper
      L = U.t();
    }
    q = ATb + rho * (z - u); // temporary value
    if (m >= n) {
      x = solve(trimatu(U), solve(trimatl(L), q));
    } else {
      x = q / rho - (A.t() * solve(trimatu(U), solve(trimatl(L), A * q))) / rho;
    }
    
    // 4-2. update 'z' with adaptation
    // see Boyd et al (2011) page 21
    z_old = z;
    z     = lasso_prox(x + u, lambda / rho);

    // 4-3. update 'u'
    u = u + x - z;

    // 4-3. dianostics, reporting
    h_objval(k) = lasso_objfun_fast(ATA_CHOL_U, ATb, b2, lambda, x, z, m, n);
    h_r_norm(k) = norm(x - z, 2);
    h_s_norm(k) = norm(-rho * (z - z_old), 2);
    if (norm(x, 2) > norm(-z, 2)) {
      h_eps_pri(k) = sqrtn * abstol + reltol * norm(x, 2);
    } else {
      h_eps_pri(k) = sqrtn * abstol + reltol * norm(-z, 2);
    }
    h_eps_dual(k) = sqrtn * abstol + reltol * norm(rho * u, 2);
        
    /* :::::::::::::::::::::::::::::::::::::::::::::::::::::::: */
    /* Relaxation                                               */
    rho_old = rho;
    if (rho_adaptation) {
      if (h_r_norm(k) > mu * h_s_norm(k)) {
        rho = rho * tau;
        u   = u / tau;
      } else if (h_s_norm(k) > mu * h_r_norm(k)) {
        rho = rho / tau;
        u   = u * tau;
      }
    }
    rho_store(k+1) = rho;
        
    /* :::::::::::::::::::::::::::::::::::::::::::::::::::::::: */
    /* Print to screen                                          */
    if (ping > 0) {
      if ((ping > 0) && (k < maxiter)) {
        if (rho_adaptation) {
          if (((k+1) % ping) == 0) {
            Rcpp::Rcout << "\n\n\n" << std::endl;
            Rcpp::Rcout << "ADMM with adaptation for LASSO is running!" << std::endl;
            Rcpp::Rcout << "Iteration n.: " << k+1 << " of " << maxiter << "\n" << std::endl;
          }
        } else {
          if (((k+1) % ping) == 0) {
            Rcpp::Rcout << "\n\n\n" << std::endl;
            Rcpp::Rcout << "ADMM for LASSO is running!" << std::endl;
            Rcpp::Rcout << "Iteration n.: " << k+1 << " of " << maxiter << "\n" << std::endl;
          }
        }
      }
    }
    
    // 4-4. termination
    if ((h_r_norm(k) < h_eps_pri(k)) && (h_s_norm(k) < h_eps_dual(k))) {
      break;
    }
  }

  /* ::::::::::::::::::::::::::::::::::::::::::::::
   Compute elapsed time                                */
  elTime = timer.toc();
  
  /* ::::::::::::::::::::::::::::::::::::::::::::::
   Compute post-estimation quantities                   */
  fitted    = A * x;
  residuals = b - fitted;
  mse       = as_scalar(residuals.t() * residuals) / (static_cast<float>(m));
  
  /* Get output         */
  List output;
  output["x"]      = x;               // coefficient function
  output["u"]      = u;
  output["z"]      = z;
  output["eltime"] = elTime;
  if (k < maxiter) {
    output["convergence"] = 1.0;
    output["niter"]       = k+1;               // number of iterations
    output["objval"]      = h_objval.subvec(0, k);        // |x|_1
    output["r_norm"]      = h_r_norm.subvec(0, k);
    output["s_norm"]      = h_s_norm.subvec(0, k);
    output["eps_pri"]     = h_eps_pri.subvec(0, k);
    output["eps_dual"]    = h_eps_dual.subvec(0, k);
    output["rho"]         = rho_store.subvec(0, k+1);
  } else {
    output["convergence"] = 0.0;
    output["niter"]       = maxiter;               // number of iterations
    output["objval"]      = h_objval;        // |x|_1
    output["r_norm"]      = h_r_norm;
    output["s_norm"]      = h_s_norm;
    output["eps_pri"]     = h_eps_pri;
    output["eps_dual"]    = h_eps_dual;
    output["rho"]         = rho_store;
  }
  output["residuals"]     = residuals;
  output["fitted.values"] = fitted;
  output["mse"]           = mse;
  output["rho.updated"]   = rho;
  
  /* Return output      */
  return(output);
}

// =========================================================
// admm_glasso
// ADMM for linear regression models with group-LASSO penalty

/* Group Lasso vector (or block) soft thresholding operator        */
arma::colvec glasso_prox(arma::colvec a, const double kappa) {

  const int n = a.n_elem;
  double x;
  arma::colvec y(n, fill::zeros);

  x = 1.0 - kappa / norm(a);
  if (x > 0.0) {
    y = a * x;
  }
  return(y);
}

// Linear regression model with GLASSO penalty: evaluate the objective function
double glasso_objfun(arma::mat A, arma::colvec b, arma::vec Glen,
                     const double lambda, arma::colvec x, arma::colvec z, const int G) {
  const int m = A.n_rows;
  double pen = 0.0, feval = 0.0;
  uword g_id_init, g_id_end;
  
  pen       = 0.0;
  g_id_init = 0;
  for (int g=0; g<G; g++) {
    g_id_end      = g_id_init + Glen(g) - 1;
    pen           = pen + norm(z.subvec(g_id_init, g_id_end), 2);
    g_id_init     = g_id_end + 1;
  }
  feval = 0.5 * norm(A * x - b, 2) / static_cast<float>(m) + lambda * pen;
  
  return(feval);
}

double glasso_objfun_fast(arma::mat ATA_CHOL_U, arma::colvec ATb, const double b2,
                          arma::vec Glen, const double lambda, arma::colvec x, arma::colvec z, const int m, const int n, const int G) {
  
  /* Variable delcaration           */
  int n1;
  double ll = 0.0, objfun = 0.0;
  double pen = 0.0;
  uword g_id_init, g_id_end;
  
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
  
  // evaluate the penalty function
  pen       = 0.0;
  g_id_init = 0;
  for (int g=0; g<G; g++) {
    g_id_end  = g_id_init + Glen(g) - 1;
    pen       = pen + norm(z.subvec(g_id_init, g_id_end), 2);
    g_id_init = g_id_end + 1;
  }
  
  // evaluate the objective function
  objfun = 0.5 * ll + lambda * pen;
  
  /* Return output      */
  return(objfun);
}

// Linear regression model with sparse GLASSO penalty: evaluate the objective function
double spglasso_objfun(arma::mat A, arma::colvec b, arma::vec Glen,
                       const double lambda, const double alpha,
                       arma::colvec x, arma::colvec z, const int G) {
  const int m = A.n_rows;
  const int n = A.n_cols;
  double pen = 0.0, feval = 0.0;
  uword g_id_init, g_id_end;
  
  pen       = 0.0;
  g_id_init = 0;
  for (int g=0; g<G; g++) {
    g_id_end  = g_id_init + Glen(g) - 1;
    pen       = pen + norm(z.subvec(g_id_init, g_id_end), 2);
    g_id_init = g_id_end + 1;
  }
  pen      = lambda * (1.0 - alpha) * pen;
  g_id_end = g_id_init + n - 1;
  pen      = pen + lambda * alpha * norm(z.subvec(g_id_init, g_id_end), 1);
  feval    = 0.5 * norm(A * x - b, 2) / static_cast<float>(m) + pen;
  
  return(feval);
}

double spglasso_objfun_fast(arma::mat ATA_CHOL_U, arma::colvec ATb, const double b2,
                            arma::vec Glen, const double lambda, const double alpha,
                            arma::colvec x, arma::colvec z, const int m, const int n, const int G) {
  
  /* Variable delcaration           */
  int n1;
  double ll = 0.0, objfun = 0.0;
  double pen = 0.0;
  uword g_id_init, g_id_end;
  
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
  
  // evaluate the penalty function
  pen       = 0.0;
  g_id_init = 0;
  for (int g=0; g<G; g++) {
    g_id_end  = g_id_init + Glen(g) - 1;
    pen       = pen + norm(z.subvec(g_id_init, g_id_end), 2);
    g_id_init = g_id_end + 1;
  }
  pen      = lambda * (1.0 - alpha) * pen;
  g_id_end = g_id_init + n - 1;
  pen      = pen + lambda * alpha * norm(z.subvec(g_id_init, g_id_end), 1);
  
  // evaluate the objective function
  objfun = 0.5 * ll + lambda * pen;
  
  /* Return output      */
  return(objfun);
}

// Evaluate the RIDGE matrix: returns the Cholesky (Upper triangular matrix)
arma::mat glasso_factor(arma::mat A, arma::mat FTF, double rho) {

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
arma::mat glasso_factor_fast_large_m(arma::mat ATA, arma::mat FTF, double rho) {

  /* Variable delcaration           */
  const int n = ATA.n_cols;
  arma::mat U(n, n, fill::zeros);
  
  /* Compute the Cholesky factor of the RIDGE matrix        */
  U = chol(ATA + rho * diagmat(FTF));
  
  /* return output         */
  return(U);
}

arma::mat glasso_factor_fast_large_n(arma::mat X, double rho, const int m) {
  
  /* Definition of vectors and matrices     */
  arma::mat U(m, m, fill::zeros);
  arma::vec eyeM(m, fill::ones);

  /* Compute the Cholesky factor of the RIDGE matrix        */
  U = chol(diagmat(eyeM * rho * static_cast<float>(m)) + X);;

  /* return output         */
  return(U);
}

arma::vec glasso_residual(arma::mat F, arma::colvec x, arma::colvec z) {
  return(F * x - z);
}

arma::vec glasso_dual_residual(arma::mat F, arma::colvec z, arma::colvec z_old, double rho) {
  return(rho * F.t() * (z - z_old));
}

arma::vec glasso_residual_sparse(arma::sp_mat F, arma::colvec x, arma::colvec z) {
  return(F * x - z);
}

arma::vec glasso_dual_residual_sparse(arma::sp_mat F, arma::colvec z, arma::colvec z_old, double rho) {
  return(rho * F.t() * (z - z_old));
}

arma::mat glasso_Gvec2F1mat(arma::rowvec Gvec) {
    
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

arma::sp_mat glasso_Gvec2F1mat_spmat(arma::rowvec Gvec) {
    
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

arma::mat glasso_Gmat2Fmat(arma::mat Gmat,
                           arma::vec group_weights,
                           arma::vec var_weights) {
    
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

arma::sp_mat glasso_Gmat2Fmat_sparse(arma::mat Gmat,
                                     arma::vec group_weights,
                                     arma::vec var_weights) {
    
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
    arma::sp_mat F1mat = glasso_Gvec2F1mat_spmat(Gmat.row(g));
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

arma::mat spglasso_Gmat2Fmat(arma::mat Gmat,
                             arma::vec group_weights,
                             arma::vec var_weights,
                             arma::vec var_weights_L1) {
    
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

arma::sp_mat spglasso_Gmat2Fmat_sparse(arma::mat Gmat,
                                       arma::vec group_weights,
                                       arma::vec var_weights,
                                       arma::vec var_weights_L1) {
    
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
    arma::sp_mat F1mat = glasso_Gvec2F1mat_spmat(Gmat.row(g));
    for (int r=0; r<(int)F1mat.n_rows; r++) {
      F1mat.row(r) *= group_weights(g);
    }
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

arma::mat glasso_Gmat2Dmat(arma::mat Gmat) {
    
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

/*
 Group-LASSO via ADMM, just to compute the whole path fast.
*/
Rcpp::List admm_glasso_large_m_update_1lambda(const arma::mat& A, const arma::vec& b,
                                              const arma::colvec Glen, const arma::sp_mat F, const arma::mat FTF,
                                              const arma::mat ATA, const arma::mat& ATA_CHOL_U, const arma::colvec& ATb,
                                              const double b2, const int m, const int n, const int G,
                                              arma::colvec u, arma::colvec z, const double lambda,
                                              bool rho_adaptation, double rho, const double tau, const double mu,
                                              const double reltol, const double abstol, const int maxiter, const int ping) {
  
  /* Variable delcaration           */
  int k, dim_z;
  double rho_old, elTime = 0.0, mse;
  uword g_id_init, g_id_end;
  
  /* Get dimensions                 */
  dim_z = z.n_elem;
  
  /* Definition of vectors and matrices     */
  arma::colvec x(n, fill::zeros);
  arma::colvec q(n, fill::zeros);
  arma::colvec z_old(dim_z, fill::zeros);
  arma::mat U(n, n, fill::zeros);
  arma::mat L(n, n, fill::zeros);
  arma::vec h_objval(maxiter, fill::zeros);
  arma::vec h_r_norm(maxiter, fill::zeros);
  arma::vec h_s_norm(maxiter, fill::zeros);
  arma::vec h_eps_pri(maxiter, fill::zeros);
  arma::vec h_eps_dual(maxiter, fill::zeros);
  arma::vec rho_store(maxiter+1, fill::zeros);
  arma::colvec fitted(m, fill::zeros);
  arma::colvec residuals(m, fill::zeros);

  /* ::::::::::::::::::::::::::::::::::::::::::::::
   Set the computational time                                 */
  wall_clock timer;
  timer.tic();
  
  /* store rho (only for adaptation)         */
  rho_store(0) = rho;
  rho_old      = rho;

  /* Precompute relevant quantities         */
  U = glasso_factor_fast_large_m(ATA, FTF, rho); // returns upper
  L = U.t();
  
  /* main loop  */
  for (k=0; k<maxiter; k++) {

    // 4-1. update 'x'
    if (rho != rho_old) {
      U = glasso_factor_fast_large_m(ATA, FTF, rho); // returns upper
      L = U.t();
    }
    q = ATb + rho * spmat2vec_mult(F.t(), z - u); // temporary value
    x = solve(trimatu(U), solve(trimatl(L), q));
    
    // 4-2. update 'z' with adaptation
    // see Boyd et al (2011) page 21
    z_old     = z;
    g_id_init = 0;
    for (int g=0; g<G; g++) {
      // precompute the quantities for updating z
      g_id_end         = g_id_init + Glen(g) - 1;
      arma::vec u_g    = u.subvec(g_id_init, g_id_end);
      arma::sp_mat F_g = F.submat(g_id_init, 0, g_id_end, n-1);

      // update z
      arma::vec z_g                 = glasso_prox(F_g * x + u_g, lambda / rho);
      z.subvec(g_id_init, g_id_end) = z_g;
      g_id_init                     = g_id_end + 1;
    }

    // 4-3. update 'u'
    u += spmat2vec_mult(F, x) - z;
  
    // 4-3. dianostics, reporting
    h_objval(k) = glasso_objfun_fast(ATA_CHOL_U, ATb, b2, Glen, lambda, x, z, m, n, G);
    h_r_norm(k) = norm(glasso_residual_sparse(F, x, z), 2);
    h_s_norm(k) = norm(glasso_dual_residual_sparse(F, z, z_old, rho), 2);
    if (norm(F * x, 2) > norm(-z, 2)) {
      h_eps_pri(k) = std::sqrt(z.n_elem) * abstol + reltol * norm(F * x, 2);
    } else {
      h_eps_pri(k) = std::sqrt(z.n_elem) * abstol + reltol * norm(-z, 2);
    }
    h_eps_dual(k) = std::sqrt(z.n_elem) * abstol + reltol * norm(F.t() * rho * u, 2);
    
    /* :::::::::::::::::::::::::::::::::::::::::::::::::::::::: */
    /* Relaxation                                               */
    rho_old = rho;
    if (rho_adaptation) {
      if (h_r_norm(k) > mu * h_s_norm(k)) {
        rho = rho * tau;
        u   = u / tau;
      } else if (h_s_norm(k) > mu * h_r_norm(k)) {
        rho = rho / tau;
        u   = u * tau;
      }
    }
    rho_store(k+1) = rho;
        
    /* :::::::::::::::::::::::::::::::::::::::::::::::::::::::: */
    /* Print to screen                                          */
    if ((ping > 0) && (k < maxiter)) {
      if (rho_adaptation) {
        if (((k+1) % ping) == 0) {
          Rcpp::Rcout << "\n\n\n" << std::endl;
          Rcpp::Rcout << "ADMM with adaptation for group-LASSO is running!" << std::endl;
          Rcpp::Rcout << "Iteration n.: " << k+1 << " of " << maxiter << "\n" << std::endl;
        }
      } else {
        if (((k+1) % ping) == 0) {
          Rcpp::Rcout << "\n\n\n" << std::endl;
          Rcpp::Rcout << "ADMM for group-LASSO is running!" << std::endl;
          Rcpp::Rcout << "Iteration n.: " << k+1 << " of " << maxiter << "\n" << std::endl;
        }
      }
    }

    // 4-4. termination
    if ((h_r_norm(k) < h_eps_pri(k)) && (h_s_norm(k) < h_eps_dual(k))) {
      break;
    }
  }

  /* ::::::::::::::::::::::::::::::::::::::::::::::
   Compute elapsed time                                */
  elTime = timer.toc();
  
  /* ::::::::::::::::::::::::::::::::::::::::::::::
   Compute post-estimation quantities                   */
  fitted    = A * x;
  residuals = b - fitted;
  mse       = as_scalar(residuals.t() * residuals) / (static_cast<float>(m));
  
  /* Get output         */
  List output;
  output["x"]      = x;               // coefficient function
  output["u"]      = u;
  output["z"]      = z;
  output["eltime"] = elTime;
  if (k < maxiter) {
    output["convergence"] = 1.0;
    output["niter"]       = k+1;               // number of iterations
    output["objval"]      = h_objval.subvec(0, k);        // |x|_1
    output["r_norm"]      = h_r_norm.subvec(0, k);
    output["s_norm"]      = h_s_norm.subvec(0, k);
    output["eps_pri"]     = h_eps_pri.subvec(0, k);
    output["eps_dual"]    = h_eps_dual.subvec(0, k);
    output["rho"]         = rho_store.subvec(0, k+1);
  } else {
    output["convergence"] = 0.0;
    output["niter"]       = maxiter;               // number of iterations
    output["objval"]      = h_objval;        // |x|_1
    output["r_norm"]      = h_r_norm;
    output["s_norm"]      = h_s_norm;
    output["eps_pri"]     = h_eps_pri;
    output["eps_dual"]    = h_eps_dual;
    output["rho"]         = rho_store;
  }
  output["residuals"]     = residuals;
  output["fitted.values"] = fitted;
  output["mse"]           = mse;
  output["rho.updated"]   = rho;
  
  /* Return output      */
  return(output);
}

Rcpp::List admm_glasso_large_n_update_1lambda(const arma::mat& A, const arma::vec& b, const arma::colvec Glen, const arma::sp_mat F,
                                              const arma::mat FTF_INV, const arma::mat AFTF_INVA,
                                              const arma::mat& ATA_CHOL_U, const arma::colvec& ATb,
                                              const double b2, const int m, const int n, const int G,
                                              arma::colvec u, arma::colvec z, const double lambda,
                                              bool rho_adaptation, double rho, const double tau, const double mu,
                                              const double reltol, const double abstol, const int maxiter, const int ping) {
  
  /* Variable delcaration           */
  int k, dim_z;
  double rho_old, elTime = 0.0, mse;
  uword g_id_init, g_id_end;
  
  /* Get dimensions                 */
  dim_z = z.n_elem;
  
  /* Definition of vectors and matrices     */
  arma::colvec x(n, fill::zeros);
  arma::colvec x_star(n, fill::zeros);
  arma::colvec q(n, fill::zeros);
  arma::colvec z_old(dim_z, fill::zeros);
  arma::mat U(m, m, fill::zeros);
  arma::mat L(m, m, fill::zeros);
  arma::vec h_objval(maxiter, fill::zeros);
  arma::vec h_r_norm(maxiter, fill::zeros);
  arma::vec h_s_norm(maxiter, fill::zeros);
  arma::vec h_eps_pri(maxiter, fill::zeros);
  arma::vec h_eps_dual(maxiter, fill::zeros);
  arma::vec rho_store(maxiter+1, fill::zeros);
  arma::colvec fitted(m, fill::zeros);
  arma::colvec residuals(m, fill::zeros);

  /* ::::::::::::::::::::::::::::::::::::::::::::::
   Set the computational time                                 */
  wall_clock timer;
  timer.tic();
  
  /* store rho (only for adaptation)         */
  rho_store(0) = rho;
  rho_old      = rho;

  /* Precompute relevant quantities         */
  U = glasso_factor_fast_large_n(AFTF_INVA, rho, m); // returns upper
  L = U.t();
  
  /* main loop  */
  for (k=0; k<maxiter; k++) {

    // 4-1. update 'x'
    if (rho != rho_old) {
      U = glasso_factor_fast_large_n(AFTF_INVA, rho, m); // returns upper
      L = U.t();
    }
    //q      = ATb + rho * F.t() * (z - u); // temporary value
    q      = ATb + rho * spmat2vec_mult(F.t(), z - u); // temporary value
    
    x_star = (diagmat(FTF_INV) * q) / rho;
    x      = x_star - diagmat(FTF_INV) * A.t() * solve(trimatu(U), solve(trimatl(L), A * x_star));
    
    // 4-2. update 'z' with adaptation
    // see Boyd et al (2011) page 21
    z_old     = z;
    g_id_init = 0;
    for (int g=0; g<G; g++) {
      // precompute the quantities for updating z
      g_id_end         = g_id_init + Glen(g) - 1;
      arma::vec u_g    = u.subvec(g_id_init, g_id_end);
      arma::sp_mat F_g = F.submat(g_id_init, 0, g_id_end, n-1);

      // update z
      arma::vec z_g                 = glasso_prox(F_g * x + u_g, lambda / rho);
      z.subvec(g_id_init, g_id_end) = z_g;
      g_id_init                     = g_id_end + 1;
    }

    // 4-3. update 'u'
    //u = u + F * x - z;
    u += spmat2vec_mult(F, x) - z;
  
    // 4-3. dianostics, reporting
    h_objval(k) = glasso_objfun_fast(ATA_CHOL_U, ATb, b2, Glen, lambda, x, z, m, n, G);
    h_r_norm(k) = norm(glasso_residual_sparse(F, x, z), 2);
    h_s_norm(k) = norm(glasso_dual_residual_sparse(F, z, z_old, rho), 2);
    if (norm(F * x, 2) > norm(-z, 2)) {
      h_eps_pri(k) = std::sqrt(z.n_elem) * abstol + reltol * norm(F * x, 2);
    } else {
      h_eps_pri(k) = std::sqrt(z.n_elem) * abstol + reltol * norm(-z, 2);
    }
    h_eps_dual(k) = std::sqrt(z.n_elem) * abstol + reltol * norm(F.t() * rho * u, 2);
    
    /* :::::::::::::::::::::::::::::::::::::::::::::::::::::::: */
    /* Relaxation                                               */
    rho_old = rho;
    if (rho_adaptation) {
      if (h_r_norm(k) > mu * h_s_norm(k)) {
        rho = rho * tau;
        u   = u / tau;
      } else if (h_s_norm(k) > mu * h_r_norm(k)) {
        rho = rho / tau;
        u   = u * tau;
      }
    }
    rho_store(k+1) = rho;
        
    /* :::::::::::::::::::::::::::::::::::::::::::::::::::::::: */
    /* Print to screen                                          */
    if ((ping > 0) && (k < maxiter)) {
      if (rho_adaptation) {
        if (((k+1) % ping) == 0) {
          Rcpp::Rcout << "\n\n\n" << std::endl;
          Rcpp::Rcout << "ADMM with adaptation for group-LASSO is running!" << std::endl;
          Rcpp::Rcout << "Iteration n.: " << k+1 << " of " << maxiter << "\n" << std::endl;
        }
      } else {
        if (((k+1) % ping) == 0) {
          Rcpp::Rcout << "\n\n\n" << std::endl;
          Rcpp::Rcout << "ADMM for group-LASSO is running!" << std::endl;
          Rcpp::Rcout << "Iteration n.: " << k+1 << " of " << maxiter << "\n" << std::endl;
        }
      }
    }

    // 4-4. termination
    if ((h_r_norm(k) < h_eps_pri(k)) && (h_s_norm(k) < h_eps_dual(k))) {
      break;
    }
  }

  /* ::::::::::::::::::::::::::::::::::::::::::::::
   Compute elapsed time                                */
  elTime = timer.toc();
  
  /* ::::::::::::::::::::::::::::::::::::::::::::::
   Compute post-estimation quantities                   */
  fitted    = A * x;
  residuals = b - fitted;
  mse       = as_scalar(residuals.t() * residuals) / (static_cast<float>(m));
  
  /* Get output         */
  List output;
  output["x"]      = x;               // coefficient function
  output["u"]      = u;
  output["z"]      = z;
  output["eltime"] = elTime;
  if (k < maxiter) {
    output["convergence"] = 1.0;
    output["niter"]       = k+1;               // number of iterations
    output["objval"]      = h_objval.subvec(0, k);        // |x|_1
    output["r_norm"]      = h_r_norm.subvec(0, k);
    output["s_norm"]      = h_s_norm.subvec(0, k);
    output["eps_pri"]     = h_eps_pri.subvec(0, k);
    output["eps_dual"]    = h_eps_dual.subvec(0, k);
    output["rho"]         = rho_store.subvec(0, k+1);
  } else {
    output["convergence"] = 0.0;
    output["niter"]       = maxiter;               // number of iterations
    output["objval"]      = h_objval;        // |x|_1
    output["r_norm"]      = h_r_norm;
    output["s_norm"]      = h_s_norm;
    output["eps_pri"]     = h_eps_pri;
    output["eps_dual"]    = h_eps_dual;
    output["rho"]         = rho_store;
  }
  output["residuals"]     = residuals;
  output["fitted.values"] = fitted;
  output["mse"]           = mse;
  output["rho.updated"]   = rho;
  
  /* Return output      */
  return(output);
}

/*
 Sparse Group-LASSO via ADMM, just to compute the whole path fast.
*/
Rcpp::List admm_spglasso_large_m_update_1lambda(const arma::mat& A, const arma::vec& b,
                                                const arma::colvec Glen, const arma::sp_mat F, const arma::mat FTF,
                                                const arma::mat ATA, const arma::mat& ATA_CHOL_U, const arma::colvec& ATb,
                                                const double b2, const int m, const int n, const int G,
                                                arma::colvec u, arma::colvec z, const double lambda, const double alpha,
                                                bool rho_adaptation, double rho, const double tau, const double mu,
                                                const double reltol, const double abstol, const int maxiter, const int ping) {
  
  /* Variable delcaration           */
  int k, dim_z;
  double rho_old, elTime = 0.0, mse;
  uword g_id_init, g_id_end;
  
  /* Get dimensions                 */
  dim_z = z.n_elem;
  
  /* Definition of vectors and matrices     */
  arma::colvec x(n, fill::zeros);
  arma::colvec q(n, fill::zeros);
  arma::colvec z_old(dim_z, fill::zeros);
  arma::mat U(n, n, fill::zeros);
  arma::mat L(n, n, fill::zeros);
  arma::vec h_objval(maxiter, fill::zeros);
  arma::vec h_r_norm(maxiter, fill::zeros);
  arma::vec h_s_norm(maxiter, fill::zeros);
  arma::vec h_eps_pri(maxiter, fill::zeros);
  arma::vec h_eps_dual(maxiter, fill::zeros);
  arma::vec rho_store(maxiter+1, fill::zeros);
  arma::colvec fitted(m, fill::zeros);
  arma::colvec residuals(m, fill::zeros);

  /* ::::::::::::::::::::::::::::::::::::::::::::::
   Set the computational time                                 */
  wall_clock timer;
  timer.tic();
  
  /* store rho (only for adaptation)         */
  rho_store(0) = rho;
  rho_old      = rho;

  /* Precompute relevant quantities         */
  U = glasso_factor_fast_large_m(ATA, FTF, rho); // returns upper
  L = U.t();
  
  /* main loop  */
  for (k=0; k<maxiter; k++) {

    // 4-1. update 'x'
    if (rho != rho_old) {
      U = glasso_factor_fast_large_m(ATA, FTF, rho); // returns upper
      L = U.t();
    }
    q = ATb + rho * spmat2vec_mult(F.t(), z - u); // temporary value
    x = solve(trimatu(U), solve(trimatl(L), q));
    
    // 4-2. update 'z' with adaptation
    // see Boyd et al (2011) page 21
    z_old     = z;
    g_id_init = 0;
    for (int g=0; g<G; g++) {
      // precompute the quantities for updating z
      g_id_end         = g_id_init + Glen(g) - 1;
      arma::vec u_g    = u.subvec(g_id_init, g_id_end);
      arma::sp_mat F_g = F.submat(g_id_init, 0, g_id_end, n-1);

      // update z
      arma::vec z_g                 = glasso_prox(F_g * x + u_g, lambda * (1.0 - alpha) / rho);
      z.subvec(g_id_init, g_id_end) = z_g;
      g_id_init                     = g_id_end + 1;
    }
    // update z for the last group
    g_id_end                      = g_id_init + n - 1;
    arma::vec u_g                 = u.subvec(g_id_init, g_id_end);
    arma::sp_mat F_g              = F.submat(g_id_init, 0, g_id_end, n-1);
    arma::vec z_g                 = lasso_prox(F_g * x + u_g, lambda * alpha / rho);
    z.subvec(g_id_init, g_id_end) = z_g;

    // 4-3. update 'u'
    u += spmat2vec_mult(F, x) - z;
      
    // 4-3. dianostics, reporting
    h_objval(k) = spglasso_objfun_fast(ATA_CHOL_U, ATb, b2, Glen, lambda, alpha, x, z, m, n, G);
    h_r_norm(k) = norm(glasso_residual_sparse(F, x, z), 2);
    h_s_norm(k) = norm(glasso_dual_residual_sparse(F, z, z_old, rho), 2);
    if (norm(F * x, 2) > norm(-z, 2)) {
      h_eps_pri(k) = std::sqrt(z.n_elem) * abstol + reltol * norm(F * x, 2);
    } else {
      h_eps_pri(k) = std::sqrt(z.n_elem) * abstol + reltol * norm(-z, 2);
    }
    h_eps_dual(k) = std::sqrt(z.n_elem) * abstol + reltol * norm(F.t() * rho * u, 2);
    
    /* :::::::::::::::::::::::::::::::::::::::::::::::::::::::: */
    /* Relaxation                                               */
    rho_old = rho;
    if (rho_adaptation) {
      if (h_r_norm(k) > mu * h_s_norm(k)) {
        rho = rho * tau;
        u   = u / tau;
      } else if (h_s_norm(k) > mu * h_r_norm(k)) {
        rho = rho / tau;
        u   = u * tau;
      }
    }
    rho_store(k+1) = rho;
        
    /* :::::::::::::::::::::::::::::::::::::::::::::::::::::::: */
    /* Print to screen                                          */
    if ((ping > 0) && (k < maxiter)) {
      if (rho_adaptation) {
        if (((k+1) % ping) == 0) {
          Rcpp::Rcout << "\n\n\n" << std::endl;
          Rcpp::Rcout << "ADMM with adaptation for sparse group-LASSO is running!" << std::endl;
          Rcpp::Rcout << "Iteration n.: " << k+1 << " of " << maxiter << "\n" << std::endl;
        }
      } else {
        if (((k+1) % ping) == 0) {
          Rcpp::Rcout << "\n\n\n" << std::endl;
          Rcpp::Rcout << "ADMM for sparse group-LASSO is running!" << std::endl;
          Rcpp::Rcout << "Iteration n.: " << k+1 << " of " << maxiter << "\n" << std::endl;
        }
      }
    }

    // 4-4. termination
    if ((h_r_norm(k) < h_eps_pri(k)) && (h_s_norm(k) < h_eps_dual(k))) {
      break;
    }
  }

  /* ::::::::::::::::::::::::::::::::::::::::::::::
   Compute elapsed time                                */
  elTime = timer.toc();
  
  /* ::::::::::::::::::::::::::::::::::::::::::::::
   Compute post-estimation quantities                   */
  fitted    = A * x;
  residuals = b - fitted;
  mse       = as_scalar(residuals.t() * residuals) / (static_cast<float>(m));
  
  /* Get output         */
  List output;
  output["x"]      = x;               // coefficient function
  output["u"]      = u;
  output["z"]      = z;
  output["eltime"] = elTime;
  if (k < maxiter) {
    output["convergence"] = 1.0;
    output["niter"]       = k+1;               // number of iterations
    output["objval"]      = h_objval.subvec(0, k);        // |x|_1
    output["r_norm"]      = h_r_norm.subvec(0, k);
    output["s_norm"]      = h_s_norm.subvec(0, k);
    output["eps_pri"]     = h_eps_pri.subvec(0, k);
    output["eps_dual"]    = h_eps_dual.subvec(0, k);
    output["rho"]         = rho_store.subvec(0, k+1);
  } else {
    output["convergence"] = 0.0;
    output["niter"]       = maxiter;               // number of iterations
    output["objval"]      = h_objval;        // |x|_1
    output["r_norm"]      = h_r_norm;
    output["s_norm"]      = h_s_norm;
    output["eps_pri"]     = h_eps_pri;
    output["eps_dual"]    = h_eps_dual;
    output["rho"]         = rho_store;
  }
  output["residuals"]     = residuals;
  output["fitted.values"] = fitted;
  output["mse"]           = mse;
  output["rho.updated"]   = rho;
  
  /* Return output      */
  return(output);
}

Rcpp::List admm_spglasso_large_n_update_1lambda(const arma::mat& A, arma::vec& b, const arma::colvec Glen, const arma::sp_mat F,
                                                const arma::mat FTF_INV, const arma::mat AFTF_INVA,
                                                const arma::mat& ATA_CHOL_U, const arma::colvec& ATb,
                                                const double b2, const int m, const int n, const int G,
                                                arma::colvec u, arma::colvec z, const double lambda, const double alpha,
                                                bool rho_adaptation, double rho, const double tau, const double mu,
                                                const double reltol, const double abstol, const int maxiter, const int ping) {
  
  /* Variable delcaration           */
  int k, dim_z;
  double rho_old, elTime = 0.0, mse;
  uword g_id_init, g_id_end;
  
  /* Get dimensions                 */
  dim_z = z.n_elem;
  
  /* Definition of vectors and matrices     */
  arma::colvec x(n, fill::zeros);
  arma::colvec x_star(n, fill::zeros);
  arma::colvec q(n, fill::zeros);
  arma::colvec z_old(dim_z, fill::zeros);
  arma::mat U(m, m, fill::zeros);
  arma::mat L(m, m, fill::zeros);
  arma::vec h_objval(maxiter, fill::zeros);
  arma::vec h_r_norm(maxiter, fill::zeros);
  arma::vec h_s_norm(maxiter, fill::zeros);
  arma::vec h_eps_pri(maxiter, fill::zeros);
  arma::vec h_eps_dual(maxiter, fill::zeros);
  arma::vec rho_store(maxiter+1, fill::zeros);
  arma::colvec fitted(m, fill::zeros);
  arma::colvec residuals(m, fill::zeros);

  /* ::::::::::::::::::::::::::::::::::::::::::::::
   Set the computational time                                 */
  wall_clock timer;
  timer.tic();
  
  /* store rho (only for adaptation)         */
  rho_store(0) = rho;
  rho_old      = rho;

  /* Precompute relevant quantities         */
  U = glasso_factor_fast_large_n(AFTF_INVA, rho, m); // returns upper
  L = U.t();
  
  /* main loop  */
  for (k=0; k<maxiter; k++) {

    // 4-1. update 'x'
    if (rho != rho_old) {
      U = glasso_factor_fast_large_n(AFTF_INVA, rho, m); // returns upper
      L = U.t();
    }
    //q      = ATb + rho * F.t() * (z - u); // temporary value
    q      = ATb + rho * spmat2vec_mult(F.t(), z - u);
    x_star = (diagmat(FTF_INV) * q) / rho;
    x      = x_star - diagmat(FTF_INV) * A.t() * solve(trimatu(U), solve(trimatl(L), A * x_star));
    
    // 4-2. update 'z' with adaptation
    // see Boyd et al (2011) page 21
    z_old     = z;
    g_id_init = 0;
    for (int g=0; g<G; g++) {
      // precompute the quantities for updating z
      g_id_end         = g_id_init + Glen(g) - 1;
      arma::vec u_g    = u.subvec(g_id_init, g_id_end);
      arma::sp_mat F_g = F.submat(g_id_init, 0, g_id_end, n-1);

      // update z
      arma::vec z_g                 = glasso_prox(F_g * x + u_g, lambda * (1.0 - alpha) / rho);
      z.subvec(g_id_init, g_id_end) = z_g;
      g_id_init                     = g_id_end + 1;
    }
    // update z for the last group
    g_id_end                      = g_id_init + n - 1;
    arma::vec u_g                 = u.subvec(g_id_init, g_id_end);
    arma::sp_mat F_g              = F.submat(g_id_init, 0, g_id_end, n-1);
    arma::vec z_g                 = lasso_prox(F_g * x + u_g, lambda * alpha / rho);
    z.subvec(g_id_init, g_id_end) = z_g;

    // 4-3. update 'u'
    //u = u + F * x - z;
    u += spmat2vec_mult(F, x) - z;
  
    // 4-3. dianostics, reporting
    h_objval(k) = spglasso_objfun_fast(ATA_CHOL_U, ATb, b2, Glen, lambda, alpha, x, z, m, n, G);
    h_r_norm(k) = norm(glasso_residual_sparse(F, x, z), 2);
    h_s_norm(k) = norm(glasso_dual_residual_sparse(F, z, z_old, rho), 2);
    if (norm(F * x, 2) > norm(-z, 2)) {
      h_eps_pri(k) = std::sqrt(z.n_elem) * abstol + reltol * norm(F * x, 2);
    } else {
      h_eps_pri(k) = std::sqrt(z.n_elem) * abstol + reltol * norm(-z, 2);
    }
    h_eps_dual(k) = std::sqrt(z.n_elem) * abstol + reltol * norm(F.t() * rho * u, 2);
    
    /* :::::::::::::::::::::::::::::::::::::::::::::::::::::::: */
    /* Relaxation                                               */
    rho_old = rho;
    if (rho_adaptation) {
      if (h_r_norm(k) > mu * h_s_norm(k)) {
        rho = rho * tau;
        u   = u / tau;
      } else if (h_s_norm(k) > mu * h_r_norm(k)) {
        rho = rho / tau;
        u   = u * tau;
      }
    }
    rho_store(k+1) = rho;
        
    /* :::::::::::::::::::::::::::::::::::::::::::::::::::::::: */
    /* Print to screen                                          */
    if ((ping > 0) && (k < maxiter)) {
      if (rho_adaptation) {
        if (((k+1) % ping) == 0) {
          Rcpp::Rcout << "\n\n\n" << std::endl;
          Rcpp::Rcout << "ADMM with adaptation for group-LASSO is running!" << std::endl;
          Rcpp::Rcout << "Iteration n.: " << k+1 << " of " << maxiter << "\n" << std::endl;
        }
      } else {
        if (((k+1) % ping) == 0) {
          Rcpp::Rcout << "\n\n\n" << std::endl;
          Rcpp::Rcout << "ADMM for group-LASSO is running!" << std::endl;
          Rcpp::Rcout << "Iteration n.: " << k+1 << " of " << maxiter << "\n" << std::endl;
        }
      }
    }

    // 4-4. termination
    if ((h_r_norm(k) < h_eps_pri(k)) && (h_s_norm(k) < h_eps_dual(k))) {
      break;
    }
  }

  /* ::::::::::::::::::::::::::::::::::::::::::::::
   Compute elapsed time                                */
  elTime = timer.toc();
  
  /* ::::::::::::::::::::::::::::::::::::::::::::::
   Compute post-estimation quantities                   */
  fitted    = A * x;
  residuals = b - fitted;
  mse       = as_scalar(residuals.t() * residuals) / (static_cast<float>(m));
  
  /* Get output         */
  List output;
  output["x"]      = x;               // coefficient function
  output["u"]      = u;
  output["z"]      = z;
  output["eltime"] = elTime;
  if (k < maxiter) {
    output["convergence"] = 1.0;
    output["niter"]       = k+1;               // number of iterations
    output["objval"]      = h_objval.subvec(0, k);        // |x|_1
    output["r_norm"]      = h_r_norm.subvec(0, k);
    output["s_norm"]      = h_s_norm.subvec(0, k);
    output["eps_pri"]     = h_eps_pri.subvec(0, k);
    output["eps_dual"]    = h_eps_dual.subvec(0, k);
    output["rho"]         = rho_store.subvec(0, k+1);
  } else {
    output["convergence"] = 0.0;
    output["niter"]       = maxiter;               // number of iterations
    output["objval"]      = h_objval;        // |x|_1
    output["r_norm"]      = h_r_norm;
    output["s_norm"]      = h_s_norm;
    output["eps_pri"]     = h_eps_pri;
    output["eps_dual"]    = h_eps_dual;
    output["rho"]         = rho_store;
  }
  output["residuals"]     = residuals;
  output["fitted.values"] = fitted;
  output["mse"]           = mse;
  output["rho.updated"]   = rho;
  
  /* Return output      */
  return(output);
}

/*
 Overlap Group-LASSO via ADMM, just to compute the whole path fast.
*/
Rcpp::List admm_ovglasso_large_m_update_1lambda(const arma::mat& A, const arma::vec& b,
                                                const arma::colvec Glen, const arma::sp_mat F, const arma::mat FTF,
                                                const arma::mat ATA, const arma::mat& ATA_CHOL_U, const arma::colvec& ATb,
                                                const double b2, const int m, const int n, const int G,
                                                arma::colvec u, arma::colvec z, const double lambda,
                                                bool rho_adaptation, double rho, const double tau, const double mu,
                                                const double reltol, const double abstol, const int maxiter, const int ping) {
  
  /* Variable delcaration           */
  int k, dim_z;
  double rho_old, elTime = 0.0, mse;
  uword g_id_init, g_id_end;
  
  /* Get dimensions                 */
  dim_z = z.n_elem;
  
  /* Definition of vectors and matrices     */
  arma::colvec x(n, fill::zeros);
  arma::colvec q(n, fill::zeros);
  arma::colvec z_old(dim_z, fill::zeros);
  arma::mat U(n, n, fill::zeros);
  arma::mat L(n, n, fill::zeros);
  arma::vec h_objval(maxiter, fill::zeros);
  arma::vec h_r_norm(maxiter, fill::zeros);
  arma::vec h_s_norm(maxiter, fill::zeros);
  arma::vec h_eps_pri(maxiter, fill::zeros);
  arma::vec h_eps_dual(maxiter, fill::zeros);
  arma::vec rho_store(maxiter+1, fill::zeros);
  arma::colvec fitted(m, fill::zeros);
  arma::colvec residuals(m, fill::zeros);

  /* ::::::::::::::::::::::::::::::::::::::::::::::
   Set the computational time                                 */
  wall_clock timer;
  timer.tic();
  
  /* store rho (only for adaptation)         */
  rho_store(0) = rho;
  rho_old      = rho;

  /* Precompute relevant quantities         */
  U = glasso_factor_fast_large_m(ATA, FTF, rho); // returns upper
  L = U.t();
  
  /* main loop  */
  for (k=0; k<maxiter; k++) {

    // 4-1. update 'x'
    if (rho != rho_old) {
      U = glasso_factor_fast_large_m(ATA, FTF, rho); // returns upper
      L = U.t();
    }
    //q = ATb + rho * F.t() * (z - u); // temporary value
    q = ATb + rho * spmat2vec_mult(F.t(), z - u);
    x = solve(trimatu(U), solve(trimatl(L), q));
    
    // 4-2. update 'z' with adaptation
    // see Boyd et al (2011) page 21
    z_old     = z;
    g_id_init = 0;
    for (int g=0; g<G; g++) {
      // precompute the quantities for updating z
      g_id_end         = g_id_init + Glen(g) - 1;
      arma::vec u_g    = u.subvec(g_id_init, g_id_end);
      arma::sp_mat F_g = F.submat(g_id_init, 0, g_id_end, n-1);
      
      // update z
      arma::vec z_g                 = glasso_prox(F_g * x + u_g, lambda / rho);
      z.subvec(g_id_init, g_id_end) = z_g;
      g_id_init                     = g_id_end + 1;
    }

    // 4-3. update 'u'
    u += spmat2vec_mult(F, x) - z;
  
    // 4-3. dianostics, reporting
    h_objval(k) = glasso_objfun_fast(ATA_CHOL_U, ATb, b2, Glen, lambda, x, z, m, n, G);
    h_r_norm(k) = norm(glasso_residual_sparse(F, x, z), 2);
    h_s_norm(k) = norm(glasso_dual_residual_sparse(F, z, z_old, rho), 2);
    if (norm(F * x, 2) > norm(-z, 2)) {
      h_eps_pri(k) = std::sqrt(z.n_elem) * abstol + reltol * norm(F * x, 2);
    } else {
      h_eps_pri(k) = std::sqrt(z.n_elem) * abstol + reltol * norm(-z, 2);
    }
    h_eps_dual(k) = std::sqrt(z.n_elem) * abstol + reltol * norm(F.t() * rho * u, 2);
    
    /* :::::::::::::::::::::::::::::::::::::::::::::::::::::::: */
    /* Relaxation                                               */
    rho_old = rho;
    if (rho_adaptation) {
      if (h_r_norm(k) > mu * h_s_norm(k)) {
        rho = rho * tau;
        u   = u / tau;
      } else if (h_s_norm(k) > mu * h_r_norm(k)) {
        rho = rho / tau;
        u   = u * tau;
      }
    }
    rho_store(k+1) = rho;
        
    /* :::::::::::::::::::::::::::::::::::::::::::::::::::::::: */
    /* Print to screen                                          */
    if ((ping > 0) && (k < maxiter)) {
      if (rho_adaptation) {
        if (((k+1) % ping) == 0) {
          Rcpp::Rcout << "\n\n\n" << std::endl;
          Rcpp::Rcout << "ADMM with adaptation for overlap group-LASSO is running!" << std::endl;
          Rcpp::Rcout << "Iteration n.: " << k+1 << " of " << maxiter << "\n" << std::endl;
        }
      } else {
        if (((k+1) % ping) == 0) {
          Rcpp::Rcout << "\n\n\n" << std::endl;
          Rcpp::Rcout << "ADMM for overlap group-LASSO is running!" << std::endl;
          Rcpp::Rcout << "Iteration n.: " << k+1 << " of " << maxiter << "\n" << std::endl;
        }
      }
    }

    // 4-4. termination
    if ((h_r_norm(k) < h_eps_pri(k)) && (h_s_norm(k) < h_eps_dual(k))) {
      break;
    }
  }

  /* ::::::::::::::::::::::::::::::::::::::::::::::
   Compute elapsed time                                */
  elTime = timer.toc();
  
  /* ::::::::::::::::::::::::::::::::::::::::::::::
   Compute post-estimation quantities                   */
  fitted    = A * x;
  residuals = b - fitted;
  mse       = as_scalar(residuals.t() * residuals) / (static_cast<float>(m));
  
  /* Get output         */
  List output;
  output["x"]      = x;               // coefficient function
  output["u"]      = u;
  output["z"]      = z;
  output["eltime"] = elTime;
  if (k < maxiter) {
    output["convergence"] = 1.0;
    output["niter"]       = k+1;               // number of iterations
    output["objval"]      = h_objval.subvec(0, k);        // |x|_1
    output["r_norm"]      = h_r_norm.subvec(0, k);
    output["s_norm"]      = h_s_norm.subvec(0, k);
    output["eps_pri"]     = h_eps_pri.subvec(0, k);
    output["eps_dual"]    = h_eps_dual.subvec(0, k);
    output["rho"]         = rho_store.subvec(0, k+1);
  } else {
    output["convergence"] = 0.0;
    output["niter"]       = maxiter;               // number of iterations
    output["objval"]      = h_objval;        // |x|_1
    output["r_norm"]      = h_r_norm;
    output["s_norm"]      = h_s_norm;
    output["eps_pri"]     = h_eps_pri;
    output["eps_dual"]    = h_eps_dual;
    output["rho"]         = rho_store;
  }
  output["residuals"]     = residuals;
  output["fitted.values"] = fitted;
  output["mse"]           = mse;
  output["rho.updated"]   = rho;
  
  /* Return output      */
  return(output);
}

Rcpp::List admm_ovglasso_large_n_update_1lambda(const arma::mat& A, const arma::vec& b, const arma::colvec Glen, const arma::sp_mat F,
                                                const arma::mat FTF_INV, const arma::mat AFTF_INVA,
                                                const arma::mat& ATA_CHOL_U, const arma::colvec& ATb,
                                                const double b2, const int m, const int n, const int G,
                                                arma::colvec u, arma::colvec z, const double lambda,
                                                bool rho_adaptation, double rho, const double tau, const double mu,
                                                const double reltol, const double abstol, const int maxiter, const int ping) {
  
  /* Variable delcaration           */
  int k, dim_z;
  double rho_old, elTime = 0.0, mse;
  uword g_id_init, g_id_end;
  
  /* Get dimensions                 */
  dim_z = z.n_elem;
  
  /* Definition of vectors and matrices     */
  arma::colvec x(n, fill::zeros);
  arma::colvec x_star(n, fill::zeros);
  arma::colvec q(n, fill::zeros);
  arma::colvec z_old(dim_z, fill::zeros);
  arma::mat U(m, m, fill::zeros);
  arma::mat L(m, m, fill::zeros);
  arma::vec h_objval(maxiter, fill::zeros);
  arma::vec h_r_norm(maxiter, fill::zeros);
  arma::vec h_s_norm(maxiter, fill::zeros);
  arma::vec h_eps_pri(maxiter, fill::zeros);
  arma::vec h_eps_dual(maxiter, fill::zeros);
  arma::vec rho_store(maxiter+1, fill::zeros);
  arma::colvec fitted(m, fill::zeros);
  arma::colvec residuals(m, fill::zeros);

  /* ::::::::::::::::::::::::::::::::::::::::::::::
   Set the computational time                                 */
  wall_clock timer;
  timer.tic();
  
  /* store rho (only for adaptation)         */
  rho_store(0) = rho;
  rho_old      = rho;

  /* Precompute relevant quantities         */
  U = glasso_factor_fast_large_n(AFTF_INVA, rho, m); // returns upper
  L = U.t();
  
  /* main loop  */
  for (k=0; k<maxiter; k++) {

    // 4-1. update 'x'
    if (rho != rho_old) {
      U = glasso_factor_fast_large_n(AFTF_INVA, rho, m); // returns upper
      L = U.t();
    }
    //q      = ATb + rho * F.t() * (z - u); // temporary value
    q      = ATb + rho * spmat2vec_mult(F.t(), z - u);
    x_star = (diagmat(FTF_INV) * q) / rho;
    x      = x_star - diagmat(FTF_INV) * A.t() * solve(trimatu(U), solve(trimatl(L), A * x_star));
    
    // 4-2. update 'z' with adaptation
    // see Boyd et al (2011) page 21
    z_old     = z;
    g_id_init = 0;
    for (int g=0; g<G; g++) {
      // precompute the quantities for updating z
      g_id_end         = g_id_init + Glen(g) - 1;
      arma::vec u_g    = u.subvec(g_id_init, g_id_end);
      arma::sp_mat F_g = F.submat(g_id_init, 0, g_id_end, n-1);

      // update z
      arma::vec z_g                 = glasso_prox(F_g * x + u_g, lambda / rho);
      z.subvec(g_id_init, g_id_end) = z_g;
      g_id_init                     = g_id_end + 1;
    }

    // 4-3. update 'u'
    //u = u + F * x - z;
    u += spmat2vec_mult(F, x) - z;
  
    // 4-3. dianostics, reporting
    h_objval(k) = glasso_objfun_fast(ATA_CHOL_U, ATb, b2, Glen, lambda, x, z, m, n, G);
    h_r_norm(k) = norm(glasso_residual_sparse(F, x, z), 2);
    h_s_norm(k) = norm(glasso_dual_residual_sparse(F, z, z_old, rho), 2);
    if (norm(F * x, 2) > norm(-z, 2)) {
      h_eps_pri(k) = std::sqrt(z.n_elem) * abstol + reltol * norm(F * x, 2);
    } else {
      h_eps_pri(k) = std::sqrt(z.n_elem) * abstol + reltol * norm(-z, 2);
    }
    h_eps_dual(k) = std::sqrt(z.n_elem) * abstol + reltol * norm(F.t() * rho * u, 2);
    
    /* :::::::::::::::::::::::::::::::::::::::::::::::::::::::: */
    /* Relaxation                                               */
    rho_old = rho;
    if (rho_adaptation) {
      if (h_r_norm(k) > mu * h_s_norm(k)) {
        rho = rho * tau;
        u   = u / tau;
      } else if (h_s_norm(k) > mu * h_r_norm(k)) {
        rho = rho / tau;
        u   = u * tau;
      }
    }
    rho_store(k+1) = rho;
        
    /* :::::::::::::::::::::::::::::::::::::::::::::::::::::::: */
    /* Print to screen                                          */
    if ((ping > 0) && (k < maxiter)) {
      if (rho_adaptation) {
        if (((k+1) % ping) == 0) {
          Rcpp::Rcout << "\n\n\n" << std::endl;
          Rcpp::Rcout << "ADMM with adaptation for overlap group-LASSO is running!" << std::endl;
          Rcpp::Rcout << "Iteration n.: " << k+1 << " of " << maxiter << "\n" << std::endl;
        }
      } else {
        if (((k+1) % ping) == 0) {
          Rcpp::Rcout << "\n\n\n" << std::endl;
          Rcpp::Rcout << "ADMM for overlap group-LASSO is running!" << std::endl;
          Rcpp::Rcout << "Iteration n.: " << k+1 << " of " << maxiter << "\n" << std::endl;
        }
      }
    }

    // 4-4. termination
    if ((h_r_norm(k) < h_eps_pri(k)) && (h_s_norm(k) < h_eps_dual(k))) {
      break;
    }
  }

  /* ::::::::::::::::::::::::::::::::::::::::::::::
   Compute elapsed time                                */
  elTime = timer.toc();
  
  /* ::::::::::::::::::::::::::::::::::::::::::::::
   Compute post-estimation quantities                   */
  fitted    = A * x;
  residuals = b - fitted;
  mse       = as_scalar(residuals.t() * residuals) / (static_cast<float>(m));
  
  /* Get output         */
  List output;
  output["x"]      = x;               // coefficient function
  output["u"]      = u;
  output["z"]      = z;
  output["eltime"] = elTime;
  if (k < maxiter) {
    output["convergence"] = 1.0;
    output["niter"]       = k+1;               // number of iterations
    output["objval"]      = h_objval.subvec(0, k);        // |x|_1
    output["r_norm"]      = h_r_norm.subvec(0, k);
    output["s_norm"]      = h_s_norm.subvec(0, k);
    output["eps_pri"]     = h_eps_pri.subvec(0, k);
    output["eps_dual"]    = h_eps_dual.subvec(0, k);
    output["rho"]         = rho_store.subvec(0, k+1);
  } else {
    output["convergence"] = 0.0;
    output["niter"]       = maxiter;               // number of iterations
    output["objval"]      = h_objval;        // |x|_1
    output["r_norm"]      = h_r_norm;
    output["s_norm"]      = h_s_norm;
    output["eps_pri"]     = h_eps_pri;
    output["eps_dual"]    = h_eps_dual;
    output["rho"]         = rho_store;
  }
  output["residuals"]     = residuals;
  output["fitted.values"] = fitted;
  output["mse"]           = mse;
  output["rho.updated"]   = rho;
  
  /* Return output      */
  return(output);
}

/*
 Adaptive-LASSO via ADMM, just to compute the whole path fast.
*/
Rcpp::List admm_adalasso_large_m_update_1lambda(const arma::mat& A, const arma::vec& b,
                                                const arma::mat F, const arma::mat FTF,
                                                const arma::mat ATA, const arma::mat& ATA_CHOL_U, const arma::colvec& ATb,
                                                const double b2, const int m, const int n,
                                                arma::colvec u, arma::colvec z, const double lambda,
                                                bool rho_adaptation, double rho, const double tau, const double mu,
                                                const double reltol, const double abstol, const int maxiter, const int ping) {
  
  /* Variable delcaration           */
  int k, dim_z;
  double rho_old, elTime = 0.0, mse, sqrtn;
  
  /* Get dimensions                 */
  dim_z = z.n_elem;
  
  /* Variable definition            */
  sqrtn = std::sqrt(static_cast<float>(n));
  
  /* Definition of vectors and matrices     */
  arma::colvec x(n, fill::zeros);
  arma::colvec q(n, fill::zeros);
  arma::colvec z_old(dim_z, fill::zeros);
  arma::mat U(n, n, fill::zeros);
  arma::mat L(n, n, fill::zeros);
  arma::vec h_objval(maxiter, fill::zeros);
  arma::vec h_r_norm(maxiter, fill::zeros);
  arma::vec h_s_norm(maxiter, fill::zeros);
  arma::vec h_eps_pri(maxiter, fill::zeros);
  arma::vec h_eps_dual(maxiter, fill::zeros);
  arma::vec rho_store(maxiter+1, fill::zeros);
  arma::colvec fitted(m, fill::zeros);
  arma::colvec residuals(m, fill::zeros);
  
  /* ::::::::::::::::::::::::::::::::::::::::::::::
   Set the computational time                                 */
  wall_clock timer;
  timer.tic();
  
  /* store rho (only for adaptation)         */
  rho_store(0) = rho;
  rho_old      = rho;

  /* Precompute relevant quantities         */
  U = adalasso_factor_fast_large_m(ATA, FTF, rho); // returns upper
  L = U.t();
  
  /* main loop  */
  for (k=0; k<maxiter; k++) {

    // 4-1. update 'x'
    if (rho != rho_old) {
      U = adalasso_factor_fast_large_m(ATA, FTF, rho); // returns upper
      L = U.t();
    }
    q = ATb + rho * diagmat(F) * (z - u); // temporary value
    x = solve(trimatu(U), solve(trimatl(L), q));
    
    // 4-2. update 'z' with adaptation
    // see Boyd et al (2011) page 21
    z_old = z;
    z     = lasso_prox(diagmat(F) * x + u, lambda / rho);
        
    // 4-3. update 'u'
    u = u + diagmat(F) * x - z;
    
    // 4-3. dianostics, reporting
    h_objval(k) = adalasso_objfun_fast(ATA_CHOL_U, ATb, b2, lambda, x, z, m, n, diagvec(F));
    h_r_norm(k) = norm(adalasso_residual(F, x, z), 2);
    h_s_norm(k) = norm(adalasso_dual_residual(F, z, z_old, rho), 2);
    if (norm(diagmat(F) * x, 2) > norm(-z, 2)) {
      h_eps_pri(k) = sqrtn * abstol + reltol * norm(diagmat(F) * x, 2);
    } else {
      h_eps_pri(k) = sqrtn * abstol + reltol * norm(-z, 2);
    }
    h_eps_dual(k) = sqrtn * abstol + reltol * norm(diagmat(F) * rho * u, 2);
    
    /* :::::::::::::::::::::::::::::::::::::::::::::::::::::::: */
    /* Relaxation                                               */
    rho_old = rho;
    if (rho_adaptation) {
      if (h_r_norm(k) > mu * h_s_norm(k)) {
        rho = rho * tau;
        u   = u / tau;
      } else if (h_s_norm(k) > mu * h_r_norm(k)) {
        rho = rho / tau;
        u   = u * tau;
      }
    }
    rho_store(k+1) = rho;
        
    /* :::::::::::::::::::::::::::::::::::::::::::::::::::::::: */
    /* Print to screen                                          */
    if ((ping > 0) && (k < maxiter)) {
      if (rho_adaptation) {
        if (((k+1) % ping) == 0) {
          Rcpp::Rcout << "\n\n\n" << std::endl;
          Rcpp::Rcout << "ADMM with adaptation for adaptive-LASSO is running!" << std::endl;
          Rcpp::Rcout << "Iteration n.: " << k+1 << " of " << maxiter << "\n" << std::endl;
        }
      } else {
        if (((k+1) % ping) == 0) {
          Rcpp::Rcout << "\n\n\n" << std::endl;
          Rcpp::Rcout << "ADMM for adaptive-LASSO is running!" << std::endl;
          Rcpp::Rcout << "Iteration n.: " << k+1 << " of " << maxiter << "\n" << std::endl;
        }
      }
    }

    // 4-4. termination
    if ((h_r_norm(k) < h_eps_pri(k)) && (h_s_norm(k) < h_eps_dual(k))) {
      break;
    }
  }

  /* ::::::::::::::::::::::::::::::::::::::::::::::
   Compute elapsed time                                */
  elTime = timer.toc();
  
  /* ::::::::::::::::::::::::::::::::::::::::::::::
   Compute post-estimation quantities                   */
  fitted    = A * x;
  residuals = b - fitted;
  mse       = as_scalar(residuals.t() * residuals) / (static_cast<float>(m));
  
  /* Get output         */
  List output;
  output["x"]      = x;               // coefficient function
  output["u"]      = u;
  output["z"]      = z;
  output["eltime"] = elTime;
  if (k < maxiter) {
    output["convergence"] = 1.0;
    output["niter"]       = k+1;               // number of iterations
    output["objval"]      = h_objval.subvec(0, k);        // |x|_1
    output["r_norm"]      = h_r_norm.subvec(0, k);
    output["s_norm"]      = h_s_norm.subvec(0, k);
    output["eps_pri"]     = h_eps_pri.subvec(0, k);
    output["eps_dual"]    = h_eps_dual.subvec(0, k);
    output["rho"]         = rho_store.subvec(0, k+1);
  } else {
    output["convergence"] = 0.0;
    output["niter"]       = maxiter;               // number of iterations
    output["objval"]      = h_objval;        // |x|_1
    output["r_norm"]      = h_r_norm;
    output["s_norm"]      = h_s_norm;
    output["eps_pri"]     = h_eps_pri;
    output["eps_dual"]    = h_eps_dual;
    output["rho"]         = rho_store;
  }
  output["residuals"]     = residuals;
  output["fitted.values"] = fitted;
  output["mse"]           = mse;
  output["rho.updated"]   = rho;
  
  /* Return output      */
  return(output);
}

Rcpp::List admm_adalasso_large_n_update_1lambda(const arma::mat& A, const arma::vec& b, const arma::mat F,
                                                const arma::mat FTF_INV, const arma::mat AFTF_INVA,
                                                const arma::mat& ATA_CHOL_U, const arma::colvec& ATb,
                                                const double b2, const int m, const int n,
                                                arma::colvec u, arma::colvec z, const double lambda,
                                                bool rho_adaptation, double rho, const double tau, const double mu,
                                                const double reltol, const double abstol, const int maxiter, const int ping) {
  
  /* Variable delcaration           */
  int k, dim_z;
  double rho_old, elTime = 0.0, mse, sqrtn;
 
  /* Get dimensions                 */
  dim_z = z.n_elem;
  
  /* Variable definition            */
  sqrtn = std::sqrt(static_cast<float>(n));
  
  /* Definition of vectors and matrices     */
  arma::colvec x(n, fill::zeros);
  arma::colvec x_star(n, fill::zeros);
  arma::colvec q(n, fill::zeros);
  arma::colvec z_old(dim_z, fill::zeros);
  arma::mat U(m, m, fill::zeros);
  arma::mat L(m, m, fill::zeros);
  arma::vec h_objval(maxiter, fill::zeros);
  arma::vec h_r_norm(maxiter, fill::zeros);
  arma::vec h_s_norm(maxiter, fill::zeros);
  arma::vec h_eps_pri(maxiter, fill::zeros);
  arma::vec h_eps_dual(maxiter, fill::zeros);
  arma::vec rho_store(maxiter+1, fill::zeros);
  arma::colvec fitted(m, fill::zeros);
  arma::colvec residuals(m, fill::zeros);

  /* ::::::::::::::::::::::::::::::::::::::::::::::
   Set the computational time                                 */
  wall_clock timer;
  timer.tic();
  
  /* store rho (only for adaptation)         */
  rho_store(0) = rho;
  rho_old      = rho;

  /* Precompute relevant quantities         */
  U = adalasso_factor_fast_large_n(AFTF_INVA, rho, m);
  L = U.t();
  
  /* main loop  */
  for (k=0; k<maxiter; k++) {

    // 4-1. update 'x'
    if (rho != rho_old) {
      U = adalasso_factor_fast_large_n(AFTF_INVA, rho, m);
      L = U.t();
    }
    q      = ATb + rho * diagmat(F) * (z - u); // temporary value
    x_star = (diagmat(FTF_INV) * q) / rho;
    x      = x_star - diagmat(FTF_INV) * (A.t() * solve(trimatu(U), solve(trimatl(L), A * x_star)));
    
    // 4-2. update 'z' with adaptation
    // see Boyd et al (2011) page 21
    z_old = z;
    z     = lasso_prox(diagmat(F) * x + u, lambda / rho);
    
    // 4-3. update 'u'
    u = u + diagmat(F) * x - z;
    
    // 4-3. dianostics, reporting
    h_objval(k) = adalasso_objfun_fast(ATA_CHOL_U, ATb, b2, lambda, x, z, m, n, diagvec(F));
    h_r_norm(k) = norm(adalasso_residual(F, x, z), 2);
    h_s_norm(k) = norm(adalasso_dual_residual(F, z, z_old, rho), 2);
    if (norm(diagmat(F) * x, 2) > norm(-z, 2)) {
      h_eps_pri(k) = sqrtn * abstol + reltol * norm(diagmat(F) * x, 2);
    } else {
      h_eps_pri(k) = sqrtn * abstol + reltol * norm(-z, 2);
    }
    h_eps_dual(k) = sqrtn * abstol + reltol * norm(diagmat(F) * rho * u, 2);
        
    /* :::::::::::::::::::::::::::::::::::::::::::::::::::::::: */
    /* Relaxation                                               */
    rho_old = rho;
    if (rho_adaptation) {
      if (h_r_norm(k) > mu * h_s_norm(k)) {
        rho = rho * tau;
        u   = u / tau;
      } else if (h_s_norm(k) > mu * h_r_norm(k)) {
        rho = rho / tau;
        u   = u * tau;
      }
    }
    rho_store(k+1) = rho;
            
    /* :::::::::::::::::::::::::::::::::::::::::::::::::::::::: */
    /* Print to screen                                          */
    if ((ping > 0) && (k < maxiter)) {
      if (rho_adaptation) {
        if (((k+1) % ping) == 0) {
          Rcpp::Rcout << "\n\n\n" << std::endl;
          Rcpp::Rcout << "ADMM with adaptation for adaptive-LASSO is running!" << std::endl;
          Rcpp::Rcout << "Iteration n.: " << k+1 << " of " << maxiter << "\n" << std::endl;
        }
      } else {
        if (((k+1) % ping) == 0) {
          Rcpp::Rcout << "\n\n\n" << std::endl;
          Rcpp::Rcout << "ADMM for adaptive-LASSO is running!" << std::endl;
          Rcpp::Rcout << "Iteration n.: " << k+1 << " of " << maxiter << "\n" << std::endl;
        }
      }
    }

    // 4-4. termination
    if ((h_r_norm(k) < h_eps_pri(k)) && (h_s_norm(k) < h_eps_dual(k))) {
      break;
    }
  }

  /* ::::::::::::::::::::::::::::::::::::::::::::::
   Compute elapsed time                                */
  elTime = timer.toc();
  
  /* ::::::::::::::::::::::::::::::::::::::::::::::
   Compute post-estimation quantities                   */
  fitted    = A * x;
  residuals = b - fitted;
  mse       = as_scalar(residuals.t() * residuals) / (static_cast<float>(m));
  
  /* Get output         */
  List output;
  output["x"]      = x;               // coefficient function
  output["u"]      = u;
  output["z"]      = z;
  output["eltime"] = elTime;
  if (k < maxiter) {
    output["convergence"] = 1.0;
    output["niter"]       = k+1;               // number of iterations
    output["objval"]      = h_objval.subvec(0, k);        // |x|_1
    output["r_norm"]      = h_r_norm.subvec(0, k);
    output["s_norm"]      = h_s_norm.subvec(0, k);
    output["eps_pri"]     = h_eps_pri.subvec(0, k);
    output["eps_dual"]    = h_eps_dual.subvec(0, k);
    output["rho"]         = rho_store.subvec(0, k+1);
  } else {
    output["convergence"] = 0.0;
    output["niter"]       = maxiter;               // number of iterations
    output["objval"]      = h_objval;        // |x|_1
    output["r_norm"]      = h_r_norm;
    output["s_norm"]      = h_s_norm;
    output["eps_pri"]     = h_eps_pri;
    output["eps_dual"]    = h_eps_dual;
    output["rho"]         = rho_store;
  }
  output["residuals"]     = residuals;
  output["fitted.values"] = fitted;
  output["mse"]           = mse;
  output["rho.updated"]   = rho;
  
  /* Return output      */
  return(output);
}


