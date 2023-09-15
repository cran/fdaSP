// Utility routines

// Authors:
//           Bernardi Mauro, University of Padova
//           Last update: July 12, 2022

// List of implemented methods:



// [[Rcpp::depends(RcppArmadillo)]]
#include "linreg_ADMM.h"

// ==========================================================
// admm_enet: Linear regression model with ENET penalty
// Linear regression model with ENET penalty: evaluate the objective function
Rcpp::List admm_enet(const arma::mat& A, const arma::colvec& b,  const double lambda, const double alpha,
                     const double reltol, const double abstol, const int maxiter, const double rho) {
  
  /* Variables declaration       */
  double gamma, sqrtn;
  int k;

  /* Get dimensions             */
  const int n = A.n_cols;
  
  /* Vectors and matrices declaration       */
  arma::colvec x(n, fill::zeros);
  arma::colvec z(n, fill::zeros);
  arma::colvec u(n, fill::zeros);
  arma::colvec q(n, fill::zeros);
  arma::colvec z_old(n, fill::zeros);
  arma::colvec x_hat(n, fill::zeros);
  arma::vec h_objval(maxiter, fill::zeros);
  arma::vec h_r_norm(maxiter, fill::zeros);
  arma::vec h_s_norm(maxiter, fill::zeros);
  arma::vec h_eps_pri(maxiter, fill::zeros);
  arma::vec h_eps_dual(maxiter, fill::zeros);

  /* Define relevant quantities     */
  gamma         = lambda * (1.0 - alpha) + rho;
  sqrtn         = std::sqrt(static_cast<float>(n));
  arma::mat ATb = A.t() * b;
  arma::mat U   = enet_factor(A, gamma); // returns upper
  arma::mat L   = U.t();

  for (k=0; k<maxiter; k++) {
    // 4-1. update 'x'
    q = ATb + rho * (z - u); // temporary value
    x = solve(trimatu(U), solve(trimatl(L), q));

    // 4-2. update 'z'
    z_old = z;
    z     = enet_prox(x + u, lambda * alpha / rho);

    // 4-3. update 'u'
    u = u + x - z;

    // 4-3. dianostics, reporting
    h_objval(k) = enet_objfun(A, b, lambda, alpha, x, z);
    h_r_norm(k) = arma::norm(x - z);
    h_s_norm(k) = arma::norm(-rho * (z - z_old));
    if (norm(x) > norm(-z)) {
      h_eps_pri(k) = sqrtn * abstol + reltol * norm(x);
    } else {
      h_eps_pri(k) = sqrtn * abstol + reltol * norm(-z);
    }
    h_eps_dual(k) = sqrtn * abstol + reltol * norm(rho * u);
    // 4-4. termination
    if ((h_r_norm(k) < h_eps_pri(k)) && (h_s_norm(k)<h_eps_dual(k))) {
      break;
    }
  }
  
  /* Get output         */
  Rcpp::List output;
  output["x"]        = x;             // coefficient function
  output["objval"]   = h_objval;      // |x|_1
  output["niter"]    = k;             // number of iterations
  output["r_norm"]   = h_r_norm;
  output["s_norm"]   = h_s_norm;
  output["eps_pri"]  = h_eps_pri;
  output["eps_dual"] = h_eps_dual;

  /* Return output      */
  return(output);
}

// ======================================================
// admm_genlasso: Linear regression model with generalized LASSO penalty
// Linear regression model with generalized LASSO penalty: evaluate the objective function
Rcpp::List admm_genlasso(const arma::mat& A, const arma::colvec& b, const arma::mat &D,
                         const double lambda, const double reltol, const double abstol,
                         const int maxiter, const double rho) {
  // 1. get parameters
  const int n = A.n_cols;

  // 2. set ready
  arma::colvec x(n,fill::randn); x/=10.0;
  arma::colvec z(D*x);
  arma::colvec u(D*x-z);
  arma::colvec q(n,fill::zeros);
  arma::colvec zold(z);
  arma::colvec x_hat(n,fill::zeros);

  // 3. precompute static variables for x-update and factorization
  arma::mat Atb = A.t()*b;
  arma::mat U   = genlasso_factor(A,rho,D); // returns upper
  arma::mat L   = U.t();

  // 4. iteration
  arma::vec h_objval(maxiter,fill::zeros);
  arma::vec h_r_norm(maxiter,fill::zeros);
  arma::vec h_s_norm(maxiter,fill::zeros);
  arma::vec h_eps_pri(maxiter,fill::zeros);
  arma::vec h_eps_dual(maxiter,fill::zeros);

  double sqrtn = std::sqrt(static_cast<float>(n));
  int k;
  for (k=0; k<maxiter; k++) {
    // 4-1. update 'x'
    q = Atb + rho*D.t()*(z-u); // temporary value
    x = solve(trimatu(U),solve(trimatl(L),q));
    //        if (m >= n){
    //            x = solve(trimatu(U),solve(trimatl(L),q));
    //        } else {
    //            x = q/rho - (A.t()*solve(trimatu(U),solve(trimatl(L),A*q)))/rho2;
    //        }

    // 4-2. update 'z'
    zold = z;
    z = genlasso_shrinkage(D*x + u, lambda/rho);

    // 4-3. update 'u'
    u = u + D*x - z;

    // 4-3. dianostics, reporting
    h_objval(k) = genlasso_objective(A,b,D,lambda,x,z);
    h_r_norm(k) = arma::norm(D*x-z);
    h_s_norm(k) = arma::norm(-rho*(z-zold));
    if (norm(x)>norm(-z)){
      h_eps_pri(k) = sqrtn*abstol + reltol*norm(x);
    } else {
      h_eps_pri(k) = sqrtn*abstol + reltol*norm(-z);
    }
    h_eps_dual(k) = sqrtn*abstol + reltol*norm(rho*u);

    // 4-4. termination
    if ((h_r_norm(k) < h_eps_pri(k))&&(h_s_norm(k)<h_eps_dual(k))){
      break;
    }
  }

  // 5. report results
  List output;
  output["x"]        = x;             // coefficient function
  output["objval"]   = h_objval;      // |x|_1
  output["k"]        = k;             // number of iterations
  output["r_norm"]   = h_r_norm;
  output["s_norm"]   = h_s_norm;
  output["eps_pri"]  = h_eps_pri;
  output["eps_dual"] = h_eps_dual;
  
  /* Return output      */
  return(output);
}

// =========================================================
// admm_lasso
// ADMM for linear regression models with LASSO penalty
// elementwise soft thresholding operator
/*
* LASSO via ADMM (from Stanford)
* http://stanford.edu/~boyd/papers/pdf/admm_distr_stats.pdf
* page 19 : section 3.3.1 : stopping criteria part (3.12).
*/
//' @name  fdaSP.internal
//' @keywords internal
// [[Rcpp::export(.admm_lasso)]]
Rcpp::List admm_lasso(const arma::mat& A, const arma::colvec& b, arma::colvec& u, arma::colvec& z, const double lambda,
                      bool rho_adaptation, double rho, const double tau, const double mu,
                      const double reltol, const double abstol, const int maxiter, const int ping) {
    
  /* Variable delcaration           */
  double sqrtn, rho_old, elTime;
  int k, n1;
  
  /* Get dimensions             */
  const int m = A.n_rows;
  const int n = A.n_cols;
    
  /* Variable definition            */
  sqrtn = std::sqrt(static_cast<float>(n));
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
  arma::vec ATb(n, fill::zeros);
  arma::vec h_objval(maxiter, fill::zeros);
  arma::vec h_r_norm(maxiter, fill::zeros);
  arma::vec h_s_norm(maxiter, fill::zeros);
  arma::vec h_eps_pri(maxiter, fill::zeros);
  arma::vec h_eps_dual(maxiter, fill::zeros);
  arma::vec rho_store(maxiter+1, fill::zeros);
  arma::mat AA(n1, n1, fill::zeros);
  
  /* ::::::::::::::::::::::::::::::::::::::::::::::
   Set the computational time                                 */
  wall_clock timer;
  timer.tic();
  
  /* store rho (only for adaptation)         */
  rho_store(0) = rho;
  rho_old      = rho;

  /* Precompute relevant quantities         */
  ATb = A.t() * b / static_cast<float>(m);
  squaredmat(AA, A, m, n);
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
      //x = q / rho - (A.t() * solve(trimatu(U), solve(trimatl(L), A * q))) / rho2;
      x = q / rho - (A.t() * solve(trimatu(U), solve(trimatl(L), A * q))) / rho;
    }

    // 4-2. update 'z' with adaptation
    // see Boyd et al (2011) page 21
    z_old = z;
    z     = lasso_prox(x + u, lambda / rho);
    
    // 4-3. update 'u'
    u = u + x - z;

    // 4-3. dianostics, reporting
    h_objval(k) = lasso_objfun(A, b, lambda, x, z);
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

  /* Get output         */
  List output;
  output["x"] = x;               // coefficient function
  output["u"] = u;
  output["z"] = z;
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
  output["eltime"]   = elTime;
  if ((k+1) < maxiter) {
    output["convergence"] = 1.0;
  } else {
    output["convergence"] = 0.0;
  }

  /* Return output      */
  return(output);
}

/*
* Adaptive LASSO via ADMM (from Stanford)
* http://stanford.edu/~boyd/papers/pdf/admm_distr_stats.pdf
* page 19 : section 3.3.1 : stopping criteria part (3.12).
*/
//' @name  fdaSP.internal
//' @keywords internal
// [[Rcpp::export(.admm_adalasso)]]
Rcpp::List admm_adalasso(const arma::mat& A, const arma::colvec& b, arma::vec& var_weights,
                         arma::colvec& u, arma::colvec& z, const double lambda,
                         bool rho_adaptation, double rho, const double tau, const double mu,
                         const double reltol, const double abstol, const int maxiter, const int ping) {
  
  /* Get dimensions             */
  const int m = A.n_rows;
  const int n = A.n_cols;
  
  /* Variable delcaration           */
  List out;
    
  /* Run ADMM            */
  if (m >= n) {
    /* large n problems            */
    out = admm_adalasso_large_m(A, b, var_weights, u, z, lambda, rho_adaptation,
                                rho, tau, mu, reltol, abstol, maxiter, ping);
  } else {
    /* large p problems            */
    out = admm_adalasso_large_n(A, b, var_weights, u, z, lambda, rho_adaptation,
                                rho, tau, mu, reltol, abstol, maxiter, ping);
  }

  /* Return output      */
  return(out);
}

Rcpp::List admm_adalasso_large_m(const arma::mat& A, const arma::colvec& b, arma::vec& var_weights,
                                 arma::colvec& u, arma::colvec& z, const double lambda,
                                 bool rho_adaptation, double rho, const double tau, const double mu,
                                 const double reltol, const double abstol, const int maxiter, const int ping) {
    
  /* Variable delcaration           */
  double sqrtn, rho_old, elTime;
  int k;
  
  /* Get dimensions             */
  const int m = A.n_rows;
  const int n = A.n_cols;
    
  /* Variable definition            */
  sqrtn = std::sqrt(static_cast<float>(n));
  
  /* Definition of vectors and matrices     */
  arma::colvec x(n, fill::zeros);
  arma::colvec q(n, fill::zeros);
  arma::colvec z_old(n, fill::zeros);
  arma::mat U(n, n, fill::zeros);
  arma::mat L(n, n, fill::zeros);
  arma::mat ATA(n, n, fill::zeros);
  arma::vec ATb(n, fill::zeros);
  arma::mat F(n, n, fill::zeros);
  arma::mat FTF(n, n, fill::zeros);
  arma::vec h_objval(maxiter, fill::zeros);
  arma::vec h_r_norm(maxiter, fill::zeros);
  arma::vec h_s_norm(maxiter, fill::zeros);
  arma::vec h_eps_pri(maxiter, fill::zeros);
  arma::vec h_eps_dual(maxiter, fill::zeros);
  arma::vec rho_store(maxiter+1, fill::zeros);
  
  /* ::::::::::::::::::::::::::::::::::::::::::::::
   Set the computational time                                 */
  wall_clock timer;
  timer.tic();
  
  /* store rho (only for adaptation)         */
  rho_store(0) = rho;
  rho_old      = rho;

  /* Precompute relevant quantities         */
  ATA = A.t() * A / static_cast<float>(m);
  ATb = A.t() * b / static_cast<float>(m);
  F   = adalasso_Fmat(var_weights);
  FTF = diagmat(F) * diagmat(F);
  U   = adalasso_factor_fast_large_m(ATA, FTF, rho); // returns upper
  L   = U.t();
  
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
    h_objval(k) = adalasso_objfun(A, b, lambda, x, z, var_weights);
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
    if (ping > 0) {
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
    }
    
    // 4-4. termination
    if ((h_r_norm(k) < h_eps_pri(k)) && (h_s_norm(k) < h_eps_dual(k))) {
      break;
    }
  }
  
  /* ::::::::::::::::::::::::::::::::::::::::::::::
   Compute elapsed time                                */
  elTime = timer.toc();

  /* Get output         */
  List output;
  output["niter"] = k;               // number of iterations
  output["x"]     = x;               // coefficient function
  output["u"]     = u;
  output["z"]     = z;
  if (k < maxiter) {
    output["objval"]   = h_objval.subvec(0, k);        // |x|_1
    output["r_norm"]   = h_r_norm.subvec(0, k);
    output["s_norm"]   = h_s_norm.subvec(0, k);
    output["eps_pri"]  = h_eps_pri.subvec(0, k);
    output["eps_dual"] = h_eps_dual.subvec(0, k);
    output["rho"]      = rho_store.subvec(0, k+1);
  } else {
    output["objval"]   = h_objval;        // |x|_1
    output["r_norm"]   = h_r_norm;
    output["s_norm"]   = h_s_norm;
    output["eps_pri"]  = h_eps_pri;
    output["eps_dual"] = h_eps_dual;
    output["rho"]      = rho_store;
  }
  output["eltime"] = elTime;
  if ((k+1) < maxiter) {
    output["convergence"] = 1.0;
  } else {
    output["convergence"] = 1.0;
  }

  /* Return output      */
  return(output);
}

Rcpp::List admm_adalasso_large_n(const arma::mat& A, const arma::colvec& b, arma::vec& var_weights,
                                 arma::colvec& u, arma::colvec& z, const double lambda,
                                 bool rho_adaptation, double rho, const double tau, const double mu,
                                 const double reltol, const double abstol, const int maxiter, const int ping) {
    
  /* Variable delcaration           */
  double sqrtn, rho_old, elTime;
  int k;
  
  /* Get dimensions             */
  const int m = A.n_rows;
  const int n = A.n_cols;
    
  /* Variable definition            */
  sqrtn = std::sqrt(static_cast<float>(n));
    
  /* Definition of vectors and matrices     */
  arma::colvec x(n, fill::zeros);
  arma::colvec x_star(n, fill::zeros);
  arma::colvec q(n, fill::zeros);
  arma::colvec z_old(n, fill::zeros);
  arma::mat U(m, m, fill::zeros);
  arma::mat L(m, m, fill::zeros);
  arma::vec ATb(n, fill::zeros);
  arma::mat F(n, n, fill::zeros);
  arma::mat FTF(n, n, fill::zeros);
  arma::mat FTF_INV(n, n, fill::zeros);
  arma::mat AFTF_INVA(m, m, fill::zeros);
  arma::vec h_objval(maxiter, fill::zeros);
  arma::vec h_r_norm(maxiter, fill::zeros);
  arma::vec h_s_norm(maxiter, fill::zeros);
  arma::vec h_eps_pri(maxiter, fill::zeros);
  arma::vec h_eps_dual(maxiter, fill::zeros);
  arma::vec rho_store(maxiter+1, fill::zeros);
    
  /* ::::::::::::::::::::::::::::::::::::::::::::::
   Set the computational time                                 */
  wall_clock timer;
  timer.tic();
  
  /* store rho (only for adaptation)         */
  rho_store(0) = rho;
  rho_old      = rho;
  
  /* Precompute relevant quantities         */
  ATb       = A.t() * b / static_cast<float>(m);
  F         = adalasso_Fmat(var_weights);
  FTF       = diagmat(F) * diagmat(F);
  FTF_INV   = inv(diagmat(FTF));
  AFTF_INVA = A * diagmat(FTF_INV) * A.t();
  U         = adalasso_factor_fast_large_n(AFTF_INVA, rho, m);
  L         = U.t();

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
    h_objval(k) = adalasso_objfun(A, b, lambda, x, z, var_weights);
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
    if (ping > 0) {
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
    }
    
    // 4-4. termination
    if ((h_r_norm(k) < h_eps_pri(k)) && (h_s_norm(k) < h_eps_dual(k))) {
      break;
    }
  }
  
  /* ::::::::::::::::::::::::::::::::::::::::::::::
   Compute elapsed time                                */
  elTime = timer.toc();

  /* Get output         */
  List output;
  output["niter"] = k;               // number of iterations
  output["x"]     = x;               // coefficient function
  output["u"]     = u;
  output["z"]     = z;
  if (k < maxiter) {
    output["objval"]   = h_objval.subvec(0, k);        // |x|_1
    output["r_norm"]   = h_r_norm.subvec(0, k);
    output["s_norm"]   = h_s_norm.subvec(0, k);
    output["eps_pri"]  = h_eps_pri.subvec(0, k);
    output["eps_dual"] = h_eps_dual.subvec(0, k);
    output["rho"]      = rho_store.subvec(0, k+1);
  } else {
    output["objval"]   = h_objval;        // |x|_1
    output["r_norm"]   = h_r_norm;
    output["s_norm"]   = h_s_norm;
    output["eps_pri"]  = h_eps_pri;
    output["eps_dual"] = h_eps_dual;
    output["rho"]      = rho_store;
  }
  output["eltime"] = elTime;
  if ((k+1) < maxiter) {
    output["convergence"] = 1.0;
  } else {
    output["convergence"] = 1.0;
  }

  /* Return output      */
  return(output);
}

// =========================================================
// admm_glasso
// ADMM for linear regression models with group-LASSO penalty

/* Group Lasso vector (or block) soft thresholding operator        */
/*
* GLASSO via ADMM (from Stanford)
*/
//' @name  fdaSP.internal
//' @keywords internal
// [[Rcpp::export(.admm_glasso)]]
Rcpp::List admm_glasso(const arma::mat& A, const arma::colvec& b,
                       arma::mat& groups, arma::vec& group_weights, arma::vec& var_weights,
                       arma::colvec& u, arma::colvec& z, const double lambda,
                       bool rho_adaptation, double rho, const double tau, const double mu,
                       const double reltol, const double abstol, const int maxiter, const int ping) {
  
  /* Get dimensions             */
  const int m = A.n_rows;
  const int n = A.n_cols;
  
  /* Variable delcaration           */
  List out;
    
  /* Run ADMM            */
  if (m >= n) {
    /* large n problems            */
    out = admm_glasso_large_m(A, b, groups, group_weights, var_weights, u, z, lambda,
                              rho_adaptation, rho, tau, mu, reltol, abstol, maxiter, ping);
  } else {
    /* large p problems            */
    out = admm_glasso_large_n(A, b, groups, group_weights, var_weights, u, z, lambda,
                              rho_adaptation, rho, tau, mu, reltol, abstol, maxiter, ping);
  }

  /* Return output      */
  return(out);
}

Rcpp::List admm_glasso_large_m(const arma::mat& A, const arma::colvec& b,
                               arma::mat& groups, arma::vec& group_weights, arma::vec& var_weights,
                               arma::colvec& u, arma::colvec& z, const double lambda,
                               bool rho_adaptation, double rho, const double tau, const double mu,
                               const double reltol, const double abstol, const int maxiter, const int ping) {
    
  /* Variable delcaration           */
  int k;
  uword g_id_init, g_id_end;
  double rho_old, elTime;
  
  /* Get dimensions             */
  const int m     = A.n_rows;
  const int n     = A.n_cols;
  const int G     = groups.n_rows;
  const int dim_z = z.n_elem;
      
  /* Definition of vectors and matrices     */
  arma::colvec x(n, fill::zeros);
  arma::colvec q(n, fill::zeros);
  arma::colvec z_old(dim_z, fill::zeros);
  arma::mat ATA(n, n, fill::zeros);
  arma::mat FTF(n, n, fill::zeros);
  arma::vec ATb(n, fill::zeros);
  arma::mat U(n, n, fill::zeros);
  arma::mat L(n, n, fill::zeros);
  arma::colvec Glen(G, fill::zeros);
  arma::vec h_objval(maxiter, fill::zeros);
  arma::vec h_r_norm(maxiter, fill::zeros);
  arma::vec h_s_norm(maxiter, fill::zeros);
  arma::vec h_eps_pri(maxiter, fill::zeros);
  arma::vec h_eps_dual(maxiter, fill::zeros);
  arma::vec rho_store(maxiter+1, fill::zeros);
    
  /* ::::::::::::::::::::::::::::::::::::::::::::::
   Set the computational time                                 */
  wall_clock timer;
  timer.tic();
  
  /* store rho (only for adaptation)         */
  rho_store(0) = rho;
  rho_old      = rho;
  
  /* Precompute relevant quantities         */
  ATA            = A.t() * A / static_cast<float>(m);
  ATb            = A.t() * b / static_cast<float>(m);
  Glen           = sum(groups, 1);
  arma::sp_mat F = glasso_Gmat2Fmat_sparse(groups, group_weights, var_weights);
  FTF            = F.t() * F;
  U              = glasso_factor_fast_large_m(ATA, FTF, rho); // returns upper
  L              = U.t();
  
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
    h_objval(k) = glasso_objfun(A, b, Glen, lambda, x, z, G);
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
  
  /* :::::::::::::::::::::::::::::::::::::::::::::::::::::::: */
  /* Print to screen                                          */
  if (ping > 0) {
    Rcpp::Rcout << "ADMM for group-LASSO" << std::endl;
    Rcpp::Rcout << "Iterations n.: " << k << std::endl;
    if ((k+1) < maxiter) {
      Rcpp::Rcout << "Convergence achieved!" << std::endl;
    }
  }

  /* Get output         */
  List output;
  output["x"] = x;               // coefficient function
  output["u"] = u;
  output["z"] = z;
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
  output["eltime"]   = elTime;
  if ((k+1) < maxiter) {
    output["convergence"] = 1.0;
  } else {
    output["convergence"] = 0.0;
  }

  /* Return output      */
  return(output);
}

Rcpp::List admm_glasso_large_n(const arma::mat& A, const arma::colvec& b,
                               arma::mat& groups, arma::vec& group_weights, arma::vec& var_weights,
                               arma::colvec& u, arma::colvec& z, const double lambda,
                               bool rho_adaptation, double rho, const double tau, const double mu,
                               const double reltol, const double abstol, const int maxiter, const int ping) {
    
  /* Variable delcaration           */
  int k;
  uword g_id_init, g_id_end;
  double rho_old, elTime;
  
  /* Get dimensions             */
  const int m     = A.n_rows;
  const int n     = A.n_cols;
  const int G     = groups.n_rows;
  const int dim_z = z.n_elem;
      
  /* Definition of vectors and matrices     */
  arma::colvec x(n, fill::zeros);
  arma::colvec x_star(n, fill::zeros);
  arma::colvec q(n, fill::zeros);
  arma::colvec z_old(dim_z, fill::zeros);
  arma::mat FTF(n, n, fill::zeros);
  arma::mat FTF_INV(n, n, fill::zeros);
  arma::mat AFTF_INVA(m, m, fill::zeros);
  arma::vec ATb(n, fill::zeros);
  arma::mat U(m, m, fill::zeros);
  arma::mat L(m, m, fill::zeros);
  arma::colvec Glen(G, fill::zeros);
  arma::vec h_objval(maxiter, fill::zeros);
  arma::vec h_r_norm(maxiter, fill::zeros);
  arma::vec h_s_norm(maxiter, fill::zeros);
  arma::vec h_eps_pri(maxiter, fill::zeros);
  arma::vec h_eps_dual(maxiter, fill::zeros);
  arma::vec rho_store(maxiter+1, fill::zeros);
  
  /* ::::::::::::::::::::::::::::::::::::::::::::::
   Set the computational time                                 */
  wall_clock timer;
  timer.tic();
  
  /* store rho (only for adaptation)         */
  rho_store(0) = rho;
  rho_old      = rho;
    
  /* Precompute relevant quantities         */
  ATb            = A.t() * b / static_cast<float>(m);
  Glen           = sum(groups, 1);
  arma::sp_mat F = glasso_Gmat2Fmat_sparse(groups, group_weights, var_weights);
  FTF            = F.t() * F;
  FTF_INV        = inv(diagmat(FTF));
  AFTF_INVA      = A * FTF_INV * A.t();
  U              = glasso_factor_fast_large_n(AFTF_INVA, rho, m);
  L              = U.t();

  /* main loop  */
  for (k=0; k<maxiter; k++) {
    // 4-1. update 'x'
    if (rho != rho_old) {
      U = glasso_factor_fast_large_n(AFTF_INVA, rho, m);
      L = U.t();
    }
    q      = ATb + rho * spmat2vec_mult(F.t(), z - u); // temporary value
    x_star = (diagmat(FTF_INV) * q) / rho;
    x      = x_star - FTF_INV * A.t() * solve(trimatu(U), solve(trimatl(L), A * x_star));

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
    h_objval(k) = glasso_objfun(A, b, Glen, lambda, x, z, G);
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
  
  /* :::::::::::::::::::::::::::::::::::::::::::::::::::::::: */
  /* Print to screen                                          */
  if (ping > 0) {
    Rcpp::Rcout << "ADMM for group-LASSO" << std::endl;
    Rcpp::Rcout << "Iterations n.: " << k << std::endl;
    if ((k+1) < maxiter) {
      Rcpp::Rcout << "Convergence achieved!" << std::endl;
    }
  }

  /* Get output         */
  List output;
  output["x"] = x;               // coefficient function
  output["u"] = u;
  output["z"] = z;
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
  output["eltime"]   = elTime;
  if ((k+1) < maxiter) {
    output["convergence"] = 1.0;
  } else {
    output["convergence"] = 0.0;
  }

  /* Return output      */
  return(output);
}

// =========================================================
// admm_spglasso
// ADMM for linear regression models with sparse group-LASSO penalty
/*
* SPGLASSO via ADMM (from Stanford)
*/
//' @name  fdaSP.internal
//' @keywords internal
// [[Rcpp::export(.admm_spglasso)]]
Rcpp::List admm_spglasso(const arma::mat& A, const arma::colvec& b,
                         arma::mat& groups, arma::vec& group_weights,
                         arma::vec& var_weights, arma::vec& var_weights_L1,
                         arma::colvec& u, arma::colvec& z, const double lambda1, const double lambda2,
                         bool rho_adaptation, double rho, const double tau, const double mu,
                         const double reltol, const double abstol, const int maxiter, const int ping) {
  
  /* Get dimensions             */
  const int m = A.n_rows;
  const int n = A.n_cols;
  
  /* Variable delcaration           */
  List out;
    
  /* Run ADMM            */
  if (m >= n) {
    /* large n problems            */
    if (lambda2 == 0) {
      out = admm_glasso_large_m(A, b, groups, group_weights, var_weights, u, z, lambda1, rho_adaptation, rho, tau, mu, reltol, abstol, maxiter, ping);
    } else if (lambda2 == 1) {
      out = admm_adalasso(A, b, var_weights_L1, u, z, lambda1, rho_adaptation, rho, tau, mu, reltol, abstol, maxiter, ping);
    } else {
      out = admm_spglasso_large_m(A, b, groups, group_weights, var_weights, var_weights_L1,
                                  u, z, lambda1, lambda2, rho_adaptation, rho, tau, mu,
                                  reltol, abstol, maxiter, ping);
    }
  } else {
    /* large p problems            */
    if (lambda2 == 0) {
      out = admm_glasso_large_n(A, b, groups, group_weights, var_weights, u, z, lambda1, rho_adaptation, rho, tau, mu, reltol, abstol, maxiter, ping);
    } else if (lambda2 == 1) {
      out = admm_adalasso(A, b, var_weights_L1, u, z, lambda1, rho_adaptation, rho, tau, mu, reltol, abstol, maxiter, ping);
    } else {
      out = admm_spglasso_large_n(A, b, groups, group_weights, var_weights, var_weights_L1,
                                  u, z, lambda1, lambda2, rho_adaptation, rho, tau, mu,
                                  reltol, abstol, maxiter, ping);
    }
  }

  /* Return output      */
  return(out);
}

Rcpp::List admm_spglasso_large_m(const arma::mat& A, const arma::colvec& b,
                                 arma::mat& groups, arma::vec& group_weights,
                                 arma::vec& var_weights, arma::vec& var_weights_L1,
                                 arma::colvec& u, arma::colvec& z, const double lambda1, const double lambda2,
                                 bool rho_adaptation, double rho, const double tau, const double mu,
                                 const double reltol, const double abstol, const int maxiter, const int ping) {
    
  /* Variable delcaration           */
  //double rho2;
  int k;
  uword g_id_init, g_id_end;
  double spglasso_alpha, spglasso_lambda, rho_old, elTime;
  
  /* Get dimensions             */
  const int m     = A.n_rows;
  const int n     = A.n_cols;
  const int G     = groups.n_rows;
  const int dim_z = z.n_elem;
      
  /* Definition of vectors and matrices     */
  arma::colvec x(n, fill::zeros);
  arma::colvec q(n, fill::zeros);
  arma::colvec z_old(dim_z, fill::zeros);
  arma::mat ATA(n, n, fill::zeros);
  arma::mat FTF(n, n, fill::zeros);
  arma::vec ATb(n, fill::zeros);
  arma::mat U(n, n, fill::zeros);
  arma::mat L(n, n, fill::zeros);
  arma::colvec Glen(G, fill::zeros);
  arma::vec h_objval(maxiter, fill::zeros);
  arma::vec h_r_norm(maxiter, fill::zeros);
  arma::vec h_s_norm(maxiter, fill::zeros);
  arma::vec h_eps_pri(maxiter, fill::zeros);
  arma::vec h_eps_dual(maxiter, fill::zeros);
  arma::vec rho_store(maxiter+1, fill::zeros);
  
  /* ::::::::::::::::::::::::::::::::::::::::::::::
   Set the computational time                                 */
  wall_clock timer;
  timer.tic();
  
  /* store rho (only for adaptation)         */
  rho_store(0) = rho;
  rho_old      = rho;
  
  /* Precompute relevant quantities         */
  spglasso_lambda = lambda1;
  spglasso_alpha  = lambda2;
  ATA             = A.t() * A / static_cast<float>(m);
  ATb             = A.t() * b / static_cast<float>(m);
  Glen            = sum(groups, 1);
  arma::sp_mat F  = spglasso_Gmat2Fmat_sparse(groups, group_weights, var_weights, var_weights_L1);
  FTF             = F.t() * F;
  U               = glasso_factor_fast_large_m(ATA, FTF, rho); // returns upper
  L               = U.t();
  
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
      arma::vec z_g                 = glasso_prox(F_g * x + u_g, spglasso_lambda * (1.0 - spglasso_alpha) / rho);
      z.subvec(g_id_init, g_id_end) = z_g;
      g_id_init                     = g_id_end + 1;
    }
    // update z for the last group
    g_id_end                      = g_id_init + n - 1;
    arma::vec u_g                 = u.subvec(g_id_init, g_id_end);
    arma::sp_mat F_g              = F.submat(g_id_init, 0, g_id_end, n-1);
    arma::vec z_g                 = lasso_prox(F_g * x + u_g, spglasso_lambda * spglasso_alpha / rho);
    z.subvec(g_id_init, g_id_end) = z_g;

    // 4-3. update 'u'
    u += spmat2vec_mult(F, x) - z;
  
    // 4-3. dianostics, reporting
    h_objval(k) = spglasso_objfun(A, b, Glen, spglasso_lambda, spglasso_alpha, x, z, G);
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
    
    /* ::::::::::::::::::::::::::::::::::::::::::::::
     Compute elapsed time                                */
    elTime = timer.toc();
    
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
  
  /* :::::::::::::::::::::::::::::::::::::::::::::::::::::::: */
  /* Print to screen                                          */
  if (ping > 0) {
    Rcpp::Rcout << "ADMM for sparse group-LASSO" << std::endl;
    Rcpp::Rcout << "Iterations n.: " << k << std::endl;
    if ((k+1) < maxiter) {
      Rcpp::Rcout << "Convergence achieved!" << std::endl;
    }
  }

  /* Get output         */
  List output;
  output["x"] = x;               // coefficient function
  output["u"] = u;
  output["z"] = z;
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
  output["eltime"]   = elTime;
  if ((k+1) < maxiter) {
    output["convergence"] = 1.0;
  } else {
    output["convergence"] = 0.0;
  }

  /* Return output      */
  return(output);
}

Rcpp::List admm_spglasso_large_n(const arma::mat& A, const arma::colvec& b,
                                 arma::mat& groups, arma::vec& group_weights,
                                 arma::vec& var_weights, arma::vec& var_weights_L1,
                                 arma::colvec& u, arma::colvec& z, const double lambda1, const double lambda2,
                                 bool rho_adaptation, double rho, const double tau, const double mu,
                                 const double reltol, const double abstol, const int maxiter, const int ping) {
    
  /* Variable delcaration           */
  //double rho2;
  int k;
  uword g_id_init, g_id_end;
  double spglasso_alpha, spglasso_lambda, rho_old, elTime;
  
  /* Get dimensions             */
  const int m     = A.n_rows;
  const int n     = A.n_cols;
  const int G     = groups.n_rows;
  const int dim_z = z.n_elem;
      
  /* Definition of vectors and matrices     */
  arma::colvec x(n, fill::zeros);
  arma::colvec x_star(n, fill::zeros);
  arma::colvec q(n, fill::zeros);
  arma::colvec z_old(dim_z, fill::zeros);
  arma::mat FTF(n, n, fill::zeros);
  arma::mat FTF_INV(n, n, fill::zeros);
  arma::mat AFTF_INVA(m, m, fill::zeros);
  arma::vec ATb(n, fill::zeros);
  arma::mat U(m, m, fill::zeros);
  arma::mat L(m, m, fill::zeros);
  arma::colvec Glen(G, fill::zeros);
  arma::vec h_objval(maxiter, fill::zeros);
  arma::vec h_r_norm(maxiter, fill::zeros);
  arma::vec h_s_norm(maxiter, fill::zeros);
  arma::vec h_eps_pri(maxiter, fill::zeros);
  arma::vec h_eps_dual(maxiter, fill::zeros);
  arma::vec rho_store(maxiter+1, fill::zeros);

  /* ::::::::::::::::::::::::::::::::::::::::::::::
   Set the computational time                                 */
  wall_clock timer;
  timer.tic();
  
  /* store rho (only for adaptation)         */
  rho_store(0) = rho;
  rho_old      = rho;
  
  /* Precompute relevant quantities         */
  spglasso_lambda = lambda1;
  spglasso_alpha  = lambda2;
  ATb             = A.t() * b / static_cast<float>(m);
  Glen            = sum(groups, 1);
  arma::sp_mat F  = spglasso_Gmat2Fmat_sparse(groups, group_weights, var_weights, var_weights_L1);
  FTF             = F.t() * F;
  FTF_INV         = inv(diagmat(FTF));
  AFTF_INVA       = A * FTF_INV * A.t();
  U               = glasso_factor_fast_large_n(AFTF_INVA, rho, m);
  L               = U.t();
  
  /* main loop  */
  for (k=0; k<maxiter; k++) {
    // 4-1. update 'x'
    if (rho != rho_old) {
      U = glasso_factor_fast_large_n(AFTF_INVA, rho, m); // returns upper
      L = U.t();
    }
    q      = ATb + rho * spmat2vec_mult(F.t(), z - u); // temporary value
    x_star = (diagmat(FTF_INV) * q) / rho;
    x      = x_star - FTF_INV * A.t() * solve(trimatu(U), solve(trimatl(L), A * x_star));

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
      arma::vec z_g                 = glasso_prox(F_g * x + u_g, spglasso_lambda * (1.0 - spglasso_alpha) / rho);
      z.subvec(g_id_init, g_id_end) = z_g;
      g_id_init                     = g_id_end + 1;
    }
    // update z for the last group
    g_id_end                      = g_id_init + n - 1;
    arma::vec u_g                 = u.subvec(g_id_init, g_id_end);
    arma::sp_mat F_g              = F.submat(g_id_init, 0, g_id_end, n-1);
    arma::vec z_g                 = lasso_prox(F_g * x + u_g, spglasso_lambda * spglasso_alpha / rho);
    z.subvec(g_id_init, g_id_end) = z_g;

    // 4-3. update 'u'
    u += spmat2vec_mult(F, x) - z;
  
    // 4-3. dianostics, reporting
    h_objval(k) = spglasso_objfun(A, b, Glen, spglasso_lambda, spglasso_alpha, x, z, G);
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
    
    /* ::::::::::::::::::::::::::::::::::::::::::::::
     Compute elapsed time                                */
    elTime = timer.toc();
    
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
  
  /* :::::::::::::::::::::::::::::::::::::::::::::::::::::::: */
  /* Print to screen                                          */
  if (ping > 0) {
    Rcpp::Rcout << "ADMM for sparse group-LASSO" << std::endl;
    Rcpp::Rcout << "Iterations n.: " << k << std::endl;
    if ((k+1) < maxiter) {
      Rcpp::Rcout << "Convergence achieved!" << std::endl;
    }
  }

  /* Get output         */
  List output;
  output["x"] = x;               // coefficient function
  output["u"] = u;
  output["z"] = z;
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
  output["eltime"]   = elTime;
  if ((k+1) < maxiter) {
    output["convergence"] = 1.0;
  } else {
    output["convergence"] = 0.0;
  }

  /* Return output      */
  return(output);
}

// =========================================================
// admm_ovglasso
// ADMM for linear regression models with overlap group-LASSO penalty

/*
* Overlap Group-LASSO via ADMM (from Stanford)
*/
//' @name  fdaSP.internal
//' @keywords internal
// [[Rcpp::export(.admm_ovglasso)]]
Rcpp::List admm_ovglasso(const arma::mat& A, const arma::colvec& b,
                         arma::mat& groups, arma::vec& group_weights, arma::vec& var_weights,
                         arma::colvec& u, arma::colvec& z, const double lambda,
                         bool rho_adaptation, double rho, const double tau, const double mu,
                         const double reltol, const double abstol, const int maxiter, const int ping) {
  
  /* Get dimensions             */
  const int m = A.n_rows;
  const int n = A.n_cols;
  
  /* Variable delcaration           */
  List out;
    
  /* Run ADMM            */
  if (m >= n) {
    /* large n problems            */
    out = admm_ovglasso_large_m(A, b, groups, group_weights, var_weights,
                                u, z, lambda, rho_adaptation, rho, tau, mu,
                                reltol, abstol, maxiter, ping);
  } else {
    /* large p problems            */
    out = admm_ovglasso_large_m(A, b, groups, group_weights, var_weights,
                                u, z, lambda, rho_adaptation, rho, tau, mu,
                                reltol, abstol, maxiter, ping);
  }

  /* Return output      */
  return(out);
}

Rcpp::List admm_ovglasso_large_m(const arma::mat& A, const arma::colvec& b,
                                 arma::mat& groups, arma::vec& group_weights, arma::vec& var_weights,
                                 arma::colvec& u, arma::colvec& z, const double lambda,
                                 bool rho_adaptation, double rho, const double tau, const double mu,
                                 const double reltol, const double abstol, const int maxiter, const int ping) {
    
  /* Variable delcaration           */
  int k;
  uword g_id_init, g_id_end;
  double rho_old, elTime;
  
  /* Get dimensions             */
  const int m     = A.n_rows;
  const int n     = A.n_cols;
  const int G     = groups.n_rows;
  const int dim_z = z.n_elem;
  
  /* Definition of vectors and matrices     */
  arma::colvec x(n, fill::zeros);
  arma::colvec q(n, fill::zeros);
  arma::colvec z_old(dim_z, fill::zeros);
  arma::mat ATA(n, n, fill::zeros);
  arma::mat FTF(n, n, fill::zeros);
  arma::vec ATb(n, fill::zeros);
  arma::mat U(n, n, fill::zeros);
  arma::mat L(n, n, fill::zeros);
  arma::colvec Glen(G, fill::zeros);
  arma::vec h_objval(maxiter, fill::zeros);
  arma::vec h_r_norm(maxiter, fill::zeros);
  arma::vec h_s_norm(maxiter, fill::zeros);
  arma::vec h_eps_pri(maxiter, fill::zeros);
  arma::vec h_eps_dual(maxiter, fill::zeros);
  arma::vec rho_store(maxiter+1, fill::zeros);
  
  /* ::::::::::::::::::::::::::::::::::::::::::::::
   Set the computational time                                 */
  wall_clock timer;
  timer.tic();
  
  /* store rho (only for adaptation)         */
  rho_store(0) = rho;
  rho_old      = rho;

  /* Precompute relevant quantities         */
  ATA            = A.t() * A / static_cast<float>(m);
  ATb            = A.t() * b / static_cast<float>(m);
  Glen           = sum(groups, 1);
  arma::sp_mat F = glasso_Gmat2Fmat_sparse(groups, group_weights, var_weights);
  FTF            = F.t() * F;
  U              = glasso_factor_fast_large_m(ATA, FTF, rho); // returns upper
  L              = U.t();
  
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
    h_objval(k) = glasso_objfun(A, b, Glen, lambda, x, z, G);
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
    
    /* ::::::::::::::::::::::::::::::::::::::::::::::
     Compute elapsed time                                */
    elTime = timer.toc();
    
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

  /* Get output         */
  List output;
  output["x"] = x;               // coefficient function
  output["u"] = u;
  output["z"] = z;
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
  output["eltime"]   = elTime;
  if ((k+1) < maxiter) {
    output["convergence"] = 1.0;
  } else {
    output["convergence"] = 0.0;
  }

  /* Return output      */
  return(output);
}

Rcpp::List admm_ovglasso_large_n(const arma::mat& A, const arma::colvec& b,
                                 arma::mat& groups, arma::vec& group_weights, arma::vec& var_weights,
                                 arma::colvec& u, arma::colvec& z, const double lambda,
                                 bool rho_adaptation, double rho, const double tau, const double mu,
                                 const double reltol, const double abstol, const int maxiter, const int ping) {
    
  /* Variable delcaration           */
  int k;
  uword g_id_init, g_id_end;
  double rho_old, elTime;
  
  /* Get dimensions             */
  const int m     = A.n_rows;
  const int n     = A.n_cols;
  const int G     = groups.n_rows;
  const int dim_z = z.n_elem;
  
  /* Definition of vectors and matrices     */
  arma::colvec x(n, fill::zeros);
  arma::colvec x_star(n, fill::zeros);
  arma::colvec q(n, fill::zeros);
  arma::colvec z_old(dim_z, fill::zeros);
  arma::mat FTF(n, n, fill::zeros);
  arma::mat FTF_INV(n, n, fill::zeros);
  arma::mat AFTF_INVA(m, m, fill::zeros);
  arma::vec ATb(n, fill::zeros);
  arma::mat U(m, m, fill::zeros);
  arma::mat L(m, m, fill::zeros);
  arma::colvec Glen(G, fill::zeros);
  arma::vec h_objval(maxiter, fill::zeros);
  arma::vec h_r_norm(maxiter, fill::zeros);
  arma::vec h_s_norm(maxiter, fill::zeros);
  arma::vec h_eps_pri(maxiter, fill::zeros);
  arma::vec h_eps_dual(maxiter, fill::zeros);
  arma::vec rho_store(maxiter+1, fill::zeros);
  
  /* ::::::::::::::::::::::::::::::::::::::::::::::
   Set the computational time                                 */
  wall_clock timer;
  timer.tic();
  
  /* store rho (only for adaptation)         */
  rho_store(0) = rho;
  rho_old      = rho;

  /* Precompute relevant quantities         */
  ATb            = A.t() * b / static_cast<float>(m);
  Glen           = sum(groups, 1);
  arma::sp_mat F = glasso_Gmat2Fmat_sparse(groups, group_weights, var_weights);
  FTF            = F.t() * F;
  FTF_INV        = inv(diagmat(FTF));
  AFTF_INVA      = A * FTF_INV * A.t();
  U              = glasso_factor_fast_large_n(AFTF_INVA, rho, m);
  L              = U.t();
  
  /* main loop  */
  for (k=0; k<maxiter; k++) {
    // 4-1. update 'x'
    if (rho != rho_old) {
      U = glasso_factor_fast_large_n(AFTF_INVA, rho, m); // returns upper
      L = U.t();
    }
    q      = ATb + rho * spmat2vec_mult(F.t(), z - u); // temporary value
    x_star = (diagmat(FTF_INV) * q) / rho;
    x      = x_star - FTF_INV * A.t() * solve(trimatu(U), solve(trimatl(L), A * x_star));

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
    h_objval(k) = glasso_objfun(A, b, Glen, lambda, x, z, G);
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
    
    /* ::::::::::::::::::::::::::::::::::::::::::::::
     Compute elapsed time                                */
    elTime = timer.toc();
    
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

  /* Get output         */
  List output;
  output["x"] = x;               // coefficient function
  output["u"] = u;
  output["z"] = z;
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
  output["eltime"]   = elTime;
  if ((k+1) < maxiter) {
    output["convergence"] = 1.0;
  } else {
    output["convergence"] = 0.0;
  }

  /* Return output      */
  return(output);
}

/*
 LASSO via ADMM: fast routine
*/
//' @name  fdaSP.internal
//' @keywords internal
// [[Rcpp::export(.admm_lasso_fast)]]
Rcpp::List admm_lasso_fast(const arma::mat& A, arma::vec& b, arma::vec lambda,
                           bool rho_adaptation, double rho, const double tau, const double mu,
                           const double reltol, const double abstol, const int maxiter, const int ping) {
    
  /* Variable delcaration           */
  double b2, min_mse;
  int n1;
  uword idx_min_mse;
  
  /* Get dimensions             */
  const int m = A.n_rows;
  const int n = A.n_cols;
  
  /* Get number of lambda           */
  const int nlambda = lambda.n_elem;
  
  /* Get output lists          */
  List out, output;

  /* Variable definition            */
  if (m >= n) {
    n1 = n;
  } else {
    n1 = m;
  }
  
  /* Definition of vectors and matrices     */
  arma::vec ATb(n, fill::zeros);
  arma::mat AA(n1, n1, fill::zeros);
  arma::mat ATA(n, n, fill::zeros);
  arma::mat ATA_CHOL_U(n1, n, fill::zeros);
  arma::colvec u(n, fill::zeros);
  arma::colvec z(n, fill::zeros);
  arma::colvec x(n, fill::zeros);
  
  /* Store output    */
  arma::field<vec> h_objval(nlambda);
  arma::field<vec> h_r_norm(nlambda);
  arma::field<vec> h_s_norm(nlambda);
  arma::field<vec> h_eps_pri(nlambda);
  arma::field<vec> h_eps_dual(nlambda);
  arma::field<vec> h_rho(nlambda);
  arma::vec niter(nlambda, fill::zeros);
  arma::vec eltime(nlambda, fill::zeros);
  arma::mat X(nlambda, n, fill::zeros);
  arma::vec conv(nlambda, fill::zeros);
  arma::mat residuals(nlambda, m, fill::zeros);
  arma::mat fitted(nlambda, m, fill::zeros);
  arma::vec mse(nlambda, fill::zeros);
  arma::mat Umat(nlambda, n, fill::zeros);
  arma::mat Zmat(nlambda, n, fill::zeros);
  
  /* Precompute relevant quantities         */
  ATb        = A.t() * b / static_cast<float>(m);
  ATA        = A.t() * A / static_cast<float>(m);
  ATA_CHOL_U = chol_qr_fact(A, m, n);
  if (m >= n) {
    AA = ATA * m;
  } else {
    squaredmat(AA, A, m, n);
  }
  b2 = as_scalar(b.t() * b) / static_cast<float>(m);
    
  /* main loop  */
  for (int j=0; j<nlambda; j++) {
    /* perform linear regression with lasso penalty for the j-th lambda       */
    out = admm_lasso_update_1lambda(A, b, ATA_CHOL_U, AA, ATb, b2, m, n, u, z, lambda(j),
                                    rho_adaptation, rho, tau, mu, reltol, abstol, maxiter, ping);
    
    /* Retrieve output      */
    u = as<arma::colvec>(out["u"]);
    z = as<arma::colvec>(out["z"]);
    x = as<arma::colvec>(out["x"]);

    /* Store output      */
    niter(j)         = out["niter"];
    eltime(j)        = out["eltime"];
    conv(j)          = out["convergence"];
    X.row(j)         = x.t();
    h_objval(j)      = as<arma::vec>(out["objval"]);
    h_r_norm(j)      = as<arma::vec>(out["r_norm"]);
    h_s_norm(j)      = as<arma::vec>(out["s_norm"]);
    h_eps_pri(j)     = as<arma::vec>(out["eps_pri"]);
    h_eps_dual(j)    = as<arma::vec>(out["eps_dual"]);
    h_rho(j)         = (as<arma::vec>(out["rho"])).subvec(0, niter(j)-1);
    rho              = out["rho.updated"];
    residuals.row(j) = (as<arma::vec>(out["residuals"])).t();
    fitted.row(j)    = (as<arma::vec>(out["fitted.values"])).t();
    mse(j)           = out["mse"];
    Umat.row(j)      = u.t();
    Zmat.row(j)      = z.t();
  }
  
  /* Find min MSE in-sample      */
  idx_min_mse = index_min(mse);
  min_mse     = mse(idx_min_mse);
  
  /* Get output         */
  output["coef.path"]    = X;
  output["iternum"]      = niter;               // number of iterations
  output["objfun"]       = h_objval;        // |x|_1
  output["r_norm"]       = h_r_norm;
  output["s_norm"]       = h_s_norm;
  output["err_pri"]      = h_eps_pri;
  output["err_dual"]     = h_eps_dual;
  output["convergence"]  = conv;
  output["elapsedTime"]  = eltime;
  output["rho"]          = h_rho;
  output["lambda"]       = lambda;
  output["min.mse"]      = min_mse;
  output["indi.min.mse"] = idx_min_mse;
  output["lambda.min"]   = lambda[idx_min_mse];
  output["coefficients"] = (X.row(idx_min_mse)).t();
  output["mse"]          = mse;
  output["U"]            = Umat;
  output["Z"]            = Zmat;
  
  /* Return output      */
  return(output);
}

/*
 Group-LASSO via ADMM: fast routine
*/
//' @name  fdaSP.internal
//' @keywords internal
// [[Rcpp::export(.admm_glasso_fast)]]
Rcpp::List admm_glasso_fast(const arma::mat& A, arma::vec& b,
                            arma::mat& groups, arma::vec& group_weights, arma::vec& var_weights,
                            const arma::vec lambda, bool rho_adaptation, double rho, const double tau, const double mu,
                            const double reltol, const double abstol, const int maxiter, const int ping) {

  /* Get dimensions             */
  const int m = A.n_rows;
  const int n = A.n_cols;

  /* Variable delcaration           */
  List out;

  /* Run ADMM            */
  if (m >= n) {
    /* large n problems            */
    out = admm_glasso_large_m_fast(A, b, groups, group_weights, var_weights,
                                   lambda, rho_adaptation, rho, tau, mu,
                                   reltol, abstol, maxiter, ping);
  } else {
    /* large p problems            */
    out = admm_glasso_large_n_fast(A, b, groups, group_weights, var_weights,
                                   lambda, rho_adaptation, rho, tau, mu,
                                   reltol, abstol, maxiter, ping);
  }

  /* Return output      */
  return(out);
}

/*
 Group-LASSO via ADMM: fast routine for large m
 where m denotes the number of observations
*/
Rcpp::List admm_glasso_large_m_fast(const arma::mat& A, arma::vec& b,
                                    arma::mat& groups, arma::vec& group_weights, arma::vec& var_weights,
                                    const arma::vec lambda, bool rho_adaptation, double rho, const double tau, const double mu,
                                    const double reltol, const double abstol, const int maxiter, const int ping) {

  /* Variable delcaration           */
  double b2, min_mse;
  int n1, dim_z;
  uword idx_min_mse;

  /* Get dimensions             */
  const int m = A.n_rows;
  const int n = A.n_cols;
  const int G = groups.n_rows;

  /* Get number of lambda           */
  const int nlambda = lambda.n_elem;

  /* Get output lists          */
  List out, output;

  /* Variable definition            */
  if (m >= n) {
    n1 = n;
  } else {
    n1 = m;
  }

  /* Definition of vectors and matrices     */
  arma::mat ATA(n, n, fill::zeros);
  arma::vec ATb(n, fill::zeros);
  arma::mat ATA_CHOL_U(n1, n, fill::zeros);
  arma::mat FTF(n, n, fill::zeros);
  arma::colvec x(n, fill::zeros);
  arma::colvec Glen(G, fill::zeros);

  /* Store output    */
  arma::field<vec> h_objval(nlambda);
  arma::field<vec> h_r_norm(nlambda);
  arma::field<vec> h_s_norm(nlambda);
  arma::field<vec> h_eps_pri(nlambda);
  arma::field<vec> h_eps_dual(nlambda);
  arma::field<vec> h_rho(nlambda);
  arma::vec niter(nlambda, fill::zeros);
  arma::vec eltime(nlambda, fill::zeros);
  arma::mat X(nlambda, n, fill::zeros);
  arma::vec conv(nlambda, fill::zeros);
  arma::mat residuals(nlambda, m, fill::zeros);
  arma::mat fitted(nlambda, m, fill::zeros);
  arma::vec mse(nlambda, fill::zeros);

  /* Precompute relevant quantities         */
  ATA            = A.t() * A / static_cast<float>(m);
  ATb            = A.t() * b / static_cast<float>(m);
  ATA_CHOL_U     = chol_qr_fact(A, m, n);
  b2             = as_scalar(b.t() * b) / static_cast<float>(m);
  Glen           = sum(groups, 1);
  arma::sp_mat F = glasso_Gmat2Fmat_sparse(groups, group_weights, var_weights);
  FTF            = F.t() * F;
  
  /* Initialize vectors u and z         */
  dim_z = sum(Glen);
  arma::colvec u(dim_z, fill::zeros);
  arma::colvec z(dim_z, fill::zeros);
  arma::mat Umat(nlambda, dim_z, fill::zeros);
  arma::mat Zmat(nlambda, dim_z, fill::zeros);

  /* main loop  */
  for (int j=0; j<nlambda; j++) {
    /* perform linear regression with lasso penalty for the j-th lambda       */
    out = admm_glasso_large_m_update_1lambda(A, b, Glen, F, FTF, ATA, ATA_CHOL_U, ATb,
                                             b2, m, n, G, u, z, lambda(j),
                                             rho_adaptation, rho, tau, mu,
                                             reltol, abstol, maxiter, ping);

    /* Retrieve output      */
    u = as<arma::colvec>(out["u"]);
    z = as<arma::colvec>(out["z"]);
    x = as<arma::colvec>(out["x"]);

    /* Store output      */
    niter(j)         = out["niter"];
    eltime(j)        = out["eltime"];
    conv(j)          = out["convergence"];
    X.row(j)         = x.t();
    h_objval(j)      = as<arma::vec>(out["objval"]);
    h_r_norm(j)      = as<arma::vec>(out["r_norm"]);
    h_s_norm(j)      = as<arma::vec>(out["s_norm"]);
    h_eps_pri(j)     = as<arma::vec>(out["eps_pri"]);
    h_eps_dual(j)    = as<arma::vec>(out["eps_dual"]);
    h_rho(j)         = (as<arma::vec>(out["rho"])).subvec(0, niter(j)-1);
    rho              = out["rho.updated"];
    residuals.row(j) = (as<arma::vec>(out["residuals"])).t();
    fitted.row(j)    = (as<arma::vec>(out["fitted.values"])).t();
    mse(j)           = out["mse"];
    Umat.row(j)      = u.t();
    Zmat.row(j)      = z.t();
  }

  /* Find min MSE in-sample      */
  idx_min_mse = index_min(mse);
  min_mse     = mse(idx_min_mse);

  /* Get output         */
  output["coef.path"]    = X;
  output["iternum"]      = niter;               // number of iterations
  output["objfun"]       = h_objval;        // |x|_1
  output["r_norm"]       = h_r_norm;
  output["s_norm"]       = h_s_norm;
  output["err_pri"]      = h_eps_pri;
  output["err_dual"]     = h_eps_dual;
  output["convergence"]  = conv;
  output["elapsedTime"]  = eltime;
  output["rho"]          = h_rho;
  output["lambda"]       = lambda;
  output["min.mse"]      = min_mse;
  output["indi.min.mse"] = idx_min_mse;
  output["lambda.min"]   = lambda[idx_min_mse];
  output["coefficients"] = (X.row(idx_min_mse)).t();
  output["mse"]          = mse;
  output["U"]            = Umat;
  output["Z"]            = Zmat;
  

  /* Return output      */
  return(output);
}

/*
 Group-LASSO via ADMM: fast routine for large n
 where n denotes the number of parameters
*/
Rcpp::List admm_glasso_large_n_fast(const arma::mat& A, arma::vec& b,
                                    arma::mat& groups, arma::vec& group_weights, arma::vec& var_weights,
                                    const arma::vec lambda, bool rho_adaptation, double rho, const double tau, const double mu,
                                    const double reltol, const double abstol, const int maxiter, const int ping) {
  
  /* Variable delcaration           */
  double b2, min_mse;
  int n1, dim_z;
  uword idx_min_mse;

  /* Get dimensions             */
  const int m = A.n_rows;
  const int n = A.n_cols;
  const int G = groups.n_rows;

  /* Get number of lambda           */
  const int nlambda = lambda.n_elem;

  /* Get output lists          */
  List out, output;

  /* Variable definition            */
  if (m >= n) {
    n1 = n;
  } else {
    n1 = m;
  }

  /* Definition of vectors and matrices     */
  arma::vec ATb(n, fill::zeros);
  arma::mat ATA_CHOL_U(n1, n, fill::zeros);
  arma::colvec x(n, fill::zeros);
  arma::colvec Glen(G, fill::zeros);
  arma::mat FTF(n, n, fill::zeros);
  arma::mat FTF_INV(n, n, fill::zeros);
  arma::mat AFTF_INVA(m, m, fill::zeros);

  /* Store output    */
  arma::field<vec> h_objval(nlambda);
  arma::field<vec> h_r_norm(nlambda);
  arma::field<vec> h_s_norm(nlambda);
  arma::field<vec> h_eps_pri(nlambda);
  arma::field<vec> h_eps_dual(nlambda);
  arma::field<vec> h_rho(nlambda);
  arma::vec niter(nlambda, fill::zeros);
  arma::vec eltime(nlambda, fill::zeros);
  arma::mat X(nlambda, n, fill::zeros);
  arma::vec conv(nlambda, fill::zeros);
  arma::mat residuals(nlambda, m, fill::zeros);
  arma::mat fitted(nlambda, m, fill::zeros);
  arma::vec mse(nlambda, fill::zeros);

  /* Precompute relevant quantities         */
  ATb            = A.t() * b / static_cast<float>(m);
  ATA_CHOL_U     = chol_qr_fact(A, m, n);
  b2             = as_scalar(b.t() * b) / static_cast<float>(m);
  Glen           = sum(groups, 1);
  arma::sp_mat F = glasso_Gmat2Fmat_sparse(groups, group_weights, var_weights);
  FTF            = diagmat(spmat2spmat_mult(F.t(), F));
  FTF_INV        = diagmat(1.0 / diagvec(FTF));
  AFTF_INVA      = A * diagmat(FTF_INV) * A.t();

  /* Initialize vectors u and z         */
  dim_z = sum(Glen);
  arma::colvec u(dim_z, fill::zeros);
  arma::colvec z(dim_z, fill::zeros);
  arma::mat Umat(nlambda, dim_z, fill::zeros);
  arma::mat Zmat(nlambda, dim_z, fill::zeros);

  /* main loop  */
  for (int j=0; j<nlambda; j++) {
    /* perform linear regression with lasso penalty for the j-th lambda       */
    out = admm_glasso_large_n_update_1lambda(A, b, Glen, F,
                                             FTF_INV, AFTF_INVA,
                                             ATA_CHOL_U, ATb,
                                             b2, m, n, G,
                                             u, z, lambda(j),
                                             rho_adaptation, rho, tau, mu,
                                             reltol, abstol, maxiter, ping);

    /* Retrieve output      */
    u = as<arma::colvec>(out["u"]);
    z = as<arma::colvec>(out["z"]);
    x = as<arma::colvec>(out["x"]);

    /* Store output      */
    niter(j)         = out["niter"];
    eltime(j)        = out["eltime"];
    conv(j)          = out["convergence"];
    X.row(j)         = x.t();
    h_objval(j)      = as<arma::vec>(out["objval"]);
    h_r_norm(j)      = as<arma::vec>(out["r_norm"]);
    h_s_norm(j)      = as<arma::vec>(out["s_norm"]);
    h_eps_pri(j)     = as<arma::vec>(out["eps_pri"]);
    h_eps_dual(j)    = as<arma::vec>(out["eps_dual"]);
    h_rho(j)         = (as<arma::vec>(out["rho"])).subvec(0, niter(j)-1);
    rho              = out["rho.updated"];
    residuals.row(j) = (as<arma::vec>(out["residuals"])).t();
    fitted.row(j)    = (as<arma::vec>(out["fitted.values"])).t();
    mse(j)           = out["mse"];
    Umat.row(j)      = u.t();
    Zmat.row(j)      = z.t();
  }

  /* Find min MSE in-sample      */
  idx_min_mse = index_min(mse);
  min_mse     = mse(idx_min_mse);

  /* Get output         */
  output["coef.path"]    = X;
  output["iternum"]      = niter;               // number of iterations
  output["objfun"]       = h_objval;        // |x|_1
  output["r_norm"]       = h_r_norm;
  output["s_norm"]       = h_s_norm;
  output["err_pri"]      = h_eps_pri;
  output["err_dual"]     = h_eps_dual;
  output["convergence"]  = conv;
  output["elapsedTime"]  = eltime;
  output["rho"]          = h_rho;
  output["lambda"]       = lambda;
  output["min.mse"]      = min_mse;
  output["indi.min.mse"] = idx_min_mse;
  output["lambda.min"]   = lambda[idx_min_mse];
  output["coefficients"] = (X.row(idx_min_mse)).t();
  output["U"]            = Umat;
  output["Z"]            = Zmat;

  /* Return output      */
  return(output);
}







// =========================================================
// admm_spglasso
// ADMM for linear regression models with sparse group-LASSO penalty
/*
* SPGLASSO via ADMM FAST VERSION
*/
//' @name  fdaSP.internal
//' @keywords internal
// [[Rcpp::export(.admm_spglasso_fast)]]
Rcpp::List admm_spglasso_fast(const arma::mat& A, arma::vec& b,
                              arma::mat& groups, arma::vec& group_weights,
                              arma::vec& var_weights, arma::vec& var_weights_L1,
                              const arma::vec lambda, const double alpha,
                              bool rho_adaptation, double rho, const double tau, const double mu,
                              const double reltol, const double abstol, const int maxiter, const int ping) {
  
  /* Get dimensions             */
  const int m = A.n_rows;
  const int n = A.n_cols;
  
  /* Variable delcaration           */
  List out;
    
  /* Run ADMM            */
  if (m >= n) {
    /* large n problems            */
    if (alpha == 0) {
      out = admm_glasso_large_m_fast(A, b, groups, group_weights, var_weights,
                                     lambda, rho_adaptation, rho, tau, mu,
                                     reltol, abstol, maxiter, ping);
    } else if (alpha == 1) {
      arma::vec ones_n(n, fill::ones);
      if (sum(var_weights_L1-ones_n) == 0.0) {
        out = admm_lasso_fast(A, b, lambda, rho_adaptation, rho, tau, mu, reltol, abstol, maxiter, ping);
      } else {
        out = admm_adalasso_fast(A, b, var_weights_L1, lambda, rho_adaptation, rho, tau, mu, reltol, abstol, maxiter, ping);
      }
    } else {
      out = admm_spglasso_large_m_fast(A, b, groups, group_weights, var_weights, var_weights_L1,
                                       lambda, alpha, rho_adaptation, rho, tau, mu,
                                       reltol, abstol, maxiter, ping);
    }
  } else {
    /* large p problems            */
    if (alpha == 0) {
      out = admm_glasso_large_n_fast(A, b, groups, group_weights, var_weights,
                                     lambda, rho_adaptation, rho, tau, mu,
                                     reltol, abstol, maxiter, ping);
    } else if (alpha == 1) {
      arma::vec ones_n(n, fill::ones);
      if (sum(var_weights_L1-ones_n) == 0.0) {
        out = admm_lasso_fast(A, b, lambda, rho_adaptation, rho, tau, mu, reltol, abstol, maxiter, ping);
      } else {
        out = admm_adalasso_fast(A, b, var_weights_L1, lambda, rho_adaptation, rho, tau, mu, reltol, abstol, maxiter, ping);
      }
    } else {
      out = admm_spglasso_large_n_fast(A, b, groups, group_weights, var_weights, var_weights_L1,
                                       lambda, alpha, rho_adaptation, rho, tau, mu,
                                       reltol, abstol, maxiter, ping);
    }
  }

  /* Return output      */
  return(out);
}

/*
 Sparse Group-LASSO via ADMM: fast routine for large m
 where m denotes the number of observations
*/
Rcpp::List admm_spglasso_large_m_fast(const arma::mat& A, arma::vec& b,
                                      arma::mat& groups, arma::vec& group_weights,
                                      arma::vec& var_weights, arma::vec& var_weights_L1,
                                      const arma::vec lambda, const double alpha,
                                      bool rho_adaptation, double rho, const double tau, const double mu,
                                      const double reltol, const double abstol, const int maxiter, const int ping) {
    
  /* Variable delcaration           */
  //double rho2;
  double b2, min_mse;
  int n1, dim_z;
  uword idx_min_mse;
  
  /* Get dimensions             */
  const int m = A.n_rows;
  const int n = A.n_cols;
  const int G = groups.n_rows;
  
  /* Get number of lambda           */
  const int nlambda = lambda.n_elem;

  /* Get output lists          */
  List out, output;
  
  /* Variable definition            */
  if (m >= n) {
    n1 = n;
  } else {
    n1 = m;
  }

  /* Definition of vectors and matrices     */
  arma::mat ATA(n, n, fill::zeros);
  arma::vec ATb(n, fill::zeros);
  arma::mat ATA_CHOL_U(n1, n, fill::zeros);
  arma::mat FTF(n, n, fill::zeros);
  arma::colvec x(n, fill::zeros);
  arma::colvec Glen(G, fill::zeros);

  /* Store output    */
  arma::field<vec> h_objval(nlambda);
  arma::field<vec> h_r_norm(nlambda);
  arma::field<vec> h_s_norm(nlambda);
  arma::field<vec> h_eps_pri(nlambda);
  arma::field<vec> h_eps_dual(nlambda);
  arma::field<vec> h_rho(nlambda);
  arma::vec niter(nlambda, fill::zeros);
  arma::vec eltime(nlambda, fill::zeros);
  arma::mat X(nlambda, n, fill::zeros);
  arma::vec conv(nlambda, fill::zeros);
  arma::mat residuals(nlambda, m, fill::zeros);
  arma::mat fitted(nlambda, m, fill::zeros);
  arma::vec mse(nlambda, fill::zeros);

  /* Precompute relevant quantities         */
  ATA            = A.t() * A / static_cast<float>(m);
  ATb            = A.t() * b / static_cast<float>(m);
  ATA_CHOL_U     = chol_qr_fact(A, m, n);
  b2             = as_scalar(b.t() * b) / static_cast<float>(m);
  Glen           = sum(groups, 1);
  arma::sp_mat F = spglasso_Gmat2Fmat_sparse(groups, group_weights, var_weights, var_weights_L1);
  FTF            = diagmat(spmat2spmat_mult(F.t(), F));
  
  /* Initialize vectors u and z         */
  dim_z = n + sum(Glen);
  arma::colvec u(dim_z, fill::zeros);
  arma::colvec z(dim_z, fill::zeros);
  arma::mat Umat(nlambda, dim_z, fill::zeros);
  arma::mat Zmat(nlambda, dim_z, fill::zeros);

  /* main loop  */
  for (int j=0; j<nlambda; j++) {
    /* perform linear regression with lasso penalty for the j-th lambda       */
    out = admm_spglasso_large_m_update_1lambda(A, b, Glen, F, FTF, ATA, ATA_CHOL_U, ATb, b2, m, n, G,
                                               u, z, lambda(j), alpha, rho_adaptation, rho, tau, mu,
                                               reltol, abstol, maxiter, ping);

    /* Retrieve output      */
    u = as<arma::colvec>(out["u"]);
    z = as<arma::colvec>(out["z"]);
    x = as<arma::colvec>(out["x"]);

    /* Store output      */
    niter(j)         = out["niter"];
    eltime(j)        = out["eltime"];
    conv(j)          = out["convergence"];
    X.row(j)         = x.t();
    h_objval(j)      = as<arma::vec>(out["objval"]);
    h_r_norm(j)      = as<arma::vec>(out["r_norm"]);
    h_s_norm(j)      = as<arma::vec>(out["s_norm"]);
    h_eps_pri(j)     = as<arma::vec>(out["eps_pri"]);
    h_eps_dual(j)    = as<arma::vec>(out["eps_dual"]);
    h_rho(j)         = (as<arma::vec>(out["rho"])).subvec(0, niter(j)-1);
    rho              = out["rho.updated"];
    residuals.row(j) = (as<arma::vec>(out["residuals"])).t();
    fitted.row(j)    = (as<arma::vec>(out["fitted.values"])).t();
    mse(j)           = out["mse"];
    Umat.row(j)      = u.t();
    Zmat.row(j)      = z.t();
  }

  /* Find min MSE in-sample      */
  idx_min_mse = index_min(mse);
  min_mse     = mse(idx_min_mse);

  /* Get output         */
  output["coef.path"]    = X;
  output["iternum"]      = niter;               // number of iterations
  output["objfun"]       = h_objval;        // |x|_1
  output["r_norm"]       = h_r_norm;
  output["s_norm"]       = h_s_norm;
  output["err_pri"]      = h_eps_pri;
  output["err_dual"]     = h_eps_dual;
  output["convergence"]  = conv;
  output["elapsedTime"]  = eltime;
  output["rho"]          = h_rho;
  output["lambda"]       = lambda;
  output["min.mse"]      = min_mse;
  output["indi.min.mse"] = idx_min_mse;
  output["lambda.min"]   = lambda[idx_min_mse];
  output["coefficients"] = (X.row(idx_min_mse)).t();
  output["mse"]          = mse;
  output["U"]            = Umat;
  output["Z"]            = Zmat;

  /* Return output      */
  return(output);
}

/*
 Group-LASSO via ADMM: fast routine for large n
 where n denotes the number of parameters
*/
Rcpp::List admm_spglasso_large_n_fast(const arma::mat& A, arma::vec& b,
                                      arma::mat& groups, arma::vec& group_weights, arma::vec& var_weights, arma::vec& var_weights_L1,
                                      const arma::vec lambda, const double alpha, bool rho_adaptation, double rho, const double tau, const double mu,
                                      const double reltol, const double abstol, const int maxiter, const int ping) {

  /* Variable delcaration           */
  double b2, min_mse;
  int n1, dim_z;
  uword idx_min_mse;

  /* Get dimensions             */
  const int m = A.n_rows;
  const int n = A.n_cols;
  const int G = groups.n_rows;

  /* Get number of lambda           */
  const int nlambda = lambda.n_elem;

  /* Get output lists          */
  List out, output;

  /* Variable definition            */
  if (m >= n) {
    n1 = n;
  } else {
    n1 = m;
  }

  /* Definition of vectors and matrices     */
  arma::vec ATb(n, fill::zeros);
  arma::mat ATA_CHOL_U(n1, n, fill::zeros);
  arma::mat FTF(n, n, fill::zeros);
  arma::colvec x(n, fill::zeros);
  arma::colvec Glen(G, fill::zeros);
  arma::mat FTF_INV(n, n, fill::zeros);
  arma::mat AFTF_INVA(m, m, fill::zeros);

  /* Store output    */
  arma::field<vec> h_objval(nlambda);
  arma::field<vec> h_r_norm(nlambda);
  arma::field<vec> h_s_norm(nlambda);
  arma::field<vec> h_eps_pri(nlambda);
  arma::field<vec> h_eps_dual(nlambda);
  arma::field<vec> h_rho(nlambda);
  arma::vec niter(nlambda, fill::zeros);
  arma::vec eltime(nlambda, fill::zeros);
  arma::mat X(nlambda, n, fill::zeros);
  arma::vec conv(nlambda, fill::zeros);
  arma::mat residuals(nlambda, m, fill::zeros);
  arma::mat fitted(nlambda, m, fill::zeros);
  arma::vec mse(nlambda, fill::zeros);

  /* Precompute relevant quantities         */
  ATb            = A.t() * b / static_cast<float>(m);
  ATA_CHOL_U     = chol_qr_fact(A, m, n);
  b2             = as_scalar(b.t() * b) / static_cast<float>(m);
  Glen           = sum(groups, 1);
  arma::sp_mat F = spglasso_Gmat2Fmat_sparse(groups, group_weights, var_weights, var_weights_L1);
  FTF            = diagmat(spmat2spmat_mult(F.t(), F));
  FTF_INV        = diagmat(1.0 / diagvec(FTF));
  AFTF_INVA      = A * diagmat(FTF_INV) * A.t();
  
  /* Initialize vectors u and z         */
  dim_z = n + sum(Glen);
  arma::colvec u(dim_z, fill::zeros);
  arma::colvec z(dim_z, fill::zeros);
  arma::mat Umat(nlambda, dim_z, fill::zeros);
  arma::mat Zmat(nlambda, dim_z, fill::zeros);

  /* main loop  */
  for (int j=0; j<nlambda; j++) {
    /* perform linear regression with lasso penalty for the j-th lambda       */
    out = admm_spglasso_large_n_update_1lambda(A, b, Glen, F, FTF_INV, AFTF_INVA,
                                               ATA_CHOL_U, ATb, b2, m, n, G, u, z, lambda(j), alpha,
                                               rho_adaptation,  rho,  tau,  mu, reltol,  abstol,  maxiter,  ping);

    /* Retrieve output      */
    u = as<arma::colvec>(out["u"]);
    z = as<arma::colvec>(out["z"]);
    x = as<arma::colvec>(out["x"]);

    /* Store output      */
    niter(j)         = out["niter"];
    eltime(j)        = out["eltime"];
    conv(j)          = out["convergence"];
    X.row(j)         = x.t();
    h_objval(j)      = as<arma::vec>(out["objval"]);
    h_r_norm(j)      = as<arma::vec>(out["r_norm"]);
    h_s_norm(j)      = as<arma::vec>(out["s_norm"]);
    h_eps_pri(j)     = as<arma::vec>(out["eps_pri"]);
    h_eps_dual(j)    = as<arma::vec>(out["eps_dual"]);
    h_rho(j)         = (as<arma::vec>(out["rho"])).subvec(0, niter(j)-1);
    rho              = out["rho.updated"];
    residuals.row(j) = (as<arma::vec>(out["residuals"])).t();
    fitted.row(j)    = (as<arma::vec>(out["fitted.values"])).t();
    mse(j)           = out["mse"];
    Umat.row(j)      = u.t();
    Zmat.row(j)      = z.t();
  }

  /* Find min MSE in-sample      */
  idx_min_mse = index_min(mse);
  min_mse     = mse(idx_min_mse);

  /* Get output         */
  output["coef.path"]    = X;
  output["iternum"]      = niter;               // number of iterations
  output["objfun"]       = h_objval;        // |x|_1
  output["r_norm"]       = h_r_norm;
  output["s_norm"]       = h_s_norm;
  output["err_pri"]      = h_eps_pri;
  output["err_dual"]     = h_eps_dual;
  output["convergence"]  = conv;
  output["elapsedTime"]  = eltime;
  output["rho"]          = h_rho;
  output["lambda"]       = lambda;
  output["min.mse"]      = min_mse;
  output["indi.min.mse"] = idx_min_mse;
  output["lambda.min"]   = lambda[idx_min_mse];
  output["coefficients"] = (X.row(idx_min_mse)).t();
  output["mse"]          = mse;
  output["U"]            = Umat;
  output["Z"]            = Zmat;

  /* Return output      */
  return(output);
}

/*
 Group-LASSO via ADMM: fast routine
*/
//' @name  fdaSP.internal
//' @keywords internal
// [[Rcpp::export(.admm_ovglasso_fast)]]
Rcpp::List admm_ovglasso_fast(const arma::mat& A, arma::vec& b,
                              arma::mat& groups, arma::vec& group_weights, arma::vec& var_weights,
                              const arma::vec lambda, bool rho_adaptation, double rho, const double tau, const double mu,
                              const double reltol, const double abstol, const int maxiter, const int ping) {

  /* Get dimensions             */
  const int m = A.n_rows;
  const int n = A.n_cols;

  /* Variable delcaration           */
  List out;

  /* Run ADMM            */
  if (m >= n) {
    /* large n problems            */
    out = admm_ovglasso_large_m_fast(A, b, groups, group_weights, var_weights,
                                     lambda, rho_adaptation, rho, tau, mu,
                                     reltol, abstol, maxiter, ping);
  } else {
    /* large p problems            */
    out = admm_ovglasso_large_n_fast(A, b, groups, group_weights, var_weights,
                                     lambda, rho_adaptation, rho, tau, mu,
                                     reltol, abstol, maxiter, ping);
  }

  /* Return output      */
  return(out);
}

/*
 Group-LASSO via ADMM: fast routine for large m
 where m denotes the number of observations
*/
Rcpp::List admm_ovglasso_large_m_fast(const arma::mat& A, arma::vec& b,
                                      arma::mat& groups, arma::vec& group_weights, arma::vec& var_weights,
                                      const arma::vec lambda, bool rho_adaptation, double rho, const double tau, const double mu,
                                      const double reltol, const double abstol, const int maxiter, const int ping) {

  /* Variable delcaration           */
  double b2, min_mse;
  int n1, dim_z;
  uword idx_min_mse;

  /* Get dimensions             */
  const int m = A.n_rows;
  const int n = A.n_cols;
  const int G = groups.n_rows;

  /* Get number of lambda           */
  const int nlambda = lambda.n_elem;

  /* Get output lists          */
  List out, output;

  /* Variable definition            */
  if (m >= n) {
    n1 = n;
  } else {
    n1 = m;
  }

  /* Definition of vectors and matrices     */
  arma::mat ATA(n, n, fill::zeros);
  arma::vec ATb(n, fill::zeros);
  arma::mat ATA_CHOL_U(n1, n, fill::zeros);
  arma::mat FTF(n, n, fill::zeros);
  arma::colvec x(n, fill::zeros);
  arma::colvec Glen(G, fill::zeros);

  /* Store output    */
  arma::field<vec> h_objval(nlambda);
  arma::field<vec> h_r_norm(nlambda);
  arma::field<vec> h_s_norm(nlambda);
  arma::field<vec> h_eps_pri(nlambda);
  arma::field<vec> h_eps_dual(nlambda);
  arma::field<vec> h_rho(nlambda);
  arma::vec niter(nlambda, fill::zeros);
  arma::vec eltime(nlambda, fill::zeros);
  arma::mat X(nlambda, n, fill::zeros);
  arma::vec conv(nlambda, fill::zeros);
  arma::mat residuals(nlambda, m, fill::zeros);
  arma::mat fitted(nlambda, m, fill::zeros);
  arma::vec mse(nlambda, fill::zeros);
  
  /* Precompute relevant quantities         */
  ATA            = A.t() * A / static_cast<float>(m);
  ATb            = A.t() * b / static_cast<float>(m);
  ATA_CHOL_U     = chol_qr_fact(A, m, n);
  b2             = as_scalar(b.t() * b) / static_cast<float>(m);
  Glen           = sum(groups, 1);
  arma::sp_mat F = glasso_Gmat2Fmat_sparse(groups, group_weights, var_weights);
  FTF            = F.t() * F;
    
  /* Initialize vectors u and z         */
  dim_z = sum(Glen);
  arma::colvec u(dim_z, fill::zeros);
  arma::colvec z(dim_z, fill::zeros);
  arma::mat Umat(nlambda, dim_z, fill::zeros);
  arma::mat Zmat(nlambda, dim_z, fill::zeros);

  /* main loop  */
  for (int j=0; j<nlambda; j++) {
    /* perform linear regression with lasso penalty for the j-th lambda       */
    out = admm_ovglasso_large_m_update_1lambda(A, b, Glen, F, FTF, ATA, ATA_CHOL_U, ATb,
                                               b2, m, n, G, u, z, lambda(j),
                                               rho_adaptation, rho, tau, mu,
                                               reltol, abstol, maxiter, ping);

    /* Retrieve output      */
    u = as<arma::colvec>(out["u"]);
    z = as<arma::colvec>(out["z"]);
    x = as<arma::colvec>(out["x"]);

    /* Store output      */
    niter(j)         = out["niter"];
    eltime(j)        = out["eltime"];
    conv(j)          = out["convergence"];
    X.row(j)         = x.t();
    h_objval(j)      = as<arma::vec>(out["objval"]);
    h_r_norm(j)      = as<arma::vec>(out["r_norm"]);
    h_s_norm(j)      = as<arma::vec>(out["s_norm"]);
    h_eps_pri(j)     = as<arma::vec>(out["eps_pri"]);
    h_eps_dual(j)    = as<arma::vec>(out["eps_dual"]);
    h_rho(j)         = (as<arma::vec>(out["rho"])).subvec(0, niter(j)-1);
    rho              = out["rho.updated"];
    residuals.row(j) = (as<arma::vec>(out["residuals"])).t();
    fitted.row(j)    = (as<arma::vec>(out["fitted.values"])).t();
    mse(j)           = out["mse"];
    Umat.row(j)      = u.t();
    Zmat.row(j)      = z.t();
  }

  /* Find min MSE in-sample      */
  idx_min_mse = index_min(mse);
  min_mse     = mse(idx_min_mse);

  /* Get output         */
  output["coef.path"]    = X;
  output["iternum"]      = niter;               // number of iterations
  output["objfun"]       = h_objval;        // |x|_1
  output["r_norm"]       = h_r_norm;
  output["s_norm"]       = h_s_norm;
  output["err_pri"]      = h_eps_pri;
  output["err_dual"]     = h_eps_dual;
  output["convergence"]  = conv;
  output["elapsedTime"]  = eltime;
  output["rho"]          = h_rho;
  output["lambda"]       = lambda;
  output["min.mse"]      = min_mse;
  output["indi.min.mse"] = idx_min_mse;
  output["lambda.min"]   = lambda[idx_min_mse];
  output["coefficients"] = (X.row(idx_min_mse)).t();
  output["mse"]          = mse;
  output["U"]            = Umat;
  output["Z"]            = Zmat;

  /* Return output      */
  return(output);
}

/*
 Group-LASSO via ADMM: fast routine for large n
 where n denotes the number of parameters
*/
Rcpp::List admm_ovglasso_large_n_fast(const arma::mat& A, arma::vec& b,
                                      arma::mat& groups, arma::vec& group_weights, arma::vec& var_weights,
                                      const arma::vec lambda, bool rho_adaptation, double rho, const double tau, const double mu,
                                      const double reltol, const double abstol, const int maxiter, const int ping) {

  /* Variable delcaration           */
  double b2, min_mse;
  int n1, dim_z;
  uword idx_min_mse;

  /* Get dimensions             */
  const int m = A.n_rows;
  const int n = A.n_cols;
  const int G = groups.n_rows;

  /* Get number of lambda           */
  const int nlambda = lambda.n_elem;

  /* Get output lists          */
  List out, output;

  /* Variable definition            */
  if (m >= n) {
    n1 = n;
  } else {
    n1 = m;
  }

  /* Definition of vectors and matrices     */
  arma::vec ATb(n, fill::zeros);
  arma::mat ATA_CHOL_U(n1, n, fill::zeros);
  arma::mat FTF(n, n, fill::zeros);
  arma::colvec x(n, fill::zeros);
  arma::colvec Glen(G, fill::zeros);
  arma::mat FTF_INV(n, n, fill::zeros);
  arma::mat AFTF_INVA(m, m, fill::zeros);

  /* Store output    */
  arma::field<vec> h_objval(nlambda);
  arma::field<vec> h_r_norm(nlambda);
  arma::field<vec> h_s_norm(nlambda);
  arma::field<vec> h_eps_pri(nlambda);
  arma::field<vec> h_eps_dual(nlambda);
  arma::field<vec> h_rho(nlambda);
  arma::vec niter(nlambda, fill::zeros);
  arma::vec eltime(nlambda, fill::zeros);
  arma::mat X(nlambda, n, fill::zeros);
  arma::vec conv(nlambda, fill::zeros);
  arma::mat residuals(nlambda, m, fill::zeros);
  arma::mat fitted(nlambda, m, fill::zeros);
  arma::vec mse(nlambda, fill::zeros);

  /* Precompute relevant quantities         */
  ATb            = A.t() * b / static_cast<float>(m);
  ATA_CHOL_U     = chol_qr_fact(A, m, n);
  b2             = as_scalar(b.t() * b) / static_cast<float>(m);
  Glen           = sum(groups, 1);
  arma::sp_mat F = glasso_Gmat2Fmat_sparse(groups, group_weights, var_weights);
  FTF            = diagmat(spmat2spmat_mult(F.t(), F));
  FTF_INV        = diagmat(1.0 / diagvec(FTF));
  AFTF_INVA      = A * diagmat(FTF_INV) * A.t();

  /* Initialize vectors u and z         */
  dim_z = sum(Glen);
  arma::colvec u(dim_z, fill::zeros);
  arma::colvec z(dim_z, fill::zeros);
  arma::mat Umat(nlambda, dim_z, fill::zeros);
  arma::mat Zmat(nlambda, dim_z, fill::zeros);

  /* main loop  */
  for (int j=0; j<nlambda; j++) {
    /* perform linear regression with lasso penalty for the j-th lambda       */
    out = admm_ovglasso_large_n_update_1lambda(A, b, Glen, F,
                                               FTF_INV, AFTF_INVA,
                                               ATA_CHOL_U, ATb,
                                               b2, m, n, G,
                                               u, z, lambda(j),
                                               rho_adaptation, rho, tau, mu,
                                               reltol, abstol, maxiter, ping);

    /* Retrieve output      */
    u = as<arma::colvec>(out["u"]);
    z = as<arma::colvec>(out["z"]);
    x = as<arma::colvec>(out["x"]);

    /* Store output      */
    niter(j)         = out["niter"];
    eltime(j)        = out["eltime"];
    conv(j)          = out["convergence"];
    X.row(j)         = x.t();
    h_objval(j)      = as<arma::vec>(out["objval"]);
    h_r_norm(j)      = as<arma::vec>(out["r_norm"]);
    h_s_norm(j)      = as<arma::vec>(out["s_norm"]);
    h_eps_pri(j)     = as<arma::vec>(out["eps_pri"]);
    h_eps_dual(j)    = as<arma::vec>(out["eps_dual"]);
    h_rho(j)         = (as<arma::vec>(out["rho"])).subvec(0, niter(j)-1);
    rho              = out["rho.updated"];
    residuals.row(j) = (as<arma::vec>(out["residuals"])).t();
    fitted.row(j)    = (as<arma::vec>(out["fitted.values"])).t();
    mse(j)           = out["mse"];
    Umat.row(j)      = u.t();
    Zmat.row(j)      = z.t();
  }

  /* Find min MSE in-sample      */
  idx_min_mse = index_min(mse);
  min_mse     = mse(idx_min_mse);

  /* Get output         */
  output["coef.path"]    = X;
  output["iternum"]      = niter;               // number of iterations
  output["objfun"]       = h_objval;        // |x|_1
  output["r_norm"]       = h_r_norm;
  output["s_norm"]       = h_s_norm;
  output["err_pri"]      = h_eps_pri;
  output["err_dual"]     = h_eps_dual;
  output["convergence"]  = conv;
  output["elapsedTime"]  = eltime;
  output["rho"]          = h_rho;
  output["lambda"]       = lambda;
  output["min.mse"]      = min_mse;
  output["indi.min.mse"] = idx_min_mse;
  output["lambda.min"]   = lambda[idx_min_mse];
  output["coefficients"] = (X.row(idx_min_mse)).t();
  output["mse"]          = mse;
  output["U"]            = Umat;
  output["Z"]            = Zmat;

  /* Return output      */
  return(output);
}

/*
 Adaptive-LASSO via ADMM: fast routine
*/
//' @name  fdaSP.internal
//' @keywords internal
// [[Rcpp::export(.admm_adalasso_fast)]]
Rcpp::List admm_adalasso_fast(const arma::mat& A, arma::vec& b,
                              arma::vec& var_weights, const arma::vec lambda,
                              bool rho_adaptation, double rho, const double tau, const double mu,
                              const double reltol, const double abstol, const int maxiter, const int ping) {

  /* Get dimensions             */
  const int m = A.n_rows;
  const int n = A.n_cols;

  /* Variable delcaration           */
  List out;

  /* Run ADMM            */
  if (m >= n) {
    /* large n problems            */
    out = admm_adalasso_large_m_fast(A, b, var_weights, lambda,
                                     rho_adaptation, rho, tau, mu,
                                     reltol, abstol, maxiter, ping);
  } else {
    /* large p problems            */
    out = admm_adalasso_large_n_fast(A, b, var_weights, lambda,
                                     rho_adaptation, rho, tau, mu,
                                     reltol, abstol, maxiter, ping);
  }

  /* Return output      */
  return(out);
}

/*
 Adaptive-LASSO via ADMM: fast routine for large m
 where m denotes the number of observations
*/
Rcpp::List admm_adalasso_large_m_fast(const arma::mat& A, arma::vec& b, arma::vec& var_weights, const arma::vec lambda,
                                      bool rho_adaptation, double rho, const double tau, const double mu,
                                      const double reltol, const double abstol, const int maxiter, const int ping) {

  /* Variable delcaration           */
  double b2, min_mse;
  int n1;
  uword idx_min_mse;

  /* Get dimensions             */
  const int m = A.n_rows;
  const int n = A.n_cols;
  
  /* Get number of lambda           */
  const int nlambda = lambda.n_elem;

  /* Get output lists          */
  List out, output;

  /* Variable definition            */
  if (m >= n) {
    n1 = n;
  } else {
    n1 = m;
  }

  /* Definition of vectors and matrices     */
  arma::mat ATA(n, n, fill::zeros);
  arma::vec ATb(n, fill::zeros);
  arma::mat ATA_CHOL_U(n1, n, fill::zeros);
  arma::mat F(n, n, fill::zeros);
  arma::mat FTF(n, n, fill::zeros);
  arma::colvec x(n, fill::zeros);
  arma::mat Umat(nlambda, n, fill::zeros);
  arma::mat Zmat(nlambda, n, fill::zeros);
  
  /* Store output    */
  arma::field<vec> h_objval(nlambda);
  arma::field<vec> h_r_norm(nlambda);
  arma::field<vec> h_s_norm(nlambda);
  arma::field<vec> h_eps_pri(nlambda);
  arma::field<vec> h_eps_dual(nlambda);
  arma::field<vec> h_rho(nlambda);
  arma::vec niter(nlambda, fill::zeros);
  arma::vec eltime(nlambda, fill::zeros);
  arma::mat X(nlambda, n, fill::zeros);
  arma::vec conv(nlambda, fill::zeros);
  arma::mat residuals(nlambda, m, fill::zeros);
  arma::mat fitted(nlambda, m, fill::zeros);
  arma::vec mse(nlambda, fill::zeros);

  /* Precompute relevant quantities         */
  ATA        = A.t() * A / static_cast<float>(m);
  ATb        = A.t() * b / static_cast<float>(m);
  ATA_CHOL_U = chol_qr_fact(A, m, n);
  b2         = as_scalar(b.t() * b) / static_cast<float>(m);
  F          = adalasso_Fmat(var_weights);
  FTF        = diagmat(F) * diagmat(F);
  
  /* Initialize vectors u and z         */
  arma::colvec u(n, fill::zeros);
  arma::colvec z(n, fill::zeros);
  
  /* main loop  */
  for (int j=0; j<nlambda; j++) {
    /* perform linear regression with lasso penalty for the j-th lambda       */
    out = admm_adalasso_large_m_update_1lambda(A, b, F, FTF, ATA, ATA_CHOL_U, ATb, b2, m, n,
                                               u, z,  lambda(j), rho_adaptation, rho, tau, mu,
                                               reltol, abstol, maxiter, ping);

    /* Retrieve output      */
    u = as<arma::colvec>(out["u"]);
    z = as<arma::colvec>(out["z"]);
    x = as<arma::colvec>(out["x"]);

    /* Store output      */
    niter(j)         = out["niter"];
    eltime(j)        = out["eltime"];
    conv(j)          = out["convergence"];
    X.row(j)         = x.t();
    h_objval(j)      = as<arma::vec>(out["objval"]);
    h_r_norm(j)      = as<arma::vec>(out["r_norm"]);
    h_s_norm(j)      = as<arma::vec>(out["s_norm"]);
    h_eps_pri(j)     = as<arma::vec>(out["eps_pri"]);
    h_eps_dual(j)    = as<arma::vec>(out["eps_dual"]);
    h_rho(j)         = (as<arma::vec>(out["rho"])).subvec(0, niter(j)-1);
    rho              = out["rho.updated"];
    residuals.row(j) = (as<arma::vec>(out["residuals"])).t();
    fitted.row(j)    = (as<arma::vec>(out["fitted.values"])).t();
    mse(j)           = out["mse"];
    Umat.row(j)      = u.t();
    Zmat.row(j)      = z.t();
  }

  /* Find min MSE in-sample      */
  idx_min_mse = index_min(mse);
  min_mse     = mse(idx_min_mse);

  /* Get output         */
  output["coef.path"]    = X;
  output["iternum"]      = niter;               // number of iterations
  output["objfun"]       = h_objval;        // |x|_1
  output["r_norm"]       = h_r_norm;
  output["s_norm"]       = h_s_norm;
  output["err_pri"]      = h_eps_pri;
  output["err_dual"]     = h_eps_dual;
  output["convergence"]  = conv;
  output["elapsedTime"]  = eltime;
  output["rho"]          = h_rho;
  output["lambda"]       = lambda;
  output["min.mse"]      = min_mse;
  output["indi.min.mse"] = idx_min_mse;
  output["lambda.min"]   = lambda[idx_min_mse];
  output["coefficients"] = (X.row(idx_min_mse)).t();
  output["mse"]          = mse;
  output["U"]            = Umat;
  output["Z"]            = Zmat;

  /* Return output      */
  return(output);
}

/*
 Adaptive-LASSO via ADMM: fast routine for large n
 where n denotes the number of parameters
*/
Rcpp::List admm_adalasso_large_n_fast(const arma::mat& A, arma::vec& b, arma::vec& var_weights, const arma::vec lambda,
                                      bool rho_adaptation, double rho, const double tau, const double mu,
                                      const double reltol, const double abstol, const int maxiter, const int ping) {

  /* Variable delcaration           */
  double b2, min_mse;
  int n1;
  uword idx_min_mse;

  /* Get dimensions             */
  const int m = A.n_rows;
  const int n = A.n_cols;

  /* Get number of lambda           */
  const int nlambda = lambda.n_elem;

  /* Get output lists          */
  List out, output;

  /* Variable definition            */
  if (m >= n) {
    n1 = n;
  } else {
    n1 = m;
  }

  /* Definition of vectors and matrices     */
  arma::vec ATb(n, fill::zeros);
  arma::mat ATA_CHOL_U(n1, n, fill::zeros);
  arma::mat F(n, n, fill::zeros);
  arma::mat FTF(n, n, fill::zeros);
  arma::colvec x(n, fill::zeros);
  arma::mat FTF_INV(n, n, fill::zeros);
  arma::mat AFTF_INVA(m, m, fill::zeros);

  /* Store output    */
  arma::field<vec> h_objval(nlambda);
  arma::field<vec> h_r_norm(nlambda);
  arma::field<vec> h_s_norm(nlambda);
  arma::field<vec> h_eps_pri(nlambda);
  arma::field<vec> h_eps_dual(nlambda);
  arma::field<vec> h_rho(nlambda);
  arma::vec niter(nlambda, fill::zeros);
  arma::vec eltime(nlambda, fill::zeros);
  arma::mat X(nlambda, n, fill::zeros);
  arma::vec conv(nlambda, fill::zeros);
  arma::mat residuals(nlambda, m, fill::zeros);
  arma::mat fitted(nlambda, m, fill::zeros);
  arma::vec mse(nlambda, fill::zeros);
  arma::mat Umat(nlambda, n, fill::zeros);
  arma::mat Zmat(nlambda, n, fill::zeros);

  /* Precompute relevant quantities         */
  ATb        = A.t() * b / static_cast<float>(m);
  ATA_CHOL_U = chol_qr_fact(A, m, n);
  b2         = as_scalar(b.t() * b) / static_cast<float>(m);
  F          = adalasso_Fmat(var_weights);
  FTF        = diagmat(F) * diagmat(F);
  FTF_INV    = inv(diagmat(FTF));
  AFTF_INVA  = A * diagmat(FTF_INV) * A.t();
  
  /* Initialize vectors u and z         */
  arma::colvec u(n, fill::zeros);
  arma::colvec z(n, fill::zeros);

  /* main loop  */
  for (int j=0; j<nlambda; j++) {
    /* perform linear regression with lasso penalty for the j-th lambda       */
    out = admm_adalasso_large_n_update_1lambda(A, b, F, FTF_INV, AFTF_INVA, ATA_CHOL_U,
                                               ATb, b2, m, n, u, z, lambda(j),
                                               rho_adaptation, rho, tau, mu,
                                               reltol, abstol, maxiter, ping);

    /* Retrieve output      */
    u = as<arma::colvec>(out["u"]);
    z = as<arma::colvec>(out["z"]);
    x = as<arma::colvec>(out["x"]);

    /* Store output      */
    niter(j)         = out["niter"];
    eltime(j)        = out["eltime"];
    conv(j)          = out["convergence"];
    X.row(j)         = x.t();
    h_objval(j)      = as<arma::vec>(out["objval"]);
    h_r_norm(j)      = as<arma::vec>(out["r_norm"]);
    h_s_norm(j)      = as<arma::vec>(out["s_norm"]);
    h_eps_pri(j)     = as<arma::vec>(out["eps_pri"]);
    h_eps_dual(j)    = as<arma::vec>(out["eps_dual"]);
    h_rho(j)         = (as<arma::vec>(out["rho"])).subvec(0, niter(j)-1);
    rho              = out["rho.updated"];
    residuals.row(j) = (as<arma::vec>(out["residuals"])).t();
    fitted.row(j)    = (as<arma::vec>(out["fitted.values"])).t();
    mse(j)           = out["mse"];
    Umat.row(j)      = u.t();
    Zmat.row(j)      = z.t();
  }

  /* Find min MSE in-sample      */
  idx_min_mse = index_min(mse);
  min_mse     = mse(idx_min_mse);

  /* Get output         */
  output["coef.path"]    = X;
  output["iternum"]      = niter;               // number of iterations
  output["objfun"]       = h_objval;        // |x|_1
  output["r_norm"]       = h_r_norm;
  output["s_norm"]       = h_s_norm;
  output["err_pri"]      = h_eps_pri;
  output["err_dual"]     = h_eps_dual;
  output["convergence"]  = conv;
  output["elapsedTime"]  = eltime;
  output["rho"]          = h_rho;
  output["lambda"]       = lambda;
  output["min.mse"]      = min_mse;
  output["indi.min.mse"] = idx_min_mse;
  output["lambda.min"]   = lambda[idx_min_mse];
  output["coefficients"] = (X.row(idx_min_mse)).t();
  output["mse"]          = mse;
  output["U"]            = Umat;
  output["Z"]            = Zmat;

  /* Return output      */
  return(output);
}





