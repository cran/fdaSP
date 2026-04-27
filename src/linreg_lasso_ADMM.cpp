// Routines for:
//      1. admm_lasso
//      2. admm_lasso_cov
//      3. admm_adalasso
//      3. admm_enet
//      4. admm_genlasso
//
//      ADMM for linear regression models with
//          overlap group-LASSO penalty with non penalized covariates

// Authors:
//           Bernardi Mauro, University of Padova
//           Last update: July 12, 2022

// List of implemented methods:



// [[Rcpp::depends(RcppArmadillo)]]
#include "linreg_lasso_ADMM.h"

/*  
    LASSO via ADMM
                                            */

// admm_enet: Linear regression model with ENET penalty
// Linear regression model with ENET penalty: evaluate the objective function
Rcpp::List admm_enet(const arma::mat& A,
                     const arma::colvec& b,
                     const double lambda,
                     const double alpha,
                     const double reltol,
                     const double abstol,
                     const int maxiter,
                     const double rho) {
  
  /* Variables declaration       */
  double gamma = 0.0, sqrtn = 0.0;
  int k = 0, n = 0;

  /* Get dimensions             */
  n = A.n_cols;
  
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

// admm_genlasso: Linear regression model with generalized LASSO penalty
// Linear regression model with generalized LASSO penalty: evaluate the objective function
Rcpp::List admm_genlasso(const arma::mat& A,
                         const arma::colvec& b,
                         const arma::mat& D,
                         const double lambda,
                         const double reltol,
                         const double abstol,
                         const int maxiter,
                         const double rho) {
  // 1. get parameters
  const int n = A.n_cols;
  int k = 0;
  double sqrtn = std::sqrt(static_cast<float>(n));
  
  // 2. set ready
  arma::colvec x(n, fill::randn); x/=10.0;
  arma::colvec z(D * x);
  arma::colvec u(D * x - z);
  arma::colvec q(n, fill::zeros);
  arma::colvec zold(z);
  arma::colvec x_hat(n, fill::zeros);

  // 3. precompute static variables for x-update and factorization
  arma::mat Atb = A.t() * b;
  arma::mat U   = genlasso_factor(A, rho, D); // returns upper
  arma::mat L   = U.t();

  // 4. iteration
  arma::vec h_objval(maxiter,fill::zeros);
  arma::vec h_r_norm(maxiter,fill::zeros);
  arma::vec h_s_norm(maxiter,fill::zeros);
  arma::vec h_eps_pri(maxiter,fill::zeros);
  arma::vec h_eps_dual(maxiter,fill::zeros);

  for (k=0; k<maxiter; k++) {
    // 4-1. update 'x'
    q = Atb + rho * D.t() * (z-u); // temporary value
    x = solve(trimatu(U),solve(trimatl(L),q));
    //        if (m >= n){
    //            x = solve(trimatu(U),solve(trimatl(L),q));
    //        } else {
    //            x = q/rho - (A.t()*solve(trimatu(U),solve(trimatl(L),A*q)))/rho2;
    //        }

    // 4-2. update 'z'
    zold = z;
    z    = genlasso_shrinkage(D * x + u, lambda / rho);

    // 4-3. update 'u'
    u = u + D*x - z;

    // 4-3. dianostics, reporting
    h_objval(k) = genlasso_objective(A, b, D, lambda, x, z);
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

/*
 Attention: the dimension of the vector x is p, where p is the number of columns
            of W (penalised covariates) and q is the dimension of the vector of
            regression parameters that are not penalized which correspond to
            the design matrix Z (non-penalised covariates),
            while the dimension of the vectors u and z are p                */

/*
    LASSO and adaptive LASSO via ADMM: wrap functions
                                                                        */

/* ADMM with LASSO                      */
//' @name  fdaSP.internal
//' @keywords internal
// [[Rcpp::export(.admm_lasso)]]
Rcpp::List admm_lasso(const arma::mat& A,
                      const arma::colvec& b,
                      arma::colvec& u,
                      arma::colvec& z,
                      const double lambda,
                      const bool rho_adaptation,
                      double rho,
                      const double tau,
                      const double mu,
                      const double reltol,
                      const double abstol,
                      const int maxiter,
                      const int ping) {
    
  /* Variable delcaration           */
  double sqrtn = 0.0, rho_old = 0.0, elTime = 0.0, rss = 0.0, penalty = 0.0;
  int k = 0, n1 = 0;
  
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
    h_objval(k) = lasso_objfun(A, b, lambda, x, rss, penalty);
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
  output["x"] = x;
  output["u"] = u;
  output["z"] = z;
  if (k < maxiter) {
    output["convergence"] = 1.0;
    output["niter"]       = k+1;
    output["objval"]      = h_objval.subvec(0, k);
    output["r_norm"]      = h_r_norm.subvec(0, k);
    output["s_norm"]      = h_s_norm.subvec(0, k);
    output["eps_pri"]     = h_eps_pri.subvec(0, k);
    output["eps_dual"]    = h_eps_dual.subvec(0, k);
    output["rho"]         = rho_store.subvec(0, k+1);
  } else {
    output["convergence"] = 0.0;
    output["niter"]       = maxiter;
    output["objval"]      = h_objval;
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

/* ADMM with LASSO and additional covariates                                     */
//' @name  fdaSP.internal
//' @keywords internal
// [[Rcpp::export(.admm_lasso_cov)]]
Rcpp::List admm_lasso_cov(const arma::mat& W,
                          const arma::mat& Z,
                          const arma::colvec& y,
                          arma::colvec& u,
                          arma::colvec& z,
                          const double lambda,
                          const bool rho_adaptation,
                          double rho,
                          const double tau,
                          const double mu,
                          const double reltol,
                          const double abstol,
                          const int maxiter,
                          const int ping) {
    
    /* Get dimensions             */
    const int m       = W.n_rows;
    const int n_spcov = W.n_cols;

    /* Variable delcaration           */
    List out;

    /* Run ADMM            */
    if (m >= n_spcov) {
      /* large n problems            */
      out = admm_lasso_cov_large_m(W, Z, y, u, z, lambda,
                                   rho_adaptation, rho, tau, mu,
                                   reltol, abstol, maxiter, ping);
    } else {
      /* large p problems            */
      out = admm_lasso_cov_large_n(W, Z, y, u, z, lambda,
                                   rho_adaptation, rho, tau, mu,
                                   reltol, abstol, maxiter, ping);
    }

    /* Return output      */
    return(out);
}

/* ADMM with adaptive LASSO                                         */
//' @name  fdaSP.internal
//' @keywords internal
// [[Rcpp::export(.admm_adalasso)]]
Rcpp::List admm_adalasso(const arma::mat& A,
                         const arma::colvec& b,
                         const arma::vec& var_weights,
                         arma::colvec& u,
                         arma::colvec& z,
                         const double lambda,
                         const bool rho_adaptation,
                         double rho,
                         const double tau,
                         const double mu,
                         const double reltol,
                         const double abstol,
                         const int maxiter,
                         const int ping) {
  
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

/* ADMM with adaptive LASSO and additional covariates                                     */
//' @name  fdaSP.internal
//' @keywords internal
// [[Rcpp::export(.admm_adalasso_cov)]]
Rcpp::List admm_adalasso_cov(const arma::mat& W,
                             const arma::mat& Z,
                             const arma::colvec& y,
                             const arma::vec& var_weights,
                             arma::colvec& u,
                             arma::colvec& z,
                             const double lambda,
                             const bool rho_adaptation,
                             double rho,
                             const double tau,
                             const double mu,
                             const double reltol,
                             const double abstol,
                             const int maxiter,
                             const int ping) {
  
  /* Get dimensions             */
  const int m       = W.n_rows;
  const int n_spcov = W.n_cols;
  
  /* Variable delcaration           */
  List out;
    
  /* Run ADMM            */
  if (m >= n_spcov) {
    /* large n problems            */
    out = admm_adalasso_cov_large_m(W, Z, y, var_weights, u, z,
                                    lambda, rho_adaptation, rho, tau, mu,
                                    reltol, abstol, maxiter, ping);
  } else {
    /* large p problems            */
    out = admm_adalasso_cov_large_n(W, Z, y, var_weights, u, z,
                                    lambda, rho_adaptation, rho, tau, mu,
                                    reltol, abstol, maxiter, ping);
  }

  /* Return output      */
  return(out);
}

/* ADMM with LASSO, fast functions                                   */
//' @name  fdaSP.internal
//' @keywords internal
// [[Rcpp::export(.admm_lasso_fast)]]
Rcpp::List admm_lasso_fast(const arma::mat& A,
                           const arma::vec& b,
                           const arma::vec& lambda,
                           const bool rho_adaptation,
                           double rho,
                           const double tau,
                           const double mu,
                           const double reltol,
                           const double abstol,
                           const int maxiter,
                           const int ping) {
    
  /* Variable delcaration           */
  double b2 = 0.0, min_mse = 0.0;
  int n1 = 0;
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
  output["iternum"]      = niter;
  output["objfun"]       = h_objval;
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

/* ADMM with LASSO and additional covariates, fast functions                                   */
//' @name  fdaSP.internal
//' @keywords internal
// [[Rcpp::export(.admm_lasso_cov_fast)]]
Rcpp::List admm_lasso_cov_fast(const arma::mat& W,
                               const arma::mat& Z,
                               const arma::vec& y,
                               const arma::vec& lambda,
                               const bool rho_adaptation,
                               double rho,
                               const double tau,
                               const double mu,
                               const double reltol,
                               const double abstol,
                               const int maxiter,
                               const int ping) {

  /* Get dimensions             */
  const int m       = W.n_rows;
  const int n_spcov = W.n_cols;

  /* Variable delcaration           */
  List out;

  /* Run ADMM            */
  if (m >= n_spcov) {
    /* large n problems            */
    out = admm_lasso_cov_large_m_fast(W, Z, y, lambda, rho_adaptation, rho, tau, mu,
                                      reltol, abstol, maxiter, ping);
  } else {
    /* large p problems            */
    out = admm_lasso_cov_large_n_fast(W, Z, y, lambda, rho_adaptation, rho, tau, mu,
                                      reltol, abstol, maxiter, ping);
  }

  /* Return output      */
  return(out);
}

/* ADMM with adaptive LASSO, fast functions                                   */
//' @name  fdaSP.internal
//' @keywords internal
// [[Rcpp::export(.admm_adalasso_fast)]]
Rcpp::List admm_adalasso_fast(const arma::mat& A,
                              const arma::vec& b,
                              const arma::vec& var_weights,
                              const arma::vec lambda,
                              const bool rho_adaptation,
                              double rho,
                              const double tau,
                              const double mu,
                              const double reltol,
                              const double abstol,
                              const int maxiter,
                              const int ping) {

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

/* ADMM with adaptive LASSO and additional covariates, fast functions                                   */
//' @name  fdaSP.internal
//' @keywords internal
// [[Rcpp::export(.admm_adalasso_cov_fast)]]
Rcpp::List admm_adalasso_cov_fast(const arma::mat& W,
                                  const arma::mat& Z,
                                  const arma::vec& y,
                                  const arma::vec& var_weights,
                                  const arma::vec& lambda,
                                  const bool rho_adaptation,
                                  double rho,
                                  const double tau,
                                  const double mu,
                                  const double reltol,
                                  const double abstol,
                                  const int maxiter,
                                  const int ping) {

  /* Get dimensions             */
  const int m       = W.n_rows;
  const int n_spcov = W.n_cols;

  /* Variable delcaration           */
  List out;

  /* Run ADMM            */
  if (m >= n_spcov) {
    /* large n problems            */
    out = admm_adalasso_cov_large_m_fast(W, Z, y, var_weights,
                                         lambda, rho_adaptation, rho, tau, mu,
                                         reltol, abstol, maxiter, ping);
  } else {
    /* large p problems            */
    out = admm_adalasso_cov_large_n_fast(W, Z, y, var_weights,
                                         lambda, rho_adaptation, rho, tau, mu,
                                         reltol, abstol, maxiter, ping);
  }

  /* Return output      */
  return(out);
}

/*
    LASSO and adaptive LASSO via ADMM: main functions
                                                                        */

Rcpp::List admm_adalasso_cov_large_m(const arma::mat& W,
                                     const arma::mat& Z,
                                     const arma::vec& y,
                                     const arma::vec& var_weights,
                                     arma::colvec& u,
                                     arma::colvec& z,
                                     const double lambda,
                                     const bool rho_adaptation,
                                     double rho,
                                     const double tau,
                                     const double mu,
                                     const double reltol,
                                     const double abstol,
                                     const int maxiter,
                                     const int ping) {

  /* Variable delcaration           */
  double y2 = 0.0, sqrtm = 0.0, rho_old = 0.0, elTime = 0.0, rss = 0.0, pen = 0.0;
  int n1 = 0, k = 0;

  /* Get dimensions             */
  const int m       = W.n_rows;
  const int n_spcov = W.n_cols;
  const int n_cov   = Z.n_cols;
  
  /* Variable definition            */
  //sqrtn = std::sqrt(static_cast<float>(n_spcov));
  sqrtm = std::sqrt(static_cast<float>(m));

  /* Get output lists          */
  List out, output;

  /* Variable definition            */
  if (m >= n_spcov) {
    n1 = n_spcov;
  } else {
    n1 = m;
  }

  /* Definition of vectors and matrices     */
  arma::colvec x(n_spcov, fill::zeros);
  arma::colvec v(n_cov, fill::zeros);
  arma::mat W_(m, n_spcov, fill::zeros);
  arma::mat Z_(m, n_cov, fill::zeros);
  arma::colvec q(n_spcov, fill::zeros);
  arma::colvec z_old(n_spcov, fill::zeros);
  arma::mat U(n_spcov, n_spcov, fill::zeros);
  arma::mat L(n_spcov, n_spcov, fill::zeros);
  arma::mat WW_CHOL_U(n1, n_spcov, fill::zeros);
  arma::mat ZZ_CHOL_U(n_cov, n_cov, fill::zeros);
  arma::vec Wy(n_spcov, fill::zeros);
  arma::vec ones_m(m, fill::ones);
  arma::mat eye_m(m, m, fill::eye);
  arma::mat ZZ(n_cov, n_cov, fill::zeros);
  arma::mat L_ZZ(n_cov, n_cov, fill::zeros);
  arma::mat L_ZZ_INV(n_cov, n_cov, fill::zeros);
  arma::mat ZZ_INV(n_cov, n_cov, fill::zeros);
  arma::mat P_ZZ(m, m, fill::zeros);
  arma::mat M_ZZ(m, m, fill::zeros);
  arma::mat WMW(n_spcov, n_spcov, fill::zeros);
  arma::vec WMy(n_spcov, fill::zeros);
  arma::vec Zy(n_cov, fill::zeros);
  arma::vec v_LS(n_cov, fill::zeros);
  arma::mat ZW(n_cov, n_spcov, fill::zeros);
  arma::mat P_ZW(n_cov, n_spcov, fill::zeros);
  arma::vec rho_store(maxiter+1, fill::zeros);
  arma::vec h_objval(maxiter, fill::zeros);
  arma::vec h_r_norm(maxiter, fill::zeros);
  arma::vec h_s_norm(maxiter, fill::zeros);
  arma::vec h_eps_pri(maxiter, fill::zeros);
  arma::vec h_eps_dual(maxiter, fill::zeros);
  
  /* Precompute relevant quantities         */
  Z_             = Z.each_col() % (ones_m * (1.0 / sqrtm));                   // 19 Nov 25, la divisione per n è corretta
  W_             = W.each_col() % (ones_m * (1.0 / sqrtm));
  ZZ             = Z_.t() * Z_;
  L_ZZ           = (chol(ZZ)).t();
  L_ZZ_INV       = inv(trimatl(L_ZZ));
  ZZ_INV         = L_ZZ_INV.t() * L_ZZ_INV;
  P_ZZ           = Z_ * ZZ_INV * Z_.t();
  M_ZZ           = eye_m - P_ZZ;
  WMW            = W_.t() * M_ZZ * W_;
  WMy            = W_.t() * M_ZZ * y / static_cast<float>(sqrtm);
  Zy             = Z_.t() * y / static_cast<float>(sqrtm);
  v_LS           = L_ZZ_INV.t() * L_ZZ_INV * Zy;
  ZW             = Z_.t() * W_;
  P_ZW           = L_ZZ_INV.t() * L_ZZ_INV * ZW;
  WW_CHOL_U      = chol_qr_fact(W_, m, n_spcov);
  ZZ_CHOL_U      = chol_qr_fact(Z_, m, n_cov);
  y2             = as_scalar(y.t() * y) / static_cast<float>(m);
  Wy             = W_.t() * y / static_cast<float>(sqrtm);
  
  /* ::::::::::::::::::::::::::::::::::::::::::::::
   Set the computational time                                 */
  wall_clock timer;
  timer.tic();
  
  /* store rho (only for adaptation)         */
  rho_store(0) = rho;
  rho_old      = rho;

  /* Precompute relevant quantities         */
  //U = glasso_factor_fast_large_m(ATA, FTF, rho); // returns upper
  //U = glasso_factor_cov_fast_large_m(WMW, FTF, rho);
  U = adalasso_factor_fast_large_m(WMW, rho);
  L = U.t();
  
  /* main loop  */
  for (k=0; k<maxiter; k++) {

    // 4-1. update 'x'
    if (rho != rho_old) {
      //U = glasso_factor_fast_large_m(ATA, FTF, rho); // returns upper
      //U = glasso_factor_cov_fast_large_m(WMW, FTF, rho);
      U = adalasso_factor_fast_large_m(WMW, rho);
      L = U.t();
    }
    q = WMy + rho * (z - u);           // temporary value
    x = solve(trimatu(U), solve(trimatl(L), q));         // updated of regression parameters associated to covariates W (sparsified covariates, x)
    v = v_LS - P_ZW * x;                                 // updated of regression parameters associated to covariates Z (covariates not sparsified, v)
    
    // 4-2. update 'z' with adaptation
    // see Boyd et al (2011) page 21
    z_old = z;
    z     = adalasso_prox(x + u, lambda * var_weights / rho);

    // 4-3. update 'u'
    u += x - z;
  
    // 4-3. dianostics, reporting
    rss         = lm_cov_rss_fast(WW_CHOL_U, ZZ_CHOL_U, ZW, Wy, Zy, y2, x, v, m);
    pen         = adalasso_penalty(x, var_weights);
    h_objval(k) = 0.5 *  rss + lambda * pen;            // RSS è già diviso per numerosità campionaria
    h_r_norm(k) = norm(adalasso_residual(x, z), 2);
    h_s_norm(k) = norm(adalasso_dual_residual(z, z_old, rho), 2);
    if (norm(x, 2) > norm(-z, 2)) {
      h_eps_pri(k) = std::sqrt(z.n_elem) * abstol + reltol * norm(x, 2);
    } else {
      h_eps_pri(k) = std::sqrt(z.n_elem) * abstol + reltol * norm(-z, 2);
    }
    h_eps_dual(k) = std::sqrt(z.n_elem) * abstol + reltol * norm(rho * u, 2);
    
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
  
  /* Get output         */
  output["x"]      = x;
  output["v"]      = v;
  output["u"]      = u;
  output["z"]      = z;
  output["eltime"] = elTime;
  if (k < maxiter) {
    output["convergence"] = 1.0;
    output["niter"]       = k+1;
    output["objval"]      = h_objval.subvec(0, k);
    output["r_norm"]      = h_r_norm.subvec(0, k);
    output["s_norm"]      = h_s_norm.subvec(0, k);
    output["eps_pri"]     = h_eps_pri.subvec(0, k);
    output["eps_dual"]    = h_eps_dual.subvec(0, k);
    output["rho"]         = rho_store.subvec(0, k+1);
  } else {
    output["convergence"] = 0.0;
    output["niter"]       = maxiter;
    output["objval"]      = h_objval;
    output["r_norm"]      = h_r_norm;
    output["s_norm"]      = h_s_norm;
    output["eps_pri"]     = h_eps_pri;
    output["eps_dual"]    = h_eps_dual;
    output["rho"]         = rho_store;
  }
  
  /* Return output      */
  return(output);
}

Rcpp::List admm_adalasso_cov_large_n(const arma::mat& W,
                                     const arma::mat& Z,
                                     const arma::vec& y,
                                     const arma::vec& var_weights,
                                     arma::colvec& u,
                                     arma::colvec& z,
                                     double lambda,
                                     const bool rho_adaptation,
                                     double rho,
                                     const double tau,
                                     const double mu,
                                     const double reltol,
                                     const double abstol,
                                     const int maxiter,
                                     const int ping) {

  /* Variable delcaration           */
  double y2 = 0.0, sqrtm = 0.0, rho_old = 0.0, elTime = 0.0, rss = 0.0, pen = 0.0;
  int n1 = 0, k = 0;

  /* Get dimensions             */
  const int m       = W.n_rows;
  const int n_spcov = W.n_cols;
  const int n_cov   = Z.n_cols;
  
  /* Variable definition            */
  //sqrtn = std::sqrt(static_cast<float>(n_spcov));
  sqrtm = std::sqrt(static_cast<float>(m));

  /* Get output lists          */
  List out, output;

  /* Variable definition            */
  if (m >= n_spcov) {
    n1 = n_spcov;
  } else {
    n1 = m;
  }

  /* Definition of vectors and matrices     */
  arma::colvec x(n_spcov, fill::zeros);
  arma::colvec v(n_cov, fill::zeros);
  arma::mat Z_(m, n_cov, fill::zeros);
  arma::colvec q(n_spcov, fill::zeros);
  arma::colvec x_star(n_spcov, fill::zeros);
  arma::colvec z_old(n_spcov, fill::zeros);
  arma::mat U(m, m, fill::zeros);
  arma::mat L(m, m, fill::zeros);
  arma::mat WW_CHOL_U(n1, n_spcov, fill::zeros);
  arma::mat ZZ_CHOL_U(n_cov, n_cov, fill::zeros);
  arma::vec Wy(n_spcov, fill::zeros);
  arma::vec ones_m(m, fill::ones);
  arma::mat eye_m(m, m, fill::eye);
  arma::mat ZZ(n_cov, n_cov, fill::zeros);
  arma::mat L_ZZ(n_cov, n_cov, fill::zeros);
  arma::mat L_ZZ_INV(n_cov, n_cov, fill::zeros);
  arma::mat ZZ_INV(n_cov, n_cov, fill::zeros);
  arma::mat P_ZZ(m, m, fill::zeros);
  arma::mat M_ZZ(m, m, fill::zeros);
  arma::mat WMW(n_spcov, n_spcov, fill::zeros);
  arma::vec WMy(n_spcov, fill::zeros);
  arma::vec Zy(n_cov, fill::zeros);
  arma::vec v_LS(n_cov, fill::zeros);
  arma::mat ZW(n_cov, n_spcov, fill::zeros);
  arma::mat P_ZW(n_cov, n_spcov, fill::zeros);
  arma::mat WW(m, m, fill::zeros);
  arma::mat M_ZZW(m, n_spcov, fill::zeros);
  arma::mat M_ZZW2(m, m, fill::zeros);
  arma::vec rho_store(maxiter+1, fill::zeros);
  arma::vec h_objval(maxiter, fill::zeros);
  arma::vec h_r_norm(maxiter, fill::zeros);
  arma::vec h_s_norm(maxiter, fill::zeros);
  arma::vec h_eps_pri(maxiter, fill::zeros);
  arma::vec h_eps_dual(maxiter, fill::zeros);

  /* Precompute relevant quantities         */
  Z_        = Z.each_col() % (ones_m * (1.0 / sqrtm));
  //W         = W.each_col() % (ones_m * (1.0 / sqrtm));      // è corretto non dividere W per la numerosità campionaria m
  ZZ        = Z_.t() * Z_;
  L_ZZ      = (chol(ZZ)).t();
  L_ZZ_INV  = inv(trimatl(L_ZZ));
  ZZ_INV    = L_ZZ_INV.t() * L_ZZ_INV;
  P_ZZ      = Z_ * ZZ_INV * Z_.t();
  M_ZZ      = eye_m - P_ZZ;
  WMW       = W.t() * M_ZZ * W;                               // 19Nov25 qui è stata corretta la divisione per n, il resto è giusto
  WMy       = W.t() * M_ZZ * y;
  Zy        = Z_.t() * y / static_cast<float>(sqrtm);
  v_LS      = L_ZZ_INV.t() * L_ZZ_INV * Zy;
  ZW        = Z_.t() * W / static_cast<float>(sqrtm);
  P_ZW      = L_ZZ_INV.t() * L_ZZ_INV * ZW;
  WW_CHOL_U = chol_qr_fact(W, m, n_spcov);
  ZZ_CHOL_U = chol_qr_fact(Z_, m, n_cov);
  y2        = as_scalar(y.t() * y) / static_cast<float>(m);
  Wy        = W.t() * y;
  M_ZZW     = M_ZZ * W;                                       // 19Nov25 qui è stata corretta la divisione per n, il resto è giusto
  
  /* Define the F matrix         */
  M_ZZW2 = M_ZZW * M_ZZW.t();
  
  /* ::::::::::::::::::::::::::::::::::::::::::::::
   Set the computational time                                 */
  wall_clock timer;
  timer.tic();
  
  /* store rho (only for adaptation)         */
  rho_store(0) = rho;
  rho_old      = rho;

  /* Precompute relevant quantities         */
  U = glasso_factor_cov_fast_large_n(M_ZZW2, rho, m); // returns upper
  L = U.t();
  
  /* main loop  */
  for (k=0; k<maxiter; k++) {

    // 4-1. update 'x'
    if (rho != rho_old) {
      U = glasso_factor_cov_fast_large_n(M_ZZW2, rho, m); // returns upper
      L = U.t();
    }
    q      = WMy + rho * (z - u); // temporary value
    x_star = q / rho;
    x      = x_star - M_ZZW.t() * solve(trimatu(U), solve(trimatl(L), M_ZZW * x_star)); // updated of regression parameters associated to covariates W
    v      = v_LS - P_ZW * x;                                 // updated of regression parameters associated to covariates Z (covariates not sparsified, v)

    // 4-2. update 'z' with adaptation
    // see Boyd et al (2011) page 21
    z_old = z;
    z     = adalasso_prox(x + u, lambda * var_weights / rho);

    // 4-3. update 'u'
    u = u + x - z;
  
    // 4-3. dianostics, reporting
    rss         = lm_cov_rss_fast(WW_CHOL_U, ZZ_CHOL_U, ZW, Wy, Zy, y2, x, v, m);
    pen         = adalasso_penalty(x, var_weights);
    h_objval(k) = 0.5 *  rss / static_cast<float>(m) + lambda * pen;
    h_r_norm(k) = norm(adalasso_residual(x, z), 2);
    h_s_norm(k) = norm(adalasso_dual_residual(z, z_old, rho), 2);
    if (norm(x, 2) > norm(-z, 2)) {
      h_eps_pri(k) = std::sqrt(z.n_elem) * abstol + reltol * norm(x, 2);
    } else {
      h_eps_pri(k) = std::sqrt(z.n_elem) * abstol + reltol * norm(-z, 2);
    }
    h_eps_dual(k) = std::sqrt(z.n_elem) * abstol + reltol * norm(rho * u, 2);
    
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
    
  /* Get output         */
  output["x"]      = x;
  output["v"]      = v;
  output["u"]      = u;
  output["z"]      = z;
  output["eltime"] = elTime;
  if (k < maxiter) {
    output["convergence"] = 1.0;
    output["niter"]       = k+1;
    output["objval"]      = h_objval.subvec(0, k);
    output["r_norm"]      = h_r_norm.subvec(0, k);
    output["s_norm"]      = h_s_norm.subvec(0, k);
    output["eps_pri"]     = h_eps_pri.subvec(0, k);
    output["eps_dual"]    = h_eps_dual.subvec(0, k);
    output["rho"]         = rho_store.subvec(0, k+1);
  } else {
    output["convergence"] = 0.0;
    output["niter"]       = maxiter;
    output["objval"]      = h_objval;
    output["r_norm"]      = h_r_norm;
    output["s_norm"]      = h_s_norm;
    output["eps_pri"]     = h_eps_pri;
    output["eps_dual"]    = h_eps_dual;
    output["rho"]         = rho_store;
  }
  
  /* Return output      */
  return(output);
}

Rcpp::List admm_adalasso_large_m(const arma::mat& A,
                                 const arma::colvec& b,
                                 const arma::vec& var_weights,
                                 arma::colvec& u,
                                 arma::colvec& z,
                                 const double lambda,
                                 const bool rho_adaptation,
                                 double rho,
                                 const double tau,
                                 const double mu,
                                 const double reltol,
                                 const double abstol,
                                 const int maxiter,
                                 const int ping) {
    
  /* Variable delcaration           */
  double sqrtn = 0.0, rho_old = 0.0, elTime = 0.0, rss = 0.0, pen = 0.0;
  int k = 0, m = 0, n = 0;
  
  /* Get dimensions             */
  m = A.n_rows;
  n = A.n_cols;
    
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
  U   = adalasso_factor_fast_large_m(ATA, rho); // returns upper
  L   = U.t();
  
  /* main loop  */
  for (k=0; k<maxiter; k++) {
    // 4-1. update 'x'
    if (rho != rho_old) {
      U = adalasso_factor_fast_large_m(ATA, rho); // returns upper
      L = U.t();
    }
    q = ATb + rho * (z - u);
    x = solve(trimatu(U), solve(trimatl(L), q));
    
    // 4-2. update 'z' with adaptation
    // see Boyd et al (2011) page 21
    z_old = z;
    z     = adalasso_prox(x + u, lambda * var_weights / rho);
        
    // 4-3. update 'u'
    u = u + x - z;
        
    // 4-3. dianostics, reporting
    h_objval(k) = adalasso_objfun(A, b, lambda, x, var_weights, rss, pen);
    h_r_norm(k) = norm(adalasso_residual(x, z), 2);
    h_s_norm(k) = norm(adalasso_dual_residual(z, z_old, rho), 2);
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
  output["niter"] = k;
  output["x"]     = x;
  output["u"]     = u;
  output["z"]     = z;
  if (k < maxiter) {
    output["objval"]   = h_objval.subvec(0, k);
    output["r_norm"]   = h_r_norm.subvec(0, k);
    output["s_norm"]   = h_s_norm.subvec(0, k);
    output["eps_pri"]  = h_eps_pri.subvec(0, k);
    output["eps_dual"] = h_eps_dual.subvec(0, k);
    output["rho"]      = rho_store.subvec(0, k+1);
  } else {
    output["objval"]   = h_objval;
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

Rcpp::List admm_adalasso_large_n(const arma::mat& A,
                                 const arma::colvec& b,
                                 const arma::vec& var_weights,
                                 arma::colvec& u,
                                 arma::colvec& z,
                                 const double lambda,
                                 const bool rho_adaptation,
                                 double rho,
                                 const double tau,
                                 const double mu,
                                 const double reltol,
                                 const double abstol,
                                 const int maxiter,
                                 const int ping) {
    
  /* Variable delcaration           */
  double sqrtn = 0.0, rho_old = 0.0, elTime = 0.0, rss = 0.0, pen = 0.0;
  int k = 0, m = 0, n = 0;
  
  /* Get dimensions             */
  m = A.n_rows;
  n = A.n_cols;
    
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
  AFTF_INVA = A * A.t();
  U         = adalasso_factor_fast_large_n(AFTF_INVA, rho, m);
  L         = U.t();

  /* main loop  */
  for (k=0; k<maxiter; k++) {
    // 4-1. update 'x'
    if (rho != rho_old) {
      U = adalasso_factor_fast_large_n(AFTF_INVA, rho, m);
      L = U.t();
    }
    q      = ATb + rho * (z - u); // temporary value
    x_star = q / rho;
    x      = x_star - (A.t() * solve(trimatu(U), solve(trimatl(L), A * x_star)));
            
    // 4-2. update 'z' with adaptation
    // see Boyd et al (2011) page 21
    z_old = z;
    z     = adalasso_prox(x + u, lambda * var_weights / rho);
    
    // 4-3. update 'u'
    u = u + x - z;

    // 4-3. dianostics, reporting
    h_objval(k) = adalasso_objfun(A, b, lambda, x, var_weights, rss, pen);
    h_r_norm(k) = norm(adalasso_residual(x, z), 2);
    h_s_norm(k) = norm(adalasso_dual_residual(z, z_old, rho), 2);
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
  output["niter"] = k;
  output["x"]     = x;
  output["u"]     = u;
  output["z"]     = z;
  if (k < maxiter) {
    output["objval"]   = h_objval.subvec(0, k);
    output["r_norm"]   = h_r_norm.subvec(0, k);
    output["s_norm"]   = h_s_norm.subvec(0, k);
    output["eps_pri"]  = h_eps_pri.subvec(0, k);
    output["eps_dual"] = h_eps_dual.subvec(0, k);
    output["rho"]      = rho_store.subvec(0, k+1);
  } else {
    output["objval"]   = h_objval;
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

Rcpp::List admm_adalasso_large_m_fast(const arma::mat& A, 
                                      const arma::vec& b,
                                      const arma::vec& var_weights,
                                      const arma::vec& lambda,
                                      const bool rho_adaptation,
                                      double rho,
                                      const double tau,
                                      const double mu,
                                      const double reltol, 
                                      const double abstol,
                                      const int maxiter,
                                      const int ping) {

  /* Variable delcaration           */
  double b2 = 0.0, min_mse = 0.0;
  int n1 = 0;
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
  
  /* Initialize vectors u and z         */
  arma::colvec u(n, fill::zeros);
  arma::colvec z(n, fill::zeros);
  
  /* main loop  */
  for (int j=0; j<nlambda; j++) {
    /* perform linear regression with lasso penalty for the j-th lambda       */
    out = admm_adalasso_large_m_update_1lambda(A, b, ATA, ATA_CHOL_U, ATb, b2, var_weights, m, n,
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
  output["iternum"]      = niter;
  output["objfun"]       = h_objval;
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

Rcpp::List admm_adalasso_large_n_fast(const arma::mat& A, 
                                      const arma::vec& b,
                                      const arma::vec& var_weights,
                                      const arma::vec& lambda,
                                      const bool rho_adaptation,
                                      double rho,
                                      const double tau,
                                      const double mu,
                                      const double reltol, 
                                      const double abstol,
                                      const int maxiter,
                                      const int ping) {

  /* Variable delcaration           */
  double b2 = 0.0, min_mse = 0.0;
  int n1 = 0;
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
  arma::colvec x(n, fill::zeros);
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
  AFTF_INVA  = A * A.t();
  
  /* Initialize vectors u and z         */
  arma::colvec u(n, fill::zeros);
  arma::colvec z(n, fill::zeros);

  /* main loop  */
  for (int j=0; j<nlambda; j++) {
    /* perform linear regression with lasso penalty for the j-th lambda       */
    out = admm_adalasso_large_n_update_1lambda(A, b, AFTF_INVA, ATA_CHOL_U,
                                               ATb, b2, var_weights,
                                               m, n, u, z, lambda(j),
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
  output["iternum"]      = niter;
  output["objfun"]       = h_objval;
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

/*  Code to compute the whole path fast.    */

/*  LASSO via ADMM, just to compute the whole path fast.     */
Rcpp::List admm_lasso_update_1lambda(const arma::mat& A, 
                                     const arma::vec& b,
                                     const arma::mat& ATA_CHOL_U,
                                     const arma::mat& AA,
                                     const arma::vec& ATb,
                                     const double b2,
                                     const int m,
                                     const int n,
                                     arma::colvec& u,
                                     arma::colvec& z,
                                     const double lambda,
                                     const bool rho_adaptation,
                                     double rho,
                                     const double tau,
                                     const double mu,
                                     const double reltol, 
                                     const double abstol,
                                     const int maxiter,
                                     const int ping) {
  
  /* Variable delcaration           */
  int k = 0, n1 = 0;
  double sqrtn = 0.0, rho_old = 0.0, elTime = 0.0, mse = 0.0, rss = 0.0, pen = 0.0;
  if (m >= n) {
    n1 = n;
  } else {
    n1 = m;
  }
  
  /* Definition of vectors and matrices     */
  arma::vec x(n, fill::zeros);
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
    rss         = lm_rss_fast(ATA_CHOL_U, ATb, b2, x, m, n);
    pen         = lasso_penalty(x);
    h_objval(k) = 0.5 *  rss / static_cast<float>(m) + lambda * pen;
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
  output["x"]      = x;
  output["u"]      = u;
  output["z"]      = z;
  output["eltime"] = elTime;
  if (k < maxiter) {
    output["convergence"] = 1.0;
    output["niter"]       = k+1;
    output["objval"]      = h_objval.subvec(0, k);
    output["r_norm"]      = h_r_norm.subvec(0, k);
    output["s_norm"]      = h_s_norm.subvec(0, k);
    output["eps_pri"]     = h_eps_pri.subvec(0, k);
    output["eps_dual"]    = h_eps_dual.subvec(0, k);
    output["rho"]         = rho_store.subvec(0, k+1);
  } else {
    output["convergence"] = 0.0;
    output["niter"]       = maxiter;
    output["objval"]      = h_objval;
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

Rcpp::List admm_adalasso_large_m_update_1lambda(const arma::mat& A,
                                                const arma::vec& b,
                                                const arma::mat& ATA,
                                                const arma::mat& ATA_CHOL_U,
                                                const arma::vec& ATb,
                                                const double b2,
                                                const arma::vec& var_weights,
                                                const int m,
                                                const int n,
                                                arma::colvec& u,
                                                arma::colvec& z,
                                                const double lambda,
                                                const bool rho_adaptation,
                                                double rho,
                                                const double tau,
                                                const double mu,
                                                const double reltol, 
                                                const double abstol,
                                                const int maxiter,
                                                const int ping) {
  
  /* Variable delcaration           */
  int k = 0, dim_z = 0;
  double rho_old = 0.0, elTime = 0.0, mse = 0.0, sqrtn = 0.0, rss = 0.0, pen = 0.0;
  
  /* Get dimensions                 */
  dim_z = z.n_elem;
  
  /* Variable definition            */
  sqrtn = std::sqrt(static_cast<float>(n));
  
  /* Definition of vectors and matrices     */
  arma::vec x(n, fill::zeros);
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
  U = adalasso_factor_fast_large_m(ATA, rho); // returns upper
  L = U.t();
  
  /* main loop  */
  for (k=0; k<maxiter; k++) {
  
    // 4-1. update 'x'
    if (rho != rho_old) {
      U = adalasso_factor_fast_large_m(ATA, rho); // returns upper
      L = U.t();
    }
    q = ATb + rho * (z - u);
    x = solve(trimatu(U), solve(trimatl(L), q));
    
    // 4-2. update 'z' with adaptation
    // see Boyd et al (2011) page 21
    z_old = z;
    z     = adalasso_prox(x + u, lambda * var_weights / rho);
    
    // 4-3. update 'u'
    u = u + x - z;
    
    // 4-3. dianostics, reporting
    rss         = lm_rss_fast(ATA_CHOL_U, ATb, b2, x, m, n);
    pen         = adalasso_penalty(x, var_weights);
    h_objval(k) = 0.5 *  rss / static_cast<float>(m) + lambda * pen;
    h_r_norm(k) = norm(adalasso_residual(x, z), 2);
    h_s_norm(k) = norm(adalasso_dual_residual(z, z_old, rho), 2);
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
  output["x"]      = x;
  output["u"]      = u;
  output["z"]      = z;
  output["eltime"] = elTime;
  if (k < maxiter) {
    output["convergence"] = 1.0;
    output["niter"]       = k+1;
    output["objval"]      = h_objval.subvec(0, k);
    output["r_norm"]      = h_r_norm.subvec(0, k);
    output["s_norm"]      = h_s_norm.subvec(0, k);
    output["eps_pri"]     = h_eps_pri.subvec(0, k);
    output["eps_dual"]    = h_eps_dual.subvec(0, k);
    output["rho"]         = rho_store.subvec(0, k+1);
  } else {
    output["convergence"] = 0.0;
    output["niter"]       = maxiter;
    output["objval"]      = h_objval;
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

Rcpp::List admm_adalasso_large_n_update_1lambda(const arma::mat& A, 
                                                const arma::vec& b,
                                                const arma::mat& AFTF_INVA,
                                                const arma::mat& ATA_CHOL_U,
                                                const arma::vec& ATb,
                                                const double b2,
                                                const arma::vec& var_weights,
                                                const int m,
                                                const int n,
                                                arma::colvec& u,
                                                arma::colvec& z,
                                                const double lambda,
                                                const bool rho_adaptation,
                                                double rho,
                                                const double tau,
                                                const double mu,
                                                const double reltol, 
                                                const double abstol,
                                                const int maxiter,
                                                const int ping) {
  
  /* Variable delcaration           */
  int k= 0, dim_z= 0;
  double rho_old= 0.0, elTime = 0.0, mse= 0.0, sqrtn= 0.0, rss = 0.0, pen = 0.0;
 
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
    q      = ATb + rho * (z - u); // temporary value
    x_star = q / rho;
    x      = x_star - (A.t() * solve(trimatu(U), solve(trimatl(L), A * x_star)));
    
    // 4-2. update 'z' with adaptation
    // see Boyd et al (2011) page 21
    z_old = z;
    z     = adalasso_prox(x + u, lambda * var_weights / rho);
    
    // 4-3. update 'u'
    u = u + x - z;
    
    // 4-3. dianostics, reporting
    rss         = lm_rss_fast(ATA_CHOL_U, ATb, b2, x, m, n);
    pen         = adalasso_penalty(x, var_weights);
    h_objval(k) = 0.5 *  rss / static_cast<float>(m) + lambda * pen;
    h_r_norm(k) = norm(adalasso_residual(x, z), 2);
    h_s_norm(k) = norm(adalasso_dual_residual(z, z_old, rho), 2);
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
  output["x"]      = x;
  output["u"]      = u;
  output["z"]      = z;
  output["eltime"] = elTime;
  if (k < maxiter) {
    output["convergence"] = 1.0;
    output["niter"]       = k+1;
    output["objval"]      = h_objval.subvec(0, k);
    output["r_norm"]      = h_r_norm.subvec(0, k);
    output["s_norm"]      = h_s_norm.subvec(0, k);
    output["eps_pri"]     = h_eps_pri.subvec(0, k);
    output["eps_dual"]    = h_eps_dual.subvec(0, k);
    output["rho"]         = rho_store.subvec(0, k+1);
  } else {
    output["convergence"] = 0.0;
    output["niter"]       = maxiter;
    output["objval"]      = h_objval;
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

Rcpp::List admm_lasso_cov_large_m_update_1lambda(const arma::mat& W,
                                                 const arma::mat& Z,
                                                 const arma::vec& y,
                                                 const arma::mat& WMW,
                                                 const arma::vec& WMy,
                                                 const arma::vec& v_LS,
                                                 const arma::mat& P_ZW,
                                                 const arma::mat& WW_CHOL_U,
                                                 const arma::vec& Wy,
                                                 const double y2,
                                                 const arma::mat& ZZ_CHOL_U, 
                                                 const arma::vec& Zy,
                                                 const arma::mat& ZW,
                                                 arma::colvec& u,
                                                 arma::colvec& z,
                                                 const double lambda,
                                                 const bool rho_adaptation,
                                                 double rho,
                                                 const double tau,
                                                 const double mu,
                                                 const double reltol, 
                                                 const double abstol,
                                                 const int maxiter, 
                                                 const int ping) {
  
  /* Variable delcaration           */
  int k = 0;
  double rho_old = 0.0, elTime = 0.0, sqrtn = 0.0, rss = 0.0, pen = 0.0;
  
  /* Get dimensions             */
  const int m       = W.n_rows;
  const int n_spcov = W.n_cols;
  const int n_cov   = Z.n_cols;
  
  /* Variable definition            */
  sqrtn = std::sqrt(static_cast<float>(n_spcov));
  
  /* Definition of vectors and matrices     */
  arma::colvec x(n_spcov, fill::zeros);
  arma::colvec v(n_cov, fill::zeros);
  arma::colvec q(n_spcov, fill::zeros);
  arma::colvec z_old(n_spcov, fill::zeros);
  arma::mat U(n_spcov, n_spcov, fill::zeros);
  arma::mat L(n_spcov, n_spcov, fill::zeros);
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
  U = lasso_factor_cov_fast_large_m(WMW, rho);    // returns upper
  L = U.t();
  
  /* main loop  */
  for (k=0; k<maxiter; k++) {

    // 4-1. update 'x'
    if (rho != rho_old) {
      U = lasso_factor_cov_fast_large_m(WMW, rho);    // returns upper
      L = U.t();
    }
    q = WMy + rho * (z - u);                             // temporary value
    x = solve(trimatu(U), solve(trimatl(L), q));         // updated of regression parameters associated to covariates W (sparsified covariates, x)
    v = v_LS - P_ZW * x;                                 // updated of regression parameters associated to covariates Z (covariates not sparsified, v)
    
    // 4-2. update 'z' with adaptation
    // see Boyd et al (2011) page 21
    z_old = z;
    z     = lasso_prox(x + u, lambda / rho);
    
    // 4-3. update 'u'
    u = u + x - z;
  
    // 4-3. dianostics, reporting
    rss         = lm_cov_rss_fast(WW_CHOL_U, ZZ_CHOL_U, ZW, Wy, Zy, y2, x, v, m);
    pen         = lasso_penalty(x);
    h_objval(k) = 0.5 *  rss / static_cast<float>(m) + lambda * pen;
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
            Rcpp::Rcout << "ADMM with adaptation for LASSO with covariates is running!" << std::endl;
            Rcpp::Rcout << "Iteration n.: " << k+1 << " of " << maxiter << "\n" << std::endl;
          }
        } else {
          if (((k+1) % ping) == 0) {
            Rcpp::Rcout << "\n\n\n" << std::endl;
            Rcpp::Rcout << "ADMM for LASSO with covariates is running!" << std::endl;
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
  output["x"]      = x;
  output["v"]      = v;
  output["u"]      = u;
  output["z"]      = z;
  output["eltime"] = elTime;
  if (k < maxiter) {
    output["convergence"] = 1.0;
    output["niter"]       = k+1;
    output["objval"]      = h_objval.subvec(0, k);
    output["r_norm"]      = h_r_norm.subvec(0, k);
    output["s_norm"]      = h_s_norm.subvec(0, k);
    output["eps_pri"]     = h_eps_pri.subvec(0, k);
    output["eps_dual"]    = h_eps_dual.subvec(0, k);
    output["rho"]         = rho_store.subvec(0, k+1);
  } else {
    output["convergence"] = 0.0;
    output["niter"]       = maxiter;
    output["objval"]      = h_objval;
    output["r_norm"]      = h_r_norm;
    output["s_norm"]      = h_s_norm;
    output["eps_pri"]     = h_eps_pri;
    output["eps_dual"]    = h_eps_dual;
    output["rho"]         = rho_store;
  }
  //output["residuals"]     = residuals;
  //output["fitted.values"] = fitted;
  //output["mse"]           = mse;
  output["rho.updated"]   = rho;
  
  /* Return output      */
  return(output);
}

Rcpp::List admm_lasso_cov_large_n_update_1lambda(const arma::mat& W,
                                                 const arma::mat& Z,
                                                 const arma::vec& y,
                                                 const arma::mat& M_ZZW,
                                                 const arma::mat& M_ZZW2,
                                                 const arma::mat& WMW,
                                                 const arma::vec& WMy,
                                                 const arma::vec& v_LS,
                                                 const arma::mat& P_ZW,
                                                 const arma::mat& WW_CHOL_U,
                                                 const arma::colvec& Wy,
                                                 const double y2,
                                                 const arma::mat& ZZ_CHOL_U, 
                                                 const arma::colvec& Zy,
                                                 const arma::mat& ZW,
                                                 arma::colvec& u,
                                                 arma::colvec& z,
                                                 const double lambda,
                                                 bool rho_adaptation,
                                                 double rho,
                                                 const double tau,
                                                 const double mu,
                                                 const double reltol, 
                                                 const double abstol,
                                                 const int maxiter,
                                                 const int ping) {
  
  /* Variable delcaration           */
  int k = 0;
  double rho_old = 0.0, elTime = 0.0, sqrtn = 0.0, rss = 0.0, pen = 0.0;
  
  /* Get dimensions             */
  const int m       = W.n_rows;
  const int n_spcov = W.n_cols;
  const int n_cov   = Z.n_cols;
  
  /* Variable definition            */
  sqrtn = std::sqrt(static_cast<float>(n_spcov));
  
  /* Definition of vectors and matrices     */
  arma::vec x(n_spcov, fill::zeros);
  arma::vec v(n_cov, fill::zeros);
  arma::colvec q(n_spcov, fill::zeros);
  arma::colvec z_old(n_spcov, fill::zeros);
  arma::mat U(m, m, fill::zeros);
  arma::mat L(m, m, fill::zeros);
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
  U = lasso_factor_cov_fast_large_n(M_ZZW2, rho, m);    // returns upper
  L = U.t();
  
  /* main loop  */
  for (k=0; k<maxiter; k++) {

    // 4-1. update 'x'
    if (rho != rho_old) {
      U = lasso_factor_cov_fast_large_n(M_ZZW2, rho, m);    // returns upper
      L = U.t();
    }
    q = WMy + rho * (z - u);                             // temporary value
    x = q / rho - (M_ZZW.t() * solve(trimatu(U), solve(trimatl(L), M_ZZW * q))) / rho;         // updated of regression parameters associated to covariates W
    v = v_LS - P_ZW * x;                                 // updated of regression parameters associated to covariates Z (covariates not sparsified, v)
    
    // 4-2. update 'z' with adaptation
    // see Boyd et al (2011) page 21
    z_old = z;
    z     = lasso_prox(x + u, lambda / rho);
    
    // 4-3. update 'u'
    u = u + x - z;
  
    // 4-3. dianostics, reporting
    rss         = lm_cov_rss_fast(WW_CHOL_U, ZZ_CHOL_U, ZW, Wy, Zy, y2, x, v, m);
    pen         = lasso_penalty(x);
    h_objval(k) = 0.5 *  rss / static_cast<float>(m) + lambda * pen;
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
            Rcpp::Rcout << "ADMM with adaptation for LASSO with covariates is running!" << std::endl;
            Rcpp::Rcout << "Iteration n.: " << k+1 << " of " << maxiter << "\n" << std::endl;
          }
        } else {
          if (((k+1) % ping) == 0) {
            Rcpp::Rcout << "\n\n\n" << std::endl;
            Rcpp::Rcout << "ADMM for LASSO with covariates is running!" << std::endl;
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
  output["x"]      = x;
  output["v"]      = v;
  output["u"]      = u;
  output["z"]      = z;
  output["eltime"] = elTime;
  if (k < maxiter) {
    output["convergence"] = 1.0;
    output["niter"]       = k+1;
    output["objval"]      = h_objval.subvec(0, k);
    output["r_norm"]      = h_r_norm.subvec(0, k);
    output["s_norm"]      = h_s_norm.subvec(0, k);
    output["eps_pri"]     = h_eps_pri.subvec(0, k);
    output["eps_dual"]    = h_eps_dual.subvec(0, k);
    output["rho"]         = rho_store.subvec(0, k+1);
  } else {
    output["convergence"] = 0.0;
    output["niter"]       = maxiter;
    output["objval"]      = h_objval;
    output["r_norm"]      = h_r_norm;
    output["s_norm"]      = h_s_norm;
    output["eps_pri"]     = h_eps_pri;
    output["eps_dual"]    = h_eps_dual;
    output["rho"]         = rho_store;
  }
  //output["residuals"]     = residuals;
  //output["fitted.values"] = fitted;
  //output["mse"]           = mse;
  output["rho.updated"]   = rho;
  
  /* Return output      */
  return(output);
}

Rcpp::List admm_lasso_cov_large_m_fast(const arma::mat& W,
                                       const arma::mat& Z,
                                       const arma::vec& y,
                                       const arma::vec& lambda,
                                       const bool rho_adaptation,
                                       double rho,
                                       const double tau,
                                       const double mu,
                                       const double reltol, 
                                       const double abstol,
                                       const int maxiter,
                                       const int ping) {

  /* Variable delcaration           */
  double y2 = 0.0, min_mse = 0.0, sqrtm = 0.0;
  int n1 = 0;
  uword idx_min_mse;

  /* Get dimensions             */
  const int m       = W.n_rows;
  const int n_spcov = W.n_cols;
  const int n_cov   = Z.n_cols;
  
  /* Variable definition            */
  sqrtm = std::sqrt(static_cast<float>(m));

  /* Get number of lambda           */
  const int nlambda = lambda.n_elem;

  /* Get output lists          */
  List out, output;

  /* Variable definition            */
  if (m >= n_spcov) {
    n1 = n_spcov;
  } else {
    n1 = m;
  }

  /* Definition of vectors and matrices     */
  arma::colvec x(n_spcov, fill::zeros);
  arma::colvec v(n_cov, fill::zeros);
  arma::colvec u(n_spcov, fill::zeros);
  arma::colvec z(n_spcov, fill::zeros);
  arma::mat W_(m, n_spcov, fill::zeros);
  arma::mat Z_(m, n_cov, fill::zeros);
  arma::mat WW_CHOL_U(n1, n_spcov, fill::zeros);
  arma::mat ZZ_CHOL_U(n_cov, n_cov, fill::zeros);
  arma::vec Wy(n_spcov, fill::zeros);
  arma::vec ones_m(m, fill::ones);
  arma::mat eye_m(m, m, fill::eye);
  arma::mat ZZ(n_cov, n_cov, fill::zeros);
  arma::mat L_ZZ(n_cov, n_cov, fill::zeros);
  arma::mat L_ZZ_INV(n_cov, n_cov, fill::zeros);
  arma::mat ZZ_INV(n_cov, n_cov, fill::zeros);
  arma::mat P_ZZ(m, m, fill::zeros);
  arma::mat M_ZZ(m, m, fill::zeros);
  arma::mat WMW(n_spcov, n_spcov, fill::zeros);
  arma::vec WMy(n_spcov, fill::zeros);
  arma::vec Zy(n_cov, fill::zeros);
  arma::vec v_LS(n_cov, fill::zeros);
  arma::mat ZW(n_cov, n_spcov, fill::zeros);
  arma::mat P_ZW(n_cov, n_spcov, fill::zeros);
  arma::colvec fitted_(m, fill::zeros);
  arma::colvec residuals_(m, fill::zeros);
  arma::vec mse(nlambda, fill::zeros);
  
  /* Store output    */
  arma::field<vec> h_objval(nlambda);
  arma::field<vec> h_r_norm(nlambda);
  arma::field<vec> h_s_norm(nlambda);
  arma::field<vec> h_eps_pri(nlambda);
  arma::field<vec> h_eps_dual(nlambda);
  arma::field<vec> h_rho(nlambda);
  arma::vec niter(nlambda, fill::zeros);
  arma::vec eltime(nlambda, fill::zeros);
  arma::mat X(nlambda, n_spcov, fill::zeros);
  arma::mat V(nlambda, n_cov, fill::zeros);
  arma::vec conv(nlambda, fill::zeros);
  arma::mat residuals(nlambda, m, fill::zeros);
  arma::mat fitted(nlambda, m, fill::zeros);
  arma::mat Umat(nlambda, n_spcov, fill::zeros);
  arma::mat Zmat(nlambda, n_spcov, fill::zeros);
  
  /* Precompute relevant quantities         */
  Z_        = Z.each_col() % (ones_m * (1.0 / sqrtm));                // 19 Nov 25: la divisione per n è stata controllata ed è corretta
  W_        = W.each_col() % (ones_m * (1.0 / sqrtm));
  ZZ        = Z_.t() * Z_;
  L_ZZ      = (chol(ZZ)).t();
  L_ZZ_INV  = inv(trimatl(L_ZZ));
  ZZ_INV    = L_ZZ_INV.t() * L_ZZ_INV;
  P_ZZ      = Z_ * ZZ_INV * Z_.t();
  M_ZZ      = eye_m - P_ZZ;
  WMW       = W_.t() * M_ZZ * W_;
  WMy       = W_.t() * M_ZZ * y / static_cast<float>(sqrtm);
  Zy        = Z_.t() * y / static_cast<float>(sqrtm);
  v_LS      = L_ZZ_INV.t() * L_ZZ_INV * Zy;
  ZW        = Z_.t() * W_;
  P_ZW      = L_ZZ_INV.t() * L_ZZ_INV * ZW;
  WW_CHOL_U = chol_qr_fact(W_, m, n_spcov);
  ZZ_CHOL_U = chol_qr_fact(Z_, m, n_cov);
  y2        = as_scalar(y.t() * y) / static_cast<float>(m);
  Wy        = W_.t() * y / static_cast<float>(sqrtm);

  /* main loop  */
  for (int j=0; j<nlambda; j++) {
    /* perform linear regression with lasso penalty for the j-th lambda       */
    out = admm_lasso_cov_large_m_update_1lambda(W_, Z_, y, WMW, WMy, v_LS, P_ZW, WW_CHOL_U, Wy, y2,
                                                ZZ_CHOL_U, Zy, ZW, u, z, lambda(j),
                                                rho_adaptation, rho, tau, mu, reltol, abstol, maxiter, ping);

    /* Retrieve output      */
    u = as<arma::colvec>(out["u"]);
    z = as<arma::colvec>(out["z"]);
    x = as<arma::colvec>(out["x"]);
    v = as<arma::colvec>(out["v"]);
    
    /* Retrieve output      */
    fitted_    = W * x + Z * v;
    residuals_ = y - fitted_;
    
    /* Store output      */
    niter(j)         = out["niter"];
    eltime(j)        = out["eltime"];
    conv(j)          = out["convergence"];
    X.row(j)         = x.t();
    V.row(j)         = v.t();
    h_objval(j)      = as<arma::vec>(out["objval"]);
    h_r_norm(j)      = as<arma::vec>(out["r_norm"]);
    h_s_norm(j)      = as<arma::vec>(out["s_norm"]);
    h_eps_pri(j)     = as<arma::vec>(out["eps_pri"]);
    h_eps_dual(j)    = as<arma::vec>(out["eps_dual"]);
    h_rho(j)         = (as<arma::vec>(out["rho"])).subvec(0, niter(j)-1);
    rho              = out["rho.updated"];
    //residuals.row(j) = (as<arma::vec>(out["residuals"])).t();
    //fitted.row(j)    = (as<arma::vec>(out["fitted.values"])).t();
    residuals.row(j) = fitted_.t();
    fitted.row(j)    = residuals_.t();
    //mse(j)           = out["mse"];
    mse(j)           = as_scalar(residuals_.t() * residuals_) / (static_cast<float>(m));
    Umat.row(j)      = u.t();
    Zmat.row(j)      = z.t();
  }

  /* Find min MSE in-sample      */
  idx_min_mse = index_min(mse);
  min_mse     = mse(idx_min_mse);

  /* Get output         */
  output["sp.coef.path"]    = X;
  output["coef.path"]       = V;
  output["iternum"]         = niter;
  output["objfun"]          = h_objval;
  output["r_norm"]          = h_r_norm;
  output["s_norm"]          = h_s_norm;
  output["err_pri"]         = h_eps_pri;
  output["err_dual"]        = h_eps_dual;
  output["convergence"]     = conv;
  output["elapsedTime"]     = eltime;
  output["rho"]             = h_rho;
  output["lambda"]          = lambda;
  output["min.mse"]         = min_mse;
  output["indi.min.mse"]    = idx_min_mse;
  output["lambda.min"]      = lambda[idx_min_mse];
  output["sp.coefficients"] = (X.row(idx_min_mse)).t();
  output["coefficients"]    = (V.row(idx_min_mse)).t();
  output["mse"]             = mse;
  output["U"]               = Umat;
  output["Z"]               = Zmat;
  
  /* Return output      */
  return(output);
}

Rcpp::List admm_lasso_cov_large_n_fast(const arma::mat& W,
                                       const arma::mat& Z,
                                       const arma::vec& y,
                                       const arma::vec& lambda,
                                       const bool rho_adaptation,
                                       double rho,
                                       const double tau,
                                       const double mu,
                                       const double reltol, 
                                       const double abstol,
                                       const int maxiter,
                                       const int ping) {

  /* Variable delcaration           */
  double y2 = 0.0, min_mse = 0.0, sqrtm = 0.0;
  int n1 = 0;
  uword idx_min_mse;

  /* Get dimensions             */
  const int m       = W.n_rows;
  const int n_spcov = W.n_cols;
  const int n_cov   = Z.n_cols;
  
  /* Variable definition            */
  sqrtm = std::sqrt(static_cast<float>(m));

  /* Get number of lambda           */
  const int nlambda = lambda.n_elem;

  /* Get output lists          */
  List out, output;

  /* Variable definition            */
  if (m >= n_spcov) {
    n1 = n_spcov;
  } else {
    n1 = m;
  }

  /* Definition of vectors and matrices     */
  arma::colvec x(n_spcov, fill::zeros);
  arma::colvec v(n_cov, fill::zeros);
  arma::colvec u(n_spcov, fill::zeros);
  arma::colvec z(n_spcov, fill::zeros);
  arma::mat W_(m, n_spcov, fill::zeros);
  arma::mat Z_(m, n_cov, fill::zeros);
  arma::mat WW_CHOL_U(n1, n_spcov, fill::zeros);
  arma::mat ZZ_CHOL_U(n_cov, n_cov, fill::zeros);
  arma::vec Wy(n_spcov, fill::zeros);
  arma::vec ones_m(m, fill::ones);
  arma::mat eye_m(m, m, fill::eye);
  arma::mat ZZ(n_cov, n_cov, fill::zeros);
  arma::mat L_ZZ(n_cov, n_cov, fill::zeros);
  arma::mat L_ZZ_INV(n_cov, n_cov, fill::zeros);
  arma::mat ZZ_INV(n_cov, n_cov, fill::zeros);
  arma::mat P_ZZ(m, m, fill::zeros);
  arma::mat M_ZZ(m, m, fill::zeros);
  arma::mat WMW(n_spcov, n_spcov, fill::zeros);
  arma::vec WMy(n_spcov, fill::zeros);
  arma::vec Zy(n_cov, fill::zeros);
  arma::vec v_LS(n_cov, fill::zeros);
  arma::mat ZW(n_cov, n_spcov, fill::zeros);
  arma::mat P_ZW(n_cov, n_spcov, fill::zeros);
  arma::mat M_ZZW(m, n_spcov, fill::zeros);
  arma::mat M_ZZW2(m, m, fill::zeros);
  arma::colvec fitted_(m, fill::zeros);
  arma::colvec residuals_(m, fill::zeros);
  arma::vec mse(nlambda, fill::zeros);
  
  /* Store output    */
  arma::field<vec> h_objval(nlambda);
  arma::field<vec> h_r_norm(nlambda);
  arma::field<vec> h_s_norm(nlambda);
  arma::field<vec> h_eps_pri(nlambda);
  arma::field<vec> h_eps_dual(nlambda);
  arma::field<vec> h_rho(nlambda);
  arma::vec niter(nlambda, fill::zeros);
  arma::vec eltime(nlambda, fill::zeros);
  arma::mat X(nlambda, n_spcov, fill::zeros);
  arma::mat V(nlambda, n_cov, fill::zeros);
  arma::vec conv(nlambda, fill::zeros);
  arma::mat residuals(nlambda, m, fill::zeros);
  arma::mat fitted(nlambda, m, fill::zeros);
  arma::mat Umat(nlambda, n_spcov, fill::zeros);
  arma::mat Zmat(nlambda, n_spcov, fill::zeros);
  
  /* Precompute relevant quantities         */
  Z_        = Z.each_col() % (ones_m * (1.0 / sqrtm));
  W_        = W.each_col() % (ones_m * (1.0 / sqrtm));
  ZZ        = Z_.t() * Z_;
  L_ZZ      = (chol(ZZ)).t();
  L_ZZ_INV  = inv(trimatl(L_ZZ));
  ZZ_INV    = L_ZZ_INV.t() * L_ZZ_INV;
  P_ZZ      = Z_ * ZZ_INV * Z_.t();
  M_ZZ      = eye_m - P_ZZ;
  WMW       = W_.t() * M_ZZ * W_;
  WMy       = W_.t() * M_ZZ * y / static_cast<float>(sqrtm);            // 19Nov25: divisione per n corretta così come è ora
  Zy        = Z_.t() * y / static_cast<float>(sqrtm);
  v_LS      = L_ZZ_INV.t() * L_ZZ_INV * Zy;
  ZW        = Z_.t() * W_;                                              // 19Nov25: divisione per n corretta così come è ora
  P_ZW      = L_ZZ_INV.t() * L_ZZ_INV * ZW;
  WW_CHOL_U = chol_qr_fact(W_, m, n_spcov);
  ZZ_CHOL_U = chol_qr_fact(Z_, m, n_cov);
  y2        = as_scalar(y.t() * y) / static_cast<float>(m);
  Wy        = W_.t() * y / static_cast<float>(sqrtm);                   // 19Nov25: divisione per n corretta così come è ora
  M_ZZW     = M_ZZ * W_;
  M_ZZW2    = M_ZZW * M_ZZW.t();
  
  /* main loop  */
  for (int j=0; j<nlambda; j++) {
    /* perform linear regression with lasso penalty for the j-th lambda       */
    out = admm_lasso_cov_large_n_update_1lambda(W_, Z_, y, M_ZZW, M_ZZW2, WMW, WMy, v_LS, P_ZW, WW_CHOL_U, Wy, y2,
                                                ZZ_CHOL_U, Zy, ZW, u, z, lambda(j),
                                                rho_adaptation, rho, tau, mu, reltol, abstol, maxiter, ping);
    
    /* Retrieve output      */
    u = as<arma::colvec>(out["u"]);
    z = as<arma::colvec>(out["z"]);
    x = as<arma::colvec>(out["x"]);
    v = as<arma::colvec>(out["v"]);
    
    /* Retrieve output      */
    fitted_    = W * x + Z * v;
    residuals_ = y - fitted_;

    /* Store output      */
    niter(j)         = out["niter"];
    eltime(j)        = out["eltime"];
    conv(j)          = out["convergence"];
    X.row(j)         = x.t();
    V.row(j)         = v.t();
    h_objval(j)      = as<arma::vec>(out["objval"]);
    h_r_norm(j)      = as<arma::vec>(out["r_norm"]);
    h_s_norm(j)      = as<arma::vec>(out["s_norm"]);
    h_eps_pri(j)     = as<arma::vec>(out["eps_pri"]);
    h_eps_dual(j)    = as<arma::vec>(out["eps_dual"]);
    h_rho(j)         = (as<arma::vec>(out["rho"])).subvec(0, niter(j)-1);
    rho              = out["rho.updated"];
    //residuals.row(j) = (as<arma::vec>(out["residuals"])).t();
    //fitted.row(j)    = (as<arma::vec>(out["fitted.values"])).t();
    residuals.row(j) = fitted_.t();
    fitted.row(j)    = residuals_.t();
    //mse(j)           = out["mse"];
    mse(j)           = as_scalar(residuals_.t() * residuals_) / (static_cast<float>(m));
    Umat.row(j)      = u.t();
    Zmat.row(j)      = z.t();
  }

  /* Find min MSE in-sample      */
  idx_min_mse = index_min(mse);
  min_mse     = mse(idx_min_mse);

  /* Get output         */
  output["sp.coef.path"]    = X;
  output["coef.path"]       = V;
  output["iternum"]         = niter;
  output["objfun"]          = h_objval;
  output["r_norm"]          = h_r_norm;
  output["s_norm"]          = h_s_norm;
  output["err_pri"]         = h_eps_pri;
  output["err_dual"]        = h_eps_dual;
  output["convergence"]     = conv;
  output["elapsedTime"]     = eltime;
  output["rho"]             = h_rho;
  output["lambda"]          = lambda;
  output["min.mse"]         = min_mse;
  output["indi.min.mse"]    = idx_min_mse;
  output["lambda.min"]      = lambda[idx_min_mse];
  output["sp.coefficients"] = (X.row(idx_min_mse)).t();
  output["coefficients"]    = (V.row(idx_min_mse)).t();
  output["mse"]             = mse;
  output["U"]               = Umat;
  output["Z"]               = Zmat;
  
  /* Return output      */
  return(output);
}

Rcpp::List admm_adalasso_cov_large_m_update_1lambda(const arma::mat& W,
                                                    const arma::mat& Z,
                                                    const arma::vec& y,
                                                    const arma::mat& WMW,
                                                    const arma::vec& WMy,
                                                    const arma::vec& v_LS,
                                                    const arma::mat& P_ZW,
                                                    const arma::mat& WW_CHOL_U,
                                                    const arma::colvec& Wy,
                                                    const double y2,
                                                    const arma::mat& ZZ_CHOL_U, 
                                                    const arma::colvec& Zy,
                                                    const arma::mat& ZW,
                                                    const arma::vec& var_weights,
                                                    arma::colvec& u,
                                                    arma::colvec& z,
                                                    const double lambda,
                                                    const bool rho_adaptation,
                                                    double rho,
                                                    const double tau,
                                                    const double mu,
                                                    const double reltol, 
                                                    const double abstol,
                                                    const int maxiter,
                                                    const int ping) {
  
  /* Variable delcaration           */
  int k = 0;
  double rho_old = 0.0, elTime = 0.0, rss = 0.0, pen = 0.0;
  
  /* Get dimensions             */
  const int m       = W.n_rows;
  const int n_spcov = W.n_cols;
  const int n_cov   = Z.n_cols;
  const int dim_z   = z.n_elem;
    
  /* Definition of vectors and matrices     */
  arma::vec x(n_spcov, fill::zeros);
  arma::vec v(n_cov, fill::zeros);
  arma::colvec q(n_spcov, fill::zeros);
  arma::colvec z_old(dim_z, fill::zeros);
  arma::mat U(n_spcov, n_spcov, fill::zeros);
  arma::mat L(n_spcov, n_spcov, fill::zeros);
  arma::vec h_objval(maxiter, fill::zeros);
  arma::vec h_r_norm(maxiter, fill::zeros);
  arma::vec h_s_norm(maxiter, fill::zeros);
  arma::vec h_eps_pri(maxiter, fill::zeros);
  arma::vec h_eps_dual(maxiter, fill::zeros);
  arma::vec rho_store(maxiter+1, fill::zeros);
  //arma::colvec fitted(m, fill::zeros);
  //arma::colvec residuals(m, fill::zeros);
  
  /* ::::::::::::::::::::::::::::::::::::::::::::::
   Set the computational time                                 */
  wall_clock timer;
  timer.tic();
  
  /* store rho (only for adaptation)         */
  rho_store(0) = rho;
  rho_old      = rho;

  /* Precompute relevant quantities         */
  //U = glasso_factor_fast_large_m(ATA, FTF, rho); // returns upper
  U = adalasso_factor_fast_large_m(WMW, rho);
  L = U.t();
  
  /* main loop  */
  for (k=0; k<maxiter; k++) {

    // 4-1. update 'x'
    if (rho != rho_old) {
      //U = glasso_factor_fast_large_m(ATA, FTF, rho); // returns upper
      U = adalasso_factor_fast_large_m(WMW, rho);
      L = U.t();
    }
    q = WMy + rho * (z - u);           // temporary value
    x = solve(trimatu(U), solve(trimatl(L), q));         // updated of regression parameters associated to covariates W (sparsified covariates, x)
    v = v_LS - P_ZW * x;                                 // updated of regression parameters associated to covariates Z (covariates not sparsified, v)
    
    // 4-2. update 'z' with adaptation
    // see Boyd et al (2011) page 21
    z_old = z;
    z     = adalasso_prox(x + u, lambda * var_weights / rho);

    // 4-3. update 'u'
    u += x - z;
  
    // 4-3. dianostics, reporting
    rss         = lm_cov_rss_fast(WW_CHOL_U, ZZ_CHOL_U, ZW, Wy, Zy, y2, x, v, m);
    pen         = adalasso_penalty(x, var_weights);
    h_objval(k) = 0.5 *  rss / static_cast<float>(m) + lambda * pen;
    h_r_norm(k) = norm(adalasso_residual(x, z), 2);
    h_s_norm(k) = norm(adalasso_dual_residual(z, z_old, rho), 2);
    if (norm(x, 2) > norm(-z, 2)) {
      h_eps_pri(k) = std::sqrt(z.n_elem) * abstol + reltol * norm(x, 2);
    } else {
      h_eps_pri(k) = std::sqrt(z.n_elem) * abstol + reltol * norm(-z, 2);
    }
    h_eps_dual(k) = std::sqrt(z.n_elem) * abstol + reltol * norm(rho * u, 2);
    
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
  
  /* Get output         */
  List output;
  output["x"]      = x;
  output["v"]      = v;
  output["u"]      = u;
  output["z"]      = z;
  output["eltime"] = elTime;
  if (k < maxiter) {
    output["convergence"] = 1.0;
    output["niter"]       = k+1;
    output["objval"]      = h_objval.subvec(0, k);
    output["r_norm"]      = h_r_norm.subvec(0, k);
    output["s_norm"]      = h_s_norm.subvec(0, k);
    output["eps_pri"]     = h_eps_pri.subvec(0, k);
    output["eps_dual"]    = h_eps_dual.subvec(0, k);
    output["rho"]         = rho_store.subvec(0, k+1);
  } else {
    output["convergence"] = 0.0;
    output["niter"]       = maxiter;
    output["objval"]      = h_objval;
    output["r_norm"]      = h_r_norm;
    output["s_norm"]      = h_s_norm;
    output["eps_pri"]     = h_eps_pri;
    output["eps_dual"]    = h_eps_dual;
    output["rho"]         = rho_store;
  }
  //output["residuals"]     = residuals;
  //output["fitted.values"] = fitted;
  //output["mse"]           = mse;
  output["rho.updated"]   = rho;
  
  /* Return output      */
  return(output);
}

Rcpp::List admm_adalasso_cov_large_m_fast(const arma::mat& W,
                                          const arma::mat& Z,
                                          const arma::vec& y,
                                          const arma::vec& var_weights,
                                          const arma::vec& lambda,
                                          const bool rho_adaptation,
                                          double rho,
                                          const double tau,
                                          const double mu,
                                          const double reltol, 
                                          const double abstol,
                                          const int maxiter,
                                          const int ping) {

  /* Variable delcaration           */
  double y2 = 0.0, min_mse = 0.0, sqrtm = 0.0;
  int n1 = 0;
  uword idx_min_mse;

  /* Get dimensions             */
  const int m       = W.n_rows;
  const int n_spcov = W.n_cols;
  const int n_cov   = Z.n_cols;
  
  /* Variable definition            */
  sqrtm = std::sqrt(static_cast<float>(m));

  /* Get number of lambda           */
  const int nlambda = lambda.n_elem;

  /* Get output lists          */
  List out, output;

  /* Variable definition            */
  if (m >= n_spcov) {
    n1 = n_spcov;
  } else {
    n1 = m;
  }

  /* Definition of vectors and matrices     */
  arma::colvec x(n_spcov, fill::zeros);
  arma::colvec v(n_cov, fill::zeros);
  arma::mat W_(m, n_spcov, fill::zeros);
  arma::mat Z_(m, n_cov, fill::zeros);
  arma::mat WW_CHOL_U(n1, n_spcov, fill::zeros);
  arma::mat ZZ_CHOL_U(n_cov, n_cov, fill::zeros);
  arma::vec Wy(n_spcov, fill::zeros);
  arma::vec ones_m(m, fill::ones);
  arma::mat eye_m(m, m, fill::eye);
  arma::mat ZZ(n_cov, n_cov, fill::zeros);
  arma::mat L_ZZ(n_cov, n_cov, fill::zeros);
  arma::mat L_ZZ_INV(n_cov, n_cov, fill::zeros);
  arma::mat ZZ_INV(n_cov, n_cov, fill::zeros);
  arma::mat P_ZZ(m, m, fill::zeros);
  arma::mat M_ZZ(m, m, fill::zeros);
  arma::mat WMW(n_spcov, n_spcov, fill::zeros);
  arma::vec WMy(n_spcov, fill::zeros);
  arma::vec Zy(n_cov, fill::zeros);
  arma::vec v_LS(n_cov, fill::zeros);
  arma::mat ZW(n_cov, n_spcov, fill::zeros);
  arma::mat P_ZW(n_cov, n_spcov, fill::zeros);
  arma::colvec fitted_(m, fill::zeros);
  arma::colvec residuals_(m, fill::zeros);
  arma::vec mse(nlambda, fill::zeros);

  /* Store output    */
  arma::field<vec> h_objval(nlambda);
  arma::field<vec> h_r_norm(nlambda);
  arma::field<vec> h_s_norm(nlambda);
  arma::field<vec> h_eps_pri(nlambda);
  arma::field<vec> h_eps_dual(nlambda);
  arma::field<vec> h_rho(nlambda);
  arma::vec niter(nlambda, fill::zeros);
  arma::vec eltime(nlambda, fill::zeros);
  arma::mat X(nlambda, n_spcov, fill::zeros);
  arma::mat V(nlambda, n_cov, fill::zeros);
  arma::vec conv(nlambda, fill::zeros);
  arma::mat residuals(nlambda, m, fill::zeros);
  arma::mat fitted(nlambda, m, fill::zeros);
  
  /* Precompute relevant quantities         */
  Z_             = Z.each_col() % (ones_m * (1.0 / sqrtm));               // 19Nov25: divisione per n controllata ed è corretta
  W_             = W.each_col() % (ones_m * (1.0 / sqrtm));
  ZZ             = Z_.t() * Z_;
  L_ZZ           = (chol(ZZ)).t();
  L_ZZ_INV       = inv(trimatl(L_ZZ));
  ZZ_INV         = L_ZZ_INV.t() * L_ZZ_INV;
  P_ZZ           = Z_ * ZZ_INV * Z_.t();
  M_ZZ           = eye_m - P_ZZ;
  WMW            = W_.t() * M_ZZ * W_;
  WMy            = W_.t() * M_ZZ * y / static_cast<float>(sqrtm);
  Zy             = Z_.t() * y / static_cast<float>(sqrtm);
  v_LS           = L_ZZ_INV.t() * L_ZZ_INV * Zy;
  ZW             = Z_.t() * W_;
  P_ZW           = L_ZZ_INV.t() * L_ZZ_INV * ZW;
  WW_CHOL_U      = chol_qr_fact(W_, m, n_spcov);
  ZZ_CHOL_U      = chol_qr_fact(Z_, m, n_cov);
  y2             = as_scalar(y.t() * y) / static_cast<float>(m);
  Wy             = W_.t() * y / static_cast<float>(sqrtm);
  
  /* Initialize vectors u and z         */
  arma::colvec u(n_spcov, fill::zeros);
  arma::colvec z(n_spcov, fill::zeros);
  arma::mat Umat(nlambda, n_spcov, fill::zeros);
  arma::mat Zmat(nlambda, n_spcov, fill::zeros);

  /* main loop  */
  for (int j=0; j<nlambda; j++) {
    /* perform linear regression with lasso penalty for the j-th lambda       */
    out = admm_adalasso_cov_large_m_update_1lambda(W_, Z_, y, WMW, WMy, v_LS, P_ZW,
                                                   WW_CHOL_U, Wy, y2, ZZ_CHOL_U, Zy, ZW, var_weights,
                                                   u, z, lambda(j), rho_adaptation, rho, tau, mu,
                                                   reltol, abstol, maxiter, ping);

    /* Retrieve output      */
    u = as<arma::colvec>(out["u"]);
    z = as<arma::colvec>(out["z"]);
    x = as<arma::colvec>(out["x"]);
    v = as<arma::colvec>(out["v"]);
    
    /* Retrieve output      */
    fitted_    = W * x + Z * v;
    residuals_ = y - fitted_;
    
    /* Store output      */
    niter(j)         = out["niter"];
    eltime(j)        = out["eltime"];
    conv(j)          = out["convergence"];
    X.row(j)         = x.t();
    V.row(j)         = v.t();
    h_objval(j)      = as<arma::vec>(out["objval"]);
    h_r_norm(j)      = as<arma::vec>(out["r_norm"]);
    h_s_norm(j)      = as<arma::vec>(out["s_norm"]);
    h_eps_pri(j)     = as<arma::vec>(out["eps_pri"]);
    h_eps_dual(j)    = as<arma::vec>(out["eps_dual"]);
    h_rho(j)         = (as<arma::vec>(out["rho"])).subvec(0, niter(j)-1);
    rho              = out["rho.updated"];
    //residuals.row(j) = (as<arma::vec>(out["residuals"])).t();
    //fitted.row(j)    = (as<arma::vec>(out["fitted.values"])).t();
    //mse(j)           = out["mse"];
    residuals.row(j) = fitted_.t();
    fitted.row(j)    = residuals_.t();
    mse(j)           = as_scalar(residuals_.t() * residuals_) / (static_cast<float>(m));
    Umat.row(j)      = u.t();
    Zmat.row(j)      = z.t();
  }

  /* Find min MSE in-sample      */
  idx_min_mse = index_min(mse);
  min_mse     = mse(idx_min_mse);

  /* Get output         */
  output["sp.coef.path"]    = X;
  output["coef.path"]       = V;
  output["iternum"]         = niter;
  output["objfun"]          = h_objval;
  output["r_norm"]          = h_r_norm;
  output["s_norm"]          = h_s_norm;
  output["err_pri"]         = h_eps_pri;
  output["err_dual"]        = h_eps_dual;
  output["convergence"]     = conv;
  output["elapsedTime"]     = eltime;
  output["rho"]             = h_rho;
  output["lambda"]          = lambda;
  output["min.mse"]         = min_mse;
  output["indi.min.mse"]    = idx_min_mse;
  output["lambda.min"]      = lambda[idx_min_mse];
  output["sp.coefficients"] = (X.row(idx_min_mse)).t();
  output["coefficients"]    = (V.row(idx_min_mse)).t();
  output["mse"]             = mse;
  output["U"]               = Umat;
  output["Z"]               = Zmat;

  /* Return output      */
  return(output);
}

Rcpp::List admm_adalasso_cov_large_n_update_1lambda(const arma::mat& W,
                                                    const arma::mat& Z,
                                                    const arma::vec& y,
                                                    const arma::mat& M_ZZW,
                                                    const arma::mat& M_ZZW2,
                                                    const arma::mat& WMW,
                                                    const arma::vec& WMy,
                                                    const arma::vec& v_LS,
                                                    const arma::mat& P_ZW,
                                                    const arma::mat& WW_CHOL_U,
                                                    const arma::colvec& Wy,
                                                    const double y2,
                                                    const arma::mat& ZZ_CHOL_U, 
                                                    const arma::colvec& Zy,
                                                    const arma::mat& ZW,
                                                    const arma::vec& var_weights,
                                                    arma::colvec& u,
                                                    arma::colvec& z,
                                                    const double lambda,
                                                    const bool rho_adaptation,
                                                    double rho,
                                                    const double tau,
                                                    const double mu,
                                                    const double reltol, 
                                                    const double abstol,
                                                    const int maxiter,
                                                    const int ping) {
  

  /* Variable delcaration           */
  int k = 0;
  double rho_old = 0.0, elTime = 0.0, pen = 0.0, rss = 0.0;

  /* Get dimensions             */
  const int m       = W.n_rows;
  const int n_spcov = W.n_cols;
  const int n_cov   = Z.n_cols;
  const int dim_z   = z.n_elem;
    
  /* Definition of vectors and matrices     */
  arma::vec x(n_spcov, fill::zeros);
  arma::vec x_star(n_spcov, fill::zeros);
  arma::colvec v(n_cov, fill::zeros);
  arma::colvec q(n_spcov, fill::zeros);
  arma::colvec z_old(dim_z, fill::zeros);
  arma::mat U(m, m, fill::zeros);
  arma::mat L(m, m, fill::zeros);
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
  U = glasso_factor_cov_fast_large_n(M_ZZW2, rho, m); // returns upper
  L = U.t();
  
  /* main loop  */
  for (k=0; k<maxiter; k++) {

    // 4-1. update 'x'
    if (rho != rho_old) {
      U = glasso_factor_cov_fast_large_n(M_ZZW2, rho, m); // returns upper
      L = U.t();
    }
    q      = WMy + rho * (z - u); // temporary value
    x_star = q / rho;
    x      = x_star - M_ZZW.t() * solve(trimatu(U), solve(trimatl(L), M_ZZW * x_star)); // updated of regression parameters associated to covariates W
    v      = v_LS - P_ZW * x;                                 // updated of regression parameters associated to covariates Z (covariates not sparsified, v)

    // 4-2. update 'z' with adaptation
    // see Boyd et al (2011) page 21
    z_old = z;
    z     = adalasso_prox(x + u, lambda * var_weights / rho);

    // 4-3. update 'u'
    u = u + x - z;
  
    // 4-3. dianostics, reporting
    rss         = lm_cov_rss_fast(WW_CHOL_U, ZZ_CHOL_U, ZW, Wy, Zy, y2, x, v, m);
    pen         = adalasso_penalty(x, var_weights);
    h_objval(k) = 0.5 *  rss / static_cast<float>(m) + lambda * pen;
    h_r_norm(k) = norm(adalasso_residual(x, z), 2);
    h_s_norm(k) = norm(adalasso_dual_residual(z, z_old, rho), 2);
    if (norm(x, 2) > norm(-z, 2)) {
      h_eps_pri(k) = std::sqrt(z.n_elem) * abstol + reltol * norm(x, 2);
    } else {
      h_eps_pri(k) = std::sqrt(z.n_elem) * abstol + reltol * norm(-z, 2);
    }
    h_eps_dual(k) = std::sqrt(z.n_elem) * abstol + reltol * norm(rho * u, 2);
    
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
  
  /* Get output         */
  List output;
  output["x"]      = x;
  output["v"]      = v;
  output["u"]      = u;
  output["z"]      = z;
  output["eltime"] = elTime;
  if (k < maxiter) {
    output["convergence"] = 1.0;
    output["niter"]       = k+1;
    output["objval"]      = h_objval.subvec(0, k);
    output["r_norm"]      = h_r_norm.subvec(0, k);
    output["s_norm"]      = h_s_norm.subvec(0, k);
    output["eps_pri"]     = h_eps_pri.subvec(0, k);
    output["eps_dual"]    = h_eps_dual.subvec(0, k);
    output["rho"]         = rho_store.subvec(0, k+1);
  } else {
    output["convergence"] = 0.0;
    output["niter"]       = maxiter;
    output["objval"]      = h_objval;
    output["r_norm"]      = h_r_norm;
    output["s_norm"]      = h_s_norm;
    output["eps_pri"]     = h_eps_pri;
    output["eps_dual"]    = h_eps_dual;
    output["rho"]         = rho_store;
  }
  //output["residuals"]     = residuals;
  //output["fitted.values"] = fitted;
  //output["mse"]           = mse;
  output["rho.updated"]   = rho;
  
  /* Return output      */
  return(output);
}

Rcpp::List admm_adalasso_cov_large_n_fast(const arma::mat& W,
                                          const arma::mat& Z,
                                          const arma::vec& y,
                                          const arma::vec& var_weights,
                                          const arma::vec& lambda,
                                          const bool rho_adaptation,
                                          double rho,
                                          const double tau,
                                          const double mu,
                                          const double reltol, 
                                          const double abstol,
                                          const int maxiter,
                                          const int ping) {

  /* Variable delcaration           */
  double y2 = 0.0, min_mse = 0.0, sqrtm = 0.0;
  int n1 = 0;
  uword idx_min_mse;

  /* Get dimensions             */
  const int m       = W.n_rows;
  const int n_spcov = W.n_cols;
  const int n_cov   = Z.n_cols;
  
  /* Variable definition            */
  sqrtm = std::sqrt(static_cast<float>(m));

  /* Get number of lambda           */
  const int nlambda = lambda.n_elem;

  /* Get output lists          */
  List out, output;

  /* Variable definition            */
  if (m >= n_spcov) {
    n1 = n_spcov;
  } else {
    n1 = m;
  }

  /* Definition of vectors and matrices     */
  arma::colvec x(n_spcov, fill::zeros);
  arma::colvec v(n_cov, fill::zeros);
  arma::mat Z_(m, n_cov, fill::zeros);
  arma::mat WW_CHOL_U(n1, n_spcov, fill::zeros);
  arma::mat ZZ_CHOL_U(n_cov, n_cov, fill::zeros);
  arma::vec Wy(n_spcov, fill::zeros);
  arma::vec ones_m(m, fill::ones);
  arma::mat eye_m(m, m, fill::eye);
  arma::mat ZZ(n_cov, n_cov, fill::zeros);
  arma::mat L_ZZ(n_cov, n_cov, fill::zeros);
  arma::mat L_ZZ_INV(n_cov, n_cov, fill::zeros);
  arma::mat ZZ_INV(n_cov, n_cov, fill::zeros);
  arma::mat P_ZZ(m, m, fill::zeros);
  arma::mat M_ZZ(m, m, fill::zeros);
  arma::mat WMW(n_spcov, n_spcov, fill::zeros);
  arma::vec WMy(n_spcov, fill::zeros);
  arma::vec Zy(n_cov, fill::zeros);
  arma::vec v_LS(n_cov, fill::zeros);
  arma::mat ZW(n_cov, n_spcov, fill::zeros);
  arma::mat P_ZW(n_cov, n_spcov, fill::zeros);
  arma::mat M_ZZW(m, n_spcov, fill::zeros);
  arma::mat M_ZZW2(m, m, fill::zeros);
  arma::colvec fitted_(m, fill::zeros);
  arma::colvec residuals_(m, fill::zeros);
  arma::vec mse(nlambda, fill::zeros);
  
  /* Store output    */
  arma::field<vec> h_objval(nlambda);
  arma::field<vec> h_r_norm(nlambda);
  arma::field<vec> h_s_norm(nlambda);
  arma::field<vec> h_eps_pri(nlambda);
  arma::field<vec> h_eps_dual(nlambda);
  arma::field<vec> h_rho(nlambda);
  arma::vec niter(nlambda, fill::zeros);
  arma::vec eltime(nlambda, fill::zeros);
  arma::mat X(nlambda, n_spcov, fill::zeros);
  arma::mat V(nlambda, n_cov, fill::zeros);
  arma::vec conv(nlambda, fill::zeros);
  arma::mat residuals(nlambda, m, fill::zeros);
  arma::mat fitted(nlambda, m, fill::zeros);
  
  /* Precompute relevant quantities         */
  Z_        = Z.each_col() % (ones_m * (1.0 / sqrtm));
  //W         = W.each_col() % (ones_m * (1.0 / sqrtm));                // è corretto non dividere W per la numerosità campionaria m
  ZZ        = Z_.t() * Z_;
  L_ZZ      = (chol(ZZ)).t();
  L_ZZ_INV  = inv(trimatl(L_ZZ));
  ZZ_INV    = L_ZZ_INV.t() * L_ZZ_INV;
  P_ZZ      = Z_ * ZZ_INV * Z_.t();
  M_ZZ      = eye_m - P_ZZ;
  WMW       = W.t() * M_ZZ * W;                                         // 19Nov25: divisione per n controllata, è corretta così
  WMy       = W.t() * M_ZZ * y;
  Zy        = Z_.t() * y / static_cast<float>(sqrtm);
  v_LS      = L_ZZ_INV.t() * L_ZZ_INV * Zy;
  ZW        = Z_.t() * W / static_cast<float>(sqrtm);
  P_ZW      = L_ZZ_INV.t() * L_ZZ_INV * ZW;
  WW_CHOL_U = chol_qr_fact(W, m, n_spcov);
  ZZ_CHOL_U = chol_qr_fact(Z_, m, n_cov);
  y2        = as_scalar(y.t() * y) / static_cast<float>(m);
  Wy        = W.t() * y;
  M_ZZW     = M_ZZ * W;                                                 // 19Nov25: divisione per n controllata, è corretta così
  
  /* Define the F matrix         */
  M_ZZW2  = M_ZZW * M_ZZW.t();
  
  /* Initialize vectors u and z         */
  arma::colvec u(n_spcov, fill::zeros);
  arma::colvec z(n_spcov, fill::zeros);
  arma::mat Umat(nlambda, n_spcov, fill::zeros);
  arma::mat Zmat(nlambda, n_spcov, fill::zeros);
  
  /* main loop  */
  for (int j=0; j<nlambda; j++) {
    /* perform linear regression with lasso penalty for the j-th lambda       */
    out = admm_adalasso_cov_large_n_update_1lambda(W, Z_, y, M_ZZW, M_ZZW2,
                                                   WMW, WMy, v_LS, P_ZW, WW_CHOL_U, Wy, y2,
                                                   ZZ_CHOL_U, Zy, ZW, var_weights, u, z, lambda(j),
                                                   rho_adaptation, rho, tau, mu, reltol, abstol, maxiter, ping);

    /* Retrieve output      */
    u = as<arma::colvec>(out["u"]);
    z = as<arma::colvec>(out["z"]);
    x = as<arma::colvec>(out["x"]);
    v = as<arma::colvec>(out["v"]);
        
    /* Retrieve output      */
    fitted_    = W * x + Z * v;
    residuals_ = y - fitted_;
    
    /* Store output      */
    niter(j)         = out["niter"];
    eltime(j)        = out["eltime"];
    conv(j)          = out["convergence"];
    X.row(j)         = x.t();
    V.row(j)         = v.t();
    h_objval(j)      = as<arma::vec>(out["objval"]);
    h_r_norm(j)      = as<arma::vec>(out["r_norm"]);
    h_s_norm(j)      = as<arma::vec>(out["s_norm"]);
    h_eps_pri(j)     = as<arma::vec>(out["eps_pri"]);
    h_eps_dual(j)    = as<arma::vec>(out["eps_dual"]);
    h_rho(j)         = (as<arma::vec>(out["rho"])).subvec(0, niter(j)-1);
    rho              = out["rho.updated"];
    //residuals.row(j) = (as<arma::vec>(out["residuals"])).t();
    //fitted.row(j)    = (as<arma::vec>(out["fitted.values"])).t();
    //mse(j)           = out["mse"];
    residuals.row(j) = fitted_.t();
    fitted.row(j)    = residuals_.t();
    mse(j)           = as_scalar(residuals_.t() * residuals_) / (static_cast<float>(m));
    Umat.row(j)      = u.t();
    Zmat.row(j)      = z.t();
  }

  /* Find min MSE in-sample      */
  idx_min_mse = index_min(mse);
  min_mse     = mse(idx_min_mse);

  /* Get output         */
  output["sp.coef.path"]    = X;
  output["coef.path"]       = V;
  output["iternum"]         = niter;
  output["objfun"]          = h_objval;            
  output["r_norm"]          = h_r_norm;
  output["s_norm"]          = h_s_norm;
  output["err_pri"]         = h_eps_pri;
  output["err_dual"]        = h_eps_dual;
  output["convergence"]     = conv;
  output["elapsedTime"]     = eltime;
  output["rho"]             = h_rho;
  output["lambda"]          = lambda;
  output["min.mse"]         = min_mse;
  output["indi.min.mse"]    = idx_min_mse;
  output["lambda.min"]      = lambda[idx_min_mse];
  output["sp.coefficients"] = (X.row(idx_min_mse)).t();
  output["coefficients"]    = (V.row(idx_min_mse)).t();
  output["mse"]             = mse;
  output["U"]               = Umat;
  output["Z"]               = Zmat;
  
  /* Return output      */
  return(output);
}

Rcpp::List admm_lasso_cov_large_m(const arma::mat& W,
                                  const arma::mat& Z,
                                  const arma::colvec& y,
                                  arma::colvec& u, 
                                  arma::colvec& z,
                                  const double lambda,
                                  const bool rho_adaptation,
                                  double rho,
                                  const double tau,
                                  const double mu,
                                  const double reltol, 
                                  const double abstol,
                                  const int maxiter,
                                  const int ping) {
    
  /* Variable delcaration           */
  double sqrtn = 0.0, sqrtm = 0.0, rho_old = 0.0, elTime = 0.0, y2 = 0.0, rss = 0.0, pen = 0.0;
  int k = 0, n1 = 0;
  
  /* Get dimensions             */
  const int m       = W.n_rows;
  const int n_spcov = W.n_cols;
  const int n_cov   = Z.n_cols;
  
  /* Variable definition            */
  if (m >= n_spcov) {
    n1 = n_spcov;
  } else {
    n1 = m;
  }
    
  /* Variable definition            */
  sqrtn = std::sqrt(static_cast<float>(n_spcov));
  sqrtm = std::sqrt(static_cast<float>(m));
  
  /* Definition of vectors and matrices     */
  arma::vec ones_m(m, fill::ones);
  arma::mat eye_m(m, m, fill::eye);
  arma::colvec x(n_spcov, fill::zeros);
  arma::colvec v(n_cov, fill::zeros);
  arma::colvec q(n_spcov, fill::zeros);
  arma::colvec z_old(n_spcov, fill::zeros);
  arma::mat W_(m, n_spcov, fill::zeros);
  arma::mat Z_(m, n_cov, fill::zeros);
  arma::mat U(n_spcov, n_spcov, fill::zeros);
  arma::mat L(n_spcov, n_spcov, fill::zeros);
  arma::mat ZZ(n_cov, n_cov, fill::zeros);
  arma::mat L_ZZ(n_cov, n_cov, fill::zeros);
  arma::mat L_ZZ_INV(n_cov, n_cov, fill::zeros);
  arma::mat ZZ_INV(n_cov, n_cov, fill::zeros);
  arma::mat P_ZZ(m, m, fill::zeros);
  arma::mat M_ZZ(m, m, fill::zeros);
  arma::mat WMW(n_spcov, n_spcov, fill::zeros);
  arma::vec WMy(n_spcov, fill::zeros);
  arma::vec Zy(n_cov, fill::zeros);
  arma::vec v_LS(n_cov, fill::zeros);
  arma::mat ZW(n_cov, n_spcov, fill::zeros);
  arma::mat P_ZW(n_cov, n_spcov, fill::zeros);
  arma::vec h_objval(maxiter, fill::zeros);
  arma::vec h_r_norm(maxiter, fill::zeros);
  arma::vec h_s_norm(maxiter, fill::zeros);
  arma::vec h_eps_pri(maxiter, fill::zeros);
  arma::vec h_eps_dual(maxiter, fill::zeros);
  arma::vec rho_store(maxiter+1, fill::zeros);
  arma::mat WW_CHOL_U(n1, n_spcov, fill::zeros);
  arma::mat ZZ_CHOL_U(n_cov, n_cov, fill::zeros);
  arma::vec Wy(n_spcov, fill::zeros);
  
  /* ::::::::::::::::::::::::::::::::::::::::::::::
   Set the computational time                                 */
  wall_clock timer;
  timer.tic();
  
  /* store rho (only for adaptation)         */
  rho_store(0) = rho;
  rho_old      = rho;

  /* Precompute relevant quantities         */
  Z_        = Z.each_col() % (ones_m * (1.0 / sqrtm));
  W_        = W.each_col() % (ones_m * (1.0 / sqrtm));
  WW_CHOL_U = chol_qr_fact(W_, m, n_spcov);
  ZZ_CHOL_U = chol_qr_fact(Z_, m, n_cov);
  y2        = as_scalar(y.t() * y) / static_cast<float>(m);
  Wy        = W_.t() * y / static_cast<float>(sqrtm);                 // 19Nov25: la divisione per m è corretta così come è ora
  ZZ        = Z_.t() * Z_;
  L_ZZ      = (chol(ZZ)).t();
  L_ZZ_INV  = inv(trimatl(L_ZZ));
  ZZ_INV    = L_ZZ_INV.t() * L_ZZ_INV;
  P_ZZ      = Z_ * ZZ_INV * Z_.t();
  M_ZZ      = eye_m - P_ZZ;
  WMW       = W_.t() * M_ZZ * W_;
  WMy       = W_.t() * M_ZZ * y / static_cast<float>(sqrtm);
  Zy        = Z_.t() * y / static_cast<float>(sqrtm);
  v_LS      = L_ZZ_INV.t() * L_ZZ_INV * Zy;
  ZW        = Z_.t() * W_;
  P_ZW      = L_ZZ_INV.t() * L_ZZ_INV * ZW;
  U         = lasso_factor_cov_fast_large_m(WMW, rho);    // returns upper
  L         = U.t();
  
  /* main loop  */
  for (k=0; k<maxiter; k++) {
        
    // 4-1. update 'x'
    if (rho != rho_old) {
      U = lasso_factor_cov_fast_large_m(WMW, rho);   // returns upper
      L = U.t();
    }
    q = WMy + rho * (z - u);                             // temporary value
    x = solve(trimatu(U), solve(trimatl(L), q));         // updated of regression parameters associated to covariates W (sparsified covariates, x)
    v = v_LS - P_ZW * x;                                 // updated of regression parameters associated to covariates Z (covariates not sparsified, v)
    
    // 4-2. update 'z' with adaptation
    // see Boyd et al (2011) page 21
    z_old = z;
    z     = lasso_prox(x + u, lambda / rho);
    
    // 4-3. update 'u'
    u = u + x - z;

    // 4-3. dianostics, reporting
    rss         = lm_cov_rss_fast(WW_CHOL_U, ZZ_CHOL_U, ZW, Wy, Zy, y2, x, v, m);
    pen         = lasso_penalty(x);
    h_objval(k) = 0.5 *  rss / static_cast<float>(m) + lambda * pen;
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
            Rcpp::Rcout << "ADMM with adaptation for LASSO with covariates is running!" << std::endl;
            Rcpp::Rcout << "Iteration n.: " << k+1 << " of " << maxiter << "\n" << std::endl;
          }
        } else {
          if (((k+1) % ping) == 0) {
            Rcpp::Rcout << "\n\n\n" << std::endl;
            Rcpp::Rcout << "ADMM for LASSO with covariates is running!" << std::endl;
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
  output["x"]  = x;
  output["v"]  = v;
  output["u"]  = u;
  output["z"]  = z;
  if (k < maxiter) {
    output["convergence"] = 1.0;
    output["niter"]       = k+1;
    output["objval"]      = h_objval.subvec(0, k);
    output["r_norm"]      = h_r_norm.subvec(0, k);
    output["s_norm"]      = h_s_norm.subvec(0, k);
    output["eps_pri"]     = h_eps_pri.subvec(0, k);
    output["eps_dual"]    = h_eps_dual.subvec(0, k);
    output["rho"]         = rho_store.subvec(0, k+1);
  } else {
    output["convergence"] = 0.0;
    output["niter"]       = maxiter;
    output["objval"]      = h_objval;
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

Rcpp::List admm_lasso_cov_large_n(const arma::mat& W,
                                  const arma::mat& Z,
                                  const arma::colvec& y,
                                  arma::colvec& u,
                                  arma::colvec& z,
                                  const double lambda,
                                  const bool rho_adaptation,
                                  double rho,
                                  const double tau,
                                  const double mu,
                                  const double reltol, 
                                  const double abstol,
                                  const int maxiter,
                                  const int ping) {
    
  /* Variable delcaration           */
  double sqrtm = 0.0, sqrtn = 0.0, rho_old = 0.0, elTime = 0.0, y2 = 0.0, pen = 0.0, rss = 0.0;
  int n1 = 0, k = 0;

  /* Get dimensions             */
  const int m       = W.n_rows;
  const int n_spcov = W.n_cols;
  const int n_cov   = Z.n_cols;
  
  /* Variable definition            */
  sqrtn = std::sqrt(static_cast<float>(n_spcov));
  sqrtm = std::sqrt(static_cast<float>(m));

  /* Get output lists          */
  List out, output;

  /* Variable definition            */
  if (m >= n_spcov) {
    n1 = n_spcov;
  } else {
    n1 = m;
  }

  /* Definition of vectors and matrices     */
  arma::colvec x(n_spcov, fill::zeros);
  arma::colvec v(n_cov, fill::zeros);
  arma::mat Z_(m, n_cov, fill::zeros);
  arma::mat WW_CHOL_U(n1, n_spcov, fill::zeros);
  arma::mat ZZ_CHOL_U(n_cov, n_cov, fill::zeros);
  arma::vec Wy(n_spcov, fill::zeros);
  arma::vec ones_m(m, fill::ones);
  arma::mat eye_m(m, m, fill::eye);
  arma::mat ZZ(n_cov, n_cov, fill::zeros);
  arma::mat L_ZZ(n_cov, n_cov, fill::zeros);
  arma::mat L_ZZ_INV(n_cov, n_cov, fill::zeros);
  arma::mat ZZ_INV(n_cov, n_cov, fill::zeros);
  arma::mat P_ZZ(m, m, fill::zeros);
  arma::mat M_ZZ(m, m, fill::zeros);
  arma::mat WMW(n_spcov, n_spcov, fill::zeros);
  arma::vec WMy(n_spcov, fill::zeros);
  arma::vec Zy(n_cov, fill::zeros);
  arma::vec v_LS(n_cov, fill::zeros);
  arma::mat ZW(n_cov, n_spcov, fill::zeros);
  arma::mat P_ZW(n_cov, n_spcov, fill::zeros);
  arma::mat WW(m, m, fill::zeros);
  arma::mat M_ZZW(m, n_spcov, fill::zeros);
  arma::mat M_ZZW2(m, m, fill::zeros);
  
  /* Definition of vectors and matrices     */
  arma::colvec q(n_spcov, fill::zeros);
  arma::colvec z_old(n_spcov, fill::zeros);
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
  Z_        = Z.each_col() % (ones_m * (1.0 / sqrtm));
  ZZ        = Z_.t() * Z_;
  L_ZZ      = (chol(ZZ)).t();
  L_ZZ_INV  = inv(trimatl(L_ZZ));
  ZZ_INV    = L_ZZ_INV.t() * L_ZZ_INV;
  P_ZZ      = Z_ * ZZ_INV * Z_.t();
  M_ZZ      = eye_m - P_ZZ;
  WMW       = W.t() * M_ZZ * W;                                           // 19Nov25: la divisione per n è corretta così come è ora
  WMy       = W.t() * M_ZZ * y;
  Zy        = Z_.t() * y / static_cast<float>(sqrtm);
  v_LS      = L_ZZ_INV.t() * L_ZZ_INV * Zy;
  ZW        = Z_.t() * W / static_cast<float>(sqrtm);
  P_ZW      = L_ZZ_INV.t() * L_ZZ_INV * ZW;
  WW_CHOL_U = chol_qr_fact(W, m, n_spcov);
  ZZ_CHOL_U = chol_qr_fact(Z_, m, n_cov);
  y2        = as_scalar(y.t() * y) / static_cast<float>(m);
  Wy        = W.t() * y;
  M_ZZW     = M_ZZ * W;                                                   // 19Nov25: la divisione per n è corretta così come è ora
  M_ZZW2    = M_ZZW * M_ZZW.t();
    
  /* Precompute relevant quantities         */
  U = lasso_factor_cov_fast_large_n(M_ZZW2, rho, m);    // returns upper
  L = U.t();
  
  /* main loop  */
  for (k=0; k<maxiter; k++) {
        
    // 4-1. update 'x'
    if (rho != rho_old) {
      U = lasso_factor_cov_fast_large_n(M_ZZW2, rho, m);   // returns upper
      L = U.t();
    }
    q = WMy + rho * (z - u);                             // temporary value
    x = q / rho - (M_ZZW.t() * solve(trimatu(U), solve(trimatl(L), M_ZZW * q))) / rho;         // updated of regression parameters associated to covariates W
    v = v_LS - P_ZW * x;                                 // updated of regression parameters associated to covariates Z (covariates not sparsified, v)
    
    // 4-2. update 'z' with adaptation
    // see Boyd et al (2011) page 21
    z_old = z;
    z     = lasso_prox(x + u, lambda / rho);
    
    // 4-3. update 'u'
    u = u + x - z;

    // 4-3. dianostics, reporting
    rss         = lm_cov_rss_fast(WW_CHOL_U, ZZ_CHOL_U, ZW, Wy, Zy, y2, x, v, m);
    pen         = lasso_penalty(x);
    h_objval(k) = 0.5 *  rss / static_cast<float>(m) + lambda * pen;
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
            Rcpp::Rcout << "ADMM with adaptation for LASSO with covariates is running!" << std::endl;
            Rcpp::Rcout << "Iteration n.: " << k+1 << " of " << maxiter << "\n" << std::endl;
          }
        } else {
          if (((k+1) % ping) == 0) {
            Rcpp::Rcout << "\n\n\n" << std::endl;
            Rcpp::Rcout << "ADMM for LASSO with covariates is running!" << std::endl;
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
  output["x"]  = x;               // coefficient function
  output["v"]  = v;
  output["u"]  = u;
  output["z"]  = z;
  if (k < maxiter) {
    output["convergence"] = 1.0;
    output["niter"]       = k+1;
    output["objval"]      = h_objval.subvec(0, k);
    output["r_norm"]      = h_r_norm.subvec(0, k);
    output["s_norm"]      = h_s_norm.subvec(0, k);
    output["eps_pri"]     = h_eps_pri.subvec(0, k);
    output["eps_dual"]    = h_eps_dual.subvec(0, k);
    output["rho"]         = rho_store.subvec(0, k+1);
  } else {
    output["convergence"] = 0.0;
    output["niter"]       = maxiter;
    output["objval"]      = h_objval;        
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












/* end of file          */



