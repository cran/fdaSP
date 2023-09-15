// Utility routines

// Authors:
//           Bernardi Mauro, University of Padova
//           Last update: July 12, 2022

// List of implemented methods:



// [[Rcpp::depends(RcppArmadillo)]]
#include "linreg_ADMM_cov.h"


// LASSO penalty: evaluate the RIDGE matrix, returns the Cholesky (Upper triangular matrix)
arma::mat lasso_factor_cov_fast_large_m(arma::mat XMX, double rho) {
  
  /* Variable delcaration           */
  const int n = XMX.n_cols;
  arma::vec eye_n(n, fill::ones);
  arma::mat U(n, n, fill::zeros);
  
  /* Compute the Cholesky factor of the RIDGE matrix        */
  U = chol(diagmat(eye_n) * rho + XMX);
  
  /* return output         */
  return(U);
}

arma::mat lasso_factor_cov_fast_large_n(arma::mat XX, double rho, const int m) {

  /* Definition of vectors and matrices     */
  arma::mat U(m, m, fill::zeros);
  arma::vec ones_m(m, fill::ones);
  
  /* Compute the Cholesky factor of the RIDGE matrix        */
  U = chol(diagmat(ones_m * rho * static_cast<float>(m)) + XX);

  /* return output         */
  return(U);
}

// Group-LASSO penalty: evaluate the RIDGE matrix, returns the Cholesky (Upper triangular matrix)
arma::mat glasso_factor_cov_fast_large_m(arma::mat XMX, arma::mat FTF, double rho) {

  /* Variable delcaration           */
  const int n = XMX.n_cols;
  arma::mat U(n, n, fill::zeros);
  
  /* Compute the Cholesky factor of the RIDGE matrix        */
  U = chol(XMX + rho * diagmat(FTF));
  
  /* return output         */
  return(U);
}

arma::mat glasso_factor_cov_fast_large_n(arma::mat XMX, double rho, const int m) {

  /* Definition of vectors and matrices     */
  arma::mat U(m, m, fill::zeros);
  arma::vec ones_m(m, fill::ones);
  
  /* Compute the Cholesky factor of the RIDGE matrix        */
  U = chol(diagmat(ones_m * rho * static_cast<float>(m)) + XMX);

  /* return output         */
  return(U);
}

// Linear regression model with covariates with LASSO penalty:
// evaluate the objective function
// this is the fast version
double lasso_cov_objfun_fast(arma::mat XX_CHOL_U, arma::mat ZZ_CHOL_U, arma::mat ZX, arma::colvec Xy, arma::colvec Zy, const double y2,
                             const double lambda, arma::colvec x, arma::colvec v, arma::colvec z, const int m) {
  
  /* Variable delcaration           */
  int n1;
  double ll = 0.0, objfun = 0.0;
  
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
  
  // evaluate the quadratic form (negative log-likelihood)
  // previously divided by m
  if (m >= n_spcov) {
    tmp_x = trimatu(XX_CHOL_U) * x;
  } else {
    tmp_x = trimatu(XX_CHOL_U.submat(0, 0, m-1, m-1)) * x.subvec(0, m-1) + XX_CHOL_U.submat(0, m, m-1, n_spcov-1) * x.subvec(m, n_spcov-1);
  }
  tmp_z = trimatu(ZZ_CHOL_U) * v;
  ll    = as_scalar(tmp_x.t() * tmp_x) + as_scalar(tmp_z.t() * tmp_z) + y2 - 2.0 * as_scalar(x.t() * Xy) - 2.0 * as_scalar(v.t() * Zy) - 2.0 * as_scalar(v.t() * ZX * x);
  
  // evaluate the objective function
  objfun = 0.5 * ll + lambda * norm(z, 1);
  
  /* Return output      */
  return(objfun);
}

// Linear regression model with covariates with LASSO penalty:
// evaluate the risk (logl) function
// this is the fast version
double linreg_cov_logl_fast(arma::mat XX_CHOL_U, arma::mat ZZ_CHOL_U, arma::mat ZX,
                            arma::colvec Xy, arma::colvec Zy, const double y2,
                            arma::colvec x, arma::colvec v, const int m) {
  
  /* Variable delcaration           */
  int n1;
  double ll = 0.0, objfun = 0.0;
  
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
  
  // evaluate the quadratic form (negative log-likelihood)
  // previously divided by m
  if (m >= n_spcov) {
    tmp_x = trimatu(XX_CHOL_U) * x;
  } else {
    tmp_x = trimatu(XX_CHOL_U.submat(0, 0, m-1, m-1)) * x.subvec(0, m-1) + XX_CHOL_U.submat(0, m, m-1, n_spcov-1) * x.subvec(m, n_spcov-1);
  }
  tmp_z = trimatu(ZZ_CHOL_U) * v;
  ll    = as_scalar(tmp_x.t() * tmp_x) + as_scalar(tmp_z.t() * tmp_z) + y2 - 2.0 * as_scalar(x.t() * Xy) - 2.0 * as_scalar(v.t() * Zy) - 2.0 * as_scalar(v.t() * ZX * x);
  
  // evaluate the objective function
  objfun = 0.5 * ll;
  
  /* Return output      */
  return(objfun);
}

// Linear regression model with covariates with LASSO penalty:
// evaluate the lasso penalty function
// this is the fast version
double lasso_cov_penfun(const arma::colvec z, const double lambda) {
  
  /* Variable delcaration           */
  double penfun = 0.0;
  
  // evaluate the objective function
  penfun = lambda * as_scalar(norm(z, 1));
  
  /* Return output      */
  return(penfun);
}

// Linear regression model with covariates with group-LASSO penalty:
// evaluate the group lasso penalty function
// this is the fast version
double glasso_cov_penfun(const arma::vec Glen, const int G, const arma::colvec z, const double lambda) {
  
  /* Variable delcaration           */
  double pen = 0.0, penfun = 0.0;
  uword g_id_init, g_id_end;
  
  // evaluate the penalty function
  pen       = 0.0;
  g_id_init = 0;
  for (int g=0; g<G; g++) {
    g_id_end  = g_id_init + Glen(g) - 1;
    pen       = pen + as_scalar(norm(z.subvec(g_id_init, g_id_end), 2));
    g_id_init = g_id_end + 1;
  }
  
  // evaluate the objective function
  penfun = lambda * pen;
  
  /* Return output      */
  return(penfun);
}

// Linear regression model with sparse GLASSO penalty: evaluate the penalty function
double spglasso_cov_penfun(const arma::vec Glen, const int G, const int n, arma::colvec z,
                           const double lambda, const double alpha) {
  
  double pen = 0.0;
  uword g_id_init, g_id_end;
  
  pen       = 0.0;
  g_id_init = 0;
  for (int g=0; g<G; g++) {
    g_id_end  = g_id_init + Glen(g) - 1;
    pen       = pen + as_scalar(norm(z.subvec(g_id_init, g_id_end), 2));
    g_id_init = g_id_end + 1;
  }
  pen      = lambda * (1.0 - alpha) * pen;
  g_id_end = g_id_init + n - 1;
  pen      = pen + lambda * alpha * norm(z.subvec(g_id_init, g_id_end), 1);
  
  /* Return output      */
  return(pen);
}

// Linear regression model with adaptive-LASSO penalty: evaluate the penalty function
double adalasso_cov_penfun(const arma::colvec z, const double lambda, const arma::colvec var_weights) {
  
  /* Variable delcaration           */
  double pen = 0.0, objfun = 0.0;
  
  // evaluate the penalty function
  pen = as_scalar(norm(diagmat(var_weights) * z, 1));
  
  // evaluate the objective function
  objfun = lambda * pen;
  
  /* Return output      */
  return(objfun);
}

/*
    LASSO via ADMM with covariate not penalized, just one lambda, large m.
*/
Rcpp::List admm_lasso_cov_large_m_update_1lambda(arma::mat& W, arma::mat Z, arma::vec& y,
                                                 const arma::mat WMW, const arma::vec WMy, const arma::vec v_LS, const arma::mat P_ZW,
                                                 const arma::mat& WW_CHOL_U, const arma::colvec& Wy, const double y2,
                                                 const arma::mat& ZZ_CHOL_U, const arma::colvec& Zy, const arma::mat ZW,
                                                 arma::colvec u, arma::colvec z, const double lambda,
                                                 bool rho_adaptation, double rho, const double tau, const double mu,
                                                 const double reltol, const double abstol, const int maxiter, const int ping) {
  
  /* Variable delcaration           */
  int k;
  double rho_old, elTime = 0.0, mse, sqrtn;
  
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
    //h_objval(k) = lasso_cov_objfun_fast(WW_CHOL_U, ZZ_CHOL_U, ZW, Wy, Zy, y2, lambda, x, v, z, m);
    h_objval(k)  = linreg_cov_logl_fast(WW_CHOL_U, ZZ_CHOL_U, ZW, Wy, Zy, y2, x, v, m);
    h_objval(k) += lasso_cov_penfun(z, lambda);
    h_r_norm(k)  = norm(x - z, 2);
    h_s_norm(k)  = norm(-rho * (z - z_old), 2);
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
  
  /* ::::::::::::::::::::::::::::::::::::::::::::::
   Compute post-estimation quantities                   */
  fitted    = W * x + Z * v;
  residuals = y - fitted;
  mse       = as_scalar(residuals.t() * residuals) / (static_cast<float>(m));
  
  /* Get output         */
  List output;
  output["x"]      = x;               // coefficient function
  output["v"]      = v;               // coefficient function
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
    LASSO via ADMM with covariate not penalized, just one lambda, large n.
*/
Rcpp::List admm_lasso_cov_large_n_update_1lambda(arma::mat& W, arma::mat Z, arma::vec& y, const arma::mat WW, const arma::mat M_ZZW, const arma::mat M_ZZW2,
                                                 const arma::mat WMW, const arma::vec WMy, const arma::vec v_LS, const arma::mat P_ZW,
                                                 const arma::mat& WW_CHOL_U, const arma::colvec& Wy, const double y2,
                                                 const arma::mat& ZZ_CHOL_U, const arma::colvec& Zy, const arma::mat ZW,
                                                 arma::colvec u, arma::colvec z, const double lambda,
                                                 bool rho_adaptation, double rho, const double tau, const double mu,
                                                 const double reltol, const double abstol, const int maxiter, const int ping) {
  
  /* Variable delcaration           */
  int k;
  double rho_old, elTime = 0.0, mse, sqrtn;
  
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
    //h_objval(k) = lasso_cov_objfun_fast(WW_CHOL_U, ZZ_CHOL_U, ZW, Wy, Zy, y2, lambda, x, v, z, m);
    h_objval(k)  = linreg_cov_logl_fast(WW_CHOL_U, ZZ_CHOL_U, ZW, Wy, Zy, y2, x, v, m);
    h_objval(k) += lasso_cov_penfun(z, lambda);
    h_r_norm(k)  = norm(x - z, 2);
    h_s_norm(k)  = norm(-rho * (z - z_old), 2);
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
  
  /* ::::::::::::::::::::::::::::::::::::::::::::::
   Compute post-estimation quantities                   */
  fitted    = W * x + Z * v;
  residuals = y - fitted;
  mse       = as_scalar(residuals.t() * residuals) / (static_cast<float>(m));
  
  /* Get output         */
  List output;
  output["x"]      = x;               // coefficient function
  output["v"]      = v;               // coefficient function
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
    LASSO via ADMM with covariate not penalized, fast routine to compute the whole path
          for large m where m denotes the number of observations
*/
Rcpp::List admm_lasso_cov_large_m_fast(arma::mat& W, arma::mat Z, arma::vec& y,
                                       const arma::vec lambda, bool rho_adaptation, double rho, const double tau, const double mu,
                                       const double reltol, const double abstol, const int maxiter, const int ping) {

  /* Variable delcaration           */
  double y2, min_mse, sqrtm;
  int n1;
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
  arma::vec mse(nlambda, fill::zeros);
  arma::mat Umat(nlambda, n_spcov, fill::zeros);
  arma::mat Zmat(nlambda, n_spcov, fill::zeros);
  
  /* Precompute relevant quantities         */
  Z         = Z.each_col() % (ones_m * (1.0 / sqrtm));
  W         = W.each_col() % (ones_m * (1.0 / sqrtm));
  ZZ        = Z.t() * Z;
  L_ZZ      = (chol(ZZ)).t();
  L_ZZ_INV  = inv(trimatl(L_ZZ));
  ZZ_INV    = L_ZZ_INV.t() * L_ZZ_INV;
  P_ZZ      = Z * ZZ_INV * Z.t();
  M_ZZ      = eye_m - P_ZZ;
  WMW       = W.t() * M_ZZ * W;
  WMy       = W.t() * M_ZZ * y / static_cast<float>(sqrtm);
  Zy        = Z.t() * y / static_cast<float>(sqrtm);
  v_LS      = L_ZZ_INV.t() * L_ZZ_INV * Zy;
  ZW        = Z.t() * W;
  P_ZW      = L_ZZ_INV.t() * L_ZZ_INV * ZW;
  WW_CHOL_U = chol_qr_fact(W, m, n_spcov);
  ZZ_CHOL_U = chol_qr_fact(Z, m, n_cov);
  y2        = as_scalar(y.t() * y) / static_cast<float>(m);
  Wy        = W.t() * y / static_cast<float>(sqrtm);

  /* main loop  */
  for (int j=0; j<nlambda; j++) {
    /* perform linear regression with lasso penalty for the j-th lambda       */
    out = admm_lasso_cov_large_m_update_1lambda(W, Z, y, WMW, WMy, v_LS, P_ZW, WW_CHOL_U, Wy, y2,
                                                ZZ_CHOL_U, Zy, ZW, u, z, lambda(j),
                                                rho_adaptation, rho, tau, mu, reltol, abstol, maxiter, ping);

    /* Retrieve output      */
    u = as<arma::colvec>(out["u"]);
    z = as<arma::colvec>(out["z"]);
    x = as<arma::colvec>(out["x"]);
    v = as<arma::colvec>(out["v"]);

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
  output["sp.coef.path"]    = X;
  output["coef.path"]       = V;
  output["iternum"]         = niter;               // number of iterations
  output["objfun"]          = h_objval;            // |x|_1
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

/*
    LASSO via ADMM with covariate not penalized, fast routine to compute the whole path
          for large n where n denotes the number of parameters
*/
Rcpp::List admm_lasso_cov_large_n_fast(arma::mat& W, arma::mat Z, arma::vec& y,
                                       const arma::vec lambda, bool rho_adaptation, double rho, const double tau, const double mu,
                                       const double reltol, const double abstol, const int maxiter, const int ping) {

  /* Variable delcaration           */
  double y2, min_mse, sqrtm;
  int n1;
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
  arma::vec mse(nlambda, fill::zeros);
  arma::mat Umat(nlambda, n_spcov, fill::zeros);
  arma::mat Zmat(nlambda, n_spcov, fill::zeros);
  
  /* Precompute relevant quantities         */
  Z         = Z.each_col() % (ones_m * (1.0 / sqrtm));
  ZZ        = Z.t() * Z;
  L_ZZ      = (chol(ZZ)).t();
  L_ZZ_INV  = inv(trimatl(L_ZZ));
  ZZ_INV    = L_ZZ_INV.t() * L_ZZ_INV;
  P_ZZ      = Z * ZZ_INV * Z.t();
  M_ZZ      = eye_m - P_ZZ;
  WMW       = W.t() * M_ZZ * W;
  WMy       = W.t() * M_ZZ * y / static_cast<float>(m);
  Zy        = Z.t() * y / static_cast<float>(sqrtm);
  v_LS      = L_ZZ_INV.t() * L_ZZ_INV * Zy;
  ZW        = Z.t() * W / static_cast<float>(sqrtm);
  P_ZW      = L_ZZ_INV.t() * L_ZZ_INV * ZW;
  WW_CHOL_U = chol_qr_fact(W, m, n_spcov);
  ZZ_CHOL_U = chol_qr_fact(Z, m, n_cov);
  y2        = as_scalar(y.t() * y) / static_cast<float>(m);
  Wy        = W.t() * y / static_cast<float>(m);
  M_ZZW     = M_ZZ * W;
  M_ZZW2    = M_ZZW * M_ZZW.t();
  
  /* main loop  */
  for (int j=0; j<nlambda; j++) {
    /* perform linear regression with lasso penalty for the j-th lambda       */
    out = admm_lasso_cov_large_n_update_1lambda(W, Z, y, WW, M_ZZW, M_ZZW2, WMW, WMy, v_LS, P_ZW, WW_CHOL_U, Wy, y2,
                                                ZZ_CHOL_U, Zy, ZW, u, z, lambda(j),
                                                rho_adaptation, rho, tau, mu, reltol, abstol, maxiter, ping);
    
    /* Retrieve output      */
    u = as<arma::colvec>(out["u"]);
    z = as<arma::colvec>(out["z"]);
    x = as<arma::colvec>(out["x"]);
    v = as<arma::colvec>(out["v"]);

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
  output["sp.coef.path"]    = X;
  output["coef.path"]       = V;
  output["iternum"]         = niter;               // number of iterations
  output["objfun"]          = h_objval;            // |x|_1
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

/*
    LASSO via ADMM with covariate not penalized, fast routine to compute the whole path
            wrap for m and n (m denotes the number of observations, while n denotes the number of parameters)
*/
//' @name  fdaSP.internal
//' @keywords internal
// [[Rcpp::export(.admm_lasso_cov_fast)]]
Rcpp::List admm_lasso_cov_fast(arma::mat& W, arma::mat Z, arma::vec& y,
                               const arma::vec lambda, bool rho_adaptation, double rho, const double tau, const double mu,
                               const double reltol, const double abstol, const int maxiter, const int ping) {

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

/*
    Group-LASSO via ADMM with covariate not penalized, just one lambda, large m.
*/
Rcpp::List admm_glasso_cov_large_m_update_1lambda(arma::mat W, arma::mat Z, arma::vec& y,
                                                  const arma::colvec Glen, const int G, const arma::sp_mat F, const arma::mat FTF,
                                                  const arma::mat WMW, const arma::vec WMy, const arma::vec v_LS, const arma::mat P_ZW,
                                                  const arma::mat& WW_CHOL_U, const arma::colvec& Wy, const double y2,
                                                  const arma::mat& ZZ_CHOL_U, const arma::colvec& Zy, const arma::mat ZW,
                                                  arma::colvec u, arma::colvec z, const double lambda,
                                                  bool rho_adaptation, double rho, const double tau, const double mu,
                                                  const double reltol, const double abstol, const int maxiter, const int ping) {
  
  /* Variable delcaration           */
  int k;
  double rho_old, elTime = 0.0, mse;
  uword g_id_init, g_id_end;
  
  /* Get dimensions             */
  const int m       = W.n_rows;
  const int n_spcov = W.n_cols;
  const int n_cov   = Z.n_cols;
  const int dim_z   = z.n_elem;
  
  /* Definition of vectors and matrices     */
  arma::colvec x(n_spcov, fill::zeros);
  arma::colvec v(n_cov, fill::zeros);
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
  //U = glasso_factor_fast_large_m(ATA, FTF, rho); // returns upper
  U = glasso_factor_cov_fast_large_m(WMW, FTF, rho);
  L = U.t();
  
  /* main loop  */
  for (k=0; k<maxiter; k++) {

    // 4-1. update 'x'
    if (rho != rho_old) {
      //U = glasso_factor_fast_large_m(ATA, FTF, rho); // returns upper
      U = glasso_factor_cov_fast_large_m(WMW, FTF, rho);
      L = U.t();
    }
    q = WMy + rho * spmat2vec_mult(F.t(), z - u);        // temporary value
    x = solve(trimatu(U), solve(trimatl(L), q));         // updated of regression parameters associated to covariates W (sparsified covariates, x)
    v = v_LS - P_ZW * x;                                 // updated of regression parameters associated to covariates Z (covariates not sparsified, v)
    
    // 4-2. update 'z' with adaptation
    // see Boyd et al (2011) page 21
    z_old     = z;
    g_id_init = 0;
    for (int g=0; g<G; g++) {
      // precompute the quantities for updating z
      g_id_end         = g_id_init + Glen(g) - 1;
      arma::vec u_g    = u.subvec(g_id_init, g_id_end);
      arma::sp_mat F_g = F.submat(g_id_init, 0, g_id_end, n_spcov-1);

      // update z
      arma::vec z_g                 = glasso_prox(F_g * x + u_g, lambda / rho);
      z.subvec(g_id_init, g_id_end) = z_g;
      g_id_init                     = g_id_end + 1;
    }

    // 4-3. update 'u'
    u += spmat2vec_mult(F, x) - z;
  
    // 4-3. dianostics, reporting
    //h_objval(k) = glasso_objfun_fast(ATA_CHOL_U, ATb, b2, Glen, lambda, x, z, m, n, G);
    h_objval(k)  = linreg_cov_logl_fast(WW_CHOL_U, ZZ_CHOL_U, ZW, Wy, Zy, y2, x, v, m);
    h_objval(k) += glasso_cov_penfun(Glen, G, z, lambda);
    h_r_norm(k)  = norm(glasso_residual_sparse(F, x, z), 2);
    h_s_norm(k)  = norm(glasso_dual_residual_sparse(F, z, z_old, rho), 2);
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
  fitted    = W * x + Z * v;
  residuals = y - fitted;
  mse       = as_scalar(residuals.t() * residuals) / (static_cast<float>(m));
  
  /* Get output         */
  List output;
  output["x"]      = x;               // coefficient function
  output["v"]      = v;               // coefficient function
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
    Group-LASSO via ADMM with covariate not penalized, fast routine to compute the whole path
          for large m where m denotes the number of observations
    exogenous covariates (not penalized) are included in this routine
*/
Rcpp::List admm_glasso_cov_large_m_fast(arma::mat& W, arma::mat& Z, arma::vec& y,
                                        arma::mat& groups, arma::vec& group_weights, arma::vec& var_weights,
                                        const arma::vec lambda, bool rho_adaptation, double rho, const double tau, const double mu,
                                        const double reltol, const double abstol, const int maxiter, const int ping) {

  /* Variable delcaration           */
  double y2, min_mse, sqrtm;
  int n1, dim_z;
  uword idx_min_mse;

  /* Get dimensions             */
  const int m       = W.n_rows;
  const int n_spcov = W.n_cols;
  const int n_cov   = Z.n_cols;
  const int G       = groups.n_rows;
    
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
  arma::mat FTF(n_spcov, n_spcov, fill::zeros);
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
  arma::mat X(nlambda, n_spcov, fill::zeros);
  arma::mat V(nlambda, n_cov, fill::zeros);
  arma::vec conv(nlambda, fill::zeros);
  arma::mat residuals(nlambda, m, fill::zeros);
  arma::mat fitted(nlambda, m, fill::zeros);
  arma::vec mse(nlambda, fill::zeros);
    
  /* Precompute relevant quantities         */
  Z              = Z.each_col() % (ones_m * (1.0 / sqrtm));
  W              = W.each_col() % (ones_m * (1.0 / sqrtm));
  ZZ             = Z.t() * Z;
  L_ZZ           = (chol(ZZ)).t();
  L_ZZ_INV       = inv(trimatl(L_ZZ));
  ZZ_INV         = L_ZZ_INV.t() * L_ZZ_INV;
  P_ZZ           = Z * ZZ_INV * Z.t();
  M_ZZ           = eye_m - P_ZZ;
  WMW            = W.t() * M_ZZ * W;
  WMy            = W.t() * M_ZZ * y / static_cast<float>(sqrtm);
  Zy             = Z.t() * y / static_cast<float>(sqrtm);
  v_LS           = L_ZZ_INV.t() * L_ZZ_INV * Zy;
  ZW             = Z.t() * W;
  P_ZW           = L_ZZ_INV.t() * L_ZZ_INV * ZW;
  WW_CHOL_U      = chol_qr_fact(W, m, n_spcov);
  ZZ_CHOL_U      = chol_qr_fact(Z, m, n_cov);
  y2             = as_scalar(y.t() * y) / static_cast<float>(m);
  Wy             = W.t() * y / static_cast<float>(sqrtm);
  Glen           = sum(groups, 1);
  arma::sp_mat F = glasso_Gmat2Fmat_sparse(groups, group_weights, var_weights);
  FTF            = diagmat(spmat2spmat_mult(F.t(), F));
    
  /* Initialize vectors u and z         */
  dim_z = sum(Glen);
  arma::colvec u(dim_z, fill::zeros);
  arma::colvec z(dim_z, fill::zeros);
  arma::mat Umat(nlambda, dim_z, fill::zeros);
  arma::mat Zmat(nlambda, dim_z, fill::zeros);
  
  /* main loop  */
  for (int j=0; j<nlambda; j++) {
    /* perform linear regression with lasso penalty for the j-th lambda       */
    out = admm_glasso_cov_large_m_update_1lambda(W, Z, y, Glen, G, F, FTF,
                                                 WMW, WMy, v_LS, P_ZW, WW_CHOL_U, Wy, y2, ZZ_CHOL_U, Zy, ZW,
                                                 u, z, lambda(j), rho_adaptation, rho, tau, mu, reltol, abstol, maxiter, ping);

    /* Retrieve output      */
    u = as<arma::colvec>(out["u"]);
    z = as<arma::colvec>(out["z"]);
    x = as<arma::colvec>(out["x"]);
    v = as<arma::colvec>(out["v"]);

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
  output["sp.coef.path"]    = X;
  output["coef.path"]       = V;
  output["iternum"]         = niter;               // number of iterations
  output["objfun"]          = h_objval;        // |x|_1
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

/*
    Group-LASSO via ADMM with covariate not penalized, just one lambda, large n.
*/
Rcpp::List admm_glasso_cov_large_n_update_1lambda(arma::mat& W, arma::mat& Z, arma::vec& y, const arma::colvec Glen, const int G,
                                                  const arma::sp_mat F, const arma::mat FF_INV,
                                                  const arma::mat WW, const arma::mat M_ZZW, const arma::mat M_ZZW2,
                                                  const arma::mat WMW, const arma::vec WMy, const arma::vec v_LS, const arma::mat P_ZW,
                                                  const arma::mat& WW_CHOL_U, const arma::colvec& Wy, const double y2,
                                                  const arma::mat& ZZ_CHOL_U, const arma::colvec& Zy, const arma::mat ZW,
                                                  arma::colvec u, arma::colvec z, const double lambda,
                                                  bool rho_adaptation, double rho, const double tau, const double mu,
                                                  const double reltol, const double abstol, const int maxiter, const int ping) {
  
  /* Variable delcaration           */
  int k;
  double rho_old, elTime = 0.0, mse;
  uword g_id_init, g_id_end;

  /* Get dimensions             */
  const int m       = W.n_rows;
  const int n_spcov = W.n_cols;
  const int n_cov   = Z.n_cols;
  const int dim_z   = z.n_elem;
    
  /* Definition of vectors and matrices     */
  arma::colvec x(n_spcov, fill::zeros);
  arma::colvec x_star(n_spcov, fill::zeros);
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
  U = glasso_factor_cov_fast_large_n(M_ZZW2, rho, m); // returns upper
  L = U.t();
  
  /* main loop  */
  for (k=0; k<maxiter; k++) {

    // 4-1. update 'x'
    if (rho != rho_old) {
      U = glasso_factor_cov_fast_large_n(M_ZZW2, rho, m); // returns upper
      L = U.t();
    }
    q      = WMy + rho * spmat2vec_mult(F.t(), z - u); // temporary value
    x_star = (diagmat(FF_INV) * q) / rho;
    x      = x_star - diagmat(FF_INV) * M_ZZW.t() * solve(trimatu(U), solve(trimatl(L), M_ZZW * x_star)); // updated of regression parameters associated to covariates W
    v      = v_LS - P_ZW * x;                                 // updated of regression parameters associated to covariates Z (covariates not sparsified, v)

    // 4-2. update 'z' with adaptation
    // see Boyd et al (2011) page 21
    z_old     = z;
    g_id_init = 0;
    for (int g=0; g<G; g++) {
      // precompute the quantities for updating z
      g_id_end         = g_id_init + Glen(g) - 1;
      arma::vec u_g    = u.subvec(g_id_init, g_id_end);
      arma::sp_mat F_g = F.submat(g_id_init, 0, g_id_end, n_spcov-1);

      // update z
      arma::vec z_g                 = glasso_prox(F_g * x + u_g, lambda / rho);
      z.subvec(g_id_init, g_id_end) = z_g;
      g_id_init                     = g_id_end + 1;
    }

    // 4-3. update 'u'
    u += spmat2vec_mult(F, x) - z;
  
    // 4-3. dianostics, reporting
    //h_objval(k) = glasso_objfun_fast(ATA_CHOL_U, ATb, b2, Glen, lambda, x, z, m, n, G);
    h_objval(k)  = linreg_cov_logl_fast(WW_CHOL_U, ZZ_CHOL_U, ZW, Wy, Zy, y2, x, v, m);
    h_objval(k) += glasso_cov_penfun(Glen, G, z, lambda);
    h_r_norm(k)  = norm(glasso_residual_sparse(F, x, z), 2);
    h_s_norm(k)  = norm(glasso_dual_residual_sparse(F, z, z_old, rho), 2);
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
  fitted    = W * x + Z * v;
  residuals = y - fitted;
  mse       = as_scalar(residuals.t() * residuals) / (static_cast<float>(m));
  
  /* Get output         */
  List output;
  output["x"]      = x;               // coefficient function
  output["v"]      = v;               // coefficient function
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
    Group-LASSO via ADMM with covariate not penalized, fast routine to compute the whole path
          for large n where n denotes the number of parameters
    exogenous covariates (not penalized) are included in this routine
*/
Rcpp::List admm_glasso_cov_large_n_fast(arma::mat& W, arma::mat Z, arma::vec& y,
                                        arma::mat& groups, arma::vec& group_weights, arma::vec& var_weights,
                                        const arma::vec lambda, bool rho_adaptation, double rho, const double tau, const double mu,
                                        const double reltol, const double abstol, const int maxiter, const int ping) {

  /* Variable delcaration           */
  double y2, min_mse, sqrtm;
  int n1, dim_z;
  uword idx_min_mse;

  /* Get dimensions             */
  const int m       = W.n_rows;
  const int n_spcov = W.n_cols;
  const int n_cov   = Z.n_cols;
  const int G = groups.n_rows;
  
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
  arma::mat FTF(n_spcov, n_spcov, fill::zeros);
  arma::mat FTF_INV(n_spcov, n_spcov, fill::zeros);
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
  arma::mat X(nlambda, n_spcov, fill::zeros);
  arma::mat V(nlambda, n_cov, fill::zeros);
  arma::vec conv(nlambda, fill::zeros);
  arma::mat residuals(nlambda, m, fill::zeros);
  arma::mat fitted(nlambda, m, fill::zeros);
  arma::vec mse(nlambda, fill::zeros);
  
  /* Precompute relevant quantities         */
  Z         = Z.each_col() % (ones_m * (1.0 / sqrtm));
  //W         = W.each_col() % (ones_m * (1.0 / sqrtm));
  ZZ        = Z.t() * Z;
  L_ZZ      = (chol(ZZ)).t();
  L_ZZ_INV  = inv(trimatl(L_ZZ));
  ZZ_INV    = L_ZZ_INV.t() * L_ZZ_INV;
  P_ZZ      = Z * ZZ_INV * Z.t();
  M_ZZ      = eye_m - P_ZZ;
  WMW       = W.t() * M_ZZ * W;
  WMy       = W.t() * M_ZZ * y / static_cast<float>(m);
  Zy        = Z.t() * y / static_cast<float>(sqrtm);
  v_LS      = L_ZZ_INV.t() * L_ZZ_INV * Zy;
  ZW        = Z.t() * W / static_cast<float>(sqrtm);
  P_ZW      = L_ZZ_INV.t() * L_ZZ_INV * ZW;
  WW_CHOL_U = chol_qr_fact(W, m, n_spcov);
  ZZ_CHOL_U = chol_qr_fact(Z, m, n_cov);
  y2        = as_scalar(y.t() * y) / static_cast<float>(m);
  Wy        = W.t() * y / static_cast<float>(m);
  M_ZZW     = M_ZZ * W;
  //M_ZZW2    = M_ZZW * M_ZZW.t();
  
  Glen           = sum(groups, 1);
  arma::sp_mat F = glasso_Gmat2Fmat_sparse(groups, group_weights, var_weights);
  FTF            = diagmat(spmat2spmat_mult(F.t(), F));
  FTF_INV        = diagmat(1.0 / diagvec(FTF));
  M_ZZW2         = M_ZZW * diagmat(FTF_INV) * M_ZZW.t();
  
  /* Initialize vectors u and z         */
  dim_z = sum(Glen);
  arma::colvec u(dim_z, fill::zeros);
  arma::colvec z(dim_z, fill::zeros);
  arma::mat Umat(nlambda, dim_z, fill::zeros);
  arma::mat Zmat(nlambda, dim_z, fill::zeros);
  
  /* main loop  */
  for (int j=0; j<nlambda; j++) {
    /* perform linear regression with lasso penalty for the j-th lambda       */
    out = admm_glasso_cov_large_n_update_1lambda(W, Z, y, Glen, G, F, FTF_INV, WW, M_ZZW, M_ZZW2,
                                                 WMW, WMy, v_LS, P_ZW, WW_CHOL_U, Wy, y2,
                                                 ZZ_CHOL_U, Zy, ZW, u, z, lambda(j),
                                                 rho_adaptation, rho, tau, mu, reltol, abstol, maxiter, ping);

    /* Retrieve output      */
    u = as<arma::colvec>(out["u"]);
    z = as<arma::colvec>(out["z"]);
    x = as<arma::colvec>(out["x"]);
    v = as<arma::colvec>(out["v"]);

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
  output["sp.coef.path"]    = X;
  output["coef.path"]       = V;
  output["iternum"]         = niter;               // number of iterations
  output["objfun"]          = h_objval;            // |x|_1
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

/*
    Group-LASSO via ADMM with covariate not penalized, fast routine to compute the whole path
            wrap for m and n (m denotes the number of observations, while n denotes the number of parameters)
*/
//' @name  fdaSP.internal
//' @keywords internal
// [[Rcpp::export(.admm_glasso_cov_fast)]]
Rcpp::List admm_glasso_cov_fast(arma::mat& W, arma::mat Z, arma::vec& y,
                                arma::mat& groups, arma::vec& group_weights, arma::vec& var_weights,
                                const arma::vec lambda, bool rho_adaptation, double rho, const double tau, const double mu,
                                const double reltol, const double abstol, const int maxiter, const int ping) {

  /* Get dimensions             */
  const int m       = W.n_rows;
  const int n_spcov = W.n_cols;

  /* Variable delcaration           */
  List out;

  /* Run ADMM            */
  if (m >= n_spcov) {
    /* large n problems            */
    out = admm_glasso_cov_large_m_fast(W, Z, y,
                                       groups, group_weights, var_weights,
                                       lambda, rho_adaptation, rho, tau, mu,
                                       reltol, abstol, maxiter, ping);
  } else {
    /* large p problems            */
    out = admm_glasso_cov_large_n_fast(W, Z, y,
                                       groups, group_weights, var_weights,
                                       lambda, rho_adaptation, rho, tau, mu,
                                       reltol, abstol, maxiter, ping);
  }

  /* Return output      */
  return(out);
}

/*
    Overlap Group-LASSO via ADMM with covariate not penalized, just one lambda, large m.
*/
Rcpp::List admm_ovglasso_cov_large_m_update_1lambda(arma::mat& W, arma::mat& Z, arma::vec& y,
                                                    const arma::colvec Glen, const int G, const arma::sp_mat F, const arma::mat FTF,
                                                    const arma::mat WMW, const arma::vec WMy, const arma::vec v_LS, const arma::mat P_ZW,
                                                    const arma::mat& WW_CHOL_U, const arma::colvec& Wy, const double y2,
                                                    const arma::mat& ZZ_CHOL_U, const arma::colvec& Zy, const arma::mat ZW,
                                                    arma::colvec u, arma::colvec z, const double lambda,
                                                    bool rho_adaptation, double rho, const double tau, const double mu,
                                                    const double reltol, const double abstol, const int maxiter, const int ping) {
  
  /* Variable delcaration           */
  int k;
  double rho_old, elTime = 0.0, mse;
  uword g_id_init, g_id_end;
  
  /* Get dimensions             */
  const int m       = W.n_rows;
  const int n_spcov = W.n_cols;
  const int n_cov   = Z.n_cols;
  const int dim_z   = z.n_elem;
    
  /* Definition of vectors and matrices     */
  arma::colvec x(n_spcov, fill::zeros);
  arma::colvec v(n_cov, fill::zeros);
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
  //U = glasso_factor_fast_large_m(ATA, FTF, rho); // returns upper
  U = glasso_factor_cov_fast_large_m(WMW, FTF, rho);
  L = U.t();
  
  /* main loop  */
  for (k=0; k<maxiter; k++) {

    // 4-1. update 'x'
    if (rho != rho_old) {
      //U = glasso_factor_fast_large_m(ATA, FTF, rho); // returns upper
      U = glasso_factor_cov_fast_large_m(WMW, FTF, rho);
      L = U.t();
    }
    q = WMy + rho * spmat2vec_mult(F.t(), z - u);        // temporary value
    x = solve(trimatu(U), solve(trimatl(L), q));         // updated of regression parameters associated to covariates W (sparsified covariates, x)
    v = v_LS - P_ZW * x;                                 // updated of regression parameters associated to covariates Z (covariates not sparsified, v)
    
    // 4-2. update 'z' with adaptation
    // see Boyd et al (2011) page 21
    z_old     = z;
    g_id_init = 0;
    for (int g=0; g<G; g++) {
      // precompute the quantities for updating z
      g_id_end         = g_id_init + Glen(g) - 1;
      arma::vec u_g    = u.subvec(g_id_init, g_id_end);
      arma::sp_mat F_g = F.submat(g_id_init, 0, g_id_end, n_spcov-1);

      // update z
      arma::vec z_g                 = glasso_prox(F_g * x + u_g, lambda / rho);
      z.subvec(g_id_init, g_id_end) = z_g;
      g_id_init                     = g_id_end + 1;
    }

    // 4-3. update 'u'
    u += spmat2vec_mult(F, x) - z;
  
    // 4-3. dianostics, reporting
    //h_objval(k) = glasso_objfun_fast(ATA_CHOL_U, ATb, b2, Glen, lambda, x, z, m, n, G);
    h_objval(k)  = linreg_cov_logl_fast(WW_CHOL_U, ZZ_CHOL_U, ZW, Wy, Zy, y2, x, v, m);
    h_objval(k) += glasso_cov_penfun(Glen, G, z, lambda);
    h_r_norm(k)  = norm(glasso_residual_sparse(F, x, z), 2);
    h_s_norm(k)  = norm(glasso_dual_residual_sparse(F, z, z_old, rho), 2);
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
  fitted    = W * x + Z * v;
  residuals = y - fitted;
  mse       = as_scalar(residuals.t() * residuals) / (static_cast<float>(m));
  
  /* Get output         */
  List output;
  output["x"]      = x;               // coefficient function
  output["v"]      = v;               // coefficient function
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
    Overlap Group-LASSO via ADMM with covariate not penalized, fast routine to compute the whole path
          for large m where m denotes the number of observations
    exogenous covariates (not penalized) are included in this routine
*/
Rcpp::List admm_ovglasso_cov_large_m_fast(arma::mat& W, arma::mat& Z, arma::vec& y,
                                          arma::mat& groups, arma::vec& group_weights, arma::vec& var_weights,
                                          const arma::vec lambda, bool rho_adaptation, double rho, const double tau, const double mu,
                                          const double reltol, const double abstol, const int maxiter, const int ping) {

  /* Variable delcaration           */
  double y2, min_mse, sqrtm;
  int n1, dim_z;
  uword idx_min_mse;

  /* Get dimensions             */
  const int m       = W.n_rows;
  const int n_spcov = W.n_cols;
  const int n_cov   = Z.n_cols;
  const int G       = groups.n_rows;
  
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
  arma::mat FTF(n_spcov, n_spcov, fill::zeros);
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
  arma::mat X(nlambda, n_spcov, fill::zeros);
  arma::mat V(nlambda, n_cov, fill::zeros);
  arma::vec conv(nlambda, fill::zeros);
  arma::mat residuals(nlambda, m, fill::zeros);
  arma::mat fitted(nlambda, m, fill::zeros);
  arma::vec mse(nlambda, fill::zeros);
  
  /* Precompute relevant quantities         */
  Z              = Z.each_col() % (ones_m * (1.0 / sqrtm));
  W              = W.each_col() % (ones_m * (1.0 / sqrtm));
  ZZ             = Z.t() * Z;
  L_ZZ           = (chol(ZZ)).t();
  L_ZZ_INV       = inv(trimatl(L_ZZ));
  ZZ_INV         = L_ZZ_INV.t() * L_ZZ_INV;
  P_ZZ           = Z * ZZ_INV * Z.t();
  M_ZZ           = eye_m - P_ZZ;
  WMW            = W.t() * M_ZZ * W;
  WMy            = W.t() * M_ZZ * y / static_cast<float>(sqrtm);
  Zy             = Z.t() * y / static_cast<float>(sqrtm);
  v_LS           = L_ZZ_INV.t() * L_ZZ_INV * Zy;
  ZW             = Z.t() * W;
  P_ZW           = L_ZZ_INV.t() * L_ZZ_INV * ZW;
  WW_CHOL_U      = chol_qr_fact(W, m, n_spcov);
  ZZ_CHOL_U      = chol_qr_fact(Z, m, n_cov);
  y2             = as_scalar(y.t() * y) / static_cast<float>(m);
  Wy             = W.t() * y / static_cast<float>(sqrtm);
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
    out = admm_ovglasso_cov_large_m_update_1lambda(W, Z, y, Glen, G, F, FTF,
                                                   WMW, WMy, v_LS, P_ZW, WW_CHOL_U, Wy, y2, ZZ_CHOL_U, Zy, ZW,
                                                   u, z, lambda(j), rho_adaptation, rho, tau, mu, reltol, abstol, maxiter, ping);

    /* Retrieve output      */
    u = as<arma::colvec>(out["u"]);
    z = as<arma::colvec>(out["z"]);
    x = as<arma::colvec>(out["x"]);
    v = as<arma::colvec>(out["v"]);

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
  output["sp.coef.path"]    = X;
  output["coef.path"]       = V;
  output["iternum"]         = niter;               // number of iterations
  output["objfun"]          = h_objval;        // |x|_1
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

/*
    Overlap Group-LASSO via ADMM with covariate not penalized, just one lambda, large n.
*/
Rcpp::List admm_ovglasso_cov_large_n_update_1lambda(arma::mat& W, arma::mat& Z, arma::vec& y, const arma::colvec Glen, const int G,
                                                    const arma::sp_mat F, const arma::mat FF_INV,
                                                    const arma::mat WW, const arma::mat M_ZZW, const arma::mat M_ZZW2,
                                                    const arma::mat WMW, const arma::vec WMy, const arma::vec v_LS, const arma::mat P_ZW,
                                                    const arma::mat& WW_CHOL_U, const arma::colvec& Wy, const double y2,
                                                    const arma::mat& ZZ_CHOL_U, const arma::colvec& Zy, const arma::mat ZW,
                                                    arma::colvec u, arma::colvec z, const double lambda,
                                                    bool rho_adaptation, double rho, const double tau, const double mu,
                                                    const double reltol, const double abstol, const int maxiter, const int ping) {
  

  /* Variable delcaration           */
  int k;
  double rho_old, elTime = 0.0, mse;
  uword g_id_init, g_id_end;

  /* Get dimensions             */
  const int m       = W.n_rows;
  const int n_spcov = W.n_cols;
  const int n_cov   = Z.n_cols;
  const int dim_z   = z.n_elem;
    
  /* Definition of vectors and matrices     */
  arma::colvec x(n_spcov, fill::zeros);
  arma::colvec x_star(n_spcov, fill::zeros);
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
  U = glasso_factor_cov_fast_large_n(M_ZZW2, rho, m); // returns upper
  L = U.t();
  
  /* main loop  */
  for (k=0; k<maxiter; k++) {

    // 4-1. update 'x'
    if (rho != rho_old) {
      U = glasso_factor_cov_fast_large_n(M_ZZW2, rho, m); // returns upper
      L = U.t();
    }
    q      = WMy + rho * spmat2vec_mult(F.t(), z - u); // temporary value
    x_star = (diagmat(FF_INV) * q) / rho;
    x      = x_star - diagmat(FF_INV) * M_ZZW.t() * solve(trimatu(U), solve(trimatl(L), M_ZZW * x_star)); // updated of regression parameters associated to covariates W
    v      = v_LS - P_ZW * x;                                 // updated of regression parameters associated to covariates Z (covariates not sparsified, v)

    // 4-2. update 'z' with adaptation
    // see Boyd et al (2011) page 21
    z_old     = z;
    g_id_init = 0;
    for (int g=0; g<G; g++) {
      // precompute the quantities for updating z
      g_id_end         = g_id_init + Glen(g) - 1;
      arma::vec u_g    = u.subvec(g_id_init, g_id_end);
      arma::sp_mat F_g = F.submat(g_id_init, 0, g_id_end, n_spcov-1);

      // update z
      arma::vec z_g                 = glasso_prox(F_g * x + u_g, lambda / rho);
      z.subvec(g_id_init, g_id_end) = z_g;
      g_id_init                     = g_id_end + 1;
    }

    // 4-3. update 'u'
    u += spmat2vec_mult(F, x) - z;
  
    // 4-3. dianostics, reporting
    //h_objval(k) = glasso_objfun_fast(ATA_CHOL_U, ATb, b2, Glen, lambda, x, z, m, n, G);
    h_objval(k)  = linreg_cov_logl_fast(WW_CHOL_U, ZZ_CHOL_U, ZW, Wy, Zy, y2, x, v, m);
    h_objval(k) += glasso_cov_penfun(Glen, G, z, lambda);
    h_r_norm(k)  = norm(glasso_residual_sparse(F, x, z), 2);
    h_s_norm(k)  = norm(glasso_dual_residual_sparse(F, z, z_old, rho), 2);
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
  fitted    = W * x + Z * v;
  residuals = y - fitted;
  mse       = as_scalar(residuals.t() * residuals) / (static_cast<float>(m));
  
  /* Get output         */
  List output;
  output["x"]      = x;               // coefficient function
  output["v"]      = v;               // coefficient function
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
    Overlap Group-LASSO via ADMM with covariate not penalized, fast routine to compute the whole path
          for large n where n denotes the number of parameters
    exogenous covariates (not penalized) are included in this routine
*/
Rcpp::List admm_ovglasso_cov_large_n_fast(arma::mat& W, arma::mat& Z, arma::vec& y,
                                          arma::mat& groups, arma::vec& group_weights, arma::vec& var_weights,
                                          const arma::vec lambda, bool rho_adaptation, double rho, const double tau, const double mu,
                                          const double reltol, const double abstol, const int maxiter, const int ping) {

  /* Variable delcaration           */
  double y2, min_mse, sqrtm;
  int n1, dim_z;
  uword idx_min_mse;

  /* Get dimensions             */
  const int m       = W.n_rows;
  const int n_spcov = W.n_cols;
  const int n_cov   = Z.n_cols;
  const int G = groups.n_rows;
  
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
  arma::mat FTF(n_spcov, n_spcov, fill::zeros);
  arma::mat FTF_INV(n_spcov, n_spcov, fill::zeros);
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
  arma::mat X(nlambda, n_spcov, fill::zeros);
  arma::mat V(nlambda, n_cov, fill::zeros);
  arma::vec conv(nlambda, fill::zeros);
  arma::mat residuals(nlambda, m, fill::zeros);
  arma::mat fitted(nlambda, m, fill::zeros);
  arma::vec mse(nlambda, fill::zeros);
  
  /* Precompute relevant quantities         */
  Z         = Z.each_col() % (ones_m * (1.0 / sqrtm));
  //W         = W.each_col() % (ones_m * (1.0 / sqrtm));
  ZZ        = Z.t() * Z;
  L_ZZ      = (chol(ZZ)).t();
  L_ZZ_INV  = inv(trimatl(L_ZZ));
  ZZ_INV    = L_ZZ_INV.t() * L_ZZ_INV;
  P_ZZ      = Z * ZZ_INV * Z.t();
  M_ZZ      = eye_m - P_ZZ;
  WMW       = W.t() * M_ZZ * W;
  WMy       = W.t() * M_ZZ * y / static_cast<float>(m);
  Zy        = Z.t() * y / static_cast<float>(sqrtm);
  v_LS      = L_ZZ_INV.t() * L_ZZ_INV * Zy;
  ZW        = Z.t() * W / static_cast<float>(sqrtm);
  P_ZW      = L_ZZ_INV.t() * L_ZZ_INV * ZW;
  WW_CHOL_U = chol_qr_fact(W, m, n_spcov);
  ZZ_CHOL_U = chol_qr_fact(Z, m, n_cov);
  y2        = as_scalar(y.t() * y) / static_cast<float>(m);
  Wy        = W.t() * y / static_cast<float>(m);
  M_ZZW     = M_ZZ * W;
  //M_ZZW2    = M_ZZW * M_ZZW.t();
  
  Glen           = sum(groups, 1);
  arma::sp_mat F = glasso_Gmat2Fmat_sparse(groups, group_weights, var_weights);
  FTF            = diagmat(spmat2spmat_mult(F.t(), F));
  FTF_INV        = diagmat(1.0 / diagvec(FTF));
  M_ZZW2         = M_ZZW * diagmat(FTF_INV) * M_ZZW.t();
  
  /* Initialize vectors u and z         */
  dim_z = sum(Glen);
  arma::colvec u(dim_z, fill::zeros);
  arma::colvec z(dim_z, fill::zeros);
  arma::mat Umat(nlambda, dim_z, fill::zeros);
  arma::mat Zmat(nlambda, dim_z, fill::zeros);
  
  /* main loop  */
  for (int j=0; j<nlambda; j++) {
    /* perform linear regression with lasso penalty for the j-th lambda       */
    out = admm_ovglasso_cov_large_n_update_1lambda(W, Z, y, Glen, G, F, FTF_INV, WW, M_ZZW, M_ZZW2,
                                                   WMW, WMy, v_LS, P_ZW, WW_CHOL_U, Wy, y2,
                                                   ZZ_CHOL_U, Zy, ZW, u, z, lambda(j),
                                                   rho_adaptation, rho, tau, mu, reltol, abstol, maxiter, ping);

    /* Retrieve output: warm start      */
    u = as<arma::colvec>(out["u"]);
    z = as<arma::colvec>(out["z"]);
    x = as<arma::colvec>(out["x"]);
    v = as<arma::colvec>(out["v"]);

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
  output["sp.coef.path"]    = X;
  output["coef.path"]       = V;
  output["iternum"]         = niter;               // number of iterations
  output["objfun"]          = h_objval;            // |x|_1
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

/*
    Overlap Group-LASSO via ADMM with covariate not penalized, fast routine to compute the whole path
            wrap for m and n (m denotes the number of observations, while n denotes the number of parameters)
*/
//' @name  fdaSP.internal
//' @keywords internal
// [[Rcpp::export(.admm_ovglasso_cov_fast)]]
Rcpp::List admm_ovglasso_cov_fast(arma::mat& W, arma::mat& Z, arma::vec& y,
                                  arma::mat& groups, arma::vec& group_weights, arma::vec& var_weights,
                                  const arma::vec lambda, bool rho_adaptation, double rho, const double tau, const double mu,
                                  const double reltol, const double abstol, const int maxiter, const int ping) {

  /* Get dimensions             */
  const int m       = W.n_rows;
  const int n_spcov = W.n_cols;

  /* Variable delcaration           */
  List out;

  /* Run ADMM            */
  if (m >= n_spcov) {
    /* large n problems            */
    out = admm_ovglasso_cov_large_m_fast(W, Z, y,
                                         groups, group_weights, var_weights,
                                         lambda, rho_adaptation, rho, tau, mu,
                                         reltol, abstol, maxiter, ping);
  } else {
    /* large p problems            */
    out = admm_ovglasso_cov_large_n_fast(W, Z, y,
                                         groups, group_weights, var_weights,
                                         lambda, rho_adaptation, rho, tau, mu,
                                         reltol, abstol, maxiter, ping);
  }

  /* Return output      */
  return(out);
}

/*
 Adaptive-LASSO with exogenous covariates via ADMM, just to compute the whole path fast.
*/
Rcpp::List admm_adalasso_cov_large_m_update_1lambda(arma::mat& W, arma::mat& Z, arma::vec& y,
                                                    const arma::mat F, const arma::mat FTF,
                                                    const arma::mat WMW, const arma::vec WMy, const arma::vec v_LS, const arma::mat P_ZW,
                                                    const arma::mat& WW_CHOL_U, const arma::colvec& Wy, const double y2,
                                                    const arma::mat& ZZ_CHOL_U, const arma::colvec& Zy, const arma::mat ZW,
                                                    arma::colvec u, arma::colvec z, const double lambda,
                                                    bool rho_adaptation, double rho, const double tau, const double mu,
                                                    const double reltol, const double abstol, const int maxiter, const int ping) {
  
  /* Variable delcaration           */
  int k;
  double rho_old, elTime = 0.0, mse;
  
  /* Get dimensions             */
  const int m       = W.n_rows;
  const int n_spcov = W.n_cols;
  const int n_cov   = Z.n_cols;
  const int dim_z   = z.n_elem;
    
  /* Definition of vectors and matrices     */
  arma::colvec x(n_spcov, fill::zeros);
  arma::colvec v(n_cov, fill::zeros);
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
  //U = glasso_factor_fast_large_m(ATA, FTF, rho); // returns upper
  U = glasso_factor_cov_fast_large_m(WMW, FTF, rho);
  L = U.t();
  
  /* main loop  */
  for (k=0; k<maxiter; k++) {

    // 4-1. update 'x'
    if (rho != rho_old) {
      //U = glasso_factor_fast_large_m(ATA, FTF, rho); // returns upper
      U = glasso_factor_cov_fast_large_m(WMW, FTF, rho);
      L = U.t();
    }
    q = WMy + rho * diagmat(F) * (z - u);           // temporary value
    x = solve(trimatu(U), solve(trimatl(L), q));         // updated of regression parameters associated to covariates W (sparsified covariates, x)
    v = v_LS - P_ZW * x;                                 // updated of regression parameters associated to covariates Z (covariates not sparsified, v)
    
    // 4-2. update 'z' with adaptation
    // see Boyd et al (2011) page 21
    z_old = z;
    z     = lasso_prox(diagmat(F) * x + u, lambda / rho);

    // 4-3. update 'u'
    u += diagmat(F) * x - z;
  
    // 4-3. dianostics, reporting
    //h_objval(k) = glasso_objfun_fast(ATA_CHOL_U, ATb, b2, Glen, lambda, x, z, m, n, G);
    h_objval(k)  = linreg_cov_logl_fast(WW_CHOL_U, ZZ_CHOL_U, ZW, Wy, Zy, y2, x, v, m);
    h_objval(k) += adalasso_cov_penfun(z, lambda, diagvec(F));
    h_r_norm(k)  = norm(adalasso_residual(F, x, z), 2);
    h_s_norm(k)  = norm(adalasso_dual_residual(F, z, z_old, rho), 2);
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
  fitted    = W * x + Z * v;
  residuals = y - fitted;
  mse       = as_scalar(residuals.t() * residuals) / (static_cast<float>(m));
  
  /* Get output         */
  List output;
  output["x"]      = x;               // coefficient function
  output["v"]      = v;               // coefficient function
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
    Adaptive-LASSO via ADMM with covariate not penalized, fast routine to compute the whole path
          for large m where m denotes the number of observations
    exogenous covariates (not penalized) are included in this routine
*/
Rcpp::List admm_adalasso_cov_large_m_fast(arma::mat& W, arma::mat& Z, arma::vec& y, arma::vec& var_weights,
                                          const arma::vec lambda, bool rho_adaptation, double rho, const double tau, const double mu,
                                          const double reltol, const double abstol, const int maxiter, const int ping) {

  /* Variable delcaration           */
  double y2, min_mse, sqrtm;
  int n1;
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
  arma::mat FTF(n_spcov, n_spcov, fill::zeros);

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
  arma::vec mse(nlambda, fill::zeros);
  
  /* Precompute relevant quantities         */
  Z              = Z.each_col() % (ones_m * (1.0 / sqrtm));
  W              = W.each_col() % (ones_m * (1.0 / sqrtm));
  ZZ             = Z.t() * Z;
  L_ZZ           = (chol(ZZ)).t();
  L_ZZ_INV       = inv(trimatl(L_ZZ));
  ZZ_INV         = L_ZZ_INV.t() * L_ZZ_INV;
  P_ZZ           = Z * ZZ_INV * Z.t();
  M_ZZ           = eye_m - P_ZZ;
  WMW            = W.t() * M_ZZ * W;
  WMy            = W.t() * M_ZZ * y / static_cast<float>(sqrtm);
  Zy             = Z.t() * y / static_cast<float>(sqrtm);
  v_LS           = L_ZZ_INV.t() * L_ZZ_INV * Zy;
  ZW             = Z.t() * W;
  P_ZW           = L_ZZ_INV.t() * L_ZZ_INV * ZW;
  WW_CHOL_U      = chol_qr_fact(W, m, n_spcov);
  ZZ_CHOL_U      = chol_qr_fact(Z, m, n_cov);
  y2             = as_scalar(y.t() * y) / static_cast<float>(m);
  Wy             = W.t() * y / static_cast<float>(sqrtm);
  
  /* Define the F matrix         */
  arma::mat F = adalasso_Fmat(var_weights);
  FTF         = diagmat(F) * diagmat(F);
  
  /* Initialize vectors u and z         */
  arma::colvec u(n_spcov, fill::zeros);
  arma::colvec z(n_spcov, fill::zeros);
  arma::mat Umat(nlambda, n_spcov, fill::zeros);
  arma::mat Zmat(nlambda, n_spcov, fill::zeros);

  /* main loop  */
  for (int j=0; j<nlambda; j++) {
    /* perform linear regression with lasso penalty for the j-th lambda       */
    out = admm_adalasso_cov_large_m_update_1lambda(W, Z, y, F, FTF, WMW, WMy, v_LS, P_ZW,
                                                   WW_CHOL_U, Wy, y2, ZZ_CHOL_U, Zy, ZW,
                                                   u, z, lambda(j), rho_adaptation, rho, tau, mu,
                                                   reltol, abstol, maxiter, ping);

    /* Retrieve output      */
    u = as<arma::colvec>(out["u"]);
    z = as<arma::colvec>(out["z"]);
    x = as<arma::colvec>(out["x"]);
    v = as<arma::colvec>(out["v"]);

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
  output["sp.coef.path"]    = X;
  output["coef.path"]       = V;
  output["iternum"]         = niter;               // number of iterations
  output["objfun"]          = h_objval;        // |x|_1
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

/*
    Adaptive-LASSO via ADMM with covariate not penalized, just one lambda, large n.
*/
Rcpp::List admm_adalasso_cov_large_n_update_1lambda(arma::mat& W, arma::mat& Z, arma::vec& y,
                                                    const arma::mat F, const arma::mat FF_INV,
                                                    const arma::mat WW, const arma::mat M_ZZW, const arma::mat M_ZZW2,
                                                    const arma::mat WMW, const arma::vec WMy, const arma::vec v_LS, const arma::mat P_ZW,
                                                    const arma::mat& WW_CHOL_U, const arma::colvec& Wy, const double y2,
                                                    const arma::mat& ZZ_CHOL_U, const arma::colvec& Zy, const arma::mat ZW,
                                                    arma::colvec u, arma::colvec z, const double lambda,
                                                    bool rho_adaptation, double rho, const double tau, const double mu,
                                                    const double reltol, const double abstol, const int maxiter, const int ping) {
  

  /* Variable delcaration           */
  int k;
  double rho_old, elTime = 0.0, mse;

  /* Get dimensions             */
  const int m       = W.n_rows;
  const int n_spcov = W.n_cols;
  const int n_cov   = Z.n_cols;
  const int dim_z   = z.n_elem;
    
  /* Definition of vectors and matrices     */
  arma::colvec x(n_spcov, fill::zeros);
  arma::colvec x_star(n_spcov, fill::zeros);
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
  U = glasso_factor_cov_fast_large_n(M_ZZW2, rho, m); // returns upper
  L = U.t();
  
  /* main loop  */
  for (k=0; k<maxiter; k++) {

    // 4-1. update 'x'
    if (rho != rho_old) {
      U = glasso_factor_cov_fast_large_n(M_ZZW2, rho, m); // returns upper
      L = U.t();
    }
    q      = WMy + rho * diagmat(F) * (z - u); // temporary value
    x_star = (diagmat(FF_INV) * q) / rho;
    x      = x_star - diagmat(FF_INV) * M_ZZW.t() * solve(trimatu(U), solve(trimatl(L), M_ZZW * x_star)); // updated of regression parameters associated to covariates W
    v      = v_LS - P_ZW * x;                                 // updated of regression parameters associated to covariates Z (covariates not sparsified, v)

    // 4-2. update 'z' with adaptation
    // see Boyd et al (2011) page 21
    z_old = z;
    z     = lasso_prox(diagmat(F) * x + u, lambda / rho);

    // 4-3. update 'u'
    u = u + diagmat(F) * x - z;
  
    // 4-3. dianostics, reporting
    //h_objval(k) = glasso_objfun_fast(ATA_CHOL_U, ATb, b2, Glen, lambda, x, z, m, n, G);
    h_objval(k)  = linreg_cov_logl_fast(WW_CHOL_U, ZZ_CHOL_U, ZW, Wy, Zy, y2, x, v, m);
    h_objval(k) += adalasso_cov_penfun(z, lambda, diagvec(F));
    h_r_norm(k)  = norm(adalasso_residual(F, x, z), 2);
    h_s_norm(k)  = norm(adalasso_dual_residual(F, z, z_old, rho), 2);
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
  fitted    = W * x + Z * v;
  residuals = y - fitted;
  mse       = as_scalar(residuals.t() * residuals) / (static_cast<float>(m));
  
  /* Get output         */
  List output;
  output["x"]      = x;               // coefficient function
  output["v"]      = v;               // coefficient function
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
    Adaptive-LASSO via ADMM with covariate not penalized, fast routine to compute the whole path
          for large n where n denotes the number of parameters
    exogenous covariates (not penalized) are included in this routine
*/
Rcpp::List admm_adalasso_cov_large_n_fast(arma::mat& W, arma::mat& Z, arma::vec& y, arma::vec& var_weights,
                                          const arma::vec lambda, bool rho_adaptation, double rho, const double tau, const double mu,
                                          const double reltol, const double abstol, const int maxiter, const int ping) {

  /* Variable delcaration           */
  double y2, min_mse, sqrtm;
  int n1;
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
  arma::mat FTF(n_spcov, n_spcov, fill::zeros);
  arma::mat FTF_INV(n_spcov, n_spcov, fill::zeros);
  
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
  arma::vec mse(nlambda, fill::zeros);
  
  /* Precompute relevant quantities         */
  Z         = Z.each_col() % (ones_m * (1.0 / sqrtm));
  ZZ        = Z.t() * Z;
  L_ZZ      = (chol(ZZ)).t();
  L_ZZ_INV  = inv(trimatl(L_ZZ));
  ZZ_INV    = L_ZZ_INV.t() * L_ZZ_INV;
  P_ZZ      = Z * ZZ_INV * Z.t();
  M_ZZ      = eye_m - P_ZZ;
  WMW       = W.t() * M_ZZ * W;
  WMy       = W.t() * M_ZZ * y / static_cast<float>(m);
  Zy        = Z.t() * y / static_cast<float>(sqrtm);
  v_LS      = L_ZZ_INV.t() * L_ZZ_INV * Zy;
  ZW        = Z.t() * W / static_cast<float>(sqrtm);
  P_ZW      = L_ZZ_INV.t() * L_ZZ_INV * ZW;
  WW_CHOL_U = chol_qr_fact(W, m, n_spcov);
  ZZ_CHOL_U = chol_qr_fact(Z, m, n_cov);
  y2        = as_scalar(y.t() * y) / static_cast<float>(m);
  Wy        = W.t() * y / static_cast<float>(m);
  M_ZZW     = M_ZZ * W;
  
  /* Define the F matrix         */
  arma::mat F = adalasso_Fmat(var_weights);
  FTF         = diagmat(F) * diagmat(F);
  FTF_INV     = diagmat(1.0 / diagvec(FTF));
  M_ZZW2      = M_ZZW * diagmat(FTF_INV) * M_ZZW.t();
  
  /* Initialize vectors u and z         */
  arma::colvec u(n_spcov, fill::zeros);
  arma::colvec z(n_spcov, fill::zeros);
  arma::mat Umat(nlambda, n_spcov, fill::zeros);
  arma::mat Zmat(nlambda, n_spcov, fill::zeros);
  
  /* main loop  */
  for (int j=0; j<nlambda; j++) {
    /* perform linear regression with lasso penalty for the j-th lambda       */
    out = admm_adalasso_cov_large_n_update_1lambda(W, Z, y, F, FTF_INV, WW, M_ZZW, M_ZZW2,
                                                   WMW, WMy, v_LS, P_ZW, WW_CHOL_U, Wy, y2,
                                                   ZZ_CHOL_U, Zy, ZW, u, z, lambda(j),
                                                   rho_adaptation, rho, tau, mu, reltol, abstol, maxiter, ping);

    /* Retrieve output      */
    u = as<arma::colvec>(out["u"]);
    z = as<arma::colvec>(out["z"]);
    x = as<arma::colvec>(out["x"]);
    v = as<arma::colvec>(out["v"]);

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
  output["sp.coef.path"]    = X;
  output["coef.path"]       = V;
  output["iternum"]         = niter;               // number of iterations
  output["objfun"]          = h_objval;            // |x|_1
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

/*
    Adaptive-LASSO via ADMM with covariate not penalized, fast routine to compute the whole path
            wrap for m and n (m denotes the number of observations, while n denotes the number of parameters)
*/
//' @name  fdaSP.internal
//' @keywords internal
// [[Rcpp::export(.admm_adalasso_cov_fast)]]
Rcpp::List admm_adalasso_cov_fast(arma::mat& W, arma::mat& Z, arma::vec& y, arma::vec& var_weights,
                                  const arma::vec lambda, bool rho_adaptation, double rho, const double tau, const double mu,
                                  const double reltol, const double abstol, const int maxiter, const int ping) {

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
    Sparse group-LASSO via ADMM with covariate not penalized, just one lambda, large m.
*/
Rcpp::List admm_spglasso_cov_large_m_update_1lambda(arma::mat& W, arma::mat& Z, arma::vec& y,
                                                    const arma::colvec Glen, const int G, const arma::sp_mat F, const arma::mat FTF,
                                                    const arma::mat WMW, const arma::vec WMy, const arma::vec v_LS, const arma::mat P_ZW,
                                                    const arma::mat& WW_CHOL_U, const arma::colvec& Wy, const double y2,
                                                    const arma::mat& ZZ_CHOL_U, const arma::colvec& Zy, const arma::mat ZW,
                                                    arma::colvec u, arma::colvec z, const double lambda, const double alpha,
                                                    bool rho_adaptation, double rho, const double tau, const double mu,
                                                    const double reltol, const double abstol, const int maxiter, const int ping) {
  
  /* Variable delcaration           */
  int k;
  double rho_old, elTime = 0.0, mse;
  uword g_id_init, g_id_end;
  
  /* Get dimensions             */
  const int m       = W.n_rows;
  const int n_spcov = W.n_cols;
  const int n_cov   = Z.n_cols;
  const int dim_z   = z.n_elem;
    
  /* Definition of vectors and matrices     */
  arma::colvec x(n_spcov, fill::zeros);
  arma::colvec v(n_cov, fill::zeros);
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
  //U = glasso_factor_fast_large_m(ATA, FTF, rho); // returns upper
  U = glasso_factor_cov_fast_large_m(WMW, FTF, rho);
  L = U.t();
  
  /* main loop  */
  for (k=0; k<maxiter; k++) {

    // 4-1. update 'x'
    if (rho != rho_old) {
      //U = glasso_factor_fast_large_m(ATA, FTF, rho); // returns upper
      U = glasso_factor_cov_fast_large_m(WMW, FTF, rho);
      L = U.t();
    }
    q = WMy + rho * spmat2vec_mult(F.t(), z - u);        // temporary value
    x = solve(trimatu(U), solve(trimatl(L), q));         // updated of regression parameters associated to covariates W (sparsified covariates, x)
    v = v_LS - P_ZW * x;                                 // updated of regression parameters associated to covariates Z (covariates not sparsified, v)
    
    // 4-2. update 'z' with adaptation
    // see Boyd et al (2011) page 21
    z_old     = z;
    g_id_init = 0;
    for (int g=0; g<G; g++) {
      // precompute the quantities for updating z
      g_id_end         = g_id_init + Glen(g) - 1;
      arma::vec u_g    = u.subvec(g_id_init, g_id_end);
      arma::sp_mat F_g = F.submat(g_id_init, 0, g_id_end, n_spcov-1);

      // update z
      arma::vec z_g                 = glasso_prox(F_g * x + u_g, lambda / rho);
      z.subvec(g_id_init, g_id_end) = z_g;
      g_id_init                     = g_id_end + 1;
    }
    // update z for the last group
    g_id_end                      = g_id_init + n_spcov - 1;
    arma::vec u_g                 = u.subvec(g_id_init, g_id_end);
    arma::sp_mat F_g              = F.submat(g_id_init, 0, g_id_end, n_spcov-1);
    arma::vec z_g                 = lasso_prox(F_g * x + u_g, lambda * alpha / rho);
    z.subvec(g_id_init, g_id_end) = z_g;

    // 4-3. update 'u'
    u += spmat2vec_mult(F, x) - z;
  
    // 4-3. dianostics, reporting
    //h_objval(k) = glasso_objfun_fast(ATA_CHOL_U, ATb, b2, Glen, lambda, x, z, m, n, G);
    h_objval(k)  = linreg_cov_logl_fast(WW_CHOL_U, ZZ_CHOL_U, ZW, Wy, Zy, y2, x, v, m);
    h_objval(k) += spglasso_cov_penfun(Glen, G, n_spcov, z, lambda, alpha);
    h_r_norm(k)  = norm(glasso_residual_sparse(F, x, z), 2);
    h_s_norm(k)  = norm(glasso_dual_residual_sparse(F, z, z_old, rho), 2);
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
  fitted    = W * x + Z * v;
  residuals = y - fitted;
  mse       = as_scalar(residuals.t() * residuals) / (static_cast<float>(m));
  
  /* Get output         */
  List output;
  output["x"]      = x;               // coefficient function
  output["v"]      = v;               // coefficient function
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
    Sparse group-LASSO via ADMM with covariate not penalized, fast routine to compute the whole path
          for large m where m denotes the number of observations
    exogenous covariates (not penalized) are included in this routine
*/
Rcpp::List admm_spglasso_cov_large_m_fast(arma::mat& W, arma::mat& Z, arma::vec& y,
                                          arma::mat& groups, arma::vec& group_weights, arma::vec& var_weights, arma::vec& var_weights_L1,
                                          const arma::vec lambda, const double alpha, bool rho_adaptation, double rho, const double tau, const double mu,
                                          const double reltol, const double abstol, const int maxiter, const int ping) {

  /* Variable delcaration           */
  double y2, min_mse, sqrtm;
  int n1, dim_z;
  uword idx_min_mse;

  /* Get dimensions             */
  const int m       = W.n_rows;
  const int n_spcov = W.n_cols;
  const int n_cov   = Z.n_cols;
  const int G       = groups.n_rows;
  
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
  arma::mat FTF(n_spcov, n_spcov, fill::zeros);
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
  arma::mat X(nlambda, n_spcov, fill::zeros);
  arma::mat V(nlambda, n_cov, fill::zeros);
  arma::vec conv(nlambda, fill::zeros);
  arma::mat residuals(nlambda, m, fill::zeros);
  arma::mat fitted(nlambda, m, fill::zeros);
  arma::vec mse(nlambda, fill::zeros);
  
  /* Precompute relevant quantities         */
  Z              = Z.each_col() % (ones_m * (1.0 / sqrtm));
  W              = W.each_col() % (ones_m * (1.0 / sqrtm));
  ZZ             = Z.t() * Z;
  L_ZZ           = (chol(ZZ)).t();
  L_ZZ_INV       = inv(trimatl(L_ZZ));
  ZZ_INV         = L_ZZ_INV.t() * L_ZZ_INV;
  P_ZZ           = Z * ZZ_INV * Z.t();
  M_ZZ           = eye_m - P_ZZ;
  WMW            = W.t() * M_ZZ * W;
  WMy            = W.t() * M_ZZ * y / static_cast<float>(sqrtm);
  Zy             = Z.t() * y / static_cast<float>(sqrtm);
  v_LS           = L_ZZ_INV.t() * L_ZZ_INV * Zy;
  ZW             = Z.t() * W;
  P_ZW           = L_ZZ_INV.t() * L_ZZ_INV * ZW;
  WW_CHOL_U      = chol_qr_fact(W, m, n_spcov);
  ZZ_CHOL_U      = chol_qr_fact(Z, m, n_cov);
  y2             = as_scalar(y.t() * y) / static_cast<float>(m);
  Wy             = W.t() * y / static_cast<float>(sqrtm);
  Glen           = sum(groups, 1);
  arma::sp_mat F = spglasso_Gmat2Fmat_sparse(groups, group_weights, var_weights, var_weights_L1);
  FTF            = diagmat(spmat2spmat_mult(F.t(), F));
  
  /* Initialize vectors u and z         */
  dim_z = n_spcov + sum(Glen);
  arma::colvec u(dim_z, fill::zeros);
  arma::colvec z(dim_z, fill::zeros);
  arma::mat Umat(nlambda, dim_z, fill::zeros);
  arma::mat Zmat(nlambda, dim_z, fill::zeros);

  /* main loop  */
  for (int j=0; j<nlambda; j++) {
    /* perform linear regression with lasso penalty for the j-th lambda       */
    out = admm_spglasso_cov_large_m_update_1lambda(W, Z, y, Glen, G, F, FTF,
                                                   WMW, WMy, v_LS, P_ZW, WW_CHOL_U, Wy, y2, ZZ_CHOL_U, Zy, ZW,
                                                   u, z, lambda(j), alpha, rho_adaptation, rho, tau, mu, reltol, abstol, maxiter, ping);

    /* Retrieve output      */
    u = as<arma::colvec>(out["u"]);
    z = as<arma::colvec>(out["z"]);
    x = as<arma::colvec>(out["x"]);
    v = as<arma::colvec>(out["v"]);

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
  output["sp.coef.path"]    = X;
  output["coef.path"]       = V;
  output["iternum"]         = niter;               // number of iterations
  output["objfun"]          = h_objval;        // |x|_1
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

/*
    Sparse group-LASSO via ADMM with covariate not penalized, just one lambda, large n.
*/
Rcpp::List admm_spglasso_cov_large_n_update_1lambda(arma::mat& W, arma::mat& Z, arma::vec& y, const arma::colvec Glen, const int G,
                                                    const arma::sp_mat F, const arma::mat FF_INV,
                                                    const arma::mat WW, const arma::mat M_ZZW, const arma::mat M_ZZW2,
                                                    const arma::mat WMW, const arma::vec WMy, const arma::vec v_LS, const arma::mat P_ZW,
                                                    const arma::mat& WW_CHOL_U, const arma::colvec& Wy, const double y2,
                                                    const arma::mat& ZZ_CHOL_U, const arma::colvec& Zy, const arma::mat ZW,
                                                    arma::colvec u, arma::colvec z, const double lambda, const double alpha,
                                                    bool rho_adaptation, double rho, const double tau, const double mu,
                                                    const double reltol, const double abstol, const int maxiter, const int ping) {
  

  /* Variable delcaration           */
  int k;
  double rho_old, elTime = 0.0, mse;
  uword g_id_init, g_id_end;

  /* Get dimensions             */
  const int m       = W.n_rows;
  const int n_spcov = W.n_cols;
  const int n_cov   = Z.n_cols;
  const int dim_z   = z.n_elem;
    
  /* Definition of vectors and matrices     */
  arma::colvec x(n_spcov, fill::zeros);
  arma::colvec x_star(n_spcov, fill::zeros);
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
  U = glasso_factor_cov_fast_large_n(M_ZZW2, rho, m); // returns upper
  L = U.t();
  
  /* main loop  */
  for (k=0; k<maxiter; k++) {

    // 4-1. update 'x'
    if (rho != rho_old) {
      U = glasso_factor_cov_fast_large_n(M_ZZW2, rho, m); // returns upper
      L = U.t();
    }
    q      = WMy + rho * spmat2vec_mult(F.t(), z - u); // temporary value
    x_star = (diagmat(FF_INV) * q) / rho;
    x      = x_star - diagmat(FF_INV) * M_ZZW.t() * solve(trimatu(U), solve(trimatl(L), M_ZZW * x_star)); // updated of regression parameters associated to covariates W
    v      = v_LS - P_ZW * x;                                 // updated of regression parameters associated to covariates Z (covariates not sparsified, v)

    // 4-2. update 'z' with adaptation
    // see Boyd et al (2011) page 21
    z_old     = z;
    g_id_init = 0;
    for (int g=0; g<G; g++) {
      // precompute the quantities for updating z
      g_id_end         = g_id_init + Glen(g) - 1;
      arma::vec u_g    = u.subvec(g_id_init, g_id_end);
      arma::sp_mat F_g = F.submat(g_id_init, 0, g_id_end, n_spcov-1);

      // update z
      arma::vec z_g                 = glasso_prox(F_g * x + u_g, lambda / rho);
      z.subvec(g_id_init, g_id_end) = z_g;
      g_id_init                     = g_id_end + 1;
    }
    // update z for the last group
    g_id_end                      = g_id_init + n_spcov - 1;
    arma::vec u_g                 = u.subvec(g_id_init, g_id_end);
    arma::sp_mat F_g              = F.submat(g_id_init, 0, g_id_end, n_spcov-1);
    arma::vec z_g                 = lasso_prox(F_g * x + u_g, lambda * alpha / rho);
    z.subvec(g_id_init, g_id_end) = z_g;

    // 4-3. update 'u'
    u += spmat2vec_mult(F, x) - z;
  
    // 4-3. dianostics, reporting
    //h_objval(k) = glasso_objfun_fast(ATA_CHOL_U, ATb, b2, Glen, lambda, x, z, m, n, G);
    h_objval(k)  = linreg_cov_logl_fast(WW_CHOL_U, ZZ_CHOL_U, ZW, Wy, Zy, y2, x, v, m);
    h_objval(k) += spglasso_cov_penfun(Glen, G, n_spcov, z, lambda, alpha);
    h_r_norm(k)  = norm(glasso_residual_sparse(F, x, z), 2);
    h_s_norm(k)  = norm(glasso_dual_residual_sparse(F, z, z_old, rho), 2);
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
  fitted    = W * x + Z * v;
  residuals = y - fitted;
  mse       = as_scalar(residuals.t() * residuals) / (static_cast<float>(m));
  
  /* Get output         */
  List output;
  output["x"]      = x;               // coefficient function
  output["v"]      = v;               // coefficient function
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
    Sparse group-LASSO via ADMM with covariate not penalized, fast routine to compute the whole path
          for large n where n denotes the number of parameters
    exogenous covariates (not penalized) are included in this routine
*/
Rcpp::List admm_spglasso_cov_large_n_fast(arma::mat& W, arma::mat& Z, arma::vec& y,
                                          arma::mat& groups, arma::vec& group_weights, arma::vec& var_weights, arma::vec& var_weights_L1,
                                          const arma::vec lambda, const double alpha, bool rho_adaptation, double rho, const double tau, const double mu,
                                          const double reltol, const double abstol, const int maxiter, const int ping) {

  /* Variable delcaration           */
  double y2, min_mse, sqrtm;
  int n1, dim_z;
  uword idx_min_mse;

  /* Get dimensions             */
  const int m       = W.n_rows;
  const int n_spcov = W.n_cols;
  const int n_cov   = Z.n_cols;
  const int G = groups.n_rows;
  
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
  arma::mat FTF(n_spcov, n_spcov, fill::zeros);
  arma::mat FTF_INV(n_spcov, n_spcov, fill::zeros);
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
  arma::mat X(nlambda, n_spcov, fill::zeros);
  arma::mat V(nlambda, n_cov, fill::zeros);
  arma::vec conv(nlambda, fill::zeros);
  arma::mat residuals(nlambda, m, fill::zeros);
  arma::mat fitted(nlambda, m, fill::zeros);
  arma::vec mse(nlambda, fill::zeros);
  
  /* Precompute relevant quantities         */
  Z         = Z.each_col() % (ones_m * (1.0 / sqrtm));
  //W         = W.each_col() % (ones_m * (1.0 / sqrtm));
  ZZ        = Z.t() * Z;
  L_ZZ      = (chol(ZZ)).t();
  L_ZZ_INV  = inv(trimatl(L_ZZ));
  ZZ_INV    = L_ZZ_INV.t() * L_ZZ_INV;
  P_ZZ      = Z * ZZ_INV * Z.t();
  M_ZZ      = eye_m - P_ZZ;
  WMW       = W.t() * M_ZZ * W;
  WMy       = W.t() * M_ZZ * y / static_cast<float>(m);
  Zy        = Z.t() * y / static_cast<float>(sqrtm);
  v_LS      = L_ZZ_INV.t() * L_ZZ_INV * Zy;
  ZW        = Z.t() * W / static_cast<float>(sqrtm);
  P_ZW      = L_ZZ_INV.t() * L_ZZ_INV * ZW;
  WW_CHOL_U = chol_qr_fact(W, m, n_spcov);
  ZZ_CHOL_U = chol_qr_fact(Z, m, n_cov);
  y2        = as_scalar(y.t() * y) / static_cast<float>(m);
  Wy        = W.t() * y / static_cast<float>(m);
  M_ZZW     = M_ZZ * W;
  //M_ZZW2    = M_ZZW * M_ZZW.t();
  
  Glen           = sum(groups, 1);
  arma::sp_mat F = spglasso_Gmat2Fmat_sparse(groups, group_weights, var_weights, var_weights_L1);
  FTF            = diagmat(spmat2spmat_mult(F.t(), F));
  FTF_INV        = diagmat(1.0 / diagvec(FTF));
  M_ZZW2         = M_ZZW * diagmat(FTF_INV) * M_ZZW.t();
  
  /* Initialize vectors u and z         */
  dim_z = n_spcov + sum(Glen);
  arma::colvec u(dim_z, fill::zeros);
  arma::colvec z(dim_z, fill::zeros);
  arma::mat Umat(nlambda, dim_z, fill::zeros);
  arma::mat Zmat(nlambda, dim_z, fill::zeros);
  
  /* main loop  */
  for (int j=0; j<nlambda; j++) {
    /* perform linear regression with lasso penalty for the j-th lambda       */
    out = admm_spglasso_cov_large_n_update_1lambda(W, Z, y, Glen, G, F, FTF_INV, WW, M_ZZW, M_ZZW2,
                                                   WMW, WMy, v_LS, P_ZW, WW_CHOL_U, Wy, y2,
                                                   ZZ_CHOL_U, Zy, ZW, u, z, lambda(j), alpha,
                                                   rho_adaptation, rho, tau, mu, reltol, abstol, maxiter, ping);

    /* Retrieve output      */
    u = as<arma::colvec>(out["u"]);
    z = as<arma::colvec>(out["z"]);
    x = as<arma::colvec>(out["x"]);
    v = as<arma::colvec>(out["v"]);

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
  output["sp.coef.path"]    = X;
  output["coef.path"]       = V;
  output["iternum"]         = niter;               // number of iterations
  output["objfun"]          = h_objval;            // |x|_1
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

/*
    Sparse Group-LASSO via ADMM with covariate not penalized, fast routine to compute the whole path
            wrap for m and n (m denotes the number of observations, while n denotes the number of parameters)
*/
//' @name  fdaSP.internal
//' @keywords internal
// [[Rcpp::export(.admm_spglasso_cov_fast)]]
Rcpp::List admm_spglasso_cov_fast(arma::mat& W, arma::mat& Z, arma::vec& y,
                                  arma::mat& groups, arma::vec& group_weights, arma::vec& var_weights, arma::vec& var_weights_L1,
                                  const arma::vec lambda, const double alpha, bool rho_adaptation, double rho, const double tau, const double mu,
                                  const double reltol, const double abstol, const int maxiter, const int ping) {

  /* Get dimensions             */
  const int m       = W.n_rows;
  const int n_spcov = W.n_cols;

  /* Variable delcaration           */
  List out;

  /* Run ADMM            */
  if (m >= n_spcov) {
    /* large n problems            */
    out = admm_spglasso_cov_large_m_fast(W, Z, y,
                                         groups, group_weights, var_weights, var_weights_L1,
                                         lambda, alpha, rho_adaptation, rho, tau, mu,
                                         reltol, abstol, maxiter, ping);
  } else {
    /* large p problems            */
    out = admm_spglasso_cov_large_n_fast(W, Z, y,
                                         groups, group_weights, var_weights, var_weights_L1,
                                         lambda, alpha, rho_adaptation, rho, tau, mu,
                                         reltol, abstol, maxiter, ping);
  }

  /* Return output      */
  return(out);
}

// =========================================================
// admm_lasso_cov
// ADMM for linear regression models with LASSO penalty
// This code includes exogenous covariates that are not penalized
// elementwise soft thresholding operator
/*
 
 Attention: the dimension of the vector x is p, where p is the number of columns
            of W (penalised covariates) and q is the dimension of the vector of
            regression parameters that are not penalized which correspond to
            the design matrix Z (non-penalised covariates),
            while the dimension of the vectors u and z are p
*/
//' @name  fdaSP.internal
//' @keywords internal
// [[Rcpp::export(.admm_lasso_cov)]]
Rcpp::List admm_lasso_cov(arma::mat& W, arma::mat& Z, const arma::colvec& y,
                          arma::colvec& u, arma::colvec& z, const double lambda,
                          bool rho_adaptation, double rho, const double tau, const double mu,
                          const double reltol, const double abstol, const int maxiter, const int ping) {
    
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

Rcpp::List admm_lasso_cov_large_m(arma::mat& W, arma::mat& Z, const arma::colvec& y,
                                  arma::colvec& u, arma::colvec& z, const double lambda,
                                  bool rho_adaptation, double rho, const double tau, const double mu,
                                  const double reltol, const double abstol, const int maxiter, const int ping) {
    
  /* Variable delcaration           */
  double sqrtn, sqrtm, rho_old, elTime, y2;
  int k, n1;
  
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
  Z         = Z.each_col() % (ones_m * (1.0 / sqrtm));
  W         = W.each_col() % (ones_m * (1.0 / sqrtm));
  WW_CHOL_U = chol_qr_fact(W, m, n_spcov);
  ZZ_CHOL_U = chol_qr_fact(Z, m, n_cov);
  y2        = as_scalar(y.t() * y) / static_cast<float>(m);
  Wy        = W.t() * y / static_cast<float>(m);
  ZZ        = Z.t() * Z;
  L_ZZ      = (chol(ZZ)).t();
  L_ZZ_INV  = inv(trimatl(L_ZZ));
  ZZ_INV    = L_ZZ_INV.t() * L_ZZ_INV;
  P_ZZ      = Z * ZZ_INV * Z.t();
  M_ZZ      = eye_m - P_ZZ;
  WMW       = W.t() * M_ZZ * W;
  WMy       = W.t() * M_ZZ * y / static_cast<float>(sqrtm);
  Zy        = Z.t() * y / static_cast<float>(sqrtm);
  v_LS      = L_ZZ_INV.t() * L_ZZ_INV * Zy;
  ZW        = Z.t() * W;
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
    //h_objval(k) = lasso_objfun(A, b, lambda, x, z);
    h_objval(k)  = linreg_cov_logl_fast(WW_CHOL_U, ZZ_CHOL_U, ZW, Wy, Zy, y2, x, v, m);
    h_objval(k) += lasso_cov_penfun(z, lambda);
    h_r_norm(k)  = norm(x - z, 2);
    h_s_norm(k)  = norm(-rho * (z - z_old), 2);
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
  output["x"]  = x;               // coefficient function
  output["v"]  = v;
  output["u"]  = u;
  output["z"]  = z;
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

Rcpp::List admm_lasso_cov_large_n(arma::mat& W, arma::mat& Z, const arma::colvec& y,
                                  arma::colvec& u, arma::colvec& z, const double lambda,
                                  bool rho_adaptation, double rho, const double tau, const double mu,
                                  const double reltol, const double abstol, const int maxiter, const int ping) {
    
  /* Variable delcaration           */
  double sqrtm, sqrtn, rho_old, elTime, y2;
  int n1, k;

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
  Z         = Z.each_col() % (ones_m * (1.0 / sqrtm));
  ZZ        = Z.t() * Z;
  L_ZZ      = (chol(ZZ)).t();
  L_ZZ_INV  = inv(trimatl(L_ZZ));
  ZZ_INV    = L_ZZ_INV.t() * L_ZZ_INV;
  P_ZZ      = Z * ZZ_INV * Z.t();
  M_ZZ      = eye_m - P_ZZ;
  WMW       = W.t() * M_ZZ * W;
  WMy       = W.t() * M_ZZ * y / static_cast<float>(m);
  Zy        = Z.t() * y / static_cast<float>(sqrtm);
  v_LS      = L_ZZ_INV.t() * L_ZZ_INV * Zy;
  ZW        = Z.t() * W / static_cast<float>(sqrtm);
  P_ZW      = L_ZZ_INV.t() * L_ZZ_INV * ZW;
  WW_CHOL_U = chol_qr_fact(W, m, n_spcov);
  ZZ_CHOL_U = chol_qr_fact(Z, m, n_cov);
  y2        = as_scalar(y.t() * y) / static_cast<float>(m);
  Wy        = W.t() * y / static_cast<float>(m);
  M_ZZW     = M_ZZ * W;
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
    //h_objval(k) = lasso_objfun(A, b, lambda, x, z);
    h_objval(k)  = linreg_cov_logl_fast(WW_CHOL_U, ZZ_CHOL_U, ZW, Wy, Zy, y2, x, v, m);
    h_objval(k) += lasso_cov_penfun(z, lambda);
    h_r_norm(k)  = norm(x - z, 2);
    h_s_norm(k)  = norm(-rho * (z - z_old), 2);
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
// admm_glasso_cov
// ADMM for linear regression models with group-LASSO penalty
// This code includes exogenous covariates that are not penalized
/*
 
 Attention: the dimension of the vector x is p, where p is the number of columns
            of W (penalised covariates) and q is the dimension of the vector of
            regression parameters that are not penalized which correspond to
            the design matrix Z (non-penalised covariates),
            while the dimension of the vectors u and z are p
*/
//' @name  fdaSP.internal
//' @keywords internal
// [[Rcpp::export(.admm_glasso_cov)]]
Rcpp::List admm_glasso_cov(arma::mat& W, arma::mat& Z, arma::vec& y, arma::colvec& u, arma::colvec& z,
                           arma::mat& groups, arma::vec& group_weights, arma::vec& var_weights,
                           double lambda, bool rho_adaptation, double rho, const double tau, const double mu,
                           const double reltol, const double abstol, const int maxiter, const int ping) {
  
  /* Get dimensions             */
  const int m       = W.n_rows;
  const int n_spcov = W.n_cols;
  
  /* Variable delcaration           */
  List out;
    
  /* Run ADMM            */
  if (m >= n_spcov) {
    /* large n problems            */
    out = admm_glasso_cov_large_m(W, Z, y, u, z,
                                  groups, group_weights, var_weights,
                                  lambda, rho_adaptation, rho, tau, mu,
                                  reltol, abstol, maxiter, ping);
  } else {
    /* large p problems            */
    out = admm_glasso_cov_large_n(W, Z, y, u, z,
                                  groups, group_weights, var_weights,
                                  lambda, rho_adaptation, rho, tau, mu,
                                  reltol, abstol, maxiter, ping);
  }

  /* Return output      */
  return(out);
}

Rcpp::List admm_glasso_cov_large_m(arma::mat& W, arma::mat& Z, arma::vec& y, arma::colvec& u, arma::colvec& z,
                                   arma::mat& groups, arma::vec& group_weights, arma::vec& var_weights,
                                   double lambda, bool rho_adaptation, double rho, const double tau, const double mu,
                                   const double reltol, const double abstol, const int maxiter, const int ping) {

  /* Variable delcaration           */
  double y2, sqrtm, rho_old, elTime;
  int n1, dim_z, k;
  uword g_id_init, g_id_end;

  /* Get dimensions             */
  const int m       = W.n_rows;
  const int n_spcov = W.n_cols;
  const int n_cov   = Z.n_cols;
  const int G       = groups.n_rows;
  
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
  arma::mat U(n_spcov, n_spcov, fill::zeros);
  arma::mat L(n_spcov, n_spcov, fill::zeros);
  arma::colvec x(n_spcov, fill::zeros);
  arma::colvec v(n_cov, fill::zeros);
  arma::colvec q(n_spcov, fill::zeros);
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
  arma::mat FTF(n_spcov, n_spcov, fill::zeros);
  arma::colvec Glen(G, fill::zeros);
  arma::vec h_objval(maxiter, fill::zeros);
  arma::vec h_r_norm(maxiter, fill::zeros);
  arma::vec h_s_norm(maxiter, fill::zeros);
  arma::vec h_eps_pri(maxiter, fill::zeros);
  arma::vec h_eps_dual(maxiter, fill::zeros);
  arma::vec rho_store(maxiter+1, fill::zeros);
  
  /* Precompute relevant quantities         */
  Z              = Z.each_col() % (ones_m * (1.0 / sqrtm));
  W              = W.each_col() % (ones_m * (1.0 / sqrtm));
  ZZ             = Z.t() * Z;
  L_ZZ           = (chol(ZZ)).t();
  L_ZZ_INV       = inv(trimatl(L_ZZ));
  ZZ_INV         = L_ZZ_INV.t() * L_ZZ_INV;
  P_ZZ           = Z * ZZ_INV * Z.t();
  M_ZZ           = eye_m - P_ZZ;
  WMW            = W.t() * M_ZZ * W;
  WMy            = W.t() * M_ZZ * y / static_cast<float>(sqrtm);
  Zy             = Z.t() * y / static_cast<float>(sqrtm);
  v_LS           = L_ZZ_INV.t() * L_ZZ_INV * Zy;
  ZW             = Z.t() * W;
  P_ZW           = L_ZZ_INV.t() * L_ZZ_INV * ZW;
  WW_CHOL_U      = chol_qr_fact(W, m, n_spcov);
  ZZ_CHOL_U      = chol_qr_fact(Z, m, n_cov);
  y2             = as_scalar(y.t() * y) / static_cast<float>(m);
  Wy             = W.t() * y / static_cast<float>(sqrtm);
  Glen           = sum(groups, 1);
  arma::sp_mat F = glasso_Gmat2Fmat_sparse(groups, group_weights, var_weights);
  FTF            = diagmat(spmat2spmat_mult(F.t(), F));
  
  /* Initialize vector z_old         */
  dim_z = sum(Glen);
  arma::colvec z_old(dim_z, fill::zeros);

  /* ::::::::::::::::::::::::::::::::::::::::::::::
   Set the computational time                                 */
  wall_clock timer;
  timer.tic();
  
  /* store rho (only for adaptation)         */
  rho_store(0) = rho;
  rho_old      = rho;

  /* Precompute relevant quantities         */
  //U = glasso_factor_fast_large_m(ATA, FTF, rho); // returns upper
  U = glasso_factor_cov_fast_large_m(WMW, FTF, rho);
  L = U.t();
  
  /* main loop  */
  for (k=0; k<maxiter; k++) {

    // 4-1. update 'x'
    if (rho != rho_old) {
      //U = glasso_factor_fast_large_m(ATA, FTF, rho); // returns upper
      U = glasso_factor_cov_fast_large_m(WMW, FTF, rho);
      L = U.t();
    }
    q = WMy + rho * spmat2vec_mult(F.t(), z - u);        // temporary value
    x = solve(trimatu(U), solve(trimatl(L), q));         // updated of regression parameters associated to covariates W (sparsified covariates, x)
    v = v_LS - P_ZW * x;                                 // updated of regression parameters associated to covariates Z (covariates not sparsified, v)
    
    // 4-2. update 'z' with adaptation
    // see Boyd et al (2011) page 21
    z_old     = z;
    g_id_init = 0;
    for (int g=0; g<G; g++) {
      // precompute the quantities for updating z
      g_id_end         = g_id_init + Glen(g) - 1;
      arma::vec u_g    = u.subvec(g_id_init, g_id_end);
      arma::sp_mat F_g = F.submat(g_id_init, 0, g_id_end, n_spcov-1);

      // update z
      arma::vec z_g                 = glasso_prox(F_g * x + u_g, lambda / rho);
      z.subvec(g_id_init, g_id_end) = z_g;
      g_id_init                     = g_id_end + 1;
    }

    // 4-3. update 'u'
    u += spmat2vec_mult(F, x) - z;
  
    // 4-3. dianostics, reporting
    //h_objval(k) = glasso_objfun_fast(ATA_CHOL_U, ATb, b2, Glen, lambda, x, z, m, n, G);
    h_objval(k)  = linreg_cov_logl_fast(WW_CHOL_U, ZZ_CHOL_U, ZW, Wy, Zy, y2, x, v, m);
    h_objval(k) += glasso_cov_penfun(Glen, G, z, lambda);
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
  
  /* Get output         */
  output["x"]      = x;               // coefficient function
  output["v"]      = v;               // coefficient function
  output["u"]      = u;
  output["z"]      = z;
  output["eltime"] = elTime;
  if (k < maxiter) {
    output["convergence"] = 1.0;
    output["niter"]       = k+1;                          // number of iterations
    output["objval"]      = h_objval.subvec(0, k);        // |x|_1
    output["r_norm"]      = h_r_norm.subvec(0, k);
    output["s_norm"]      = h_s_norm.subvec(0, k);
    output["eps_pri"]     = h_eps_pri.subvec(0, k);
    output["eps_dual"]    = h_eps_dual.subvec(0, k);
    output["rho"]         = rho_store.subvec(0, k+1);
  } else {
    output["convergence"] = 0.0;
    output["niter"]       = maxiter;               // number of iterations
    output["objval"]      = h_objval;              // |x|_1
    output["r_norm"]      = h_r_norm;
    output["s_norm"]      = h_s_norm;
    output["eps_pri"]     = h_eps_pri;
    output["eps_dual"]    = h_eps_dual;
    output["rho"]         = rho_store;
  }

  /* Return output      */
  return(output);
}

Rcpp::List admm_glasso_cov_large_n(arma::mat& W, arma::mat Z, arma::vec& y, arma::colvec& u, arma::colvec& z,
                                   arma::mat& groups, arma::vec& group_weights, arma::vec& var_weights,
                                   double lambda, bool rho_adaptation, double rho, const double tau, const double mu,
                                   const double reltol, const double abstol, const int maxiter, const int ping) {

  /* Variable delcaration           */
  double y2, sqrtm, rho_old, elTime;
  int n1, dim_z, k;
  uword g_id_init, g_id_end;

  /* Get dimensions             */
  const int m       = W.n_rows;
  const int n_spcov = W.n_cols;
  const int n_cov   = Z.n_cols;
  const int G       = groups.n_rows;
  
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
  arma::mat U(m, m, fill::zeros);
  arma::mat L(m, m, fill::zeros);
  arma::colvec x(n_spcov, fill::zeros);
  arma::colvec v(n_cov, fill::zeros);
  arma::colvec x_star(n_spcov, fill::zeros);
  arma::colvec q(n_spcov, fill::zeros);
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
  arma::mat FTF(n_spcov, n_spcov, fill::zeros);
  arma::mat FTF_INV(n_spcov, n_spcov, fill::zeros);
  arma::colvec Glen(G, fill::zeros);
  arma::vec h_objval(maxiter, fill::zeros);
  arma::vec h_r_norm(maxiter, fill::zeros);
  arma::vec h_s_norm(maxiter, fill::zeros);
  arma::vec h_eps_pri(maxiter, fill::zeros);
  arma::vec h_eps_dual(maxiter, fill::zeros);
  arma::vec rho_store(maxiter+1, fill::zeros);
  
  /* Precompute relevant quantities         */
  Z         = Z.each_col() % (ones_m * (1.0 / sqrtm));
  //W         = W.each_col() % (ones_m * (1.0 / sqrtm));
  ZZ        = Z.t() * Z;
  L_ZZ      = (chol(ZZ)).t();
  L_ZZ_INV  = inv(trimatl(L_ZZ));
  ZZ_INV    = L_ZZ_INV.t() * L_ZZ_INV;
  P_ZZ      = Z * ZZ_INV * Z.t();
  M_ZZ      = eye_m - P_ZZ;
  WMW       = W.t() * M_ZZ * W;
  WMy       = W.t() * M_ZZ * y / static_cast<float>(m);
  Zy        = Z.t() * y / static_cast<float>(sqrtm);
  v_LS      = L_ZZ_INV.t() * L_ZZ_INV * Zy;
  ZW        = Z.t() * W / static_cast<float>(sqrtm);
  P_ZW      = L_ZZ_INV.t() * L_ZZ_INV * ZW;
  WW_CHOL_U = chol_qr_fact(W, m, n_spcov);
  ZZ_CHOL_U = chol_qr_fact(Z, m, n_cov);
  y2        = as_scalar(y.t() * y) / static_cast<float>(m);
  Wy        = W.t() * y / static_cast<float>(m);
  M_ZZW     = M_ZZ * W;
  //M_ZZW2    = M_ZZW * M_ZZW.t();
  
  Glen           = sum(groups, 1);
  arma::sp_mat F = glasso_Gmat2Fmat_sparse(groups, group_weights, var_weights);
  FTF            = diagmat(spmat2spmat_mult(F.t(), F));
  FTF_INV        = diagmat(1.0 / diagvec(FTF));
  M_ZZW2         = M_ZZW * diagmat(FTF_INV) * M_ZZW.t();
  
  /* Initialize vector z_old         */
  dim_z = sum(Glen);
  arma::colvec z_old(dim_z, fill::zeros);
  
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
    q      = WMy + rho * spmat2vec_mult(F.t(), z - u); // temporary value
    x_star = (diagmat(FTF_INV) * q) / rho;
    x      = x_star - diagmat(FTF_INV) * M_ZZW.t() * solve(trimatu(U), solve(trimatl(L), M_ZZW * x_star)); // updated of regression parameters associated to covariates W
    v      = v_LS - P_ZW * x;                                 // updated of regression parameters associated to covariates Z (covariates not sparsified, v)

    // 4-2. update 'z' with adaptation
    // see Boyd et al (2011) page 21
    z_old     = z;
    g_id_init = 0;
    for (int g=0; g<G; g++) {
      // precompute the quantities for updating z
      g_id_end         = g_id_init + Glen(g) - 1;
      arma::vec u_g    = u.subvec(g_id_init, g_id_end);
      arma::sp_mat F_g = F.submat(g_id_init, 0, g_id_end, n_spcov-1);

      // update z
      arma::vec z_g                 = glasso_prox(F_g * x + u_g, lambda / rho);
      z.subvec(g_id_init, g_id_end) = z_g;
      g_id_init                     = g_id_end + 1;
    }

    // 4-3. update 'u'
    u += spmat2vec_mult(F, x) - z;
  
    // 4-3. dianostics, reporting
    //h_objval(k) = glasso_objfun_fast(ATA_CHOL_U, ATb, b2, Glen, lambda, x, z, m, n, G);
    h_objval(k)  = linreg_cov_logl_fast(WW_CHOL_U, ZZ_CHOL_U, ZW, Wy, Zy, y2, x, v, m);
    h_objval(k) += glasso_cov_penfun(Glen, G, z, lambda);
    h_r_norm(k)  = norm(glasso_residual_sparse(F, x, z), 2);
    h_s_norm(k)  = norm(glasso_dual_residual_sparse(F, z, z_old, rho), 2);
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
  
  /* Get output         */
  output["x"]      = x;               // coefficient function
  output["v"]      = v;               // coefficient function
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
  
  /* Return output      */
  return(output);
}

// =========================================================
// admm_ovglasso_cov
// ADMM for linear regression models with overlap group-LASSO penalty
// This code includes exogenous covariates that are not penalized
/*
 
 Attention: the dimension of the vector x is p, where p is the number of columns
            of W (penalised covariates) and q is the dimension of the vector of
            regression parameters that are not penalized which correspond to
            the design matrix Z (non-penalised covariates),
            while the dimension of the vectors u and z are p
*/
//' @name  fdaSP.internal
//' @keywords internal
// [[Rcpp::export(.admm_ovglasso_cov)]]
Rcpp::List admm_ovglasso_cov(arma::mat& W, arma::mat& Z, arma::vec& y, arma::colvec& u, arma::colvec& z,
                             arma::mat& groups, arma::vec& group_weights, arma::vec& var_weights,
                             double lambda, bool rho_adaptation, double rho, const double tau, const double mu,
                             const double reltol, const double abstol, const int maxiter, const int ping) {
  
  /* Get dimensions             */
  const int m       = W.n_rows;
  const int n_spcov = W.n_cols;
  
  /* Variable delcaration           */
  List out;
    
  /* Run ADMM            */
  if (m >= n_spcov) {
    /* large n problems            */
    out = admm_ovglasso_cov_large_m(W, Z, y, u, z,
                                    groups, group_weights, var_weights,
                                    lambda, rho_adaptation, rho, tau, mu,
                                    reltol, abstol, maxiter, ping);
  } else {
    /* large p problems            */
    out = admm_ovglasso_cov_large_m(W, Z, y, u, z,
                                    groups, group_weights, var_weights,
                                    lambda, rho_adaptation, rho, tau, mu,
                                    reltol, abstol, maxiter, ping);
  }

  /* Return output      */
  return(out);
}

Rcpp::List admm_ovglasso_cov_large_m(arma::mat& W, arma::mat& Z, arma::vec& y, arma::colvec& u, arma::colvec& z,
                                     arma::mat& groups, arma::vec& group_weights, arma::vec& var_weights,
                                     double lambda, bool rho_adaptation, double rho, const double tau, const double mu,
                                     const double reltol, const double abstol, const int maxiter, const int ping) {

  /* Variable delcaration           */
  double y2, sqrtm, rho_old, elTime;
  int n1, dim_z, k;
  uword g_id_init, g_id_end;

  /* Get dimensions             */
  const int m       = W.n_rows;
  const int n_spcov = W.n_cols;
  const int n_cov   = Z.n_cols;
  const int G       = groups.n_rows;
  
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
  arma::colvec q(n_spcov, fill::zeros);
  arma::colvec x_star(n_spcov, fill::zeros);
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
  arma::mat FTF(n_spcov, n_spcov, fill::zeros);
  arma::colvec Glen(G, fill::zeros);
  arma::vec rho_store(maxiter+1, fill::zeros);
  arma::vec h_objval(maxiter, fill::zeros);
  arma::vec h_r_norm(maxiter, fill::zeros);
  arma::vec h_s_norm(maxiter, fill::zeros);
  arma::vec h_eps_pri(maxiter, fill::zeros);
  arma::vec h_eps_dual(maxiter, fill::zeros);
  
  /* Precompute relevant quantities         */
  Z              = Z.each_col() % (ones_m * (1.0 / sqrtm));
  W              = W.each_col() % (ones_m * (1.0 / sqrtm));
  ZZ             = Z.t() * Z;
  L_ZZ           = (chol(ZZ)).t();
  L_ZZ_INV       = inv(trimatl(L_ZZ));
  ZZ_INV         = L_ZZ_INV.t() * L_ZZ_INV;
  P_ZZ           = Z * ZZ_INV * Z.t();
  M_ZZ           = eye_m - P_ZZ;
  WMW            = W.t() * M_ZZ * W;
  WMy            = W.t() * M_ZZ * y / static_cast<float>(sqrtm);
  Zy             = Z.t() * y / static_cast<float>(sqrtm);
  v_LS           = L_ZZ_INV.t() * L_ZZ_INV * Zy;
  ZW             = Z.t() * W;
  P_ZW           = L_ZZ_INV.t() * L_ZZ_INV * ZW;
  WW_CHOL_U      = chol_qr_fact(W, m, n_spcov);
  ZZ_CHOL_U      = chol_qr_fact(Z, m, n_cov);
  y2             = as_scalar(y.t() * y) / static_cast<float>(m);
  Wy             = W.t() * y / static_cast<float>(sqrtm);
  Glen           = sum(groups, 1);
  arma::sp_mat F = glasso_Gmat2Fmat_sparse(groups, group_weights, var_weights);
  FTF            = F.t() * F;
  
  /* Initialize vector z_old         */
  dim_z = sum(Glen);
  arma::colvec z_old(dim_z, fill::zeros);

  /* ::::::::::::::::::::::::::::::::::::::::::::::
   Set the computational time                                 */
  wall_clock timer;
  timer.tic();
  
  /* store rho (only for adaptation)         */
  rho_store(0) = rho;
  rho_old      = rho;

  /* Precompute relevant quantities         */
  //U = glasso_factor_fast_large_m(ATA, FTF, rho); // returns upper
  U = glasso_factor_cov_fast_large_m(WMW, FTF, rho);
  L = U.t();
  
  /* main loop  */
  for (k=0; k<maxiter; k++) {

    // 4-1. update 'x'
    if (rho != rho_old) {
      //U = glasso_factor_fast_large_m(ATA, FTF, rho); // returns upper
      U = glasso_factor_cov_fast_large_m(WMW, FTF, rho);
      L = U.t();
    }
    q = WMy + rho * spmat2vec_mult(F.t(), z - u);        // temporary value
    x = solve(trimatu(U), solve(trimatl(L), q));         // updated of regression parameters associated to covariates W (sparsified covariates, x)
    v = v_LS - P_ZW * x;                                 // updated of regression parameters associated to covariates Z (covariates not sparsified, v)
    
    // 4-2. update 'z' with adaptation
    // see Boyd et al (2011) page 21
    z_old     = z;
    g_id_init = 0;
    for (int g=0; g<G; g++) {
      // precompute the quantities for updating z
      g_id_end         = g_id_init + Glen(g) - 1;
      arma::vec u_g    = u.subvec(g_id_init, g_id_end);
      arma::sp_mat F_g = F.submat(g_id_init, 0, g_id_end, n_spcov-1);

      // update z
      arma::vec z_g                 = glasso_prox(F_g * x + u_g, lambda / rho);
      z.subvec(g_id_init, g_id_end) = z_g;
      g_id_init                     = g_id_end + 1;
    }

    // 4-3. update 'u'
    u += spmat2vec_mult(F, x) - z;
  
    // 4-3. dianostics, reporting
    //h_objval(k) = glasso_objfun_fast(ATA_CHOL_U, ATb, b2, Glen, lambda, x, z, m, n, G);
    h_objval(k)  = linreg_cov_logl_fast(WW_CHOL_U, ZZ_CHOL_U, ZW, Wy, Zy, y2, x, v, m);
    h_objval(k) += glasso_cov_penfun(Glen, G, z, lambda);
    h_r_norm(k)  = norm(glasso_residual_sparse(F, x, z), 2);
    h_s_norm(k)  = norm(glasso_dual_residual_sparse(F, z, z_old, rho), 2);
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
  
  /* Get output         */
  output["x"]      = x;               // coefficient function
  output["v"]      = v;               // coefficient function
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
  
  /* Return output      */
  return(output);
}

Rcpp::List admm_ovglasso_cov_large_n(arma::mat& W, arma::mat& Z, arma::vec& y, arma::colvec& u, arma::colvec& z,
                                     arma::mat& groups, arma::vec& group_weights, arma::vec& var_weights,
                                     double lambda, bool rho_adaptation, double rho, const double tau, const double mu,
                                     const double reltol, const double abstol, const int maxiter, const int ping) {

  /* Variable delcaration           */
  double y2, sqrtm, rho_old, elTime;
  int n1, dim_z, k;
  uword g_id_init, g_id_end;

  /* Get dimensions             */
  const int m       = W.n_rows;
  const int n_spcov = W.n_cols;
  const int n_cov   = Z.n_cols;
  const int G = groups.n_rows;
  
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
  arma::colvec q(n_spcov, fill::zeros);
  arma::colvec x_star(n_spcov, fill::zeros);
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
  arma::mat FTF(n_spcov, n_spcov, fill::zeros);
  arma::mat FTF_INV(n_spcov, n_spcov, fill::zeros);
  arma::colvec Glen(G, fill::zeros);
  arma::vec rho_store(maxiter+1, fill::zeros);
  arma::vec h_objval(maxiter, fill::zeros);
  arma::vec h_r_norm(maxiter, fill::zeros);
  arma::vec h_s_norm(maxiter, fill::zeros);
  arma::vec h_eps_pri(maxiter, fill::zeros);
  arma::vec h_eps_dual(maxiter, fill::zeros);
  
  /* Precompute relevant quantities         */
  Z         = Z.each_col() % (ones_m * (1.0 / sqrtm));
  //W         = W.each_col() % (ones_m * (1.0 / sqrtm));
  ZZ        = Z.t() * Z;
  L_ZZ      = (chol(ZZ)).t();
  L_ZZ_INV  = inv(trimatl(L_ZZ));
  ZZ_INV    = L_ZZ_INV.t() * L_ZZ_INV;
  P_ZZ      = Z * ZZ_INV * Z.t();
  M_ZZ      = eye_m - P_ZZ;
  WMW       = W.t() * M_ZZ * W;
  WMy       = W.t() * M_ZZ * y / static_cast<float>(m);
  Zy        = Z.t() * y / static_cast<float>(sqrtm);
  v_LS      = L_ZZ_INV.t() * L_ZZ_INV * Zy;
  ZW        = Z.t() * W / static_cast<float>(sqrtm);
  P_ZW      = L_ZZ_INV.t() * L_ZZ_INV * ZW;
  WW_CHOL_U = chol_qr_fact(W, m, n_spcov);
  ZZ_CHOL_U = chol_qr_fact(Z, m, n_cov);
  y2        = as_scalar(y.t() * y) / static_cast<float>(m);
  Wy        = W.t() * y / static_cast<float>(m);
  M_ZZW     = M_ZZ * W;
  //M_ZZW2    = M_ZZW * M_ZZW.t();
  
  Glen           = sum(groups, 1);
  arma::sp_mat F = glasso_Gmat2Fmat_sparse(groups, group_weights, var_weights);
  FTF            = diagmat(spmat2spmat_mult(F.t(), F));
  FTF_INV        = diagmat(1.0 / diagvec(FTF));
  M_ZZW2         = M_ZZW * diagmat(FTF_INV) * M_ZZW.t();
  
  /* Initialize vectors z_old         */
  dim_z = sum(Glen);
  arma::colvec z_old(dim_z, fill::zeros);
  
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
    q      = WMy + rho * spmat2vec_mult(F.t(), z - u); // temporary value
    x_star = (diagmat(FTF_INV) * q) / rho;
    x      = x_star - diagmat(FTF_INV) * M_ZZW.t() * solve(trimatu(U), solve(trimatl(L), M_ZZW * x_star)); // updated of regression parameters associated to covariates W
    v      = v_LS - P_ZW * x;                                 // updated of regression parameters associated to covariates Z (covariates not sparsified, v)

    // 4-2. update 'z' with adaptation
    // see Boyd et al (2011) page 21
    z_old     = z;
    g_id_init = 0;
    for (int g=0; g<G; g++) {
      // precompute the quantities for updating z
      g_id_end         = g_id_init + Glen(g) - 1;
      arma::vec u_g    = u.subvec(g_id_init, g_id_end);
      arma::sp_mat F_g = F.submat(g_id_init, 0, g_id_end, n_spcov-1);

      // update z
      arma::vec z_g                 = glasso_prox(F_g * x + u_g, lambda / rho);
      z.subvec(g_id_init, g_id_end) = z_g;
      g_id_init                     = g_id_end + 1;
    }

    // 4-3. update 'u'
    u += spmat2vec_mult(F, x) - z;
  
    // 4-3. dianostics, reporting
    //h_objval(k) = glasso_objfun_fast(ATA_CHOL_U, ATb, b2, Glen, lambda, x, z, m, n, G);
    h_objval(k)  = linreg_cov_logl_fast(WW_CHOL_U, ZZ_CHOL_U, ZW, Wy, Zy, y2, x, v, m);
    h_objval(k) += glasso_cov_penfun(Glen, G, z, lambda);
    h_r_norm(k)  = norm(glasso_residual_sparse(F, x, z), 2);
    h_s_norm(k)  = norm(glasso_dual_residual_sparse(F, z, z_old, rho), 2);
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
  
  /* Get output         */
  output["x"]      = x;               // coefficient function
  output["v"]      = v;               // coefficient function
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

  /* Return output      */
  return(output);
}

// =========================================================
// admm_adalasso_cov
// ADMM for linear regression models with adaptive-LASSO penalty
// This code includes exogenous covariates that are not penalized
/*
 
 Attention: the dimension of the vector x is p, where p is the number of columns
            of W (penalised covariates) and q is the dimension of the vector of
            regression parameters that are not penalized which correspond to
            the design matrix Z (non-penalised covariates),
            while the dimension of the vectors u and z are p
*/
//' @name  fdaSP.internal
//' @keywords internal
// [[Rcpp::export(.admm_adalasso_cov)]]
Rcpp::List admm_adalasso_cov(arma::mat W, arma::mat& Z, arma::colvec& y, arma::vec& var_weights,
                             arma::colvec& u, arma::colvec& z, const double lambda,
                             bool rho_adaptation, double rho, const double tau, const double mu,
                             const double reltol, const double abstol, const int maxiter, const int ping) {
  
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

Rcpp::List admm_adalasso_cov_large_m(arma::mat& W, arma::mat& Z, arma::vec& y, arma::vec& var_weights, arma::colvec& u, arma::colvec& z,
                                     double lambda, bool rho_adaptation, double rho, const double tau, const double mu,
                                     const double reltol, const double abstol, const int maxiter, const int ping) {

  /* Variable delcaration           */
  double y2, sqrtm, rho_old, elTime;
  int n1, k;

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
  arma::mat FTF(n_spcov, n_spcov, fill::zeros);
  arma::vec rho_store(maxiter+1, fill::zeros);
  arma::vec h_objval(maxiter, fill::zeros);
  arma::vec h_r_norm(maxiter, fill::zeros);
  arma::vec h_s_norm(maxiter, fill::zeros);
  arma::vec h_eps_pri(maxiter, fill::zeros);
  arma::vec h_eps_dual(maxiter, fill::zeros);
  
  /* Precompute relevant quantities         */
  Z              = Z.each_col() % (ones_m * (1.0 / sqrtm));
  W              = W.each_col() % (ones_m * (1.0 / sqrtm));
  ZZ             = Z.t() * Z;
  L_ZZ           = (chol(ZZ)).t();
  L_ZZ_INV       = inv(trimatl(L_ZZ));
  ZZ_INV         = L_ZZ_INV.t() * L_ZZ_INV;
  P_ZZ           = Z * ZZ_INV * Z.t();
  M_ZZ           = eye_m - P_ZZ;
  WMW            = W.t() * M_ZZ * W;
  WMy            = W.t() * M_ZZ * y / static_cast<float>(sqrtm);
  Zy             = Z.t() * y / static_cast<float>(sqrtm);
  v_LS           = L_ZZ_INV.t() * L_ZZ_INV * Zy;
  ZW             = Z.t() * W;
  P_ZW           = L_ZZ_INV.t() * L_ZZ_INV * ZW;
  WW_CHOL_U      = chol_qr_fact(W, m, n_spcov);
  ZZ_CHOL_U      = chol_qr_fact(Z, m, n_cov);
  y2             = as_scalar(y.t() * y) / static_cast<float>(m);
  Wy             = W.t() * y / static_cast<float>(sqrtm);
  
  /* Define the F matrix         */
  arma::mat F = adalasso_Fmat(var_weights);
  FTF         = diagmat(F) * diagmat(F);

  /* ::::::::::::::::::::::::::::::::::::::::::::::
   Set the computational time                                 */
  wall_clock timer;
  timer.tic();
  
  /* store rho (only for adaptation)         */
  rho_store(0) = rho;
  rho_old      = rho;

  /* Precompute relevant quantities         */
  //U = glasso_factor_fast_large_m(ATA, FTF, rho); // returns upper
  U = glasso_factor_cov_fast_large_m(WMW, FTF, rho);
  L = U.t();
  
  /* main loop  */
  for (k=0; k<maxiter; k++) {

    // 4-1. update 'x'
    if (rho != rho_old) {
      //U = glasso_factor_fast_large_m(ATA, FTF, rho); // returns upper
      U = glasso_factor_cov_fast_large_m(WMW, FTF, rho);
      L = U.t();
    }
    q = WMy + rho * diagmat(F) * (z - u);           // temporary value
    x = solve(trimatu(U), solve(trimatl(L), q));         // updated of regression parameters associated to covariates W (sparsified covariates, x)
    v = v_LS - P_ZW * x;                                 // updated of regression parameters associated to covariates Z (covariates not sparsified, v)
    
    // 4-2. update 'z' with adaptation
    // see Boyd et al (2011) page 21
    z_old = z;
    z     = lasso_prox(diagmat(F) * x + u, lambda / rho);

    // 4-3. update 'u'
    u += diagmat(F) * x - z;
  
    // 4-3. dianostics, reporting
    //h_objval(k) = glasso_objfun_fast(ATA_CHOL_U, ATb, b2, Glen, lambda, x, z, m, n, G);
    h_objval(k)  = linreg_cov_logl_fast(WW_CHOL_U, ZZ_CHOL_U, ZW, Wy, Zy, y2, x, v, m);
    h_objval(k) += adalasso_cov_penfun(z, lambda, diagvec(F));
    h_r_norm(k)  = norm(adalasso_residual(F, x, z), 2);
    h_s_norm(k)  = norm(adalasso_dual_residual(F, z, z_old, rho), 2);
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
  output["x"]      = x;               // coefficient function
  output["v"]      = v;               // coefficient function
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
  
  /* Return output      */
  return(output);
}

Rcpp::List admm_adalasso_cov_large_n(arma::mat& W, arma::mat& Z, arma::vec& y, arma::vec& var_weights, arma::colvec& u, arma::colvec& z,
                                     double lambda, bool rho_adaptation, double rho, const double tau, const double mu,
                                     const double reltol, const double abstol, const int maxiter, const int ping) {

  /* Variable delcaration           */
  double y2, sqrtm, rho_old, elTime;
  int n1, k;

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
  arma::mat FTF(n_spcov, n_spcov, fill::zeros);
  arma::mat FTF_INV(n_spcov, n_spcov, fill::zeros);
  arma::vec rho_store(maxiter+1, fill::zeros);
  arma::vec h_objval(maxiter, fill::zeros);
  arma::vec h_r_norm(maxiter, fill::zeros);
  arma::vec h_s_norm(maxiter, fill::zeros);
  arma::vec h_eps_pri(maxiter, fill::zeros);
  arma::vec h_eps_dual(maxiter, fill::zeros);

  /* Precompute relevant quantities         */
  Z         = Z.each_col() % (ones_m * (1.0 / sqrtm));
  ZZ        = Z.t() * Z;
  L_ZZ      = (chol(ZZ)).t();
  L_ZZ_INV  = inv(trimatl(L_ZZ));
  ZZ_INV    = L_ZZ_INV.t() * L_ZZ_INV;
  P_ZZ      = Z * ZZ_INV * Z.t();
  M_ZZ      = eye_m - P_ZZ;
  WMW       = W.t() * M_ZZ * W;
  WMy       = W.t() * M_ZZ * y / static_cast<float>(m);
  Zy        = Z.t() * y / static_cast<float>(sqrtm);
  v_LS      = L_ZZ_INV.t() * L_ZZ_INV * Zy;
  ZW        = Z.t() * W / static_cast<float>(sqrtm);
  P_ZW      = L_ZZ_INV.t() * L_ZZ_INV * ZW;
  WW_CHOL_U = chol_qr_fact(W, m, n_spcov);
  ZZ_CHOL_U = chol_qr_fact(Z, m, n_cov);
  y2        = as_scalar(y.t() * y) / static_cast<float>(m);
  Wy        = W.t() * y / static_cast<float>(m);
  M_ZZW     = M_ZZ * W;
  
  /* Define the F matrix         */
  arma::mat F = adalasso_Fmat(var_weights);
  FTF         = diagmat(F) * diagmat(F);
  FTF_INV     = diagmat(1.0 / diagvec(FTF));
  M_ZZW2      = M_ZZW * diagmat(FTF_INV) * M_ZZW.t();
  
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
    q      = WMy + rho * diagmat(F) * (z - u); // temporary value
    x_star = (diagmat(FTF_INV) * q) / rho;
    x      = x_star - diagmat(FTF_INV) * M_ZZW.t() * solve(trimatu(U), solve(trimatl(L), M_ZZW * x_star)); // updated of regression parameters associated to covariates W
    v      = v_LS - P_ZW * x;                                 // updated of regression parameters associated to covariates Z (covariates not sparsified, v)

    // 4-2. update 'z' with adaptation
    // see Boyd et al (2011) page 21
    z_old = z;
    z     = lasso_prox(diagmat(F) * x + u, lambda / rho);

    // 4-3. update 'u'
    u = u + diagmat(F) * x - z;
  
    // 4-3. dianostics, reporting
    //h_objval(k) = glasso_objfun_fast(ATA_CHOL_U, ATb, b2, Glen, lambda, x, z, m, n, G);
    h_objval(k)  = linreg_cov_logl_fast(WW_CHOL_U, ZZ_CHOL_U, ZW, Wy, Zy, y2, x, v, m);
    h_objval(k) += adalasso_cov_penfun(z, lambda, diagvec(F));
    h_r_norm(k)  = norm(adalasso_residual(F, x, z), 2);
    h_s_norm(k)  = norm(adalasso_dual_residual(F, z, z_old, rho), 2);
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
  output["x"]      = x;               // coefficient function
  output["v"]      = v;               // coefficient function
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
  
  /* Return output      */
  return(output);
}

// =========================================================
// admm_spglasso_cov
// ADMM for linear regression models with sparse group-LASSO penalty
// This code includes exogenous covariates that are not penalized
/*
 
 Attention: the dimension of the vector x is p, where p is the number of columns
            of W (penalised covariates) and q is the dimension of the vector of
            regression parameters that are not penalized which correspond to
            the design matrix Z (non-penalised covariates),
            while the dimension of the vectors u and z are p
*/
//' @name  fdaSP.internal
//' @keywords internal
// [[Rcpp::export(.admm_spglasso_cov)]]
Rcpp::List admm_spglasso_cov(arma::mat& W, arma::mat& Z, arma::vec& y,
                             arma::mat& groups, arma::vec& group_weights,
                             arma::vec& var_weights, arma::vec& var_weights_L1,
                             arma::colvec& u, arma::colvec& z, const double lambda1, const double lambda2,
                             bool rho_adaptation, double rho, const double tau, const double mu,
                             const double reltol, const double abstol, const int maxiter, const int ping) {
  
  /* Get dimensions             */
  const int m       = W.n_rows;
  const int n_spcov = W.n_cols;
  
  /* Variable delcaration           */
  List out;
    
  /* Run ADMM            */
  if (m >= n_spcov) {
    /* large n problems            */
    if (lambda2 == 0) {
      out = admm_glasso_cov_large_m(W, Z, y, u, z,
                                    groups, group_weights, var_weights,
                                    lambda1, rho_adaptation, rho, tau, mu,
                                    reltol, abstol, maxiter, ping);
    } else if (lambda2 == 1) {
      out = admm_adalasso_cov_large_m(W, Z, y, var_weights, u, z, lambda1,
                                      rho_adaptation, rho, tau, mu,
                                      reltol, abstol, maxiter, ping);
    } else {
      out = admm_spglasso_cov_large_m(W, Z, y, u, z,
                                      groups, group_weights, var_weights, var_weights_L1,
                                      lambda1, lambda2, rho_adaptation, rho, tau, mu,
                                      reltol, abstol, maxiter, ping);
    }
  } else {
    /* large p problems            */
    if (lambda2 == 0) {
      out = admm_glasso_cov_large_n(W, Z, y, u, z,
                                    groups, group_weights, var_weights,
                                    lambda1, rho_adaptation, rho, tau, mu,
                                    reltol, abstol, maxiter, ping);
    } else if (lambda2 == 1) {
      out = admm_adalasso_cov_large_n(W, Z, y, var_weights, u, z, lambda1,
                                      rho_adaptation, rho, tau, mu,
                                      reltol, abstol, maxiter, ping);
    } else {
      out = admm_spglasso_cov_large_n(W, Z, y, u, z,
                                      groups, group_weights, var_weights, var_weights_L1,
                                      lambda1, lambda2, rho_adaptation, rho, tau, mu,
                                      reltol, abstol, maxiter, ping);
    }
  }

  /* Return output      */
  return(out);
}

Rcpp::List admm_spglasso_cov_large_m(arma::mat& W, arma::mat& Z, arma::vec& y, arma::colvec& u, arma::colvec& z,
                                     arma::mat& groups, arma::vec& group_weights, arma::vec& var_weights, arma::vec& var_weights_L1,
                                     double lambda, const double alpha, bool rho_adaptation, double rho, const double tau, const double mu,
                                     const double reltol, const double abstol, const int maxiter, const int ping) {

  /* Variable delcaration           */
  double y2, sqrtm, rho_old, elTime;
  int n1, dim_z, k;
  uword g_id_init, g_id_end;

  /* Get dimensions             */
  const int m       = W.n_rows;
  const int n_spcov = W.n_cols;
  const int n_cov   = Z.n_cols;
  const int G       = groups.n_rows;
  
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
  arma::colvec q(n_spcov, fill::zeros);
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
  arma::mat FTF(n_spcov, n_spcov, fill::zeros);
  arma::colvec Glen(G, fill::zeros);
  arma::vec rho_store(maxiter+1, fill::zeros);
  arma::vec h_objval(maxiter, fill::zeros);
  arma::vec h_r_norm(maxiter, fill::zeros);
  arma::vec h_s_norm(maxiter, fill::zeros);
  arma::vec h_eps_pri(maxiter, fill::zeros);
  arma::vec h_eps_dual(maxiter, fill::zeros);
  
  /* Precompute relevant quantities         */
  Z              = Z.each_col() % (ones_m * (1.0 / sqrtm));
  W              = W.each_col() % (ones_m * (1.0 / sqrtm));
  ZZ             = Z.t() * Z;
  L_ZZ           = (chol(ZZ)).t();
  L_ZZ_INV       = inv(trimatl(L_ZZ));
  ZZ_INV         = L_ZZ_INV.t() * L_ZZ_INV;
  P_ZZ           = Z * ZZ_INV * Z.t();
  M_ZZ           = eye_m - P_ZZ;
  WMW            = W.t() * M_ZZ * W;
  WMy            = W.t() * M_ZZ * y / static_cast<float>(sqrtm);
  Zy             = Z.t() * y / static_cast<float>(sqrtm);
  v_LS           = L_ZZ_INV.t() * L_ZZ_INV * Zy;
  ZW             = Z.t() * W;
  P_ZW           = L_ZZ_INV.t() * L_ZZ_INV * ZW;
  WW_CHOL_U      = chol_qr_fact(W, m, n_spcov);
  ZZ_CHOL_U      = chol_qr_fact(Z, m, n_cov);
  y2             = as_scalar(y.t() * y) / static_cast<float>(m);
  Wy             = W.t() * y / static_cast<float>(sqrtm);
  Glen           = sum(groups, 1);
  arma::sp_mat F = spglasso_Gmat2Fmat_sparse(groups, group_weights, var_weights, var_weights_L1);
  FTF            = diagmat(spmat2spmat_mult(F.t(), F));
  
  /* Initialize vectors z_old         */
  dim_z = n_spcov + sum(Glen);
  arma::colvec z_old(dim_z, fill::zeros);

  /* ::::::::::::::::::::::::::::::::::::::::::::::
   Set the computational time                                 */
  wall_clock timer;
  timer.tic();
  
  /* store rho (only for adaptation)         */
  rho_store(0) = rho;
  rho_old      = rho;

  /* Precompute relevant quantities         */
  //U = glasso_factor_fast_large_m(ATA, FTF, rho); // returns upper
  U = glasso_factor_cov_fast_large_m(WMW, FTF, rho);
  L = U.t();
  
  /* main loop  */
  for (k=0; k<maxiter; k++) {

    // 4-1. update 'x'
    if (rho != rho_old) {
      //U = glasso_factor_fast_large_m(ATA, FTF, rho); // returns upper
      U = glasso_factor_cov_fast_large_m(WMW, FTF, rho);
      L = U.t();
    }
    q = WMy + rho * spmat2vec_mult(F.t(), z - u);        // temporary value
    x = solve(trimatu(U), solve(trimatl(L), q));         // updated of regression parameters associated to covariates W (sparsified covariates, x)
    v = v_LS - P_ZW * x;                                 // updated of regression parameters associated to covariates Z (covariates not sparsified, v)
    
    // 4-2. update 'z' with adaptation
    // see Boyd et al (2011) page 21
    z_old     = z;
    g_id_init = 0;
    for (int g=0; g<G; g++) {
      // precompute the quantities for updating z
      g_id_end         = g_id_init + Glen(g) - 1;
      arma::vec u_g    = u.subvec(g_id_init, g_id_end);
      arma::sp_mat F_g = F.submat(g_id_init, 0, g_id_end, n_spcov-1);

      // update z
      arma::vec z_g                 = glasso_prox(F_g * x + u_g, lambda / rho);
      z.subvec(g_id_init, g_id_end) = z_g;
      g_id_init                     = g_id_end + 1;
    }
    // update z for the last group
    g_id_end                      = g_id_init + n_spcov - 1;
    arma::vec u_g                 = u.subvec(g_id_init, g_id_end);
    arma::sp_mat F_g              = F.submat(g_id_init, 0, g_id_end, n_spcov-1);
    arma::vec z_g                 = lasso_prox(F_g * x + u_g, lambda * alpha / rho);
    z.subvec(g_id_init, g_id_end) = z_g;

    // 4-3. update 'u'
    u += spmat2vec_mult(F, x) - z;
  
    // 4-3. dianostics, reporting
    //h_objval(k) = glasso_objfun_fast(ATA_CHOL_U, ATb, b2, Glen, lambda, x, z, m, n, G);
    h_objval(k)  = linreg_cov_logl_fast(WW_CHOL_U, ZZ_CHOL_U, ZW, Wy, Zy, y2, x, v, m);
    h_objval(k) += spglasso_cov_penfun(Glen, G, n_spcov, z, lambda, alpha);
    h_r_norm(k)  = norm(glasso_residual_sparse(F, x, z), 2);
    h_s_norm(k)  = norm(glasso_dual_residual_sparse(F, z, z_old, rho), 2);
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
  
  /* Get output         */
  output["x"]      = x;               // coefficient function
  output["v"]      = v;               // coefficient function
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
  
  /* Return output      */
  return(output);
}

Rcpp::List admm_spglasso_cov_large_n(arma::mat& W, arma::mat& Z, arma::vec& y, arma::colvec& u, arma::colvec& z,
                                     arma::mat& groups, arma::vec& group_weights, arma::vec& var_weights, arma::vec& var_weights_L1,
                                     double lambda, const double alpha, bool rho_adaptation, double rho, const double tau, const double mu,
                                     const double reltol, const double abstol, const int maxiter, const int ping) {

  /* Variable delcaration           */
  double y2, sqrtm, rho_old, elTime;
  int n1, dim_z, k;
  uword g_id_init, g_id_end;

  /* Get dimensions             */
  const int m       = W.n_rows;
  const int n_spcov = W.n_cols;
  const int n_cov   = Z.n_cols;
  const int G = groups.n_rows;
  
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
  arma::colvec x_star(n_spcov, fill::zeros);
  arma::colvec v(n_cov, fill::zeros);
  arma::colvec q(n_spcov, fill::zeros);
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
  arma::mat FTF(n_spcov, n_spcov, fill::zeros);
  arma::mat FTF_INV(n_spcov, n_spcov, fill::zeros);
  arma::colvec Glen(G, fill::zeros);
  arma::vec rho_store(maxiter+1, fill::zeros);
  arma::vec h_objval(maxiter, fill::zeros);
  arma::vec h_r_norm(maxiter, fill::zeros);
  arma::vec h_s_norm(maxiter, fill::zeros);
  arma::vec h_eps_pri(maxiter, fill::zeros);
  arma::vec h_eps_dual(maxiter, fill::zeros);
  
  /* Precompute relevant quantities         */
  Z         = Z.each_col() % (ones_m * (1.0 / sqrtm));
  //W         = W.each_col() % (ones_m * (1.0 / sqrtm));
  ZZ        = Z.t() * Z;
  L_ZZ      = (chol(ZZ)).t();
  L_ZZ_INV  = inv(trimatl(L_ZZ));
  ZZ_INV    = L_ZZ_INV.t() * L_ZZ_INV;
  P_ZZ      = Z * ZZ_INV * Z.t();
  M_ZZ      = eye_m - P_ZZ;
  WMW       = W.t() * M_ZZ * W;
  WMy       = W.t() * M_ZZ * y / static_cast<float>(m);
  Zy        = Z.t() * y / static_cast<float>(sqrtm);
  v_LS      = L_ZZ_INV.t() * L_ZZ_INV * Zy;
  ZW        = Z.t() * W / static_cast<float>(sqrtm);
  P_ZW      = L_ZZ_INV.t() * L_ZZ_INV * ZW;
  WW_CHOL_U = chol_qr_fact(W, m, n_spcov);
  ZZ_CHOL_U = chol_qr_fact(Z, m, n_cov);
  y2        = as_scalar(y.t() * y) / static_cast<float>(m);
  Wy        = W.t() * y / static_cast<float>(m);
  M_ZZW     = M_ZZ * W;
  //M_ZZW2    = M_ZZW * M_ZZW.t();
  
  Glen           = sum(groups, 1);
  arma::sp_mat F = spglasso_Gmat2Fmat_sparse(groups, group_weights, var_weights, var_weights_L1);
  FTF            = diagmat(spmat2spmat_mult(F.t(), F));
  FTF_INV        = diagmat(1.0 / diagvec(FTF));
  M_ZZW2         = M_ZZW * diagmat(FTF_INV) * M_ZZW.t();
  
  /* Initialize vectors z_old and x_star         */
  dim_z = n_spcov + sum(Glen);
  arma::colvec z_old(dim_z, fill::zeros);

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
    q      = WMy + rho * spmat2vec_mult(F.t(), z - u); // temporary value
    x_star = (diagmat(FTF_INV) * q) / rho;
    x      = x_star - diagmat(FTF_INV) * M_ZZW.t() * solve(trimatu(U), solve(trimatl(L), M_ZZW * x_star)); // updated of regression parameters associated to covariates W
    v      = v_LS - P_ZW * x;                                 // updated of regression parameters associated to covariates Z (covariates not sparsified, v)

    // 4-2. update 'z' with adaptation
    // see Boyd et al (2011) page 21
    z_old     = z;
    g_id_init = 0;
    for (int g=0; g<G; g++) {
      // precompute the quantities for updating z
      g_id_end         = g_id_init + Glen(g) - 1;
      arma::vec u_g    = u.subvec(g_id_init, g_id_end);
      arma::sp_mat F_g = F.submat(g_id_init, 0, g_id_end, n_spcov-1);

      // update z
      arma::vec z_g                 = glasso_prox(F_g * x + u_g, lambda / rho);
      z.subvec(g_id_init, g_id_end) = z_g;
      g_id_init                     = g_id_end + 1;
    }

    // 4-3. update 'u'
    u += spmat2vec_mult(F, x) - z;
  
    // 4-3. dianostics, reporting
    //h_objval(k) = glasso_objfun_fast(ATA_CHOL_U, ATb, b2, Glen, lambda, x, z, m, n, G);
    h_objval(k)  = linreg_cov_logl_fast(WW_CHOL_U, ZZ_CHOL_U, ZW, Wy, Zy, y2, x, v, m);
    h_objval(k) += spglasso_cov_penfun(Glen, G, n_spcov, z, lambda, alpha);
    h_r_norm(k)  = norm(glasso_residual_sparse(F, x, z), 2);
    h_s_norm(k)  = norm(glasso_dual_residual_sparse(F, z, z_old, rho), 2);
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
  
  /* Get output         */
  output["x"]      = x;               // coefficient function
  output["v"]      = v;               // coefficient function
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

  /* Return output      */
  return(output);
}






/* end of file          */
