// Convex Loacal Adaptive Functional Clustering via ADMM
// Main routines

// Authors:
//           Bernardi Mauro, University of Padova
//           Last update: May 1, 2024

// List of implemented methods:

//#define ARMA_USE_SUPERLU
//#define ARMA_USE_LAPACK
//#define ARMA_USE_BLAS

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(cpp11)]]
#include <RcppArmadillo.h>

// [[Rcpp::depends(RcppArmadillo)]]
#include "dof_utils.h"

/* Utility functions for the calculation of the degrees of freedom for
        convex clustering.
 
 Functions:
 
    0.1 ovglasso_group_selmat
    0.2 lm_OVGLASSO_admm_active_sets
 
    1.1 lm_dof_LASSO_1lambda                    # linear regression, con penalty lasso
    1.2 lm_dof_OVGLASSO_1lambda                 # linear regression, con penalty ovg lasso
    1.3 lm_cov_dof_OVGLASSO_1lambda             # linear regression, con penalty ovg lasso e con Z
    1.4 lm_adaptive_dof_OVGLASSO_1lambda        # linear regression, con penalty ovg lasso e pesi adattivi
 
    2.1 f2s_dof_1lambda                         # function to scalar regression, senza Z e senza l2 penalty
    2.2 f2s_cov_dof_1lambda                     # function to scalar regression, con Z ma senza l2 penalty
    2.3 f2s_smo_dof_1lambda                     # function to scalar regression, senza Z ma con l2 penalty
    2.4 f2s_cov_smo_dof_1lambda                 # function to scalar regression, con Z e l2 penalty
    2.5 f2s_adaptive_dof_1lambda                # versione con pesi adattivi di 2.1
    2.6 f2s_cov_adaptive_dof_1lambda            # versione con pesi adattivi di 2.2
    2.7 f2s_smo_adaptive_dof_1lambda            # versione con pesi adattivi di 2.3
    2.8 f2s_adaptive_cov_smo_dof_1lambda        # versione con pesi adattivi di 2.4
                                                                                                              */

arma::umat ovglasso_group_selmat(const arma::uvec& y,
                                 const arma::uvec& x) {
  
  int ny = y.n_elem, nx = x.n_elem;
  arma::umat indi(ny, nx, fill::zeros);

  for (int i=0; i<ny; i++) {
    for (int j=0; j<nx; j++) {
      indi(i, j) = (y(i) == x(j)) ? 1 : 0;
    }
  }
  return indi;
}

Rcpp::List lm_OVGLASSO_admm_active_sets(const arma::vec& coeff,
                                        const double lambda,
                                        const arma::mat& GRmat,
                                        const arma::vec& group_weights,
                                        const arma::vec& var_weights,
                                        const arma::vec& Uvec,
                                        const double err_primal,
                                        const double err_dual,
                                        const double rho,
                                        const double toler_c,
                                        const double toler_d) {
  
  /* Variable delcaration           */
  int p = coeff.n_elem, G = GRmat.n_rows, r_init = 0, r_end = 0;
  double tau = 0.0, delta = 0.0, gnorm = 0.0, dual_cond = 0.0, dual_cond_rhs = 0.0;
  List output;
  
  /* Definition of vectors and matrices     */
  arma::vec vones = arma::ones<arma::vec>(GRmat.n_cols);
  arma::vec nG = GRmat * vones;
  arma::uvec res = arma::zeros<arma::uvec>(G);
  arma::mat coeff_nozeros(G, p, fill::zeros);
  arma::uvec coeff_idx_active(p, fill::zeros);
  
  // Construct Fmat
  arma::sp_mat Fmat = ovglasso_Gmat2Fmat_sparse(GRmat,
                                                var_weights);

  // Compute thresholds
  tau   = toler_c * err_primal / std::sqrt((double)p);
  delta = toler_d * err_dual;

  // Iterate through groups
  r_init = 0;
  r_end  = nG(0) - 1;
  for (int g = 0; g < G; ++g) {
    // 1. Primal condition
    arma::uvec idx = find(GRmat.row(g) == 1);
    gnorm = norm(coeff.elem(idx), 2);

    // 2. Dual condition
    if (g > 0) {
      r_init = r_end + 1;
      r_end  = r_init + nG(g) - 1;
    }
    arma::sp_mat Fmat_g = Fmat.rows(r_init, r_end);
    arma::vec Uvec_b    = Uvec.subvec(r_init, r_end);
    dual_cond           = norm(Fmat_g * coeff + Uvec_b, 2);
    dual_cond_rhs       = delta + lambda * group_weights(g) / rho;

    // Check both conditions
    if ((gnorm >= tau) || (dual_cond >= dual_cond_rhs)) {
      // returns 1 if the group is zero
      res(g) = 1;
      
      // Extract corresponding elements of coeff
      coeff_nozeros.row(g) = GRmat.row(g) % coeff.t();
    }
  }
  arma::uvec idx = find(res.t() * GRmat > 0.0);
  coeff_idx_active.elem(idx).ones();
  
  /* Get output         */
  output["groups_idx_active"] = res;
  output["coeff_idx_active"]  = coeff_idx_active;
  output["coeff_active"]      = coeff_nozeros;
  
  // Return output
  return output;
}

//' @name  fdaSP.internal
//' @keywords internal
// [[Rcpp::export(.lm_dof_LASSO_1lambda)]]
Rcpp::List lm_dof_LASSO_1lambda(const arma::mat& X,
                                const arma::vec& coeff,
                                const double lambda,
                                const arma::vec& Uvec,
                                const double err_primal,
                                const double err_dual,
                                const double rho,
                                const double toler_c,
                                const double toler_d) {
  
  /* Variable declaration           */
  int p = X.n_cols;
  double dof = 0.0, tau = 0.0, delta = 0.0, dual_cond_rhs = 0.0;
  List output;
  
  /* Definition of vectors and matrices     */
  arma::vec gnorms(p, fill::zeros);
  arma::uvec res(p, fill::zeros);
  arma::vec coeff_active(p, fill::zeros);
  arma::vec dual_cond(p, fill::zeros);
  
  // Compute thresholds
  tau    = toler_c * err_primal / std::sqrt((double)p);
  delta  = toler_d * err_dual;
  
  // 1. Primal condition
  gnorms = abs(coeff);
  
  // 2. Dual condition
  dual_cond     = abs(coeff + Uvec);
  dual_cond_rhs = delta + lambda / rho;
  
  // find the active sets
  for (int g = 0; g < p; ++g) {
    // Check both conditions
    if ((gnorms(g) >= tau) || (dual_cond(g) >= dual_cond_rhs)) {
      // returns 1 if the group is not zero
      res(g) = 1;
      
      // Extract corresponding elements of coeff
      coeff_active(g) = coeff(g);
    }
  }
  arma::uvec Aset = find(res == 1);
  dof             = Aset.n_elem;

  /* Get output         */
  output["coeff_active"] = coeff_active;
  output["var_active"]   = res;
  output["dof"]          = dof;
  
  // Return output
  return output;
}

//' @name  fdaSP.internal
//' @keywords internal
// [[Rcpp::export(.lm_dof_OVGLASSO_1lambda)]]
Rcpp::List lm_dof_OVGLASSO_1lambda(const arma::mat& X,
                                   const arma::vec& coeff,
                                   const double lambda,
                                   const arma::mat& GRmat,
                                   const arma::vec& group_weights,
                                   const arma::vec& var_weights,
                                   const arma::vec& Uvec,
                                   const double err_primal,
                                   const double err_dual,
                                   const double rho,
                                   const double toler_c,
                                   const double toler_d) {
  
  /* Variable declaration           */
  int n = 0, p = 0, G = 0, g = 0, ng = 0, r_init = 0, r_end = 0;
  double dof = 0.0, tau = 0.0, delta = 0.0, dual_cond = 0.0, dual_cond_rhs = 0.0;
  List output;
  
  /* Variable definition            */
  n = X.n_rows;
  p = X.n_cols;
  G = GRmat.n_rows;

  /* Definition of vectors and matrices     */
  arma::vec nG(G, fill::zeros);
  arma::uvec res(G, fill::zeros);
  arma::vec gnorms(G, fill::zeros);
  arma::mat B(n, n, fill::zeros);
  arma::mat coeff_active(G, p, fill::zeros);
  arma::uvec Aset_idx(p, fill::zeros);
  
  // compute the number of groups
  nG = arma::sum(GRmat, 1);
  
  // Compute thresholds
  tau   = toler_c * err_primal / std::sqrt((double)p);
  delta = toler_d * err_dual;
  
  // Construct Fmat
  arma::sp_mat Fmat = ovglasso_Gmat2Fmat_sparse(GRmat,
                                                var_weights);

  // find the active sets
  r_init = 0;
  r_end  = nG(0) - 1;
  for (int g = 0; g < G; ++g) {
    // 1. Primal condition
    arma::uvec idx = find(GRmat.row(g) == 1);
    if (!idx.is_empty()) {
      gnorms(g) = norm(coeff.elem(idx), 2);
    }

    // 2. Dual condition
    if (g > 0) {
      r_init = r_end + 1;
      r_end  = r_init + nG(g) - 1;
    }
    arma::sp_mat Fmat_g = Fmat.rows(r_init, r_end);
    arma::vec Uvec_b    = Uvec.subvec(r_init, r_end);
    dual_cond           = norm(Fmat_g * coeff + Uvec_b, 2);
    dual_cond_rhs       = delta + lambda * group_weights(g) / rho;

    // Check both conditions
    if ((gnorms(g) >= tau) || (dual_cond >= dual_cond_rhs)) {
      // returns 1 if the group is not zero
      res(g) = 1;
      
      // Extract corresponding elements of coeff
      coeff_active.row(g) = GRmat.row(g) % coeff.t();
    }
  }
  arma::uvec Aset_G = find(res == 1);
  //arma::uvec Aset   = find(res.t() * GRmat > 0.0);
  
  for (int s=0; s<p; s++) {
    arma::uvec id = find(GRmat.col(s) > 0);
    Aset_idx(s)   = prod(res(id));
  }
  arma::uvec Aset = find(Aset_idx > 0.0);
  arma::uvec Aset_G_idx(G, fill::zeros);
  Aset_G_idx.elem(Aset_G).ones();
  //Aset_idx.elem(Aset).ones();
  
  /* dof are zeros if Aset is empty         */
  if (!Aset.is_empty()) {
    /* compute relevant quantities         */
    arma::mat X_Aset  = X.cols(Aset);
    arma::mat XTX_inv = inv_sympd(X_Aset.t() * X_Aset);
    arma::mat P       = X_Aset * XTX_inv;
    arma::mat H       = P * X_Aset.t();

    // calculate the degrees of freedom
    arma::mat Pi_Aset(Aset.n_elem, Aset.n_elem, fill::zeros);
    for (unsigned int k=0; k<Aset_G.n_elem; k++) {
      g              = Aset_G(k);
      arma::uvec idx = find(GRmat.row(g) == 1);
      ng             = idx.n_elem;
      
      if ((gnorms(g) > 1e-15) && (ng > 0)) {
        arma::mat I_ng    = arma::eye(ng, ng);
        arma::vec coeff_g = coeff.elem(idx);
        //arma::mat Pi_g    = group_weights(g) * (I_ng / gnorms(g) - (coeff_g * coeff_g.t()) / std::pow(gnorms(g), 3));
        
        // aggiunto il 27 Oct 2025
        arma::vec var_weights_g = var_weights.elem(idx);
        arma::mat T_g           = diagmat(var_weights_g);
        arma::mat T2_g          = diagmat(var_weights_g) * diagmat(var_weights_g);
        arma::vec w_g           = T_g * coeff_g;
        arma::vec w2_g          = T2_g * coeff_g;
        arma::mat Pi_g          = group_weights(g) * (T2_g / norm(w_g, 2) - (w2_g * w2_g.t()) / std::pow(norm(w_g, 2), 3));

        // build selection matrix
        arma::umat Smat = ovglasso_group_selmat(Aset, idx);
          
        // compute the denominator
        Pi_Aset += Smat * Pi_g * Smat.t();
      }
    }
    
    /* compute dof         */
    B   = (double)n * lambda * P * Pi_Aset * P.t() + arma::eye(n, n);
    //dof = trace(inv_sympd(B) * H);
    dof = trace(inv(B) * H);
  } else {
    arma::uvec Aset_G_idx(G, fill::zeros);
    dof = 0.0;
  }
  
  /* Get output         */
  output["groups_active"] = Aset_G_idx;
  output["coeff_active"]  = coeff_active;
  output["var_active"]    = Aset_idx;
  output["dof"]           = dof;
  
  // Return output
  return output;
}

//' @name  fdaSP.internal
//' @keywords internal
// [[Rcpp::export(.lm_cov_dof_OVGLASSO_1lambda)]]
Rcpp::List lm_cov_dof_OVGLASSO_1lambda(const arma::mat& X,
                                       const arma::mat& Z,
                                       const arma::vec& coeff_X,
                                       const double lambda,
                                       const arma::mat& GRmat,
                                       const arma::vec& group_weights,
                                       const arma::vec& var_weights,
                                       const arma::vec& Uvec,
                                       const double err_primal,
                                       const double err_dual,
                                       const double rho,
                                       const double toler_c,
                                       const double toler_d) {
  
  /* Variable declaration           */
  int n = 0, p = 0, q = 0, G = 0, g = 0, ng = 0, r_init = 0, r_end = 0;
  double dof = 0.0, tau = 0.0, delta = 0.0, dual_cond = 0.0, dual_cond_rhs = 0.0;
  List output;
    
  /* Variable definition                    */
  n = X.n_rows;
  p = X.n_cols;
  q = Z.n_cols;
  G = GRmat.n_rows;

  /* Definition of vectors and matrices     */
  arma::vec nG(G, fill::zeros);
  arma::uvec res(G, fill::zeros);
  arma::vec gnorms(G, fill::zeros);
  arma::mat coeff_active(G, p, fill::zeros);
  arma::mat DTD(p, p, fill::zeros);
  arma::uvec Aset_idx(p, fill::zeros);

  // compute the number of groups
  nG = arma::sum(GRmat, 1);
  
  // Compute thresholds
  tau   = toler_c * err_primal / std::sqrt((double)p);
  delta = toler_d * err_dual;
  
  // Construct Fmat
  arma::sp_mat Fmat = ovglasso_Gmat2Fmat_sparse(GRmat,
                                                var_weights);
  
  // find the active sets
  r_init = 0;
  r_end  = nG(0) - 1;
  for (int g = 0; g < G; ++g) {
    // 1. Primal condition
    arma::uvec idx = find(GRmat.row(g) == 1);
    if (!idx.is_empty()) {
      gnorms(g) = norm(coeff_X.elem(idx), 2);
    }

    // 2. Dual condition
    if (g > 0) {
      r_init = r_end + 1;
      r_end  = r_init + nG(g) - 1;
    }
    arma::sp_mat Fmat_g = Fmat.rows(r_init, r_end);
    arma::vec Uvec_b    = Uvec.subvec(r_init, r_end);
    dual_cond           = norm(Fmat_g * coeff_X + Uvec_b, 2);
    dual_cond_rhs       = delta + lambda * group_weights(g) / rho;
    
    // Check both conditions
    if ((gnorms(g) >= tau) || (dual_cond >= dual_cond_rhs)) {
      // returns 1 if the group is not zero
      res(g) = 1;
      
      // Extract corresponding elements of coeff
      coeff_active.row(g) = GRmat.row(g) % coeff_X.t();
    }
  }
  arma::uvec Aset_G = find(res == 1);
  //arma::uvec Aset   = find(res.t() * GRmat > 0.0);
  
  for (int s=0; s<p; s++) {
    arma::uvec id = find(GRmat.col(s) > 0);
    Aset_idx(s)   = prod(res(id));
  }
  arma::uvec Aset = find(Aset_idx > 0.0);
  arma::uvec Aset_G_idx(G, fill::zeros);
  Aset_G_idx.elem(Aset_G).ones();
  //Aset_idx.elem(Aset).ones();

  /* dof are zeros if Aset is empty         */
  if (!Aset.is_empty()) {
    
    /* compute relevant quantities         */
    arma::mat Pi(Aset.n_elem + q, Aset.n_elem + q, fill::zeros);
    arma::mat Q_inv(Aset.n_elem + q, Aset.n_elem + q, fill::zeros);
    arma::mat B(Aset.n_elem + q, Aset.n_elem + q, fill::zeros);
    
    // Construct the matrix of active covariates
    arma::mat X_Aset = X.cols(Aset);
        
    // fill the matrix Q
    Q_inv.submat(0, 0, Aset.n_elem-1, Aset.n_elem-1)                         = X_Aset.t() * X_Aset;
    Q_inv.submat(0, Aset.n_elem, Aset.n_elem-1, Aset.n_elem+q-1)             = X_Aset.t() * Z;
    Q_inv.submat(Aset.n_elem, 0, Aset.n_elem+q-1, Aset.n_elem-1)             = (Q_inv.submat(0, Aset.n_elem, Aset.n_elem-1, Aset.n_elem+q-1)).t();
    Q_inv.submat(Aset.n_elem, Aset.n_elem, Aset.n_elem+q-1, Aset.n_elem+q-1) = Z.t() * Z;
    
    // calculate the degrees of freedom
    arma::mat Pi_Aset(Aset.n_elem, Aset.n_elem, fill::zeros);
    for (unsigned int k=0; k<Aset_G.n_elem; k++) {
      g                 = Aset_G(k);
      arma::uvec idx    = find(GRmat.row(g) == 1);
      ng                = idx.n_elem;
      arma::mat I_ng    = arma::eye(ng, ng);
      arma::vec coeff_g = coeff_X.elem(idx);
      //arma::mat Pi_g    = group_weights(g) * (I_ng / gnorms(g) - (coeff_g * coeff_g.t()) / std::pow(gnorms(g), 3));
      
      // aggiunto il 27 Oct 2025
      arma::vec var_weights_g = var_weights.elem(idx);
      arma::mat T_g           = diagmat(var_weights_g);
      arma::mat T2_g          = diagmat(var_weights_g) * diagmat(var_weights_g);
      arma::vec w_g           = T_g * coeff_g;
      arma::vec w2_g          = T2_g * coeff_g;
      arma::mat Pi_g          = group_weights(g) * (T2_g / norm(w_g, 2) - (w2_g * w2_g.t()) / std::pow(norm(w_g, 2), 3));

      // build selection matrix
      arma::umat Smat = ovglasso_group_selmat(Aset, idx);
        
      // compute the denominator
      Pi_Aset += Smat * Pi_g * Smat.t();
    }
    
    /* fill the matrix Pi_C             */
    Pi.submat(0, 0, Aset.n_elem-1, Aset.n_elem-1) = Pi_Aset;
    
    /* compute the matrix B         */
    B = Q_inv + (double)n * (lambda * Pi_Aset);
        
    /* compute dof         */
    //dof = trace(inv_sympd(B) * Q_inv);
    dof = trace(inv(B) * Q_inv);
  } else {
    arma::uvec Aset_G_idx(G, fill::zeros);
    dof = (double)q;
  }

  /* Get output         */
  output["groups_active"] = Aset_G_idx;
  output["coeff_active"]  = coeff_active;
  output["var_active"]    = Aset_idx;
  output["dof"]           = dof;
  
  // Return output
  return output;
}

//' @name  fdaSP.internal
//' @keywords internal
// [[Rcpp::export(.lm_adaptive_dof_OVGLASSO_1lambda)]]
Rcpp::List lm_adaptive_dof_OVGLASSO_1lambda(const arma::mat& X,
                                            const arma::vec& coeff,
                                            const arma::vec& coeff_LS,
                                            const double lambda,
                                            const arma::mat& GRmat,
                                            const arma::vec& group_weights,
                                            const arma::vec& var_weights,
                                            const arma::vec& Uvec,
                                            const double err_primal,
                                            const double err_dual,
                                            const double rho,
                                            const double toler_c,
                                            const double toler_d) {
  
  /* Variable declaration           */
  int n = 0, p = 0, G = 0, g = 0, ng = 0, r_init = 0, r_end = 0;
  double dof = 0.0, tau = 0.0, delta = 0.0, dual_cond = 0.0, dual_cond_rhs = 0.0;
  List output;
    
  /* Variable definition                    */
  n = X.n_rows;
  p = X.n_cols;
  G = GRmat.n_rows;

  /* Definition of vectors and matrices     */
  arma::vec nG(G, fill::zeros);
  arma::uvec res(G, fill::zeros);
  arma::vec gnorms(G, fill::zeros);
  arma::mat coeff_active(G, p, fill::zeros);
  arma::uvec Aset_idx(p, fill::zeros);
  
  // compute the number of groups
  nG = arma::sum(GRmat, 1);
  
  // Compute thresholds
  tau   = toler_c * err_primal / std::sqrt((double)p);
  delta = toler_d * err_dual;
  
  // Construct Fmat
  arma::sp_mat Fmat = ovglasso_Gmat2Fmat_sparse(GRmat,
                                                var_weights);

  // find the active sets
  r_init = 0;
  r_end  = nG(0) - 1;
  for (int g = 0; g < G; ++g) {
    // 1. Primal condition
    arma::uvec idx = find(GRmat.row(g) == 1);
    if (!idx.is_empty()) {
      gnorms(g) = norm(coeff.elem(idx), 2);
    }

    // 2. Dual condition
    if (g > 0) {
      r_init = r_end + 1;
      r_end  = r_init + nG(g) - 1;
    }
    arma::sp_mat Fmat_g = Fmat.rows(r_init, r_end);
    arma::vec Uvec_b    = Uvec.subvec(r_init, r_end);
    dual_cond           = norm(Fmat_g * coeff + Uvec_b, 2);
    dual_cond_rhs       = delta + lambda * group_weights(g) / rho;
    
    // Check both conditions
    if ((gnorms(g) >= tau) || (dual_cond >= dual_cond_rhs)) {
    //if (!((gnorms(g) < tau) && (dual_cond < dual_cond_rhs))) {
      // returns 1 if the group is not zero
      res(g) = 1;
      
      // Extract corresponding elements of coeff
      coeff_active.row(g) = GRmat.row(g) % coeff.t();
    }
  }
  arma::uvec Aset_G = find(res == 1);
  //arma::uvec Aset   = find(res.t() * GRmat > 0.0);
  
  for (int s=0; s<p; s++) {
    arma::uvec id = find(GRmat.col(s) > 0);
    Aset_idx(s)   = prod(res(id));
  }
  arma::uvec Aset = find(Aset_idx > 0.0);
  arma::uvec Aset_G_idx(G, fill::zeros);
  Aset_G_idx.elem(Aset_G).ones();
  //Aset_idx.elem(Aset).ones();
  
  /* dof are zeros if Aset is empty         */
  if (!Aset.is_empty()) {
    
    /* compute relevant quantities         */
    arma::mat Q_inv(Aset.n_elem, Aset.n_elem, fill::zeros);
    arma::mat B(Aset.n_elem, Aset.n_elem, fill::zeros);
    
    // Construct the matrix of active covariates
    arma::mat X_Aset = X.cols(Aset);
        
    // fill the matrix Q
    Q_inv = X_Aset.t() * X_Aset;
      
    // calculate the degrees of freedom
    arma::mat Pi_Aset(Aset.n_elem, Aset.n_elem, fill::zeros);
    arma::mat Phi_Aset(Aset.n_elem, Aset.n_elem, fill::zeros);
    for (unsigned int k=0; k<Aset_G.n_elem; k++) {
      g                 = Aset_G(k);
      arma::uvec idx    = find(GRmat.row(g) == 1);
      ng                = idx.n_elem;
      arma::mat I_ng    = arma::eye(ng, ng);
      arma::vec coeff_g = coeff.elem(idx);
      //arma::mat Pi_g    = group_weights(g) * (I_ng / gnorms(g) - (coeff_g * coeff_g.t()) / std::pow(gnorms(g), 3));
      
      // aggiunto il 27 Oct 2025
      arma::vec var_weights_g = var_weights.elem(idx);
      arma::mat T_g           = diagmat(var_weights_g);
      arma::mat T2_g          = diagmat(var_weights_g) * diagmat(var_weights_g);
      arma::vec w_g           = T_g * coeff_g;
      arma::vec w2_g          = T2_g * coeff_g;
      arma::mat Pi_g          = group_weights(g) * (T2_g / norm(w_g, 2) - (w2_g * w2_g.t()) / std::pow(norm(w_g, 2), 3));

      // build selection matrix
      arma::umat Smat = ovglasso_group_selmat(Aset, idx);
        
      // compute the denominator
      Pi_Aset += Smat * Pi_g * Smat.t();
      
      // compute the derivative of the adaptive part
      arma::vec coeff_LS_g  = coeff_LS.elem(idx);
      arma::mat Phi_g       = coeff_g * coeff_LS_g.t();
      Phi_g                /= -std::pow(norm(coeff_LS_g, 2), 3) * gnorms(g);
      Phi_Aset             += Smat * Phi_g * Smat.t();
    }
    
    /* compute the matrix B         */
    B = Q_inv + (double)n * lambda * Pi_Aset;
    
    /* compute dof         */
    //dof = trace(inv_sympd(B) * (Q_inv - lambda * Phi_Aset));
    dof = trace(inv(B) * (Q_inv - lambda * Phi_Aset));
  } else {
    arma::uvec Aset_G_idx(G, fill::zeros);
    dof = 0.0;
  }

  /* Get output         */
  output["groups_active"] = Aset_G_idx;
  output["coeff_active"]  = coeff_active;
  output["var_active"]    = Aset_idx;
  output["dof"]           = dof;
  
  // Return output
  return output;
}

//' @name  fdaSP.internal
//' @keywords internal
// [[Rcpp::export(.lm_cov_adaptive_dof_OVGLASSO_1lamba)]]
Rcpp::List lm_cov_adaptive_dof_OVGLASSO_1lamba(const arma::mat& X,
                                               const arma::mat& Z,
                                               const arma::vec& coeff_X,
                                               const arma::vec& coeff_X_LS,
                                               const double lambda,
                                               const arma::mat& GRmat,
                                               const arma::vec& group_weights,
                                               const arma::vec& var_weights,
                                               const arma::vec& Uvec,
                                               const double err_primal,
                                               const double err_dual,
                                               const double rho,
                                               const double toler_c,
                                               const double toler_d) {
          
  /* Variable declaration           */
  int n = 0, p = 0, q = 0, G = 0, g = 0, ng = 0, r_init = 0, r_end = 0;
  double dof = 0.0, tau = 0.0, delta = 0.0, dual_cond = 0.0, dual_cond_rhs = 0.0;
  List output;
    
  /* Variable definition                    */
  n = X.n_rows;
  p = X.n_cols;
  q = Z.n_cols;
  G = GRmat.n_rows;

  /* Definition of vectors and matrices     */
  arma::vec nG(G, fill::zeros);
  arma::uvec res(G, fill::zeros);
  arma::vec gnorms(G, fill::zeros);
  arma::mat coeff_active(G, p, fill::zeros);
  arma::mat DTD(p, p, fill::zeros);
  arma::uvec Aset_idx(p, fill::zeros);

  // compute the number of groups
  nG = arma::sum(GRmat, 1);
  
  // Compute thresholds
  tau   = toler_c * err_primal / std::sqrt((double)p);
  delta = toler_d * err_dual;
  
  // Construct Fmat
  arma::sp_mat Fmat = ovglasso_Gmat2Fmat_sparse(GRmat,
                                                var_weights);
  
  // find the active sets
  r_init = 0;
  r_end  = nG(0) - 1;
  for (int g = 0; g < G; ++g) {
    // 1. Primal condition
    arma::uvec idx = find(GRmat.row(g) == 1);
    if (!idx.is_empty()) {
      gnorms(g) = norm(coeff_X.elem(idx), 2);
    }

    // 2. Dual condition
    if (g > 0) {
      r_init = r_end + 1;
      r_end  = r_init + nG(g) - 1;
    }
    arma::sp_mat Fmat_g = Fmat.rows(r_init, r_end);
    arma::vec Uvec_b    = Uvec.subvec(r_init, r_end);
    dual_cond           = norm(Fmat_g * coeff_X + Uvec_b, 2);
    dual_cond_rhs       = delta + lambda * group_weights(g) / rho;
    
    // Check both conditions
    if ((gnorms(g) >= tau) || (dual_cond >= dual_cond_rhs)) {
      // returns 1 if the group is not zero
      res(g) = 1;
      
      // Extract corresponding elements of coeff
      coeff_active.row(g) = GRmat.row(g) % coeff_X.t();
    }
  }
  arma::uvec Aset_G = find(res == 1);
  //arma::uvec Aset   = find(res.t() * GRmat > 0.0);
  
  for (int s=0; s<p; s++) {
    arma::uvec id = find(GRmat.col(s) > 0);
    Aset_idx(s)   = prod(res(id));
  }
  arma::uvec Aset = find(Aset_idx > 0.0);
  arma::uvec Aset_G_idx(G, fill::zeros);
  Aset_G_idx.elem(Aset_G).ones();
  //Aset_idx.elem(Aset).ones();
  
  /* dof are zeros if Aset is empty         */
  if (!Aset.is_empty()) {
    
    /* compute relevant quantities         */
    arma::mat Pi(Aset.n_elem + q, Aset.n_elem + q, fill::zeros);
    arma::mat Phi(Aset.n_elem + q, Aset.n_elem + q, fill::zeros);
    arma::mat Q_inv(Aset.n_elem + q, Aset.n_elem + q, fill::zeros);
    arma::mat B(Aset.n_elem + q, Aset.n_elem + q, fill::zeros);
    
    // Construct the matrix of active covariates
    arma::mat X_Aset = X.cols(Aset);
        
    // fill the matrix Q
    Q_inv.submat(0, 0, Aset.n_elem-1, Aset.n_elem-1)                         = X_Aset.t() * X_Aset;
    Q_inv.submat(0, Aset.n_elem, Aset.n_elem-1, Aset.n_elem+q-1)             = X_Aset.t() * Z;
    Q_inv.submat(Aset.n_elem, 0, Aset.n_elem+q-1, Aset.n_elem-1)             = (Q_inv.submat(0, Aset.n_elem, Aset.n_elem-1, Aset.n_elem+q-1)).t();
    Q_inv.submat(Aset.n_elem, Aset.n_elem, Aset.n_elem+q-1, Aset.n_elem+q-1) = Z.t() * Z;
    
    // calculate the degrees of freedom
    arma::mat Pi_Aset(Aset.n_elem, Aset.n_elem, fill::zeros);
    arma::mat Phi_Aset(Aset.n_elem, Aset.n_elem, fill::zeros);
    for (unsigned int k=0; k<Aset_G.n_elem; k++) {
      g                 = Aset_G(k);
      arma::uvec idx    = find(GRmat.row(g) == 1);
      ng                = idx.n_elem;
      arma::mat I_ng    = arma::eye(ng, ng);
      arma::vec coeff_g = coeff_X.elem(idx);
      //arma::mat Pi_g    = group_weights(g) * (I_ng / gnorms(g) - (coeff_g * coeff_g.t()) / std::pow(gnorms(g), 3));
      
      // aggiunto il 27 Oct 2025
      arma::vec var_weights_g = var_weights.elem(idx);
      arma::mat T_g           = diagmat(var_weights_g);
      arma::mat T2_g          = diagmat(var_weights_g) * diagmat(var_weights_g);
      arma::vec w_g           = T_g * coeff_g;
      arma::vec w2_g          = T2_g * coeff_g;
      arma::mat Pi_g          = group_weights(g) * (T2_g / norm(w_g, 2) - (w2_g * w2_g.t()) / std::pow(norm(w_g, 2), 3));

      // build selection matrix
      arma::umat Smat = ovglasso_group_selmat(Aset, idx);
        
      // compute the denominator
      Pi_Aset += Smat * Pi_g * Smat.t();
      
      // compute the derivative of the adaptive part
      arma::vec coeff_X_LS_g  = coeff_X_LS.elem(idx);
      arma::mat Phi_g         = coeff_g * coeff_X_LS_g.t();
      Phi_g                  /= -std::pow(norm(coeff_X_LS_g, 2), 3) * gnorms(g);
      Phi_Aset               += Smat * Phi_g * Smat.t();
    }
    
    /* fill the matrix Pi_C             */
    Pi.submat(0, 0, Aset.n_elem-1, Aset.n_elem-1)  = Pi_Aset;
    Phi.submat(0, 0, Aset.n_elem-1, Aset.n_elem-1) = Phi_Aset;
    
    /* compute the matrix B         */
    B = Q_inv + (double)n * (lambda * Pi);
        
    /* compute dof         */
    //dof = trace(inv_sympd(B) * (Q_inv - lambda * Phi));
    dof = trace(inv(B) * (Q_inv - lambda * Phi));
  } else {
    arma::uvec Aset_G_idx(G, fill::zeros);
    dof = (double)q;
  }

  /* Get output         */
  output["groups_active"] = Aset_G_idx;
  output["coeff_active"]  = coeff_active;
  output["var_active"]    = Aset_idx;
  output["dof"]           = dof;
  
  // Return output
  return output;
}

//' @name  fdaSP.internal
//' @keywords internal
// [[Rcpp::export(.f2s_dof_1lambda)]]
Rcpp::List f2s_dof_1lambda(const arma::mat& W,
                           const arma::vec& coeff,
                           const double lambda,
                           const arma::mat& GRmat,
                           const arma::vec& group_weights,
                           const arma::vec& var_weights,
                           const arma::vec& Uvec,
                           const double err_primal,
                           const double err_dual,
                           const double rho,
                           const double toler_c,
                           const double toler_d) {
  
  /* Variable declaration           */
  int n = 0, p = 0, G = 0, g = 0, ng = 0, r_init = 0, r_end = 0;
  double dof = 0.0, tau = 0.0, delta = 0.0, dual_cond = 0.0, dual_cond_rhs = 0.0;
  List output;
    
  /* Variable definition                    */
  n = W.n_rows;
  p = W.n_cols;
  G = GRmat.n_rows;

  /* Definition of vectors and matrices     */
  arma::vec nG(G, fill::zeros);
  arma::uvec res(G, fill::zeros);
  arma::vec gnorms(G, fill::zeros);
  arma::mat coeff_active(G, p, fill::zeros);
  arma::uvec Aset_idx(p, fill::zeros);
  
  // compute the number of groups
  nG = arma::sum(GRmat, 1);
  
  // Compute thresholds
  tau   = toler_c * err_primal / std::sqrt((double)p);
  delta = toler_d * err_dual;
  
  // Construct Fmat
  arma::sp_mat Fmat = ovglasso_Gmat2Fmat_sparse(GRmat,
                                                var_weights);

  // find the active sets
  r_init = 0;
  r_end  = nG(0) - 1;
  for (int g = 0; g < G; ++g) {
    // 1. Primal condition
    arma::uvec idx = find(GRmat.row(g) == 1);
    if (!idx.is_empty()) {
      gnorms(g) = norm(coeff.elem(idx), 2);
    }

    // 2. Dual condition
    if (g > 0) {
      r_init = r_end + 1;
      r_end  = r_init + nG(g) - 1;
    }
    arma::sp_mat Fmat_g = Fmat.rows(r_init, r_end);
    arma::vec Uvec_b    = Uvec.subvec(r_init, r_end);
    dual_cond           = norm(Fmat_g * coeff + Uvec_b, 2);
    dual_cond_rhs       = delta + lambda * group_weights(g) / rho;
    
    // Check both conditions
    if ((gnorms(g) >= tau) || (dual_cond >= dual_cond_rhs)) {
    //if (!((gnorms(g) < tau) && (dual_cond < dual_cond_rhs))) {
      // returns 1 if the group is not zero
      res(g) = 1;
      
      // Extract corresponding elements of coeff
      coeff_active.row(g) = GRmat.row(g) % coeff.t();
    }
  }
  arma::uvec Aset_G = find(res == 1);
  //arma::uvec Aset   = find(res.t() * GRmat > 0.0);
  
  for (int s=0; s<p; s++) {
    arma::uvec id = find(GRmat.col(s) > 0);
    Aset_idx(s)   = prod(res(id));
  }
  arma::uvec Aset = find(Aset_idx > 0.0);
  arma::uvec Aset_G_idx(G, fill::zeros);
  Aset_G_idx.elem(Aset_G).ones();
  //Aset_idx.elem(Aset).ones();
    
  /* dof are zeros if Aset is empty         */
  if (!Aset.is_empty()) {
    
    /* compute relevant quantities         */
    arma::mat Q_inv(Aset.n_elem, Aset.n_elem, fill::zeros);
    arma::mat B(Aset.n_elem, Aset.n_elem, fill::zeros);
    
    // Construct the matrix of active covariates
    arma::mat W_Aset = W.cols(Aset);
        
    // fill the matrix Q
    Q_inv = W_Aset.t() * W_Aset;
      
    // calculate the degrees of freedom
    arma::mat Pi_Aset(Aset.n_elem, Aset.n_elem, fill::zeros);
    for (unsigned int k=0; k<Aset_G.n_elem; k++) {
      g                       = Aset_G(k);
      arma::uvec idx          = find(GRmat.row(g) == 1);
      ng                      = idx.n_elem;
      arma::mat I_ng          = arma::eye(ng, ng);
      arma::vec coeff_g       = coeff.elem(idx);
      //arma::mat Pi_g          = group_weights(g) * (I_ng / gnorms(g) - (coeff_g * coeff_g.t()) / std::pow(gnorms(g), 3));
      
      // aggiunto il 27 Oct 2025
      arma::vec var_weights_g = var_weights.elem(idx);
      arma::mat T_g           = diagmat(var_weights_g);
      arma::mat T2_g          = diagmat(var_weights_g) * diagmat(var_weights_g);
      arma::vec w_g           = T_g * coeff_g;
      arma::vec w2_g          = T2_g * coeff_g;
      arma::mat Pi_g          = group_weights(g) * (T2_g / norm(w_g, 2) - (w2_g * w2_g.t()) / std::pow(norm(w_g, 2), 3));

      // build selection matrix
      arma::umat Smat = ovglasso_group_selmat(Aset, idx);
        
      // compute the denominator
      Pi_Aset += Smat * Pi_g * Smat.t();
    }
    
    /* compute the matrix B         */
    B = Q_inv + (double)n * lambda * Pi_Aset;
        
    /* compute dof         */
    //dof = trace(inv_sympd(B) * Q_inv);
    dof = trace(inv(B) * Q_inv);
  } else {
    arma::uvec Aset_G_idx(G, fill::zeros);
    dof = 0.0;
  }

  /* Get output         */
  output["groups_active"] = Aset_G_idx;
  output["coeff_active"]  = coeff_active;
  output["var_active"]    = Aset_idx;
  output["dof"]           = dof;
  
  // Return output
  return output;
}

//' @name  fdaSP.internal
//' @keywords internal
// [[Rcpp::export(.f2s_cov_dof_1lambda)]]
Rcpp::List f2s_cov_dof_1lambda(const arma::mat& W,
                               const arma::mat& Z,
                               const arma::vec& coeff_W,
                               const double lambda,
                               const arma::mat& GRmat,
                               const arma::vec& group_weights,
                               const arma::vec& var_weights,
                               const arma::vec& Uvec,
                               const double err_primal,
                               const double err_dual,
                               const double rho,
                               const double toler_c,
                               const double toler_d) {
  
  /* Variable declaration           */
  int n = 0, p = 0, q = 0, G = 0, g = 0, ng = 0, r_init = 0, r_end = 0;
  double dof = 0.0, tau = 0.0, delta = 0.0, dual_cond = 0.0, dual_cond_rhs = 0.0;
  List output;
    
  /* Variable definition                    */
  n = W.n_rows;
  p = W.n_cols;
  q = Z.n_cols;
  G = GRmat.n_rows;

  /* Definition of vectors and matrices     */
  arma::vec nG(G, fill::zeros);
  arma::uvec res(G, fill::zeros);
  arma::vec gnorms(G, fill::zeros);
  arma::mat coeff_active(G, p, fill::zeros);
  arma::uvec Aset_idx(p, fill::zeros);
  
  // compute the number of groups
  nG = arma::sum(GRmat, 1);
  
  // Compute thresholds
  tau   = toler_c * err_primal / std::sqrt((double)p);
  delta = toler_d * err_dual;
  
  // Construct Fmat
  arma::sp_mat Fmat = ovglasso_Gmat2Fmat_sparse(GRmat,
                                                var_weights);
  
  // find the active sets
  r_init = 0;
  r_end  = nG(0) - 1;
  for (int g = 0; g < G; ++g) {
    // 1. Primal condition
    arma::uvec idx = find(GRmat.row(g) == 1);
    if (!idx.is_empty()) {
      gnorms(g) = norm(coeff_W.elem(idx), 2);
    }

    // 2. Dual condition
    if (g > 0) {
      r_init = r_end + 1;
      r_end  = r_init + nG(g) - 1;
    }
    arma::sp_mat Fmat_g = Fmat.rows(r_init, r_end);
    arma::vec Uvec_b    = Uvec.subvec(r_init, r_end);
    dual_cond           = norm(Fmat_g * coeff_W + Uvec_b, 2);
    dual_cond_rhs       = delta + lambda * group_weights(g) / rho;
    
    // Check both conditions
    if ((gnorms(g) >= tau) || (dual_cond >= dual_cond_rhs)) {
      // returns 1 if the group is not zero
      res(g) = 1;
      
      // Extract corresponding elements of coeff
      coeff_active.row(g) = GRmat.row(g) % coeff_W.t();
    }
  }
  arma::uvec Aset_G = find(res == 1);
  //arma::uvec Aset   = find(res.t() * GRmat > 0.0);
  for (int s=0; s<p; s++) {
    arma::uvec id = find(GRmat.col(s) > 0);
    Aset_idx(s)   = prod(res(id));
  }
  arma::uvec Aset = find(Aset_idx > 0.0);
  arma::uvec Aset_G_idx(G, fill::zeros);
  Aset_G_idx.elem(Aset_G).ones();
  //Aset_idx.elem(Aset).ones();
  
  /* dof are zeros if Aset is empty         */
  if (!Aset.is_empty()) {
    
    /* compute relevant quantities         */
    arma::mat Pi(Aset.n_elem + q, Aset.n_elem + q, fill::zeros);
    arma::mat Q_inv(Aset.n_elem + q, Aset.n_elem + q, fill::zeros);
    arma::mat B(Aset.n_elem + q, Aset.n_elem + q, fill::zeros);
    
    // Construct the matrix of active covariates
    arma::mat W_Aset = W.cols(Aset);
        
    // fill the matrix Q
    Q_inv.submat(0, 0, Aset.n_elem-1, Aset.n_elem-1)                         = W_Aset.t() * W_Aset;
    Q_inv.submat(0, Aset.n_elem, Aset.n_elem-1, Aset.n_elem+q-1)             = W_Aset.t() * Z;
    Q_inv.submat(Aset.n_elem, 0, Aset.n_elem+q-1, Aset.n_elem-1)             = (Q_inv.submat(0, Aset.n_elem, Aset.n_elem-1, Aset.n_elem+q-1)).t();
    Q_inv.submat(Aset.n_elem, Aset.n_elem, Aset.n_elem+q-1, Aset.n_elem+q-1) = Z.t() * Z;
    
    // calculate the degrees of freedom
    arma::mat Pi_Aset(Aset.n_elem, Aset.n_elem, fill::zeros);
    for (unsigned int k=0; k<Aset_G.n_elem; k++) {
      g                 = Aset_G(k);
      arma::uvec idx    = find(GRmat.row(g) == 1);
      ng                = idx.n_elem;
      arma::mat I_ng    = arma::eye(ng, ng);
      arma::vec coeff_g = coeff_W.elem(idx);
      //arma::mat Pi_g    = group_weights(g) * (I_ng / gnorms(g) - (coeff_g * coeff_g.t()) / std::pow(gnorms(g), 3));
      
      // aggiunto il 27 Oct 2025
      arma::vec var_weights_g = var_weights.elem(idx);
      arma::mat T_g           = diagmat(var_weights_g);
      arma::mat T2_g          = diagmat(var_weights_g) * diagmat(var_weights_g);
      arma::vec w_g           = T_g * coeff_g;
      arma::vec w2_g          = T2_g * coeff_g;
      arma::mat Pi_g          = group_weights(g) * (T2_g / norm(w_g, 2) - (w2_g * w2_g.t()) / std::pow(norm(w_g, 2), 3));

      // build selection matrix
      arma::umat Smat = ovglasso_group_selmat(Aset, idx);
        
      // compute the denominator
      Pi_Aset += Smat * Pi_g * Smat.t();
    }
    
    /* fill the matrix Pi_C             */
    Pi.submat(0, 0, Aset.n_elem-1, Aset.n_elem-1) = Pi_Aset;
    
    /* compute the matrix B         */
    B = Q_inv + (double)n * (lambda * Pi);
    
    /* compute dof         */
    //dof = trace(inv_sympd(B) * Q_inv);
    dof = trace(inv(B) * Q_inv);
  } else {
    arma::uvec Aset_G_idx(G, fill::zeros);
    dof = (double)q;
  }
  
  /* Get output         */
  output["groups_active"] = Aset_G_idx;
  output["coeff_active"]  = coeff_active;
  output["var_active"]    = Aset_idx;
  output["dof"]           = dof;
  
  // Return output
  return output;
}

//' @name  fdaSP.internal
//' @keywords internal
// [[Rcpp::export(.f2s_smo_dof_1lambda)]]
Rcpp::List f2s_smo_dof_1lambda(const arma::mat& W,
                               const arma::vec& coeff,
                               const double lambda,
                               const double lambda2,
                               const int diff_order,
                               const arma::mat& GRmat,
                               const arma::vec& group_weights,
                               const arma::vec& var_weights,
                               const arma::vec& Uvec,
                               const double err_primal,
                               const double err_dual,
                               const double rho,
                               const double toler_c,
                               const double toler_d) {
  
  /* Variable declaration           */
  int n = 0, p = 0, G = 0, g = 0, ng = 0, r_init = 0, r_end = 0;
  double dof = 0.0, tau = 0.0, delta = 0.0, dual_cond = 0.0, dual_cond_rhs = 0.0;
  List output;
    
  /* Variable definition                    */
  n = W.n_rows;
  p = W.n_cols;
  G = GRmat.n_rows;

  /* Definition of vectors and matrices     */
  arma::vec nG(G, fill::zeros);
  arma::uvec res(G, fill::zeros);
  arma::vec gnorms(G, fill::zeros);
  arma::mat coeff_active(G, p, fill::zeros);
  arma::mat DTD(p, p, fill::zeros);
  arma::uvec Aset_idx(p, fill::zeros);

  // compute the number of groups
  nG = arma::sum(GRmat, 1);
  
  // Compute thresholds
  tau   = toler_c * err_primal / std::sqrt((double)p);
  delta = toler_d * err_dual;
  
  // Construct Fmat
  arma::sp_mat Fmat = ovglasso_Gmat2Fmat_sparse(GRmat,
                                                var_weights);
  
  // Construct the penalty matrix
  DTD = forward_diff_penalty_matrix(coeff.n_elem, 1, diff_order);
    
  // find the active sets
  r_init = 0;
  r_end  = nG(0) - 1;
  for (int g = 0; g < G; ++g) {
    // 1. Primal condition
    arma::uvec idx = find(GRmat.row(g) == 1);
    if (!idx.is_empty()) {
      gnorms(g) = norm(coeff.elem(idx), 2);
    }

    // 2. Dual condition
    if (g > 0) {
      r_init = r_end + 1;
      r_end  = r_init + nG(g) - 1;
    }
    arma::sp_mat Fmat_g = Fmat.rows(r_init, r_end);
    arma::vec Uvec_b    = Uvec.subvec(r_init, r_end);
    dual_cond           = norm(Fmat_g * coeff + Uvec_b, 2);
    dual_cond_rhs       = delta + lambda * group_weights(g) / rho;
    
    // Check both conditions
    if ((gnorms(g) >= tau) || (dual_cond >= dual_cond_rhs)) {
      // returns 1 if the group is not zero
      res(g) = 1;
      
      // Extract corresponding elements of coeff
      coeff_active.row(g) = GRmat.row(g) % coeff.t();
    }
  }
  arma::uvec Aset_G = find(res == 1);
  //arma::uvec Aset   = find(res.t() * GRmat > 0.0);
  for (int s=0; s<p; s++) {
    arma::uvec id = find(GRmat.col(s) > 0);
    Aset_idx(s)   = prod(res(id));
  }
  arma::uvec Aset = find(Aset_idx > 0.0);
  arma::uvec Aset_G_idx(G, fill::zeros);
  Aset_G_idx.elem(Aset_G).ones();
  //Aset_idx.elem(Aset).ones();

  /* dof are zeros if Aset is empty         */
  if (!Aset.is_empty()) {
    
    /* compute relevant quantities         */
    arma::mat Q_inv(Aset.n_elem, Aset.n_elem, fill::zeros);
    arma::mat B(Aset.n_elem, Aset.n_elem, fill::zeros);
    
    // Construct the matrix of active penalty
    arma::mat DTD_Aset = DTD.submat(Aset, Aset);
    
    // Construct the matrix of active covariates
    arma::mat W_Aset = W.cols(Aset);
        
    // fill the matrix Q
    //Q_inv = inv_sympd(W_Aset.t() * W_Aset);
    Q_inv = W_Aset.t() * W_Aset;
      
    // calculate the degrees of freedom
    arma::mat Pi_Aset(Aset.n_elem, Aset.n_elem, fill::zeros);
    for (unsigned int k=0; k<Aset_G.n_elem; k++) {
      g                 = Aset_G(k);
      arma::uvec idx    = find(GRmat.row(g) == 1);
      ng                = idx.n_elem;
      arma::mat I_ng    = arma::eye(ng, ng);
      arma::vec coeff_g = coeff.elem(idx);
      //arma::mat Pi_g    = group_weights(g) * (I_ng / gnorms(g) - (coeff_g * coeff_g.t()) / std::pow(gnorms(g), 3));
      
      // aggiunto il 27 Oct 2025
      arma::vec var_weights_g = var_weights.elem(idx);
      arma::mat T_g           = diagmat(var_weights_g);
      arma::mat T2_g          = diagmat(var_weights_g) * diagmat(var_weights_g);
      arma::vec w_g           = T_g * coeff_g;
      arma::vec w2_g          = T2_g * coeff_g;
      arma::mat Pi_g          = group_weights(g) * (T2_g / norm(w_g, 2) - (w2_g * w2_g.t()) / std::pow(norm(w_g, 2), 3));

      // build selection matrix
      arma::umat Smat = ovglasso_group_selmat(Aset, idx);
        
      // compute the denominator
      Pi_Aset += Smat * Pi_g * Smat.t();
    }
    
    /* compute the matrix B         */
    B = Q_inv + (double)n * (lambda * Pi_Aset + lambda2 * DTD_Aset);
    
    /* compute dof         */
    //dof = trace(inv_sympd(B) * Q_inv);
    dof = trace(inv(B) * Q_inv);
  } else {
    arma::uvec Aset_G_idx(G, fill::zeros);
    dof = 0.0;
  }

  /* Get output         */
  output["groups_active"] = Aset_G_idx;
  output["coeff_active"]  = coeff_active;
  output["var_active"]    = Aset_idx;
  output["dof"]           = dof;
  
  // Return output
  return output;
}

//' @name  fdaSP.internal
//' @keywords internal
// [[Rcpp::export(.f2s_cov_smo_dof_1lambda)]]
Rcpp::List f2s_cov_smo_dof_1lambda(const arma::mat& W,
                                   const arma::mat& Z,
                                   const arma::vec& coeff_W,
                                   const double lambda,
                                   const double lambda2,
                                   const int diff_order,
                                   const arma::mat& GRmat,
                                   const arma::vec& group_weights,
                                   const arma::vec& var_weights,
                                   const arma::vec& Uvec,
                                   const double err_primal,
                                   const double err_dual,
                                   const double rho,
                                   const double toler_c,
                                   const double toler_d) {
  
  /* Variable declaration           */
  int n = 0, p = 0, q = 0, G = 0, g = 0, ng = 0, r_init = 0, r_end = 0;
  double dof = 0.0, tau = 0.0, delta = 0.0, dual_cond = 0.0, dual_cond_rhs = 0.0;
  List output;
    
  /* Variable definition                    */
  n = W.n_rows;
  p = W.n_cols;
  q = Z.n_cols;
  G = GRmat.n_rows;

  /* Definition of vectors and matrices     */
  arma::vec nG(G, fill::zeros);
  arma::uvec res(G, fill::zeros);
  arma::vec gnorms(G, fill::zeros);
  arma::mat coeff_active(G, p, fill::zeros);
  arma::mat DTD(p, p, fill::zeros);
  arma::uvec Aset_idx(p, fill::zeros);

  // compute the number of groups
  nG = arma::sum(GRmat, 1);
  
  // Compute thresholds
  tau   = toler_c * err_primal / std::sqrt((double)p);
  delta = toler_d * err_dual;
  
  // Construct Fmat
  arma::sp_mat Fmat = ovglasso_Gmat2Fmat_sparse(GRmat,
                                                var_weights);
  
  // Construct the penalty matrix
  DTD = forward_diff_penalty_matrix(coeff_W.n_elem, 1, diff_order);
      
  // find the active sets
  r_init = 0;
  r_end  = nG(0) - 1;
  for (int g = 0; g < G; ++g) {
    // 1. Primal condition
    arma::uvec idx = find(GRmat.row(g) == 1);
    if (!idx.is_empty()) {
      gnorms(g) = norm(coeff_W.elem(idx), 2);
    }

    // 2. Dual condition
    if (g > 0) {
      r_init = r_end + 1;
      r_end  = r_init + nG(g) - 1;
    }
    arma::sp_mat Fmat_g = Fmat.rows(r_init, r_end);
    arma::vec Uvec_b    = Uvec.subvec(r_init, r_end);
    dual_cond           = norm(Fmat_g * coeff_W + Uvec_b, 2);
    dual_cond_rhs       = delta + lambda * group_weights(g) / rho;
    
    // Check both conditions
    if ((gnorms(g) >= tau) || (dual_cond >= dual_cond_rhs)) {
      // returns 1 if the group is not zero
      res(g) = 1;
      
      // Extract corresponding elements of coeff
      coeff_active.row(g) = GRmat.row(g) % coeff_W.t();
    }
  }
  arma::uvec Aset_G = find(res == 1);
  //arma::uvec Aset   = find(res.t() * GRmat > 0.0);
  for (int s=0; s<p; s++) {
    arma::uvec id = find(GRmat.col(s) > 0);
    Aset_idx(s)   = prod(res(id));
  }
  arma::uvec Aset = find(Aset_idx > 0.0);
  arma::uvec Aset_G_idx(G, fill::zeros);
  Aset_G_idx.elem(Aset_G).ones();
  //Aset_idx.elem(Aset).ones();
  
  /* dof are zeros if Aset is empty         */
  if (!Aset.is_empty()) {
    
    /* compute relevant quantities         */
    arma::mat Pi(Aset.n_elem + q, Aset.n_elem + q, fill::zeros);
    arma::mat R(Aset.n_elem + q, Aset.n_elem + q, fill::zeros);
    arma::mat Q_inv(Aset.n_elem + q, Aset.n_elem + q, fill::zeros);
    arma::mat B(Aset.n_elem + q, Aset.n_elem + q, fill::zeros);
    
    // Construct the matrix of active penalty
    arma::mat DTD_Aset = DTD.submat(Aset, Aset);
        
    // Construct the matrix of active covariates
    arma::mat W_Aset = W.cols(Aset);
    
    // fill the matrix Q
    Q_inv.submat(0, 0, Aset.n_elem-1, Aset.n_elem-1)                         = W_Aset.t() * W_Aset;
    Q_inv.submat(0, Aset.n_elem, Aset.n_elem-1, Aset.n_elem+q-1)             = W_Aset.t() * Z;
    Q_inv.submat(Aset.n_elem, 0, Aset.n_elem+q-1, Aset.n_elem-1)             = (Q_inv.submat(0, Aset.n_elem, Aset.n_elem-1, Aset.n_elem+q-1)).t();
    Q_inv.submat(Aset.n_elem, Aset.n_elem, Aset.n_elem+q-1, Aset.n_elem+q-1) = Z.t() * Z;
    
    // fill the matrix R
    R.submat(0, 0, Aset.n_elem-1, Aset.n_elem-1) = DTD_Aset;
    
    // calculate the degrees of freedom
    arma::mat Pi_Aset(Aset.n_elem, Aset.n_elem, fill::zeros);
    for (unsigned int k=0; k<Aset_G.n_elem; k++) {
      g                 = Aset_G(k);
      arma::uvec idx    = find(GRmat.row(g) == 1);
      ng                = idx.n_elem;
      arma::mat I_ng    = arma::eye(ng, ng);
      arma::vec coeff_g = coeff_W.elem(idx);
      //arma::mat Pi_g    = group_weights(g) * (I_ng / gnorms(g) - (coeff_g * coeff_g.t()) / std::pow(gnorms(g), 3));
      
      // aggiunto il 27 Oct 2025
      arma::vec var_weights_g = var_weights.elem(idx);
      arma::mat T_g           = diagmat(var_weights_g);
      arma::mat T2_g          = diagmat(var_weights_g) * diagmat(var_weights_g);
      arma::vec w_g           = T_g * coeff_g;
      arma::vec w2_g          = T2_g * coeff_g;
      arma::mat Pi_g          = group_weights(g) * (T2_g / norm(w_g, 2) - (w2_g * w2_g.t()) / std::pow(norm(w_g, 2), 3));

      // build selection matrix
      arma::umat Smat = ovglasso_group_selmat(Aset, idx);
        
      // compute the denominator
      Pi_Aset += Smat * Pi_g * Smat.t();
    }
    
    /* fill the matrix Pi_C             */
    Pi.submat(0, 0, Aset.n_elem-1, Aset.n_elem-1) = Pi_Aset;
    
    /* compute the matrix B         */
    B = Q_inv + (double)n * (lambda * Pi + lambda2 * R);
    
    /* compute dof         */
    //dof = trace(inv_sympd(B) * Q_inv);
    dof = trace(inv(B) * Q_inv);
  } else {
    arma::uvec Aset_G_idx(G, fill::zeros);
    dof = (double)q;
  }
  
  /* Get output         */
  output["groups_active"] = Aset_G_idx;
  output["coeff_active"]  = coeff_active;
  output["var_active"]    = Aset_idx;
  output["dof"]           = dof;
  
  // Return output
  return output;
}

//' @name  fdaSP.internal
//' @keywords internal
// [[Rcpp::export(.f2s_adaptive_dof_1lambda)]]
Rcpp::List f2s_adaptive_dof_1lambda(const arma::mat& W,
                                    const arma::vec& coeff,
                                    const arma::vec& coeff_LS,
                                    const double lambda,
                                    const arma::mat& GRmat,
                                    const arma::vec& group_weights,
                                    const arma::vec& var_weights,
                                    const arma::vec& Uvec,
                                    const double err_primal,
                                    const double err_dual,
                                    const double rho,
                                    const double toler_c,
                                    const double toler_d) {
  
  /* Variable declaration           */
  int n = 0, p = 0, G = 0, g = 0, ng = 0, r_init = 0, r_end = 0;
  double dof = 0.0, tau = 0.0, delta = 0.0, dual_cond = 0.0, dual_cond_rhs = 0.0;
  List output;
    
  /* Variable definition                    */
  n = W.n_rows;
  p = W.n_cols;
  G = GRmat.n_rows;

  /* Definition of vectors and matrices     */
  arma::vec nG(G, fill::zeros);
  arma::uvec res(G, fill::zeros);
  arma::vec gnorms(G, fill::zeros);
  arma::mat coeff_active(G, p, fill::zeros);
  arma::uvec Aset_idx(p, fill::zeros);
  
  // compute the number of groups
  nG = arma::sum(GRmat, 1);
  
  // Compute thresholds
  tau   = toler_c * err_primal / std::sqrt((double)p);
  delta = toler_d * err_dual;
  
  // Construct Fmat
  arma::sp_mat Fmat = ovglasso_Gmat2Fmat_sparse(GRmat,
                                                var_weights);

  // find the active sets
  r_init = 0;
  r_end  = nG(0) - 1;
  for (int g = 0; g < G; ++g) {
    // 1. Primal condition
    arma::uvec idx = find(GRmat.row(g) == 1);
    if (!idx.is_empty()) {
      gnorms(g) = norm(coeff.elem(idx), 2);
    }

    // 2. Dual condition
    if (g > 0) {
      r_init = r_end + 1;
      r_end  = r_init + nG(g) - 1;
    }
    arma::sp_mat Fmat_g = Fmat.rows(r_init, r_end);
    arma::vec Uvec_b    = Uvec.subvec(r_init, r_end);
    dual_cond           = norm(Fmat_g * coeff + Uvec_b, 2);
    dual_cond_rhs       = delta + lambda * group_weights(g) / rho;
    
    // Check both conditions
    if ((gnorms(g) >= tau) || (dual_cond >= dual_cond_rhs)) {
    //if (!((gnorms(g) < tau) && (dual_cond < dual_cond_rhs))) {
      // returns 1 if the group is not zero
      res(g) = 1;
      
      // Extract corresponding elements of coeff
      coeff_active.row(g) = GRmat.row(g) % coeff.t();
    }
  }
  arma::uvec Aset_G = find(res == 1);
  //arma::uvec Aset   = find(res.t() * GRmat > 0.0);
  for (int s=0; s<p; s++) {
    arma::uvec id = find(GRmat.col(s) > 0);
    Aset_idx(s)   = prod(res(id));
  }
  arma::uvec Aset = find(Aset_idx > 0.0);
  arma::uvec Aset_G_idx(G, fill::zeros);
  Aset_G_idx.elem(Aset_G).ones();
  //Aset_idx.elem(Aset).ones();

  /* dof are zeros if Aset is empty         */
  if (!Aset.is_empty()) {
    
    /* compute relevant quantities         */
    arma::mat Q_inv(Aset.n_elem, Aset.n_elem, fill::zeros);
    arma::mat B(Aset.n_elem, Aset.n_elem, fill::zeros);
    
    // Construct the matrix of active covariates
    arma::mat W_Aset = W.cols(Aset);
        
    // fill the matrix Q
    Q_inv = W_Aset.t() * W_Aset;
      
    // calculate the degrees of freedom
    arma::mat Pi_Aset(Aset.n_elem, Aset.n_elem, fill::zeros);
    arma::mat Phi_Aset(Aset.n_elem, Aset.n_elem, fill::zeros);
    for (unsigned int k=0; k<Aset_G.n_elem; k++) {
      g                 = Aset_G(k);
      arma::uvec idx    = find(GRmat.row(g) == 1);
      ng                = idx.n_elem;
      arma::mat I_ng    = arma::eye(ng, ng);
      arma::vec coeff_g = coeff.elem(idx);
      //arma::mat Pi_g    = group_weights(g) * (I_ng / gnorms(g) - (coeff_g * coeff_g.t()) / std::pow(gnorms(g), 3));
      
      // aggiunto il 27 Oct 2025
      arma::vec var_weights_g = var_weights.elem(idx);
      arma::mat T_g           = diagmat(var_weights_g);
      arma::mat T2_g          = diagmat(var_weights_g) * diagmat(var_weights_g);
      arma::vec w_g           = T_g * coeff_g;
      arma::vec w2_g          = T2_g * coeff_g;
      arma::mat Pi_g          = group_weights(g) * (T2_g / norm(w_g, 2) - (w2_g * w2_g.t()) / std::pow(norm(w_g, 2), 3));

      // build selection matrix
      arma::umat Smat = ovglasso_group_selmat(Aset, idx);
        
      // compute the denominator
      Pi_Aset += Smat * Pi_g * Smat.t();
      
      // compute the derivative of the adaptive part
      arma::vec coeff_LS_g  = coeff_LS.elem(idx);
      arma::mat Phi_g       = coeff_g * coeff_LS_g.t();
      Phi_g                /= -std::pow(norm(coeff_LS_g, 2), 3) * gnorms(g);
      Phi_Aset             += Smat * Phi_g * Smat.t();
    }
    
    /* compute the matrix B         */
    B = Q_inv + (double)n * lambda * Pi_Aset;
    
    /* compute dof         */
    //dof = trace(inv_sympd(B) * (Q_inv - lambda * Phi_Aset));
    dof = trace(inv(B) * (Q_inv - lambda * Phi_Aset));
  } else {
    arma::uvec Aset_G_idx(G, fill::zeros);
    dof = 0.0;
  }

  /* Get output         */
  output["groups_active"] = Aset_G_idx;
  output["coeff_active"]  = coeff_active;
  output["var_active"]    = Aset_idx;
  output["dof"]           = dof;
  
  // Return output
  return output;
}

//' @name  fdaSP.internal
//' @keywords internal
// [[Rcpp::export(.f2s_cov_adaptive_dof_1lambda)]]
Rcpp::List f2s_cov_adaptive_dof_1lambda(const arma::mat& W,
                                        const arma::mat& Z,
                                        const arma::vec& coeff_W,
                                        const arma::vec& coeff_W_LS,
                                        const double lambda,
                                        const arma::mat& GRmat,
                                        const arma::vec& group_weights,
                                        const arma::vec& var_weights,
                                        const arma::vec& Uvec,
                                        const double err_primal,
                                        const double err_dual,
                                        const double rho,
                                        const double toler_c,
                                        const double toler_d) {
          
  /* Variable declaration           */
  int n = 0, p = 0, q = 0, G = 0, g = 0, ng = 0, r_init = 0, r_end = 0;
  double dof = 0.0, tau = 0.0, delta = 0.0, dual_cond = 0.0, dual_cond_rhs = 0.0;
  List output;
    
  /* Variable definition                    */
  n = W.n_rows;
  p = W.n_cols;
  q = Z.n_cols;
  G = GRmat.n_rows;

  /* Definition of vectors and matrices     */
  arma::vec nG(G, fill::zeros);
  arma::uvec res(G, fill::zeros);
  arma::vec gnorms(G, fill::zeros);
  arma::mat coeff_active(G, p, fill::zeros);
  arma::uvec Aset_idx(p, fill::zeros);
  
  // compute the number of groups
  nG = arma::sum(GRmat, 1);
  
  // Compute thresholds
  tau   = toler_c * err_primal / std::sqrt((double)p);
  delta = toler_d * err_dual;
  
  // Construct Fmat
  arma::sp_mat Fmat = ovglasso_Gmat2Fmat_sparse(GRmat,
                                                var_weights);
  
  // find the active sets
  r_init = 0;
  r_end  = nG(0) - 1;
  for (int g = 0; g < G; ++g) {
    // 1. Primal condition
    arma::uvec idx = find(GRmat.row(g) == 1);
    if (!idx.is_empty()) {
      gnorms(g) = norm(coeff_W.elem(idx), 2);
    }

    // 2. Dual condition
    if (g > 0) {
      r_init = r_end + 1;
      r_end  = r_init + nG(g) - 1;
    }
    arma::sp_mat Fmat_g = Fmat.rows(r_init, r_end);
    arma::vec Uvec_b    = Uvec.subvec(r_init, r_end);
    dual_cond           = norm(Fmat_g * coeff_W + Uvec_b, 2);
    dual_cond_rhs       = delta + lambda * group_weights(g) / rho;
    
    // Check both conditions
    if ((gnorms(g) >= tau) || (dual_cond >= dual_cond_rhs)) {
      // returns 1 if the group is not zero
      res(g) = 1;
      
      // Extract corresponding elements of coeff
      coeff_active.row(g) = GRmat.row(g) % coeff_W.t();
    }
  }
  arma::uvec Aset_G = find(res == 1);
  //arma::uvec Aset   = find(res.t() * GRmat > 0.0);
  for (int s=0; s<p; s++) {
    arma::uvec id = find(GRmat.col(s) > 0);
    Aset_idx(s)   = prod(res(id));
  }
  arma::uvec Aset = find(Aset_idx > 0.0);
  arma::uvec Aset_G_idx(G, fill::zeros);
  Aset_G_idx.elem(Aset_G).ones();
  //Aset_idx.elem(Aset).ones();

  /* dof are zeros if Aset is empty         */
  if (!Aset.is_empty()) {
    
    /* compute relevant quantities         */
    arma::mat Pi(Aset.n_elem + q, Aset.n_elem + q, fill::zeros);
    arma::mat Phi(Aset.n_elem + q, Aset.n_elem + q, fill::zeros);
    arma::mat Q_inv(Aset.n_elem + q, Aset.n_elem + q, fill::zeros);
    arma::mat B(Aset.n_elem + q, Aset.n_elem + q, fill::zeros);
    
    // Construct the matrix of active covariates
    arma::mat W_Aset = W.cols(Aset);
        
    // fill the matrix Q
    Q_inv.submat(0, 0, Aset.n_elem-1, Aset.n_elem-1)                         = W_Aset.t() * W_Aset;
    Q_inv.submat(0, Aset.n_elem, Aset.n_elem-1, Aset.n_elem+q-1)             = W_Aset.t() * Z;
    Q_inv.submat(Aset.n_elem, 0, Aset.n_elem+q-1, Aset.n_elem-1)             = (Q_inv.submat(0, Aset.n_elem, Aset.n_elem-1, Aset.n_elem+q-1)).t();
    Q_inv.submat(Aset.n_elem, Aset.n_elem, Aset.n_elem+q-1, Aset.n_elem+q-1) = Z.t() * Z;
    
    // calculate the degrees of freedom
    arma::mat Pi_Aset(Aset.n_elem, Aset.n_elem, fill::zeros);
    arma::mat Phi_Aset(Aset.n_elem, Aset.n_elem, fill::zeros);
    for (unsigned int k=0; k<Aset_G.n_elem; k++) {
      g                 = Aset_G(k);
      arma::uvec idx    = find(GRmat.row(g) == 1);
      ng                = idx.n_elem;
      arma::mat I_ng    = arma::eye(ng, ng);
      arma::vec coeff_g = coeff_W.elem(idx);
      //arma::mat Pi_g    = group_weights(g) * (I_ng / gnorms(g) - (coeff_g * coeff_g.t()) / std::pow(gnorms(g), 3));
      
      // aggiunto il 27 Oct 2025
      arma::vec var_weights_g = var_weights.elem(idx);
      arma::mat T_g           = diagmat(var_weights_g);
      arma::mat T2_g          = diagmat(var_weights_g) * diagmat(var_weights_g);
      arma::vec w_g           = T_g * coeff_g;
      arma::vec w2_g          = T2_g * coeff_g;
      arma::mat Pi_g          = group_weights(g) * (T2_g / norm(w_g, 2) - (w2_g * w2_g.t()) / std::pow(norm(w_g, 2), 3));

      // build selection matrix
      arma::umat Smat = ovglasso_group_selmat(Aset, idx);
        
      // compute the denominator
      Pi_Aset += Smat * Pi_g * Smat.t();
      
      // compute the derivative of the adaptive part
      arma::vec coeff_W_LS_g  = coeff_W_LS.elem(idx);
      arma::mat Phi_g         = coeff_g * coeff_W_LS_g.t();
      Phi_g                  /= -std::pow(norm(coeff_W_LS_g, 2), 3) * gnorms(g);
      Phi_Aset               += Smat * Phi_g * Smat.t();
    }
    
    /* fill the matrix Pi_C             */
    Pi.submat(0, 0, Aset.n_elem-1, Aset.n_elem-1)  = Pi_Aset;
    Phi.submat(0, 0, Aset.n_elem-1, Aset.n_elem-1) = Phi_Aset;
    
    /* compute the matrix B         */
    B = Q_inv + (double)n * (lambda * Pi);
        
    /* compute dof         */
    //dof = trace(inv_sympd(B) * (Q_inv - lambda * Phi));
    dof = trace(inv(B) * (Q_inv - lambda * Phi));
  } else {
    arma::uvec Aset_G_idx(G, fill::zeros);
    dof = (double)q;
  }

  /* Get output         */
  output["groups_active"] = Aset_G_idx;
  output["coeff_active"]  = coeff_active;
  output["var_active"]    = Aset_idx;
  output["dof"]           = dof;
  
  // Return output
  return output;
}

//' @name  fdaSP.internal
//' @keywords internal
// [[Rcpp::export(.f2s_smo_adaptive_dof_1lambda)]]
Rcpp::List f2s_smo_adaptive_dof_1lambda(const arma::mat& W,
                                        const arma::vec& coeff,
                                        const arma::vec& coeff_LS,
                                        const double lambda,
                                        const double lambda2,
                                        const int diff_order,
                                        const arma::mat& GRmat,
                                        const arma::vec& group_weights,
                                        const arma::vec& var_weights,
                                        const arma::vec& Uvec,
                                        const double err_primal,
                                        const double err_dual,
                                        const double rho,
                                        const double toler_c,
                                        const double toler_d) {
  
  /* Variable declaration           */
  int n = 0, p = 0, G = 0, g = 0, ng = 0, r_init = 0, r_end = 0;
  double dof = 0.0, tau = 0.0, delta = 0.0, dual_cond = 0.0, dual_cond_rhs = 0.0;
  List output;
    
  /* Variable definition                    */
  n = W.n_rows;
  p = W.n_cols;
  G = GRmat.n_rows;

  /* Definition of vectors and matrices     */
  arma::vec nG(G, fill::zeros);
  arma::uvec res(G, fill::zeros);
  arma::vec gnorms(G, fill::zeros);
  arma::mat coeff_active(G, p, fill::zeros);
  arma::mat DTD(p, p, fill::zeros);
  arma::uvec Aset_idx(p, fill::zeros);

  // compute the number of groups
  nG = arma::sum(GRmat, 1);
  
  // Compute thresholds
  tau   = toler_c * err_primal / std::sqrt((double)p);
  delta = toler_d * err_dual;
  
  // Construct Fmat
  arma::sp_mat Fmat = ovglasso_Gmat2Fmat_sparse(GRmat,
                                                var_weights);
  
  // Construct the penalty matrix
  DTD = forward_diff_penalty_matrix(coeff.n_elem, 1, diff_order);
    
  // find the active sets
  r_init = 0;
  r_end  = nG(0) - 1;
  for (int g = 0; g < G; ++g) {
    // 1. Primal condition
    arma::uvec idx = find(GRmat.row(g) == 1);
    if (!idx.is_empty()) {
      gnorms(g) = norm(coeff.elem(idx), 2);
    }

    // 2. Dual condition
    if (g > 0) {
      r_init = r_end + 1;
      r_end  = r_init + nG(g) - 1;
    }
    arma::sp_mat Fmat_g = Fmat.rows(r_init, r_end);
    arma::vec Uvec_b    = Uvec.subvec(r_init, r_end);
    dual_cond           = norm(Fmat_g * coeff + Uvec_b, 2);
    dual_cond_rhs       = delta + lambda * group_weights(g) / rho;
    
    // Check both conditions
    if ((gnorms(g) >= tau) || (dual_cond >= dual_cond_rhs)) {
      // returns 1 if the group is not zero
      res(g) = 1;
      
      // Extract corresponding elements of coeff
      coeff_active.row(g) = GRmat.row(g) % coeff.t();
    }
  }
  arma::uvec Aset_G = find(res == 1);
  //arma::uvec Aset   = find(res.t() * GRmat > 0.0);
  for (int s=0; s<p; s++) {
    arma::uvec id = find(GRmat.col(s) > 0);
    Aset_idx(s)   = prod(res(id));
  }
  arma::uvec Aset = find(Aset_idx > 0.0);
  arma::uvec Aset_G_idx(G, fill::zeros);
  Aset_G_idx.elem(Aset_G).ones();
  //Aset_idx.elem(Aset).ones();
  
  /* dof are zeros if Aset is empty         */
  if (!Aset.is_empty()) {
    
    /* compute relevant quantities         */
    arma::mat Q_inv(Aset.n_elem, Aset.n_elem, fill::zeros);
    arma::mat B(Aset.n_elem, Aset.n_elem, fill::zeros);
    
    // Construct the matrix of active penalty
    arma::mat DTD_Aset = DTD.submat(Aset, Aset);
        
    // Construct the matrix of active covariates
    arma::mat W_Aset = W.cols(Aset);
        
    // fill the matrix Q
    //Q_inv = inv_sympd(W_Aset.t() * W_Aset);
    Q_inv = W_Aset.t() * W_Aset;                                    // Modificata il 23 Nov 25
      
    // calculate the degrees of freedom
    arma::mat Pi_Aset(Aset.n_elem, Aset.n_elem, fill::zeros);
    arma::mat Phi_Aset(Aset.n_elem, Aset.n_elem, fill::zeros);
    for (unsigned int k=0; k<Aset_G.n_elem; k++) {
      g                 = Aset_G(k);
      arma::uvec idx    = find(GRmat.row(g) == 1);
      ng                = idx.n_elem;
      arma::mat I_ng    = arma::eye(ng, ng);
      arma::vec coeff_g = coeff.elem(idx);
      //arma::mat Pi_g    = group_weights(g) * (I_ng / gnorms(g) - (coeff_g * coeff_g.t()) / std::pow(gnorms(g), 3));
      
      // aggiunto il 27 Oct 2025
      arma::vec var_weights_g = var_weights.elem(idx);
      arma::mat T_g           = diagmat(var_weights_g);
      arma::mat T2_g          = diagmat(var_weights_g) * diagmat(var_weights_g);
      arma::vec w_g           = T_g * coeff_g;
      arma::vec w2_g          = T2_g * coeff_g;
      arma::mat Pi_g          = group_weights(g) * (T2_g / norm(w_g, 2) - (w2_g * w2_g.t()) / std::pow(norm(w_g, 2), 3));

      // build selection matrix
      arma::umat Smat = ovglasso_group_selmat(Aset, idx);
        
      // compute the denominator
      Pi_Aset += Smat * Pi_g * Smat.t();
      
      // compute the derivative of the adaptive part
      arma::vec coeff_LS_g  = coeff_LS.elem(idx);
      arma::mat Phi_g       = coeff_g * coeff_LS_g.t();
      Phi_g                /= -std::pow(norm(coeff_LS_g, 2), 3) * gnorms(g);
      Phi_Aset             += Smat * Phi_g * Smat.t();
    }
    
    /* compute the matrix B         */
    B = Q_inv + (double)n * (lambda * Pi_Aset + lambda2 * DTD_Aset);
    
    /* compute dof         */
    //dof = trace(inv_sympd(B) * (Q_inv - lambda * Phi_Aset));
    dof = trace(inv(B) * (Q_inv - lambda * Phi_Aset));
  } else {
    arma::uvec Aset_G_idx(G, fill::zeros);
    dof = 0.0;
  }

  /* Get output         */
  output["groups_active"] = Aset_G_idx;
  output["coeff_active"]  = coeff_active;
  output["var_active"]    = Aset_idx;
  output["dof"]           = dof;
  
  // Return output
  return output;
}

//' @name  fdaSP.internal
//' @keywords internal
// [[Rcpp::export(.f2s_adaptive_cov_smo_dof_1lambda)]]
Rcpp::List f2s_adaptive_cov_smo_dof_1lambda(const arma::mat& W,
                                            const arma::mat& Z,
                                            const arma::vec& coeff_W,
                                            const arma::vec& coeff_W_LS,
                                            const double lambda,
                                            const double lambda2,
                                            const int diff_order,
                                            const arma::mat& GRmat,
                                            const arma::vec& group_weights,
                                            const arma::vec& var_weights,
                                            const arma::vec& Uvec,
                                            const double err_primal,
                                            const double err_dual,
                                            const double rho,
                                            const double toler_c,
                                            const double toler_d) {
  
  /* Variable declaration           */
  int n = 0, p = 0, q = 0, G = 0, g = 0, ng = 0, r_init = 0, r_end = 0;
  double dof = 0.0, tau = 0.0, delta = 0.0, dual_cond = 0.0, dual_cond_rhs = 0.0;
  List output;
    
  /* Variable definition                    */
  n = W.n_rows;
  p = W.n_cols;
  q = Z.n_cols;
  G = GRmat.n_rows;

  /* Definition of vectors and matrices     */
  arma::vec nG(G, fill::zeros);
  arma::uvec res(G, fill::zeros);
  arma::vec gnorms(G, fill::zeros);
  arma::mat coeff_active(G, p, fill::zeros);
  arma::mat DTD(p, p, fill::zeros);
  arma::uvec Aset_idx(p, fill::zeros);

  // compute the number of groups
  nG = arma::sum(GRmat, 1);
  
  // Compute thresholds
  tau   = toler_c * err_primal / std::sqrt((double)p);
  delta = toler_d * err_dual;
  
  // Construct Fmat
  arma::sp_mat Fmat = ovglasso_Gmat2Fmat_sparse(GRmat,
                                                var_weights);
  
  // Construct the penalty matrix
  DTD = forward_diff_penalty_matrix(coeff_W.n_elem, 1, diff_order);
    
  // find the active sets
  r_init = 0;
  r_end  = nG(0) - 1;
  for (int g = 0; g < G; ++g) {
    // 1. Primal condition
    arma::uvec idx = find(GRmat.row(g) == 1);
    if (!idx.is_empty()) {
      gnorms(g) = norm(coeff_W.elem(idx), 2);
    }

    // 2. Dual condition
    if (g > 0) {
      r_init = r_end + 1;
      r_end  = r_init + nG(g) - 1;
    }
    arma::sp_mat Fmat_g = Fmat.rows(r_init, r_end);
    arma::vec Uvec_b    = Uvec.subvec(r_init, r_end);
    dual_cond           = norm(Fmat_g * coeff_W + Uvec_b, 2);
    dual_cond_rhs       = delta + lambda * group_weights(g) / rho;
    
    // Check both conditions
    if ((gnorms(g) >= tau) || (dual_cond >= dual_cond_rhs)) {
      // returns 1 if the group is not zero
      res(g) = 1;
      
      // Extract corresponding elements of coeff
      coeff_active.row(g) = GRmat.row(g) % coeff_W.t();
    }
  }
  arma::uvec Aset_G = find(res == 1);
  //arma::uvec Aset   = find(res.t() * GRmat > 0.0);
  for (int s=0; s<p; s++) {
    arma::uvec id = find(GRmat.col(s) > 0);
    Aset_idx(s)   = prod(res(id));
  }
  arma::uvec Aset = find(Aset_idx > 0.0);
  arma::uvec Aset_G_idx(G, fill::zeros);
  Aset_G_idx.elem(Aset_G).ones();
  //Aset_idx.elem(Aset).ones();
  
  /* dof are zeros if Aset is empty         */
  if (!Aset.is_empty()) {
    
    /* compute relevant quantities         */
    arma::mat Pi(Aset.n_elem + q, Aset.n_elem + q, fill::zeros);
    arma::mat R(Aset.n_elem + q, Aset.n_elem + q, fill::zeros);
    arma::mat Q_inv(Aset.n_elem + q, Aset.n_elem + q, fill::zeros);
    arma::mat B(Aset.n_elem + q, Aset.n_elem + q, fill::zeros);
    arma::mat Phi(Aset.n_elem + q, Aset.n_elem + q, fill::zeros);
    
    // Construct the matrix of active penalty
    arma::mat DTD_Aset = DTD.submat(Aset, Aset);
        
    // Construct the matrix of active covariates
    arma::mat W_Aset = W.cols(Aset);
        
    // fill the matrix Q
    Q_inv.submat(0, 0, Aset.n_elem-1, Aset.n_elem-1)                         = W_Aset.t() * W_Aset;
    Q_inv.submat(0, Aset.n_elem, Aset.n_elem-1, Aset.n_elem+q-1)             = W_Aset.t() * Z;
    Q_inv.submat(Aset.n_elem, 0, Aset.n_elem+q-1, Aset.n_elem-1)             = (Q_inv.submat(0, Aset.n_elem, Aset.n_elem-1, Aset.n_elem+q-1)).t();
    Q_inv.submat(Aset.n_elem, Aset.n_elem, Aset.n_elem+q-1, Aset.n_elem+q-1) = Z.t() * Z;
    
    // fill the matrix R
    R.submat(0, 0, Aset.n_elem-1, Aset.n_elem-1) = DTD_Aset;
    
    // calculate the degrees of freedom
    arma::mat Pi_Aset(Aset.n_elem, Aset.n_elem, fill::zeros);
    arma::mat Phi_Aset(Aset.n_elem, Aset.n_elem, fill::zeros);
    for (unsigned int k=0; k<Aset_G.n_elem; k++) {
      g                 = Aset_G(k);
      arma::uvec idx    = find(GRmat.row(g) == 1);
      ng                = idx.n_elem;
      arma::mat I_ng    = arma::eye(ng, ng);
      arma::vec coeff_g = coeff_W.elem(idx);
      //arma::mat Pi_g    = group_weights(g) * (I_ng / gnorms(g) - (coeff_g * coeff_g.t()) / std::pow(gnorms(g), 3));

      // aggiunto il 27 Oct 2025
      arma::vec var_weights_g = var_weights.elem(idx);
      arma::mat T_g           = diagmat(var_weights_g);
      arma::mat T2_g          = diagmat(var_weights_g) * diagmat(var_weights_g);
      arma::vec w_g           = T_g * coeff_g;
      arma::vec w2_g          = T2_g * coeff_g;
      arma::mat Pi_g          = group_weights(g) * (T2_g / norm(w_g, 2) - (w2_g * w2_g.t()) / std::pow(norm(w_g, 2), 3));
      
      // build selection matrix
      arma::umat Smat = ovglasso_group_selmat(Aset, idx);
        
      // compute the denominator
      Pi_Aset += Smat * Pi_g * Smat.t();
      
      // compute the derivative of the adaptive part
      arma::vec coeff_LS_g  = coeff_W_LS.elem(idx);
      arma::mat Phi_g       = coeff_g * coeff_LS_g.t();
      Phi_g                /= -std::pow(norm(coeff_LS_g, 2), 3) * gnorms(g);
      Phi_Aset             += Smat * Phi_g * Smat.t();
    }
    
    /* fill the matrix Pi_C             */
    Pi.submat(0, 0, Aset.n_elem-1, Aset.n_elem-1)  = Pi_Aset;
    Phi.submat(0, 0, Aset.n_elem-1, Aset.n_elem-1) = Phi_Aset;
    R.submat(0, 0, Aset.n_elem-1, Aset.n_elem-1)   = DTD_Aset;
    
    /* compute the matrix B         */
    B = Q_inv + (double)n * (lambda * Pi + lambda2 * R);
        
    /* compute dof         */
    //dof = trace(inv_sympd(B) * (Q_inv - lambda * Phi));
    dof = trace(inv(B) * (Q_inv - lambda * Phi));
  } else {
    arma::uvec Aset_G_idx(G, fill::zeros);
    dof = (double)q;
  }

  /* Get output         */
  output["groups_active"] = Aset_G_idx;
  output["coeff_active"]  = coeff_active;
  output["var_active"]    = Aset_idx;
  output["dof"]           = dof;
  
  // Return output
  return output;
}

//' @name  fdaSP.internal
//' @keywords internal
// [[Rcpp::export(.f2s_bic)]]
arma::vec f2s_bic(const arma::vec& y,
                  const arma::mat& W,
                  const arma::mat& coeff_path,
                  const arma::vec& df) {
  
  /* Variable declaration           */
  int n = 0, p = 0, nlambda = coeff_path.n_rows;
  double rss = 0.0;
  
  /* Variable definition            */
  n = W.n_rows;
  p = W.n_cols;
  
  /* Definition of vectors and matrices     */
  arma::vec coeff(p);                 coeff.zeros();
  arma::vec bic(nlambda);             bic.zeros();
  
  for (int i = 0; i < nlambda; ++i) {
    coeff  = coeff_path.row(i).t();
    rss    = pow(arma::norm(y - W * coeff, 2), 2);
    bic(i) = n * std::log(rss / (double)n) + df(i) * std::log(n);
  }
  return bic;
}

//' @name  fdaSP.internal
//' @keywords internal
// [[Rcpp::export(.f2s_cov_bic)]]
arma::vec f2s_cov_bic(const arma::vec& y,
                      const arma::mat& W,
                      const arma::mat& Z,
                      const arma::mat& W_coeff_path,
                      const arma::mat& Z_coeff_path,
                      const arma::vec& df) {
  
  /* Variable declaration           */
  int n = 0, p = 0, q = 0, nlambda = W_coeff_path.n_rows;
  double rss = 0.0;
  
  /* Variable definition            */
  n = W.n_rows;
  p = W.n_cols;
  q = Z.n_cols;
  
  /* Definition of vectors and matrices     */
  arma::vec coeff_W(p);               coeff_W.zeros();
  arma::vec coeff_Z(q);               coeff_Z.zeros();
  arma::vec bic(nlambda);             bic.zeros();
  
  for (int i = 0; i < nlambda; ++i) {
    coeff_W = W_coeff_path.row(i).t();
    coeff_Z = Z_coeff_path.row(i).t();
    rss     = pow(arma::norm(y - W * coeff_W - Z * coeff_Z, 2), 2);
    bic(i)  = n * std::log(rss / (double)n) + df(i) * std::log(n);
  }
  return bic;
}

//' @name  fdaSP.internal
//' @keywords internal
// [[Rcpp::export(.f2s_ebic)]]
arma::vec f2s_ebic(const arma::vec& y,
                   const arma::mat& X,
                   const arma::mat& coeff_path,
                   const arma::vec& df,
                   const double kappa) {
   
   /* Variable declaration           */
   int n = 0, p = 0, nlambda = coeff_path.n_rows;
   double rss = 0.0;
   
   /* Variable definition            */
   n = X.n_rows;
   p = X.n_cols;
   
   /* Definition of vectors and matrices     */
   arma::vec coeff(p);                 coeff.zeros();
   arma::vec ebic(nlambda);            ebic.zeros();
  
  for (int i = 0; i < nlambda; ++i) {
    coeff   = coeff_path.row(i).t();
    rss     = pow(arma::norm(y - X * coeff, 2), 2);
    ebic(i) = n * std::log(rss / (double)n) + kappa * df(i) * std::log(n);
  }
  return ebic;
}

//' @name  fdaSP.internal
//' @keywords internal
// [[Rcpp::export(.f2s_cov_ebic)]]
arma::vec f2s_cov_ebic(const arma::vec& y,
                       const arma::mat& W,
                       const arma::mat& Z,
                       const arma::mat& W_coeff_path,
                       const arma::mat& Z_coeff_path,
                       const arma::vec& df,
                       const double kappa) {
  
  /* Variable declaration           */
  int n = 0, p = 0, q = 0, nlambda = W_coeff_path.n_rows;
  double rss = 0.0;
  
  /* Variable definition            */
  n = W.n_rows;
  p = W.n_cols;
  q = Z.n_cols;
  
  /* Definition of vectors and matrices     */
  arma::vec coeff_W(p);               coeff_W.zeros();
  arma::vec coeff_Z(q);               coeff_Z.zeros();
  arma::vec ebic(nlambda);            ebic.zeros();
  
  for (int i = 0; i < nlambda; ++i) {
    coeff_W = W_coeff_path.row(i).t();
    coeff_Z = Z_coeff_path.row(i).t();
    rss     = pow(arma::norm(y - W * coeff_W - Z * coeff_Z, 2), 2);
    ebic(i) = n * std::log(rss / (double)n) + kappa * df(i) * std::log(n);
  }
  return ebic;
}














