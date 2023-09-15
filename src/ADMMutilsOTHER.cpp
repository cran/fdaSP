// Utility routines

// Authors:
//           Bernardi Mauro, University of Padova
//           Last update: July 12, 2022

// List of implemented methods:

// [[Rcpp::depends(RcppArmadillo)]]
#include "ADMMutilsOTHER.h"

// ============================================================
// admm_tv: Total Variation Minimization via ADMM
arma::colvec tv_shrinkage(arma::colvec a, const double kappa) {
  const int n = a.n_elem;
  arma::colvec y(n,fill::zeros);
  for (int i=0;i<n;i++){
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

double tv_objective(arma::colvec b, const double lambda, arma::mat D,
                    arma::colvec x, arma::colvec z) {
  return(pow(norm(x-b),2)/2 + lambda*norm(z,1));
}

/*
* Total Variation Minimization via ADMM (from Stanford)
* http://stanford.edu/~boyd/papers/pdf/admm_distr_stats.pdf
* page 19 : section 3.3.1 : stopping criteria part (3.12).
*/
Rcpp::List admm_tv(const arma::colvec& b, arma::colvec& xinit, const double lambda,
                   const double reltol, const double abstol, const int maxiter,
                   const double rho, const double alpha) {
  // 1. get parameters and tv difference matrix
  const int n = b.n_elem;
  arma::vec onesN(n,fill::ones);
  arma::mat D = diagmat(onesN);
  for (int i=0;i<(n-1);i++) {
    D(i,i+1) = -1;
  }
  arma::mat I = diagmat(onesN);
  arma::mat DtD = D.t()*D;

  // 2. set ready
  arma::colvec x(n,fill::zeros);
  arma::colvec z(n,fill::zeros);
  arma::colvec u(n,fill::zeros);
  arma::colvec zold(n,fill::zeros);
  arma::colvec Ax_hat(n,fill::zeros);

  // 3. iteration
  arma::vec h_objval(maxiter,fill::zeros);
  arma::vec h_r_norm(maxiter,fill::zeros);
  arma::vec h_s_norm(maxiter,fill::zeros);
  arma::vec h_eps_pri(maxiter,fill::zeros);
  arma::vec h_eps_dual(maxiter,fill::zeros);

  double sqrtn = std::sqrt(static_cast<float>(n));
  arma::vec compare2(2,fill::zeros);
  int k;
  for (k=0;k<maxiter;k++) {
    // 3-1. update 'x'
    x = solve(I+rho*DtD, b+rho*D.t()*(z-u));

    // 3-2. update 'z' with relaxation
    zold = z;
    Ax_hat = alpha*D*x + (1-alpha)*zold;
    z = tv_shrinkage(Ax_hat+u, lambda/rho);

    // 3-3. update 'u'
    u = u + Ax_hat - z;

    // 3-4.. dianostics, reporting
    h_objval(k) = tv_objective(b,lambda,D,x,z);
    h_r_norm(k) = norm(D*x-z);
    h_s_norm(k) = norm(-rho*D.t()*(z-zold));

    compare2(0) = norm(D*x);
    compare2(1) = norm(-z);

    h_eps_pri(k)  = sqrtn*abstol + reltol*max(compare2);
    h_eps_dual(k) = sqrtn*abstol + reltol*norm(rho*D.t()*u);

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
  
  // 6. return results
  return(output);
}

// ============================================================
// admm_bp: Basis Pursuit via ADMM
arma::colvec bp_shrinkage(arma::colvec a, const double kappa) {
  const int n = a.n_elem;
  arma::colvec y(n,fill::zeros);
  for (int i=0;i<n;i++) {
    // first term : max(0, a-kappa)
    if (a(i)-kappa > 0){
      y(i) = a(i)-kappa;
    }
    // second term : -max(0, -a-kappa)
    if (-a(i)-kappa > 0) {
      y(i) = y(i) + a(i) + kappa;
    }
  }
  return(y);
}

/*
* Basis Pursuit via ADMM (from Stanford)
* URL : https://web.stanford.edu/~boyd/papers/admm/basis_pursuit/basis_pursuit.html
* http://stanford.edu/~boyd/papers/pdf/admm_distr_stats.pdf
* page 19 : section 3.3.1 : stopping criteria part (3.12).
*/
Rcpp::List admm_bp(const arma::mat& A, const arma::colvec& b, arma::colvec& xinit,
                   const double reltol, const double abstol, const int maxiter,
                   const double rho, const double alpha) {
  // 1. get parameters
  const int n = A.n_cols;

  // 2. set ready
  arma::colvec x(n,fill::zeros);
  arma::colvec z(n,fill::zeros);
  arma::colvec u(n,fill::zeros);
  arma::colvec zold(n,fill::zeros);
  arma::colvec x_hat(n,fill::zeros);

  // 3. precompute static variables for x-update
  arma::vec n1s(n,fill::ones);
  arma::mat AAt  = A*A.t();
  arma::mat P    = diagmat(n1s) - A.t()*solve(AAt,A);
  arma::colvec q = A.t()*solve(AAt,b);

  // 4. iteration
  arma::vec h_objval(maxiter,fill::zeros);
  arma::vec h_r_norm(maxiter,fill::zeros);
  arma::vec h_s_norm(maxiter,fill::zeros);
  arma::vec h_eps_pri(maxiter,fill::zeros);
  arma::vec h_eps_dual(maxiter,fill::zeros);

  double sqrtn = std::sqrt(static_cast<float>(n));
  int k;
  for (k=0;k<maxiter;k++) {
    // 4-1. update 'x'
    x = P*(z-u) + q;

    // 4-2. update 'z' with relaxation
    zold = z;
    x_hat = alpha*x + (1 - alpha)*zold;
    z = bp_shrinkage(x_hat + u, 1/rho);
    u = u + (x_hat - z);

    // 4-3. dianostics, reporting
    h_objval(k) = norm(x,1);
    h_r_norm(k) = norm(x-z);
    h_s_norm(k) = norm(-rho*(z-zold));
    if (norm(x)>norm(-z)) {
      h_eps_pri(k) = sqrtn*abstol + reltol*norm(x);
    } else {
      h_eps_pri(k) = sqrtn*abstol + reltol*norm(-z);
    }
    h_eps_dual(k) = sqrtn*abstol + reltol*norm(rho*u);

    // 4-4. termination
    if ((h_r_norm(k) < h_eps_pri(k))&&(h_s_norm(k)<h_eps_dual(k))) {
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
  
  // 6. return results
  return(output);
}

// ===========================================================
// admm_lad: LAD via ADMM
arma::colvec lad_shrinkage(arma::colvec a, const double kappa){
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

/*
* LAD via ADMM (from Stanford)
* http://stanford.edu/~boyd/papers/pdf/admm_distr_stats.pdf
* page 19 : section 3.3.1 : stopping criteria part (3.12).
*/
Rcpp::List admm_lad(const arma::mat& A, const arma::colvec& b, arma::colvec& xinit,
                    const double reltol, const double abstol, const int maxiter,
                    const double rho, const double alpha) {
  // 1. get parameters
  const int m = A.n_rows;
  const int n = A.n_cols;

  // 2. set ready
  arma::colvec x(n,fill::zeros);
  arma::colvec z(m,fill::zeros);
  arma::colvec u(m,fill::zeros);
  arma::colvec zold(m,fill::zeros);
  arma::colvec Ax_hat(m,fill::zeros);

  // 3. precompute static variables for x-update
  arma::mat R = arma::chol(A.t()*A);

  // 4. iteration
  arma::vec h_objval(maxiter,fill::zeros);
  arma::vec h_r_norm(maxiter,fill::zeros);
  arma::vec h_s_norm(maxiter,fill::zeros);
  arma::vec h_eps_pri(maxiter,fill::zeros);
  arma::vec h_eps_dual(maxiter,fill::zeros);

  double sqrtn = std::sqrt(static_cast<float>(n));
  double sqrtm = std::sqrt(static_cast<float>(m));
  arma::vec compare3(3,fill::zeros);
  int k;
  for (k=0;k<maxiter;k++) {
    // 4-1. update 'x'
    x = solve(trimatu(R),solve(trimatl(R.t()),A.t()*(b+z-u)));

    // 4-2. update 'z' with relaxation
    zold = z;
    Ax_hat = alpha*A*x + (1-alpha)*(zold + b);
    z = lad_shrinkage(Ax_hat - b + u, 1/rho);
    u = u + (Ax_hat - z - b);

    // 4-3. dianostics, reporting
    h_objval(k) = norm(x,1);
    h_r_norm(k) = norm(A*x-z-b);
    h_s_norm(k) = norm(-rho*A.t()*(z-zold));

    compare3(0) = norm(A*x);
    compare3(1) = norm(-z);
    compare3(2) = norm(b);

    h_eps_pri(k) = sqrtm*abstol + reltol*max(compare3);
    h_eps_dual(k) = sqrtn*abstol + reltol*norm(rho*A.t()*u);

    // 4-4. termination
    if ((h_r_norm(k) < h_eps_pri(k))&&(h_s_norm(k)<h_eps_dual(k))) {
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

// ==========================================================
// admm_rpca
arma::vec shrink_vec_rpca(arma::vec x, double tau) {
  const int n = x.n_elem;
  arma::vec output(n,fill::zeros);
  double xij    = 0.0;
  double absxij = 0.0;
  double signer = 0.0;
  for (int i=0;i<n;i++){
    xij = x(i);
    if (xij >= 0){
      signer = 1.0;
      absxij = xij;
    } else {
      signer = 1.0;
      absxij = -xij;
    }
    if (absxij > tau){
      output(i) = signer*(absxij-tau);
    }
  }
  return(output);
}

arma::mat shrink_mat_rpca(arma::mat A, const double tau) {
  const int n = A.n_rows;
  const int p = A.n_cols;
  arma::mat output(n,p,fill::zeros);
  double zij    = 0.0;
  double abszij = 0.0;
  double signer = 0.0;
  for (int i=0;i<n;i++){
    for (int j=0;j<p;j++){
      zij = A(i,j);
      if (zij >= 0){
        signer = 1.0;
        abszij = zij;
      } else {
        signer = -1.0;
        abszij = -zij;
      }

      if (abszij > tau){
        output(i,j) = signer*(abszij-tau);
      }
    }
  }
  return(output);
}

arma::mat rpca_vectorpadding(arma::vec x, const int n, const int p) {
  arma::mat output(n,p,fill::zeros);
  if (n<p){
    for (int i=0;i<n;i++){
      output(i,i) = x(i);
    }
  } else {
    for (int j=0;j<p;j++){
      output(j,j) = x(j);
    }
  }
  return(output);
}

// ADMM for PCA
Rcpp::List admm_rpca(const arma::mat& M, const double tol, const int maxiter,
                     double mu, double lambda) {
  // 1. get parameters
  const int n1 = M.n_rows;
  const int n2 = M.n_cols;

  double invmu = 1/mu;
  double lbdmu = lambda/mu;

  // 2. set updating objects
  arma::mat Lold(n1,n2,fill::zeros);
  arma::mat Lnew(n1,n2,fill::zeros);
  arma::mat Sold(n1,n2,fill::zeros);
  arma::mat Snew(n1,n2,fill::zeros);
  arma::mat Yold(n1,n2,fill::zeros);
  arma::mat Ynew(n1,n2,fill::zeros);

  arma::mat costL(n1,n2,fill::zeros);
  arma::mat costS(n1,n2,fill::zeros);
  arma::mat costY(n1,n2,fill::zeros);
  arma::mat spadding(n1,n2,fill::zeros);

  arma::mat svdU;
  arma::vec svds;
  arma::mat svdV;
  arma::vec vecshrinkage;

  // 3. iteration records
  arma::vec vectolerance(maxiter,fill::zeros);

  // 4. main iteration
  int k=0;
  double norm1 = 0.0;                 // error LHS term
  double norm2 = arma::norm(M,"fro"); // error RHS term
  double normratio = 0.0;
  for (k=0;k<maxiter;k++){
    //  4-1. update L
    costL = (M - Sold + Yold*invmu);                   // compute term to be decomposed
    arma::svd(svdU, svds, svdV, costL);                // svd decomposition
    vecshrinkage = shrink_vec_rpca(svds, invmu);       // do shrinkage on singular vector
    spadding = rpca_vectorpadding(vecshrinkage,n1,n2); // we need zero padding on diagmat one
    Lnew = svdU*spadding*svdV.t();                     // update L

    // 4-2. update S
    costS = (M-Lnew+Yold*invmu);                // compute term to be shrinked
    Snew  = shrink_mat_rpca(costS, lbdmu);        // update S

    // 4-3. update Y
    Ynew  = Yold + mu*(M-Lnew-Snew);

    // 4-4. compute error
    norm1     = arma::norm(M-Lnew-Snew,"fro");
    normratio = norm1/norm2;
    vectolerance(k) = normratio;

    // 4-5. updating and termination
    Lold = Lnew;
    Sold = Snew;
    Yold = Ynew;

    if (normratio < tol){
      break;
    }
  }

  // 5. report results
  List output;
  output["L"]      = Lold;
  output["S"]      = Sold;
  output["k"]      = k;
  output["errors"] = vectolerance;
  
  /* Return output      */
  return(output);
}

// ==========================================================
// admm_spca: ADMM for sparse PCA
arma::vec spca_gamma(arma::vec sigma, double r) {
  const int p = sigma.n_elem;
  int indj = 0;
  double term1 = 0.0;
  double term2 = 0.0;
  for (int j=0;j<p;j++){
    term1 = sigma(j);
    for (int k=j;k<p;k++){
      term2 += sigma(k);
    }
    term2 = (term2-r)/(p-j);
    if (term1 > term2){
      indj = j;
      break;
    }
  }
  double theta = 0.0;
  for (int j=indj;j<p;j++){
    theta += sigma(j);
  }
  theta = (theta-r)/(p-indj);

  arma::vec output(p,fill::zeros);
  for (int i=0;i<p;i++){
    term1 = sigma(i)-theta;
    if (term1>0){
      output(i) = term1;
    }
  }
  return(output);
}

arma::mat spca_shrinkage(arma::mat A, const double tau) {
  const int n = A.n_rows;
  arma::mat output(n,n,fill::zeros);
  double zij    = 0.0;
  double abszij = 0.0;
  double signer = 0.0;
  for (int i=0;i<n;i++){
    for (int j=0;j<n;j++){
      zij = A(i,j);
      if (zij >= 0){
        signer = 1.0;
        abszij = zij;
      } else {
        signer = -1.0;
        abszij = -zij;
      }

      if (abszij > tau){
        output(i,j) = signer*(abszij-tau);
      }
    }
  }
  return(output);
}

// ADMM for sparse PCA
Rcpp::List admm_spca(const arma::mat& Sigma, const double reltol, const double abstol,
                     const int maxiter, double mu, double rho) {
  // 1. get parameters
  int p = Sigma.n_cols;

  // 2. set updating objects
  arma::mat Xold(p,p,fill::zeros);
  arma::mat Xnew(p,p,fill::zeros);
  arma::mat Yold(p,p,fill::zeros);
  arma::mat Ynew(p,p,fill::zeros);
  arma::mat Lold(p,p,fill::zeros);
  arma::mat Lnew(p,p,fill::zeros);

  arma::mat costX(p,p,fill::zeros);
  arma::mat costY(p,p,fill::zeros);
  arma::vec eigval(p,fill::zeros);   // for EVD of costX
  arma::mat eigvec(p,p,fill::zeros);

  // 3. iteration records
  arma::vec h_r_norm(maxiter,fill::zeros);
  arma::vec h_s_norm(maxiter,fill::zeros);
  arma::vec h_eps_pri(maxiter,fill::zeros);
  arma::vec h_eps_dual(maxiter,fill::zeros);

  // 4. main iteration
  int k=0;
  double ythr = mu*rho;
  double normX = 0.0;
  double normY = 0.0;
  for (k=0;k<maxiter;k++){
    // 4-1. update 'X'
    costX = Yold + mu*Lold + mu*Sigma;
    eig_sym(eigval, eigvec, costX);
    arma::vec gamma = spca_gamma(eigval, 1.0);
    Xnew = eigvec*arma::diagmat(gamma)*eigvec.t();

    // 4-2. update 'Y'
    costY = Xnew-mu*Lold;
    Ynew  = spca_shrinkage(costY, ythr);

    // 4-3. update 'L'
    Lnew  = Lold - (Xnew-Ynew)/mu;

    // 4-4. diagnostics for reporting
    h_r_norm(k) = arma::norm(Xnew-Ynew,"fro");
    h_s_norm(k) = arma::norm(Yold-Ynew,"fro")/mu;

    normX = arma::norm(Xnew,"fro");
    normY = arma::norm(Ynew,"fro");
    if (normX >= normY){
      h_eps_pri(k) = p*abstol + reltol*normX;
    } else {
      h_eps_pri(k) = p*abstol + reltol*normY;
    }
    h_eps_dual(k) = p*abstol + reltol*arma::norm(Lnew,"fro");

    // 4-5. updating and termination
    Xold = Xnew;
    Yold = Ynew;
    Lold = Lnew;

    if ((h_r_norm(k) < h_eps_pri(k))&&(h_s_norm(k)<h_eps_dual(k))){
      break;
    }
  }

  // 5. report results
  List output;
  output["X"]        = Xold;             // coefficient function
  output["k"]        = k;                // number of iterations
  output["r_norm"]   = h_r_norm;
  output["s_norm"]   = h_s_norm;
  output["eps_pri"]  = h_eps_pri;
  output["eps_dual"] = h_eps_dual;
  
  /* Return output      */
  return(output);
}

// ===========================================================
// admm_sdp: ADMM for sparse positive definite matrix factorization
arma::mat sdp_evdplus(arma::mat& X) {
  int n = X.n_rows;
  arma::vec eigval;
  arma::mat eigvec;
  arma::eig_sym(eigval, eigvec, X);
  for (int i=0;i<n;i++) {
    if (eigval(i) < 0) {
      eigval(i) = 0.0;
    }
  }
  arma::mat output = eigvec*arma::diagmat(eigval)*arma::trans(eigvec);
  return(output);
}

double sdp_gap(arma::vec& b, arma::vec& y, arma::mat& C, arma::mat& X) {
  double term1 = (std::abs(arma::dot(b,y) - arma::dot(C,X)));
  double term2 = 1.0 + std::abs(arma::dot(b,y)) + arma::dot(C,X);
  return(term1/term2);
}

// ADMM for sparse positive definite matrix factorization
Rcpp::List admm_sdp(arma::mat& C, arma::field<arma::mat>& listA, arma::vec b, double mymu,
                    double myrho, double mygamma, int maxiter, double abstol, bool printer) {
  // parameters
  unsigned int n = C.n_rows;
  unsigned int m = b.n_elem;

  double mu    = mymu;
  double rho   = myrho;
  double gamma = mygamma;

  // initialize
  arma::mat Xold(n,n,fill::eye);
  arma::mat Xnew(n,n,fill::zeros);
  arma::mat Xtmp(n,n,fill::zeros);
  arma::mat Sold(n,n,fill::zeros);
  arma::mat Snew(n,n,fill::zeros);
  arma::mat Vold(n,n,fill::zeros);
  arma::mat Vnew(n,n,fill::zeros);
  arma::vec yold(m,fill::zeros);
  arma::vec ynew(m,fill::zeros);
  arma::vec ytmp(m,fill::zeros);

  // preliminary computation
  // A, (AA^T)^{-1}
  arma::mat A(m,n*n,fill::zeros);
  for (int i=0; i<(int)m; i++) {
    A.row(i) = arma::vectorise(listA(i)).t();
  }
  arma::mat AAinv = arma::pinv(A*A.t());

  // main iteration
  arma::vec h_objval(maxiter,fill::zeros);
  arma::vec h_primal(maxiter,fill::zeros);
  arma::vec h_dual(maxiter,fill::zeros);
  arma::vec h_gap(maxiter,fill::zeros);
  arma::vec hvec(3,fill::zeros);


  int it;
  for (it=0;it<maxiter;it++) {
    // 1. Update
    // 1-1. update y
    ynew = -AAinv*(mu*(A*arma::vectorise(Xold)-b) + A*arma::vectorise(Sold-C));
    // 1-2. update S
    Vnew = C - mu*Xold;
    for (int i=0; i<(int)m; i++){
      Vnew = Vnew - ynew(i)*listA(i);
    }
    Snew = sdp_evdplus(Vnew);
    // 1-3. update X
    Xtmp = (1.0/mu)*(Snew-Vnew);
    Xnew = (1.0-rho)*Xold + rho*Xtmp;

    // 2. Termination Rules
    Xtmp = C-Snew;
    for (int i=0; i<(int)m; i++){
      Xtmp -= ynew(i)*listA(i);
    }
    h_objval(it) = arma::dot(C,Xnew);
    h_primal(it) = arma::norm(A*arma::vectorise(Xnew)-b,2);
    h_dual(it)   = arma::norm(Xtmp,"fro");
    h_gap(it)    = sdp_gap(b,ynew,C,Xnew);

    hvec(0) = h_primal(it);
    hvec(1) = h_dual(it);
    hvec(2) = h_gap(it);

    // 3. Penalty update (very simple way)
    if (hvec(0) < hvec(1)){
      mu *= gamma;
    } else {
      mu /= gamma;
    }

    // 4. do the stopping
    Xold = Xnew;
    Sold = Snew;
    Vold = Vnew;
    yold = ynew;
    if (printer==true){
      Rcpp::Rcout << "* admm.sdp : iteration " << it << "/" << maxiter << " complete.." << std::endl;
    }
    if (hvec.max() < abstol){
      break;
    }
  }

  // 5. report results
  List output;
  output["X"]        = Xold;
  output["objval"]   = h_objval.head(it);
  output["eps_pri"]  = h_primal.head(it);
  output["eps_dual"] = h_dual.head(it);
  output["gap"]      = h_gap.head(it);
  
  /* Return output      */
  return(output);
}









