// Utility routines

// Authors:
//           Bernardi Mauro, University of Padova
//           Last update: July 12, 2022

// List of implemented methods:

// [[Rcpp::depends(RcppArmadillo)]]
#include "MMutils.h"

/**
 * Extend division reminder to vectors
 *
 * @param   a       Dividend
 * @param   n       Divisor
 */
int mod(int a, int n) {
    return a - floor(a/n)*n;
}

arma::vec fast_XTX_plus_D_update(arma::mat mXX, arma::vec vXy, arma::vec omega) {
    
  /* ::::::::::::::::::::::::::::::::::::::::::::::
   variable declaration                                  */
  int p = 0;
  
  /* ::::::::::::::::::::::::::::::::::::::::::::::
   get dimensions                                  */
  p = mXX.n_cols;
  
  /* ::::::::::::::::::::::::::::::::::::::::::::::
   vector and matrices declaration                       */
  arma::mat mOmega(p, p);               mOmega.zeros();
  arma::mat mSig(p, p);                 mSig.zeros();
  arma::mat mQ(p, p);                   mQ.zeros();
  arma::mat mR(p, p);                   mR.zeros();
  arma::vec b(p);                       b.zeros();
  arma::vec mu_star(p);                 mu_star.zeros();
  arma::mat sigma_star(p, p);           sigma_star.zeros();
  arma::vec res(p);                     res.zeros();
  
  /* :::::::::::::::::::::::::::::::::::::::::
   Get relevant quantities                      */
  mOmega = diagmat(omega);
  mSig   = mXX + mOmega;
  
  /* :::::::::::::::::::::::::::::::::::::::::
   Perform QR update of XTX + D                      */
  qr_econ(mQ, mR, mSig);
  b = mQ.t() * vXy;
  
  /* :::::::::::::::::::::::::::::::::::::::::
   Compute the vector of regression parameters     */
  res = solve(trimatu(mR), b);  // enable fast mode
  
  /* ::::::::::::::::::::::::::::::::::::::::::::::
   return output                                        */
  return(res);
}

arma::mat fast_XTX_plus_D_inversion(arma::mat mXX, arma::vec omega) {
    
  /* ::::::::::::::::::::::::::::::::::::::::::::::
   variable declaration                                  */
  int p = 0;
  
  /* ::::::::::::::::::::::::::::::::::::::::::::::
   get dimensions                                  */
  p = mXX.n_cols;
  
  /* ::::::::::::::::::::::::::::::::::::::::::::::
   vector and matrices declaration                       */
  arma::mat mOmega(p, p);               mOmega.zeros();
  arma::mat mSig(p, p);                 mSig.zeros();
  arma::mat mQ(p, p);                   mQ.zeros();
  arma::mat mR(p, p);                   mR.zeros();
  arma::mat sigma_star(p, p);           sigma_star.zeros();
  
  /* :::::::::::::::::::::::::::::::::::::::::
   Get relevant quantities                      */
  mOmega = diagmat(omega);
  mSig   = mXX + mOmega;
  
  /* :::::::::::::::::::::::::::::::::::::::::
   Perform QR update of XTX + D                      */
  qr_econ(mQ, mR, mSig);
  
  /* :::::::::::::::::::::::::::::::::::::::::
   Compute the inverse of Sig     */
  sigma_star = solve(trimatu(mR), mQ.t());  // enable fast mode
  
  /* ::::::::::::::::::::::::::::::::::::::::::::::
   return output                                        */
  return(sigma_star);
}

arma::vec LASSO_MM_Woodbury_update(arma::mat mX, arma::vec vXy, arma::vec omega, double lambda, unsigned int n, unsigned int p) {
  
  /* ::::::::::::::::::::::::::::::::::::::::::::::
   vector and matrices declaration                       */
  arma::mat mXX(n, n);                    mXX.zeros();
  arma::mat mX_(p, n);                     mX_.zeros();
  arma::mat mSig(p, p);                   mSig.zeros();
  arma::mat mQ(p, p);                     mQ.zeros();
  arma::mat mD(p, p);                     mD.zeros();
  arma::mat mR(p, p);                     mR.zeros();
  arma::mat mOmega_INV(n, n);             mOmega_INV.zeros();
  arma::mat XWX(p, p);                    XWX.zeros();
  arma::mat mB(p, p);                     mB.zeros();
  arma::mat mEye_p(p, p);                 mEye_p.eye();
  arma::mat mEye_n(n, n);                 mEye_n.eye();
  arma::vec vRegP_update(p);              vRegP_update.zeros();
  
  /* :::::::::::::::::::::::::::::::::::::::::
   Get relevant quantities                      */
  mX_  = mX * diagmat(sqrt(omega));
  mXX  = mX_ * mX_.t();
  mSig = mXX + lambda * mEye_n;
  
  /* :::::::::::::::::::::::::::::::::::::::::
   Perform QR update of XTX + D                      */
  qr_econ(mQ, mR, mSig);
  
  /* :::::::::::::::::::::::::::::::::::::::::
   Compute the inverse of Sig                          */
  mOmega_INV = solve(trimatu(mR), mQ.t());  // enable fast mode
  
  /* :::::::::::::::::::::::::::::::::::::::::
   Update the vector of parameters                          */
  mD           = diagmat(omega);
  XWX          = mX.t() * mOmega_INV * mX;
  mB           = mD * (mEye_p - XWX * mD) / lambda;
  vRegP_update = mB * vXy;
  
  /* ::::::::::::::::::::::::::::::::::::::::::::::
   return output                                        */
  return(vRegP_update);
}

List lmridge(arma::mat mX, arma::vec vY, arma::vec lambda, unsigned int n, unsigned int p) {
    
  unsigned int nlambda;
  List out;
  
  /* ::::::::::::::::::::::::::::::::::::::::::::::
   get dimensions                                      */
  nlambda = lambda.n_elem;
  
  /* ::::::::::::::::::::::::::::::::::::::::::::::
   vector and matrices declaration                       */
  arma::mat mXX(p, p);                    mXX.zeros();
  arma::vec vXy(p);                       vXy.zeros();
  arma::vec vRegP(p);                     vRegP.zeros();
  arma::mat mRegP(p, nlambda);            mRegP.zeros();
  arma::vec lambdavec(p);                 lambdavec.zeros();
  arma::vec eigval(p);                    eigval.zeros();
  arma::mat eigvec(p, p);                 eigvec.zeros();
  arma::vec vUnit_p(p);                   vUnit_p.ones();
  arma::mat mOmega_INV(p, p);             mOmega_INV.zeros();
  arma::vec vRes(n);                      vRes.zeros();
  arma::vec vRes2(n);                     vRes2.zeros();
  arma::vec ess(nlambda);                 ess.zeros();
  
  /* :::::::::::::::::::::::::::::::::::::::::
   Get relevant quantities                              */
  mXX = mX.t() * mX / (double)n;
  vXy = mX.t() * vY / (double)n;
  
  /* :::::::::::::::::::::::::::::::::::::::::
   Perform spectral decomposition of XTX                */
  eig_sym(eigval, eigvec, mXX);
  vRegP = eigvec.t() * vXy;
  
  /* :::::::::::::::::::::::::::::::::::::::::
   Compute the inverse                          */
  for (unsigned int jt=0; jt<nlambda; jt++) {
    lambdavec     = lambda(jt) * vUnit_p;
    mOmega_INV    = diagmat(1.0 / (lambdavec + eigval));
    mRegP.col(jt) = eigvec * mOmega_INV * vRegP;
    vRes          = vY - mX * mRegP.col(jt);
    vRes2         = pow(vRes, 2);
    ess(jt)       = sum(vRes2) / (double)n;
  }
  
  /* ::::::::::::::::::::::::::::::::::::::::::::::
   get output                                           */
  out["coeff"] = mRegP;
  out["ess"]   = ess;
  
  /* ::::::::::::::::::::::::::::::::::::::::::::::
   return output                                        */
  return out;
}
