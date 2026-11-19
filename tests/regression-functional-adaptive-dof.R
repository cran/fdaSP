# Regression checks for normalized adaptive DoF and coordinate weights.
# Fixed lambda, full-rank design, away from active-set transitions.
# Weight estimation is repeated at each response perturbation.
library(fdaSP)

set.seed(990);n<-24;M<-6;p<-21
H<-splines::bs(seq(0,1,length.out=p),df=M,intercept=TRUE)
W<-matrix(rnorm(n*M),n,M); W[,2]<-W[,2]+.3*W[,1]
Xfun<-W%*%solve(crossprod(H),t(H))
Z<-cbind(1,seq(-1,1,length.out=n));y0<-drop(W%*%c(2,-1,1.5,.8,1,-.7)+Z%*%c(.6,.4)+rnorm(n,sd=.03))
ctl<-list(maxit=30000,abstol=1e-10,reltol=1e-10,print.out=FALSE)
rows<-list()
for(ad in c(FALSE,TRUE)) for(vtype in c('unit','nonunit')) for(sm in c(FALSE,TRUE)) for(z in c(FALSE,TRUE)) {
 args<-list(mX=Xfun,M=M,weights_adaptive=ad,standardize.data=FALSE,lambda1=c(.02,.01),control=ctl)
 if(z)args$mZ<-Z
 if(vtype=='unit')args$var_weights<-rep(1,M)
 if(sm){args$lambda2<-.015;args$diff_order<-2}
 ff<-function(yy)suppressWarnings(do.call(f2sSP,c(list(vY=yy),args)))
 f<-ff(y0)
 fit<-function(f){b<-matrix(f$sp.coef.path,nrow=2,ncol=M);a<-W%*%t(b);if(z)a<-a+Z%*%t(matrix(f$coef.path,nrow=2,ncol=2));a}
 num<-c(0,0);eps<-1e-4
 for(i in seq_len(n)){a<-b<-y0;a[i]<-a[i]+eps;b[i]<-b[i]-eps;num<-num+(fit(ff(a))[i,]-fit(ff(b))[i,])/(2*eps)}
 # independent KKT differentiation on all-active example
 b<-matrix(f$sp.coef.path,nrow=2,ncol=M)[2,];G<-f$GRmat;v<-f$var_weights;w<-f$group_weights
 D<-if(z)cbind(Z,W) else W;q<-if(z)2 else 0;Q<-crossprod(D)/n
 L<-solve(crossprod(D),t(D));pilot<-drop(L%*%y0)[q+seq_len(M)];Lp<-L[q+seq_len(M),,drop=FALSE]
 P<-matrix(0,M,M);J<-matrix(0,M,n);grad<-rep(0,M)
 for(g in seq_len(nrow(G))){ii<-which(G[g,]==1);r<-sqrt(sum((v[ii]*b[ii])^2));vv<-v[ii]^2*b[ii];P[ii,ii]<-P[ii,ii]+w[g]*(diag(v[ii]^2,length(ii))-tcrossprod(vv)/r^2)/r;grad[ii]<-grad[ii]+w[g]*vv/r
 if(ad){dw<--drop(crossprod(pilot[ii],Lp[ii,,drop=FALSE]))/sqrt(sum(pilot[ii]^2))^3;J[ii,]<-J[ii,,drop=FALSE]+tcrossprod(vv/r,dw)}}
 Hess<-Q;ix<-q+seq_len(M);Hess[ix,ix]<-Hess[ix,ix]+.01*P
 if(sm){DD<-diff(diag(M),differences=2);Hess[ix,ix]<-Hess[ix,ix]+.015*crossprod(DD);grad<-grad+(.015/.01)*drop(crossprod(DD,DD%*%b))}
 J<-rbind(matrix(0,q,n),J)
 reference<-sum(diag(D%*%solve(Hess,t(D)/n-.01*J)))
 resid<-y0-W%*%b;if(z)resid<-qr.resid(qr(Z),resid)
 kkt<-max(abs(drop(crossprod(W,resid))/n-.01*grad))
 row<-data.frame(adaptive=ad,weights=vtype,smooth=sm,Z=z,reported=as.numeric(f$dof)[2],numeric=num[2],independent=reference,kkt=kkt,error=as.numeric(f$dof)[2]-num[2]);print(row);rows[[length(rows)+1]]<-row
 stopifnot(all(is.finite(num)), abs(row$error)<2e-5,
           abs(reference-num[2])<2e-5, kkt<1e-6)
 # The two linear adaptive grouped kernels share the derivative formula.
 # All groups are active here, so the inactive-group dual certificate is unused.
 if(ad && !sm) {
   a<-list(lambda=.01, GRmat=G, group_weights=w, var_weights=v,
           Uvec=rep(0,sum(G)),err_primal=1e-12,err_dual=1e-12,
           rho=1,toler_c=5,toler_d=.001)
   if(z){a$X<-W;a$Z<-Z;a$coeff_X<-b;a$coeff_X_LS<-pilot
     fun<-getFromNamespace(".lm_cov_adaptive_dof_OVGLASSO_1lamba","fdaSP")
   }else{a$X<-W;a$coeff<-b;a$coeff_LS<-pilot
     fun<-getFromNamespace(".lm_adaptive_dof_OVGLASSO_1lambda","fdaSP")}
   linear<-do.call(fun,a)$dof
   stopifnot(abs(linear-num[2])<2e-5)
 }

}
cat("All 16 functional configurations and four linear grouped kernels passed.\n")
