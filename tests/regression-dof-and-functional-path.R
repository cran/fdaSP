# Regression tests for DoF argument forwarding and loss normalization.
# Compare against a central finite-difference trace away from transitions.
library(fdaSP)
for (design in c("orthogonal","correlated")) {
 for (with_z in c(FALSE,TRUE)) {
set.seed(412)
n<-24;p<-6;X<-qr.Q(qr(matrix(rnorm(n*p),n,p)))*sqrt(n)
if (design=='correlated') X[,2]<-.5*X[,1]+X[,2]
useZ<-with_z
Z<-if(useZ) matrix(1,n,1) else NULL
y<-drop(X%*%c(2,-1,1.5,.8,.4,-.6))+rnorm(n,sd=.01)
ctl<-list(maxit=10000,abstol=1e-9,reltol=1e-9,print.out=FALSE)
getfit<-function(y,method,adaptive=FALSE) suppressWarnings(lmSP(X=X,Z=Z,y=y,penalty=method,adaptive=adaptive,groups=rep(1:3,each=2),lambda=c(.12,.08),standardize.data=FALSE,control=ctl))
checks<-list()
for (case in c('LASSO','GLASSO','Adaptive Lasso')) {
 cat("Testing",case,"\n")
 method<-if(case=='Adaptive Lasso')'LASSO' else case
 adaptive<-case=='Adaptive Lasso'
 fit<-getfit(y,method,adaptive);h<-1e-4
 deriv<-sapply(seq_len(n),function(i){yp<-ym<-y;yp[i]<-yp[i]+h;ym[i]<-ym[i]-h
 fp<-getfit(yp,method,adaptive);fm<-getfit(ym,method,adaptive)
 bp<-fp$sp.coef.path[2,];bm<-fm$sp.coef.path[2,]
 value<-sum(X[i,]*(bp-bm))
 if(useZ) value<-value+sum(Z[i,]*(fp$coef.path[2,]-fm$coef.path[2,]))
 value/(2*h)})
 checks[[case]]<-data.frame(method=case,reported=fit$dof[2],finite_difference=sum(deriv),error=abs(fit$dof[2]-sum(deriv)),converged=all(fit$convergence==1))
}
checks<-do.call(rbind,checks);print(checks)
stopifnot(all(checks$error<1e-3),all(checks$converged))

 }
}

# Each returned functional path slice must match its coefficient slice.
set.seed(733)
grid <- seq(0,1,length.out=21)
H <- splines::bs(grid,df=5,intercept=TRUE)
Theta <- splines::bs(grid,df=4,intercept=TRUE)
Xfun <- matrix(rnorm(30*5),30,5) %*% t(H)/21
Psi <- matrix(seq_len(20)/20,5,4)
Yfun <- Xfun %*% H %*% Psi %*% t(Theta)
f <- f2fSP(mY=Yfun,mX=Xfun,M=5,L=4,lambda=c(.01,.001),
           standardize.data=FALSE,control=list(maxit=3000,print.out=FALSE))
for (k in seq_along(f$lambda)) {
 stopifnot(max(abs(f$sp.fun.path[k,,]-H%*%f$sp.coef.path[k,,]%*%t(Theta)))<1e-10)
}
