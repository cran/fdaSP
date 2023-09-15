# Authors: Mauro Bernardi 
#          Department Statistical Sciences
#       	 University of Padova
#      	   Via Cesare Battisti, 241
#      	   35121 PADOVA, Italy
#          E-mail: mauro.bernardi@unipd.it  

# Last change: March 16, 2023

#' Overlap Group Least Absolute Shrinkage and Selection Operator for function-on-function regression model
#'
#' Overlap Group-LASSO for function-on-function regression model solves the following optimization problem
#' \deqn{\textrm{min}_{\psi} ~ \frac{1}{2} \sum_{i=1}^n \int \left( y_i(s) - \int x_i(t) \psi(t,s) dt \right)^2 ds + \lambda \sum_{g=1}^{G} \Vert S_{g}T\psi \Vert_2}
#' to obtain a sparse coefficient vector \eqn{\psi=\mathsf{vec}(\Psi)\in\mathbb{R}^{ML}} for the functional penalized predictor \eqn{x(t)}, where the coefficient matrix \eqn{\Psi\in\mathbb{R}^{M\times L}}, 
#' the regression function \eqn{\psi(t,s)=\varphi(t)^\intercal\Psi\theta(s)}, 
#' \eqn{\varphi(t)} and \eqn{\theta(s)} are two B-splines bases of order \eqn{d} and dimension \eqn{M} and \eqn{L}, respectively. For each group \eqn{g}, each row of 
#' the matrix \eqn{S_g\in\mathbb{R}^{d\times ML}} has non-zero entries only for those bases belonging 
#' to that group. These values are provided by the arguments \code{groups} and \code{group_weights} (see below). 
#' Each basis function belongs to more than one group. The diagonal matrix \eqn{T\in\mathbb{R}^{ML\times ML}} contains 
#' the basis-specific weights. These values are provided by the argument \code{var_weights} (see below).
#' The regularization path is computed for the overlap group-LASSO penalty at a grid of values for the regularization 
#' parameter \eqn{\lambda} using the alternating direction method of multipliers (ADMM). See Boyd et al. (2011) and Lin et al. (2022) 
#' for details on the ADMM method.
#'  
#' @param mX an \eqn{(n\times r_x)} matrix of observations of the functional covariate.
#' @param mY an \eqn{(n\times r_y)} matrix of observations of the functional response variable.
#' @param L number of elements of the B-spline basis vector \eqn{\theta(s)}.
#' @param M number of elements of the B-spline basis vector \eqn{\varphi(t)}.
#' @param group_weights a vector of length \eqn{G} containing group-specific weights. The default is square root of the group cardinality, see Bernardi et al. (2022).
#' @param var_weights a vector of length \eqn{ML} containing basis-specific weights. The default is a vector where 
#' each entry is the reciprocal of the number of groups including that basis. See Bernardi et al. (2022) for details.
#' @param standardize.data logical. Should data be standardized?
#' @param splOrd the order \eqn{d} of the spline basis.
#' @param lambda either a regularization parameter or a vector of regularization parameters. 
#' In this latter case the routine computes the whole path. If it is NULL values for lambda are provided by the routine.
#' @param lambda.min.ratio smallest value for lambda, as a fraction of the maximum lambda value. If \eqn{nr_y>LM}, the default is 0.0001, and if \eqn{nr_y<LM}, the default is 0.01. 
#' @param nlambda the number of lambda values - default is 30.
#' @param overall.group logical. If it is TRUE, an overall group including all penalized covariates is added.
#' @param control	 a list of control parameters for the ADMM algorithm. See ‘Details’.
#'
#' @return A named list containing \describe{
#' \item{sp.coefficients}{an \eqn{(M\times L)} solution matrix for the parameters \eqn{\Psi}, which corresponds to the minimum in-sample MSE.}
#' \item{sp.coef.path}{an \eqn{(n_\lambda\times M \times L)} array of estimated \eqn{\Psi} coefficients for each lambda.}
#' \item{sp.fun}{an \eqn{(r_x\times r_y)} matrix providing the estimated functional coefficient for \eqn{\psi(t,s)}.}
#' \item{sp.fun.path}{an \eqn{(n_\lambda\times r_x\times r_y)} array providing the estimated functional coefficients for \eqn{\psi(t,s)} for each lambda.}
#' \item{lambda}{sequence of lambda.}
#' \item{lambda.min}{value of lambda that attains the minimum in-sample MSE.}
#' \item{mse}{in-sample mean squared error.}
#' \item{min.mse}{minimum value of the in-sample MSE for the sequence of lambda.}
#' \item{convergence}{logical. 1 denotes achieved convergence.}
#' \item{elapsedTime}{elapsed time in seconds.}
#' \item{iternum}{number of iterations.}
#' }
#'
#' When you run the algorithm, output returns not only the solution, but also the iteration history recording
#' following fields over iterates,
#' \describe{
#' \item{objval}{objective function value.}
#' \item{r_norm}{norm of primal residual.}
#' \item{s_norm}{norm of dual residual.}
#' \item{eps_pri}{feasibility tolerance for primal feasibility condition.}
#' \item{eps_dual}{feasibility tolerance for dual feasibility condition.}
#' }
#' Iteration stops when both \code{r_norm} and \code{s_norm} values
#' become smaller than \code{eps_pri} and \code{eps_dual}, respectively.
#' @section Details: The control argument is a list that can supply any of the following components:\describe{
#' \item{adaptation}{logical. If it is TRUE, ADMM with adaptation is performed. The default value is TRUE. See Boyd et al. (2011) for details.}
#' \item{rho}{an augmented Lagrangian parameter. The default value is 1.}
#' \item{tau.ada}{an adaptation parameter greater than one. Only needed if adaptation = TRUE. The default value is 2. See Boyd et al. (2011) and Lin et al. (2022) for details.}
#' \item{mu.ada}{an adaptation parameter greater than one. Only needed if adaptation = TRUE. The default value is 10. See Boyd et al. (2011) and Lin et al. (2022) for details.}
#' \item{abstol}{absolute tolerance stopping criterion. The default value is sqrt(sqrt(.Machine$double.eps)).}
#' \item{reltol}{relative tolerance stopping criterion. The default value is sqrt(.Machine$double.eps).}
#' \item{maxit}{maximum number of iterations. The default value is 100.}
#' \item{print.out}{logical. If it is TRUE, a message about the procedure is printed. The default value is TRUE.}
#' }
#' 
#' @examples 
#' 
#' ## generate sample data
#' set.seed(4321)
#' s  <- seq(0, 1, length.out = 100)
#' t  <- seq(0, 1, length.out = 100)
#' p1 <- 5
#' p2 <- 6
#' r  <- 10
#' n  <- 50
#' 
#' beta_basis1 <- splines::bs(s, df = p1, intercept = TRUE)    # first basis for beta
#' beta_basis2 <- splines::bs(s, df = p2, intercept = TRUE)    # second basis for beta
#' 
#' data_basis <- splines::bs(s, df = r, intercept = TRUE)    # basis for X
#' 
#' x_0   <- apply(matrix(rnorm(p1 * p2, sd = 1), p1, p2), 1, 
#'                fdaSP::softhresh, 1.5)  # regression coefficients 
#' x_fun <- beta_basis2 %*% x_0 %*%  t(beta_basis1)  
#' 
#' fun_data <- matrix(rnorm(n*r), n, r) %*% t(data_basis)
#' b        <- fun_data %*% x_fun + rnorm(n * 100, sd = sd(fun_data %*% x_fun )/3)
#' 
#' ## set the hyper-parameters
#' maxit          <- 1000
#' rho_adaptation <- FALSE
#' rho            <- 1
#' reltol         <- 1e-5
#' abstol         <- 1e-5
#' 
#' ## fit functional regression model
#' mod <- f2fSP(mY = b, mX = fun_data, L = p1, M = p2,
#'              group_weights = NULL, var_weights = NULL, standardize.data = FALSE, splOrd = 4,
#'              lambda = NULL, nlambda = 30, lambda.min.ratio = NULL, 
#'              control = list("abstol" = abstol, 
#'                             "reltol" = reltol, 
#'                             "maxit" = maxit, 
#'                             "adaptation" = rho_adaptation, 
#'                             rho = rho, 
#'              "print.out" = FALSE))
#'  
#' mycol <- function (n) {
#' palette <- colorRampPalette(RColorBrewer::brewer.pal(11, "Spectral"))
#' palette(n)
#' }
#' cols <- mycol(1000)
#' 
#' oldpar <- par(mfrow = c(1, 2))
#' image(x_0, col = cols)
#' image(mod$sp.coefficients, col = cols)
#' par(oldpar)
#' 
#' oldpar <- par(mfrow = c(1, 2))
#' image(x_fun, col = cols)
#' contour(x_fun, add = TRUE)
#' image(beta_basis2 %*% mod$sp.coefficients %*% t(beta_basis1), col = cols)
#' contour(beta_basis2 %*% mod$sp.coefficients %*% t(beta_basis1), add = TRUE)
#' par(oldpar)
#' 
#' @references
#' \insertRef{bernardi_etal.2022}{fdaSP}
#' 
#' \insertRef{boyd_etal.2011}{fdaSP}
#' 
#' \insertRef{jenatton_etal.2011}{fdaSP}
#' 
#' \insertRef{lin_etal.2022}{fdaSP}
#'
#' @export
f2fSP <- function(mY, mX, L, M, group_weights = NULL, var_weights = NULL, standardize.data = TRUE,
                  splOrd = 4, lambda = NULL, lambda.min.ratio = NULL, nlambda = 30, overall.group = FALSE,
                  control = list()) {
    
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # Get dimensions
  n <- dim(mX)[1]             # number of observations
  p <- dim(mX)[2]
  
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # Create the B-spline basis functions
  s      <- seq(0, 1, length.out = p)
  mH     <- t(splines::bs(s, df = M, intercept = TRUE))
  mTheta <- t(splines::bs(s, df = L, intercept = TRUE))
  
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # Define response and design matrix
  mY_B <- mY
  mX_B <- tcrossprod(mX, mH)
  
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # Vectorisation form
  X_ <- kronecker(t(mTheta), mX_B)
  y_ <- matrix(ks::vec(mY_B, byrow = FALSE), ncol = 1)
  
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # Standardise response and design matrix
  if (standardize.data == TRUE) {
    res   <- standardizemat(X_, y_)
    X.std <- res$X.std
    y.std <- res$y.std
    mU    <- res$mU
    mV    <- res$mV
  } else {
    X.std <- X_
    y.std <- y_
  }
  
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # Pre-processing: lambda parameter (in case of a single lambda)
  if (!is.null(lambda)) {
    if (length(lambda) == 1) {
      meps     <- (.Machine$double.eps)
      negsmall <- -meps
      if (!check_param_constant(lambda, negsmall)) {
        stop("* f2fSP : reg. parameter 'lambda' is invalid.")
      }
      if (lambda < meps){
        message("* f2fSP : since 'lambda' is effectively zero, a least-squares solution is returned.")
        xsol     <- as.vector(aux_pinv(X_) %*% matrix(y_))
        output   <- list()
        output$x <- xsol
        return(output)
      }
    }
  }
  
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # check for inputs
  npar   <- dim(X.std)[2]
  con    <- list(maxit                = 100L, 
                 abstol               = sqrt(.Machine$double.eps), 
                 reltol               = sqrt(sqrt(.Machine$double.eps)), 
                 adaptation           = TRUE, 
                 rho                  = 1, 
                 tau.ada              = 2,               
                 mu.ada               = 10,
                 par.init             = NULL,
                 init.rnd             = FALSE,
                 print.out            = TRUE)
  
  nmsC                          <- names(con)
  con[(namc <- names(control))] <- control
  if (length(noNms <- namc[!namc %in% nmsC])) {
    warning("unknown names in control: ", paste(noNms, collapse = ", "))
  }
  # set default values (for inputs)
  maxit                <- con$maxit
  abstol               <- con$abstol
  reltol               <- con$reltol
  adaptation           <- con$adaptation
  rho                  <- con$rho
  tau.ada              <- con$tau.ada      
  mu.ada               <- con$mu.ada
  par.init             <- con$par.init
  init.rnd             <- con$init.rnd
  print.out            <- con$print.out
  
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # Pre-processing: variable weights
  check_weights(x = var_weights, n = dim(X_)[2], algname = "f2fSP", funname = "var_weights")
  
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # Get the groups of the overlap group LASSO penalty
  if (overall.group == TRUE) {
    out <- penfun(method = "ogl&1", p = M, q = L, splOrd = splOrd, regpars = NULL)
  } else {
    out <- penfun(method = "ogl", p = M, q = L, splOrd = splOrd, regpars = NULL)
  }
  GRmat <- out$grMat
  G     <- dim(GRmat)[1]
  nG    <- rowSums(GRmat)
  
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # Manage group & overall group weights
  if (is.null(group_weights)) {
    # Define the weights as in Yuan and Lin 2006, JRSSB
    group_weights <- sqrt(diag(tcrossprod(GRmat)))
  } else {
    if (overall.group == TRUE) {
      check_group_weights(x = group_weights, n = G, algname = "f2fSP", funname = "group_weights")
      if (length(group_weights) == (G-1)) {
        group_weights <- c(group_weights, sqrt(dim(GRmat)[2]))
      }
    } else {
      check_weights(x = group_weights, n = G, algname = "f2fSP", funname = "group_weights")
    }
  } 
  
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # Manage variable weights
  if (is.null(var_weights)) {
    var_weights <- 1.0 / colSums(GRmat)
  } 
  
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # Get the sequence of lambda
  if (!is.null(nlambda) && is.null(lambda)) {
    # get the smallest value of such that the regression 
    # coefficients estimated by the lasso are all equal to zero
    lambda <- lm_lambdamax_OVGLASSO(X = X.std, y = y.std, 
                                    groups = NULL, GRmat = out$grMat, group_weights = group_weights, var_weights = var_weights, 
                                    lambda.min.ratio = lambda.min.ratio, maxl = nlambda)$lambda.seq
  }
  if (is.null(nlambda) && is.null(lambda)) {
    # get the smallest value of such that the regression 
    # coefficients estimated by the lasso are all equal to zero
    lambda <- lm_lambdamax_OVGLASSO(X = X.std, y = y.std, 
                                    groups = NULL, GRmat = out$grMat, group_weights = group_weights, var_weights = var_weights, 
                                    lambda.min.ratio = lambda.min.ratio, maxl = 30)$lambda.seq
  }
  if (!is.null(lambda)) {
    nlambda <- length(lambda)
  }
  
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # Store output
  mRegP    <- matrix(0, nrow = M * L, ncol = nlambda)
  vMse     <- rep(0, nlambda)
  vConv    <- rep(0, nlambda)
  vRunTime <- rep(0, nlambda)
  vIterN   <- rep(0, nlambda)
  objfun   <- vector(mode = "list", length = nlambda)
  r_norm   <- vector(mode = "list", length = nlambda)
  s_norm   <- vector(mode = "list", length = nlambda)
  err_pri  <- vector(mode = "list", length = nlambda)
  err_dual <- vector(mode = "list", length = nlambda)
  
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # Run the ADMM algorithm
  ret <- .admm_ovglasso_fast(A = X.std, b = y.std, groups = out$grMat, group_weights = group_weights, var_weights = var_weights,
                             lambda = lambda, rho_adaptation = adaptation, rho = rho, tau = tau.ada, mu = mu.ada,
                             reltol = reltol, abstol = abstol, maxiter = maxit, ping = 0)
  
  # ret <- .Call("admm_ovglasso_fast", 
  #              A = X.std, b = y.std, groups = out$grMat, group_weights = group_weights, var_weights = var_weights, 
  #              lambda = lambda, rho_adaptation = adaptation, rho = rho, tau = tau.ada, mu = mu.ada, 
  #              reltol = reltol, abstol = abstol, maxiter = maxit, ping = 0)
  
  # get estimated coefficients and path
  mSpRegP <- t(ret$coef.path)
  vSpRegP <- ret$coefficients
  
  if (standardize.data == TRUE) {
    # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    # Get the path and retrieve the scaled estimates
    sp.path <- array(dim = c(nlambda, M, L), data = t(apply(mSpRegP, 2, function(x) solve(mU) %*% x %*% mV)))
    
    # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    # Retrieve estimated parameters
    mSpRegP_ <- matrix(solve(mU) %*% vSpRegP %*% mV, nrow = M, ncol = L)
  } else {
    sp.path  <- array(dim = c(nlambda, M, L), data = t(mSpRegP))
    mSpRegP_ <- invvec(vSpRegP, L, M, byrow = FALSE)
  }
  # get the estimated functions
  # sp.fun.path <- t(mTheta) %*% sp.path %*% mH
  # sp.fun.path <- array(data = apply(sp.path, 1, function(x){t(mTheta) %*% x %*% mH}), 
  #                      dim = c(nlambda, p, p))
  sp.fun.path <- array(data = apply(sp.path, 1, function(x){t(mH) %*% x %*% mTheta}), 
                       dim = c(nlambda, p, p))
  # sp.fun      <- t(mTheta) %*% mSpRegP_ %*% mH
  sp.fun      <- t(mH) %*% mSpRegP_ %*% mTheta
  
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # Print to screen
  if (print.out == TRUE) {
    cat("\n\n\n")
    cat("Function to function regression model with overlap group-LASSO penalty\n")
    cat("running time (for ", reltol, " relative error):",
        sum(ret$elapsedTime), "seconds \n\n\n")
  }
  
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # Get output
  res.names <- c("sp.coefficients", 
                 "sp.coef.path", 
                 "sp.fun.path",
                 "sp.fun",
                 "lambda.min",
                 "lambda",
                 "mse",
                 "min.mse",
                 "convergence",
                 "elapsedTime",
                 "iternum",
                 "objfun",
                 "r_norm",
                 "s_norm",
                 "err_pri",
                 "err_dual",
                 "rho") 
  res                 <- vector(mode = "list", length = length(res.names))
  names(res)          <- res.names
  res$sp.coefficients <- mSpRegP_
  res$sp.coef.path    <- sp.path
  res$sp.fun.path     <- sp.fun.path
  res$sp.fun          <- sp.fun
  res$lambda.min      <- ret$lambda.min
  res$lambda          <- ret$lambda
  res$mse             <- ret$mse
  res$min.mse         <- ret$min.mse
  res$convergence     <- ret$convergence
  res$elapsedTime     <- ret$elapsedTime
  res$iternum         <- ret$iternum
  res$objfun          <- ret$objfun
  res$r_norm          <- ret$r_norm
  res$s_norm          <- ret$s_norm
  res$err_pri         <- ret$err_pri
  res$err_dual        <- ret$err_dual
  res$rho             <- ret$rho
 
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # Return output
  return(res)
}

#' Cross-validation for Overlap Group Least Absolute Shrinkage and Selection Operator for function-on-function regression model
#'
#' Overlap Group-LASSO for function-on-function regression model solves the following optimization problem
#' \deqn{\textrm{min}_{\psi} ~ \frac{1}{2} \sum_{i=1}^n \int \left( y_i(s) - \int x_i(t) \psi(t,s) dt \right)^2 ds + \lambda \sum_{g=1}^{G} \Vert S_{g}T\psi \Vert_2}
#' to obtain a sparse coefficient vector \eqn{\psi=\mathsf{vec}(\Psi)\in\mathbb{R}^{ML}} for the functional penalized predictor \eqn{x(t)}, where the coefficient matrix \eqn{\Psi\in\mathbb{R}^{M\times L}}, 
#' the regression function \eqn{\psi(t,s)=\varphi(t)^\intercal\Psi\theta(s)}, 
#' \eqn{\varphi(t)} and \eqn{\theta(s)} are two B-splines bases of order \eqn{d} and dimension \eqn{M} and \eqn{L}, respectively. For each group \eqn{g}, each row of 
#' the matrix \eqn{S_g\in\mathbb{R}^{d\times ML}} has non-zero entries only for those bases belonging 
#' to that group. These values are provided by the arguments \code{groups} and \code{group_weights} (see below). 
#' Each basis function belongs to more than one group. The diagonal matrix \eqn{T\in\mathbb{R}^{ML\times ML}} contains 
#' the basis-specific weights. These values are provided by the argument \code{var_weights} (see below).
#' The regularization path is computed for the overlap group-LASSO penalty at a grid of values for the regularization 
#' parameter \eqn{\lambda} using the alternating direction method of multipliers (ADMM). See Boyd et al. (2011) and Lin et al. (2022) 
#' for details on the ADMM method.
#'  
#' @param mX an \eqn{(n\times r_x)} matrix of observations of the functional covariate.
#' @param mY an \eqn{(n\times r_y)} matrix of observations of the functional response variable.
#' @param L number of elements of the B-spline basis vector \eqn{\theta(s)}.
#' @param M number of elements of the B-spline basis vector \eqn{\varphi(t)}.
#' @param group_weights a vector of length \eqn{G} containing group-specific weights. The default is square root of the group cardinality, see Bernardi et al. (2022).
#' @param var_weights a vector of length \eqn{ML} containing basis-specific weights. The default is a vector where 
#' each entry is the reciprocal of the number of groups including that basis. See Bernardi et al. (2022) for details.
#' @param standardize.data logical. Should data be standardized?
#' @param splOrd the order \eqn{d} of the spline basis.
#' @param lambda either a regularization parameter or a vector of regularization parameters. 
#' In this latter case the routine computes the whole path. If it is NULL values for lambda are provided by the routine.
#' @param lambda.min.ratio smallest value for lambda, as a fraction of the maximum lambda value. If \eqn{nr_y>LM}, the default is 0.0001, and if \eqn{nr_y<LM}, the default is 0.01. 
#' @param nlambda the number of lambda values - default is 30.
#' @param overall.group logical. If it is TRUE, an overall group including all penalized covariates is added.
#' @param cv.fold the number of folds - default is 5.
#' @param control	 a list of control parameters for the ADMM algorithm. See ‘Details’.
#'
#' @return A named list containing \describe{
#' \item{sp.coefficients}{an \eqn{(M\times L)} solution matrix for the parameters \eqn{\Psi}, which corresponds to the minimum cross-validated MSE.}
#' \item{sp.fun}{an \eqn{(r_x\times r_y)} matrix providing the estimated functional coefficient for \eqn{\psi(t,s)} corresponding to the minimum cross-validated MSE.} 
#' \item{lambda}{sequence of lambda.}
#' \item{lambda.min}{value of lambda that attains the cross-validated minimum mean squared error.}
#' \item{indi.min.mse}{index of the lambda sequence corresponding to lambda.min.}
#' \item{mse}{cross-validated mean squared error.}
#' \item{min.mse}{minimum value of the cross-validated MSE for the sequence of lambda.}
#' \item{mse.sd}{standard deviation of the cross-validated mean squared error.}
#' \item{convergence}{logical. 1 denotes achieved convergence.}
#' \item{elapsedTime}{elapsed time in seconds.}
#' \item{iternum}{number of iterations.}
#' }
#'
#' Iteration stops when both \code{r_norm} and \code{s_norm} values
#' become smaller than \code{eps_pri} and \code{eps_dual}, respectively.
#' @section Details: The control argument is a list that can supply any of the following components:\describe{
#' \item{adaptation}{logical. If it is TRUE, ADMM with adaptation is performed. The default value is TRUE. See Boyd et al. (2011) for details.}
#' \item{rho}{an augmented Lagrangian parameter. The default value is 1.}
#' \item{tau.ada}{an adaptation parameter greater than one. Only needed if adaptation = TRUE. The default value is 2. See Boyd et al. (2011) and Lin et al. (2022) for details.}
#' \item{mu.ada}{an adaptation parameter greater than one. Only needed if adaptation = TRUE. The default value is 10. See Boyd et al. (2011) and Lin et al. (2022) for details.}
#' \item{abstol}{absolute tolerance stopping criterion. The default value is sqrt(sqrt(.Machine$double.eps)).}
#' \item{reltol}{relative tolerance stopping criterion. The default value is sqrt(.Machine$double.eps).}
#' \item{maxit}{maximum number of iterations. The default value is 100.}
#' \item{print.out}{logical. If it is TRUE, a message about the procedure is printed. The default value is TRUE.}
#' }
#' 
#' @examples 
#' 
#' ## generate sample data
#' set.seed(4321)
#' s  <- seq(0, 1, length.out = 100)
#' t  <- seq(0, 1, length.out = 100)
#' p1 <- 5
#' p2 <- 6
#' r  <- 10
#' n  <- 50
#' 
#' beta_basis1 <- splines::bs(s, df = p1, intercept = TRUE)    # first basis for beta
#' beta_basis2 <- splines::bs(s, df = p2, intercept = TRUE)    # second basis for beta
#' 
#' data_basis <- splines::bs(s, df = r, intercept = TRUE)    # basis for X
#' 
#' x_0   <- apply(matrix(rnorm(p1 * p2, sd = 1), p1, p2), 1, 
#'                fdaSP::softhresh, 1.5)  # regression coefficients 
#' x_fun <- beta_basis2 %*% x_0 %*%  t(beta_basis1)  
#' 
#' fun_data <- matrix(rnorm(n*r), n, r) %*% t(data_basis)
#' b        <- fun_data %*% x_fun + rnorm(n * 100, sd = sd(fun_data %*% x_fun )/3)
#' 
#' ## set the hyper-parameters
#' maxit          <- 1000
#' rho_adaptation <- FALSE
#' rho            <- 0.01
#' reltol         <- 1e-5
#' abstol         <- 1e-5
#' 
#' ## fit functional regression model
#' mod_cv <- f2fSP_cv(mY = b, mX = fun_data, L = p1, M = p2,
#'                    group_weights = NULL, var_weights = NULL, 
#'                    standardize.data = FALSE, splOrd = 4,
#'                    lambda = NULL, nlambda = 30, cv.fold = 5, 
#'                    lambda.min.ratio = NULL,
#'                    control = list("abstol" = abstol, 
#'                                   "reltol" = reltol, 
#'                                   "maxit" = maxit, 
#'                                   "adaptation" = rho_adaptation, 
#'                                   "rho" = rho, 
#'                                   "print.out" = FALSE))
#'
#' ### graphical presentation
#' plot(log(mod_cv$lambda), mod_cv$mse, type = "l", col = "blue", lwd = 2, bty = "n", 
#'      xlab = latex2exp::TeX("$\\log(\\lambda)$"), ylab = "Prediction Error", 
#'      ylim = range(mod_cv$mse - mod_cv$mse.sd, mod_cv$mse + mod_cv$mse.sd),
#'      main = "Cross-validated Prediction Error")
#' fdaSP::confband(xV = log(mod_cv$lambda), yVmin = mod_cv$mse - mod_cv$mse.sd, 
#'                 yVmax = mod_cv$mse + mod_cv$mse.sd)       
#' abline(v = log(mod_cv$lambda[which(mod_cv$lambda == mod_cv$lambda.min)]), col = "red", lwd = 1.0)
#' 
#' ### comparison with oracle error
#' mod <- f2fSP(mY = b, mX = fun_data, L = p1, M = p2,
#'              group_weights = NULL, var_weights = NULL, 
#'              standardize.data = FALSE, splOrd = 4,
#'              lambda = NULL, nlambda = 30, lambda.min.ratio = NULL, 
#'              control = list("abstol" = abstol, 
#'                             "reltol" = reltol, 
#'                             "maxit" = maxit,
#'                             "adaptation" = rho_adaptation, 
#'                             "rho" = rho, 
#'                             "print.out" = FALSE))
#' err_mod <- apply(mod$sp.coef.path, 1, function(x) sum((x - x_0)^2))
#' plot(log(mod$lambda), err_mod, type = "l", col = "blue", lwd = 2, 
#'      xlab = latex2exp::TeX("$\\log(\\lambda)$"), 
#'      ylab = "Estimation Error", main = "True Estimation Error", bty = "n")
#' abline(v = log(mod$lambda[which(err_mod == min(err_mod))]), col = "red", lwd = 1.0)
#' abline(v = log(mod_cv$lambda[which(mod_cv$lambda == mod_cv$lambda.min)]), 
#'        col = "red", lwd = 1.0, lty = 2)
#' 
#' @references
#' \insertRef{bernardi_etal.2022}{fdaSP}
#' 
#' \insertRef{boyd_etal.2011}{fdaSP}
#' 
#' \insertRef{jenatton_etal.2011}{fdaSP}
#' 
#' \insertRef{lin_etal.2022}{fdaSP}
#'
#' @export
f2fSP_cv <- function(mY, mX, L, M, group_weights = NULL, var_weights = NULL, standardize.data = FALSE, splOrd = 4,
                     lambda = NULL, lambda.min.ratio = NULL, nlambda = NULL, cv.fold = 5, overall.group = FALSE, control = list()) {
    
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # Get dimensions
  n <- dim(mX)[1]             # number of observations
  p <- dim(mX)[2]
  
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # Create the B-spline basis functions
  s      <- seq(0, 1, length.out = p)
  mH     <- t(bs(s, df = M, intercept = TRUE))
  mTheta <- t(bs(s, df = L, intercept = TRUE))
  
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # Pre-processing
  # data validity
  if (!check_data_matrix(mX)) {
    stop("* f2fSP_cv : input 'mX' is invalid data matrix.")
  }
  if (!check_data_matrix(mY)) {
    stop("* f2fSP_cv : input 'mY' is invalid data matrix.")
  }
  mY <- as.matrix(mY)
  # data size
  if (n != dim(mY)[1]) {
    stop("* f2fSP_cv : two inputs 'mX' and 'mY' have non-matching dimension.")
  }
  
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # Define response and design matrix
  mY_B <- mY
  mX_B <- tcrossprod(mX, mH)

  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # Vectorisation form
  X_ <- kronecker(t(mTheta), mX_B)
  y_ <- matrix(ks::vec(mY_B, byrow = FALSE), ncol = 1)

  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # Standardise response and design matrix
  if (standardize.data == TRUE) {
    res   <- standardizemat(X_, y_)
    X.std <- res$X.std
    y.std <- res$y.std
    mU    <- res$mU
    mV    <- res$mV
  } else {
    X.std <- X_
    y.std <- y_
  }
  
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # Pre-processing: lambda parameter (in case of a single lambda)
  if (!is.null(lambda)) {
    if (length(lambda) == 1) {
      meps     <- (.Machine$double.eps)
      negsmall <- -meps
      if (!check_param_constant(lambda, negsmall)) {
        stop("* f2fSP_cv : reg. parameter 'lambda' is invalid.")
      }
      if (lambda < meps){
        message("* f2fSP_cv : since 'lambda' is effectively zero, a least-squares solution is returned.")
        xsol     <- as.vector(aux_pinv(X_) %*% matrix(y_))
        output   <- list()
        output$x <- xsol
        return(output)
      }
    }
  }
  
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # check for inputs
  npar   <- dim(X.std)[2]
  con    <- list(maxit                = 100L, 
                 abstol               = sqrt(.Machine$double.eps), 
                 reltol               = sqrt(sqrt(.Machine$double.eps)), 
                 adaptation           = TRUE, 
                 rho                  = 1, 
                 tau.ada              = 2,               
                 mu.ada               = 10,
                 par.init             = NULL,
                 init.rnd             = FALSE,
                 print.out            = TRUE)
  
  nmsC                          <- names(con)
  con[(namc <- names(control))] <- control
  if (length(noNms <- namc[!namc %in% nmsC])) {
    warning("unknown names in control: ", paste(noNms, collapse = ", "))
  }
  # set default values (for inputs)
  maxit                <- con$maxit
  abstol               <- con$abstol
  reltol               <- con$reltol
  adaptation           <- con$adaptation
  rho                  <- con$rho
  tau.ada              <- con$tau.ada      
  mu.ada               <- con$mu.ada
  par.init             <- con$par.init
  init.rnd             <- con$init.rnd
  print.out            <- con$print.out
  
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # Pre-processing
  # data validity
  if (!check_param_constant_multiple(c(abstol, reltol))) {
    stop("* f2fSP_cv : tolerance level is invalid.")
  }
  if (!check_param_integer(maxit, 0.0)) {
    stop("* f2fSP_cv : 'maxit' should be a positive integer.")
  }
  maxit = as.integer(maxit)
  if (!check_param_constant(rho, 0.0)) {
    stop("* f2fSP_cv : 'rho' should be a positive real number.")
  }
  if (!check_param_constant(mu.ada, 0.0)) {
    stop("* f2fSP_cv : 'mu' should be a positive real number.")
  }
  if ((tau.ada < 1.0) || (tau.ada > 2.0)) {
    warning("* f2fSP_cv : 'tau' value is suggested to be in [1,2].")
  }
  
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # Pre-processing: variable weights
  check_weights(x = var_weights, n = dim(X_)[2], algname = "f2fSP", funname = "var_weights")
  
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # Get the groups of the overlap group LASSO penalty
  if (overall.group == TRUE) {
    out <- penfun(method = "ogl&1", p = M, q = L, splOrd = splOrd, regpars = NULL)
  } else {
    out <- penfun(method = "ogl", p = M, q = L, splOrd = splOrd, regpars = NULL)
  }
  GRmat <- out$grMat
  G     <- dim(GRmat)[1]
  nG    <- rowSums(GRmat)
  
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # Manage group & overall group weights
  if (is.null(group_weights)) {
    # Define the weights as in Yuan and Lin 2006, JRSSB
    group_weights <- sqrt(diag(tcrossprod(GRmat)))
  } else {
    if (overall.group == TRUE) {
      check_group_weights(x = group_weights, n = G, algname = "f2fSP", funname = "group_weights")
      if (length(group_weights) == (G-1)) {
        group_weights <- c(group_weights, sqrt(dim(GRmat)[2]))
      }
    } else {
      check_weights(x = group_weights, n = G, algname = "f2fSP", funname = "group_weights")
    }
  } 
  
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # Manage variable weights
  if (is.null(var_weights)) {
    var_weights <- 1.0 / colSums(out$grMat)
  }

  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # Get the sequence of lambda
  if (!is.null(nlambda) && is.null(lambda)) {
    # get the smallest value of such that the regression 
    # coefficients estimated by the lasso are all equal to zero
    lambda <- lm_lambdamax_OVGLASSO(y = y.std, X = X.std,
                                    groups = NULL, GRmat = out$grMat, group_weights = group_weights, var_weights = var_weights, 
                                    lambda.min.ratio = lambda.min.ratio, maxl = nlambda)$lambda.seq
  }
  if (is.null(nlambda) && is.null(lambda)) {
    # get the smallest value of such that the regression 
    # coefficients estimated by the lasso are all equal to zero
    lambda <- lm_lambdamax_OVGLASSO(y = y.std, X = X.std, 
                                    groups = NULL, GRmat = out$grMat, group_weights = group_weights, var_weights = var_weights, 
                                    lambda.min.ratio = lambda.min.ratio, maxl = 30)$lambda.seq
  }
  if (!is.null(lambda)) {
    nlambda <- length(lambda)
  }
  
  # other parameters
  meps     = (.Machine$double.eps)
  negsmall = -meps
  if (!check_param_constant(min(lambda), negsmall)) {
    stop("* f2fSP_cv : reg. parameter 'lambda' is invalid.")
  }
  if (min(lambda) < meps) {
    message("* f2fSP_cv : since 'lambda' is effectively zero, a least-squares solution is returned.")
    xsol     = as.vector(aux_pinv(mX) %*% matrix(mY))
    output   = list()
    output$x = xsol
    return(output)
  }
  if (!is.null(lambda)) {
    nlambda <- length(lambda)
  }
  
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # Get groups
  ret.gr <- groups.cv(n = n, k = cv.fold)
  
  # store Lasso regression for each fold
  vMse_cv <- matrix(0.0, cv.fold, nlambda)
  
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # Initial time 
  startTime <- Sys.time()
  
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # Run lasso
  for (kt in 1:cv.fold) {
    # run ADMM
    # mX_ <- X.std[ret.gr$groups.cv[[kt]],]
    # vY_ <- y.std[ret.gr$groups.cv[[kt]]]
    data.gr <- f2freg_cv_manage_data(mY = mY, mX = mX, mTheta = mTheta, mH = mH, 
                                     ix.ins = ret.gr$groups.cv[[kt]], 
                                     ix.oos = ret.gr$groups.pred[[kt]], standardize.data = standardize.data)
    mX_ <- data.gr$X_ins.std
    vY_ <- data.gr$y_ins.std
    ret <- .admm_ovglasso_fast(A = mX_, b = vY_, groups = out$grMat, group_weights = group_weights, var_weights = var_weights, 
                               lambda = lambda, rho_adaptation = adaptation, rho = rho, tau = tau.ada, mu = mu.ada, 
                               reltol = reltol, abstol = abstol, maxiter = maxit, ping = 0)
    
    # Evaluate MSE
    mSpRegP_     <- ret$coef.path
    fit          <- data.gr$X_oos.std %*% t(mSpRegP_)
    n_           <- length(data.gr$y_oos.std)
    mse          <- apply(fit, 2, function(x) {sum((data.gr$y_oos.std - x)^2)/n_})
    vMse_cv[kt,] <- mse
  }
  
  # get mean squared error
  vMse <- apply(vMse_cv, 2, mean) 
  vSd  <- apply(vMse_cv, 2, sd) 
  
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # system time
  endTime <- Sys.time()
  
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # compute run time
  eltime <- difftime(time1 = endTime, time2 = startTime, units = "secs")
  
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # Get lasso output
  min.mse      <- min(vMse)
  indi.min.mse <- which(vMse == min.mse) 
  lambda.min   <- lambda[indi.min.mse]
  
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # get estimate over the full sample
  rho_ <- ret$rho[[indi.min.mse]]
  ret  <- .admm_ovglasso(A = X.std, b = y.std,
                         groups = out$grMat, group_weights = group_weights, var_weights = var_weights,
                         u = t(ret$U[indi.min.mse,]), z = t(ret$Z[indi.min.mse,]), lambda = lambda.min,
                         rho_adaptation = adaptation, rho = rho_[length(rho_)], tau = tau.ada, mu = mu.ada,
                         reltol = reltol, abstol = abstol, maxiter = maxit, ping = 0)
  
  # ret  <- .Call("admm_ovglasso",
  #               A = X.std, b = y.std, 
  #               groups = out$grMat, group_weights = group_weights, var_weights = var_weights, 
  #               u = t(ret$U[indi.min.mse,]), z = t(ret$Z[indi.min.mse,]), lambda = lambda.min, 
  #               rho_adaptation = adaptation, rho = rho_[length(rho_)], tau = tau.ada, mu = mu.ada,
  #               reltol = reltol, abstol = abstol, maxiter = maxit, ping = 0)
  
  # Store output: full sample estimate
  niter <- ret$niter
  if (niter < maxit) {
    converged <- TRUE
  } else {
    converged <- FALSE
  }
  vSpRegP <- ret$x
  dConv   <- ifelse(converged == TRUE, 1, 0) 
  nIterN  <- ret$niter

  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # Retrieve estimated parameters
  if (standardize.data == TRUE) {
    vSpRegP_ <- matrix(solve(mU) %*% vSpRegP %*% mV, nrow = npar, ncol = 1)
  } else {
    vSpRegP_ <- matrix(vSpRegP, nrow = npar, ncol = 1)
  }
  mSpRegP_ <- matrix(vSpRegP_, nrow = M, ncol = L, byrow = FALSE)
  
  # get the estimated functions
  # sp.fun <- t(mTheta) %*% mSpRegP_ %*% mH
  sp.fun <- t(mH) %*% mSpRegP_ %*% mTheta

  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # Print to screen
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  if (print.out == TRUE) {
    cat("\n\n\n")
    cat("function to function regression model with overlap group-LASSO penalty, Cross Validation\n")
    cat("Alternating direction method of multipliers\n")
    cat("running time (for ", reltol, " relative error):",
        eltime, "seconds \n\n\n")
  }
  
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # Get output
  res.names <- c("sp.coefficients",
                 "sp.fun",
                 "mse",
                 "mse.sd",
                 "min.mse",
                 "lambda",
                 "lambda.min",
                 "convergence",
                 "elapsedTime",
                 "iternum",
                 "indi.min.mse") 
  res        <- vector(mode = "list", length = length(res.names))
  names(res) <- res.names
  
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # Get output
  ret                 <- NULL
  ret$sp.coefficients <- mSpRegP_
  ret$sp.fun          <- sp.fun
  ret$mse             <- vMse
  ret$mse.sd          <- vSd
  ret$min.mse         <- min.mse
  ret$lambda          <- lambda
  ret$lambda.min      <- lambda.min
  ret$convergence     <- dConv
  ret$elapsedTime     <- eltime
  ret$iternum         <- nIterN
  ret$indi.min.mse    <- indi.min.mse
  
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # Return output
  return(ret)
}

f2freg_cv_manage_data <- function(mY, mX, mTheta, mH, ix.ins = NULL, ix.oos = NULL, standardize.data = TRUE) {
    
  if (!is.null(ix.ins)) {
    # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    # Define the in-sample and out-of-sample data
    mY_ins <- mY[ix.ins,]
    mY_oos <- mY[ix.oos,]
    mX_ins <- mX[ix.ins,]
    mX_oos <- mX[ix.oos,]
    
    # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    # Define response and design matrix
    mY_B_ins <- mY_ins
    mY_B_oos <- mY_oos
    mX_B_ins <- tcrossprod(mX_ins, mH)
    mX_B_oos <- tcrossprod(mX_oos, mH)
    
    # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    # Vectorisation form
    X_ins_ <- kronecker(t(mTheta), mX_B_ins)
    X_oos_ <- kronecker(t(mTheta), mX_B_oos)
    y_ins_ <- matrix(ks::vec(mY_B_ins, byrow = FALSE), ncol = 1)
    y_oos_ <- matrix(ks::vec(mY_B_oos, byrow = FALSE), ncol = 1)
    X_     <- rbind(X_ins_, X_oos_)
    y_     <- as.matrix(c(y_ins_, y_oos_))
    
    # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    # Standardise response and design matrix
    if (standardize.data == TRUE) {
      res       <- standardizemat(X_, y_)
      mU        <- res$mU
      mV        <- res$mV
      mx        <- apply(X_, 2, mean)
      my        <- mean(y_)
      y_ins.std <- (y_ins_ - my) %*% solve(mV)
      y_oos.std <- (y_oos_ - my) %*% solve(mV)
      X_ins.std <- t(apply(X_ins_, 1, function(x) {(x-mx)})) %*% solve(mU)
      X_oos.std <- t(apply(X_oos_, 1, function(x) {(x-mx)})) %*% solve(mU)
    } else {
      X_ins.std <- X_ins_
      y_ins.std <- y_ins_
      X_oos.std <- X_oos_
      y_oos.std <- y_oos_
    }
  } else {
    # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    # Define response and design matrix
    mY_B <- mY
    mX_B <- tcrossprod(mX, mH)
    
    # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    # Vectorisation form
    X_ <- kronecker(t(mTheta), mX_B)
    y_ <- matrix(ks::vec(mY_B, byrow = FALSE), ncol = 1)
    
    # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    # Standardise response and design matrix
    if (standardize.data == TRUE) {
      res   <- standardizemat(X_, y_)
      X.std <- res$X.std
      y.std <- res$y.std
    } else {
      X.std <- X_
      y.std <- y_
    }
  }
  
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # Get output
  res <- NULL
  if (!is.null(ix.ins)) {
    res$X_ins.std <- X_ins.std
    res$y_ins.std <- y_ins.std
    res$X_oos.std <- X_oos.std
    res$y_oos.std <- y_oos.std
  } else {
    res$X.std <- X.std
    res$y.std <- y.std
  }
  
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # Return output
  return(res)
}

