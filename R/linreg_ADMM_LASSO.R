# Authors: Mauro Bernardi 
#          Department Statistical Sciences
#       	 University of Padova
#      	   Via Cesare Battisti, 241
#      	   35121 PADOVA, Italy
#          E-mail: mauro.bernardi@unipd.it  

# Last change: March 16, 2023

# List of functions
# 1. linreg_ADMM_LASSO
# 2. linreg_ADMM_LASSO_1lambda
# 3. linreg_ADMM_LASSO_cv

#' Least Absolute Shrinkage and Selection Operator
#'
#' LASSO, or L1-regularized regression, solves the following optimization problem
#' \deqn{\textrm{min}_\beta ~ \frac{1}{2}\|y-X\beta-Z\gamma\|_2^2 + \lambda \|T \beta\|_1}
#' to obtain a sparse coefficient vector \eqn{\beta\in\mathbb{R}^p} and a coefficient vector \eqn{\gamma\in\mathbb{R}^q}. The diagonal matrix \eqn{T\in\mathbb{R}^{p\times p}} 
#' contains the variable-specific weights. When \eqn{T} is equal to the identity matrix the standard LASSO method is performed, otherwise the adaptive 
#' LASSO method of Zou (2006) is performed. These values are provided by the argument \code{var_weights} (see below).
#' The regularization path is computed for the LASSO penalty at a grid of values for the regularization parameter \eqn{\lambda} using the 
#' alternating direction method of multipliers (ADMM). See Boyd et al. (2011) for details on the ADMM method.
#'  
#' @param X an \eqn{(n\times p)} matrix of penalized predictors.
#' @param Z an \eqn{(n\times q)} full column rank matrix of predictors that are not penalized. 
#' @param y a length-\eqn{n} response vector.
#' @param var_weights a vector of length \eqn{p} containing variable-specific weights. The default is NULL.
#' @param standardize.data logical. Should data be standardized?
#' @param lambda either a regularization parameter or a vector of regularization parameters. 
#' In this latter case the routine computes the whole path. If it is NULL the path values for lambda are provided by the routine.
#' @param lambda.min.ratio smallest value for lambda, as a fraction of the maximum lambda value, the (data derived) entry value (i.e. the smallest value for which all coefficients are zero). 
#' The default depends on the sample size nobs relative to the number of variables nvars. If nobs > nvars, the default is 0.0001, 
#' close to zero. If nobs < nvars, the default is 0.01. A very small value of lambda.min.ratio will lead to a saturated fit 
#' in the nobs < nvars case.
#' @param nlambda the number of lambda values - default is 30.
#' @param intercept logical. If it is TRUE, a column of ones is added to the design matrix. 
#' @param control	 a list of control parameters. See ‘Details’.
#'
#' @return A named list containing \describe{
#' \item{sp.coefficients}{a length-\eqn{p} solution vector for the parameters \eqn{\beta}, which corresponds to the minimum in-sample MSE.}
#' \item{coefficients}{a length-\eqn{q} solution vector for the parameters \eqn{\gamma}, which corresponds to the minimum in-sample MSE. 
#' It is provided only when either the matrix \eqn{Z} in input is not NULL or the intercept is set to TRUE.}
#' \item{sp.coef.path}{an \eqn{(nlambda\times p)} matrix of estimated \eqn{\beta} coefficients for each lambda of the provided sequence.}
#' \item{coef.path}{an \eqn{(nlambda\times q)} matrix of estimated \eqn{\gamma} coefficients for each lambda of the provided sequence. 
#' It is provided only when either the matrix \eqn{Z} in input is not NULL or the intercept is set to TRUE.}
#' \item{lambda}{sequence of lambda.}
#' \item{lambda.min}{value of lambda that attains the minimum in-sample MSE.}
#' \item{mse}{in-sample mean squared error.}
#' \item{min.mse}{minimum value of the MSE for the sequence of lambdas.}
#' \item{convergence}{logical. 1 denotes achieved convergence.}
#' \item{elapsedTime}{elapsed time in seconds.}
#' \item{iternum}{number of iterations.}
#' }
#'
#' When you run the algorithm, output returns not only the solution, but also the iteration history recording
#' following fields over iterates:
#' \describe{
#' \item{objval}{objective function value}
#' \item{r_norm}{norm of primal residual}
#' \item{s_norm}{norm of dual residual}
#' \item{eps_pri}{feasibility tolerance for primal feasibility condition}
#' \item{eps_dual}{feasibility tolerance for dual feasibility condition.}
#' }
#' Iteration stops when both \code{r_norm} and \code{s_norm} values
#' become smaller than \code{eps_pri} and \code{eps_dual}, respectively.
#' @section Details: The control argument is a list that can supply any of the following components:\describe{
#' (To understand exactly what these do see the source code: higher levels give more detail.).
#' \item{adaptation}{logical. If it is TRUE, ADMM with adaptation is performed. The default value is TRUE.}
#' \item{rho}{an augmented Lagrangian parameter. The default value is 1.}
#' \item{tau.ada}{an adaptation parameter in [1,2]. Only needed if adaptation = TRUE. The default value is 2. See Boyd et al. (2011) for details.}
#' \item{mu.ada}{an adaptation parameter. Only needed if adaptation = TRUE. The default value is 10. See Boyd et al. (2011) for details.}
#' \item{abstol}{absolute tolerance stopping criterion. The default value is sqrt(sqrt(.Machine$double.eps)).}
#' \item{reltol}{relative tolerance stopping criterion. The default value is sqrt(.Machine$double.eps).}
#' \item{maxit}{maximum number of iterations. The default value is 100.}
#' \item{print.out}{logical. If it is TRUE, a message about the procedure is printed. The default value is TRUE.}
#' }
#' 
#' @examples
#' \donttest{
#' 
#' ### generate sample data
#' set.seed(2023)
#' n    <- 50
#' p    <- 30 
#' X    <- matrix(rnorm(n * p), n, p)
#' beta <- apply(matrix(rnorm(p, sd = 1), p, 1), 1, fdaSP:::softhresh, 1.5)
#' y    <- X %*% beta + rnorm(n, sd = sqrt(crossprod(X %*% beta)) / 20)
#' 
#' ### set regularization parameter grid
#' lam   <- 10^seq(0, -2, length.out = 30)
#' 
#' ### set the hyper-parameters of the ADMM algorithm
#' maxit      <- 1000
#' adaptation <- TRUE
#' rho        <- 1
#' reltol     <- 1e-5
#' abstol     <- 1e-5
#' 
#' ### run example
#' mod <- linreg_ADMM_LASSO(X = X, y = y, standardize.data = FALSE, intercept = FALSE, lambda = lam,
#'                          control = list("adaptation" = adaptation, "rho" = rho, 
#'                                         "maxit" = maxit, "reltol" = reltol, "abstol" = abstol, 
#'                                         "print.out" = FALSE)) 
#'                                         
#' ### graphical presentation
#' matplot(log(lam), mod$sp.coef.path, type = "l", main = "Lasso solution path",
#'         bty = "n", xlab = TeX("$\\log(\\lambda)$"), ylab = "")
#' }
#'
#' @references
#' \insertRef{tibshirani_regression_1996a}{fdaSP}
#' 
#' \insertRef{boyd_etal.2011}{fdaSP}
#' 
#' \insertRef{hastie_etal.2015}{fdaSP}
#' 
#' \insertRef{lin_etal.2022}{fdaSP} 
#' 
#' \insertRef{zou.2006}{fdaSP} 
#'
#' @noRd
linreg_ADMM_LASSO <- function(X, Z = NULL, y, var_weights = NULL, standardize.data = FALSE, 
                              lambda = NULL, lambda.min.ratio = NULL, nlambda = 30, 
                              intercept = FALSE, control = list()) {
  
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # add constant
  if (standardize.data == TRUE) {
    if (intercept == TRUE){
      intercept = FALSE
      warning("* linreg_ADMM_LASSO : intercept should be set to FALSE is data are standardized!\n")
    }
  }
  if (intercept == TRUE) {
    if (!is.null(Z)) {
      Z <- cbind(matrix(data = 1, nrow = length(y), ncol = 1), Z)
    } else {
      Z <- matrix(data = 1, nrow = length(y), ncol = 1)
    }
  }
  
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # Pre-processing: data validity
  if (!check_data_matrix(X)) {
    stop("* linreg_ADMM_LASSO : input 'X' is invalid data matrix.")  
  }
  if (!check_data_vector(y)) {
    stop("* linreg_ADMM_LASSO : input 'y' is invalid data vector.")  
  }
  y <- as.vector(y)
  # data size
  if (dim(X)[1] != length(y)) {
    stop("* linreg_ADMM_LASSO : two inputs 'X' and 'Y' have non-matching dimension.")
  }
  if (!is.null(Z)) {
    if (dim(Z)[1] != length(y)) {
      stop("* linreg_ADMM_OVGLASSO : two inputs 'Z' and 'Y' have non-matching dimension.")
    }
  }
  
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # get data dimensions
  n <- dim(X)[1]
  p <- dim(X)[2]  
  if (!is.null(Z)) {
    q <- dim(Z)[2]
  }
  
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # check for adaptive-LASSO
  if (is.null(var_weights)) {
    adalasso    <- FALSE
    var_weights <- rep(1L, p)
  } else {
    adalasso <- TRUE
  }
  
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # Standardise response and design matrix
  if (standardize.data == TRUE) {
    res   <- standardizemat(X, y)
    X.std <- res$X.std
    y.std <- res$y.std
    mU    <- res$mU
    mV    <- res$mV
  } else {
    X.std <- X
    y.std <- y
  }
  
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # Pre-processing: lambda parameter (in case of a single lambda)
  if (!is.null(lambda)) {
    if (length(lambda) == 1) {
      meps     <- (.Machine$double.eps)
      negsmall <- -meps
      if (!check_param_constant(lambda, negsmall)) {
        stop("* linreg_ADMM_LASSO : reg. parameter 'lambda' is invalid.")
      }
      if (lambda < meps){
        message("* linreg_ADMM_LASSO : since 'lambda' is effectively zero, a least-squares solution is returned.")
        xsol     <- as.vector(aux_pinv(X) %*% matrix(y))
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
                 print.out            = TRUE)
  
  nmsC                          <- names(con)
  con[(namc <- names(control))] <- control
  if (length(noNms <- namc[!namc %in% nmsC])) {
    warning("unknown names in control: ", paste(noNms, collapse = ", "))
  }
  # set default values (for inputs)
  maxit      <- con$maxit
  abstol     <- con$abstol
  reltol     <- con$reltol
  adaptation <- con$adaptation
  rho        <- con$rho
  tau.ada    <- con$tau.ada      
  mu.ada     <- con$mu.ada
  print.out  <- con$print.out
  
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # Pre-processing: other parameters
  if (!check_param_constant_multiple(c(abstol, reltol))) {
    stop("* linreg_ADMM_LASSO : tolerance level is invalid.")
  }
  if (!check_param_integer(maxit, 0.0)) {
    stop("* linreg_ADMM_LASSO : 'maxiter' should be a positive integer.")
  }
  maxit = as.integer(maxit)
  if (!check_param_constant(rho, 0.0)) {
    stop("* linreg_ADMM_LASSO : 'rho' should be a positive real number.")
  }
  if (!check_param_constant(mu.ada, 0.0)) {
    stop("* linreg_ADMM_LASSO : 'mu.ada' should be a positive real number.")
  }
  if ((tau.ada < 1.0) || (tau.ada > 2.0)) {
    warning("* linreg_ADMM_LASSO : 'tau.ada' value is suggested to be in [1,2].")
  }
  
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # Pre-processing: variable weights
  check_weights(x = var_weights, n = dim(X)[2], algname = "linreg_ADMM_LASSO", funname = "var_weights")
  
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # Get the sequence of lambdas
  if (!is.null(nlambda) && is.null(lambda)) {
    # get the smallest value of such that the regression 
    # coefficients estimated by the lasso are all equal to zero
    lambda <- lm_lambdamax_ADALASSO(y = y.std, X = X.std, Z, var_weights = var_weights, lambda.min.ratio = lambda.min.ratio, maxl = nlambda)$lambda.seq
  }
  if (is.null(nlambda) && is.null(lambda)) {
    # get the smallest value of such that the regression 
    # coefficients estimated by the lasso are all equal to zero
    lambda <- lm_lambdamax_ADALASSO(y = y.std, X = X.std, Z, var_weights = var_weights, lambda.min.ratio = lambda.min.ratio, maxl = 30)$lambda.seq
  }
  if (!is.null(lambda)) {
    nlambda <- length(lambda)
  }

  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # Run lasso
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # Run overlap group Lasso
  if (is.null(Z)) {
    if (adalasso == TRUE) {
      ret <- .admm_adalasso_fast(A = X.std, b = y.std, var_weights = var_weights, lambda = lambda, rho_adaptation = adaptation, rho = rho,
                                 tau = tau.ada, mu = mu.ada, reltol = reltol, abstol = abstol, maxiter = maxit, ping = 0)
      
      # ret <- .Call("admm_adalasso_fast", 
      #              A = X.std, b = y.std, var_weights = var_weights, lambda = lambda, rho_adaptation = adaptation, rho = rho, 
      #              tau = tau.ada, mu = mu.ada, reltol = reltol, abstol = abstol, maxiter = maxit, ping = 0)
    } else {
      ret <- .admm_lasso_fast(A = X.std, b = y.std, lambda = lambda, rho_adaptation = adaptation, rho = rho,
                              tau = tau.ada, mu = mu.ada, reltol = reltol, abstol = abstol, maxiter = maxit, ping = 0)
      
      # ret <- .Call("admm_lasso_fast", 
      #              A = X.std, b = y.std, lambda = lambda, rho_adaptation = adaptation, rho = rho, 
      #              tau = tau.ada, mu = mu.ada, reltol = reltol, abstol = abstol, maxiter = maxit, ping = 0)
    }
    # get estimated coefficients and path
    mSpRegP <- t(ret$coef.path)
    vSpRegP <- ret$coefficients
  } else {
    if (adalasso == TRUE) {
      ret <- .admm_adalasso_cov_fast(W = X.std, Z = Z, y = y.std, var_weights = var_weights, lambda = lambda, rho_adaptation = adaptation, rho = rho,
                                     tau = tau.ada, mu = mu.ada, reltol = reltol, abstol = abstol, maxiter = maxit, ping = 0)
      
      # ret <- .Call("admm_adalasso_cov_fast",
      #              W = X.std, Z = Z, y = y.std, var_weights = var_weights, lambda = lambda, rho_adaptation = adaptation, rho = rho, 
      #              tau = tau.ada, mu = mu.ada, reltol = reltol, abstol = abstol, maxiter = maxit, ping = 0)
    } else {
      ret <- .admm_lasso_cov_fast(W = X.std, Z = Z, y = y.std, lambda = lambda, rho_adaptation = adaptation, rho = rho,
                                  tau = tau.ada, mu = mu.ada, reltol = reltol, abstol = abstol, maxiter = maxit, ping = 0)
      
      # ret <- .Call("admm_lasso_cov_fast",
      #              W = X.std, Z = Z, y = y.std, lambda = lambda, rho_adaptation = adaptation, rho = rho, 
      #              tau = tau.ada, mu = mu.ada, reltol = reltol, abstol = abstol, maxiter = maxit, ping = 0)
    }
    # get estimated coefficients and path
    mSpRegP <- t(ret$sp.coef.path)
    mRegP   <- t(ret$coef.path)
    vSpRegP <- ret$sp.coefficients
    vRegP   <- ret$coefficients
  }
  
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # Get the path and retrieve the scaled estimates
  if (standardize.data == TRUE) {
    sp.path <- matrix(nlambda, p, data = t(apply(mSpRegP, 2, function(x) solve(mU) %*% x %*% mV)))
    vSpRegP <- solve(mU) %*% vSpRegP %*% mV
    if (!is.null(Z)) {
      path  <- matrix(nlambda, q, data = t(apply(mRegP, 2, function(x) x %*% mV)))
      vRegP <- vRegP %*% mV
    }
  } else {
    vSpRegP <- matrix(vSpRegP, length(vSpRegP), 1)
    sp.path <- t(mSpRegP)
    if (!is.null(Z)) {
      path <- t(mRegP)
    }
  }
  
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # Print to screen
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  if (print.out == TRUE) {
    cat("\n\n\n")
    cat("Linear regression model with LASSO penalty\n")
    cat("Alternating direction method of multipliers\n")
    cat("running time (for ", reltol, " relative error):",
        sum(ret$elapsedTime), "seconds \n\n\n")
  }
  
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # Get output
  res.names <- c("sp.coefficients", 
                 "sp.coef.path", 
                 "coefficients",
                 "coef.path",
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
  res        <- vector(mode = "list", length = length(res.names))
  names(res) <- res.names
  if (!is.null(Z)) {
    res$coefficients <- vRegP
    res$coef.path    <- path
  } 
  res$sp.coefficients <- vSpRegP
  res$sp.coef.path    <- sp.path
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





