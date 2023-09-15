# Authors: Mauro Bernardi 
#          Department Statistical Sciences
#       	 University of Padova
#      	   Via Cesare Battisti, 241
#      	   35121 PADOVA, Italy
#          E-mail: mauro.bernardi@unipd.it  

# Last change: March 16, 2023

#' Cross-validation for Least Absolute Shrinkage and Selection Operator
#'
#' LASSO, or L1-regularized regression, solves the following optimization problem
#' \deqn{\textrm{min}_\beta ~ \frac{1}{2}\|y-X\beta-Z\gamma\|_2^2 + \lambda \|\beta\|_1}
#' to obtain a sparse coefficient vector \eqn{\beta\in\mathbb{R}^p} and a coefficient vector \eqn{\gamma\in\mathbb{R}^q}. The regularization path 
#' is computed for the LASSO penalty at a grid of values for the regularization parameter \eqn{\lambda} using the 
#' alternating direction method of multipliers (ADMM). See Boyd et al. (2011) for details on the ADMM method.
#'
#' @param X an \eqn{(n\times p)} matrix of penalized predictors.
#' @param Z an \eqn{(n\times q)} full column rank matrix of predictors that are not penalized. 
#' @param y a length-\eqn{n} response vector.
#' @param var_weights a vector of length \eqn{p} containing variable-specific weights. The default is a vector of ones.
#' @param standardize.data logical. Should data be standardized?
#' @param lambda either a regularization parameter or a vector of regularization parameters. 
#' In this latter case the routine computes the whole path. If it is NULL the path values for lambda are provided by the routine.
#' @param lambda.min.ratio smallest value for lambda, as a fraction of the maximum lambda value, the (data derived) entry value 
#' (i.e. the smallest value for which all coefficients are zero). 
#' The default depends on the sample size nobs relative to the number of variables nvars. If nobs > nvars, the default is 0.0001, 
#' close to zero. If nobs < nvars, the default is 0.01. A very small value of lambda.min.ratio will lead to a saturated fit 
#' in the nobs < nvars case.
#' @param nlambda the number of lambda values - default is 30.
#' @param cv.fold the number of folds - default is 5.
#' @param intercept logical. If it is TRUE, a column of ones is added to the design matrix. 
#' @param control	 a list of control parameters. See ‘Details’.
#'
#' @return A named list containing \describe{
#' \item{sp.coefficients}{a length-\eqn{p} solution vector for the parameters \eqn{\beta}, which corresponds to the model fitted on the full sample with \eqn{\lambda} corresponding to minimum cross-validated MSE.}
#' \item{coefficients}{a length-\eqn{q} solution vector for the parameters \eqn{\gamma}, which corresponds to the model fitted on the full sample with \eqn{\lambda} corresponding to minimum cross-validated MSE. 
#' It is provided only when either the matrix \eqn{Z} in input is not NULL or the intercept is set to TRUE.}
#' \item{convergence}{logical. 1 denotes achieved convergence for the model fitted on the full sample.}
#' \item{iternum}{number of iterations for the model fitted on the full sample.}
#' \item{lambda}{sequence of lambda.}
#' \item{lambda.min}{value of lambda that attains the minimum cross-validated MSE.}
#' \item{indi.min.mse}{index of the lambda sequence corresponding to lambda.min.}
#' \item{mse}{cross-validated mean squared error.}
#' \item{min.mse}{minimum value of the cross-validated MSE for the sequence of lambdas.}
#' \item{mse.sd}{standard deviation of the cross-validated mean squared error.}
#' \item{elapsedTime}{elapsed time in seconds for the whole procedure.}
#' }
#'
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
#' @examples
#' 
#' ### generate sample data
#' set.seed(2023)
#' n    <- 50
#' p    <- 30 
#' X    <- matrix(rnorm(n * p), n, p)
#' beta <- apply(matrix(rnorm(p, sd = 1), p, 1), 1, fdaSP:::softhresh, 1.5)
#' y    <- X %*% beta + rnorm(n, sd = sqrt(crossprod(X %*% beta)) / 20)
#' 
#' ### set the hyper-parameters of the ADMM algorithm
#' maxit      <- 1000
#' adaptation <- TRUE
#' rho        <- 1
#' reltol     <- 1e-5
#' abstol     <- 1e-5
#' 
#' ### run cross-validation
#' mod_cv <- fdaSP:::linreg_ADMM_LASSO_cv(X = X, y = y, standardize.data = FALSE, intercept = FALSE,
#'                                        cv.fold = 5, nlambda = 30, 
#'                                        control = list("adaptation" = adaptation, "rho" = rho, 
#'                                                       "maxit" = maxit, "reltol" = reltol, "abstol" = abstol, 
#'                                                       "print.out" = FALSE)) 
#' 
#' ### graphical presentation
#' plot(log(mod_cv$lambda), mod_cv$mse, type = "l", col = "blue", lwd = 2, bty = "n", 
#'      xlab = TeX("$\\log(\\lambda)$"), ylab = "Prediction Error", 
#'      ylim = range(mod_cv$mse - mod_cv$mse.sd, mod_cv$mse + mod_cv$mse.sd),
#'      main = "Cross-validated Prediction Error")
#' fdaSP:::confband(xV = log(mod_cv$lambda), yVmin = mod_cv$mse - mod_cv$mse.sd, yVmax = mod_cv$mse + mod_cv$mse.sd)       
#' abline(v = log(mod_cv$lambda[which(mod_cv$lambda == mod_cv$lambda.min)]), col = "red", lwd = 1.0)
#' 
#' ### comparison with oracle error
#' mod <- fdaSP:::linreg_ADMM_LASSO(X = X, y = y, standardize.data = FALSE, intercept = FALSE,
#'                                  nlambda = 30, control = list("adaptation" = adaptation, "rho" = rho, 
#'                                                               "maxit" = maxit, "reltol" = reltol, "abstol" = abstol, 
#'                                                               "print.out" = FALSE)) 
#' err_mod <- apply(mod$sp.coef.path, 1, function(x) sum((x - beta)^2))
#' plot(log(mod$lambda), err_mod, type = "l", col = "blue", lwd = 2, xlab = TeX("$\\log(\\lambda)$"), 
#'      ylab = "Estimation Error", main = "True Estimation Error", bty = "n")
#' abline(v = log(mod$lambda[which(err_mod == min(err_mod))]), col = "red", lwd = 1.0)
#' abline(v = log(mod_cv$lambda[which(mod_cv$lambda == mod_cv$lambda.min)]), col = "red", lwd = 1.0, lty = 2)
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
linreg_ADMM_LASSO_cv <- function(X, Z = NULL, y, var_weights = NULL, standardize.data = TRUE, lambda = NULL, nlambda = NULL, lambda.min.ratio = NULL, 
                                 cv.fold = 5, intercept = TRUE, control = list()) {
  
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # Get dimensions
  n <- dim(X)[1]
  p <- dim(X)[2]  
  
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # Pre-processing
  # data validity
  if (!check_data_matrix(X)) {
    stop("* linreg_ADMM_LASSO_cv : input 'X' is invalid data matrix.")  
  }
  if (!check_data_vector(y)) {
    stop("* linreg_ADMM_LASSO_cv : input 'y' is invalid data vector.")  
  }
  y = as.vector(y)
  # data size
  if (n != length(y)) {
    stop("* linreg_ADMM_LASSO_cv : two inputs 'X' and 'y' have non-matching dimension.")
  }
  
  if (!is.null(var_weights)) {
    if (!check_data_positive_vector(var_weights)) {
      stop("* linreg_ADMM_LASSO : input 'var_weights' is invalid vector of weights.")  
    } else {
      adalasso <- TRUE
    }
  } else {
    adalasso    <- FALSE
    var_weights <- rep(1L, p)
  }
  
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # Get dimensions
  if (standardize.data == TRUE) {
    if (intercept == TRUE){
      intercept = FALSE
      warning("* linreg_ADMM_LASSO_cv : intercept should be set to FALSE is data are standardized!\n")
    }
  }
  
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # add constant
  if (intercept == TRUE) {
    if (!is.null(Z)) {
      Z <- cbind(matrix(data = 1, nrow = length(y), ncol = 1), Z)
    } else {
      Z <- matrix(data = 1, nrow = length(y), ncol = 1)
    }
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
  maxit                <- con$maxit
  abstol               <- con$abstol
  reltol               <- con$reltol
  adaptation           <- con$adaptation
  rho                  <- con$rho
  tau.ada              <- con$tau.ada      
  mu.ada               <- con$mu.ada
  print.out            <- con$print.out
  
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # data validity
  if (!check_param_constant_multiple(c(abstol, reltol))) {
    stop("* linreg_ADMM_LASSO_cv : tolerance level is invalid.")
  }
  if (!check_param_integer(maxit, 0.0)) {
    stop("* linreg_ADMM_LASSO_cv : 'maxiter' should be a positive integer.")
  }
  maxiter = as.integer(maxit)
  if (!check_param_constant(rho, 0.0)) {
    stop("* linreg_ADMM_LASSO_cv : 'rho' should be a positive real number.")
  }
  if (!check_param_constant(mu.ada, 0.0)) {
    stop("* linreg_ADMM_LASSO_cv : 'alpha' should be a positive real number.")
  }
  if ((tau.ada < 1.0) || (tau.ada > 2.0)) {
    warning("* linreg_ADMM_LASSO_cv : 'tau' value is suggested to be in [1,2].")
  }

  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # Get the sequence of lambdas
  if (!is.null(nlambda) && is.null(lambda)) {
    # get the smallest value of such that the regression 
    # coefficients estimated by the lasso are all equal to zero
    lambda <- lm_lambdamax_ADALASSO(y = y.std, X = X.std, Z = Z, var_weights = var_weights, lambda.min.ratio = lambda.min.ratio, maxl = nlambda)$lambda.seq
  }
  if (is.null(nlambda) && is.null(lambda)) {
    # get the smallest value of such that the regression 
    # coefficients estimated by the lasso are all equal to zero
    lambda <- lm_lambdamax_ADALASSO(y = y.std, X = X.std, Z = Z, var_weights = var_weights, lambda.min.ratio = lambda.min.ratio, maxl = 30)$lambda.seq
  }
  if (!is.null(lambda)) {
    nlambda <- length(lambda)
  }
  
  # other parameters
  meps     = (.Machine$double.eps)
  negsmall = -meps
  if (!check_param_constant(min(lambda), negsmall)) {
    stop("* linreg_ADMM_LASSO_cv : reg. parameter 'lambda' is invalid.")
  }
  if (min(lambda) < meps) {
    message("* linreg_ADMM_LASSO_cv : since 'lambda' is effectively zero, a least-squares solution is returned.")
    xsol     = as.vector(aux_pinv(X) %*% matrix(y))
    output   = list()
    output$x = xsol
    return(output)
  }
  
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # Output
  vMse     <- rep(0, nlambda)
  vSd      <- rep(0, nlambda)
  vRunTime <- rep(0, nlambda)

  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # Store
  u_store   <- matrix(0L, nrow = p, ncol = nlambda)
  z_store   <- matrix(0L, nrow = p, ncol = nlambda)
  rho_store <- rep(0L, nlambda)
  
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
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # Run overlap Group Lasso
  for (kt in 1:cv.fold) {
    # run ADMM
    mX_ <- X.std[ret.gr$groups.cv[[kt]],]
    vY_ <- y.std[ret.gr$groups.cv[[kt]]]
    if (is.null(Z)) {
      if (adalasso == FALSE) {
        ret <- .admm_lasso_fast(A = mX_, b = vY_, lambda = lambda, rho_adaptation = adaptation, rho = rho,
                                tau = tau.ada, mu = mu.ada, reltol = reltol, abstol = abstol, maxiter = maxit, ping = 0)
        
        # ret <- .Call("admm_lasso_fast", 
        #              A = mX_, b = vY_, lambda = lambda, rho_adaptation = adaptation, rho = rho, 
        #              tau = tau.ada, mu = mu.ada, reltol = reltol, abstol = abstol, maxiter = maxit, ping = 0)
      } else {
        ret <- .admm_adalasso_fast(A = mX_, b = vY_, var_weights = var_weights, lambda = lambda, rho_adaptation = adaptation, rho = rho,
                                   tau = tau.ada, mu = mu.ada, reltol = reltol, abstol = abstol, maxiter = maxit, ping = 0)
        
        # ret <- .Call("admm_adalasso_fast", 
        #              A = mX_, b = vY_, var_weights = var_weights, lambda = lambda, rho_adaptation = adaptation, rho = rho, 
        #              tau = tau.ada, mu = mu.ada, reltol = reltol, abstol = abstol, maxiter = maxit, ping = 0)
      }
      
      # fit
      mSpRegP_     <- ret$coef.path
      fit          <- X.std[ret.gr$groups.pred[[kt]],] %*% t(mSpRegP_)
    } else {
      mZ_ <- Z[ret.gr$groups.cv[[kt]],]
      if (adalasso == FALSE) {
        ret <- .admm_lasso_cov_fast(W = mX_, Z = mZ_, y = vY_, lambda = lambda, rho_adaptation = adaptation, rho = rho,
                                    tau = tau.ada, mu = mu.ada, reltol = reltol, abstol = abstol, maxiter = maxit, ping = 0)
        
        # ret <- .Call("admm_lasso_cov_fast",
        #              W = mX_, Z = mZ_, y = vY_, lambda = lambda, rho_adaptation = adaptation, rho = rho, 
        #              tau = tau.ada, mu = mu.ada, reltol = reltol, abstol = abstol, maxiter = maxit, ping = 0)
      } else {
        ret <- .admm_adalasso_cov_fast(W = mX_, Z = mZ_, y = vY_, var_weights = var_weights, lambda = lambda, rho_adaptation = adaptation, rho = rho,
                                       tau = tau.ada, mu = mu.ada, reltol = reltol, abstol = abstol, maxiter = maxit, ping = 0)
        
        # ret <- .Call("admm_adalasso_cov_fast",
        #              W = mX_, Z = mZ_, y = vY_, var_weights = var_weights, lambda = lambda, rho_adaptation = adaptation, rho = rho, 
        #              tau = tau.ada, mu = mu.ada, reltol = reltol, abstol = abstol, maxiter = maxit, ping = 0)
      }
      
      # fit
      mSpRegP_     <- ret$sp.coef.path
      mRegP_       <- ret$coef.path
      fit          <- X.std[ret.gr$groups.pred[[kt]],] %*% t(mSpRegP_) + Z[ret.gr$groups.pred[[kt]],] %*% t(mRegP_)
    }
    
    # Evaluate MSE
    n_           <- length(y.std[ret.gr$groups.pred[[kt]]])
    mse          <- apply(fit, 2, function(x) {sum((y.std[ret.gr$groups.pred[[kt]]] - x)^2)/n_})
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
  #rho_ <- ret$rho[indi.min.mse]
  rho_ <- ret$rho[[indi.min.mse]]
  if (is.null(Z)) {
    if (adalasso == FALSE) {
      ret <- .admm_lasso(A = X.std, b = y.std, u = t(ret$U[indi.min.mse,]), z = t(ret$Z[indi.min.mse,]), lambda = lambda.min,
                         rho_adaptation = adaptation, rho = rho_[length(rho_)], tau = tau.ada, mu = mu.ada,
                         reltol = reltol, abstol = abstol, maxiter = maxit, ping = 0)
      
      # ret <- .Call("admm_lasso", 
      #              A = X.std, b = y.std, u = t(ret$U[indi.min.mse,]), z = t(ret$Z[indi.min.mse,]), lambda = lambda.min, 
      #              rho_adaptation = adaptation, rho = rho_[length(rho_)], tau = tau.ada, mu = mu.ada, 
      #              reltol = reltol, abstol = abstol, maxiter = maxit, ping = 0)
    } else {
      ret <- .admm_adalasso(A = X.std, b = y.std, var_weights = var_weights, u = t(ret$U[indi.min.mse,]), z = t(ret$Z[indi.min.mse,]), lambda = lambda.min,
                            rho_adaptation = adaptation, rho = rho_[length(rho_)], tau = tau.ada, mu = mu.ada,
                            reltol = reltol, abstol = abstol, maxiter = maxit, ping = 0)
      
      # ret <- .Call("_admm_adalasso", 
      #              A = X.std, b = y.std, var_weights = var_weights, u = t(ret$U[indi.min.mse,]), z = t(ret$Z[indi.min.mse,]), lambda = lambda.min, 
      #              rho_adaptation = adaptation, rho = rho_[length(rho_)], tau = tau.ada, mu = mu.ada, 
      #              reltol = reltol, abstol = abstol, maxiter = maxit, ping = 0)
    }
    # get estimated parameters
    vSpRegP <- ret$x
  } else {
    if (adalasso == FALSE) {
      ret <- .admm_lasso_cov(W = X.std, Z = Z, y = y.std, u = t(ret$U[indi.min.mse,]), z = t(ret$Z[indi.min.mse,]), lambda = lambda.min,
                             rho_adaptation = adaptation, rho = rho_[length(rho_)], tau = tau.ada, mu = mu.ada,
                             reltol = reltol, abstol = abstol, maxiter = maxit, ping = 0)
      
      # ret <- .Call("admm_lasso_cov", 
      #              W = X.std, Z = Z, y = y.std, u = t(ret$U[indi.min.mse,]), z = t(ret$Z[indi.min.mse,]), lambda = lambda.min, 
      #              rho_adaptation = adaptation, rho = rho_[length(rho_)], tau = tau.ada, mu = mu.ada, 
      #              reltol = reltol, abstol = abstol, maxiter = maxit, ping = 0)
      
    } else {
      ret <- .admm_adalasso_cov(W = X.std, Z = Z, y = y.std, var_weights = var_weights, u = t(ret$U[indi.min.mse,]), z = t(ret$Z[indi.min.mse,]), lambda = lambda.min,
                                rho_adaptation = adaptation, rho = rho_[length(rho_)], tau = tau.ada, mu = mu.ada,
                                reltol = reltol, abstol = abstol, maxiter = maxit, ping = 0)
      
      # ret <- .Call("admm_adalasso_cov", 
      #              W = X.std, Z = Z, y = y.std, var_weights = var_weights, u = t(ret$U[indi.min.mse,]), z = t(ret$Z[indi.min.mse,]), lambda = lambda.min, 
      #              rho_adaptation = adaptation, rho = rho_[length(rho_)], tau = tau.ada, mu = mu.ada, 
      #              reltol = reltol, abstol = abstol, maxiter = maxit, ping = 0)
    }
    
    # get estimated parameters
    vSpRegP <- ret$x
    vRegP   <- ret$v
  }
  
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # Store output: full sample estimate
  niter <- ret$niter
  if (niter < maxiter) {
    converged <- TRUE
  } else {
    converged <- FALSE
  }
  dConv  <- ifelse(converged == TRUE, 1, 0) 
  nIterN <- ret$niter
  
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # Retrieve estimated parameters
  if (standardize.data == TRUE) {
    vSpRegP_ <- matrix(solve(mU) %*% vSpRegP %*% mV, nrow = p, ncol = 1)
    if (!is.null(Z)) {
      vRegP <- vRegP %*% mV
    }
  } else {
    vSpRegP_ <- matrix(vSpRegP, nrow = p, ncol = 1)
  }
  
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # Print to screen
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  if (print.out == TRUE) {
    cat("\n\n\n")
    cat("Linear regression model with LASSO penalty, Cross Validation\n")
    cat("Alternating direction method of multipliers\n")
    cat("running time (for ", reltol, " relative error):",
        endTime - startTime, "seconds \n\n\n")
  }
  
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # Get output
  res.names <- c("sp.coefficients", 
                 "coefficients",
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
  ret$sp.coefficients <- vSpRegP_
  if (!is.null(Z)) {
    ret$coefficients <- vRegP
  }
  ret$mse          <- vMse
  ret$mse.sd       <- vSd
  ret$min.mse      <- min.mse
  ret$lambda       <- lambda
  ret$lambda.min   <- lambda.min
  ret$convergence  <- dConv
  ret$elapsedTime  <- eltime
  ret$iternum      <- nIterN
  ret$indi.min.mse <- indi.min.mse
  
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # Return output
  return(ret)
}



