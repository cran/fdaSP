# Authors: Mauro Bernardi 
#          Department Statistical Sciences
#       	 University of Padova
#      	   Via Cesare Battisti, 241
#      	   35121 PADOVA, Italy
#          E-mail: mauro.bernardi@unipd.it  

# Last change: March 16, 2023


#' Overlap Group Least Absolute Shrinkage and Selection Operator
#'
#' Overlap group-LASSO, or overlap group L2-regularized regression, solves the following optimization problem
#' \deqn{\textrm{min}_\beta ~ \frac{1}{2}\|y-X\beta-Z\gamma\|_2^2 + \lambda\sum_{g=1}^G \|S_g T\beta\|_2}
#' to obtain a sparse coefficient vector \eqn{\beta\in\mathbb{R}^p} and a coefficient vector \eqn{\gamma\in\mathbb{R}^q}. For each group, each row of 
#' the matrix \eqn{S_g\in\mathbb{R}^{n_g\times p}} has non-zero entries only for those variables belonging 
#' to that group. These values are provided by the arguments \code{groups} and \code{group_weights} (see below). 
#' Each variable can belong to more than one group. The diagonal matrix \eqn{T\in\mathbb{R}^{p\times p}} contains 
#' the variable-specific weights. These values are provided by the argument \code{var_weights} (see below).
#' The regularization path is computed for the overlap group-LASSO penalty at a grid of values for the regularization 
#' parameter \eqn{\lambda} using the alternating direction method of multipliers (ADMM). See Boyd et al. (2011) 
#' for details on the ADMM method.
#'  
#' @param X an \eqn{(n\times p)} matrix of penalized predictors.
#' @param Z an \eqn{(n\times q)} full column rank matrix of predictors that are not penalized. 
#' @param y a length-\eqn{n} response vector.
#' @param groups a list with two elements: the first element is a vector containing the variables belonging to each group, 
#' while the second element is a vector containing the group lengths (see example below).
#' @param group_weights a vector of length G containing group-specific weights. The default is square root of the group cardinality, see Bernardi et al. (2022).
#' @param var_weights a vector of length p containing variable-specific weights. The default is a vector where 
#' each entry is the reciprocal of the number of groups including that variable.
#' @param standardize.data logical. Should data be standardized?
#' @param lambda either a regularization parameter or a vector of regularization parameters. 
#' In this latter case the routine computes the whole path. If it is NULL the path values for lambda are provided by the routine.
#' @param lambda.min.ratio smallest value for lambda, as a fraction of the maximum lambda value, the (data derived) entry value (i.e. the smallest value for which all coefficients are zero). 
#' The default depends on the sample size nobs relative to the number of variables nvars. If nobs > nvars, the default is 0.0001, 
#' close to zero. If nobs < nvars, the default is 0.01. A very small value of lambda.min.ratio will lead to a saturated fit 
#' in the nobs < nvars case.
#' @param nlambda the number of lambda values - default is 30.
#' @param intercept logical. If it is TRUE, a column of ones is added to the design matrix. 
#' @param overall.group logical. If it is TRUE, an overall group including all penalized covariates is added.
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
#' \item{lambda.min}{value of lambda that attains the in-sample minimum in-sample MSE.}
#' \item{mse}{in-sample mean squared error.}
#' \item{min.mse}{minimum value of the in-sample MSE for the sequence of lambdas.}
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
#' X    <- matrix(rnorm(n*p), n, p)
#' beta <- rep(c(rep(4, 4), rep(0, 3), rep(-4, 3)), 3)
#' y    <- X %*% beta + rnorm(n, sd = sqrt(crossprod(X %*% beta )) / 20)
#' 
#' ## define overlapping groups of dimension 3 each 
#' group1 <- NULL
#' for(k in 0:(p - 3)) group1 <- c(group1, 1:3 + k)
#' Glen <- rep(3, p - 2)
#' 
#' ### set regularization parameter grid
#' lam   <- 10^seq(1, -2, length.out = 30)
#' 
#' ### set the hyper-parameters of the ADMM algorithm
#' maxit         <- 1000
#' adaptation    <- TRUE
#' rho           <- 1
#' reltol        <- 1e-5
#' abstol        <- 1e-5
#' 
#' ### run example
#' mod <- fdaSP:::linreg_ADMM_OVGLASSO(X = X, y = y, groups = list("groups" = group1, "Glen" = Glen),
#'                                     standardize.data = FALSE,  intercept = FALSE, lambda = lam,
#'                                     control = list("adaptation" = adaptation, "rho" = rho, 
#'                                                    "maxit" = maxit, "abstol" = abstol, "reltol" = reltol, 
#'                                                    "print.out" = FALSE))
#' 
#' ### graphical presentation
#' matplot(log(lam), mod$sp.coef.path, type = "l", main = "Overlap Group Lasso solution path", 
#'         bty = "n", xlab = TeX("$\\log(\\lambda)$"), ylab = "")
#' }
#' 
#' @references
#' \insertRef{bernardi_etal.2022}{fdaSP}
#' 
#' \insertRef{jenatton_etal.2011}{fdaSP}
#' 
#' \insertRef{boyd_etal.2011}{fdaSP}
#' 
#' \insertRef{lin_etal.2022}{fdaSP}
#'
#' @noRd
linreg_ADMM_OVGLASSO <- function(X, Z = NULL, y, groups, group_weights = NULL, var_weights = NULL, standardize.data = TRUE,
                                 lambda = NULL, lambda.min.ratio = NULL, nlambda = 30, 
                                 intercept = FALSE, overall.group = FALSE, control = list()) {
  
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # add constant
  if (standardize.data == TRUE) {
    if (intercept == TRUE){
      intercept = FALSE
      warning("* linreg_ADMM_OVGLASSO : intercept should be set to FALSE is data are standardized!\n")
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
    stop("* linreg_ADMM_OVGLASSO : input 'X' is invalid data matrix.")  
  }
  if (!check_data_vector(y)) {
    stop("* linreg_ADMM_OVGLASSO : input 'y' is invalid data vector.")  
  }
  y <- as.vector(y)
  # data size
  if (dim(X)[1] != length(y)) {
    stop("* linreg_ADMM_OVGLASSO : two inputs 'X' and 'Y' have non-matching dimension.")
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
        stop("* linreg_ADMM_OVGLASSO : reg. parameter 'lambda' is invalid.")
      }
      if (lambda < meps){
        message("* linreg_ADMM_OVGLASSO : since 'lambda' is effectively zero, a least-squares solution is returned.")
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
    stop("* linreg_ADMM_OVGLASSO : tolerance level is invalid.")
  }
  if (!check_param_integer(maxit, 0.0)) {
    stop("* linreg_ADMM_OVGLASSO : 'maxiter' should be a positive integer.")
  }
  maxit = as.integer(maxit)
  if (!check_param_constant(rho, 0.0)) {
    stop("* linreg_ADMM_OVGLASSO : 'rho' should be a positive real number.")
  }
  if (!check_param_constant(mu.ada, 0.0)) {
    stop("* linreg_ADMM_OVGLASSO : 'mu.ada' should be a positive real number.")
  }
  if ((tau.ada < 1.0) || (tau.ada > 2.0)) {
    warning("* linreg_ADMM_OVGLASSO : 'tau.ada' value is suggested to be in [1,2].")
  }
  
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # Pre-processing: variable weights
  check_weights(x = var_weights, n = dim(X)[2], algname = "linreg_ADMM_OVGLASSO", funname = "var_weights")
  
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # Manage groups
  if (is.list(groups) && !is.null(groups$Glen)) {
    nG      <- groups$Glen
    groups_ <- groups$groups
    GRmat   <- groups2mat_OVGLASSO(groups = groups_, Glen = nG)
    G       <- dim(GRmat)[1]
  } else {
    groups_ <- groups
    GRmat   <- groups2mat_OVGLASSO(groups = groups_, Glen = NULL)
    G       <- dim(GRmat)[1]
    nG      <- rowSums(GRmat)
  }
  
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # Check overlapping groups
  res.method <- check_group_overlaps(GRmat)
  if (res.method != "ovglasso") {
    warning("* linreg_ADMM_OVGLASSO : this routine is for fitting the linear regression model with overlap group-LASSO penalty. 
            Use (faster) routine 'linreg_ADMM_GLASSO' instead or check the provided groups!")
  }
  
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # Manage overall group
  if ((overall.group == TRUE) && (!any(nG == dim(X)[2]))) {
    GRmat <- rbind(GRmat, rep(1, dim(GRmat)[2]))
    G     <- G + 1
    nG    <- c(nG, dim(GRmat)[2])
  }
  
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # Manage group & overall group weights
  if (is.null(group_weights)) {
    # Define the weights as in Yuan and Lin 2006, JRSSB
    group_weights <- sqrt(diag(tcrossprod(GRmat)))
  } else {
    if (overall.group == TRUE) {
      check_group_weights(x = group_weights, n = G, algname = "linreg_ADMM_OVGLASSO", funname = "group_weights")
      if (length(group_weights) == (G-1)) {
        group_weights <- c(group_weights, sqrt(dim(GRmat)[2]))
      }
    } else {
      check_weights(x = group_weights, n = G, algname = "linreg_ADMM_OVGLASSO", funname = "group_weights")
    }
  } 

  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # Manage variable weights
  if (is.null(var_weights)) {
    var_weights <- 1.0 / colSums(GRmat)
  } 
  
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # Get the sequence of lambdas
  if (!is.null(nlambda) && is.null(lambda)) {
    # get the smallest value of such that the regression 
    # coefficients estimated by the lasso are all equal to zero
    lambda <- lm_lambdamax_OVGLASSO(y = y.std, X = X.std, Z = Z, 
                                    GRmat = GRmat, group_weights = group_weights, var_weights = var_weights, 
                                    lambda.min.ratio = lambda.min.ratio, maxl = nlambda)$lambda.seq
  }
  if (is.null(nlambda) && is.null(lambda)) {
    # get the smallest value of such that the regression 
    # coefficients estimated by the lasso are all equal to zero
    lambda <- lm_lambdamax_OVGLASSO(y = y.std, X = X.std, Z = Z, 
                                    GRmat = GRmat, group_weights = group_weights, var_weights = var_weights, 
                                    lambda.min.ratio = lambda.min.ratio, maxl = 30)$lambda.seq
  }
  if (!is.null(lambda)) {
    nlambda <- length(lambda)
  }
  
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # Run overlap group Lasso
  if (is.null(Z)) {
    ret <- .admm_ovglasso_fast(A = X.std, b = y.std, groups = GRmat, group_weights = group_weights, var_weights = var_weights,
                               lambda = lambda, rho_adaptation = adaptation, rho = rho, tau = tau.ada, mu = mu.ada,
                               reltol = reltol, abstol = abstol, maxiter = maxit, ping = 0)
    
    # ret <- .Call("admm_ovglasso_fast", 
    #              A = X.std, b = y.std, groups = GRmat, group_weights = group_weights, var_weights = var_weights,
    #              lambda = lambda, rho_adaptation = adaptation, rho = rho, tau = tau.ada, mu = mu.ada,
    #              reltol = reltol, abstol = abstol, maxiter = maxit, ping = 0)
    
    # get estimated coefficients and path
    mSpRegP <- t(ret$coef.path)
    vSpRegP <- ret$coefficients
  } else {
    ret <- .admm_ovglasso_cov_fast(W = X.std, Z = Z, y = y.std,
                                   groups = GRmat, group_weights = group_weights, var_weights = var_weights, 
                                   lambda = lambda, rho_adaptation = adaptation, rho = rho, tau = tau.ada, mu.ada,
                                   reltol = reltol, abstol = abstol, maxiter = maxit, ping = 0)
    
    # ret <- .Call("admm_ovglasso_cov_fast",
    #              W = X.std, Z = Z, y = y.std,
    #              groups = GRmat, group_weights = group_weights, var_weights = var_weights, 
    #              lambda = lambda, rho_adaptation = adaptation, rho = rho, tau = tau.ada, mu.ada,
    #              reltol = reltol, abstol = abstol, maxiter = maxit, ping = 0)
    
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
    cat("Linear regression model with overlap group-LASSO penalty\n")
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


