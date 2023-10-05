# Authors: Mauro Bernardi
#          Department Statistical Sciences
#       	 University of Padova
#      	   Via Cesare Battisti, 241
#      	   35121 PADOVA, Italy
#          E-mail: mauro.bernardi@unipd.it  

# Last change: March 16, 2023

#' Sparse Group Least Absolute Shrinkage and Selection Operator
#'
#' Sparse group-LASSO, or sparse group \eqn{L_2}-regularized regression, solves the following optimization problem
#' \deqn{\textrm{min}_\beta ~ \frac{1}{2}\|y-X\beta-Z\gamma\|_2^2 + \lambda[(1-\alpha) \sum_{g=1}^G \|S_g T\beta\|_2+\alpha\Vert T_1\beta\Vert_1]}
#' to obtain a sparse coefficient vector \eqn{\beta\in\mathbb{R}^p} and a coefficient vector \eqn{\gamma\in\mathbb{R}^q}. For each group, each row of 
#' the matrix \eqn{S_g\in\mathbb{R}^{n_g\times p}} has non-zero entries only for those variables belonging 
#' to that group. These values are provided by the arguments \code{groups} and \code{group_weights} (see below). 
#' Each variable must belong to only one group. The diagonal matrix \eqn{T\in\mathbb{R}^{p\times p}} contains the variable-specific weights. These values are
#' provided by the argument \code{var_weights} (see below). The diagonal matrix \eqn{T_1\in\mathbb{R}^{p\times p}} contains 
#' the variable-specific \eqn{L_1} weights. These values are provided by the argument \code{var_weights_L1} (see below).
#' The regularization path is computed for the sparse group-LASSO penalty at a grid of values for the regularization 
#' parameter \eqn{\lambda} using the alternating direction method of multipliers (ADMM). See Boyd et al. (2011) 
#' for details on the ADMM method. The elastic net regularization is a combination of \eqn{\ell_2} stability and
#' \eqn{\ell_1} sparsity constraint simultenously and can be obtained by 
#' specifying \eqn{G=1}, the matrix \eqn{T} and \eqn{T_1} as the identity matrix of dimension \eqn{p}, and the selection matrix 
#' \eqn{S_1} as a row vector of ones of dimension \eqn{p}. The adaptive LASSO can be obtained as a special 
#' case of the sparse group-LASSO penalty by setting \eqn{\alpha=1}, the matrix \eqn{T} as the identity matrix of dimension \eqn{p} and the
#' matrix \eqn{T_1} as the diagonal matrix containing the \eqn{L_1} variable-specific weights.
#'  
#' @param X an \eqn{(n\times p)} matrix of penalized predictors.
#' @param Z an \eqn{(n\times q)} full column rank matrix of predictors that are not penalized. 
#' @param y a length-\eqn{n} response vector.
#' @param groups either a vector of consecutive integers describing the grouping of the coefficients, 
#' or a list with two elements: the first element is a vector containing the variables belonging to each group, 
#' while the second element is a vector containing the group lenghts, (see example below).
#' @param group_weights a vector of length \eqn{G} containing group-specific weights. The default is square root of the group cardinality, see Yuan and Lin (2006).
#' @param var_weights a vector of length \eqn{p} containing variable-specific weights. The default is a vector of ones.
#' @param var_weights_L1 a vector of length \eqn{p} containing variable-specific weights for the \eqn{L_1} penalty. The default is a vector of ones.
#' @param standardize.data logical. Should data be standardized?
#' @param lambda either a regularization parameter or a vector of regularization parameters. In this latter case the routine computes the whole path. If it is NULL the path values for lambda are provided by the routine.
#' @param alpha the sparse group-LASSO mixing parameter, with \eqn{0\leq\alpha\leq1}. The penalty is defined as
#' \eqn{\alpha = 1} is the LASSO penalty, and \eqn{\alpha = 0} the group-LASSO penalty.
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
#' beta <- c(rep(4, 12), rep(0, p - 13), -2)
#' y    <- X %*% beta + rnorm(n, sd = sqrt(crossprod(X %*% beta)) / 20)
#' 
#' ### define groups of dimension 3 each
#' group1 <- rep(1:10, each = 3)
#' 
#' ### set regularization parameter grid
#' lam   <- 10^seq(1, -2, length.out = 30)
#' 
#' ### set the alpha parameter 
#' alpha <- 0.5
#' 
#' ### set the hyper-parameters of the ADMM algorithm
#' maxit         <- 1000
#' adaptation    <- TRUE
#' rho           <- 1
#' reltol        <- 1e-5
#' abstol        <- 1e-5
#' 
#' ### run example
#' mod <- fdaSP:::linreg_ADMM_spGLASSO(X = X, y = y,  groups = group1, standardize.data = FALSE, intercept = FALSE,
#'                                     lambda = lam, alpha = alpha, control = list("adaptation" = adaptation, 
#'                                                                                 "rho" = rho, "maxit" = maxit, 
#'                                                                                 "abstol" = abstol, "reltol" = reltol, 
#'                                                                                 "print.out" = FALSE))
#' 
#' ### graphical presentation
#' matplot(log(lam), mod$sp.coef.path, type = "l", main = "Sparse Group Lasso solution path",
#'         bty = "n", xlab = TeX("$\\log(\\lambda)$"), ylab = "")
#' } 
#' 
#' @references
#' \insertRef{simon_etal.2013}{fdaSP}
#' 
#' \insertRef{yuan_lin.2006}{fdaSP}
#' 
#' \insertRef{zou_hastie.2005}{fdaSP}
#' 
#' \insertRef{zou.2006}{fdaSP}
#' 
#' \insertRef{hastie_etal.2015}{fdaSP}
#' 
#' \insertRef{boyd_etal.2011}{fdaSP}
#' 
#' \insertRef{lin_etal.2022}{fdaSP} 
#'
#' @noRd
linreg_ADMM_spGLASSO <- function(X, Z = NULL, y, groups, group_weights = NULL, var_weights = NULL, var_weights_L1 = NULL, 
                                 standardize.data = TRUE, lambda = NULL, alpha = NULL, lambda.min.ratio = NULL, nlambda = 30, 
                                 intercept = FALSE, control = list()) {
  
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # add constant
  if (standardize.data == TRUE) {
    if (intercept == TRUE){
      intercept = FALSE
      warning("* linreg_ADMM_spGLASSO : intercept should be set to FALSE is data are standardized!\n")
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
    stop("* linreg_ADMM_spGLASSO : input 'X' is invalid data matrix.")  
  }
  if (!check_data_vector(y)) {
    stop("* linreg_ADMM_spGLASSO : input 'y' is invalid data vector.")  
  }
  y <- as.vector(y)
  # data size
  if (dim(X)[1] != length(y)) {
    stop("* linreg_ADMM_spGLASSO : two inputs 'X' and 'Y' have non-matching dimension.")
  }
  if (!is.null(Z)) {
    if (dim(Z)[1] != length(y)) {
      stop("* linreg_ADMM_spGLASSO : two inputs 'Z' and 'Y' have non-matching dimension.")
    }
  }
  
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # Pre-processing: lambda parameter (in case of a single lambda)
  if (!is.null(lambda)) {
    if (length(lambda) == 1) {
      meps     <- (.Machine$double.eps)
      negsmall <- -meps
      if (!check_param_constant(lambda, negsmall)) {
        stop("* linreg_ADMM_spGLASSO : reg. parameter 'lambda' is invalid.")
      }
      if (lambda < meps){
        message("* linreg_ADMM_spGLASSO : since 'lambda' is effectively zero, a least-squares solution is returned.")
        xsol     <- as.vector(aux_pinv(X) %*% matrix(y))
        output   <- list()
        output$x <- xsol
        return(output)
      }
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
  # Pre-processing: alpha parameter
  if (is.null(alpha)) {
    alpha <- 0.5
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
    stop("* linreg_ADMM_spGLASSO : tolerance level is invalid.")
  }
  if (!check_param_integer(maxit, 0.0)) {
    stop("* linreg_ADMM_spGLASSO : 'maxiter' should be a positive integer.")
  }
  maxit = as.integer(maxit)
  if (!check_param_constant(rho, 0.0)) {
    stop("* linreg_ADMM_spGLASSO : 'rho' should be a positive real number.")
  }
  if (!check_param_constant(mu.ada, 0.0)) {
    stop("* linreg_ADMM_spGLASSO : 'mu.ada' should be a positive real number.")
  }
  if ((tau.ada < 1.0) || (tau.ada > 2.0)) {
    warning("* linreg_spADMM_GLASSO : 'tau.ada' value is suggested to be in [1,2].")
  }
  
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # Pre-processing: variable weights
  check_weights(x = var_weights, n = dim(X)[2], algname = "linreg_ADMM_spGLASSO", funname = "var_weights")
  
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # Pre-processing: variable weights (L1)
  check_weights(x = var_weights_L1, n = dim(X)[2], algname = "linreg_ADMM_spGLASSO", funname = "var_weights_L1")
  
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # Manage groups
  if (is.list(groups) && !is.null(groups$Glen)) {
    nG      <- groups$Glen
    groups_ <- groups$groups
    GRmat   <- groups2mat_GLASSO(groups = groups_, Glen = nG)
    G       <- dim(GRmat)[1]
  } else {
    groups_ <- groups
    GRmat   <- groups2mat_GLASSO(groups = groups_, Glen = NULL)
    G       <- dim(GRmat)[1]
    nG      <- rowSums(GRmat)
  }
  
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # Check overlapping groups
  res.method <- check_group_overlaps(GRmat)
  if (res.method == "ovglasso") {
    stop("* linreg_ADMM_spGLASSO : this routine is for fitting the linear regression model with sparse group-LASSO penalty. 
            Use routine 'linreg_ADMM_OVGLASSO' for fitting the linear regression model with overlap group-LASSO penalty!")
  }
  if (res.method == "lasso") {
    warning("* linreg_spADMM_GLASSO : this routine is for fitting the linear regression model with sparse group-LASSO penalty. 
            Use (faster) routine 'linreg_ADMM_LASSO' instead or check the provided groups!")
  }
  
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # Manage group weights
  if (is.null(group_weights)) {
    # Define the weights as in Yuan and Lin 2006, JRSSB
    group_weights <- sqrt(diag(tcrossprod(GRmat)))
  } else {
    check_weights(x = group_weights, n = G, algname = "linreg_ADMM_spGLASSO", funname = "group_weights")
  } 
  
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # Manage variable weights
  if (is.null(var_weights)) {
    var_weights <- 1.0 / colSums(GRmat)
  }
  
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # Manage variable weights: adaptive Lasso
  if (is.null(var_weights_L1)) {
    var_weights_L1 <- rep(1, p)
  }
 
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # Get the sequence of lambdas
  if (!is.null(nlambda) && is.null(lambda)) {
    # get the smallest value of such that the regression 
    # coefficients estimated by the lasso are all equal to zero
    lambda <- lm_lambdamax_spGLASSO(y = y.std, X = X.std, Z = Z, 
                                    groups = groups, group_weights = group_weights, var_weights = var_weights, var_weights_L1 = var_weights_L1, 
                                    lambda.min.ratio = lambda.min.ratio, maxl = nlambda, alpha = alpha)$lambda.seq
  }
  if (is.null(nlambda) && is.null(lambda)) {
    # get the smallest value of such that the regression 
    # coefficients estimated by the lasso are all equal to zero
    lambda <- lm_lambdamax_spGLASSO(y = y.std, X = X.std, Z = Z, 
                                    groups = groups, group_weights = group_weights, var_weights = var_weights, var_weights_L1 = var_weights_L1, 
                                    lambda.min.ratio = lambda.min.ratio, maxl = 30, alpha = alpha)$lambda.seq
  }
  if (!is.null(lambda)) {
    nlambda <- length(lambda)
  }

  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # Run lasso
  if (is.null(Z)) {
    ret <- .admm_spglasso_fast(A = X.std, b = y.std, groups = GRmat,
                               group_weights = group_weights, var_weights = var_weights, var_weights_L1 = var_weights_L1,
                               lambda = lambda, alpha = alpha, rho_adaptation = adaptation, rho = rho, tau = tau.ada, mu = mu.ada,
                               reltol = reltol, abstol = abstol, maxiter = maxit, ping = 0)
    
    # ret <- .Call("admm_spglasso_fast", 
    #              A = X.std, b = y.std, groups = GRmat, 
    #              group_weights = group_weights, var_weights = var_weights, var_weights_L1 = var_weights_L1,
    #              lambda = lambda, alpha = alpha, rho_adaptation = adaptation, rho = rho, tau = tau.ada, mu = mu.ada, 
    #              reltol = reltol, abstol = abstol, maxiter = maxit, ping = 0)
    
    # get estimated coefficients and path
    mSpRegP <- t(ret$coef.path)
    vSpRegP <- ret$coefficients
  } else {
    ret <- .admm_spglasso_cov_fast(W = X.std, Z = Z, y = y.std,
                                   groups = GRmat, group_weights = group_weights, var_weights = var_weights, var_weights_L1 = var_weights_L1,
                                   lambda = lambda, alpha = alpha, rho_adaptation = adaptation, rho = rho, tau = tau.ada, mu = mu.ada,
                                   reltol = reltol, abstol = abstol, maxiter = maxit, ping = 0)
    
    # ret <- .Call("admm_spglasso_cov_fast", 
    #              W = X.std, Z = Z, y = y.std,
    #              groups = GRmat, group_weights = group_weights, var_weights = var_weights, var_weights_L1 = var_weights_L1,
    #              lambda = lambda, alpha = alpha, rho_adaptation = adaptation, rho = rho, tau = tau.ada, mu = mu.ada, 
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
    # if (alpha == 0) {
    #   dim_ <- sum(nG)
    # } else if (alpha == 1) {
    #   dim_ <- p
    # } else {
    #   # dim_ <- sum(nG) + p
    #   dim_ <- p
    # }
    dim_    <- p
    sp.path <- matrix(nlambda, dim_, data = t(apply(mSpRegP, 2, function(x) solve(mU) %*% x %*% mV)))
    vSpRegP <- solve(mU) %*% vSpRegP %*% mV
    if (!is.null(Z)) {
      path  <- matrix(nlambda, q, data = t(apply(mRegP, 2, function(x) x %*% mV)))
      vRegP <- vRegP %*% mV
    }
  } else {
    vSpRegP <- matrix(vSpRegP, length(vSpRegP), 1)
    sp.path  <- t(mSpRegP)
    if (!is.null(Z)) {
      path <- t(mRegP)
    }
  }
  
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # Print to screen
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  if (print.out == TRUE) {
    cat("\n\n\n")
    cat("Linear regression model with sparse group-LASSO penalty\n")
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














