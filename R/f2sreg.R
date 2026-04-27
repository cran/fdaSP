# Authors: Mauro Bernardi 
#          Department Statistical Sciences
#       	 University of Padova
#      	   Via Cesare Battisti, 241
#      	   35121 PADOVA, Italy
#          E-mail: mauro.bernardi@unipd.it  

# Last change: October 21, 2025


#' Overlap Group Least Absolute Shrinkage and Selection Operator for scalar-on-function regression model
#'
#' Overlap Group-LASSO for scalar-on-function regression model solves the following optimization problem
#' \deqn{\textrm{min}_{\psi,\gamma} ~ \frac{1}{2} \sum_{i=1}^n \left( y_i - \int x_i(t) \psi(t) dt-z_i^\intercal\gamma \right)^2 + \lambda \sum_{g=1}^{G} \Vert S_{g}T\psi \Vert_2}
#' to obtain a sparse coefficient vector \eqn{\psi\in\mathbb{R}^{M}} for the functional penalized predictor \eqn{x(t)} and a coefficient vector \eqn{\gamma\in\mathbb{R}^q} for the unpenalized scalar predictors \eqn{z_1,\dots,z_q}. The regression function is \eqn{\psi(t)=\varphi(t)^\intercal\psi}
#' where \eqn{\varphi(t)} is a B-spline basis of order \eqn{d} and dimension \eqn{M}. For each group \eqn{g}, each row of 
#' the matrix \eqn{S_g\in\mathbb{R}^{d\times M}} has non-zero entries only for those bases belonging 
#' to that group. These values are provided by the arguments \code{groups} and \code{group_weights} (see below). 
#' Each basis function belongs to more than one group. The diagonal matrix \eqn{T\in\mathbb{R}^{M\times M}} contains 
#' the basis-specific weights. These values are provided by the argument \code{var_weights} (see below).
#' The regularization path is computed for the overlap group-LASSO penalty at a grid of values for the regularization 
#' parameter \eqn{\lambda} using the alternating direction method of multipliers (ADMM). See Boyd et al. (2011) and Lin et al. (2022) 
#' for details on the ADMM method.
#'  
#' @param vY a length-\eqn{n} vector of observations of the scalar response variable.
#' @param mX a \eqn{(n\times r)} matrix of observations of the functional covariate.
#' @param mZ an \eqn{(n\times q)} full column rank matrix of scalar predictors that are not penalized. 
#' @param M number of elements of the B-spline basis vector \eqn{\varphi(t)}.
#' @param group_weights a vector of length \eqn{G} containing group-specific weights. The default is square root of the group cardinality, see Bernardi et al. (2022).
#' @param var_weights a vector of length \eqn{M} containing basis-specific weights. The default is a vector where 
#' each entry is the reciprocal of the number of groups including that basis. See Bernardi et al. (2022) for details.
#' @param weights_adaptive logical. If \code{TRUE}, adaptive weights are computed from an initial 
#' least squares estimate. This option is ignored (and set to \code{FALSE}) when 
#' \code{group_weights} are provided by the user, since external weights override the adaptive scheme.
#' @param standardize.data logical. Should data be standardized?
#' @param splOrd the order \eqn{d} of the spline basis.
#' @param diff_order order of the discrete difference operator used in the smoothness penalty.
#' The default is 1.
#' @param lambda either a regularization parameter or a vector of regularization parameters.
#' In this latter case the routine computes the whole path. If it is NULL values for lambda are provided by the routine.
#' @param lambda2 either a non-negative smoothing regularization parameter or a vector of
#' smoothing regularization parameters. If \code{NULL}, no smoothness penalty is used.
#' @param lambda.min.ratio smallest value for lambda, as a fraction of the maximum lambda value. If \eqn{n>M}, the default is 0.0001, and if \eqn{n<M}, the default is 0.01.
#' @param nlambda the number of lambda values - default is 30.
#' @param intercept logical. If it is TRUE, a column of ones is added to the design matrix. 
#' @param overall.group logical. If it is TRUE, an overall group including all penalized covariates is added.
#' @param control	 a list of control parameters for the ADMM algorithm. See ‘Details’.
#'
#' @return A named list containing \describe{
#' \item{sp.coefficients}{a length-\eqn{M} solution vector for the parameters \eqn{\psi}, which corresponds to the minimum in-sample MSE.}
#' \item{sp.coef.path}{an \eqn{(n_\lambda\times M)} matrix of estimated \eqn{\psi} coefficients for each lambda.}
#' \item{sp.fun}{a length-\eqn{r} vector providing the estimated functional coefficient for \eqn{\psi(t)}.}
#' \item{sp.fun.path}{an \eqn{(n_\lambda\times r)} matrix providing the estimated functional coefficients for \eqn{\psi(t)} for each lambda.}
#' \item{coefficients}{a length-\eqn{q} solution vector for the parameters \eqn{\gamma}, which corresponds to the minimum in-sample MSE. 
#' It is provided only when either the matrix \eqn{Z} in input is not NULL or the intercept is set to TRUE.}
#' \item{coef.path}{an \eqn{(n_\lambda\times q)} matrix of estimated \eqn{\gamma} coefficients for each lambda. 
#' It is provided only when either the matrix \eqn{Z} in input is not NULL or the intercept is set to TRUE.}
#' \item{dof}{estimated degrees of freedom for each value of the regularization parameter.}
#' \item{dof.groups_active}{active groups used in the degrees-of-freedom computation for each value of the regularization parameter.}
#' \item{dof.coeff_active}{active coefficient indices used in the degrees-of-freedom computation for each value of the regularization parameter.}
#' \item{dof.vars_active}{active basis coefficients used in the degrees-of-freedom computation for each value of the regularization parameter.}
#' \item{bic}{Bayesian information criterion computed along the regularization path.}
#' \item{lambda.min}{value of lambda that attains the minimum in-sample MSE.}
#' \item{lambda}{sequence of lambda.}
#' \item{lambda2}{sequence of smoothing regularization parameters. It is \code{NULL} when no smoothness penalty is used.}
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
#' \item{dof.toler_c}{tolerance parameter used in the degrees-of-freedom computation to determine active constraints.}
#' \item{dof.toler_d}{tolerance parameter used in the degrees-of-freedom computation to determine active coefficients or groups.}
#' \item{abstol}{absolute tolerance stopping criterion. The default value is sqrt(sqrt(.Machine$double.eps)).}
#' \item{reltol}{relative tolerance stopping criterion. The default value is sqrt(.Machine$double.eps).}
#' \item{maxit}{maximum number of iterations. The default value is 100.}
#' \item{print.out}{logical. If it is TRUE, a message about the procedure is printed. The default value is TRUE.}
#' }
#' 
#' @examples 
#' 
#' ## generate sample data
#' set.seed(1)
#' n     <- 40
#' p     <- 18                                  # number of basis to GENERATE beta
#' r     <- 100
#' s     <- seq(0, 1, length.out = r)
#' 
#' beta_basis <- splines::bs(s, df = p, intercept = TRUE)    # basis
#' coef_data  <- matrix(rnorm(n*floor(p/2)), n, floor(p/2))        
#' fun_data   <- coef_data %*% t(splines::bs(s, df = floor(p/2), intercept = TRUE))     
#' 
#' x_0   <- apply(matrix(rnorm(p, sd=1),p,1), 1, fdaSP::softhresh, 1)  # regression coefficients 
#' x_fun <- beta_basis %*% x_0                
#' 
#' b     <- fun_data %*% x_fun + rnorm(n, sd = sqrt(crossprod(fun_data %*% x_fun ))/10)
#' l     <- 10^seq(2, -4, length.out = 30)
#' maxit <- 1000
#' 
#' 
#' ## set the hyper-parameters
#' maxit          <- 1000
#' rho_adaptation <- TRUE
#' rho            <- 1
#' reltol         <- 1e-5
#' abstol         <- 1e-5
#' 
#' mod <- f2sSP(vY = b, mX = fun_data, M = p,
#'              group_weights = NULL, var_weights = NULL, standardize.data = FALSE, splOrd = 4,
#'              lambda = NULL, nlambda = 30, lambda.min = NULL, overall.group = FALSE, 
#'              control = list("abstol" = abstol, 
#'                             "reltol" = reltol, 
#'                             "adaptation" = rho_adaptation, 
#'                             "rho" = rho, 
#'                             "print.out" = FALSE)) 
#' 
#' # plot coefficiente path
#' matplot(log(mod$lambda), mod$sp.coef.path, type = "l", 
#'         xlab = latex2exp::TeX("$\\log(\\lambda)$"), ylab = "", bty = "n", lwd = 1.2)
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
f2sSP <- function(vY, mX, mZ = NULL, M, 
                  group_weights = NULL, var_weights = NULL, weights_adaptive = FALSE,
                  standardize.data = TRUE, splOrd = 4, diff_order = 1,
                  lambda = NULL, lambda2 = NULL, nlambda = 30, lambda.min.ratio = NULL, 
                  intercept = FALSE, overall.group = FALSE, control = list()) {
    
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # Pre-processing: data validity
  if (!check_data_matrix(mX)) {
    stop("* f2sSP : input 'mX' is invalid data matrix.")
  }
  if (!check_data_vector(vY)) {
    stop("* f2sSP : input 'vY' is invalid data vector.")
  }
  vY <- as.vector(vY)
  # data size
  if (dim(mX)[1] != length(vY)) {
    stop("* f2sSP : two inputs 'mX' and 'vY' have non-matching dimension.")
  }
  if (!is.null(mZ)) {
    if (dim(mZ)[1] != length(vY)) {
      stop("* f2sSP : two inputs 'mZ' and 'vY' have non-matching dimension.")
    }
  }
  if (M <= 0) {
    stop("* f2sSP : the number of basis must be a positive integer!")
  }
  if (!is.null(diff_order)) {
    if (diff_order < 1) {
      stop("* f2sSP : the parameter diff_order is less than one!")
    }
  }
  
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # Get dimensions
  n <- dim(mX)[1]             # number of observations
  p <- dim(mX)[2]
  
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # Create the B-spline basis functions
  s  <- seq(0, 1, length.out = p)
  mH <- t(splines::bs(s, df = M, intercept = TRUE))
  
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # Define response and design matrix
  mX_B <- tcrossprod(mX, mH)
  
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # Vectorisation form
  X_ <- mX_B
  y_ <- matrix(vY, nrow = n, ncol = 1)
  
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # add constant
  if (standardize.data == TRUE) {
    if (intercept == TRUE){
      intercept = FALSE
      warning("* f2sSP : intercept should be set to FALSE is data are standardized!\n")
    }
  }
  if (intercept == TRUE) {
    if (!is.null(mZ)) {
      mZ <- cbind(matrix(data = 1, nrow = n, ncol = 1), mZ)
    } else {
      mZ <- matrix(data = 1, nrow = n, ncol = 1)
    }
  }
  q <- dim(mZ)[2]
  
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
  output <- f2s_reg_RIDGE(y = y.std, Z = mZ, X = X.std, 
                          diff_order = diff_order, 
                          lambda = lambda, lambda2 = lambda2)

  # retrieve estimate coefficients in case standardize is TRUE
  if (!is.null(output)) {
    mSpRegP <- output$sp.coef.path
    vSpRegP <- output$vSpRegP
    if (!is.null(mZ)) { 
      mRegP <- output$coef.path
      vRegP <- output$vRegP
    }
    # Get the scaled path
    if (standardize.data == TRUE) {
      if (length(lambda2) > 1) {
        sp.path <- matrix(data = t(apply(mSpRegP, 2, function(x) solve(mU) %*% x %*% mV)), nrow = length(lambda2), ncol = M)
      } else {
        sp.path <- matrix(data = t(solve(mU) %*% mSpRegP %*% mV), nrow = 1, ncol = M)
      }
      vSpRegP <- matrix(solve(mU) %*% vSpRegP %*% mV, nrow = 1, ncol = M)
      if (!is.null(mZ)) {
        if (length(lambda2) > 1) {
          path <- matrix(length(lambda2), q, data = t(apply(mRegP, 2, function(x) x %*% mV)))
        } else {
          path <- matrix(1, q, data = t(mRegP %*% mV))
        }
        vRegP <- vRegP %*% mV
      }
    } else {
      sp.path <- matrix(data = t(mSpRegP), length(lambda2), M)
      if (!is.null(mZ)) {
        path  <- t(mRegP)
      }
    }
  }
  
  # retrieve estimated functins and path
  if (!is.null(output)) {
    sp.fun.path <- sp.path %*% mH
    sp.fun      <- vSpRegP %*% mH
  }
  
  # Get output
  if (!is.null(output)) {
    res.names <- c("sp.coefficients", 
                   "sp.coef.path", 
                   "sp.fun.path",
                   "sp.fun",
                   "coefficients",
                   "coef.path",
                   "dof",
                   "lambda",
                   "lambda2",
                   "mse",
                   "min.mse",
                   "bic") 
    res        <- vector(mode = "list", length = length(res.names))
    names(res) <- res.names
    if (!is.null(mZ)) {
      res$coefficients <- as.numeric(vRegP)
      res$coef.path    <- path
    } 
    res$sp.coefficients   <- t(mSpRegP)
    res$sp.coef.path      <- sp.path
    res$sp.fun.path       <- sp.fun.path
    res$sp.fun            <- t(sp.fun)
    res$dof               <- output$dof
    res$lambda            <- lambda
    res$lambda2           <- lambda2
    res$mse               <- output$mse
    res$min.mse           <- output$min.mse
    res$bic               <- output$bic
    
    # Return output
    return(res)
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
                 dof.toler_c          = 5,
                 dof.toler_d          = 0.001,
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
  toler_c              <- con$dof.toler_c
  toler_d              <- con$dof.toler_d
  par.init             <- con$par.init
  init.rnd             <- con$init.rnd
  print.out            <- con$print.out
  
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # Pre-processing: other parameters
  if (!check_param_constant_multiple(c(abstol, reltol))) {
    stop("* f2sSP : tolerance level is invalid.")
  }
  if (!check_param_integer(maxit, 0.0)) {
    stop("* f2sSP : 'maxiter' should be a positive integer.")
  }
  maxit = as.integer(maxit)
  if (!check_param_constant(rho, 0.0)) {
    stop("* f2sSP : 'rho' should be a positive real number.")
  }
  if (!check_param_constant(mu.ada, 0.0)) {
    stop("* f2sSP : 'mu.ada' should be a positive real number.")
  }
  if ((tau.ada < 1.0) || (tau.ada > 2.0)) {
    warning("* f2sSP : 'tau.ada' value is suggested to be in [1,2].")
  }
  
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # Pre-processing: variable weights
  check_weights(x = var_weights, n = dim(X_)[2], algname = "f2sSP", funname = "var_weights")
  
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # Get the groups of the overlap group LASSO penalty
  if (overall.group == TRUE) {
    out <- penfun(method = "ogl&1", p = M, q = 1, splOrd = splOrd, regpars = NULL)
  } else {
    out <- penfun(method = "ogl", p = M, q = 1, splOrd = splOrd, regpars = NULL)
  }
  GRmat <- out$grMat
  G     <- dim(GRmat)[1]
  nG    <- rowSums(GRmat)
  
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # Manage group & overall group weights
  # set weights_adaptive to FALSE if group_weights are provided by the user!
  if (!is.null(group_weights)) {
    if (weights_adaptive == TRUE) {
      warning("* f2sSP : weights_adaptive is set to FALSE if group_weights are provided by the user!\n")
      weights_adaptive <- FALSE
    }
  }
  # compute group_weights if they are not provided by the user
  if (is.null(group_weights)) {
    # Define the weights as in Yuan and Lin 2006, JRSSB
    #group_weights <- sqrt(diag(tcrossprod(GRmat)))
    #group_weights <- rep(1, G)
    if (!is.null(weights_adaptive)) {
      if (weights_adaptive == TRUE) {
        if (is.null(mZ)) {
          coeff_LS <- lm_ols(X = X.std, y = y.std)
          for (g in 1:G) {
            idx              <- which(GRmat[g,] == 1)
            group_weights[g] <- 1.0 / norm(coeff_LS[idx], type = "2")
          }
        } else {
          coeff_LS <- lm_ols_FWL(X = X.std, Z = mZ, y = y.std)$coeff_X
          for (g in 1:G) {
            idx              <- which(GRmat[g,] == 1)
            group_weights[g] <- 1.0 / norm(coeff_LS[idx], type = "2")
          }
        }
      } else {
        group_weights <- rep(1, G)
      }
    } else {
      group_weights <- rep(1, G)
    }
  } else {
    if (overall.group == TRUE) {
      check_group_weights(x = group_weights, n = G, algname = "f2sSP", funname = "group_weights")
      if (length(group_weights) == (G-1)) {
        group_weights <- c(group_weights, sqrt(dim(GRmat)[2]))
      }
    } else {
      check_weights(x = group_weights, n = G, algname = "f2sSP", funname = "group_weights")
    }
  } 
  
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # Manage variable weights
  # le righe che seguono sono state cambiate il 27 ottobre 2025 da Mauro e Marco
  if (is.null(var_weights)) {
   var_weights <- 1.0 / colSums(GRmat)
  }
  # if (is.null(var_weights)) {
  #   var_weights <- 1.0 / sqrt(colSums(GRmat))
  # }
  
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # Get the sequence of lambda
  if (!is.null(nlambda) && is.null(lambda)) {
    # get the smallest value of such that the regression 
    # coefficients estimated by the lasso are all equal to zero
    lambda <- lm_lambdamax_OVGLASSO(y = y.std, X = X.std, mZ,
                                    GRmat = GRmat, group_weights = group_weights, var_weights = var_weights, 
                                    lambda.min.ratio = lambda.min.ratio, maxl = nlambda)$lambda.seq
  }
  if (is.null(nlambda) && is.null(lambda)) {
    # get the smallest value of such that the regression 
    # coefficients estimated by the lasso are all equal to zero
    lambda <- lm_lambdamax_OVGLASSO(y = y.std, X = X.std, mZ,
                                    GRmat = GRmat, group_weights = group_weights, var_weights = var_weights, 
                                    lambda.min.ratio = lambda.min.ratio, maxl = 30)$lambda.seq
  }
  if (!is.null(lambda)) {
    nlambda <- length(lambda)
  }
  
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # Run the ADMM algorithm
  if (is.null(mZ)) {
  
    # check for that diff_order and lambda2 are not NULL
    if ((is.null(diff_order)) || (is.null(lambda2))) {
      ret <- .admm_ovglasso_fast(A = X.std, b = y.std, groups = GRmat,
                                 group_weights = group_weights, var_weights = var_weights,
                                 lambda = lambda, rho_adaptation = adaptation,
                                 rho = rho, tau = tau.ada, mu = mu.ada,
                                 reltol = reltol, abstol = abstol, maxiter = maxit, ping = 0)
    } else {
      if (length(lambda2) == 1) {
        if (lambda2 == 0) {
          ret <- .admm_ovglasso_fast(A = X.std, b = y.std, groups = GRmat,
                                     group_weights = group_weights, var_weights = var_weights,
                                     lambda = lambda, rho_adaptation = adaptation,
                                     rho = rho, tau = tau.ada, mu = mu.ada,
                                     reltol = reltol, abstol = abstol, maxiter = maxit, ping = 0)
        } else {
          ret <- .admm_ovglasso_smo_fast(A = X.std, b = y.std, groups = GRmat,
                                         group_weights = group_weights, var_weights = var_weights,
                                         lambda = lambda, lambda2 = lambda2, diff_order = diff_order, 
                                         rho_adaptation = adaptation,
                                         rho = rho, tau = tau.ada, mu = mu.ada,
                                         reltol = reltol, abstol = abstol, maxiter = maxit, ping = 0)
        }
      } else {
        ret <- .admm_ovglasso_smo_fast(A = X.std, b = y.std, groups = GRmat,
                                       group_weights = group_weights, var_weights = var_weights,
                                       lambda = lambda, lambda2 = lambda2, diff_order = diff_order, 
                                       rho_adaptation = adaptation,
                                       rho = rho, tau = tau.ada, mu = mu.ada,
                                       reltol = reltol, abstol = abstol, maxiter = maxit, ping = 0)
      }
    }
                               
    # get estimated coefficients and path
    if (length(dim(ret$coef.path)) == 2) {
      mSpRegP <- t(ret$coef.path)
      
    } else {
      mSpRegP <- ret$coef.path
    }
    vSpRegP <- ret$coefficients
  } else {
    # check for that diff_order and lambda2 are not NULL
    if ((is.null(diff_order)) || (is.null(lambda2))) {
      ret <- .admm_ovglasso_cov_fast(W = X.std, Z = mZ, y = y.std,
                                     groups = GRmat, group_weights = group_weights, var_weights = var_weights,
                                     lambda = lambda, rho_adaptation = adaptation, rho = rho, tau = tau.ada, mu.ada,
                                     reltol = reltol, abstol = abstol, maxiter = maxit, ping = 0)
    } else {
      if (length(lambda2) == 1) {
        if (lambda2 == 0) {
          ret <- .admm_ovglasso_cov_fast(W = X.std, Z = mZ, y = y.std,
                                         groups = GRmat, group_weights = group_weights, var_weights = var_weights,
                                         lambda = lambda, rho_adaptation = adaptation, rho = rho, tau = tau.ada, mu.ada,
                                         reltol = reltol, abstol = abstol, maxiter = maxit, ping = 0)
        } else {
          ret <- .admm_ovglasso_cov_smo_fast(W = X.std, Z = mZ, y = y.std,
                                             groups = GRmat, group_weights = group_weights, var_weights = var_weights,
                                             lambda = lambda, lambda2 = lambda2, diff_order = diff_order,
                                             rho_adaptation = adaptation, rho = rho, tau = tau.ada, mu.ada,
                                             reltol = reltol, abstol = abstol, maxiter = maxit, ping = 0)
        }
      } else {
        ret <- .admm_ovglasso_cov_smo_fast(W = X.std, Z = mZ, y = y.std,
                                           groups = GRmat, group_weights = group_weights, var_weights = var_weights,
                                           lambda = lambda, lambda2 = lambda2, diff_order = diff_order,
                                           rho_adaptation = adaptation, rho = rho, tau = tau.ada, mu.ada,
                                           reltol = reltol, abstol = abstol, maxiter = maxit, ping = 0)
      }
    }
        
    # get estimated coefficients and path
    if (length(dim(ret$sp.coef.path)) == 2) {
      mSpRegP <- t(ret$sp.coef.path)
      mRegP   <- t(ret$coef.path)
    } else {
      mSpRegP <- ret$sp.coef.path
      mRegP   <- ret$coef.path
    }
    vSpRegP <- ret$sp.coefficients
    vRegP   <- ret$coefficients
  }
  
  if (standardize.data == TRUE) {
    # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    # Get the path and retrieve the scaled estimates
    #path <- array(dim = c(nlambda, L, M), data = t(apply(mRegP, 2, function(x) solve(mU) %*% x %*% mV)))
    #sp.path <- matrix(data = t(apply(mSpRegP, 2, function(x) solve(mU) %*% x %*% mV)), nlambda, M)
    if (length(dim(mSpRegP)) == 2) {
      sp.path <- matrix(data = t(apply(mSpRegP, 2, function(x) solve(mU) %*% x %*% mV)), nlambda, M)
    } else {
      sp.path <- array(
        data = 0,
        dim  = dim(mSpRegP)
      )
      for (k in 1:length(lambda2)) {
        sp.path[,k,] <- matrix(data = t(apply(t(mSpRegP[,k,]), 2, function(x) solve(mU) %*% x %*% mV)), nlambda, M)
      }
    }
    
    # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    # Retrieve estimated parameters
    mSpRegP_ <- matrix(solve(mU) %*% vSpRegP %*% mV, nrow = 1, ncol = M)
    
    # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    # Retrieve estimated parameters (non penalized covariates)
    if (!is.null(mZ)) {
      if (length(dim(mSpRegP)) == 2) {
        path  <- matrix(nlambda, q, data = t(apply(mRegP, 2, function(x) x %*% mV)))
      } else {
        path <- array(
          data = 0,
          dim  = dim(mRegP)
        )
        for (k in 1:length(lambda2)) {
          path[,k,] <- matrix(nlambda, q, data = apply(mRegP[,k,], 2, function(x) x %*% mV))
        }
      }
      vRegP <- vRegP %*% mV
    }
  } else {
    #path   <- array(dim = c(nlambda, L, M), data = t(mRegP))
    #sp.path  <- matrix(data = t(mSpRegP), nlambda, M)
    #mSpRegP_ <- matrix(vSpRegP, nrow = 1, ncol = M)
    #if (!is.null(mZ)) {
    #  path <- t(mRegP)
    #}
    if (length(dim(mSpRegP)) == 2) {
      sp.path  <- matrix(data = t(mSpRegP), nlambda, M)
      if (!is.null(mZ)) {
        path <- t(mRegP)
      }
    } else {
      sp.path <- array(
        data = 0,
        dim  = c(dim(mSpRegP)[1], dim(mSpRegP)[2], dim(mSpRegP)[3])
      )
      if (!is.null(mZ)) {
        path <- array(
          data = 0,
          dim = c(dim(mRegP)[1], dim(mRegP)[2], dim(mRegP)[3])
        )
      }
      for (k in 1:length(lambda2)) {
        sp.path[,k,] <- matrix(data = mSpRegP[,k,], nlambda, M)
        if (!is.null(mZ)) {
          path[,k,] <- mRegP[,k,]
        }
      }
    }
    mSpRegP_ <- matrix(vSpRegP, nrow = 1, ncol = M)
  }
  # get the estimated functions
  # sp.fun.path <- sp.path %*% mH
  # sp.fun      <- mSpRegP_ %*% mH
  if (length(dim(sp.path)) == 2) {
    sp.fun.path <- sp.path %*% mH
  } else {
    sp.fun.path <- array(
      data = 0,
      dim  = c(dim(mSpRegP)[1], dim(mSpRegP)[2], dim(mH)[2])
    )
    for (k in 1:length(lambda2)) {
      sp.fun.path[,k,] <- sp.path[,k,] %*% mH
    }
  }
  sp.fun <- mSpRegP_ %*% mH
  
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # Compute the degrees of freedom 
  if (is.null(mZ)) {
    # compute the DoF
    dof.out <- f2s_dof(W                = X.std, 
                       y                = y.std,
                       coeff            = ret$coef.path, 
                       lambda           = lambda, 
                       lambda2          = lambda2, 
                       diff_order       = diff_order, 
                       adaptive_weights = weights_adaptive,
                       GRmat            = GRmat, 
                       group_weights    = group_weights, 
                       var_weights      = var_weights, 
                       Umat             = ret$U,
                       err_primal       = ret$err_pri,
                       err_dual         = ret$err_dual,
                       rho              = ret$rho,
                       toler_c          = toler_c,
                       toler_d          = toler_d)
  } else {
    dof.out <- f2s_cov_dof(W                = X.std, 
                           Z                = mZ, 
                           y                = y.std,
                           coeff_W          = ret$sp.coef.path, 
                           coeff_Z          = ret$coef.path, 
                           lambda           = lambda, 
                           lambda2          = lambda2, 
                           diff_order       = diff_order,
                           adaptive_weights = weights_adaptive,
                           GRmat            = GRmat, 
                           group_weights    = group_weights, 
                           var_weights      = var_weights, 
                           Umat             = ret$U,
                           err_primal       = ret$err_pri,
                           err_dual         = ret$err_dual,
                           rho              = ret$rho,
                           toler_c          = toler_c,
                           toler_d          = toler_d)
  }
  
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # compute the BIC
  bic     <- dim(X.std)[1] * log(ret$mse) + dof.out$dof * log(dim(X.std)[1]) 
  
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # Print to screen
  if (print.out == TRUE) {
    cat("\n\n\n")
    cat("Function to scalar regression model with overlap group-LASSO penalty\n")
    cat("running time (for ", reltol, " relative error):",
        sum(ret$elapsedTime), "seconds \n\n\n")
  }
  
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # Get output
  res.names <- c("sp.coefficients", 
                 "sp.coef.path", 
                 "sp.fun.path",
                 "sp.fun",
                 "coefficients",
                 "coef.path",
                 "dof",
                 "dof.groups_active",
                 "dof.coef_active",
                 "dof.vars_active",
                 "bic",
                 "lambda.min",
                 "lambda",
                 "lambda2",
                 "GRmat",
                 "group_weights",
                 "var_weights",
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
  if (!is.null(mZ)) {
    res$coefficients <- as.numeric(vRegP)
    res$coef.path    <- path
  } 
  res$sp.coefficients   <- t(mSpRegP_)
  res$sp.coef.path      <- sp.path
  res$sp.fun.path       <- sp.fun.path
  res$sp.fun            <- t(sp.fun)
  res$dof               <- dof.out$dof
  res$dof.groups_active <- dof.out$groups_active
  res$dof.coef_active   <- dof.out$coeff_active
  res$dof.vars_active   <- dof.out$vars_active
  res$bic               <- bic
  res$lambda.min        <- c(ret$lambda.min, ret$lambda2.min)
  res$lambda            <- lambda
  res$lambda2           <- lambda2
  res$GRmat             <- GRmat
  res$group_weights     <- group_weights
  res$var_weights       <- var_weights
  res$mse               <- ret$mse
  res$min.mse           <- ret$min.mse
  res$convergence       <- ret$convergence
  res$elapsedTime       <- ret$elapsedTime
  res$iternum           <- ret$iternum
  res$objfun            <- ret$objfun
  res$r_norm            <- ret$r_norm
  res$s_norm            <- ret$s_norm
  res$err_pri           <- ret$err_pri
  res$err_dual          <- ret$err_dual
  res$rho               <- ret$rho
  
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # Return output
  return(res)
}

#' Cross-validation for Overlap Group Least Absolute Shrinkage and Selection Operator on scalar-on-function regression model
#'
#' Overlap Group-LASSO for scalar-on-function regression model solves the following optimization problem
#' \deqn{\textrm{min}_{\psi,\gamma} ~ \frac{1}{2} \sum_{i=1}^n \left( y_i - \int x_i(t) \psi(t) dt-z_i^\intercal\gamma \right)^2 + \lambda \sum_{g=1}^{G} \Vert S_{g}T\psi \Vert_2}
#' to obtain a sparse coefficient vector \eqn{\psi\in\mathbb{R}^{M}} for the functional penalized predictor \eqn{x(t)} and a coefficient vector \eqn{\gamma\in\mathbb{R}^q} for the unpenalized scalar predictors \eqn{z_1,\dots,z_q}. The regression function is \eqn{\psi(t)=\varphi(t)^\intercal\psi}
#' where \eqn{\varphi(t)} is a B-spline basis of order \eqn{d} and dimension \eqn{M}. 
#' For each group \eqn{g}, each row of the matrix \eqn{S_g\in\mathbb{R}^{d\times M}} has non-zero entries only for those bases belonging 
#' to that group. These values are provided by the arguments \code{groups} and \code{group_weights} (see below). 
#' Each basis function belongs to more than one group. The diagonal matrix \eqn{T\in\mathbb{R}^{M\times M}} contains 
#' the basis specific weights. These values are provided by the argument \code{var_weights} (see below).
#' The regularization path is computed for the overlap group-LASSO penalty at a grid of values for the regularization 
#' parameter \eqn{\lambda} using the alternating direction method of multipliers (ADMM). See Boyd et al. (2011) and Lin et al. (2022) 
#' for details on the ADMM method.
#'  
#' @param vY a length-\eqn{n} vector of observations of the scalar response variable.
#' @param mX a \eqn{(n\times r)} matrix of observations of the functional covariate.
#' @param mZ an \eqn{(n\times q)} full column rank matrix of scalar predictors that are not penalized.  
#' @param M number of elements of the B-spline basis vector \eqn{\varphi(t)}.
#' @param group_weights a vector of length \eqn{G} containing group-specific weights. The default is square root of the group cardinality, see Bernardi et al. (2022).
#' @param var_weights a vector of length \eqn{M} containing basis-specific weights. The default is a vector where 
#' each entry is the reciprocal of the number of groups including that basis. See Bernardi et al. (2022) for details.
#' @param standardize.data logical. Should data be standardized?
#' @param splOrd the order \eqn{d} of the spline basis.
#' @param diff_order order of the discrete difference operator used in the smoothness penalty.
#' The default is 1.
#' @param lambda either a regularization parameter or a vector of regularization parameters.
#' @param lambda2 either a non-negative smoothing regularization parameter or a vector of
#' smoothing regularization parameters. If \code{NULL}, no smoothness penalty is used.
#' In this latter case the routine computes the whole path. If it is NULL values for lambda are provided by the routine.
#' @param lambda.min.ratio smallest value for lambda, as a fraction of the maximum lambda value. If \eqn{n>M}, the default is 0.0001, and if \eqn{n<M}, the default is 0.01. 
#' @param nlambda the number of lambda values - default is 30.
#' @param intercept logical. If it is TRUE, a column of ones is added to the design matrix. 
#' @param overall.group logical. If it is TRUE, an overall group including all penalized covariates is added.
#' @param cv.fold the number of folds - default is 5.
#' @param control	 a list of control parameters for the ADMM algorithm. See ‘Details’.
#'
#' @return A named list containing \describe{
#' \item{sp.coefficients}{a length-\eqn{M} solution vector for the parameters \eqn{\psi}, which corresponds to the minimum cross-validated MSE.}
#' \item{sp.fun}{a length-\eqn{r} vector providing the estimated functional coefficient for \eqn{\psi(t)} corresponding to the minimum cross-validated MSE.}
#' \item{coefficients}{a length-\eqn{q} solution vector for the parameters \eqn{\gamma}, which corresponds to the minimum cross-validated MSE.
#' It is provided only when either the matrix \eqn{Z} in input is not NULL or the intercept is set to TRUE.} 
#' \item{lambda}{sequence of lambda.}
#' \item{lambda.min}{value of lambda that attains the minimum cross-validated MSE.}
#' \item{mse}{cross-validated mean squared error.}
#' \item{min.mse}{minimum value of the cross-validated MSE for the sequence of lambda.}
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
#' ## generate sample data and functional coefficients
#' set.seed(1)
#' n     <- 40
#' p     <- 18                                 
#' r     <- 100
#' s     <- seq(0, 1, length.out = r)
#' 
#' beta_basis <- splines::bs(s, df = p, intercept = TRUE)    # basis
#' coef_data  <- matrix(rnorm(n*floor(p/2)), n, floor(p/2))        
#' fun_data   <- coef_data %*% t(splines::bs(s, df = floor(p/2), intercept = TRUE))     
#' 
#' x_0   <- apply(matrix(rnorm(p, sd=1),p,1), 1, fdaSP::softhresh, 1)  
#' x_fun <- beta_basis %*% x_0                
#' 
#' b     <- fun_data %*% x_fun + rnorm(n, sd = sqrt(crossprod(fun_data %*% x_fun ))/10)
#' l     <- 10^seq(2, -4, length.out = 30)
#' maxit <- 1000
#' 
#' 
#' ## set the hyper-parameters
#' maxit          <- 1000
#' rho_adaptation <- TRUE
#' rho            <- 1
#' reltol         <- 1e-5
#' abstol         <- 1e-5
#' 
#' ## run cross-validation
#' mod_cv <- f2sSP_cv(vY = b, mX = fun_data, M = p,
#'                    group_weights = NULL, var_weights = NULL, standardize.data = FALSE, splOrd = 4,
#'                    lambda = NULL, lambda.min = 1e-5, nlambda = 30, cv.fold = 5, intercept = FALSE, 
#'                    control = list("abstol" = abstol, 
#'                                   "reltol" = reltol, 
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
#' abline(v = log(mod_cv$lambda[which(mod_cv$lambda == mod_cv$lambda.min)]), 
#'        col = "red", lwd = 1.0)
#' 
#' ### comparison with oracle error
#' mod <- f2sSP(vY = b, mX = fun_data, M = p, 
#'              group_weights = NULL, var_weights = NULL, 
#'              standardize.data = FALSE, splOrd = 4,
#'              lambda = NULL, nlambda = 30, 
#'              lambda.min = 1e-5, intercept = FALSE,
#'              control = list("abstol" = abstol, 
#'                             "reltol" = reltol, 
#'                             "adaptation" = rho_adaptation, 
#'                             "rho" = rho, 
#'                             "print.out" = FALSE))
#'                                     
#' err_mod <- apply(mod$sp.coef.path, 1, function(x) sum((x - x_0)^2))
#' plot(log(mod$lambda), err_mod, type = "l", col = "blue", 
#'      lwd = 2, xlab = latex2exp::TeX("$\\log(\\lambda)$"), 
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
f2sSP_cv <- function(vY, mX, mZ = NULL, M,
                     group_weights = NULL, var_weights = NULL, standardize.data = FALSE, 
                     splOrd = 4, diff_order = 1,
                     lambda = NULL, lambda.min.ratio = NULL, nlambda = NULL, lambda2 = NULL,
                     cv.fold = 5, intercept = FALSE, overall.group = FALSE, control = list()) {
    
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # Pre-processing
  # data validity
  if (!check_data_matrix(mX)) {
    stop("* f2sSP_cv : input 'mX' is invalid data matrix.")
  }
  if (!check_data_vector(vY)) {
    stop("* f2sSP_cv : input 'vY' is invalid data vector.")
  }
  # data size
  if (dim(mX)[1] != length(vY)) {
    stop("* f2sSP_cv : two inputs 'mX' and 'vY' have non-matching dimension.")
  }
  if (!is.null(mZ)) {
    if (dim(mZ)[1] != length(vY)) {
      stop("* f2sSP_cv : two inputs 'mZ' and 'vY' have non-matching dimension.")
    }
  }
  if (M <= 0) {
    stop("* f2sSP_cv : the number of basis must be a positive integer!")
  }
  if (is.null(diff_order) || is.null(lambda2)) {
    diff_order <- NULL
    lambda2    <- NULL
  }
  if (!is.null(diff_order)) {
    if (diff_order < 1) {
      stop("* f2sSP : the parameter diff_order is less than one!")
    }
  }
  
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # Get dimensions
  n <- dim(mX)[1]             # number of observations
  p <- dim(mX)[2]
  
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # Create the B-spline basis functions
  s  <- seq(0, 1, length.out = p)
  mH <- t(splines::bs(s, df = M, intercept = TRUE))
  
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # Define response and design matrix
  mX_B <- tcrossprod(mX, mH)
  
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # Vectorisation form
  X_ <- mX_B
  y_ <- matrix(vY, nrow = n, ncol = 1)
  
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # add constant
  if (standardize.data == TRUE) {
    if (intercept == TRUE){
      intercept = FALSE
      warning("* f2sSP_cv : intercept should be set to FALSE is data are standardized!\n")
    }
  }
  if (intercept == TRUE) {
    if (!is.null(mZ)) {
      mZ <- cbind(matrix(data = 1, nrow = n, ncol = 1), mZ)
    } else {
      mZ <- matrix(data = 1, nrow = n, ncol = 1)
    }
  }
  #q <- dim(mZ)[2]
  q <- if (is.null(mZ)) 0L else ncol(mZ)
  
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
  output <- f2s_reg_RIDGE(y = y.std, Z = mZ, X = X.std, 
                          diff_order = diff_order, 
                          lambda = lambda, lambda2 = lambda2)
  
  # retrieve estimate coefficients in case standardize is TRUE
  if (!is.null(output)) {
    mSpRegP <- output$sp.coef.path
    vSpRegP <- output$vSpRegP
    if (!is.null(mZ)) { 
      mRegP <- output$coef.path
      vRegP <- output$vRegP
    }
    # Get the scaled path
    if (standardize.data == TRUE) {
      if (length(lambda2) > 1) {
        sp.path <- matrix(data = t(apply(mSpRegP, 2, function(x) solve(mU) %*% x %*% mV)), nrow = length(lambda2), ncol = M)
      } else {
        sp.path <- matrix(data = t(solve(mU) %*% mSpRegP %*% mV), nrow = 1, ncol = M)
      }
      vSpRegP <- matrix(solve(mU) %*% vSpRegP %*% mV, nrow = 1, ncol = M)
      if (!is.null(mZ)) {
        if (length(lambda2) > 1) {
          path <- matrix(length(lambda2), q, data = t(apply(mRegP, 2, function(x) x %*% mV)))
        } else {
          path <- matrix(1, q, data = t(mRegP %*% mV))
        }
        vRegP <- vRegP %*% mV
      }
    } else {
      sp.path <- matrix(data = t(mSpRegP), length(lambda2), M)
      if (!is.null(mZ)) {
        path  <- t(mRegP)
      }
    }
  }
  
  # retrieve estimated functins and path
  if (!is.null(output)) {
    sp.fun.path <- sp.path %*% mH
    sp.fun      <- vSpRegP %*% mH
  }
  
  # Get output
  if (!is.null(output)) {
    res.names <- c("sp.coefficients", 
                   "coefficients",
                   "sp.fun",
                   "mse",
                   "min.mse",
                   "lambda",
                   "lambda2") 
    res        <- vector(mode = "list", length = length(res.names))
    names(res) <- res.names
    if (!is.null(mZ)) {
      res$coefficients <- as.numeric(vRegP)
    } 
    res$sp.coefficients   <- t(mSpRegP)
    res$sp.fun            <- t(sp.fun)
    res$lambda            <- lambda
    res$lambda2           <- lambda2
    res$mse               <- output$mse
    res$min.mse           <- output$min.mse
    
    # Return output
    return(res)
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
  maxit      <- con$maxit
  abstol     <- con$abstol
  reltol     <- con$reltol
  adaptation <- con$adaptation
  rho        <- con$rho
  tau.ada    <- con$tau.ada      
  mu.ada     <- con$mu.ada
  par.init   <- con$par.init
  init.rnd   <- con$init.rnd
  print.out  <- con$print.out
  
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # Pre-processing
  # data validity
  if (!check_param_constant_multiple(c(abstol, reltol))) {
    stop("* f2sSP_cv : tolerance level is invalid.")
  }
  if (!check_param_integer(maxit, 0.0)) {
    stop("* f2sSP_cv : 'maxit' should be a positive integer.")
  }
  maxit = as.integer(maxit)
  if (!check_param_constant(rho, 0.0)) {
    stop("* f2sSP_cv : 'rho' should be a positive real number.")
  }
  if (!check_param_constant(mu.ada, 0.0)) {
    stop("* f2sSP_cv : 'mu' should be a positive real number.")
  }
  if ((tau.ada < 1.0) || (tau.ada > 2.0)) {
    warning("* f2sSP_cv : 'tau' value is suggested to be in [1,2].")
  }
  
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # Pre-processing: variable weights
  check_weights(x = var_weights, n = dim(X_)[2], algname = "f2sSP", funname = "var_weights")
  
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # Get the groups of the overlap group LASSO penalty
  if (overall.group == TRUE) {
    out <- penfun(method = "ogl&1", p = M, q = 1, splOrd = splOrd, regpars = NULL)
  } else {
    out <- penfun(method = "ogl", p = M, q = 1, splOrd = splOrd, regpars = NULL)
  }
  GRmat <- out$grMat
  G     <- dim(GRmat)[1]
  nG    <- rowSums(GRmat)
  
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # Manage group & overall group weights
  if (is.null(group_weights)) {
    # Define the weights as in Yuan and Lin 2006, JRSSB
    #group_weights <- sqrt(diag(tcrossprod(GRmat)))
    group_weights <- rep(1, G)
  } else {
    if (overall.group == TRUE) {
      check_group_weights(x = group_weights, n = G, algname = "f2sSP_cv", funname = "group_weights")
      if (length(group_weights) == (G-1)) {
        group_weights <- c(group_weights, sqrt(dim(GRmat)[2]))
      }
    } else {
      check_weights(x = group_weights, n = G, algname = "f2sSP_cv", funname = "group_weights")
    }
  } 
  
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # Manage variable weights
  #if (is.null(var_weights)) {
  #  var_weights <- 1.0 / colSums(GRmat)
  #}
  if (is.null(var_weights)) {
    var_weights <- 1.0 / sqrt(colSums(GRmat))
  }

  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # Get the sequence of lambda
  if (!is.null(nlambda) && is.null(lambda)) {
    # get the smallest value of such that the regression 
    # coefficients estimated by the lasso are all equal to zero
    lambda <- lm_lambdamax_OVGLASSO(y = y.std, X = X.std, mZ,
                                    GRmat = GRmat, group_weights = group_weights, var_weights = var_weights, 
                                    lambda.min.ratio = lambda.min.ratio, maxl = nlambda)$lambda.seq
  }
  if (is.null(nlambda) && is.null(lambda)) {
    # get the smallest value of such that the regression 
    # coefficients estimated by the lasso are all equal to zero
    lambda <- lm_lambdamax_OVGLASSO(y = y.std, X = X.std, mZ,
                                    GRmat = GRmat, group_weights = group_weights, var_weights = var_weights, 
                                    lambda.min.ratio = lambda.min.ratio, maxl = 30)$lambda.seq
  }
  if (!is.null(lambda)) {
    nlambda <- length(lambda)
  }

  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # Get groups
  ret.gr <- groups.cv(n = n, k = cv.fold)
  
  # store Lasso regression for each fold
  if (!is.null(lambda2)) {
    aMse_cv <- array(data = 0.0, dim = c(cv.fold, nlambda, length(lambda2)))
  } else {
    #aMse_cv <- matrix(data = 0.0, dim = c(cv.fold, nlambda))
    aMse_cv <- matrix(data = 0.0, nrow = cv.fold, ncol = nlambda)
  }
  
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # Initial time 
  startTime <- Sys.time()

  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # Run lasso
  for (kt in 1:cv.fold) {
    # run ADMM
    mX_ <- X.std[ret.gr$groups.cv[[kt]],]
    vY_ <- y.std[ret.gr$groups.cv[[kt]]]
    if (is.null(mZ)) {

      # check for that diff_order and lambda2 are not NULL
      if ((is.null(diff_order)) || (is.null(lambda2))) {
        ret <- .admm_ovglasso_fast(A = mX_, b = vY_, groups = GRmat, group_weights = group_weights, var_weights = var_weights,
                                   lambda = lambda, rho_adaptation = adaptation, rho = rho, tau = tau.ada, mu = mu.ada,
                                   reltol = reltol, abstol = abstol, maxiter = maxit, ping = 0)
      } else {
        ret <- .admm_ovglasso_smo_fast(A = mX_, b = vY_, groups = GRmat, group_weights = group_weights, var_weights = var_weights,
                                       lambda = lambda, lambda2 = lambda2, diff_order = diff_order, 
                                       rho_adaptation = adaptation, rho = rho, tau = tau.ada, mu = mu.ada,
                                       reltol = reltol, abstol = abstol, maxiter = maxit, ping = 0)
      }

      # fit 
      mSpRegP_ <- ret$coef.path
      if (length(dim(mSpRegP_)) == 3) {
        fit      <- array(
          data = 0,
          dim  = c(n-dim(mX_)[1], dim(mSpRegP_)[1], dim(mSpRegP_)[2])
        )
        for (k in 1:length(lambda2)) {
          fit[,,k] <- X.std[ret.gr$groups.pred[[kt]],] %*% t(mSpRegP_[,k,])
        }
      } else {
        fit <- X.std[ret.gr$groups.pred[[kt]],] %*% t(mSpRegP_)
      }
    } else {
      mZ_ <- as.matrix(mZ[ret.gr$groups.cv[[kt]],], ncol = 1)
    
      # check for that diff_order and lambda2 are not NULL
      if ((is.null(diff_order)) || (is.null(lambda2))) {
        ret <- .admm_ovglasso_cov_fast(W = mX_, Z = mZ_, y = vY_, groups = GRmat, 
                                       group_weights = group_weights, var_weights = var_weights,
                                       lambda = lambda, rho_adaptation = adaptation, rho = rho, tau = tau.ada, mu = mu.ada,
                                       reltol = reltol, abstol = abstol, maxiter = maxit, ping = 0)
      } else {
        ret <- .admm_ovglasso_cov_smo_fast(W = mX_, Z = mZ_, y = vY_, groups = GRmat, 
                                           group_weights = group_weights, var_weights = var_weights,
                                           lambda = lambda, lambda2 = lambda2, diff_order = diff_order, 
                                           rho_adaptation = adaptation, rho = rho, tau = tau.ada, mu = mu.ada,
                                           reltol = reltol, abstol = abstol, maxiter = maxit, ping = 0)
      }
      
      # fit
      mSpRegP_ <- ret$sp.coef.path
      mRegP_   <- ret$coef.path
      if (length(dim(mSpRegP_)) == 3) {
        fit      <- array(
          data = 0,
          dim  = c(n-dim(mX_)[1], dim(mSpRegP_)[1], dim(mSpRegP_)[2])
        )
        for (k in 1:length(lambda2)) {
          fit[,,k] <- X.std[ret.gr$groups.pred[[kt]],] %*% t(mSpRegP_[,k,]) + mZ[ret.gr$groups.pred[[kt]],] %*% t(mRegP_[,k,])
        }
      } else {
        fit <- X.std[ret.gr$groups.pred[[kt]],] %*% t(mSpRegP_) + mZ[ret.gr$groups.pred[[kt]],] %*% t(mRegP_)
      }
    }
    # Evaluate MSE
    if (length(dim(mSpRegP_)) == 3) {
      n_            <- length(y.std[ret.gr$groups.pred[[kt]]])
      mse           <- apply(fit, c(2, 3), function(x) {sum((y.std[ret.gr$groups.pred[[kt]]] - x)^2)/n_})
      aMse_cv[kt,,] <- mse
    } else {
      n_           <- length(y.std[ret.gr$groups.pred[[kt]]])
      mse          <- apply(fit, 2, function(x) {sum((y.std[ret.gr$groups.pred[[kt]]] - x)^2)/n_})
      aMse_cv[kt,] <- mse
    }
  }
  
  # get mean squared error
  if (length(dim(mSpRegP_)) == 2) {
    mMse <- apply(aMse_cv, 2, mean) 
    mSd  <- apply(aMse_cv, 2, sd) 
  } else {
    mMse <- apply(aMse_cv, c(2, 3), mean) 
    mSd  <- apply(aMse_cv, c(2, 3), sd) 
  }
  
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # system time
  endTime <- Sys.time()
  
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # compute run time
  eltime <- difftime(time1 = endTime, time2 = startTime, units = "secs")
  
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # Get lasso output
  # min.mse      <- min(vMse)
  # indi.min.mse <- which(vMse == min.mse) 
  # lambda.min   <- lambda[indi.min.mse]
  if (length(dim(mSpRegP_)) == 2) {
    min.mse      <- min(mMse)
    indi.min.mse <- which(mMse == min.mse) 
    lambda.min   <- lambda[indi.min.mse]
  } else {
    min.mse      <- min(mMse)
    indi.min.mse <- which(mMse == min.mse, arr.ind = TRUE)[1,, drop = FALSE]
    lambda.grid  <- expand.grid(lambda = lambda, lambda2 = lambda2)
    lambda.min   <- c(lambda[indi.min.mse[1]], lambda2[indi.min.mse[2]])
  }
  
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # get estimate over the full sample
  if (length(dim(mSpRegP_)) == 2) {
    rho_ <- ret$rho[[indi.min.mse]]
    u_   <- t(ret$U[indi.min.mse,])
    z_   <- t(ret$Z[indi.min.mse,])
  } else {
    rho_ <- ret$rho[[indi.min.mse[1], indi.min.mse[2]]]
    u_   <- t(ret$U[indi.min.mse[1],indi.min.mse[2],])
    z_   <- t(ret$Z[indi.min.mse[1],indi.min.mse[2],])
  }
  if (is.null(mZ)) {
    
    # check for that diff_order and lambda2 are not NULL
    if ((is.null(diff_order)) || (is.null(lambda2))) {
      ret  <- .admm_ovglasso(A = X.std, b = y.std,
                             groups = GRmat, group_weights = group_weights, var_weights = var_weights,
                             u = u_, z = z_, lambda = lambda.min,
                             rho_adaptation = adaptation, rho = rho_[length(rho_)], tau = tau.ada, mu = mu.ada,
                             reltol = reltol, abstol = abstol, maxiter = maxit, ping = 0)
    } else {
      ret  <- .admm_ovglasso_smo(A = X.std, b = y.std,
                                 groups = GRmat, group_weights = group_weights, var_weights = var_weights,
                                 u = u_, z = z_, lambda = lambda.min[1], lambda2 = lambda.min[2], diff_order = diff_order,
                                 rho_adaptation = adaptation, rho = rho_[length(rho_)], tau = tau.ada, mu = mu.ada,
                                 reltol = reltol, abstol = abstol, maxiter = maxit, ping = 0)
    }
    
    # get estimated parameters
    vSpRegP <- ret$x
  } else {
    # check for that diff_order and lambda2 are not NULL
    if ((is.null(diff_order)) || (is.null(lambda2))) {
      ret  <- .admm_ovglasso_cov(W = X.std, Z = mZ, y = y.std,
                                 groups = GRmat, group_weights = group_weights, var_weights = var_weights,
                                 u = u_, z = z_, lambda = lambda.min,
                                 rho_adaptation = adaptation, rho = rho_[length(rho_)], tau = tau.ada, mu = mu.ada,
                                 reltol = reltol, abstol = abstol, maxiter = maxit, ping = 0)
    } else {
      ret  <- .admm_ovglasso_cov_smo(W = X.std, Z = mZ, y = y.std,
                                     groups = GRmat, group_weights = group_weights, var_weights = var_weights,
                                     u = u_, z = z_, lambda = lambda.min[1], lambda2 = lambda.min[2], diff_order = diff_order,
                                     rho_adaptation = adaptation, rho = rho_[length(rho_)], tau = tau.ada, mu = mu.ada,
                                     reltol = reltol, abstol = abstol, maxiter = maxit, ping = 0)
    }
  
    # get estimated parameters
    vSpRegP <- ret$x
    vRegP   <- ret$v
  }
  
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # Store output: full sample estimate
  niter <- ret$niter
  if (niter < maxit) {
    converged <- TRUE
  } else {
    converged <- FALSE
  }
  dConv  <- ifelse(converged == TRUE, 1, 0) 
  nIterN <- ret$niter
  
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # Retrieve estimated parameters
  if (standardize.data == TRUE) {
    # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    # Retrieve estimated parameters
    mSpRegP <- matrix(solve(mU) %*% vSpRegP %*% mV, nrow = 1, ncol = M)
    
    # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    # Retrieve estimated parameters (non penalized covariates)
    if (!is.null(mZ)) {
      vRegP <- vRegP %*% mV
    }
  } else {
    mSpRegP <- matrix(vSpRegP, nrow = 1, ncol = M)
  }
  # get the estimated functions
  sp.fun <- mSpRegP %*% mH

  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # Print to screen
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  if (print.out == TRUE) {
    cat("\n\n\n")
    cat("Function to scalar regression model with overlap group-LASSO penalty, Cross Validation\n")
    cat("Alternating direction method of multipliers\n")
    cat("running time (for ", reltol, " relative error):",
        eltime, "seconds \n\n\n")
  }
  
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # Get output
  res.names <- c("sp.coefficients", 
                 "coefficients",
                 "sp.fun",
                 "mse",
                 "mse.sd",
                 "min.mse",
                 "lambda",
                 "lambda2",
                 "lambda.min",
                 "lambda2.min",
                 "convergence",
                 "elapsedTime",
                 "iternum",
                 "indi.min.mse") 
  res        <- vector(mode = "list", length = length(res.names))
  names(res) <- res.names
  
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # Get output
  ret                 <- NULL
  ret$sp.coefficients <- t(mSpRegP)
  if (!is.null(mZ)) {
    ret$coefficients <- t(vRegP)
  }
  ret$sp.fun       <- t(sp.fun)
  ret$mse          <- mMse
  ret$mse.sd       <- mSd
  ret$min.mse      <- min.mse
  ret$lambda       <- lambda
  ret$lambda2      <- lambda2
  ret$lambda.min   <- lambda.min[1]
  #ret$lambda2.min  <- lambda.min[2]
  ret$lambda2.min  <- if (!is.null(lambda2)) lambda.min[2] else NULL
  ret$convergence  <- dConv
  ret$elapsedTime  <- eltime
  ret$iternum      <- nIterN
  ret$indi.min.mse <- indi.min.mse

  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # Return output
  return(ret)
}

