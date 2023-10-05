# Authors: Mauro Bernardi 
#          Department Statistical Sciences
#       	 University of Padova
#      	   Via Cesare Battisti, 241
#      	   35121 PADOVA, Italy
#          E-mail: mauro.bernardi@unipd.it  

# Last change: March 16, 2023

#' Sparse Adaptive Overlap Group Least Absolute Shrinkage and Selection Operator
#'
#' Sparse Adaptive overlap group-LASSO, or sparse adaptive group \eqn{L_2}-regularized regression, solves the following optimization problem
#' \deqn{\textrm{min}_{\beta,\gamma} ~ \frac{1}{2}\|y-X\beta-Z\gamma\|_2^2 + \lambda\Big[(1-\alpha) \sum_{g=1}^G \|S_g T\beta\|_2+\alpha\Vert T_1\beta\Vert_1\Big]}
#' to obtain a sparse coefficient vector \eqn{\beta\in\mathbb{R}^p} for the matrix of penalized predictors \eqn{X} and a coefficient vector \eqn{\gamma\in\mathbb{R}^q} 
#' for the matrix of unpenalized predictors \eqn{Z}. For each group \eqn{g}, each row of 
#' the matrix \eqn{S_g\in\mathbb{R}^{n_g\times p}} has non-zero entries only for those variables belonging 
#' to that group. These values are provided by the arguments \code{groups} and \code{group_weights} (see below). 
#' Each variable can belong to more than one group. The diagonal matrix \eqn{T\in\mathbb{R}^{p\times p}} contains the variable-specific weights. These values are
#' provided by the argument \code{var_weights} (see below). The diagonal matrix \eqn{T_1\in\mathbb{R}^{p\times p}} contains 
#' the variable-specific \eqn{L_1} weights. These values are provided by the argument \code{var_weights_L1} (see below).
#' The regularization path is computed for the sparse adaptive overlap group-LASSO penalty at a grid of values for the regularization 
#' parameter \eqn{\lambda} using the alternating direction method of multipliers (ADMM). See Boyd et al. (2011) and Lin et al. (2022) 
#' for details on the ADMM method. The regularization is a combination of \eqn{L_2} and
#' \eqn{L_1} simultaneous constraints. Different specifications of the \code{penalty} argument lead to different models choice:
#' \describe{
#' \item{LASSO}{The classical Lasso regularization (Tibshirani, 1996) can be obtained by specifying \eqn{\alpha = 1} 
#' and the matrix \eqn{T_1} as the \eqn{p \times p} identity matrix. An adaptive version of this model (Zou, 2006) can be obtained if \eqn{T_1} is 
#' a \eqn{p \times p}  diagonal matrix of adaptive weights. See also Hastie et al. (2015) for further details.}
#' \item{GLASSO}{The group-Lasso regularization (Yuan and Lin, 2006) can be obtained by specifying \eqn{\alpha = 0}, 
#' non-overlapping groups in \eqn{S_g} and by setting the matrix \eqn{T} equal to the \eqn{p \times p} identity matrix. 
#' An adaptive version of this model can be obtained if the matrix \eqn{T} is a \eqn{p \times p} diagonal matrix of adaptive weights. See also Hastie et al. (2015) for further details.}
#' \item{spGLASSO}{The sparse group-Lasso regularization (Simon et al., 2011) can be obtained by specifying \eqn{\alpha\in(0,1)},  
#' non-overlapping groups in \eqn{S_g} and by setting the matrices \eqn{T} and \eqn{T_1} equal to the \eqn{p \times p} identity matrix. 
#' An adaptive version of this model can be obtained if the matrices \eqn{T} and \eqn{T_1} are \eqn{p \times p}  
#' diagonal matrices of adaptive weights.}
#' \item{OVGLASSO}{The overlap group-Lasso regularization (Jenatton et al., 2011) can be obtained by specifying 
#' \eqn{\alpha = 0}, overlapping groups in \eqn{S_g} and by setting the matrix \eqn{T} equal to the \eqn{p \times p} identity matrix. An adaptive version of this model can be obtained if the matrix \eqn{T} is a \eqn{p \times p}  
#' diagonal matrix of adaptive weights.}
#' \item{spOVGLASSO}{The sparse overlap group-Lasso regularization (Jenatton et al., 2011) can be obtained by specifying 
#' \eqn{\alpha\in(0,1)}, overlapping groups in \eqn{S_g} and by setting the matrices \eqn{T} and \eqn{T_1} equal to the \eqn{p \times p} identity matrix. 
#' An adaptive version of this model can be obtained if the matrices \eqn{T} and \eqn{T_1} are \eqn{p \times p}  diagonal matrices of adaptive weights.}
#' }
#'  
#' @param X an \eqn{(n\times p)} matrix of penalized predictors.
#' @param Z an \eqn{(n\times q)} full column rank matrix of predictors that are not penalized. 
#' @param y a length-\eqn{n} response vector.
#' @param penalty choose one from the following options: 'LASSO', for the or adaptive-Lasso penalties, 'GLASSO', 
#' for the group-Lasso penalty, 'spGLASSO', for the sparse group-Lasso penalty, 'OVGLASSO', 
#' for the overlap group-Lasso penalty and 'spOVGLASSO', for the sparse overlap group-Lasso penalty.
#' @param groups either a vector of length \eqn{p} of consecutive integers describing the grouping of the coefficients, 
#' or a list with two elements: the first element is a vector of length \eqn{\sum_{g=1}^G n_g} containing the variables belonging to each group, where \eqn{n_g} is the cardinality of the \eqn{g}-th group, 
#' while the second element is a vector of length \eqn{G} containing the group lengths (see example below).
#' @param group_weights a vector of length \eqn{G} containing group-specific weights. The default is square root of the group cardinality, see Yuan and Lin (2006).
#' @param var_weights a vector of length \eqn{p} containing variable-specific weights. The default is a vector of ones.
#' @param var_weights_L1 a vector of length \eqn{p} containing variable-specific weights for the \eqn{L_1} penalty. The default is a vector of ones.
#' @param standardize.data logical. Should data be standardized?
#' @param lambda either a regularization parameter or a vector of regularization parameters. In this latter case the routine computes the whole path. If it is NULL values for lambda are provided by the routine.
#' @param alpha the sparse overlap group-LASSO mixing parameter, with \eqn{0\leq\alpha\leq1}. This setting is only available for the sparse group-LASSO and the sparse overlap group-LASSO penalties, otherwise it is set to NULL. The LASSO and group-LASSO penalties are obtained
#' by specifying \eqn{\alpha = 1} and \eqn{\alpha = 0}, respectively.
#' @param lambda.min.ratio smallest value for lambda, as a fraction of the maximum lambda value. If \eqn{n>p}, the default is 0.0001, and if \eqn{n<p}, the default is 0.01. 
#' @param nlambda the number of lambda values - default is 30.
#' @param intercept logical. If it is TRUE, a column of ones is added to the design matrix. 
#' @param overall.group logical. This setting is only available for the overlap group-LASSO and the sparse overlap group-LASSO penalties, otherwise it is set to NULL. If it is TRUE, an overall group including all penalized covariates is added.
#' @param control	 a list of control parameters for the ADMM algorithm. See ‘Details’.
#'
#' @return A named list containing \describe{
#' \item{sp.coefficients}{a length-\eqn{p} solution vector for the parameters \eqn{\beta}. If \eqn{n_\lambda>1} then  the provided vector corresponds to the minimum in-sample MSE.}
#' \item{coefficients}{a length-\eqn{q} solution vector for the parameters \eqn{\gamma}. If \eqn{n_\lambda>1} then the provided vector corresponds to the minimum in-sample MSE.
#' It is provided only when either the matrix \eqn{Z} in input is not NULL or the intercept is set to TRUE.}
#' \item{sp.coef.path}{an \eqn{(n_\lambda\times p)} matrix of estimated \eqn{\beta} coefficients for each lambda of the provided sequence.}
#' \item{coef.path}{an \eqn{(n_\lambda\times q)} matrix of estimated \eqn{\gamma} coefficients for each lambda of the provided sequence.
#' It is provided only when either the matrix \eqn{Z} in input is not NULL or the intercept is set to TRUE.}
#' \item{lambda}{sequence of lambda.}
#' \item{lambda.min}{value of lambda that attains the minimum in sample MSE.}
#' \item{mse}{in-sample mean squared error.}
#' \item{min.mse}{minimum value of the in-sample MSE for the sequence of lambda.}
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
#' \item{adaptation}{logical. If it is TRUE, ADMM with adaptation is performed. The default value is TRUE. See Boyd et al. (2011) for details.}
#' \item{rho}{an augmented Lagrangian parameter. The default value is 1.}
#' \item{tau.ada}{an adaptation parameter greater than one. Only needed if adaptation = TRUE. The default value is 2. See Boyd et al. (2011) for details.}
#' \item{mu.ada}{an adaptation parameter greater than one. Only needed if adaptation = TRUE. The default value is 10. See Boyd et al. (2011) for details.}
#' \item{abstol}{absolute tolerance stopping criterion. The default value is sqrt(sqrt(.Machine$double.eps)).}
#' \item{reltol}{relative tolerance stopping criterion. The default value is sqrt(.Machine$double.eps).}
#' \item{maxit}{maximum number of iterations. The default value is 100.}
#' \item{print.out}{logical. If it is TRUE, a message about the procedure is printed. The default value is TRUE.}
#' }
#' 
#' @examples
#' 
#' ### generate sample data
#' set.seed(2023)
#' n    <- 50
#' p    <- 30 
#' X    <- matrix(rnorm(n*p), n, p)
#' 
#' ### Example 1, LASSO penalty
#' 
#' beta <- apply(matrix(rnorm(p, sd = 1), p, 1), 1, fdaSP::softhresh, 1.5)
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
#' mod <- lmSP(X = X, y = y, penalty = "LASSO", standardize.data = FALSE, intercept = FALSE, 
#'             lambda = lam, control = list("adaptation" = adaptation, "rho" = rho, 
#'                                          "maxit" = maxit, "reltol" = reltol, 
#'                                          "abstol" = abstol, "print.out" = FALSE)) 
#' 
#' ### graphical presentation
#' matplot(log(lam), mod$sp.coef.path, type = "l", main = "Lasso solution path",
#'         bty = "n", xlab = latex2exp::TeX("$\\log(\\lambda)$"), ylab = "")
#' 
#' ### Example 2, sparse group-LASSO penalty
#' 
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
#' mod <- lmSP(X = X, y = y, penalty = "spGLASSO", groups = group1, standardize.data = FALSE,  
#'             intercept = FALSE, lambda = lam, alpha = 0.5, 
#'             control = list("adaptation" = adaptation, "rho" = rho, 
#'                            "maxit" = maxit, "reltol" = reltol, "abstol" = abstol, 
#'                            "print.out" = FALSE)) 
#' 
#' ### graphical presentation
#' matplot(log(lam), mod$sp.coef.path, type = "l", main = "Sparse Group Lasso solution path",
#'         bty = "n", xlab = latex2exp::TeX("$\\log(\\lambda)$"), ylab = "")
#' 
#' @references
#' \insertRef{bernardi_etal.2022}{fdaSP}
#' 
#' \insertRef{boyd_etal.2011}{fdaSP}
#'  
#' \insertRef{hastie_etal.2015}{fdaSP}
#' 
#' \insertRef{jenatton_etal.2011}{fdaSP}
#' 
#' \insertRef{lin_etal.2022}{fdaSP} 
#' 
#' \insertRef{simon_etal.2013}{fdaSP}
#' 
#' \insertRef{yuan_lin.2006}{fdaSP}
#' 
#' \insertRef{zou.2006}{fdaSP}
#' 
#' 
#' @export lmSP
lmSP <- function(X, Z = NULL, y, penalty = c("LASSO", "GLASSO", "spGLASSO", "OVGLASSO", "spOVGLASSO"), 
                 groups, group_weights = NULL, var_weights = NULL, var_weights_L1 = NULL, 
                 standardize.data = TRUE, intercept = FALSE, overall.group = FALSE,
                 lambda = NULL, alpha = NULL, lambda.min.ratio = NULL, nlambda = 30, 
                 control = list()) {
    
  # :::::::::::::::::::::::::::::::::::::::::::::::::::::
  # check the penalty input
  if (!strcmpi(penalty, "LASSO") && !strcmpi(penalty, "GLASSO") && !strcmpi(penalty, "spGLASSO") && !strcmpi(penalty, "OVGLASSO")) {
    stop("* lmSP : input 'penalty' must be correctly specified: 'LASSO', 'GLASSO', 'spGLASSO', 'OVGLASSO', 'spOVGLASSO'.") 
  }
  
  # :::::::::::::::::::::::::::::::::::::::::::::::::::::
  # run linear regression with LASSO-type penalty (means, LASSO or adaptive-LASSO)
  if (strcmpi(penalty, "LASSO")) {
    res <- linreg_ADMM_LASSO(X = X, Z = Z, y = y, var_weights = var_weights, standardize.data = standardize.data, 
                             lambda = lambda, lambda.min.ratio = lambda.min.ratio, nlambda = nlambda, 
                             intercept = intercept, control = control) 
  }
  # :::::::::::::::::::::::::::::::::::::::::::::::::::::
  # run linear regression with group LASSO-type penalty 
  if (strcmpi(penalty, "GLASSO")) {
    res <- linreg_ADMM_GLASSO(X = X, Z = Z, y = y, groups = groups, group_weights = group_weights, var_weights = var_weights, 
                              standardize.data = standardize.data,
                              lambda = lambda, lambda.min.ratio = lambda.min.ratio, nlambda = nlambda, 
                              intercept = intercept, control = control)
  }
  # :::::::::::::::::::::::::::::::::::::::::::::::::::::
  # run linear regression with sparse group LASSO-type penalty 
  if (strcmpi(penalty, "spGLASSO")) {
    res <- linreg_ADMM_spGLASSO(X = X, Z = Z, y = y, groups = groups, group_weights = group_weights, var_weights = var_weights, 
                                var_weights_L1 = var_weights_L1, standardize.data = standardize.data, 
                                lambda = lambda, alpha = alpha, lambda.min.ratio = lambda.min.ratio, nlambda = nlambda, 
                                intercept = intercept, control = control)
  }
  # :::::::::::::::::::::::::::::::::::::::::::::::::::::
  # run linear regression with overlap group LASSO-type penalty 
  if (strcmpi(penalty, "OVGLASSO")) {
    res <- linreg_ADMM_OVGLASSO(X = X, Z = Z, y = y, groups = groups, group_weights = group_weights, var_weights = var_weights, 
                                standardize.data = standardize.data,
                                lambda = lambda, lambda.min.ratio = lambda.min.ratio, nlambda = nlambda,
                                intercept = intercept, overall.group = overall.group, control = control)
  }
  # :::::::::::::::::::::::::::::::::::::::::::::::::::::
  # run linear regression with overlap group LASSO-type penalty 
  if (strcmpi(penalty, "spOVGLASSO")) {
    res <- linreg_ADMM_spOVGLASSO(X = X, Z = Z, y = y, groups = groups, group_weights = group_weights, var_weights = var_weights, 
                                  var_weights_L1 = var_weights_L1, standardize.data = standardize.data,
                                  lambda = lambda, alpha = alpha, lambda.min.ratio = lambda.min.ratio, nlambda = nlambda,
                                  intercept = intercept, overall.group = overall.group, control = control)
  }
  # :::::::::::::::::::::::::::::::::::::::::::::::::::::
  # return
  return(res)
}

#' Cross-validation for Sparse Adaptive Overlap Group Least Absolute Shrinkage and Selection Operator
#'
#' Sparse Adaptive overlap group-LASSO, or sparse adaptive group \eqn{L_2}-regularized regression, solves the following optimization problem
#' \deqn{\textrm{min}_{\beta,\gamma} ~ \frac{1}{2}\|y-X\beta-Z\gamma\|_2^2 + \lambda\Big[(1-\alpha) \sum_{g=1}^G \|S_g T\beta\|_2+\alpha\Vert T_1\beta\Vert_1\Big]}
#' to obtain a sparse coefficient vector \eqn{\beta\in\mathbb{R}^p} for the matrix of penalized predictors \eqn{X} and a coefficient vector \eqn{\gamma\in\mathbb{R}^q} 
#' for the matrix of unpenalized predictors \eqn{Z}. For each group \eqn{g}, each row of 
#' the matrix \eqn{S_g\in\mathbb{R}^{n_g\times p}} has non-zero entries only for those variables belonging 
#' to that group. These values are provided by the arguments \code{groups} and \code{group_weights} (see below). 
#' Each variable can belong to more than one group. The diagonal matrix \eqn{T\in\mathbb{R}^{p\times p}} contains the variable-specific weights. These values are
#' provided by the argument \code{var_weights} (see below). The diagonal matrix \eqn{T_1\in\mathbb{R}^{p\times p}} contains 
#' the variable-specific \eqn{L_1} weights. These values are provided by the argument \code{var_weights_L1} (see below).
#' The regularization path is computed for the sparse adaptive overlap group-LASSO penalty at a grid of values for the regularization 
#' parameter \eqn{\lambda} using the alternating direction method of multipliers (ADMM). See Boyd et al. (2011) and Lin et al. (2022) 
#' for details on the ADMM method. The regularization is a combination of \eqn{L_2} and
#' \eqn{L_1} simultaneous constraints. Different specifications of the \code{penalty} argument lead to different models choice:
#' \describe{
#' \item{LASSO}{The classical Lasso regularization (Tibshirani, 1996) can be obtained by specifying \eqn{\alpha = 1} 
#' and the matrix \eqn{T_1} as the \eqn{p \times p} identity matrix. An adaptive version of this model (Zou, 2006) can be obtained if \eqn{T_1} is 
#' a \eqn{p \times p}  diagonal matrix of adaptive weights. See also Hastie et al. (2015) for further details.}
#' \item{GLASSO}{The group-Lasso regularization (Yuan and Lin, 2006) can be obtained by specifying \eqn{\alpha = 0}, 
#' non-overlapping groups in \eqn{S_g} and by setting the matrix \eqn{T} equal to the \eqn{p \times p} identity matrix. 
#' An adaptive version of this model can be obtained if the matrix \eqn{T} is a \eqn{p \times p} diagonal matrix of adaptive weights. See also Hastie et al. (2015) for further details.}
#' \item{spGLASSO}{The sparse group-Lasso regularization (Simon et al., 2011) can be obtained by specifying \eqn{\alpha\in(0,1)},  
#' non-overlapping groups in \eqn{S_g} and by setting the matrices \eqn{T} and \eqn{T_1} equal to the \eqn{p \times p} identity matrix. 
#' An adaptive version of this model can be obtained if the matrices \eqn{T} and \eqn{T_1} are \eqn{p \times p}  
#' diagonal matrices of adaptive weights.}
#' \item{OVGLASSO}{The overlap group-Lasso regularization (Jenatton et al., 2011) can be obtained by specifying 
#' \eqn{\alpha = 0}, overlapping groups in \eqn{S_g} and by setting the matrix \eqn{T} equal to the \eqn{p \times p} identity matrix. An adaptive version of this model can be obtained if the matrix \eqn{T} is a \eqn{p \times p}  
#' diagonal matrix of adaptive weights.}
#' \item{spOVGLASSO}{The sparse overlap group-Lasso regularization (Jenatton et al., 2011) can be obtained by specifying 
#' \eqn{\alpha\in(0,1)}, overlapping groups in \eqn{S_g} and by setting the matrices \eqn{T} and \eqn{T_1} equal to the \eqn{p \times p} identity matrix. 
#' An adaptive version of this model can be obtained if the matrices \eqn{T} and \eqn{T_1} are \eqn{p \times p}  diagonal matrices of adaptive weights.}
#' }
#'  
#' @param X an \eqn{(n\times p)} matrix of penalized predictors.
#' @param Z an \eqn{(n\times q)} full column rank matrix of predictors that are not penalized. 
#' @param y a length-\eqn{n} response vector.
#' @param penalty choose one from the following options: 'LASSO', for the or adaptive-Lasso penalties, 'GLASSO', 
#' for the group-Lasso penalty, 'spGLASSO', for the sparse group-Lasso penalty, 'OVGLASSO', 
#' for the overlap group-Lasso penalty and 'spOVGLASSO', for the sparse overlap group-Lasso penalty.
#' @param groups either a vector of length \eqn{p} of consecutive integers describing the grouping of the coefficients, 
#' or a list with two elements: the first element is a vector of length \eqn{\sum_{g=1}^G n_g} containing the variables belonging to each group, where \eqn{n_g} is the cardinality of the \eqn{g}-th group, 
#' while the second element is a vector of length \eqn{G} containing the group lengths (see example below).
#' @param group_weights a vector of length \eqn{G} containing group-specific weights. The default is square root of the group cardinality, see Yuan and Lin (2006).
#' @param var_weights a vector of length \eqn{p} containing variable-specific weights. The default is a vector of ones.
#' @param var_weights_L1 a vector of length \eqn{p} containing variable-specific weights for the \eqn{L_1} penalty. The default is a vector of ones.
#' @param cv.fold the number of folds - default is 5.
#' @param standardize.data logical. Should data be standardized?
#' @param lambda either a regularization parameter or a vector of regularization parameters. In this latter case the routine computes the whole path. If it is NULL values for lambda are provided by the routine.
#' @param alpha the sparse overlap group-LASSO mixing parameter, with \eqn{0\leq\alpha\leq1}. This setting is only available for the sparse group-LASSO and the sparse overlap group-LASSO penalties, otherwise it is set to NULL. The LASSO and group-LASSO penalties are obtained
#' by specifying \eqn{\alpha = 1} and \eqn{\alpha = 0}, respectively.
#' @param lambda.min.ratio smallest value for lambda, as a fraction of the maximum lambda value. If \eqn{n>p}, the default is 0.0001, and if \eqn{n<p}, the default is 0.01. 
#' @param nlambda the number of lambda values - default is 30.
#' @param intercept logical. If it is TRUE, a column of ones is added to the design matrix. 
#' @param overall.group logical. This setting is only available for the overlap group-LASSO and the sparse overlap group-LASSO penalties, otherwise it is set to NULL. If it is TRUE, an overall group including all penalized covariates is added.
#' @param control	 a list of control parameters for the ADMM algorithm. See ‘Details’.
#' 
#' @return A named list containing \describe{
#' \item{sp.coefficients}{a length-\eqn{p} solution vector for the parameters \eqn{\beta}. If \eqn{n_\lambda>1} then  the provided vector corresponds to the minimum cross-validated MSE.}
#' \item{coefficients}{a length-\eqn{q} solution vector for the parameters \eqn{\gamma}. If \eqn{n_\lambda>1} then  the provided vector corresponds to the minimum cross-validated MSE.
#' It is provided only when either the matrix \eqn{Z} in input is not NULL or the intercept is set to TRUE.}
#' \item{sp.coef.path}{an \eqn{(n_\lambda\times p)} matrix of estimated \eqn{\beta} coefficients for each lambda of the provided sequence.}
#' \item{coef.path}{an \eqn{(n_\lambda\times q)} matrix of estimated \eqn{\gamma} coefficients for each lambda of the provided sequence. 
#' It is provided only when either the matrix \eqn{Z} in input is not NULL or the intercept is set to TRUE.}
#' \item{lambda}{sequence of lambda.}
#' \item{lambda.min}{value of lambda that attains the minimum cross-validated MSE.}
#' \item{mse}{cross-validated mean squared error.}
#' \item{min.mse}{minimum value of the cross-validated MSE for the sequence of lambda.}
#' \item{convergence}{logical. 1 denotes achieved convergence.}
#' \item{elapsedTime}{elapsed time in seconds.}
#' \item{iternum}{number of iterations.}
#' \item{foldid}{a vector of values between 1 and cv.fold identifying what fold each observation is in.}
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
#' \item{adaptation}{logical. If it is TRUE, ADMM with adaptation is performed. The default value is TRUE. See Boyd et al. (2011) for details.}
#' \item{rho}{an augmented Lagrangian parameter. The default value is 1.}
#' \item{tau.ada}{an adaptation parameter greater than one. Only needed if adaptation = TRUE. The default value is 2. See Boyd et al. (2011) for details.}
#' \item{mu.ada}{an adaptation parameter greater than one. Only needed if adaptation = TRUE. The default value is 10. See Boyd et al. (2011) for details.}
#' \item{abstol}{absolute tolerance stopping criterion. The default value is sqrt(sqrt(.Machine$double.eps)).}
#' \item{reltol}{relative tolerance stopping criterion. The default value is sqrt(.Machine$double.eps).}
#' \item{maxit}{maximum number of iterations. The default value is 100.}
#' \item{print.out}{logical. If it is TRUE, a message about the procedure is printed. The default value is TRUE.}
#' }
#' 
#' @examples 
#' 
#' ### generate sample data
#' set.seed(2023)
#' n    <- 50
#' p    <- 30 
#' X    <- matrix(rnorm(n * p), n, p)
#' 
#' ### Example 1, LASSO penalty
#' 
#' beta <- apply(matrix(rnorm(p, sd = 1), p, 1), 1, fdaSP::softhresh, 1.5)
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
#' mod_cv <- lmSP_cv(X = X, y = y, penalty = "LASSO", 
#'                   standardize.data = FALSE, intercept = FALSE,
#'                   cv.fold = 5, nlambda = 30, 
#'                   control = list("adaptation" = adaptation, 
#'                                  "rho" = rho, 
#'                                  "maxit" = maxit, "reltol" = reltol, 
#'                                  "abstol" = abstol, 
#'                                  "print.out" = FALSE)) 
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
#' mod <- lmSP(X = X, y = y, penalty = "LASSO", 
#'             standardize.data = FALSE, 
#'             intercept = FALSE,
#'             nlambda = 30, 
#'             control = list("adaptation" = adaptation, 
#'                            "rho" = rho, 
#'                            "maxit" = maxit, "reltol" = reltol, 
#'                            "abstol" = abstol, 
#'                            "print.out" = FALSE)) 
#'                                          
#' err_mod <- apply(mod$sp.coef.path, 1, function(x) sum((x - beta)^2))
#' plot(log(mod$lambda), err_mod, type = "l", col = "blue", lwd = 2, 
#'      xlab = latex2exp::TeX("$\\log(\\lambda)$"), 
#'      ylab = "Estimation Error", main = "True Estimation Error", bty = "n")
#' abline(v = log(mod$lambda[which(err_mod == min(err_mod))]), col = "red", lwd = 1.0)
#' abline(v = log(mod_cv$lambda[which(mod_cv$lambda == mod_cv$lambda.min)]), 
#'        col = "red", lwd = 1.0, lty = 2)
#' 
#' ### Example 2, sparse group-LASSO penalty
#' 
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
#' ### run cross-validation
#' mod_cv <- lmSP_cv(X = X, y = y, penalty = "spGLASSO", 
#'                   groups = group1, cv.fold = 5, 
#'                   standardize.data = FALSE,  intercept = FALSE, 
#'                   lambda = lam, alpha = 0.5, 
#'                   control = list("adaptation" = adaptation, 
#'                                  "rho" = rho,
#'                                  "maxit" = maxit, "reltol" = reltol, 
#'                                  "abstol" = abstol, 
#'                                  "print.out" = FALSE)) 
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
#' mod <- lmSP(X = X, y = y, 
#'             penalty = "spGLASSO", 
#'             groups = group1, 
#'             standardize.data = FALSE, 
#'             intercept = FALSE,
#'             lambda = lam, 
#'             alpha = 0.5, 
#'             control = list("adaptation" = adaptation, "rho" = rho, 
#'                            "maxit" = maxit, "reltol" = reltol, "abstol" = abstol, 
#'                            "print.out" = FALSE)) 
#'                                          
#' err_mod <- apply(mod$sp.coef.path, 1, function(x) sum((x - beta)^2))
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
#' \insertRef{hastie_etal.2015}{fdaSP}
#' 
#' \insertRef{jenatton_etal.2011}{fdaSP}
#' 
#' \insertRef{lin_etal.2022}{fdaSP} 
#' 
#' \insertRef{simon_etal.2013}{fdaSP}
#' 
#' \insertRef{yuan_lin.2006}{fdaSP}
#' 
#' \insertRef{zou.2006}{fdaSP}
#' 
#' 
#' @export
lmSP_cv <- function(X, Z = NULL, y, penalty = c("LASSO", "GLASSO", "spGLASSO", "OVGLASSO", "spOVGLASSO"), 
                    groups, group_weights = NULL, var_weights = NULL, var_weights_L1 = NULL, cv.fold = 5,
                    standardize.data = TRUE, intercept = FALSE, overall.group = FALSE,
                    lambda = NULL, alpha = NULL, lambda.min.ratio = NULL, nlambda = 30, 
                    control = list()) {
    
  # :::::::::::::::::::::::::::::::::::::::::::::::::::::
  # check the penalty input
  if (!strcmpi(penalty, "LASSO") && !strcmpi(penalty, "GLASSO") && !strcmpi(penalty, "spGLASSO") && !strcmpi(penalty, "OVGLASSO") && !strcmpi(penalty, "spOVGLASSO")) {
    stop("* lmSP : input 'penalty' must be correctly specified: 'LASSO', 'GLASSO', 'spGLASSO', 'OVGLASSO', 'spOVGLASSO'.") 
  }
  
  # :::::::::::::::::::::::::::::::::::::::::::::::::::::
  # run linear regression with LASSO-type penalty (means, LASSO or adaptive-LASSO)
  if (strcmpi(penalty, "LASSO")) {
    res <- linreg_ADMM_LASSO_cv(X = X, Z = Z, y = y, var_weights = var_weights, standardize.data = standardize.data, 
                                lambda = lambda, lambda.min.ratio = lambda.min.ratio, nlambda = nlambda, cv.fold = cv.fold,
                                intercept = intercept, control = control) 
  }
  # :::::::::::::::::::::::::::::::::::::::::::::::::::::
  # run linear regression with group LASSO-type penalty 
  if (strcmpi(penalty, "GLASSO")) {
    res <- linreg_ADMM_GLASSO_cv(X = X, Z = Z, y = y, groups = groups, group_weights = group_weights, var_weights = var_weights, 
                                 standardize.data = standardize.data,
                                 lambda = lambda, lambda.min.ratio = lambda.min.ratio, nlambda = nlambda, cv.fold = cv.fold,
                                 intercept = intercept, control = control)
  }
  # :::::::::::::::::::::::::::::::::::::::::::::::::::::
  # run linear regression with sparse group LASSO-type penalty 
  if (strcmpi(penalty, "spGLASSO")) {
    res <- linreg_ADMM_spGLASSO_cv(X = X, Z = Z, y = y, groups = groups, group_weights = group_weights, var_weights = var_weights, 
                                   var_weights_L1 = var_weights_L1, standardize.data = standardize.data, 
                                   lambda = lambda, alpha = alpha, lambda.min.ratio = lambda.min.ratio, nlambda = nlambda, cv.fold = cv.fold,
                                   intercept = intercept, control = control)
  }
  # :::::::::::::::::::::::::::::::::::::::::::::::::::::
  # run linear regression with overlap group LASSO-type penalty 
  if (strcmpi(penalty, "OVGLASSO")) {
    res <- linreg_ADMM_OVGLASSO_cv(X = X, Z = Z, y = y, groups = groups, group_weights = group_weights, var_weights = var_weights, 
                                   standardize.data = standardize.data,
                                   lambda = lambda, lambda.min.ratio = lambda.min.ratio, nlambda = nlambda, cv.fold = cv.fold,
                                   intercept = intercept, overall.group = overall.group, control = control)
  }
  # :::::::::::::::::::::::::::::::::::::::::::::::::::::
  # run linear regression with overlap group LASSO-type penalty 
  if (strcmpi(penalty, "spOVGLASSO")) {
    res <- linreg_ADMM_spOVGLASSO_cv(X = X, Z = Z, y = y, groups = groups, group_weights = group_weights, var_weights = var_weights, 
                                     var_weights_L1 = var_weights_L1, standardize.data = standardize.data,
                                     lambda = lambda, alpha = alpha, lambda.min.ratio = lambda.min.ratio, nlambda = nlambda, cv.fold = cv.fold,
                                     intercept = intercept, overall.group = overall.group, control = control)
  }
  # :::::::::::::::::::::::::::::::::::::::::::::::::::::
  # return
  return(res)
}

