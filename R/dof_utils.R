#
## last update: 15 October 2025
#

#
## Description: compute the degrees of freedom for linear regression 
##              model with several Lasso-type penalties
#

# Functions: 
#
#           1. ovglasso_group_selmat
#           2. lm_dof_OVGLASSO
#           3. f2s_dof
#           4. f2s_cov_dof
#           5. lm_dof

####
#     Linear regression models
####

#' @keywords internal
#' @export
lm_dof <- function(y, 
                   X, 
                   Z = NULL,  
                   coeff,
                   lambda,
                   GRmat,
                   group_weights,
                   var_weights,
                   var_weights_L1 = NULL,
                   Umat,
                   err_primal,
                   err_dual,
                   rho,
                   toler_c,
                   toler_d,
                   method = c("lasso", 
                              "glasso", 
                              "ovglasso")) {
  
  # :::::::::::::::::::::::::::::::::::::::::::
  # check maxl
  
  # compute DoF
  if (strcmpi(method, "glasso") || strcmpi(method, "ovglasso")) {
    res <- lm_dof_OVGLASSO(X = X,
                           Z = Z,
                           coeff = coeff,
                           lambda = lambda,
                           GRmat = GRmat,
                           group_weights = group_weights,
                           var_weights = var_weights,
                           Umat = Umat,
                           err_primal = err_primal,
                           err_dual = err_dual,
                           rho = rho,
                           toler_c = toler_c,
                           toler_d = toler_d)
  } else if (strcmpi(method, "lasso")) {
    # da fare
    res <- lm_dof_LASSO(X = X, 
                        Z = Z, 
                        coeff = coeff, 
                        lambda = lambda,
                        var_weights = var_weights,
                        Umat = Umat,
                        err_primal = err_primal,
                        err_dual = err_dual,
                        rho = rho,
                        toler_c = toler_c,
                        toler_d = toler_d)
  }
  
  # return output
  return(res)
}

#' @keywords internal
lm_dof_LASSO <- function(X,
                         Z = NULL,
                         coeff,
                         lambda,
                         var_weights = NULL,
                         Umat,
                         err_primal,
                         err_dual,
                         rho,
                         toler_c,
                         toler_d) {
  
  # get dimensions
  nlambda <- length(lambda)
  p       <- dim(X)[2]
  
  # store output
  dof_STORE           <- rep(0.0, nlambda)
  coeff_active_STORE  <- vector("list", nlambda)
  
  # compute the degrees of freedom
  for (k in 1:nlambda) {
    if (is.null(Z)) {
      if (is.null(var_weights)) {
        # attenzione che con var_weights NULL o altrimenti è uguale
        res <- .lm_dof_LASSO_1lambda(X = X,
                                     coeff = coeff[k,],
                                     lambda = lambda[k],
                                     Uvec = Umat[k,],
                                     err_primal = err_primal[[k]][length(err_primal[[k]])],
                                     err_dual = err_dual[[k]][length(err_dual[[k]])],
                                     rho = rho[[k]][length(rho[[k]])],
                                     toler_c = toler_c,
                                     toler_d = toler_d)
      } else {
        res <- .lm_dof_LASSO_1lambda(X = X,
                                     coeff = coeff[k,],
                                     lambda = lambda[k],
                                     Uvec = Umat[k,],
                                     err_primal = err_primal[[k]][length(err_primal[[k]])],
                                     err_dual = err_dual[[k]][length(err_dual[[k]])],
                                     rho = rho[[k]][length(rho[[k]])],
                                     toler_c = toler_c,
                                     toler_d = toler_d)
      }
    } else {
      
      # define the relevant quantities to specify the LASSO penalty using the OVGLASSO one.
      p             <- dim(X)[2]
      GRmat         <- diag(1, p)
      group_weights <- rep(1, p)
      var_weights   <- rep(1, p)
      
      # run dop
      res <- .lm_cov_dof_OVGLASSO_1lambda(X = X, 
                                          Z = Z,
                                          coeff_X = coeff[k,],
                                          lambda = lambda[k],
                                          GRmat = GRmat,
                                          group_weights = group_weights,
                                          var_weights = var_weights,
                                          Uvec = Umat[k,],
                                          err_primal = err_primal[[k]][length(err_primal[[k]])],
                                          err_dual = err_dual[[k]][length(err_dual[[k]])],
                                          rho = rho[[k]][length(rho[[k]])],
                                          toler_c = toler_c,
                                          toler_d = toler_d)
    }
    
    # store output
    dof_STORE[k]            <- res$dof
    coeff_active_STORE[[k]] <- res$coeff_active
  }
  
  # get output
  output               <- NULL
  output$dof           <- dof_STORE
  output$coeff_active  <- coeff_active_STORE
  
  # return output
  return(output)
}

#
## Overlap Group-Lasso DoF
#

#' @keywords internal
ovglasso_group_selmat <- function(y, x) {
  indi <- outer(y, x, function(xi, yj) {as.integer(yj == xi)})
  return(indi)
}

#' @keywords internal
lm_dof_OVGLASSO <- function(y,
                            X,
                            Z = NULL,
                            coeff,
                            lambda,
                            adaptive_weights = TRUE,
                            GRmat,
                            group_weights,
                            var_weights,
                            Umat,
                            err_primal,
                            err_dual,
                            rho,
                            toler_c,
                            toler_d) {
  
  # get dimensions
  nlambda <- length(lambda)
  p       <- dim(X)[2]
  G       <- length(group_weights)
  
  # store output
  dof_STORE           <- rep(0.0, nlambda)
  coeff_active_STORE  <- vector("list", nlambda)
  groups_active_STORE <- matrix(0, nlambda, G)
  
  # compute LS estimate for adaptive weights
  if (adaptive_weights == TRUE) {
    coeff_LS <- lm_ols(X = X, y = y)
  }
  
  # compute the degrees of freedom
  for (k in 1:nlambda) {
    if (is.null(Z)) {
      if (adaptive_weights == TRUE) {
        res <- .lm_adaptive_dof_OVGLASSO_1lambda(X = X,
                                                 coeff = coeff[k,],
                                                 coeff_LS = coeff_LS,
                                                 lambda = lambda[k],
                                                 GRmat = GRmat,
                                                 group_weights = group_weights,
                                                 var_weights = var_weights,
                                                 Uvec = Umat[k,],
                                                 err_primal = err_primal[[k]][length(err_primal[[k]])],
                                                 err_dual = err_dual[[k]][length(err_dual[[k]])],
                                                 rho = rho[[k]][length(rho[[k]])],
                                                 toler_c = toler_c,
                                                 toler_d = toler_d)
      } else {
        res <- .lm_dof_OVGLASSO_1lambda(X = X,
                                        coeff = coeff[k,], 
                                        lambda = lambda[k],
                                        GRmat = GRmat,
                                        group_weights = group_weights,
                                        var_weights = var_weights,
                                        Uvec = Umat[k,],
                                        err_primal = err_primal[[k]][length(err_primal[[k]])],
                                        err_dual = err_dual[[k]][length(err_dual[[k]])],
                                        rho = rho[[k]][length(rho[[k]])],
                                        toler_c = toler_c,
                                        toler_d = toler_d)
      }
    } else {
      if (adaptive_weights == TRUE) {
        res <- .lm_cov_adaptive_dof_OVGLASSO_1lamba(X = X, 
                                                    Z = Z,
                                                    coeff_X = coeff[k,],
                                                    coeff_X_LS = coeff_LS,
                                                    lambda = lambda[k],
                                                    GRmat = GRmat,
                                                    group_weights = group_weights,
                                                    var_weights = var_weights,
                                                    Uvec = Umat[k,],
                                                    err_primal = err_primal[[k]][length(err_primal[[k]])],
                                                    err_dual = err_dual[[k]][length(err_dual[[k]])],
                                                    rho = rho[[k]][length(rho[[k]])],
                                                    toler_c = toler_c,
                                                    toler_d = toler_d)
      } else {
        res <- .lm_cov_dof_OVGLASSO_1lambda(X = X, 
                                            Z = Z,
                                            coeff_X = coeff[k,],
                                            lambda = lambda[k],
                                            GRmat = GRmat,
                                            group_weights = group_weights,
                                            var_weights = var_weights,
                                            Uvec = Umat[k,],
                                            err_primal = err_primal[[k]][length(err_primal[[k]])],
                                            err_dual = err_dual[[k]][length(err_dual[[k]])],
                                            rho = rho[[k]][length(rho[[k]])],
                                            toler_c = toler_c,
                                            toler_d = toler_d)
      }
    }
    
    # store output
    dof_STORE[k]            <- res$dof
    coeff_active_STORE[[k]] <- res$coeff_active
    groups_active_STORE[k,] <- t(res$groups_active)
  }
  
  # get output
  output               <- NULL
  output$dof           <- dof_STORE
  output$coeff_active  <- coeff_active_STORE
  output$groups_active <- groups_active_STORE
  
  # return output
  return(output)
}


####
#     Scalar on function regression
####

#' @keywords internal
f2s_dof <- function(W,
                    y,
                    coeff,
                    lambda,
                    lambda2 = NULL,
                    diff_order = 1,
                    adaptive_weights = TRUE,
                    GRmat,
                    group_weights,
                    var_weights,
                    Umat,
                    err_primal,
                    err_dual,
                    rho,
                    toler_c,
                    toler_d) {
  
  # get dimensions
  nlambda  <- length(lambda)
  nlambda2 <- length(lambda2)
  p        <- dim(W)[2]
  G        <- length(group_weights)
  
  # store output
  if (is.null(lambda2)) {
    # se lambda2 è NULL lo considero zero
    dof_STORE               <- rep(0.0, nlambda)
    groups_active_STORE     <- matrix(data = 0, nrow = nlambda, ncol = G)
    vars_active_STORE       <- matrix(data = 0, nrow = nlambda, ncol = p)
    coeff_active_STORE      <- vector("list", nlambda)
    dim(coeff_active_STORE) <- nlambda
    lambda2                 <- 0                                    # 27 april 2026: set to zero in case NULL
  } else {
    if (nlambda2 == 1) {
      if (lambda2 == 0) {
        # qui ho un solo lambda che è zero
        dof_STORE               <- rep(0.0, nlambda)
        groups_active_STORE     <- matrix(data = 0, nrow = nlambda, ncol = G)
        vars_active_STORE       <- matrix(data = 0, nrow = nlambda, ncol = p)
        coeff_active_STORE      <- vector("list", nlambda)
        dim(coeff_active_STORE) <- nlambda
      } else {
        # qui ho un solo lambda che non è zero
        dof_STORE               <- matrix(data = 0.0, nrow = nlambda, ncol = nlambda2)
        groups_active_STORE     <- array(data = 0, dim = c(nlambda, nlambda2, G))
        vars_active_STORE       <- array(data = 0, dim = c(nlambda, nlambda2, p))
        coeff_active_STORE      <- vector("list", nlambda * nlambda2)
        dim(coeff_active_STORE) <- c(nlambda, nlambda2)
      }
    } else {
      dof_STORE               <- matrix(data = 0.0, nrow = nlambda, ncol = nlambda2)
      groups_active_STORE     <- array(data = 0, dim = c(nlambda, nlambda2, G))
      vars_active_STORE       <- array(data = 0, dim = c(nlambda, nlambda2, p))
      coeff_active_STORE      <- vector("list", nlambda * nlambda2)
      dim(coeff_active_STORE) <- c(nlambda, nlambda2)
    }
  }
  
  # compute LS estimate for adaptive weights
  if (adaptive_weights == TRUE) {
    coeff_LS <- lm_ols(X = W, y = y)
  }
  
  # compute the degrees of freedom
  if ((!is.null(lambda2)) || (!is.null(diff_order))) {
    if (length(lambda2) == 1) {
      if (lambda2 == 0.0) {
        for (j in 1:nlambda) {
          if (adaptive_weights == TRUE) {
            # compute degrees of freedom: adaptive weights
            res <- .f2s_adaptive_dof_1lambda(W = W,
                                             coeff = coeff[j,],
                                             coeff_LS = coeff_LS,
                                             lambda = lambda[j],
                                             GRmat = GRmat,
                                             group_weights = group_weights,
                                             var_weights = var_weights,
                                             Uvec = Umat[j,],
                                             err_primal = err_primal[[j]][length(err_primal[[j]])],
                                             err_dual = err_dual[[j]][length(err_dual[[j]])],
                                             rho = rho[[j]][length(rho[[j]])],
                                             toler_c = toler_c,
                                             toler_d = toler_d)
          } else {
            # compute degrees of freedom: weights are not adaptive
            res <- .f2s_dof_1lambda(W = W,
                                    coeff = coeff[j,],
                                    lambda = lambda[j],
                                    GRmat = GRmat,
                                    group_weights = group_weights,
                                    var_weights = var_weights,
                                    Uvec = Umat[j,],
                                    err_primal = err_primal[[j]][length(err_primal[[j]])],
                                    err_dual = err_dual[[j]][length(err_dual[[j]])],
                                    rho = rho[[j]][length(rho[[j]])],
                                    toler_c = toler_c,
                                    toler_d = toler_d)
          }
          
          # store output
          dof_STORE[j]            <- res$dof
          coeff_active_STORE[[j]] <- res$coeff_active
          groups_active_STORE[j,] <- as.numeric(res$groups_active)
          vars_active_STORE[j,]   <- as.numeric(res$var_active)
        }
      } else {
        for (j in 1:nlambda) {
          for (k in 1:nlambda2) {
            if (adaptive_weights == TRUE) {
              # compute degrees of freedom: adaptive weights
              res <- .f2s_smo_adaptive_dof_1lambda(W = W, 
                                                   coeff = coeff[j,k,],
                                                   coeff_LS = coeff_LS,
                                                   lambda = lambda[j], 
                                                   lambda2 = lambda2[k], 
                                                   diff_order = diff_order,
                                                   GRmat = GRmat,
                                                   group_weights = group_weights,
                                                   var_weights = var_weights,
                                                   Uvec = Umat[j,k,],
                                                   err_primal = err_primal[[j, k]][length(err_primal[[j, k]])], 
                                                   err_dual = err_dual[[j, k]][length(err_dual[[j, k]])], 
                                                   rho = rho[[j, k]][length(rho[[j, k]])],
                                                   toler_c = toler_c,
                                                   toler_d = toler_d)
            } else {
              res <- .f2s_smo_dof_1lambda(W = W, 
                                          coeff = coeff[j,k,],
                                          lambda = lambda[j], 
                                          lambda2 = lambda2[k], 
                                          diff_order = diff_order,
                                          GRmat = GRmat,
                                          group_weights = group_weights,
                                          var_weights = var_weights,
                                          Uvec = Umat[j,k,],
                                          err_primal = err_primal[[j, k]][length(err_primal[[j, k]])], 
                                          err_dual = err_dual[[j, k]][length(err_dual[[j, k]])], 
                                          rho = rho[[j, k]][length(rho[[j, k]])],
                                          toler_c = toler_c,
                                          toler_d = toler_d)
            }
            
            # store output
            dof_STORE[j, k]            <- res$dof
            coeff_active_STORE[[j, k]] <- res$coeff_active
            groups_active_STORE[j, k,] <- as.numeric(res$groups_active)
            vars_active_STORE[j, k,]   <- as.numeric(res$var_active)
          }
        }
      }
    } else {
      for (j in 1:nlambda) {
        for (k in 1:nlambda2) {
          if (adaptive_weights == TRUE) {
            # compute degrees of freedom: adaptive weights
            res <- .f2s_smo_adaptive_dof_1lambda(W = W, 
                                                 coeff = coeff[j,k,],
                                                 coeff_LS = coeff_LS,
                                                 lambda = lambda[j], 
                                                 lambda2 = lambda2[k], 
                                                 diff_order = diff_order,
                                                 GRmat = GRmat,
                                                 group_weights = group_weights,
                                                 var_weights = var_weights,
                                                 Uvec = Umat[j,k,],
                                                 err_primal = err_primal[[j, k]][length(err_primal[[j, k]])], 
                                                 err_dual = err_dual[[j, k]][length(err_dual[[j, k]])], 
                                                 rho = rho[[j, k]][length(rho[[j, k]])],
                                                 toler_c = toler_c,
                                                 toler_d = toler_d)
          } else {
            res <- .f2s_smo_dof_1lambda(W = W, 
                                        coeff = coeff[j,k,],
                                        lambda = lambda[j], 
                                        lambda2 = lambda2[k], 
                                        diff_order = diff_order,
                                        GRmat = GRmat,
                                        group_weights = group_weights,
                                        var_weights = var_weights,
                                        Uvec = Umat[j,k,],
                                        err_primal = err_primal[[j, k]][length(err_primal[[j, k]])], 
                                        err_dual = err_dual[[j, k]][length(err_dual[[j, k]])], 
                                        rho = rho[[j, k]][length(rho[[j, k]])],
                                        toler_c = toler_c,
                                        toler_d = toler_d)
          }
          
          # store output
          dof_STORE[j, k]            <- res$dof
          coeff_active_STORE[[j, k]] <- res$coeff_active
          groups_active_STORE[j, k,] <- as.numeric(res$groups_active)
          vars_active_STORE[j, k,]   <- as.numeric(res$var_active)
        }
      }
    }
  } else {
    cat("lambda2 is set to NULL!\n")
  }
  
  # get output
  output               <- NULL
  output$dof           <- dof_STORE
  output$coeff_active  <- coeff_active_STORE
  output$groups_active <- groups_active_STORE
  output$vars_active   <- vars_active_STORE
  
  # return output
  return(output)
}

#' @keywords internal
f2s_cov_dof <- function(W, 
                        Z,
                        y,
                        coeff_W,
                        coeff_Z,
                        coeff_W_LS,
                        lambda,
                        lambda2 = NULL,
                        diff_order = 1,
                        adaptive_weights = TRUE,
                        GRmat,
                        group_weights,
                        var_weights,
                        Umat,
                        err_primal,
                        err_dual,
                        rho,
                        toler_c,
                        toler_d) {
  
  # get dimensions
  nlambda  <- length(lambda)
  nlambda2 <- length(lambda2)
  p        <- dim(W)[2]
  G        <- length(group_weights)
  
  # store output
  if (is.null(lambda2)) {
    # se lambda2 è NULL lo considero zero
    dof_STORE               <- rep(0.0, nlambda)
    groups_active_STORE     <- matrix(data = 0, nrow = nlambda, ncol = G)
    vars_active_STORE       <- matrix(data = 0, nrow = nlambda, ncol = p)
    coeff_active_STORE      <- vector("list", nlambda)
    dim(coeff_active_STORE) <- nlambda
    lambda2                 <- 0
  } else {
    if (nlambda2 == 1) {
      if (lambda2 == 0) {
        # qui ho un solo lambda che è zero
        dof_STORE               <- rep(0.0, nlambda)
        groups_active_STORE     <- matrix(data = 0, nrow = nlambda, ncol = G)
        vars_active_STORE       <- matrix(data = 0, nrow = nlambda, ncol = p)
        coeff_active_STORE      <- vector("list", nlambda)
        dim(coeff_active_STORE) <- nlambda
      } else {
        # qui ho un solo lambda che non è zero
        dof_STORE               <- matrix(data = 0.0, nrow = nlambda, ncol = nlambda2)
        groups_active_STORE     <- array(data = 0, dim = c(nlambda, nlambda2, G))
        vars_active_STORE       <- array(data = 0, dim = c(nlambda, nlambda2, p))
        coeff_active_STORE      <- vector("list", nlambda * nlambda2)
        dim(coeff_active_STORE) <- c(nlambda, nlambda2)
      }
    } else {
      dof_STORE               <- matrix(data = 0.0, nrow = nlambda, ncol = nlambda2)
      groups_active_STORE     <- array(data = 0, dim = c(nlambda, nlambda2, G))
      vars_active_STORE       <- array(data = 0, dim = c(nlambda, nlambda2, p))
      coeff_active_STORE      <- vector("list", nlambda * nlambda2)
      dim(coeff_active_STORE) <- c(nlambda, nlambda2)
    }
  }
  
  # compute LS estimate for adaptive weights
  if (adaptive_weights == TRUE) {
    out        <- lm_ols_FWL(X = W, Z = Z, y = y)
    coeff_W_LS <- out$coeff_X 
  }
  
  # compute the degrees of freedom
  if ((!is.null(lambda2)) || (!is.null(diff_order))) {
    if (length(lambda2) == 1) {
      if (lambda2 == 0.0) {
        for (j in 1:nlambda) {
          if (adaptive_weights == TRUE) {
            res <- .f2s_cov_adaptive_dof_1lambda(W = W, 
                                                 Z = Z, 
                                                 coeff_W = coeff_W[j,],
                                                 coeff_W_LS = coeff_W_LS,
                                                 lambda = lambda[j], 
                                                 GRmat = GRmat,
                                                 group_weights = group_weights,
                                                 var_weights = var_weights,
                                                 Uvec = Umat[j,],
                                                 err_primal = err_primal[[j]][length(err_primal[[j]])],
                                                 err_dual = err_dual[[j]][length(err_dual[[j]])],
                                                 rho = rho[[j]][length(rho[[j]])],
                                                 toler_c = toler_c,
                                                 toler_d = toler_d)
          } else {
            res <- .f2s_cov_dof_1lambda(W = W, 
                                        Z = Z, 
                                        coeff_W = coeff_W[j,],
                                        lambda = lambda[j], 
                                        GRmat = GRmat,
                                        group_weights = group_weights,
                                        var_weights = var_weights,
                                        Uvec = Umat[j,],
                                        err_primal = err_primal[[j]][length(err_primal[[j]])],
                                        err_dual = err_dual[[j]][length(err_dual[[j]])],
                                        rho = rho[[j]][length(rho[[j]])],
                                        toler_c = toler_c,
                                        toler_d = toler_d)
          }
          
          # store output
          dof_STORE[j]            <- res$dof
          coeff_active_STORE[[j]] <- res$coeff_active
          groups_active_STORE[j,] <- as.numeric(res$groups_active)
          vars_active_STORE[j,]   <- as.numeric(res$var_active)
        }
      } else {
        for (j in 1:nlambda) {
          for (k in 1:nlambda2) {
            if (adaptive_weights == TRUE) {
              res <- .f2s_adaptive_cov_smo_dof_1lambda(W = W, 
                                                       Z = Z, 
                                                       coeff_W = coeff_W[j,k,],
                                                       coeff_W_LS = coeff_W_LS,
                                                       lambda = lambda[j], 
                                                       lambda2 = lambda2[k], 
                                                       diff_order = diff_order,
                                                       GRmat = GRmat,
                                                       group_weights = group_weights,
                                                       var_weights = var_weights,
                                                       Uvec = Umat[j,k,],
                                                       err_primal = err_primal[[j, k]][length(err_primal[[j, k]])], 
                                                       err_dual = err_dual[[j, k]][length(err_dual[[j, k]])], 
                                                       rho = rho[[j, k]][length(rho[[j, k]])],
                                                       toler_c = toler_c,
                                                       toler_d = toler_d)
            } else {
              res <- .f2s_cov_smo_dof_1lambda(W = W, 
                                              Z = Z, 
                                              coeff_W = coeff_W[j,k,],
                                              lambda = lambda[j], 
                                              lambda2 = lambda2[k], 
                                              diff_order = diff_order,
                                              GRmat = GRmat,
                                              group_weights = group_weights,
                                              var_weights = var_weights,
                                              Uvec = Umat[j,k,],
                                              err_primal = err_primal[[j, k]][length(err_primal[[j, k]])], 
                                              err_dual = err_dual[[j, k]][length(err_dual[[j, k]])], 
                                              rho = rho[[j, k]][length(rho[[j, k]])],
                                              toler_c = toler_c,
                                              toler_d = toler_d)
            }
            
            # store output
            dof_STORE[j, k]            <- res$dof
            coeff_active_STORE[[j, k]] <- res$coeff_active
            groups_active_STORE[j, k,] <- as.numeric(res$groups_active)
            vars_active_STORE[j, k,]   <- as.numeric(res$var_active)
          }
        }
      }
    } else {
      for (j in 1:nlambda) {
        for (k in 1:nlambda2) {
          if (adaptive_weights == TRUE) {
            res <- .f2s_adaptive_cov_smo_dof_1lambda(W = W, 
                                                     Z = Z, 
                                                     coeff_W = coeff_W[j,k,],
                                                     coeff_W_LS = coeff_W_LS,
                                                     lambda = lambda[j], 
                                                     lambda2 = lambda2[k], 
                                                     diff_order = diff_order,
                                                     GRmat = GRmat,
                                                     group_weights = group_weights,
                                                     var_weights = var_weights,
                                                     Uvec = Umat[j,k,],
                                                     err_primal = err_primal[[j, k]][length(err_primal[[j, k]])], 
                                                     err_dual = err_dual[[j, k]][length(err_dual[[j, k]])], 
                                                     rho = rho[[j, k]][length(rho[[j, k]])],
                                                     toler_c = toler_c,
                                                     toler_d = toler_d)
          } else {
            res <- .f2s_cov_smo_dof_1lambda(W = W, 
                                            Z = Z, 
                                            coeff_W = coeff_W[j,k,],
                                            lambda = lambda[j], 
                                            lambda2 = lambda2[k], 
                                            diff_order = diff_order,
                                            GRmat = GRmat,
                                            group_weights = group_weights,
                                            var_weights = var_weights,
                                            Uvec = Umat[j,k,],
                                            err_primal = err_primal[[j, k]][length(err_primal[[j, k]])], 
                                            err_dual = err_dual[[j, k]][length(err_dual[[j, k]])], 
                                            rho = rho[[j, k]][length(rho[[j, k]])],
                                            toler_c = toler_c,
                                            toler_d = toler_d)
          }
          
          # store output
          dof_STORE[j, k]            <- res$dof
          coeff_active_STORE[[j, k]] <- res$coeff_active
          groups_active_STORE[j, k,] <- as.numeric(res$groups_active)
          vars_active_STORE[j, k,]   <- as.numeric(res$var_active)
        }
      }
    }
  } else {
    cat("lambda2 is set to NULL!\n")
  }
  
  # get output
  output               <- NULL
  output$dof           <- dof_STORE
  output$coeff_active  <- coeff_active_STORE
  output$groups_active <- groups_active_STORE
  output$vars_active   <- vars_active_STORE

  # return output
  return(output)
}







