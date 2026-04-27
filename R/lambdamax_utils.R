#
## last update: 30 september 2025
#

#
## Description: compute the lambda max for linear regression 
##              model with several Lasso-type penalties
#

# Functions: 
#
#           1. lm_lambdamax_LASSO
#           2. lm_lambdamax_ADLASSO
#           3. lm_lambdamax_GLASSO
#           4. lm_lambdamax_spGLASSO
#           5. lm_lambdamax_OVGLASSO
#           6. lm_lambdamax

#' @keywords internal
lm_lambdamax_LASSO <- function(y, X, Z = NULL, 
                               lambda.min.ratio = NULL, 
                               maxl = NULL) {
  
  # :::::::::::::::::::::::::::::::::::::::::::
  # check maxl
  if (is.null(maxl)) {
    maxl <- 30
  }
  
  # :::::::::::::::::::::::::::::::::::::::::::
  # get the lambda max for LASSO
  n <- dim(X)[1]
  p <- dim(X)[2]
  if (is.null(Z)) {
    vXy        <- crossprod(X, y)
    lambda.max <- max(abs(vXy)) / n
  } else {
    regp       <- solve(crossprod(Z)) %*% crossprod(Z, y)
    y_         <- y - Z %*% regp
    vXy        <- crossprod(X, y_)
    lambda.max <- max(abs(vXy)) / n
  }
  
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # get the lambda min for LASSO
  if (is.null(lambda.min.ratio)) {
    lambda.min.ratio <- ifelse(n < p, 0.01, 1e-04)
  }
  lambda.min <- lambda.min.ratio * lambda.max
  
  # :::::::::::::::::::::::::::::::::::::::::::
  # get the sequence of lambdas
  lambda.seq <- logseq(from = lambda.max, to = lambda.min, length.out = maxl)
  
  # :::::::::::::::::::::::::::::::::::::::::::
  # get output
  res            <- NULL
  res$lambda.min <- lambda.min
  res$lambda.max <- lambda.max
  res$lambda.seq <- lambda.seq
  
  # :::::::::::::::::::::::::::::::::::::::::::
  # get output
  return(res)
}

#' @keywords internal
#' @export
lm_lambdamax_ADALASSO <- function(y, X, Z = NULL, 
                                  var_weights = NULL, 
                                  lambda.min.ratio = NULL, 
                                  maxl = NULL) {
  
  # :::::::::::::::::::::::::::::::::::::::::::
  # check maxl
  if (is.null(maxl)) {
    maxl <- 30
  }
  
  # :::::::::::::::::::::::::::::::::::::::::::
  # get the dimensions
  n <- dim(X)[1]
  p <- dim(X)[2]
  
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # check for adaptive-LASSO
  if (is.null(var_weights)) {
    var_weights <- rep(1L, p)
  } 
  
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # standardize data (to account adaptive weights)
  X_ <- X %*% diag(1.0 / var_weights)
  
  # :::::::::::::::::::::::::::::::::::::::::::
  # get the lambda max for LASSO
  if (is.null(Z)) {
    vXy        <- crossprod(X_, y)
    lambda.max <- max(abs(vXy)) / n
  } else {
    regp       <- solve(crossprod(Z)) %*% crossprod(Z, y)
    y_         <- y - Z %*% regp
    vXy        <- crossprod(X_, y_)
    lambda.max <- max(abs(vXy)) / n
  }
  
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # get the lambda min for LASSO
  if (is.null(lambda.min.ratio)) {
    lambda.min.ratio <- ifelse(n < p, 0.01, 1e-04)
  }
  lambda.min <- lambda.min.ratio * lambda.max
  
  # :::::::::::::::::::::::::::::::::::::::::::
  # get the sequence of lambdas
  lambda.seq <- logseq(from = lambda.max, to = lambda.min, length.out = maxl)
  
  # :::::::::::::::::::::::::::::::::::::::::::
  # get output
  res            <- NULL
  res$lambda.min <- lambda.min
  res$lambda.max <- lambda.max
  res$lambda.seq <- lambda.seq
  
  # :::::::::::::::::::::::::::::::::::::::::::
  # get output
  return(res)
}

#' @keywords internal
#' @export
lm_lambdamax_GLASSO <- function(y, X, Z = NULL, 
                                groups = NULL, 
                                GRmat = NULL, 
                                group_weights = NULL, 
                                var_weights = NULL, 
                                lambda.min.ratio = NULL, 
                                maxl = NULL) {
  
  # :::::::::::::::::::::::::::::::::::::::::::
  # check maxl
  if (is.null(maxl)) {
    maxl <- 30
  }
  
  # :::::::::::::::::::::::::::::::::::::::::::
  # manage the non-penalized covariates (Z)
  if (is.null(Z)) {
    y_ <- y
  } else {
    regp <- solve(crossprod(Z)) %*% crossprod(Z, y)
    y_   <- y - Z %*% regp
  } 
  
  # :::::::::::::::::::::::::::::::::::::::::::
  # get the relevant quantities
  n <- dim(X)[1]
  p <- dim(X)[2]
  
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # Manage groups
  if (is.null(GRmat)) {
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
  } else {
    G  <- dim(GRmat)[1]
    nG <- rowSums(GRmat)
  }
  
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # Manage group weights
  if (is.null(group_weights)) {
    # Define the weights as in Yuan and Lin 2006, JRSSB
    group_weights <- sqrt(diag(tcrossprod(GRmat)))
  }
  
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # Manage variable weights
  if (is.null(var_weights)) {
    var_weights <- 1.0 / colSums(GRmat)
  }
  Tmat_INV <- diag(1.0 / var_weights)
  
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # standardize data (to account adaptive weights)
  X_ <- X %*% Tmat_INV
  
  # :::::::::::::::::::::::::::::::::::::::::::
  # get the lambda max for Group LASSO
  lambda_ <- NULL
  for (g in 1:G) {
    idx     <- which(GRmat[g, ] == 1) 
    vXy_    <- as.numeric(crossprod(X_[,idx], y_))
    lam     <- norm(vXy_, type = "2") / group_weights[g]
    lambda_ <- c(lambda_, lam)
  }
  lambda.max <- max(lambda_)  / n
  
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # get the lambda min for LASSO
  if (is.null(lambda.min.ratio)) {
    lambda.min.ratio <- ifelse(n < p, 0.01, 1e-04)
  }
  lambda.min <- lambda.min.ratio * lambda.max
  
  # :::::::::::::::::::::::::::::::::::::::::::
  # get the sequence of lambdas
  lambda.seq <- logseq(from = lambda.max, to = lambda.min, length.out = maxl)
  
  # :::::::::::::::::::::::::::::::::::::::::::
  # get output
  res            <- NULL
  res$lambda.min <- lambda.min
  res$lambda.max <- lambda.max
  res$lambda.seq <- lambda.seq
  
  # :::::::::::::::::::::::::::::::::::::::::::
  # get output
  return(res)
}

# lambdamax for sparse group lasso as in the original paper of
# Noah Simon, Jerome Friedman, Trevor Hastie & Robert Tibshirani, JoCGS 2013

#' @keywords internal
#' @export
spglasso_lambdamax_objfun <- function(x, X, y, 
                                      alpha, gwei) {
  
  # get dimensions
  n <- length(y)
  
  # get the relevant quantities
  XTy    <- as.numeric(crossprod(X, y)) / n
  lambda <- x
  
  # Evaluate the objective function
  Svec   <- softthreshold_VEC(xx = XTy, lambda = (lambda * alpha))
  objfun <- as.numeric(crossprod(Svec)) - gwei^2 * (1.0 - alpha)^2 * lambda^2
  
  # return
  return(objfun)
}
spglasso_lambdamax_objfun_VEC <- Vectorize(spglasso_lambdamax_objfun, vectorize.args = "x")
softthreshold_VEC <- Vectorize(FUN = softthreshold, vectorize.args = "xx")

lm_lambdamax_spGLASSO <- function(y, X, Z = NULL, groups, group_weights = NULL, var_weights = NULL, 
                                  var_weights_L1 = NULL, lambda.min.ratio = NULL, maxl = NULL, alpha = NULL, 
                                  tol = .Machine$double.eps^0.25) {
  
  # :::::::::::::::::::::::::::::::::::::::::::
  # check maxl
  if (is.null(maxl)) {
    maxl <- 30
  }
  
  # :::::::::::::::::::::::::::::::::::::::::::
  # manage the non-penalized covariates (Z)
  if (is.null(Z)) {
    y_ <- y
  } else {
    regp <- solve(crossprod(Z)) %*% crossprod(Z, y)
    y_   <- y - Z %*% regp
  } 
  
  # :::::::::::::::::::::::::::::::::::::::::::
  # get the relevant quantities
  n <- dim(X)[1]
  p <- dim(X)[2]
  
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # get the lambda min for LASSO
  if (is.null(alpha)) {
    alpha <- 0.5
  }
  
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
  # Manage group weights
  if (is.null(group_weights)) {
    # Define the weights as in Yuan and Lin 2006, JRSSB
    group_weights <- sqrt(diag(tcrossprod(GRmat)))
  }
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # Manage variable weights: adaptive Lasso
  if (is.null(var_weights_L1)) {
    var_weights_L1 <- rep(1, p)
  }
  
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # Manage variable weights
  if (is.null(var_weights)) {
    var_weights <- 1.0 / colSums(GRmat)
  }
  Tmat_INV         <- diag(1.0 / var_weights)
  Tmat_L1_INV_SQRT <- diag(1.0 / sqrt(var_weights_L1))
  
  # :::::::::::::::::::::::::::::::::::::::::::
  # get the lambda max for Group LASSO
  if (alpha == 0) {
    X_ <- X %*% Tmat_INV
    # Group LASSO
    lambda_ <- NULL
    for (g in 1:G) {
      idx     <- which(GRmat[g, ] == 1) 
      vXy_    <- as.numeric(crossprod(X_[,idx], y_))
      lam     <- norm(vXy_, type = "2") / group_weights[g]
      lambda_ <- c(lambda_, lam)
    }
    lambda.max <- max(lambda_)  / n
  } else if (alpha == 1) {
    # LASSO
    X_         <- X %*% Tmat_L1_INV_SQRT
    vXy        <- crossprod(X_, y_)
    lambda.max <- max(abs(vXy)) / n
  } else {
    X_ <- X %*% Tmat_INV
    # sparse GLASSO
    lambda_ <- NULL
    for (g in 1:G) {
      idx     <- which(GRmat[g, ] == 1) 
      lam     <- uniroot(f = spglasso_lambdamax_objfun, interval = c(0.0, 100.0),
                         X = X_[,idx], y = y_, alpha = alpha, gwei = group_weights[g], extendInt = "yes")$root
      lambda_ <- c(lambda_, lam)
    }
    lambda.max <- max(lambda_)  
  }
  
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # get the lambda min for LASSO
  if (is.null(lambda.min.ratio)) {
    lambda.min.ratio <- ifelse(n < p, 0.01, 1e-04)
  }
  lambda.min <- lambda.min.ratio * lambda.max
  
  # :::::::::::::::::::::::::::::::::::::::::::
  # get the sequence of lambdas
  lambda.seq <- logseq(from = lambda.max, to = lambda.min, length.out = maxl)
  
  # :::::::::::::::::::::::::::::::::::::::::::
  # get output
  res            <- NULL
  res$lambda.min <- lambda.min
  res$lambda.max <- lambda.max
  res$lambda.seq <- lambda.seq
  
  # :::::::::::::::::::::::::::::::::::::::::::
  # get output
  return(res)
}

#' @keywords internal
#' @export
lm_lambdamax_GLASSO_yuan_lin_2006_JRSSB <- function(y, X, groups, 
                                                    lambda.min.ratio = NULL, 
                                                    maxl = NULL) {
  
  # :::::::::::::::::::::::::::::::::::::::::::
  # check maxl
  if (is.null(maxl)) {
    maxl <- 30
  }
  
  # :::::::::::::::::::::::::::::::::::::::::::
  # get the relevant quantities
  n <- dim(X)[1]
  p <- dim(X)[2]
  
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
  
  # :::::::::::::::::::::::::::::::::::::::::::
  # get the lambda max for Group LASSO
  # (https://stats.stackexchange.com/questions/340486/how-is-the-minimum-lambda-computed-in-group-lasso)
  lambda_ <- NULL
  for (g in 1:G) {
    idx     <- which(GRmat[g, ] == 1) 
    X_      <- X[,idx] 
    vXy_    <- as.numeric(crossprod(X_, y))
    lam     <- norm(vXy_, type = "2") / sqrt(nG[g])
    lambda_ <- c(lambda_, lam)
  }
  lambda.max <- max(lambda_)  / n
  
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # get the lambda min for LASSO
  if (is.null(lambda.min.ratio)) {
    lambda.min.ratio <- ifelse(n < p, 0.01, 1e-04)
  }
  lambda.min <- lambda.min.ratio * lambda.max
  
  # :::::::::::::::::::::::::::::::::::::::::::
  # get the sequence of lambdas
  lambda.seq <- logseq(from = lambda.max, to = lambda.min, length.out = maxl)
  
  # :::::::::::::::::::::::::::::::::::::::::::
  # get output
  res            <- NULL
  res$lambda.min <- lambda.min
  res$lambda.max <- lambda.max
  res$lambda.seq <- lambda.seq
  
  # :::::::::::::::::::::::::::::::::::::::::::
  # get output
  return(res)
}

#' @keywords internal
#' @export
lm_lambdamax_OVGLASSO <- function(y, X, Z = NULL, groups = NULL, GRmat = NULL, 
                                  group_weights = NULL, var_weights = NULL, 
                                  lambda.min.ratio = NULL, maxl = NULL) {
  
  # :::::::::::::::::::::::::::::::::::::::::::
  # check maxl
  if (is.null(maxl)) {
    maxl <- 30
  }
  
  # :::::::::::::::::::::::::::::::::::::::::::
  # get the relevant quantities
  n <- dim(X)[1]
  p <- dim(X)[2]
  
  # :::::::::::::::::::::::::::::::::::::::::::
  # manage the non-penalized covariates (Z)
  if (is.null(Z)) {
    y_ <- y
  } else {
    regp <- solve(crossprod(Z)) %*% crossprod(Z, y)
    y_   <- y - Z %*% regp
  } 
  
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # Manage groups
  if (is.null(GRmat)) {
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
  } else {
    G  <- dim(GRmat)[1]
    nG <- rowSums(GRmat)
  }
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # Manage group weights
  if (is.null(group_weights)) {
    # Define the weights as in Yuan and Lin 2006, JRSSB
    #group_weights <- sqrt(diag(tcrossprod(GRmat)))
    # Modificata il 19 novembre 2024
    group_weights <- rep(1, G)
  }
  
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # Manage variable weights
  if (is.null(var_weights)) {
    var_weights <- 1.0 / sqrt(colSums(GRmat))
  }
  Tmat_INV <- diag(1.0 / var_weights)
  
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # standardize data (to account adaptive weights)
  X_ <- X %*% Tmat_INV
  #X_ <- X 
  
  # :::::::::::::::::::::::::::::::::::::::::::
  # get the lambda max for Overlap Group LASSO
  # lambda_ <- NULL
  # for (g in 1:G) {
  #   idx     <- which(GRmat[g, ] == 1) 
  #   vXy_    <- as.numeric(crossprod(X_[,idx], y_))
  #   lam     <- norm(vXy_, type = "2") / group_weights[g]
  #   lambda_ <- c(lambda_, lam)
  # }
  # lambda.max <- max(lambda_)  / n
  vXy_    <- as.numeric(crossprod(X_, y_))
  lambda_ <- ovglasso_dual_norm_upper_bound(x = vXy_, 
                                            groups = GRmat, 
                                            weights = group_weights)
  lambda.max <- lambda_ / n
  
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # get the lambda min for LASSO
  if (is.null(lambda.min.ratio)) {
    lambda.min.ratio <- ifelse(n < p, 0.01, 1e-04)
  }
  lambda.min <- lambda.min.ratio * lambda.max
  
  # :::::::::::::::::::::::::::::::::::::::::::
  # get the sequence of lambdas
  lambda.seq <- logseq(from = lambda.max, to = lambda.min, length.out = maxl)
  
  # :::::::::::::::::::::::::::::::::::::::::::
  # get output
  res            <- NULL
  res$lambda.min <- lambda.min
  res$lambda.max <- lambda.max
  res$lambda.seq <- lambda.seq
  
  # :::::::::::::::::::::::::::::::::::::::::::
  # get output
  return(res)
}

#' @keywords internal
#' @export
lm_lambdamax_spOVGLASSO <- function(y, X, Z = NULL, groups = NULL, GRmat = NULL, 
                                    group_weights = NULL, var_weights = NULL, 
                                    lambda.min.ratio = NULL, maxl = NULL) {
  
  # :::::::::::::::::::::::::::::::::::::::::::
  # check maxl
  if (is.null(maxl)) {
    maxl <- 30
  }
  
  # :::::::::::::::::::::::::::::::::::::::::::
  # get the relevant quantities
  n <- dim(X)[1]
  p <- dim(X)[2]
  
  # :::::::::::::::::::::::::::::::::::::::::::
  # manage the non-penalized covariates (Z)
  if (is.null(Z)) {
    y_ <- y
  } else {
    regp <- solve(crossprod(Z)) %*% crossprod(Z, y)
    y_   <- y - Z %*% regp
  } 
  
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # Manage groups
  if (is.null(GRmat)) {
    if (is.list(groups) && !is.null(groups$Glen)) {
      nG      <- groups$Glen
      groups_ <- groups$groups
      GRmat   <- groups2mat_OVGLASSO(groups = groups_, Glen = nG)
      GRmat   <- rbind(GRmat, diag(1, p))
      G       <- dim(GRmat)[1]
    } else {
      groups_ <- groups
      GRmat   <- groups2mat_OVGLASSO(groups = groups_, Glen = NULL)
      GRmat   <- rbind(GRmat, diag(1, p))
      G       <- dim(GRmat)[1]
      nG      <- rowSums(GRmat)
    }
  } else {
    G  <- dim(GRmat)[1]
    nG <- rowSums(GRmat)
  }
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # Manage group weights
  if (is.null(group_weights)) {
    # Define the weights as in Yuan and Lin 2006, JRSSB
    #group_weights <- sqrt(diag(tcrossprod(GRmat)))
    # Modificata il 19 novembre 2024
    group_weights <- rep(1, G)
  }
  
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # Manage variable weights
  if (is.null(var_weights)) {
    var_weights <- 1.0 / sqrt(colSums(GRmat))
  }
  Tmat_INV <- diag(1.0 / var_weights)
  
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # standardize data (to account adaptive weights)
  X_ <- X %*% Tmat_INV
  #X_ <- X
  
  # :::::::::::::::::::::::::::::::::::::::::::
  # get the lambda max for Overlap Group LASSO
  # lambda_ <- NULL
  # for (g in 1:G) {
  #   idx     <- which(GRmat[g, ] == 1) 
  #   vXy_    <- as.numeric(crossprod(X_[,idx], y_))
  #   lam     <- norm(vXy_, type = "2") / group_weights[g]
  #   lambda_ <- c(lambda_, lam)
  # }
  # lambda.max <- max(lambda_)  / n
  vXy_    <- as.numeric(crossprod(X_, y_))
  lambda_ <- ovglasso_dual_norm_upper_bound(x = vXy_, 
                                            groups = GRmat, 
                                            weights = group_weights)
  lambda.max <- max(lambda_)  / n
  
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # get the lambda min for LASSO
  if (is.null(lambda.min.ratio)) {
    lambda.min.ratio <- ifelse(n < p, 0.01, 1e-04)
  }
  lambda.min <- lambda.min.ratio * lambda.max
  
  # :::::::::::::::::::::::::::::::::::::::::::
  # get the sequence of lambdas
  lambda.seq <- logseq(from = lambda.max, to = lambda.min, length.out = maxl)
  
  # :::::::::::::::::::::::::::::::::::::::::::
  # get output
  res            <- NULL
  res$lambda.min <- lambda.min
  res$lambda.max <- lambda.max
  res$lambda.seq <- lambda.seq
  
  # :::::::::::::::::::::::::::::::::::::::::::
  # get output
  return(res)
}

#' @keywords internal
#' @export
lm_lambdamax <- function(method = c("lasso", "ovglasso"),
                         y, X, p = NULL, q = NULL, groups = NULL, 
                         lambda.min = NULL, maxl = NULL) {
    
  # :::::::::::::::::::::::::::::::::::::::::::
  # check maxl
  if (is.null(maxl)) {
    maxl <- 30
  }

  if (strcmpi(method, "lasso")) {
    res <- lm_lambdamax_LASSO(y, X, lambda.min, maxl)
  } else if (strcmpi(method, "ovglasso")) {
    res <- lm_lambdamax_OVGLASSO(y, X, groups, lambda.min, maxl)
  }
  return(res)
}

# for non consecutive grouping structures, nG specifies the number of variables
# belonging to each group
groups2mat_GLASSO <- function(groups, Glen = NULL) {
  if (is.null(Glen)) {
    grVec <- unique(groups)
    G     <- length(grVec)
    p     <- length(groups)
    GRmat <- matrix(data = 0.0, nrow = G, ncol = p)
    for (i in 1:G) {
      idx          <- groups == grVec[i]
      GRmat[i,idx] <- 1
    }
  } else {
    # non consecutive indeces
    G     <- length(Glen)
    p     <- length(groups)
    GRmat <- matrix(data = 0.0, nrow = G, ncol = p)
    idx1  <- 1
    for (i in 1:G) {
      idx2         <- idx1 + Glen[i] - 1
      idx          <- groups[idx1:idx2]
      GRmat[i,idx] <- 1
      idx1         <- idx2 + 1
    }
  }
  if (max(colSums(GRmat))>1) {
    stop("* groups2mat_GLASSO : groups overlaps.")  
  }
  return(GRmat)
}

# for non consecutive grouping structures, nG specifies the number of variables
# belonging to each group

#' @keywords internal
#' @export
groups2mat_OVGLASSO <- function(groups, Glen = NULL) {
  if (is.null(Glen)) {
    grVec <- unique(groups)
    G     <- length(grVec)
    p     <- length(groups)
    GRmat <- matrix(data = 0.0, nrow = G, ncol = p)
    for (i in 1:G) {
      idx          <- groups == grVec[i]
      GRmat[i,idx] <- 1
    }
  } else {
    # non consecutive indeces
    G     <- length(Glen)
    p     <- length(unique(groups))
    GRmat <- matrix(data = 0.0, nrow = G, ncol = p)
    idx1  <- 1
    for (i in 1:G) {
      idx2 <- idx1 + Glen[i] - 1
      idx          <- groups[idx1:idx2]
      GRmat[i,idx] <- 1
      idx1         <- idx2 + 1
    }
  }
  return(GRmat)
}
# Dual norm of overlap group-lasso via convex optimization
#' @keywords internal
matrix_to_groups <- function(GRmat) {
  apply(GRmat, 1, function(row) which(row == 1))
}
#' @keywords internal
#' @export
ovglasso_dual_norm <- function(x, groups, weights = NULL) {
  
  # require library
  #require(CVXR)
  
  # get dimensions
  p <- length(x)
  
  # convert groups matrix to list
  groups_list <- matrix_to_groups(groups)
  
  if (is.null(weights)) {
    weights <- rep(1, length(groups_list))
  }
  
  # primal variable
  z <- CVXR::Variable(p)
  
  # define primal OGL norm
  group_terms <- lapply(seq_along(groups_list), function(i) {
    g <- groups_list[[i]]
    w <- weights[i]
    w * CVXR::norm2(z[g])
  })
  ogl_norm <- Reduce("+", group_terms)
  
  # dual norm = sup { <x,z> : Omega(z) <= 1 }
  obj    <- CVXR::Maximize(t(x) %*% z)
  constr <- list(ogl_norm <= 1)
  prob   <- CVXR::Problem(obj, constr)
  result <- CVXR::psolve(prob)
  
  # return output
  return(result$value)
}
