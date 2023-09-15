# Authors: Mauro Bernardi 
#          Department Statistical Sciences
#       	 University of Padova
#      	   Via Cesare Battisti, 241
#      	   35121 PADOVA, Italy
#          E-mail: mauro.bernardi@unipd.it  

# Last change: July 20, 2020

# :::::::::::::::::::::::::::::::::::::::::::::::::
# Functions
# 0. lambdamax (wrap)
# 1. getlambda_LASSO
# 2. getlambda_ELNET
# 3. getlambda_OVGLASSO
# 4. penfun
# 5. getGroupMat_OVGLASSO

# 4. penfun
penfun <- function(method, p, q, splOrd, regpars) {
  
  # get the number of groups
  if (q == 1) {
    B <- p - splOrd + 1
  } else {
    B <- (p - splOrd + 1) * (q - splOrd + 1)
  }
  
  # lasso
  if (method == "lasso") {
    Cmat         <- diag(1L, p * q)
    Pmat         <- Cmat
    Cvec         <- rep(1L, p * q)
    grMat        <- NULL
    grMatWeights <- diag(1L, p * q)
    norme        <- NULL
  }
  
  # ridge
  if (method == "ridge") {
    Cmat         <- diag(1L, p * q)
    Pmat         <- Cmat
    Cvec         <- rep(1L, p * q)
    grMat        <- NULL
    grMatWeights <- diag(1L, p * q)
    norme        <- NULL
  }
  
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # overlap group lasso (function-to-function)
  # no overall group
  if ((method == "ogl") & (q > 1)) {
    
    # get the group matrices
    grMat        <- getGroupMat_OVGLASSO(p = p, q = q, splOrd = splOrd)
    grMatWeights <- 1.0 / apply(grMat, 2, sum)
    
    if (is.null(regpars)) {
      Cpen <- NULL
      Cmat <- NULL
      Cvec <- NULL
      norme <- NULL
    } else {
      
      # get the weights
      norme_     <- grMat %*% (regpars^2)
      
      # Set the norm to a given threshold
      norme <- fSet2GivenTolerance(x = norme_, n = B, toler = .Machine$double.eps)
      
      Cvec       <- as.vector(t(grMat) %*% (1.0 / (2.0 * sqrt(norme))))
      Cpen       <- diag(1, p * q)
      diag(Cpen) <- Cvec
      Cmat       <- invvec(Cvec, p, q)
    }
  }
  
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # overlap group lasso (function-to-function)
  # overall group
  if ((method == "ogl&1") & (q > 1)) {
    # get the group matrices
    grMat        <- getGroupMat_OVGLASSO(p = p, q = q, splOrd = splOrd)
    grMat        <- rbind(grMat, rep(1L, p * q))
    grMatWeights <- 1.0 / apply(grMat, 2, sum)
    
    # get the weights
    if (is.null(regpars)) {
      Cpen  <- NULL
      Cmat  <- NULL
      Cvec  <- NULL
      norme <- NULL
    } else {
      # get the weights
      # Cvec   <- 1.0 / (2.0 * sqrt(crossprod(regpars)))
      norme_ <- grMat %*% (regpars^2)
      
      # Set the norm to a given threshold
      norme <- fSet2GivenTolerance(x = norme_, n = B, toler = .Machine$double.eps)
      
      # add overall group
      #Cvec       <- rep(Cvec, p*q) + as.vector(t(grMat) %*% (1.0 / (2.0 * sqrt(norme))))
      Cvec       <- as.vector(t(grMat) %*% (1.0 / (2.0 * sqrt(norme))))
      Cpen       <- diag(1, p * q)
      diag(Cpen) <- Cvec
      Cmat       <- invvec(Cvec, p, q)
    }
  }
  
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # overlap group lasso (function-to-scalar)
  # no overall group
  if ((method == "ogl") & (q == 1)) {
    
    # get the group matrices
    grMat        <- getGroupMat_OVGLASSO(p = p, q = 1, splOrd = splOrd)
    grMatWeights <- 1.0 / apply(grMat, 2, sum)
    
    if (is.null(regpars)) {
      Cpen <- NULL
      Cmat <- NULL
      Cvec <- NULL
      norme <- NULL
    } else {
      
      # get the weights
      norme_    <- grMat %*% (regpars^2)
      
      # Set the norm to a given threshold
      norme <- fSet2GivenTolerance(x = norme_, n = B, toler = .Machine$double.eps)
      
      Cvec       <- as.vector(t(grMat) %*% (1.0 / (2.0 * sqrt(norme))))
      Cpen       <- diag(1, p)
      diag(Cpen) <- Cvec
      Cmat       <- Cpen
    }
  }
  
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # overlap group lasso (function-to-scalar)
  # no overall group
  if ((method == "ogl&1") & (q == 1)) {
    
    # get the group matrices
    grMat        <- getGroupMat_OVGLASSO(p = p, q = 1, splOrd = splOrd)
    grMat        <- rbind(grMat, rep(1L, p))
    grMatWeights <- 1.0 / apply(grMat, 2, sum)
    
    if (is.null(regpars)) {
      Cpen  <- NULL
      Cmat  <- NULL
      Cvec  <- NULL
      norme <- NULL
    } else {
      
      # get the weights
      norme_    <- grMat %*% (regpars^2)
      
      # Set the norm to a given threshold
      norme <- fSet2GivenTolerance(x = norme_, n = B, toler = .Machine$double.eps)
      
      Cvec       <- as.vector(t(grMat) %*% (1.0 / (2.0 * sqrt(norme))))
      Cpen       <- diag(1, p)
      diag(Cpen) <- Cvec
      Cmat       <- Cpen
    }
  }
  
  # get the output
  res              <- NULL
  res$Cpen         <- Cpen
  res$Cmat         <- Cmat
  res$Cvec         <- Cvec
  res$grMat        <- grMat
  res$grMatWeights <- grMatWeights
  res$regp.grnorm  <- norme
  
  # Return output
  return(res)
}

# 5. getGroupMat_OVGLASSO
# get groups of overlap group LASSO
getGroupMat_OVGLASSO <- function(p, q, splOrd = c(3, 4)) {
  
  if (q > 1) {
    # get the number of groups
    B     <- (p - splOrd + 1) * (q - splOrd + 1)
    grMat <- matrix(0, B, p * q)
    
    # get the groups matrix
    if ((splOrd == 3) || (splOrd == 4)) {
      # spline of order 3
      if (splOrd == 3) {
        cnt <- 1
        for (i in 0:(p-splOrd)) {
          for (j in 1:(q-splOrd+1)){
            ind             <- c(q*i+j, q*i+j+1, q*i+j+2, p+q*i+j, p+q*i+j+1, p+q*i+j+2, 2*p+q*i+j, 2*p+q*i+j+1, 2*p+q*i+j+2)
            grMat[cnt, ind] <- 1
            cnt             <- cnt + 1
          }
        }
      }
      
      # spline of order 4
      if (splOrd == 4) {
        # cnt <- 1
        # for (i in 0:(p-splOrd)) {
        #   for (j in 1:(q-splOrd+1)) {
        #     ind             <- c(q*i+j, q*i+j+1, q*i+j+2, q*i+j+3, p+q*i+j, p+q*i+j+1, p+q*i+j+2, p+q*i+j+3, 2*p+q*i+j, 2*p+q*i+j+1, 2*p+q*i+j+2, 2*p+q*i+j+3, 3*p+q*i+j, 3*p+q*i+j+1, 3*p+q*i+j+2, 3*p+q*i+j+3)
        #     grMat[cnt, ind] <- 1
        #     cnt             <- cnt + 1
        #   }
        # }
        # cnt <- 1
        # for (i in 0:(p - splOrd)) {
        #   init_row <- i * q
        #   for (j in 1:(q - splOrd + 1)) {
        #     init_col <- j
        #     v1       <- init_row + init_col + 0:(splOrd-1)
        #     v2       <- q + init_row + init_col + 0:(splOrd-1)
        #     v3       <- 2*q + init_row + init_col + 0:(splOrd-1)
        #     v4       <- 3*q + init_row + init_col + 0:(splOrd-1)
        #     ind      <- c(v1, v2, v3, v4)
        # 
        #     grMat[cnt, ind] <- 1
        #     cnt             <- cnt + 1
        #   }
        # }
        B <- (p - splOrd + 1) * (q - splOrd + 1)
        grMat <- matrix(0, B, p * q)
        cnt <- 1
        for (j in 1:(q - splOrd + 1)) {
          init_col <- j
          for (i in 0:(p - splOrd)) {
            init_row <- i*q
            
            v1  <- init_row + init_col + 0:3
            v2  <- q + init_row + init_col + 0:3
            v3  <- 2*q + init_row + init_col + 0:3
            v4  <- 3*q + init_row + init_col + 0:3
            ind <- c(v1, v2, v3, v4)
            
            grMat[cnt, ind] <- 1
            cnt <- cnt + 1
          }
        }
      }
    } else {
      cat("Error in getGroupMat_OVGLASSO, the spline order in not equat to 3 or 4!\n")
    }
  } else if (q == 1) {
    # get the number of groups
    B     <- (p - splOrd + 1)
    grMat <- matrix(0, B, p)
    
    # get the groups matrix
    if ((splOrd == 3) || (splOrd == 4)) {
      # spline of order 3
      if (splOrd == 3) {
        cnt <- 1
        for (i in 0:(p-splOrd)) {
          ind             <- c(i+1, i+2, i+3)
          grMat[cnt, ind] <- 1
          cnt             <- cnt + 1
        }
      }
      
      # spline of order 4
      if (splOrd == 4) {
        cnt <- 1
        for (i in 0:(p-splOrd)) {
          ind             <- c(i+1, i+2, i+3, i+4)
          grMat[cnt, ind] <- 1
          cnt             <- cnt + 1
        }
      }
    } else {
      cat("Error in getGroupMat_OVGLASSO, the spline order in not equat to 3 or 4!\n")
    }
  } else {
    cat("q is negative or equal to zero in getGroupMat_OVGLASSO!\n")
  }
  
  # return output
  return(grMat)
}

