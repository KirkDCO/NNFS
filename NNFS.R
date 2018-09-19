# Neural Net from scratch
# Robert Kirk DeLisle
# 2018.07.05
#

#######################
## Activation functions
#######################

linear = function(a=NULL) {
  a
}

sigmoid = function(a=NULL) {
  1/(1+exp(-a))
}

leaky.relu = function(a=NULL) {
  a[which(a<0)] = .1 *a[which(a<0)]
  a
}

relu = function(a=NULL) {
  a[qhich(a<0)] = 0
  a
}

softmax = function(a=NULL) {
  t(apply(a, 1, function(r) {
    exp(r)/sum(exp(r))
  }))
}

# tanh is already defined in R

######################################
## Derivatives of activation functions
######################################

d.linear = function(z=NULL) {
  # z is the matrix of activations and only supplies the dimensions
  # for the ouput
  
  matrix(1, nrow=dim(z)[1], ncol=dim(z)[2])
}

d.sigmoid = function(z=NULL) {
  # z is the activation of the layer of interest
  # in other words, it is z = sigmoid(a)
  # derivative can be achieved with simple calculation if
  # the original activations are cached
  
  z * (1-z)
}

d.tanh = function(z=NULL) {
  # z is the activation of the layer of interest
  # in other words, it is z = tanh(z)
  # derivative can be achieved with simple calculation if
  # the original activations are cached
  
  1 - z*z
}

d.softmax = function(z=NULL, y=NULL) {
  # z = the set of activations at the softmax layer
  # y = the target classes as one-hot vector
  # assumed that the loss function is the log loss = -sum(y(i) * log(p(i)))
  # with p(i) = softmax(z)
  
  z - y 
}

d.leaky.relu = function(z=NULL) {
  # z is the activation of the layer of interest
  
  z.ret = rep(.1,length(z))
  z.ret[ which(z>0) ] = 1
  z.ret
}

d.relu = function(z=NULL) {
  # z is the activation of the layer of interest
  
  z.ret = rep(0,length(z))
  z.ret[ which(z>0) ] = 1
  z.ret
}


#################
# Error Functions
#################

error.linear = function(Y=NULL, Y.hat=NULL) {
  Y - Y.hat
}

error.sigmoid = function(Y=NULL, Y.hat=NULL) {
  Y - Y.hat
}

error.relu = function(Y=NULL, Y.hat=NULL) {
  Y - Y.hat
}

error.leaky.relu = function(Y=NULL, Y.hat=NULL) {
  Y - Y.hat
}

error.tanh = function(Y=NULL, Y.hat=NULL) {
  Y - Y.hat
}

error.softmax = function(Y=NULL, Y.hat=NULL) {
  Y - Y.hat
}


###############
# Define the NN
###############

NNModel = function( input.dim = NULL, layers = NULL, activations = NULL, 
                    batch.norm = NULL, drop.out.rate = NULL, seed=20150808) {
  
  # input.dim = dimensions of input
  # layers = vector of number of nodes per hidden/output layer
  # activations = vector of activations types per layer
  #               can be linear, tanh, sigmoid, relu, leaky.relu, and softmax
  #               softmax is assumed to be the last layer for multicategory output
  # batch.nrom = vector of TRUE/FALSE designating whether batch norm is performed for each layer
  # drop.out.rate = vector of values for drop-out rate per layer
  # assumed all vectors same length = number of layers
  
  # returns a list structured for later computations
  
  #set the random seed to ensure ability to regenerate this network
  set.seed(seed)
  
  nn = list( layers = list())
  
  nn$layers = lapply( 1:(length(layers)), function(l) {
    if( l == 1){
      weights = matrix( rnorm( input.dim*layers[l], 0, 0.01), 
                        nrow = input.dim )
    }else{
      weights = matrix( rnorm(layers[l-1] * layers[l], 0, 0.01), 
                        nrow = layers[l-1] )
    }
    list( activation = activations[l],
          weights = weights, 
          bias = matrix(rnorm(layers[l]), ncol=layers[l]))
  })
  
  names(nn$layers) = paste('L', 1:(length(layers)), sep='')
  nn
}

#################
# Model functions
#################

forward.prop = function(NNmod = NULL, X.trn=NULL){
  
  # NNmod = list generated from NNmod() function above
  # X = matrix of inputs (1 x m) of appropriate dimensions for the NNmod
  #     effectively one row from the training matrix
  
  # returns a list of the outputs generated at each layer
  #         for use in back propagation
  
  layers = names(NNmod$layers)
  
  for(l in 1:length(layers)){
    
    if( l == 1 ){ 
      a = X.trn %*% NNmod$layers[[layers[l]]]$weights
    }else{
      a = NNmod$layers[[layers[l-1]]]$z %*% NNmod$layers[[layers[l]]]$weights
    }
    a = a + rep(NNmod$layer[[layers[l]]]$bias, each=nrow(a))
    NNmod$layers[[layers[l]]]$a = a
    
    if( NNmod$layers[[layers[l]]]$activation == 'linear' ){
      NNmod$layers[[layers[l]]]$z = linear(a)
    }else if( NNmod$layers[[layers[l]]]$activation == 'sigmoid' ){
      NNmod$layers[[layers[l]]]$z = sigmoid(a)
    }else if( NNmod$layers[[layers[l]]]$activation == 'relu' ){
      NNmod$layers[[layers[l]]]$z = relu(a)
    }else if( NNmod$layers[[layers[l]]]$activation == 'leaky.relu' ){
      NNmod$layers[[layers[l]]]$z = leaky.relu(a)
    }else if( NNmod$layers[[layers[l]]]$activation == 'tanh' ){
      NNmod$layers[[layers[l]]]$z = tanh(a)
    }else if( NNmod$layers[[layers[l]]]$activation == 'softmax' ){
      NNmod$layers[[layers[l]]]$z = softmax(a)
    }
  }
  
  NNmod
}

back.prop = function(NNmod=NULL, X.trn=NULL, Y.trn=NULL, learning.rate=NULL) {
  
  # NNModel = list generated from NNModel() function above
  # Y.trn = vector of the actual y-values being modeled
  # X.trn = training set (or minibatch) for this round
  
  # returns a NNModel list with updated weights
  
  NNmod.old = NNmod
  
  layers = names(NNmod.old$layers)
  for( l in length(layers):1 ){
    layer = layers[l]
    
    #get the right derivative function
    if( NNmod.old$layers[[layer]]$activation == 'linear' ){
      d = d.linear
      error = error.linear
    }else if( NNmod.old$layers[[layer]]$activation == 'sigmoid' ){
      d = d.sigmoid
      error = error.sigmoid
    }else if( NNmod.old$layers[[layer]]$activation == 'relu' ){
      d = d.relu
      error = error.relu
    }else if( NNmod.old$layers[[layer]]$activation == 'leaky.relu' ){
      d = d.leaky.relu
      error = error.leaky.relu
    }else if(NNmod.old$layers[[layer]]$activation == 'tanh' ){
      d = d.tanh
      error = error.tanh
    }else if(NNmod.old$layers[[layer]]$activation == 'softmax' ){
      d = d.softmax
      error = error.softmax
    }
    
    # compute deltas
    if( l == length(layers) ){  #we're at the output layer
      if( NNmod.old$layers[[layer]]$activation == 'softmax'){
        err = t(error(Y.trn, NNmod.old$layers[[layer]]$z))
        delta = lapply(seq_len(ncol(err)), function(i) as.matrix(err[,i], ncol=1))
      }else{
        delta = lapply( error(Y.trn, NNmod.old$layers[[layer]]$z), function(v) {v}) 
      }
    }else{
      next.layer = layers[l+1]
      delta = mapply(function(del, z) {
        wt.del = NNmod.old$layers[[next.layer]]$weights %*% del
        wt.del * d(as.matrix(z, nrow=dim(wt.del)[1], ncol=dim(wt.del[2])))},
          delta, split(NNmod.old$layers[[layer]]$z, 
                       row(NNmod.old$layers[[layer]]$z), drop=FALSE),
        SIMPLIFY=FALSE)
    }

    #compute weight adjustments
    if( l == 1 ) {   #we're at the first layer
      wt.adj = mapply( function(del,x) {
        del %*% x}, delta, split(X.trn, row(X.trn)),
        SIMPLIFY=FALSE)
    }else{
      prev.layer = layers[l-1]
      wt.adj = mapply(function(del, z) {del %*% z}, 
                      delta, split(NNmod.old$layers[[prev.layer]]$z, 
                                   row(NNmod.old$layers[[prev.layer]]$z)),
        SIMPLIFY=FALSE)
    }
    avg.wt.adj = Reduce('+', wt.adj)/length(wt.adj)
    
    #compute bias adjustments
    bias.adj = lapply(delta, function(del) {del})
    avg.bias.adj = Reduce('+', bias.adj)/length(bias.adj)
    
    #adjust weights and biases by average gradient
    NNmod$layers[[layer]]$weights = NNmod$layers[[layer]]$weights +
      learning.rate * t(avg.wt.adj)
    NNmod$layers[[layer]]$bias = NNmod$layers[[layer]]$bias +
      learning.rate * t(avg.bias.adj)
  }
  
  NNmod
}

train = function(NNmod=NULL, X.trn=NULL, Y.trn=NULL, X.tst=NULL, Y.tst=NULL,
                 mini.batch.size=NULL, epochs=NULL, learning.rate=0.1, seed=20180808) {
  
  # NNModel =list generated from NNModel() function above
  # X.trn = the training dataset, organized as observations X covariates (nXm)
  # Y.trn = the expected outputs for each observations in X
  # X.tst = a test set of data for monitoring during training
  # Y.tst = expected ouptus for the test set
  # mini.batch.size = number of observations used in each training step
  # epochs = number of total passes through the dataset
  
  # returns a NNModel list with trained weights
  
  #set the random seed here
  set.seed(seed)
  
  #compute number of batches needed for a full epoch
  batches = ceiling(dim(X.trn)[1]/mini.batch.size)
    
  for( e in 1:epochs ){
    mini.batch.order = sample(dim(X.trn)[1])
    for( mb in 0:(batches-1) ){
      mb.start = (mini.batch.size*mb+1)
      mb.stop = min((mini.batch.size*(mb+1)), dim(X.trn)[1])
      mini.batch.X = X.trn[ mini.batch.order[mb.start:mb.stop], , drop=FALSE]
      mini.batch.Y = Y.trn[ mini.batch.order[mb.start:mb.stop], , drop=FALSE]
      
      NNmod = forward.prop(NNmod, X.trn=mini.batch.X)
      NNmod = back.prop(NNmod, X.trn=mini.batch.X, Y.trn=mini.batch.Y,
                        learning.rate=learning.rate)
    }
  }
  NNmod
}

predict = function( NNModel=NULL, X=NULL) {
  
  # NNModel =list generated from NNModel() function above
  # X = matrix the full training dataset organized as observations X covariates (nXm)
  
  # returns a vector of predictions for each observation in the X set
  # assume input dimensions and order of covariates are consistent with the
  # supplied model
  
  prd.net = forward.prop(nn.trn, X)
  output.layer = names(NNModel$layers)[length(names(NNModel$layers))]
  prd.net$layers[[output.layer]]$z
}
