# Linear regression, single layer
#################################
nn = NNModel(input.dim=3, layers=1, activations='linear')
X.trn = matrix(rnorm(300,0,1), nrow=100)
Y.trn = matrix(5 * X.trn[,1] - 7.25 * X.trn[, 2] + 6.83 * X.trn[, 3], nrow=100)

nn.trn = train(nn,X.trn,Y.trn, epochs=10, mini.batch.size=15, learning.rate=0.5)

nn.prd = forward.prop(nn.trn, X.trn)
Y.trn - nn.prd$layers$L1$z


# Linear regression, single layer with bias
###########################################
nn = NNModel(input.dim=3, layers=1, activations='linear')
X.trn = matrix(rnorm(300,0,1), nrow=100)
Y.trn = matrix(5 * X.trn[,1] - 7.25 * X.trn[, 2] + 6.83 * X.trn[, 3] + 3.14159, nrow=100)

nn.trn = train(nn,X.trn,Y.trn, epochs=10, mini.batch.size=15, learning.rate=0.5)

nn.prd = forward.prop(nn.trn, X.trn)
Y.trn - nn.prd$layers$L1$z


# 2 layer, linear
#################
nn.trg = NNModel(input.dim=2, layers=c(2,1), activations=c('linear','linear'), seed=20180917)
nn=NNModel(input.dim=2, layers=c(2,1), activations=c('linear','linear'), seed=71908102)
nn.trg$layers$L1$weights = matrix(c(1,1,2,2), nrow=2)
nn.trg$layers$L2$weights = matrix(c(1,1), nrow=2)
X.trn = matrix(rnorm(200,0,1), nrow=100)
nn.fp = forward.prop(nn.trg,X.trn)
Y.trn = nn.fp$layers$L2$z

nn.trn = train(nn,X.trn,Y.trn, epochs=500, mini.batch.size=15, learning.rate=0.01)

nn.prd = forward.prop(nn.trn, X.trn)
Y.trn - nn.prd$layers$L2$z


# 2 layer, linear, with bias
############################
nn.trg = NNModel(input.dim=2, layers=c(2,1), activations=c('linear','linear'), seed=20180917)
nn=NNModel(input.dim=2, layers=c(2,1), activations=c('linear','linear'), seed=71908102)
nn.trg$layers$L1$weights = matrix(c(1,1,2,2), nrow=2)
nn.trg$layers$L1$bias = matrix(c(.3,1), nrow=1)
nn.trg$layers$L2$weights = matrix(c(1,1), nrow=2)
nn.trg$layers$L2$bias = matrix(0, nrow=1, ncol=1)
X.trn = matrix(rnorm(200,0,1), nrow=100)
nn.fp = forward.prop(nn.trg,X.trn)
Y.trn = nn.fp$layers$L2$z

nn.trn = train(nn,X.trn,Y.trn, epochs=500, mini.batch.size=15, learning.rate=0.01)

nn.prd = forward.prop(nn.trn, X.trn)
Y.trn - nn.prd$layers$L2$z



# 3 layer, linear
#################
nn.trg = NNModel(input.dim=4, layers=c(3,2,1), activations=c('linear','linear','linear'))
nn = nn.trg
nn.trg$layers$L1$weights = matrix(c(rep(1,4),rep(2,4),rep(3,4)), nrow=4)
nn.trg$layers$L2$weights = matrix(c(rep(1,3),rep(2,3)), nrow=3)
nn.trg$layers$L3$weights = matrix(rep(3,2),nrow=2)
X.trn = matrix(rnorm(400,0,1), nrow=100)
nn.fp = forward.prop(nn.trg,X.trn)
Y.trn = nn.fp$layers$L3$z

nn.trn = train(nn,X.trn,Y.trn, epochs=500, mini.batch.size=5, learning.rate=0.001)

nn.prd = forward.prop(nn.trn, X.trn)
Y.trn - nn.prd$layers$L3$z

# 3 layer, linear with bias
#################
nn.trg = NNModel(input.dim=4, layers=c(3,2,1), activations=c('linear','linear','linear'))
nn = nn.trg
nn.trg$layers$L1$weights = matrix(c(rep(1,4),rep(2,4),rep(3,4)), nrow=4)
nn.trg$layers$L1$bias = matrix( c(.3,.2,.1), nrow=1)
nn.trg$layers$L2$weights = matrix(c(rep(1,3),rep(2,3)), nrow=3)
nn.trg$layers$L2$bias = matrix( c(0,0), nrow=1)
nn.trg$layers$L3$weights = matrix(rep(3,2),nrow=2)
nn.trg$layers$L3$bias = matrix(.5, nrow=1, ncol=1)
X.trn = matrix(rnorm(400,0,1), nrow=100)
nn.fp = forward.prop(nn.trg,X.trn)
Y.trn = nn.fp$layers$L3$z

nn.trn = train(nn,X.trn,Y.trn, epochs=500, mini.batch.size=5, learning.rate=0.001)

nn.prd = forward.prop(nn.trn, X.trn)
Y.trn - nn.prd$layers$L3$z



# 3 layer, small, linear
########################
nn.trg = NNModel(input.dim=2, layers=c(2,2,1), activations=c('linear','linear','linear'))
nn = nn.trg
nn.trg$layers$L1$weights = matrix(c(1,2,3,4), nrow=2)
nn.trg$layers$L2$weights = matrix(c(1,2,3,4), nrow=2)
nn.trg$layers$L3$weights = matrix(c(1,2),nrow=2)
X.trn = matrix(rnorm(200,0,1), nrow=100)
nn.fp = forward.prop(nn.trg,X.trn)
Y.trn = nn.fp$layers$L3$z

nn.trn = train(nn,X.trn,Y.trn, epochs=1000, mini.batch.size=15, learning.rate=0.001)

nn.prd = forward.prop(nn.trn, X.trn)
Y.trn - nn.prd$layers$L3$z


# logistic regression
#####################
nn = NNModel(input.dim = 1, layers=1, activation='sigmoid')
X.trn = matrix(c(rnorm(50,mean=2,sd=.5), rnorm(50,mean=4,sd=.5)), nrow=100)
Y.trn = matrix(c(rep(0,50), rep(1,50)))

nn.trn = train(nn,X.trn,Y.trn, epochs=10000, mini.batch.size=15, learning.rate=0.5)

nn.prd = forward.prop(nn.trn, X.trn)
Y.trn - nn.prd$layers$L1$z

plot(X.trn, Y.trn)
x.plt = seq(0,5.5,.05)
y.plt = forward.prop(nn.trn, x.plt)$layers$L1$z
points(x.plt, y.plt, pch=19)

# 3 layer, logistic
###################
library(MASS) #for mvrnorm
nn = NNModel(input.dim=2, layers=c(7,7,1), activations=c('sigmoid','sigmoid','sigmoid'))
X.trn = rbind( mvrnorm(50, mu=c(2,4), Sigma = diag(1,nrow=2,ncol=2)),
               mvrnorm(50, mu=c(4,2), Sigma = diag(1,nrow=2,ncol=2)))
Y.trn = matrix( c(rep(0,50), rep(1,50)) )

plot(X.trn[,1], X.trn[,2], col=c('red','blue')[Y.trn+1],
     pch=19)

nn.trn = train(nn,X.trn,Y.trn, epochs=1000, mini.batch.size=5, learning.rate=0.0001)

nn.prd = forward.prop(nn.trn, X.trn)
Y.trn - nn.prd$layers$L3$z

#plotted boundary always looks linear
cut.point = mean(nn.prd$layers$L3$z)
for( x1 in seq(from=min(X.trn[,1]), to=max(X.trn[,1]), length.out=100) ) {
  for( x2 in seq(from=min(X.trn[,2]), to=max(X.trn[,2]), length.out=100) ){
    prd = forward.prop(nn.trn, matrix(c(x1,x2), nrow=1))$layers$L3$z
    clr = c('red', 'blue')[ (prd<cut.point)+1 ]
    points(x1,x2,pch=19,cex=.25,col=clr)
  }
}

