# Linear regression, single layer
#################################
nn = NNModel(input.dim=3, layers=1, activations='linear')
X.trn = matrix(rnorm(300,0,1), nrow=100)
Y.trn = matrix(5 * X.trn[,1] - 7.25 * X.trn[, 2] + 6.83 * X.trn[, 3], nrow=100)

nn.trn = train(nn,X.trn,Y.trn, epochs=10, mini.batch.size=15, learning.rate=0.5)

nn.prd = predict(nn.trn, X.trn)
Y.trn - nn.prd

plot(Y.trn, nn.prd)
abline(0,1)

# Linear regression, single layer with bias
###########################################
nn = NNModel(input.dim=3, layers=1, activations='linear')
X.trn = matrix(rnorm(300,0,1), nrow=100)
Y.trn = matrix(5 * X.trn[,1] - 7.25 * X.trn[, 2] + 6.83 * X.trn[, 3] + 3.14159, nrow=100)

nn.trn = train(nn,X.trn,Y.trn, epochs=10, mini.batch.size=15, learning.rate=0.5)

nn.prd = predict(nn.trn, X.trn)
Y.trn - nn.prd

plot(Y.trn, nn.prd)
abline(0,1)

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

nn.prd = predict(nn.trn, X.trn)
Y.trn - nn.prd

plot(Y.trn, nn.prd)
abline(0,1)

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

nn.prd = predict(nn.trn, X.trn)
Y.trn - nn.prd

plot(Y.trn, nn.prd)
abline(0,1)

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

nn.prd = predict(nn.trn, X.trn)
Y.trn - nn.prd

plot(Y.trn, nn.prd)
abline(0,1)

# 3 layer, linear with bias
###########################
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

nn.prd = predict(nn.trn, X.trn)
Y.trn - nn.prd

plot(Y.trn, nn.prd)
abline(0,1)



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

nn.prd = predict(nn.trn, X.trn)
Y.trn - nn.prd

plot(Y.trn, nn.prd)
abline(0,1)


# logistic regression
#####################
nn = NNModel(input.dim = 1, layers=1, activation='sigmoid')
X.trn = matrix(c(rnorm(50,mean=2,sd=.5), rnorm(50,mean=4,sd=.5)), nrow=100)
Y.trn = matrix(c(rep(0,50), rep(1,50)))

nn.trn = train(nn,X.trn,Y.trn, epochs=1000, mini.batch.size=15, learning.rate=0.5)

nn.prd = predict(nn.trn, X.trn)
Y.trn - nn.prd

plot(X.trn, Y.trn)
x.plt = seq(0,5.5,.05)
y.plt = predict(nn.trn, x.plt)
points(x.plt, y.plt, pch=19)

# 2 layer, logistic
###################

library(MASS) #for mvrnorm
nn = NNModel(input.dim=2, layers=c(7,1), activations=c('sigmoid','sigmoid'))
X.trn = rbind( mvrnorm(50, mu=c(1,2), Sigma = diag(1,nrow=2,ncol=2)),
               mvrnorm(50, mu=c(3,4), Sigma = diag(1,nrow=2,ncol=2)))
Y.trn = matrix( c(rep(0,50), rep(1,50)) )

nn.trn = train(nn,X.trn,Y.trn, epochs=2500, mini.batch.size=5, learning.rate=0.1)

nn.prd = predict(nn.trn, X.trn)
Y.trn - nn.prd

#plot classes and decision boundary
plot(X.trn[,1], X.trn[,2], col=c('red','blue')[Y.trn+1],
     pch=19)
cut.point = 0.5
for( x1 in seq(from=min(X.trn[,1]), to=max(X.trn[,1]), length.out=50) ) {
  for( x2 in seq(from=min(X.trn[,2]), to=max(X.trn[,2]), length.out=50) ){
    prd = predict(nn.trn, matrix(c(x1,x2), nrow=1))
    clr = c('red', 'blue')[ (prd>cut.point)+1 ]
    points(x1,x2,pch=19,cex=.15,col=clr)
  }
}


# 3 layer, logistic
###################

library(MASS) #for mvrnorm
nn = NNModel(input.dim=2, layers=c(7,7,1), activations=c('sigmoid','sigmoid','sigmoid'))
X.trn = rbind( mvrnorm(50, mu=c(1,2), Sigma = diag(1,nrow=2,ncol=2)),
               mvrnorm(50, mu=c(3,4), Sigma = diag(1,nrow=2,ncol=2)))
Y.trn = matrix( c(rep(0,50), rep(1,50)) )

nn.trn = train(nn,X.trn,Y.trn, epochs=10000, mini.batch.size=5, learning.rate=0.1)

nn.prd = predict(nn.trn, X.trn)
Y.trn - nn.prd

#plot classes and decision boundary
plot(X.trn[,1], X.trn[,2], col=c('red','blue')[Y.trn+1],
     pch=19)
cut.point = 0.5
for( x1 in seq(from=min(X.trn[,1]), to=max(X.trn[,1]), length.out=200) ) {
  for( x2 in seq(from=min(X.trn[,2]), to=max(X.trn[,2]), length.out=200) ){
    prd = predict(nn.trn, matrix(c(x1,x2), nrow=1))
    clr = c('red', 'blue')[ (prd>cut.point)+1 ]
    points(x1,x2,pch=19,cex=.15,col=clr)
  }
}


# 3-layer leaky relu classification
#############################
nn = NNModel(input.dim = 1, layers=c(3,3,1), activation=c('leaky.relu','leaky.relu','leaky.relu'))
X.trn = matrix(c(rnorm(50,mean=2,sd=.5), rnorm(50,mean=4,sd=.5)), nrow=100)
Y.trn = matrix(c(rep(0,50), rep(1,50)))

nn.trn = train(nn,X.trn,Y.trn, epochs=15000, mini.batch.size=15, learning.rate=0.25)

nn.prd = predict(nn.trn, X.trn)
Y.trn - nn.prd

plot(X.trn, Y.trn)
x.plt = seq(0,5.5,.05)
y.plt = predict(nn.trn, x.plt)
points(x.plt, y.plt, pch=19)

# 3 layer, relu with sigmoid out
################################

library(MASS) #for mvrnorm
nn = NNModel(input.dim=2, layers=c(7,7,1), activations=c('relu','relu','sigmoid'))
X.trn = rbind( mvrnorm(100, mu=c(3,3), Sigma = diag(.5,nrow=2,ncol=2)),
               mvrnorm(100, mu=c(4,2), Sigma = diag(.5,nrow=2,ncol=2)))
Y.trn = matrix( c(rep(0,100), rep(1,100)) )

nn.trn = train(nn,X.trn,Y.trn, epochs=500, mini.batch.size=5, learning.rate=0.1)

nn.prd = predict(nn.trn, X.trn)
Y.trn - nn.prd

#plot clsses and decision boundary
plot(X.trn[,1], X.trn[,2], col=c('red','blue')[Y.trn+1],
     pch=19)
cut.point = .5
for( x1 in seq(from=min(X.trn[,1]), to=max(X.trn[,1]), length.out=100) ) {
  for( x2 in seq(from=min(X.trn[,2]), to=max(X.trn[,2]), length.out=100) ){
    prd = predict(nn.trn, matrix(c(x1,x2), nrow=1))
    clr = c('red', 'blue')[ (prd>cut.point)+1 ]
    points(x1,x2,pch=19,cex=.25,col=clr)
  }
}

# 2-layer tanh classification
#############################
library(MASS)
nn = NNModel(input.dim = 2, layers=c(5,1), activation=c('tanh','tanh'))
X.trn = rbind( mvrnorm(50, mu=c(1,2), Sigma = diag(1,nrow=2,ncol=2)),
               mvrnorm(50, mu=c(3,4), Sigma = diag(1,nrow=2,ncol=2)))
Y.trn = matrix( c(rep(-1,50), rep(1,50)) )

nn.trn = train(nn,X.trn,Y.trn, epochs=2000, mini.batch.size=15, learning.rate=0.01)

nn.prd = predict(nn.trn, X.trn)
Y.trn - nn.prd

#plot classes and decision boundary
plot(X.trn[,1], X.trn[,2], col=c(rep('red',50),rep('blue',50)),
     pch=19)
cut.point = 0
for( x1 in seq(from=min(X.trn[,1]), to=max(X.trn[,1]), length.out=200) ) {
  for( x2 in seq(from=min(X.trn[,2]), to=max(X.trn[,2]), length.out=200) ){
    prd = predict(nn.trn, matrix(c(x1,x2), nrow=1))
    clr = c('red', 'blue')[ (prd>cut.point)+1 ]
    points(x1,x2,pch=19,cex=.15,col=clr)
  }
}

# 3 layer, logistic
###################

library(MASS) #for mvrnorm
nn = NNModel(input.dim=2, layers=c(7,7,1), activations=c('sigmoid','sigmoid','sigmoid'))
X.trn = rbind( mvrnorm(50, mu=c(1,2), Sigma = diag(1,nrow=2,ncol=2)),
               mvrnorm(50, mu=c(1.5,2.5), Sigma = diag(5,nrow=2,ncol=2)))
Y.trn = matrix( c(rep(0,50), rep(1,50)) )

nn.trn = train(nn,X.trn,Y.trn, epochs=2500, mini.batch.size=5, learning.rate=0.1)

nn.prd = predict(nn.trn, X.trn)
Y.trn - nn.prd

#plot classes and decision boundary
plot(X.trn[,1], X.trn[,2], col=c('red','blue')[Y.trn+1],
     pch=19)
cut.point = 0.5
for( x1 in seq(from=min(X.trn[,1]), to=max(X.trn[,1]), length.out=200) ) {
  for( x2 in seq(from=min(X.trn[,2]), to=max(X.trn[,2]), length.out=200) ){
    prd = predict(nn.trn, matrix(c(x1,x2), nrow=1))
    clr = c('red', 'blue')[ (prd>cut.point)+1 ]
    points(x1,x2,pch=19,cex=.15,col=clr)
  }
}

# 3-layer tanh classification, clusters
#######################################
library(MASS)
nn = NNModel(input.dim = 2, layers=c(5,1), activation=c('tanh','tanh'))
X.trn = rbind( mvrnorm(75, mu=c(1,2), Sigma = diag(.25,nrow=2,ncol=2)),
               mvrnorm(75, mu=c(3,4), Sigma = diag(.25,nrow=2,ncol=2)),
               mvrnorm(100, mu=c(2,3), Sigma = diag(5,nrow=2,ncol=2)))
Y.trn = matrix( c(rep(-1,150), rep(1,100)) )

nn.trn = train(nn,X.trn,Y.trn, epochs=5000, mini.batch.size=15, learning.rate=0.01)

nn.prd = predict(nn.trn, X.trn)
Y.trn - nn.prd

#plot classes and decision boundary
plot(X.trn[,1], X.trn[,2], col=c(rep('red',150),rep('blue',100)),
     pch=19)
cut.point = 0
for( x1 in seq(from=min(X.trn[,1]), to=max(X.trn[,1]), length.out=200) ) {
  for( x2 in seq(from=min(X.trn[,2]), to=max(X.trn[,2]), length.out=200) ){
    prd = predict(nn.trn, matrix(c(x1,x2), nrow=1))
    clr = c('red', 'blue')[ (prd>cut.point)+1 ]
    points(x1,x2,pch=19,cex=.15,col=clr)
  }
}


# 3-layer sigmoid classification, contained clusters
####################################################
library(MASS)
nn = NNModel(input.dim = 2, layers=c(5,5,1), activation=c('sigmoid','sigmoid','sigmoid'))
X.trn = rbind( mvrnorm(75, mu=c(2,2), Sigma = diag(.25,nrow=2,ncol=2)),
               mvrnorm(75, mu=c(4,4), Sigma = diag(.25,nrow=2,ncol=2)),
               mvrnorm(150, mu=c(3,3), Sigma = diag(5,nrow=2,ncol=2)))
Y.trn = matrix( c(rep(0,150), rep(1,150)) )

nn.trn = train(nn,X.trn,Y.trn, epochs=25000, mini.batch.size=15, learning.rate=0.1)

nn.prd = predict(nn.trn, X.trn)
Y.trn - nn.prd

#plot classes and decision boundary
plot(X.trn[,1], X.trn[,2], col=c(rep('red',150),rep('blue',150)),
     pch=19)
cut.point = 0.5
for( x1 in seq(from=min(X.trn[,1]), to=max(X.trn[,1]), length.out=200) ) {
  for( x2 in seq(from=min(X.trn[,2]), to=max(X.trn[,2]), length.out=200) ){
    prd = predict(nn.trn, matrix(c(x1,x2), nrow=1))
    clr = c('red', 'blue')[ (prd>cut.point)+1 ]
    points(x1,x2,pch=19,cex=.15,col=clr)
  }
}


# 3-layer sigmoid classification, 3 contained clusters
######################################################
library(MASS)
nn = NNModel(input.dim = 2, layers=c(5,5,5,1), activation=c('sigmoid','sigmoid','sigmoid','sigmoid'))
X.trn = rbind( mvrnorm(75, mu=c(2,0), Sigma = diag(.25,nrow=2,ncol=2)),
               mvrnorm(75, mu=c(6,4), Sigma = diag(.25,nrow=2,ncol=2)),
               mvrnorm(75, mu=c(2,6), Sigma = diag(.25,nrow=2,ncol=2)),
               mvrnorm(200, mu=c(3,3), Sigma = diag(5,nrow=2,ncol=2)))
Y.trn = matrix( c(rep(0,225), rep(1,200)) )

nn.trn = train(nn,X.trn,Y.trn, epochs=50000, mini.batch.size=15, learning.rate=0.1)

nn.prd = predict(nn.trn, X.trn)
Y.trn - nn.prd

#plot classes and decision boundary
plot(X.trn[,1], X.trn[,2], col=c(rep('red',225),rep('blue',200)),
     pch=19)
cut.point = 0.5
for( x1 in seq(from=min(X.trn[,1]), to=max(X.trn[,1]), length.out=200) ) {
  for( x2 in seq(from=min(X.trn[,2]), to=max(X.trn[,2]), length.out=200) ){
    prd = predict(nn.trn, matrix(c(x1,x2), nrow=1))
    clr = c('red', 'blue')[ (prd>cut.point)+1 ]
    points(x1,x2,pch=19,cex=.15,col=clr)
  }
}

# 2-layer leaky relu classification, clusters
#############################################
library(MASS)
nn = NNModel(input.dim = 2, layers=c(3,1), activation=c('leaky.relu','sigmoid'))
X.trn = rbind( mvrnorm(75, mu=c(1,2), Sigma = diag(.25,nrow=2,ncol=2)),
               mvrnorm(75, mu=c(3,4), Sigma = diag(.25,nrow=2,ncol=2)),
               mvrnorm(150, mu=c(2,3), Sigma = diag(5,nrow=2,ncol=2)))
Y.trn = matrix( c(rep(0,150), rep(1,150)) )

nn.trn = train(nn,X.trn,Y.trn, epochs=10000, mini.batch.size=15, learning.rate=0.05)

nn.prd = predict(nn.trn, X.trn)
Y.trn - nn.prd

#plot classes and decision boundary
plot(X.trn[,1], X.trn[,2], col=c(rep('red',150),rep('blue',150)),
     pch=19)
cut.point = .5
for( x1 in seq(from=min(X.trn[,1]), to=max(X.trn[,1]), length.out=200) ) {
  for( x2 in seq(from=min(X.trn[,2]), to=max(X.trn[,2]), length.out=200) ){
    prd = predict(nn.trn, matrix(c(x1,x2), nrow=1))
    clr = c('red', 'blue')[ (prd>cut.point)+1 ]
    points(x1,x2,pch=19,cex=.15,col=clr)
  }
}

# 3-layer multinomial, 3 clusters
######################################################
library(MASS)
nn = NNModel(input.dim = 2, layers=c(5,5,3), activation=c('sigmoid','sigmoid','softmax'))
X.trn = rbind( mvrnorm(75, mu=c(2,0), Sigma = diag(.25,nrow=2,ncol=2)),
               mvrnorm(75, mu=c(6,4), Sigma = diag(.25,nrow=2,ncol=2)),
               mvrnorm(75, mu=c(2,6), Sigma = diag(.25,nrow=2,ncol=2)))
Y.trn = matrix( c(rep(c(1,0,0), 75),
                  rep(c(0,1,0), 75),
                  rep(c(0,0,1), 75)), ncol=3, byrow=TRUE )

nn.trn = train(nn,X.trn,Y.trn, epochs=2500, mini.batch.size=15, learning.rate=0.1)

nn.prd = predict(nn.trn, X.trn)
Y.trn - nn.prd

#plot classes and decision boundary
plot(X.trn[,1], X.trn[,2], col=c(rep('red',75),rep('blue',75),rep('green',75)),
     pch=19)

for( x1 in seq(from=min(X.trn[,1]), to=max(X.trn[,1]), length.out=200) ) {
  for( x2 in seq(from=min(X.trn[,2]), to=max(X.trn[,2]), length.out=200) ){
    prd = predict(nn.trn, matrix(c(x1,x2), nrow=1))
    clr = c('red', 'blue', 'green')[ which.max(prd) ]
    points(x1,x2,pch=19,cex=.15,col=clr)
  }
}

# 3-layer multinomial, 2 clusters embedded in larger cluster
############################################################
library(MASS)
nn = NNModel(input.dim = 2, layers=c(5,5,3), activation=c('sigmoid','sigmoid','softmax'))
X.trn = rbind( mvrnorm(150, mu=c(0,0), Sigma = diag(3,nrow=2,ncol=2)),
               mvrnorm(150, mu=c(-1,-1), Sigma = diag(.5,nrow=2,ncol=2)),
               mvrnorm(150, mu=c(1,1), Sigma = diag(.5,nrow=2,ncol=2)))
Y.trn = matrix( c(rep(c(1,0,0), 150),
                  rep(c(0,1,0), 150),
                  rep(c(0,0,1), 150)), ncol=3, byrow=TRUE )

nn.trn = train(nn,X.trn,Y.trn, epochs=2500, mini.batch.size=15, learning.rate=0.1)

nn.prd = predict(nn.trn, X.trn)
Y.trn - nn.prd

#plot classes and decision boundary
plot(X.trn[,1], X.trn[,2], col=c(rep('red',150),rep('blue',150),rep('green',150)),
     pch=19)

for( x1 in seq(from=min(X.trn[,1]), to=max(X.trn[,1]), length.out=200) ) {
  for( x2 in seq(from=min(X.trn[,2]), to=max(X.trn[,2]), length.out=200) ){
    prd = predict(nn.trn, matrix(c(x1,x2), nrow=1))
    clr = c('red', 'blue', 'green')[ which.max(prd) ]
    points(x1,x2,pch=19,cex=.15,col=clr)
  }
}

# 3-layer multinomial, 6 clusters
#################################
library(MASS)
nn = NNModel(input.dim = 2, layers=c(5,5,6), activation=c('sigmoid','sigmoid','softmax'))
X.trn = rbind( mvrnorm(100, mu=c(0,.75), Sigma = diag(.1,nrow=2,ncol=2)),
               mvrnorm(100, mu=c(.75,.5), Sigma = diag(.1,nrow=2,ncol=2)),
               mvrnorm(100, mu=c(.75,-.5), Sigma = diag(.1,nrow=2,ncol=2)),
               mvrnorm(100, mu=c(0,-.75), Sigma = diag(.1,nrow=2,ncol=2)),
               mvrnorm(100, mu=c(-.75,-.5), Sigma = diag(.1,nrow=2,ncol=2)),
               mvrnorm(100, mu=c(-.75,.5), Sigma = diag(.1,nrow=2,ncol=2)))
Y.trn = matrix( c(rep(c(1,0,0,0,0,0), 100),
                  rep(c(0,1,0,0,0,0), 100),
                  rep(c(0,0,1,0,0,0), 100),
                  rep(c(0,0,0,1,0,0), 100),
                  rep(c(0,0,0,0,1,0), 100),
                  rep(c(0,0,0,0,0,1), 100)), ncol=6, byrow=TRUE )

nn.trn = train(nn,X.trn,Y.trn, epochs=2500, mini.batch.size=15, learning.rate=0.1)

nn.prd = predict(nn.trn, X.trn)
Y.trn - nn.prd

#plot classes and decision boundary
plot(X.trn[,1], X.trn[,2], col=c(rep('red',100),rep('blue',100),rep('green',100),
                                 rep('purple',100), rep('darkorange',100), rep('cyan',100)),
     pch=19)

prd.old = which.max(predict(nn.trn, matrix(c(min(X.trn[,1]),min(X.trn[,2])),nrow=1)))
x2.old = min(X.trn[,2])

for( x1 in seq(from=min(X.trn[,1]), to=max(X.trn[,1]), length.out=200) ) {
  for( x2 in seq(from=min(X.trn[,2]), to=max(X.trn[,2]), length.out=200) ) {
    prd = predict(nn.trn, matrix(c(x1,x2), nrow=1))
    clr = c('red', 'blue', 'green','purple','darkorange','cyan')[ which.max(prd) ]
    points(x1,x2,pch=19,cex=.15,col=clr)
    if( which.max(prd) != prd.old & x2 != min(X.trn[,2]) ){
      points(x1,(x2+x2.old)/2,pch=19,cex=0.15,col='black')
    }
    prd.old = which.max(prd)
    x2.old = x2
  }
}


# 3-layer multinomial, 10 randomly placed clusters
##################################################
library(MASS)
nn = NNModel(input.dim = 2, layers=c(5,5,10), activation=c('sigmoid','sigmoid','softmax'),seed=180930)
x1.rnd = runif(10,-1,1)
x2.rnd = runif(10,-1,1)
s.rnd = rnorm(n=10,mean=0.1,sd=.05)
X.trn = rbind( mvrnorm(100, mu=c(x1.rnd[1], x2.rnd[1]), Sigma = diag(s.rnd[1],nrow=2,ncol=2)),
               mvrnorm(100, mu=c(x1.rnd[2], x2.rnd[2]), Sigma = diag(s.rnd[2],nrow=2,ncol=2)),
               mvrnorm(100, mu=c(x1.rnd[3], x2.rnd[3]), Sigma = diag(s.rnd[3],nrow=2,ncol=2)),
               mvrnorm(100, mu=c(x1.rnd[4], x2.rnd[4]), Sigma = diag(s.rnd[4],nrow=2,ncol=2)),
               mvrnorm(100, mu=c(x1.rnd[5], x2.rnd[5]), Sigma = diag(s.rnd[5],nrow=2,ncol=2)),
               mvrnorm(100, mu=c(x1.rnd[6], x2.rnd[6]), Sigma = diag(s.rnd[6],nrow=2,ncol=2)),
               mvrnorm(100, mu=c(x1.rnd[7], x2.rnd[7]), Sigma = diag(s.rnd[7],nrow=2,ncol=2)),
               mvrnorm(100, mu=c(x1.rnd[8], x2.rnd[8]), Sigma = diag(s.rnd[8],nrow=2,ncol=2)),
               mvrnorm(100, mu=c(x1.rnd[9], x2.rnd[9]), Sigma = diag(s.rnd[9],nrow=2,ncol=2)),
               mvrnorm(100, mu=c(x1.rnd[10], x2.rnd[10]), Sigma = diag(s.rnd[10],nrow=2,ncol=2)))
Y.trn = matrix( c(rep(c(1,0,0,0,0,0,0,0,0,0), 100),
                  rep(c(0,1,0,0,0,0,0,0,0,0), 100),
                  rep(c(0,0,1,0,0,0,0,0,0,0), 100),
                  rep(c(0,0,0,1,0,0,0,0,0,0), 100),
                  rep(c(0,0,0,0,1,0,0,0,0,0), 100),
                  rep(c(0,0,0,0,0,1,0,0,0,0), 100),
                  rep(c(0,0,0,0,0,0,1,0,0,0), 100),
                  rep(c(0,0,0,0,0,0,0,1,0,0), 100),
                  rep(c(0,0,0,0,0,0,0,0,1,0), 100),
                  rep(c(0,0,0,0,0,0,0,0,0,1), 100)), ncol=10, byrow=TRUE )

nn.trn = train(nn,X.trn,Y.trn, epochs=1000, mini.batch.size=25, learning.rate=0.1)

nn.prd = predict(nn.trn, X.trn)
Y.trn - nn.prd

#plot classes and decision boundary
plot(X.trn[,1], X.trn[,2], col=rep(rainbow(10),each=100), pch=19)

prd.old = which.max(predict(nn.trn, matrix(c(min(X.trn[,1]),min(X.trn[,2])),nrow=1)))
x2.old = min(X.trn[,2])

for( x1 in seq(from=min(X.trn[,1]), to=max(X.trn[,1]), length.out=200) ) {
  for( x2 in seq(from=min(X.trn[,2]), to=max(X.trn[,2]), length.out=200) ) {
    prd = predict(nn.trn, matrix(c(x1,x2), nrow=1))
    clr = rainbow(10)[ which.max(prd) ]
    points(x1,x2,pch=19,cex=.15,col=clr)
    if( which.max(prd) != prd.old & x2 != min(X.trn[,2]) ){
      points(x1,(x2+x2.old)/2,pch=19,cex=0.15,col='black')
    }
    prd.old = which.max(prd)
    x2.old = x2
  }
}


# Spiral problem
################

#data generation modified from:  https://stats.stackexchange.com/questions/164048/can-a-random-forest-be-used-for-feature-selection-in-multiple-linear-regression
#basic
n <- 1:1500
r <- 0.05*n +1 
th <- n*(4*pi)/max(n)

#polar to cartesian
x1=r*cos(th) 
y1=r*sin(th)

#add noise
x2 <- x1+0.1*r*runif(min = -1,max = 1,n=length(n))
y2 <- y1+0.1*r*runif(min = -1,max = 1,n=length(n))

#append salt and pepper
x3 <- runif(min = min(x2),max = max(x2),n=length(n)/2)
y3 <- runif(min = min(y2),max = max(y2),n=length(n)/2)

#assemble data into frame 
X.trn <- matrix(c(x2,x3,y2,y3), nrow=length(n)*1.5, byrow=FALSE)
Y.trn <- matrix(c(rep(0,length(n)), rep(1,length(n)/2)), nrow=length(n)*1.5, byrow=FALSE)

nn = NNModel(input.dim=2, layers=c(5,5,5,1), activations=c('leaky.relu','leaky.relu','leaky.relu','sigmoid'))
nn.trn = train(nn,X.trn,Y.trn, epochs=2500, mini.batch.size=25, learning.rate=0.1)

#plot 
plot(X.trn[,1], X.trn[,2],pch=19, col=c('red','blue')[Y.trn +1])

#plot classes and decision boundary
cut.point = 0.5

prd.old = predict(nn.trn, matrix(c(min(X.trn[,1]),min(X.trn[,2])),nrow=1)) > cut.point
x2.old = min(X.trn[,2])

for( x1 in seq(from=min(X.trn[,1]), to=max(X.trn[,1]), length.out=200)) {
  for( x2 in seq(from=min(X.trn[,2]), to=max(X.trn[,2]), length.out=200)) {
    prd = predict(nn.trn, matrix(c(x1,x2), nrow=1))
    clr = c('red', 'blue')[ (prd>cut.point)+1 ]
    points(x1,x2,pch=19,cex=.15,col=clr)
    if( (prd > cut.point) != (prd.old > cut.point) & x2 != min(X.trn[,2]) ){
      points(x1,(x2+x2.old)/2,pch=19,cex=0.25,col='black')
    }
    prd.old = prd
    x2.old = x2
  }
}

MNIST
#####

# the MNIST dataset was downloaded from the Digit Recognizer project on Kaggle
# https://www.kaggle.com/c/digit-recognizer
# the Training set file was converted into a pixels file and a file with one hot labels, used here
# using the Convert_MNIST_to_NNFS.R script 
# Data is not included as I don't know if I can distribute it and it is quite large

# get accuracy function for later use
get.acc = function(nn, X.trn, target, show.conf.mat = FALSE) {
  prd = apply( X.trn, 1, function(r) {
    which.max(predict(nn, matrix(r, nrow=1))) - 1
  })
  t = table(target, prd)
  if( show.conf.mat ){
    print(t)
  }
  sum(diag(t))/sum(t)
}

#plot a random set of misclassified examples and show actual/predicted class
plot.random.misclass = function(nn.trn, X, Y) {
  #get 9 random mistakes
  prd = apply( X, 1, function(r) {
    which.max(predict(nn.trn, matrix(r, nrow=1))) - 1
  })
  trg = apply(Y, 1, function(r) {
    which.max(r) - 1
  })
  e = which(prd != trg)
  e.pick = sample(e, size = min(9,length(e)))
  
  plot(1:100,1:100, type='n', axes=FALSE, xlab='', ylab='', ylim=c(100,0) )
  
  for( i in 0:2 ){
    for( j in 0:2 ) {
      
      q = i + j*3 + 1
      if( q <= length(e)) {
        z = matrix(X[e.pick[q], ], nrow = 28, byrow=FALSE)
        actual = trg[e.pick[q]]
        pred = prd[e.pick[q]]
        
        start.x = 2 + i * 34
        start.y = 2 + j * 34
        delta.x = start.x - 1
        delta.y = start.y - 1
        
        text(start.x-1,start.y-2, pos = 4,
             sprintf('Actual: %d Predicted %d', actual, pred))
        for( x in start.x:(start.x + 27) ){
          for( y in start.y:(start.y + 27) ){
            rect(x, y+1, x+1, y, 
                 col = rgb(z[x-delta.x,y-delta.y],
                           z[x-delta.x,y-delta.y],
                           z[x-delta.x,y-delta.y], maxColorValue = 255),
                 border = NA)
          }
        }
      }
    }
  }
}

# get orginal data
d.pixels = read.csv('../MNIST_dataset/mnist_pixels.csv', header = TRUE)
d.onehot.labels = read.csv('../MNIST_dataset/mnist_labels_onehot.csv', header = TRUE)

# build training, validation, test sets
set.seed(181117)
indices = sample(1:42000, 42000, replace = FALSE)
train.set = indices[1:21000]
val.set = indices[21001:31500]
test.set = indices[31501:42000]

# reduced set sizes for experiments and tuning
train.set = indices[1:2100]
val.set = indices[2101:3150]
test.set = indices[3151:4200]

X.trn = as.matrix(d.pixels[train.set, ])
Y.trn = as.matrix(d.onehot.labels[train.set, ])
digit.trn = apply(Y.trn, 1, function(r) {
  which.max(r) - 1
})

X.val = as.matrix(d.pixels[val.set, ])
Y.val = as.matrix(d.onehot.labels[val.set, ])
digit.val = apply(Y.val, 1, function(r) {
  which.max(r) -1 
})

X.tst = as.matrix(d.pixels[test.set, ])
Y.tst = as.matrix(d.onehot.labels[test.set, ])
digit.tst = apply(Y.tst, 1, function(r) {
  which.max(r) - 1
})

# 2-layer
nn.trn = NNModel(input.dim = 784, layers=c(15, 10), activation=c('sigmoid', 'softmax'))
learning.rate = 0.1
n.epochs = 250

# 3-layer
nn.trn = NNModel(input.dim = 784, layers=c(20, 20, 10), 
                 activation=c('sigmoid', 'sigmoid', 'softmax'))
learning.rate = 0.1
n.epochs = 1000

# 5-layer
nn.trn = NNModel(input.dim = 784, layers=c(25, 20, 15, 10, 10), 
                 activation=c('leaky.relu', 'leaky.relu', 'leaky.relu', 'leaky.relu', 'softmax'))
learning.rate = 0.01
n.epochs = 5000

# multiple epochs with plotting between epochs
# set up accuracy plotting
plot( 0:n.epochs, seq(0,1,length.out=(n.epochs+1)), type = 'n',
      xlab = 'Epoch #', ylab = 'Accuracy')
grid()
points( c(0,0,0), c(get.acc(nn.trn, X.trn, digit.trn),
                    get.acc(nn.trn, X.val, digit.val),
                    get.acc(nn.trn, X.tst, digit.tst)),
        bg = c('blue', 'darkorange', 'red'), pch = 21, col='black')
legend('bottomright', legend=c('Training', 'Validation', 'Test'),
       pt.bg = c('blue', 'darkorange', 'red'), pch=21, col = 'black')

#look at the confusion matrices before training
get.acc(nn.trn, X.trn, digit.trn, show.conf.mat = TRUE)
get.acc(nn.trn, X.val, digit.val, show.conf.mat = TRUE)
get.acc(nn.trn, X.tst, digit.tst, show.conf.mat = TRUE)

for( e in 1:n.epochs) {
  nn.trn = train(nn.trn,X.trn,Y.trn, epochs=1, mini.batch.size=100, learning.rate=learning.rate)
  points( rep(e,3), c(get.acc(nn.trn, X.trn, digit.trn),
                      get.acc(nn.trn, X.val, digit.val),
                      get.acc(nn.trn, X.tst, digit.tst)),
          bg = c('blue', 'darkorange', 'red'), pch = 21, col='black')
}

 #confusion matrices after training
get.acc(nn.trn, X.trn, digit.trn, show.conf.mat = TRUE)
get.acc(nn.trn, X.val, digit.val, show.conf.mat = TRUE)
get.acc(nn.trn, X.tst, digit.tst, show.conf.mat = TRUE)

plot.random.misclass(nn.trn,X.trn,Y.trn)
plot.random.misclass(nn.trn,X.val,Y.val)
plot.random.misclass(nn.trn,X.tst,Y.tst)



  