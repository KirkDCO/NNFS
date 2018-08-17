# Linear regression, single layer
nn = NNModel(input.dim=3, layers=1, activations='linear')
X.trn = matrix(rnorm(300,0,1), nrow=100)
Y.trn = matrix(5 * X.trn[,1] - 7.25 * X.trn[, 2] + 6.83 * X.trn[, 3], nrow=100)
nn.trn = train(nn,X.trn,Y.trn, epochs=10, mini.batch.size=15, learning.rate=0.5)

# 2 layer, linear
nn.orig = NNModel(input.dim=3, layers=c(3,1), activations=c('linear','linear'))
nn=nn.orig
nn.orig$layers$L1$weights = matrix(c(1,1,1,2,2,2,3,3,3), nrow=3)
nn.orig$layers$L2$weights = matrix(c(1,1,1), nrow=3)
X.trn = matrix(rnorm(300,0,1), nrow=100)
nn.fp = forward.prop(nn.orig,X.trn)
Y.trn = nn.fp$layers$L2$z
nn.trn = train(nn,X.trn,Y.trn, epochs=50, mini.batch.size=15, learning.rate=0.25)

# 3 layer, linear
nn.orig = NNModel(input.dim=4, layers=c(3,2,1), activations=c('linear','linear','linear'))
nn = nn.orig
nn.orig$layers$L1$weights = matrix(c(rep(1,4),rep(2,4),rep(3,4)), nrow=4)
nn.orig$layers$L2$weights = matrix(c(rep(1,3),rep(2,3)), nrow=3)
nn.orig$layers$L3$weights = matrix(rep(3,2),nrow=2)
X.trn = matrix(rnorm(400,0,1), nrow=100)
nn.fp = forward.prop(nn.orig,X.trn)
Y.trn = nn.fp$layers$L3$z

nn.trn = train(nn,X.trn,Y.trn, epochs=10, mini.batch.size=15, learning.rate=0.1)

nn.tst = nn.orig
nn.tst$layers$L1$weights = matrix(seq(0.01,0.12,0.01), nrow=4)
nn.tst$layers$L2$weights = matrix(seq(0.13,0.18,0.01), nrow=3)
nn.tst$layers$L3$weights = matrix(seq(0.19,0.20,0.01),nrow=2)
nn.trn = train(nn.tst,X.trn,Y.trn, epochs=10, mini.batch.size=15, learning.rate=0.1)

# 3 layer, small, linear
nn.orig = NNModel(input.dim=2, layers=c(2,2,1), activations=c('linear','linear','linear'))
nn = nn.orig
nn.orig$layers$L1$weights = matrix(c(1,2,3,4), nrow=2)
nn.orig$layers$L2$weights = matrix(c(1,2,3,4), nrow=2)
nn.orig$layers$L3$weights = matrix(c(1,2),nrow=2)
X.trn = matrix(rnorm(200,0,1), nrow=100)
nn.fp = forward.prop(nn.orig,X.trn)
Y.trn = nn.fp$layers$L3$z

nn.trn = train(nn,X.trn,Y.trn, epochs=10, mini.batch.size=15, learning.rate=0.1)

nn.tst = nn.orig
nn.tst$layers$L1$weights = matrix(seq(0.01,0.04,0.01), nrow=2)
nn.tst$layers$L2$weights = matrix(seq(0.05,0.08,0.01), nrow=2)
nn.tst$layers$L3$weights = matrix(seq(0.09,0.10,0.01),nrow=2)
nn.trn = train(nn.tst,X.trn,Y.trn, epochs=10, mini.batch.size=15, learning.rate=0.1)
