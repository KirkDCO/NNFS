# Linear regression, single layer
#################################
nn = NNModel(input.dim=3, layers=1, activations='linear')
X.trn = matrix(rnorm(300,0,1), nrow=100)
Y.trn = matrix(5 * X.trn[,1] - 7.25 * X.trn[, 2] + 6.83 * X.trn[, 3], nrow=100)
nn.trn = train(nn,X.trn,Y.trn, epochs=10, mini.batch.size=15, learning.rate=0.5)

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

