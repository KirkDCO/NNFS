# Linear regression, single layer
nn = NNModel(input.dim=3, layers=1, activations='linear', learning.rate=0.1)
X.trn = matrix(rnorm(300,0,1), nrow=100)
Y.trn = matrix(5 * X.trn[,1] - 7.25 * X.trn[, 2] + 6.83 * X.trn[, 3], nrow=100)
nn.trn = train(nn,X.trn,Y.trn, epochs=10, mini.batch.size=15)

# 2 layer, linear
nn.orig = NNModel(input.dim=3, layers=c(3,1), activations=c('linear','linear'), learning.rate=0.5)
nn.orig$layers$L1$weights = matrix(c(1,1,1,2,2,2,3,3,3), nrow=3)
nn.orig$layers$L2$weights = matrix(c(1,1,1), nrow=3)
X.trn = matrix(rnorm(300,0,1), nrow=100)
nn.fp = forward.prop(nn.orig,X.trn)
Y.trn = nn.fp$layers$L2$z

nn = NNModel(input.dim=3, layers=c(3,1), activations=c('linear','linear'), learning.rate=0.5)
nn.trn = train(nn,X.trn,Y.trn, epochs=10, mini.batch.size=15)
