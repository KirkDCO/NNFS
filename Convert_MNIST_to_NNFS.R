# make training set compatible with NNFS.R

tr = read.csv('train.csv', header=TRUE)
tr.label = tr[,1]
tr.pixels = tr[,-1]

tr.label.onehot = t(sapply(tr.label, function(l) {
  out = rep(0, 10)
  out[l+1] = 1
  out
} ))

write.csv(tr.pixels, 'mnist_pixels.csv', row.names = FALSE)
write.csv(tr.label.onehot, 'mnist_labels_onehot.csv', row.names = FALSE)
