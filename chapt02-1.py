import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training, datasets, iterators, optimizers
from chainer.training import extensions
import numpy as np

batch_size = 10
uses_device = 0

class MNIST_Conv_NN(chainer.Chain):
    def __init__(self):
        super(MNIST_Conv_NN, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(1, 8, ksize=3)
            self.linear1 = L.Linear(1352, 10)

    def __call__(self, x, t=None, train=True):
        h1 = self.conv1(x)
        h2 = F.relu(h1)
        h3 = F.max_pooling_2d(h2, 2)
        h4 = self.linear1(h3)

        return F.softmax_cross_entropy(h4, t) if train else F.Softmax(h4)

model = MNIST_Conv_NN()

if uses_device >= 0:
    chainer.cuda.get_device_from_id(0).use()
    chainer.cuda.check_cuda_available()
    model.to_gpu()

train, test = chainer.datasets.get_mnist(ndim=3)

train_iter = iterators.SerialIterator(train, batch_size, shuffle=True)
test_iter = iterators.SerialIterator(test, batch_size, repeat=False, shuffle=False)

optimizer = optimizers.Adam()
optimizer.setup(model)

updater = training.StandardUpdater(train_iter, optimizer, device=uses_device)
trainer = training.Trainer(updater, (5, 'epoch'), out="result")

trainer.extend(extensions.Evaluator(test_iter, model, device=uses_device))
trainer.extend(extensions.ProgressBar())

trainer.run()

chainer.serializers.save_hdf5('chapt02.hdf5', model)
