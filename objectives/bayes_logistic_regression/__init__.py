import numpy as np
from objectives import Energy
import torch


class BayesianLogisticRegression(Energy):
    def __init__(self, data, labels, batch_size=None,
                 loc=0.0, scale=1.0):
        """
        Bayesian Logistic Regression model (assume Normal prior)
        :param data: data for Logistic Regression task
        :param labels: label for Logistic Regression task
        :param batch_size: batch size for Logistic Regression; setting it to None
        adds flexibility at the cost of speed.
        :param loc: mean of the Normal prior
        :param scale: std of the Normal prior
        """
        super(BayesianLogisticRegression, self).__init__()
        self.x_dim = int(data.shape[1])
        self.y_dim = int(labels.shape[1])
        self.dim = self.x_dim * self.y_dim + self.y_dim

        self.mu_prior = torch.ones([self.dim]).cuda() * loc
        self.sig_prior = torch.ones([self.dim]).cuda() * scale


        self.data = torch.from_numpy(data).to(torch.float32).cuda()
        self.labels = torch.from_numpy(labels).to(torch.float32).cuda()
        self.z = torch.zeros(batch_size, self.dim)

        if batch_size:
            self.data = (self.data).view(1, -1, self.x_dim).tile((batch_size, 1, 1))
            self.labels = (self.labels).view(1, -1, self.y_dim).tile((batch_size, 1, 1))
        else:
            self.data = (self.data).view(1, -1, self.x_dim).tile((self.z.shape[0], 1, 1))
            self.labels = (self.labels).view(1, -1, self.y_dim).tile((self.z.shape[0], 1, 1))
            
    def _vector_to_model(self, v):
        w = v[:, :-self.y_dim]
        b = v[:, -self.y_dim:]
        w = torch.reshape(w, [-1, self.x_dim, self.y_dim])
        b = torch.reshape(b, [-1, 1, self.y_dim])
        return w, b

    def energy_fn(self, v, x, y):
        w, b = self._vector_to_model(v)
        logits = torch.matmul(x, w) + b
        ll = torch.nn.BCEWithLogitsLoss(reduction='none')(logits, y)
        ll = torch.sum(ll, axis=[1, 2])
        pr = torch.square((v - self.mu_prior) / self.sig_prior)
        pr = 0.5 * torch.sum(pr, axis=1)
        return pr + ll

    def __call__(self, v):
        return self.energy_fn(v, self.data, self.labels)

    def evaluate(self, zv, path=None):
        z, v = zv
        z_ = np.reshape(z, [-1, z.shape[-1]])
        m = np.mean(z_, axis=0, dtype=np.float64)
        v = np.std(z_, axis=0, dtype=np.float64)
        print('mean: {}'.format(m))
        print('std: {}'.format(v))



    @staticmethod
    def mean():
        return None

    @staticmethod
    def std():
        return None
