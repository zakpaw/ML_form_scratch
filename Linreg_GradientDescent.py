#%%
class GradientDescent:
    def __init__(self):
        self.w, self.b = 0.0, 0.0

    def train(self, x, y, alpha=1e-3, epchos=1500):
        for e in range(epchos):
            self.epoch(x, y, alpha)

            if e % 400 == 0:
                print(f"epoch:{e}\tloss:{GradientDescent.avg_loss(x, y, self.w, self.b)}")

    def predict(self, x):
        return [self.w*xi + self.b for xi in x]

    def epoch(self, x, y, alpha):
        dl_dw = 0.0
        dl_bw = 0.0
        N = len(x)

        for i in range(N):
            dl_dw += -2*x[i]*(y[i]-(self.w*x[i]+self.b))
            dl_bw += -2*(y[i]-(self.w*x[i]+self.b))

        self.w = self.w - (1/float(N))*dl_dw*alpha
        self.b = self.b - (1/float(N))*dl_bw*alpha

    @staticmethod
    def avg_loss(x, y, w, b):
        error = 0.0
        for i in range(len(x)):
            error += (y[i] - (w*x[i] + b))**2
        return error / float(len(x))

#%%
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

x, y = make_regression(n_samples=500, n_features=1, noise=30)
plt.scatter(x, y, c='orange', alpha=0.6)
# %%
linreg = GradientDescent()
linreg.train(x, y)
pred = linreg.predict(x)
# %%
plt.scatter(x, y)
plt.plot(x, pred, c='orange')
