import numpy as np

# in progress
class SGD:
	DEF_SLOPE = 1
	DEF_INTERCEPT = 0
	def __init__(self, iters=1000, lr=0.01, batch_size=1):
		self.slope = DEF_SLOPE
		self.intercept = DEF_INTERCEPT
		self.lr = lr
		self.iters = iters
		self.batch_size = batch_size

# done
def SGD(X, y, lr=0.05, epoch=10, batch_size=1):
    '''
    Stochastic Gradient Descent for a single feature
    '''
    
    m, b = 0.5, 0.5 # initial parameters
    log, mse = [], [] # storage for learning process
    
    for _ in range(epoch):
        
        indexes = np.random.randint(0, len(X), batch_size) # random sample
        
        Xs = np.take(X, indexes)
        ys = np.take(y, indexes)
        N = len(Xs)
        
        f = ys - (m*Xs + b)
        
        # Updating parameters m and b
        m -= lr * (-2 * Xs.dot(f).sum() / N)
        b -= lr * (-2 * f.sum() / N)
        
        log.append((m, b))
        mse.append(mean_squared_error(y, m*X+b))        
    
    return m, b, log, mse
