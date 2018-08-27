import numpy as np
            
class RegLinReg:
    def __init__(self, l_reg, nlt=False):
        self.l_reg = l_reg # lambda
        self.nlt = nlt == True # whether to use nonlinear

    def set_lambda(self, l_reg):
        self.l_reg = l_reg

    def set_nlt(self, nlt):
        self.nlt = nlt == True

    def X_reshape(self,X):
        nb_ex = X.shape[0]
        if self.nlt == False:
            real_X = np.c_[np.ones(num_examples), X]
        else:
            # transform to (1,x1,x2,x1x2,x1^2,x2^2)
            X_m = np.prod(X, axis=1) # the x1x2 term
            real_X = np.c_[np.ones(nb_ex), X, X_m, np.square(X)]
        return real_X
    
    def predict(self,X):
        real_X = self.X_reshape(X)
        cur_h = np.matmul(real_X, self.weights)
        return cur_h

    def calc_error(self, X, y):
        nb_ex = X.shape[0]
        preds = np.sign(self.predict(X))
        nb_incorrect = np.sum(np.not_equal(preds, np.sign(y)))
        prob_incorrect = float(nb_incorrect) / float(nb_ex)
        return prob_incorrect

    def train_reg(self,X,y):
        # one-shot learning with w = (ZT*Z + lambda*I)^(-1) * ZT*y
        real_X = self.X_reshape(X)
        xTx = np.dot(real_X.T, real_X)
        lambdaI = np.multiply(self.l_reg, np.eye(xtx.shape[0]))
        X_inv = np.linalg.inv(np.add(xTx, lambdaI))
        self.weights = np.dot(X_inv, np.dot(real_X.T, y))