import cvxopt as cvo
import numpy as np

#using cvxopt notation, it takes minimizes x in the following equation:
# 0.5 * xT * P * x + qT * x with constrants G*x <= h, Ax = b

class SVM:
    def __init__(self):
        self.thresh = 1.0e-5
        #suppress output
        cvo.solvers.options['show_progress'] = False
        
    def kernel_calc(self, X):
        return X.dot(X.T)
    
    def get_constraints(self, nb_examples):
        G = cvo.matrix(np.multiply(-1, np.eye(nb_examples)))
        h = cvo.matrix(np.zeros(nb_examples))
        
        return G, h
    
    def X_reshape(self, X):
        nb_ex = X.shape[0]
        real_X = np.c_[np.ones(nb_ex), X]
        return real_X
    
    def calc_error(self, X, Y):
        nb_ex = X.shape[0]
        predicted = np.sign(self.predict(X))
        nb_incorrect = np.sum(np.not_equal(predicted, np.sign(Y)))
        prob_incorrect = float(nb_incorrect) / float(nb_ex)
        
        return prob_incorrect
    
    def predict(self, X):
        real_X = self.X_reshape(X)
        cur_h = np.matmul(real_X, self.weights)
        return cur_h
    
    def train(self, X, Y):
        #expecting X as Nxd matrix and Y as a Nx1 matrix
        X = X.astype(float)
        Y = Y.astype(float)
        nb_ex, cur_dim = X.shape
        
        q = cvo.matrix(np.multiply(-1, np.ones((nb_ex,1))))
        P = cvo.matrix(np.multiply(np.outer(Y, Y), self.kernel_calc(X)))
        A = cvo.matrix(Y.reshape(1, nb_ex), tc='d')
        b = cvo.matrix(0.0)
        G, h = self.get_constraints(nb_ex)
        
        cvo_sol = cvo.solvers.qp(P,q,G,h,A,b)
        alphas = np.ravel(cvo_sol['x'])
        
        # find the weight vector = sum(i=1,N) an*yn*xn
        yx = np.multiply(Y.reshape((nb_ex, 1)),X)
        weights= np.sum(np.multiply(alphas.reshape(nb_ex,1), yx), axis=0)
        
        
        #now we want to find the w0 term so pick an sv and solve
        #yn(wTxn + b) = 1
        #-> 1/yn = wTxn + b, (1/yn)-wTxn = b
        alphas_thresh = np.greater_equal(alphas,self.thresh)
        sv_idx = np.argmax(alphas_thresh)
        wtxn = np.dot(weights, X[sv_idx])
        cur_b = (1.0/Y[sv_idx]) - wtxn
        
        self.weights = np.concatenate(([cur_b], weights))
        self.alphas = alphas
        self.num_alphas = np.sum(alphas_thresh)