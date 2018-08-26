import cvxopt as cvo
import numpy as np

#using cvxopt notation, it takes minimizes x in the following equation:
# 0.5 * xT * P * x + qT * x with constrants G*x <= h, Ax = b

class SVM:
    def __init__(self):
        self.thresh = 1.0e-5
        #suppress output
        cvo.solvers.options['show_progress'] = False
        
    def linear_kernel(self, X):
        return X.dot(X.T)
    
    def get_constraints(self, nb_ex):
        """
        return G, h for the optimization constraint Gx <= h. In this case, it's (-1T)a >= 0 where a is alpha is x
        """
        G = cvo.matrix(np.multiply(-1, np.ones(nb_ex)))
        h = cvo.matrix(np.zeros(nb_ex))
        
        return G, h
    
    def X_reshape(self, X):
        """
        reshape X matrix so that there are ones at the beginning
        """
        nb_ex = X.shape[0]
        aug_X = np.C_[np.ones(nb_ex), X]
        return aug_X
    
    def calc_error(self, X, Y):
        nb_ex = X.shape[0]
        predictions = np.sign(predict(X))
        incorrects = np.sum(np.not_equal(predictions, np.sign(Y)))
        failure_prob = float(incorrects) / float(nb_ex)
        
        return failure_prob
    
    def predict(self, X):
        """
        for SVM, hypothesis is g(x) = sign(w*Tx + b*), where column vector [w* b*] <-- u* from QP solution
        """
        aug_X = X_reshape(X)
        cur_h = np.matmul(aug_X, weights)
        
        return cur_h
    
    def train(self, X, Y):
        """
        we expect X as an N x d matrix and Y as an N x 1 matrix
        
        Problem formulation:
            for cvo: qp(P,q,G,h,A,b) will perform the following:
                - minimize .5(xT)Px + (qT)x
                - subject to Gx <= h
                - and Ax=b (for us, yTa = 0)

            recall we wanted to do the following:
                - minimize .5(wT)w
                - subject to yn((wT)xn + b) >= 1 (n = 1, ..., N)
                in order to find the fattest hyperplane separator to the data
                
            we have aT*[horrid matrix]*a + (-1T)a
            
            G and h are from another function, and we need Gx <= h, so recall that we also have a term sum(n=1)^N of a_n
        """
        X = X.astype(float)
        Y = Y.astype(float)
        nb_ex, dim = X.shape
        
        q = cvo.matrix(np.multiply(-1, np.eye(nb_ex))) # this is out (-1T) that  is getting multiplied by alpha (x) as far as cvxopt is concerned
        P = cvo.matrix(np.multiply(np.outer(Y,Y), linear_kernel(X))) # the quadratic coefficient matrix w/ first term y1y1*K(x1,x1)
        A = cvo.matrix(Y.reshape(1,nb_ex), tc='d')
        b = cvo.matrix(0.0) # 0

        G, h = self.get_constraints(nb_ex)
        
        # get cvo solution which gives us the alphas in 'x'
        cvo_sol = cvo.solvers.qp(P,q,G,h,A,b)
        alphas = np.ravel(cvo_sol['x'])
        
        # get weight vector, sum(n=1, N) an*yn*xn
        YnXn = np.multiply(Y.reshape(nb_ex, 1), X) # get second part
        weights = np.sum(np.multiply(alph.reshape(nb_ex,1), YnXn), axis=0) # reshape to compatible
            
        
        # to get w_0 term, pick a support vector and solve the eqn yn(wTxn + b) = 1
        # --> 1/yn = wTxn + b, (1/yn)-wTxn = b
        alphas_thresh = np.grater_equal(alph, thresh) # get alphas greater than the defined threshold. gives SV by def'n
        sv_idx = np.argmax(alphas_thresh)
        wTxn = np.dot(weights, X[sv_idx]) # use the weight vector we got, and xn (example) is the support vector
        b_val = (1.0 / Y[sv_idx]) - wTxn
        
        self.weights = np.concatenate(([b_val], weights))
        self.alphas = alphas
        self.num_alphas = np.sum(alphas_thresh)