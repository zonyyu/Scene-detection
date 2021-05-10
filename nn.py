import numpy as np
from utils import *

class Layer:
    def __init__(self, n, act, dropout):
        self.features = n 
        self.activation = act
        self.dropout = dropout

        self.A = None
        self.Z = None
        self.W = None
        self.b = None

        self.D = None

        self.dA = None
        self.dZ = None
        self.dW = None
        self.db = None

        self.vdW = None
        self.vdb = None
        self.sdW = None
        self.sdb = None

    def init_weights(self, this_n, prev_n):
        if self.activation == "relu":
            xavier = np.sqrt(2/prev_n)
        else:
            xavier = np.sqrt(1/prev_n)
        self.W = (np.random.rand(this_n, prev_n) - np.ones((this_n, prev_n))*0.5 )*xavier
        self.b = np.zeros((this_n, 1))

    def act(self, Z):
        if self.activation == "relu":
            return np.maximum(0, Z)
        elif self.activation == "sigmoid":
            return sigmoid(Z)
        elif self.activation == "linear":
            return Z
        else:
            print("\"" + self.activation + "\" is not recognized.")
            raise NameError

    def dact(self, Z):
        if self.activation == "relu":
            return (Z > 0)
        if self.activation == "sigmoid":
            return sigmoid(Z) * (1-sigmoid(Z))
        if self.activation == "linear":
            return np.ones(Z.shape)
        else: 
            print("\"" + self.activation + "\" is not recognized.")
            raise NameError


class Model:
    def __init__(self, lr, reg, momentum, rmsprop):
        self.lr = lr
        self.reg = reg
        self.momentum = momentum
        self.rmsprop = rmsprop

        self.l = []
        self.X_train = None
        self.Y_train = None
        self.X_test = None
        self.Y_test = None

    def add_layer(self, features, activation, dropout):
        self.l.append(Layer(features, activation, dropout))

    def summary(self):
        print("#================ Model Summary =================#")
        print("Input: nodes:", self.l[0].features)
        for i in range(1, len(self.l)):
            print("Layer " + str(i) + ": nodes:", self.l[i].features, 
                " Weights shape:", self.l[i].W.shape, " Bias shape:", 
                self.l[i].b.shape, " Activation:", self.l[i].activation)

    def connect(self):
        for i in range(1, len(self.l)):
            self.l[i].init_weights(self.l[i].features, self.l[i-1].features)


    def forward_prop(self, X):
        self.l[0].A = X
        L = len(self.l)-1
        for i in range(1, L+1):
            self.l[i].Z = np.dot(self.l[i].W, self.l[i-1].A) + self.l[i].b
            self.l[i].A = self.l[i].act(self.l[i].Z)

            # Dropout
            self.l[i].D = np.random.rand(self.l[i].A.shape[0], self.l[i].A.shape[1]) > self.l[i].dropout
            self.l[i].A *= self.l[i].D
            self.l[i].A /= (1-self.l[i].dropout)

    def back_prop(self, A, Y):
        L = len(self.l) - 1
        m = Y.shape[1]

        self.l[L].dZ = A - Y
        self.l[L].dW = np.dot(self.l[L].dZ, self.l[L-1].A.T)/m + self.reg/m * self.l[L].W
        self.l[L].db = np.sum(self.l[L].dZ, axis=1, keepdims=True)/m
        for i in range(1, L):
            self.l[L-i].dA = (np.dot(self.l[L-i+1].W.T, self.l[L-i+1].dZ) * self.l[L-i].D)
            self.l[L-i].dA /= (1-self.l[L-i].dropout)
            self.l[L-i].dZ = self.l[L-i].dA * self.l[L-i].dact(self.l[L-i].Z)
            self.l[L-i].dW = np.dot(self.l[L-i].dZ, self.l[L-i-1].A.T)/m + self.reg/m * self.l[L-i].W
            self.l[L-i].db = np.sum(self.l[L-i].dZ, axis=1, keepdims=True)/m

    def cost(self, A, Y):
        m = Y.shape[1]
        n = Y.shape[0]
        bound = 0.999999
        A = bound*A + 0.5*(1-bound)
        C = np.sum(np.sum(-Y*np.log(A)-(1-Y)*np.log(1-A), axis=1, keepdims=True)/m, axis=0, keepdims=True)/n
        return C[0][0]
    def Adam(self, momentum, rmsprop):
        L = len(self.l) - 1
        for i in range(1, L + 1):
            self.l[i].vdW = momentum*self.l[i].vdW + (1-momentum)*self.l[i].dW
            self.l[i].vdb = momentum*self.l[i].vdb + (1-momentum)*self.l[i].db
            self.l[i].sdW = rmsprop*self.l[i].sdW + (1-rmsprop)*self.l[i].dW*self.l[i].dW
            self.l[i].sdb = rmsprop*self.l[i].sdb + (1-rmsprop)*self.l[i].db*self.l[i].db

    def reset_optimizer(self):
        L = len(self.l) - 1
        for i in range(1, L + 1):
            self.l[i].vdW = np.zeros(self.l[i].W.shape)
            self.l[i].vdb = np.zeros(self.l[i].b.shape)
            self.l[i].sdW = np.zeros(self.l[i].W.shape)
            self.l[i].sdb = np.zeros(self.l[i].b.shape)

    def update_params(self):
        L = len(self.l) - 1
        for i in range(1, L + 1):
            self.l[i].W -= self.lr*self.l[i].vdW/(np.sqrt(self.l[i].sdW)+1e-8)
            self.l[i].b -= self.lr*self.l[i].vdb/(np.sqrt(self.l[i].sdb)+1e-8)

    def train(self, X_train, Y_train, X_test=None, Y_test=None, epochs=1, batch_size=32):

        for i in range(epochs):
            m_tot = Y_train.shape[1]
            batches = m_tot // batch_size
            if m_tot%batch_size > 0:
                batches += 1

            self.reset_optimizer()
            
            for j in range(batches):
                if j != batches-1:
                    X = X_train[:, j*batch_size:(j+1)*batch_size]
                    Y = Y_train[:, j*batch_size:(j+1)*batch_size]
                else:
                    X = X_train[:, j*batch_size:m_tot]
                    Y = Y_train[:, j*batch_size:m_tot]

                self.forward_prop(X)
                A = self.l[len(self.l)-1].A

                training_cost = self.cost(A, Y)
                print("Iteration", j, "out of " + str(batches) + ", Training Cost: ", training_cost)

                self.back_prop(A, Y)

                self.Adam(self.momentum, self.rmsprop)
                self.update_params()








if __name__  == "__main__":
    model = Model(lr=1e-5, reg=0.3, momentum=0, rmsprop=0)

    X_train, Y_train, X_test, Y_test = load_data("Dataset")

    model.add_layer(features=64*64*3, activation="relu", dropout=0)
    model.add_layer(features=256, activation="relu", dropout=0)
    model.add_layer(features=256, activation="relu", dropout=0)
    model.add_layer(features=6, activation="sigmoid", dropout=0)
    model.connect()
    model.summary()


    model.train(X_train, Y_train, epochs=1, batch_size=32)

    
    


    
