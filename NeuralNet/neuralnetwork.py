import numpy as np
class NeuralNetwork:
    def __init__(self, structure, lamda, alpha):
        #structure = number of neurons per layer in each element
        self.weights = None
        self.structure = structure
        self.initialize_weights()
        self.lamda=lamda
        self.alpha=alpha

    def initialize_weights(self):
        self.weights = []
        for layer in range(len(self.structure) - 1):
            curTheta = self.calcThetaArr(layer)
            self.weights.append(curTheta)

    def calcThetaArr(self, layer):
            cols = self.structure[layer]+1  # Number of neurons in the current layer
            rows = self.structure[layer+1]  # Number of neurons in the next layer

            theta_arr = np.random.randn(rows, cols) #gaussian dist between -1 to 1
            
            return theta_arr
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def forward_propagation(self, x_i):
        a = np.array(x_i) 
        for theta in self.weights:
            a_biased = np.insert(a, 0, 1)
            z = np.dot(theta, a_biased)  
            a = self.sigmoid(z)  
        return a  
    
    def costFunction(self, X, y): #minibatch, and their true vals
        J=0
        m=len(X)
        predictions=[]
        for i in range(m): #iterate over minibatch, find costs and accumulate in J
            x_i=X[i]
            y_i=y[i]

            predict=self.forward_propagation(x_i)
            predictions.append(predict)
            cost_i= np.sum(-np.array(y_i) * np.log(predict) - (1-np.array(y_i))*np.log(1-predict))
            J=J+cost_i

        J/=m

        regularization = 0
        for theta in self.weights:
            regularization += np.sum(np.square(theta[:, 1:])) 

        regularization = (self.lamda / (2 * m)) * regularization
        regularCost = J + regularization
        return [regularCost, predictions]


    def backProp(self, X, y):
        m = len(X)  
        Gradients = [np.zeros_like(theta) for theta in self.weights]

        for i in range(m):
            x_i = X[i]
            y_i = y[i]

            activations = [x_i]
            a = np.array(x_i)
          
            for theta in self.weights:
                a_biased = np.insert(a, 0, 1)
                z = np.dot(theta, a_biased)
                a = self.sigmoid(z)
                activations.append(a)

            deltas = []
            delta = activations[-1] - y_i  # output delta
            deltas.append(delta)

            for i in range(len(self.structure) - 2, 0, -1):  # throw out input and output layers
                theta = self.weights[i]

                activations_i = np.insert(activations[i], 0, 1)
                deltaCur = np.matmul(theta.T, deltas[-1]) * activations_i * (1 - activations_i)
                deltas.append(deltaCur[1:])
            #print("Deltas:", deltas)


            for j in range(len(self.weights)):
                activations_j = np.insert(activations[j], 0, 1)
                Gradients[j] += np.outer(deltas[-(j+1)], activations_j)
            #print("Gradients:",Gradients)

        for j in range(len(self.weights)):
            P = self.lamda * self.weights[j]
            P[:, 0] = 0
            Gradients[j] += P
            Gradients[j] /= m
        #print("Gradients:",Gradients)
        return Gradients
    
    def descent(self, Gradients):
        newWeights=[]
        for j in range(len(self.weights)):
            newWeights.append(self.weights[j] - self.alpha * Gradients[j])
        self.weights=newWeights
            

#-------------------------------------------------------------







