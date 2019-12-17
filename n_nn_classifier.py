#Importing the required libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

def sigmoid(x):
    return 1/(1 + np.exp(-x))

#Derivative of the sigmoid function
def sigmoid_prime(x):
    return sigmoid(x)*(1.0 - sigmoid(x))

class NeuralNetwork(object):
    def __init__(self, architecture, classification):
        #architecture - numpy array with ith element representing the number of neurons in the ith layer.
        
        #Initialize the network architecture
        self.L = architecture.size - 1 #L corresponds to the last layer of the network.
        self.n = architecture #n stores the number of neurons in each layer
        #input_size is the number of neurons in the first layer i.e. n[0]
        #output_size is the number of neurons in the last layer i.e. n[L]
        
        self.n_class = classification[1]
        self.classes = classification[0]
        
        #Parameters will store the network parameters, i.e. the weights and biases
        self.parameters = {}
        
        #Initialize the network weights and biases:
        for i in range (1, self.L + 1): 
            #Initialize weights to small random values
            self.parameters['W' + str(i)] = np.random.randn(self.n[i], self.n[i - 1]) * 0.01
            
            #Initialize rest of the parameters to 1
            self.parameters['b' + str(i)] = np.random.randn(self.n[i], self.n_class)
            self.parameters['z' + str(i)] = np.random.randn(self.n[i], self.n_class)
            self.parameters['a' + str(i)] = np.random.randn(self.n[i], self.n_class)
        
        #As we started the loop from 1, we haven't initialized a[0]:
        self.parameters['a0'] = np.ones((self.n[0], self.n_class))
        
        #Initialize the cost:
        self.parameters['C'] = np.ones((1, self.n_class))
        
        #Create a dictionary for storing the derivatives:
        self.derivatives = {}
    
    def printShape(self, label, x):
        print('Shape of ' + label)
        print(x.shape)
        print(x)
        k=0
                    
    def forward_propagate(self, X):
        
        #Note that X here, is just one training example
        self.parameters['a0'] = np.hstack([X]*self.n_class)
        
        #Calculate the activations for every layer l
        for l in range(1, self.L + 1):
            self.parameters['z' + str(l)] = np.add(np.dot(self.parameters['W' + str(l)], self.parameters['a' + str(l - 1)]), self.parameters['b' + str(l)])
            #self.printShape('z' + str(l), self.parameters['z' + str(l)])
            self.parameters['a' + str(l)] = sigmoid(self.parameters['z' + str(l)])
        
    def compute_cost(self, y):
        self.parameters['C'] = -(y*np.log(self.parameters['a' + str(self.L)]) + (1-y)*np.log( 1 - self.parameters['a' + str(self.L)]))
    
    def compute_derivatives(self, y):
        #Partial derivatives of the cost function with respect to z[L], W[L] and b[L]:        
        #dzL
        self.derivatives['dz' + str(self.L)] = self.parameters['a' + str(self.L)] - y
        
        #self.printShape('dz' + str(self.L), self.derivatives['dz' + str(self.L)])
        #self.printShape('a' + str(self.L), self.parameters['a' + str(self.L)])
        #self.printShape('y', y)
        
        #dWL
        self.derivatives['dW' + str(self.L)] = np.dot(self.derivatives['dz' + str(self.L)], np.transpose(self.parameters['a' + str(self.L - 1)]))
        #self.printShape('dW' + str(self.L), self.derivatives['dW' + str(self.L)])
        #self.printShape('a' + str(self.L - 1), self.parameters['a' + str(self.L - 1)])
        #dbL
        self.derivatives['db' + str(self.L)] = self.derivatives['dz' + str(self.L)]
        #self.printShape('db' + str(self.L), self.derivatives['db' + str(self.L)])
        

        #Partial derivatives of the cost function with respect to z[l], W[l] and b[l]
        for l in range(self.L-1, 0, -1):
            self.derivatives['dz' + str(l)] = np.dot(np.transpose(self.parameters['W' + str(l + 1)]), self.derivatives['dz' + str(l + 1)])*sigmoid_prime(self.parameters['z' + str(l)])
            #self.printShape('dz' + str(l), self.derivatives['dz' + str(l)])
            self.derivatives['dW' + str(l)] = np.dot(self.derivatives['dz' + str(l)], np.transpose(self.parameters['a' + str(l - 1)]))
            #self.printShape('dW' + str(l), self.derivatives['dW' + str(l)])
            self.derivatives['db' + str(l)] = self.derivatives['dz' + str(l)]
            #self.printShape('db' + str(l), self.derivatives['db' + str(l)])
            
    def update_parameters(self, alpha):
        for l in range(1, self.L+1):
            self.parameters['W' + str(l)] -= alpha*self.derivatives['dW' + str(l)]
            self.parameters['b' + str(l)] -= alpha*self.derivatives['db' + str(l)]
        
    def predict(self, x):
        self.forward_propagate(x)
        return self.parameters['a' + str(self.L)]
        
    def fit(self, X, Y, num_iter, alpha = 0.01):
        for iter in range(0, num_iter):
            c =  np.zeros((1, self.n_class))#Stores the cost
            n_c = 0 #Stores the number of correct predictions
            
            print(X.shape)
            for i in range(0, X.shape[0]):
                x = X[i].reshape((X[i].size, 1))
                y = Y[i]
                
                #print(str(i) + 'f')
                self.forward_propagate(x)
                #print(str(i) + 'c')
                self.compute_cost(self.classes[y])
                #print(str(i) + 'd')
                self.compute_derivatives(self.classes[y])
                #print(str(i) + 'u')
                self.update_parameters(alpha)

                c += self.parameters['C']

                y_pred = self.predict(x)
                yi = np.argmax(y_pred, 1)
                if (yi == y):
                    n_c += 1
                if i%10000 == 0:
                    print(str(i) + 'looping')
            
            c = c/X.shape[0]
            print('Iteration: ', iter)
            print("Cost: ", c)
            print("Accuracy:", (n_c/X.shape[0])*100)

print('1')
onehotencoder = OneHotEncoder(categories='auto', sparse=False)
categories = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
classes = onehotencoder.fit_transform(categories.reshape(-1,1))
classification = (classes, 10)

print('2')
train_data = np.load('datasets/nmist/train_60000_785.npy')
X_train, Y_train = train_data[:,0:784], train_data[:,784]

print('3')
test_data = np.load('datasets/nmist/test_10000_785.npy')
X_test, Y_test = test_data[:,0:784], test_data[:,784]

print('4')
#Defining the model architecture
architecture = np.array([784, 600, 400, 200, 1])

print('5')
#Creating the classifier
classifier = NeuralNetwork(architecture, classification)


print('6')
#Training the classifier
classifier.fit(X_train, Y_train, 10)


#Predicting the test set results:
n_c = 0
for i in range(0, X_test.shape[0]):
  x = X_test[i].reshape((X_test[i].size, 1))
  y = Y_test[i]
  
  y_pred = classifier.predict(x)
  yi = np.argmax(y_pred, 1)
  

  if (yi == y):
      n_c += 1

print("Test Accuracy", (n_c/X_test.shape[0])*100)
