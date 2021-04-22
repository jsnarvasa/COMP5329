#Import relevant libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#Read data
train_data = np.load("./Assignment1-Dataset/train_data.npy")
train_label = np.load("./Assignment1-Dataset/train_label.npy")
test_data = np.load("./Assignment1-Dataset/test_data.npy")
test_label = np.load("./Assignment1-Dataset/test_label.npy")

'''
# Generate dataset
class_1 = np.hstack([np.random.normal( 1, 1, size=(25, 2)),  np.ones(shape=(25, 1))])
class_2 = np.hstack([np.random.normal(-1, 1, size=(25, 2)), -np.ones(shape=(25, 1))])
dataset = np.vstack([class_1, class_2])

train_data = dataset[:,0:2]
train_label = dataset[:,2]
'''

#Check shape
print(train_data.shape)
print(train_label.shape)
#print(test_data.shape)
#print(test_label.shape)

def normalize(X, new_min=0, new_max=1):
  #Min-Max Normalization- (x-xmin)/(xmax-xmin)
  norm_data = (X - np.min(X))/(np.max(X) - np.min(X))
  X = norm_data
  return X
train_data=normalize(train_data)

#Create Activation Class

class Activation(object):
    
    def __tanh(self, x):
        return np.tanh(x)
    
    def __tanh_deriv(self, a):
        #where a = np.tanh(x)
        return 1.0 - a**2
        
    def __logistic(self, x):
        return 1.0 /(1.0 + np.exp(-x))
    
    def __logistic_deriv(self, a):
        #where a = logistic(x)
        return a * (1-a)
        
    def __relu(self,x):
        return np.maximum(0,x)
  
    def __relu_deriv(self,a):
        return np.heaviside(a, 0)

    def __softmax(self, z):
      # return np.exp(z) / np.sum(np.exp(z))
        z = z - np.max(z) # we're adding this such that it doesn't overflow with very large nums - numerical stability
        return np.divide(np.exp(z), np.sum(np.exp(z)))

    def __softmax_deriv(self, a, b):
        #return a * (1 - a)
        for i in range(len(a)):
            for j in range(len(b)):
                if i == j:
                    deriv[i] = a[i] * (1-a[i])
                else:
                    deriv[i] = -a[i]*a[j]
        return deriv


        
    #Initialise & set the default activation functions
    def __init__(self, activation = 'relu'):
        if activation == "logistic":
            self.f = self.__logistic
            self.f_deriv = self.__logistic_deriv
        elif activation == "tanh":
            self.f = self.__tanh
            self.f_deriv = self.__tanh_deriv    
        elif activation == 'relu': 
            self.f = self.__relu
            self.f_deriv= self.__relu_deriv
        elif activation == "softmax":
            self.f = self.__softmax
            self.f_deriv = self.__softmax_deriv

drop_prob=0.5
def dropout(X, p_dropout):
    u = np.random.binomial(1, p_dropout, size=X.shape)# / p_dropout
    out = X * u
    binomial_array=u
    return out, binomial_array
def dropout_backward(delta, binomial_array, layer_num):
    delta*=nn.layers[layer_num - 1].binomial_array
    return delta

#Create Hidden Layer Class

class HiddenLayer(object):
    def __init__(self, 
                 n_in, 
                 n_out, 
                 activation_last_layer = 'relu',
                 activation = 'relu',
                 W = None,
                 b = None,
                 v_W = None,
                 v_b = None,
                 last_hidden_layer = False):
    
        self.last_hidden_layer = last_hidden_layer
        self.input = None
        self.activation = Activation(activation).f
    
        #Derivative of last layer activation
        #Note methods/links to Activation class
    
        self.activation_deriv = None
        if activation_last_layer:
            self.activation_deriv = Activation(activation_last_layer).f_deriv

        #Initialisation - assign random small values (from uniform dist)
        
        self.W = np.random.uniform(low = -np.sqrt(6. / (n_in + n_out)),
                                   high = np.sqrt(6. / (n_in + n_out )),
                                   size = (n_in, n_out))
        
        self.b = np.zeros(n_out,)
        
        if activation == 'logistic':
           self.W *= 4 #*= being similar to +=

        self.grad_W = np.zeros(self.W.shape)
        self.grad_b = np.zeros(self.b.shape)

        #Initialise velocities for Momentum SGD
        self.v_W = np.zeros_like(self.grad_W)
        self.v_b = np.zeros_like(self.grad_b)

        
          #size of bias = size of output
        
    
          #size of weight gradient = size of weight
        
        self.binomial_array=np.zeros(n_out)
    #forward progress for training epoch:
    def forward(self, input):
        lin_output = np.dot(input, self.W) + self.b #simple perceptron output
        self.output = (
            lin_output if self.activation is None #linear if no activation specified
            else self.activation(lin_output) #activation fn on w*I + b  (i.e. activation function on linear output)
        ) 

        if not self.last_hidden_layer:
            self.output, self.binomial_array = dropout(self.output, drop_prob)

        self.input = input
        return self.output

    #backpropagation
    def backward(self, delta, layer_num, output_layer = False):
        self.grad_W = np.atleast_2d(self.input).T.dot(np.atleast_2d(delta))
        self.grad_b = delta
        if self.activation == 'softmax':
            delta = delta.dot(self.W.T) * self.activation_deriv(self.input, self.output)

        if self.activation_deriv:
            #print(delta.shape, self.W.T.shape,self.activation_deriv(self.input).shape)
            delta = delta.dot(self.W.T) * self.activation_deriv(self.input)
        if layer_num != 0:
            delta=dropout_backward(delta, self.binomial_array, layer_num)
        return delta

class MLP:
    def __init__(self, layers=4, activation = [None, 'relu', 'relu','relu']):
        
        self.layers = []
        self.params = []
        
        self.activation = activation
        
        for i in range(len(layers)-1):

            last_hidden_layer = False

            if i == len(layers) - 2: # -2 because -1 for output layer, and another -1 since it's index 0
                last_hidden_layer = True

            self.layers.append(HiddenLayer(layers[i], 
                                           layers[i+1], 
                                           activation[i], 
                                           activation[i+1],
                                           last_hidden_layer=last_hidden_layer))

            
            #note HiddenLayer(n_in, n_out, activation_last_layer, activation) - type(layers) = int
        #Forward - pass info through layer, return result of final output
    def forward(self, input):
        for layer in self.layers: #iterate forward through layers
            output = layer.forward(input) #call forward method from HiddenLayer
           
            input = output #feedforward to next layer
        return output
         
    def criterion_MSE(self, y, y_hat):
        activation_deriv = Activation(self.activation[-1]).f_deriv #Derivate of last layer activation
        #MSE
        error = y - y_hat
        #Squared error
        loss = error**2
        loss= loss*0.98
        #Delta of output
        delta = -error * activation_deriv(y_hat)
        return loss, delta

    def CE_loss(self, y, y_hat):
      #print(y, np.sum(y_hat))
      softmax_y_hat = np.exp(y_hat) / np.sum(np.exp(y_hat))
      loss = -np.sum(y * np.log(y_hat)) 
      loss=loss*0.98
      delta = y - y_hat
      return loss, delta

    def backward(self, delta):
        delta = self.layers[-1].backward(delta, len(self.layers) -1, output_layer = True)
        for layer_num, layer in enumerate(reversed(self.layers[:-1])):
            delta = layer.backward(delta, layer_num)
                
    #Update weights after backward function - lr = learning rate    
    def update(self, lr, SGD_optim):
      if SGD_optim is None:
          for layer in self.layers:
                # Obtaining the averages
              grad_W_avg = np.average(layer.grad_W, axis=0)
              grad_b_avg = np.average(layer.grad_b, axis=0)
              layer.W -= lr * grad_W_avg
              layer.b -= lr * grad_b_avg

        # Need to make this compatible with batches
      elif SGD_optim['Type'] == 'Momentum':
          for layer in self.layers:
              layer.v_W = (SGD_optim['Parameter'] * layer.v_W) + (lr * layer.grad_W)
              layer.v_b = (SGD_optim['Parameter'] * layer.v_b) + (lr * layer.grad_b)
              layer.W = layer.W - layer.v_W
              layer.b = layer.b - layer.v_b

    #fit/training function - returns all losses within whole training process
    def fit(self, X, y, learning_rate = 0.1, epochs = 100, SGD_optim = None):
        #X = input data/features
        #y = input target
        #learning rate = param for speed of learning
        #epochs - # times dataset is presented to network for learning
            
        X = np.array(X)
        y = np.array(y)
        to_return = np.zeros(epochs)
            
        for k in range(epochs):
            if k % 5 == 0:
              print("training epoch", k)
            loss = np.zeros(X.shape[0]) #initialise as zeros with same length as input
            for it in range(X.shape[0]):
                i = np.random.randint(X.shape[0])
                #forward pass 
                y_hat = self.forward(X[i]) #note forward returns output of final layer
                #print(y_hat)

                #backward pass
                loss[it], delta = self.CE_loss(y[i], y_hat) #note criterion_MSE returns loss, delta
                #loss[it], delta = self.criterion_MSE(y[i], y_hat)
                self.backward(delta) 

                    
                #update
                self.update(learning_rate, SGD_optim)
            to_return[k] = np.mean(loss)

            print(to_return)
         
        return to_return
            
        #prediction function
    def predict(self, x):
        x = np.array(x)
        output = [i for i in range(x.shape[0])]
        for i in np.arange(x.shape[0]):
            output[i] = self.forward(x[i, :])
        output = np.array(output)
        return output

class Preprocessing:
    '''
    Class used to contain any pre-processing that may occur
    '''

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def normalize(self, new_min=0, new_max=1):     
        #Min-Max Normalization- (x-xmin)/(xmax-xmin)
        norm_data = (self.X - np.min(self.X))/(np.max(self.X) - np.min(self.X))
        self.X = norm_data

    def label_encode(self):
        '''
        Encode the label vector of our dataset, such that we can use it for the computation of the MSE.
        This is because the labels are labelled as integers 0 to 9, whereas for MSE to work, we need to
        create a array vector of size 10 for every single observation, where the value 1 is set on the index of the correct observation
        '''
        
        num_classes = np.unique(self.y).size
        
        encoded_label_vector = []
        
        for label in self.y:
            encoded_label = np.zeros(num_classes)
            encoded_label[int(label)-1] = 1
            encoded_label_vector.append(encoded_label)
        
        self.y = np.array(encoded_label_vector)


class Utils:
    '''
    Class used to contain miscellaneous methods used for Neural Network
    '''

    @staticmethod
    def shuffle(X, y):
        # create an index vector with the size of X
        shuffled_idx = np.arange(X.shape[0])
        np.random.shuffle(shuffled_idx)
        X = X[shuffled_idx]
        y = y[shuffled_idx]

        return X, y


train_df = Preprocessing(train_data, train_label)
train_df.normalize()
train_df.label_encode()

test_df = Preprocessing(test_data, test_label)
test_df.normalize()

print(np.min(train_df.X))
print(np.max(train_df.X))

nn = MLP([train_df.X.shape[1], 100, 10], [None, 'logistic', 'softmax'])
trial1 = nn.fit(train_df.X, train_df.y, learning_rate = 0.001, epochs = 2, SGD_optim = {'Type' : 'Momentum', 'Parameter': 0.9})
#trial1 = nn.fit(train_df.X, train_df.y, learning_rate = 0.001, epochs = 10, SGD_optim = None)
print(trial1)

predictions = nn.predict(test_df.X)

print(predictions)

        
unencoded_pred = []
for i in range(0, len(predictions)):
    unencoded_pred.append(np.array(predictions[i].argmax()))
unencoded_pred = np.array(unencoded_pred)

def model_accuracy(true_label, pred_label):
    accuracy = (np.sum(true_label == pred_label) / len(true_label)) *100
    return accuracy

print(model_accuracy(test_df.y, unencoded_pred))

