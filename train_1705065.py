import numpy as np
import math
import sys
from PIL import Image
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from layers import *
from time import *
from tqdm import tqdm
import pickle


class convolution: 
    def __init__(self, kernel_no, kernel_dim, padding, stride, lr=0.01):
        self.kernel_no = kernel_no
        self.kernel_dim = kernel_dim
        self.padding = padding
        self.stride = stride 
        self.lr = lr
        
        self.kernels = None
        self.bias = None
    
    def padding_by_0(self, input2d, padding):
        """
        input2d -> 2d matrix
        padding -> length of padding on all sides
        returns -> padded 2d matrix
        """
        input2d_height = input2d.shape[0]
        input2d_width = input2d.shape[1]
        new_width = 2 * padding + input2d_width
        new_height = 2 * padding + input2d_height
        output_2d = np.zeros((new_height, new_width))
        output_2d[padding:input2d_height+padding, padding:input2d_width+padding] = np.copy(input2d)
        return output_2d      
    
    def forward(self, input):
        """
        input -> array of 2d matrices (nc, nh, nw)
        to do-> xaviar initialization
        """
        input_channels = input.shape[0]
        input_height = input.shape[1]
        input_width = input.shape[2]
        
        padded_input_C = input_channels
        padded_input_H = input_height + 2*self.padding
        padded_input_W = input_width + 2*self.padding
        
        self.padded_inputs = np.zeros((padded_input_C,padded_input_H,padded_input_W))
        
        
        for i in range(input_channels):
            self.padded_inputs[i,:,:] = self.padding_by_0(input[i,:,:],self.padding)
        
        #create kernel randomly
        if self.kernels is None:
            self.kernels = np.random.randn(self.kernel_no, input_channels , self.kernel_dim, self.kernel_dim)
            # xavier
            for i in range(0,self.kernel_no):
                self.kernels[i,:,:,:] = np.random.normal(loc=0, scale=np.sqrt(1./(padded_input_C*self.kernel_dim*self.kernel_dim)), size=(padded_input_C, self.kernel_dim, self.kernel_dim))
            
        if self.bias is None:    
            self.bias = np.zeros((self.kernel_no,1))
        
        feature_channel = self.kernel_no
        feature_height = (padded_input_H - self.kernel_dim) // self.stride + 1
        feature_width = (padded_input_W - self.kernel_dim) // self.stride + 1
        
        feature_maps = np.zeros((feature_channel,feature_height,feature_width))
        
        for c in range(feature_channel):
            for h in range(0, feature_height):
                for w in range(0, feature_width):
                    input_height_start = h*self.stride
                    input_height_end = h*self.stride + self.kernel_dim
                    input_width_start = w*self.stride
                    input_width_end = w*self.stride + self.kernel_dim
                    feature_maps[c, h, w]=np.sum(self.padded_inputs[:, input_height_start:input_height_end, input_width_start:input_width_end]*self.kernels[c,:,:,:]) + self.bias[c]
                    
        return feature_maps 

    def backward(self,output):
        
        output = np.array(output)
        
        channel_in, row, col = output.shape
        
            
        d_input = np.zeros(self.padded_inputs.shape)
        dw = np.zeros(self.kernels.shape)
        
        # calculating db
        db = np.sum(output, axis=(1, 2)).reshape(self.bias.shape)

        # calculating dw
        dz_sparsed_row = self.stride*(row -1) + 1
        dz_sparsed_col = self.stride*(col -1) + 1
        dz_sparsed = np.zeros((self.kernel_no, dz_sparsed_row, dz_sparsed_col))
        dz_sparsed[:, ::self.stride, ::self.stride] = output

        
        channel_out, channel_in, row_, col_ = dw.shape
        
        input_strides = self.padded_inputs.strides
        input_strided = np.lib.stride_tricks.as_strided(self.padded_inputs, 
            shape=(channel_in, row_, col_, dz_sparsed_row, dz_sparsed_col),
            strides=(input_strides[0], input_strides[1], input_strides[2], input_strides[1], input_strides[2]))


        dot = np.tensordot(input_strided, dz_sparsed, axes=((3, 4), (1, 2)))
        dw = np.transpose(dot, (3, 0, 1, 2))
        

        # calculating d_input
        padding = self.kernel_dim-1
        dz_sparsed_paded = np.pad(dz_sparsed, ((0, 0), (padding, padding) , (padding, padding)))
        w_rotated = np.flip(np.flip(self.kernels, 2), 3)
        

        _, row_, col_ = dz_sparsed_paded.shape
        _, ch_in, w_rotated_row, w_rotated_col = w_rotated.shape


        dz_sp_pad_strides = dz_sparsed_paded.strides
        dz_sparsed_paded_strided = np.lib.stride_tricks.as_strided(dz_sparsed_paded, 
            shape=(self.kernel_no , row_ - w_rotated_row + 1, col_ - w_rotated_col + 1, w_rotated_row, w_rotated_col),
            strides=(dz_sp_pad_strides[0], dz_sp_pad_strides[1], dz_sp_pad_strides[2], dz_sp_pad_strides[1], dz_sp_pad_strides[2]))

        dot = np.tensordot(dz_sparsed_paded_strided, w_rotated, axes=((0, 3, 4), (0, 2, 3)))
        dot = np.transpose(dot, (2, 0, 1))

        d_input[:, :dot.shape[1], :dot.shape[2]] = dot
        d_input = d_input[:, self.padding:d_input.shape[1]-self.padding, self.padding:d_input.shape[2]-self.padding]

        # update
        self.kernels = self.kernels - self.lr * dw
        self.bias = self.bias - self.lr * db

        
        return d_input

class MaxPool:
    def __init__(self, dimension, stride):
        self.pool_dim = dimension
        self.stride = stride
    
    def forward(self,input):
        self.input = input 
        input_channels = input.shape[0]
        input_height = input.shape[1]
        input_width = input.shape[2]
        
        
        feature_channel = input_channels
        feature_height = (input_height - self.pool_dim) // self.stride + 1
        feature_width = (input_width - self.pool_dim) // self.stride + 1
        
        feature_maps = np.zeros((feature_channel,feature_height,feature_width))
        
        for c in range(feature_channel):
            for h in range(feature_height):
                for w in range(feature_width):
                    feature_maps[c,h,w] = input[c, h*self.stride:h*self.stride+self.pool_dim, w*self.stride:w*self.stride+self.pool_dim]
                    
         
        return feature_maps 

    def backward(self,output): 
         channel_in, row, col = output.shape
         
         input_gradient = np.zeros(self.input.shape)
         
         for c in range(channel_in):
            for h in range(row):
                for w in range(col):
                    idx = self.input[c, h*self.stride:h*self.stride+self.pool_dim, w*self.stride:w*self.stride+self.pool_dim]
                    max_idx = np.unravel_index(np.argmax(idx), idx.shape)
                    input_gradient[c, h*self.stride:h*self.stride+self.pool_dim, w*self.stride:w*self.stride+self.pool_dim][max_idx] = output[c, h, w]
        

    
class flatten:
    def __init__(self):
        pass
    
    def forward(self, input):
        self.input_channels = input.shape[0]
        self.input_height = input.shape[1]
        self.input_width = input.shape[2] 
        
        flattened = input.reshape(1, self.input_channels*self.input_height*self.input_width)
        
        return flattened
    
    def backward(self,output):
        input = output.reshape(self.input_channels, self.input_height, self.input_width)
        
        return input
    
    def extract(self):
        return
    
    
class ReLu:
    def __init__(self):
        pass
    
    def forward(self, input):
        self.input = input
        output = input.copy()
        output[output < 0] = 0
        return output
    
    def backward(self, output):
        d_input = self.input.copy()
        d_input[d_input < 0] = 0
        d_input[d_input > 0] = 1
        ret = d_input * output
        return ret
    
    def extract(self):
        return
    
class Softmax:
    def ___init__(self):
        pass
    
    def forward(self,input):
        # nan??
        exp = np.exp(input, dtype=np.float)
        self.output = exp/np.sum(exp)
        return self.output
        
        # f = np.exp(input - np.max(input))  # shift values
        # self.output = f / np.sum(f,)
        # return self.output          # predicted y
    
    
    def backward(self, out):
       return out
   
    def extract(self):
        return
   

class FullyConnected:
    def __init__(self,output_no, learning_rate, name):
        self.output_no = output_no
        self.lr = learning_rate
        self.weights = None
        self.bias = None
        self.name = name
        
    # def set(self, weights):
    #     self.weights = weights
        
    def forward(self,input):
        #input is a row vector
        self.input = input
        self.input_no = self.input.shape[1]
        
        if self.weights is None:
            self.weights = np.random.rand(self.input_no, self.output_no)
            self.weights = np.random.normal(loc=0, scale=np.sqrt(1./(self.output_no*self.input_no)), size=(self.input_no, self.output_no))
        
        
        self.bias = np.zeros((1, self.output_no))
        
        # if w is not None:
        #     self.set(w)
        
        output = np.dot(self.input, self.weights) + self.bias
        
        return output
    
    def  backward(self, dy):
        dw = (self.input.T).dot(dy)
        db = np.sum(dy, axis=1, keepdims=True)
        dx = dy.dot(self.weights.T)
        
        self.weights -= self.lr * dw
        self.bias -= self.lr * db
        
        return dx
    
    def extract(self):
        return {self.name+'.weights':self.weights, self.name+'.bias':self.bias}
        


class network:
    #LENET arch
    def __init__(self,batch_size,epoch):
        self.layers = []
        self.batch_size = batch_size
        self.epoch = epoch
    
    def build_net(self, layer):
        self.layers.append(layer)
     
    def batch_normalize(self, X_batch):
        """
        Parameters:
            X_batch is an array of inputs
            Shape (batch size, 1,h,w)
    
        Returns:
            Normalized X_batch
        """
        return (X_batch - np.mean(X_batch)) / np.std(X_batch)
        
     
    def train(self, training_data, training_label):
        train_data_no =  training_data.shape[0]
        
        
        for i in tqdm(range(self.epoch), desc='Train in progress' ):
            training_loss = 0
            for batch_index in range(0, train_data_no, self.batch_size):
                if batch_index + self.batch_size < train_data_no:
                   data = training_data[ batch_index : batch_index + self.batch_size ]
                   label = training_label[ batch_index : batch_index + self.batch_size]
                else:
                   data = training_data[batch_index : train_data_no]
                   label = training_label[batch_index : train_data_no]
                
                batch_acc = 0
                
                for batch in range(data.shape[0]):
                    X = data[batch]
                    Y = label[batch]
                    
                    # Forward pass
                    output = np.copy(X)
                    output = self.batch_normalize(output)
                    for layer in self.layers:
                        output = layer.forward(output)
                    
                    training_loss += cross_entropy_loss(Y, output)
                    
                    # backward pass
                    dy = np.copy(Y)
                    dy = (output - dy) / self.batch_size
                    for l in range(len(self.layers)-1, -1, -1):
                        d_out = self.layers[l].backward(dy)
                        dy = d_out
                        
                
            training_loss /= train_data_no
            validation_loss, acc_score, f1_score, conf_mat = self.test(training_data,training_label)
            print("------------------------------------------------------------------------" + '\n')
            print("Epoch --- " + str(i) + " Training Loss --- " + str(training_loss) + '\n')
            print("Epoch --- " + str(i) + " validation Loss --- " + str(validation_loss) + '\n')
            print("Epoch --- " + str(i) + " Accuracy Score --- " + str(acc_score) + '\n') 
            print("Epoch --- " + str(i) + " f1 score --- " + str(f1_score) + '\n')
            print("Epoch --- " + str(i) + " Confusion Matrix ---\n" + str(conf_mat) + '\n')
            
            
    def test(self, test_data, test_label):
        # test lable is a array of row matrices
        pred_label = []
        validation_loss = 0
        
        for i in tqdm(range(test_data.shape[0]), desc='Test in progress', position = 0, leave = True):
            x = test_data[i]
            y = test_label[i]
            
            output = np.copy(x)
            output = self.batch_normalize(output)
            for layer in self.layers:
                output = layer.forward(output)
            pred_label.append(output)
            validation_loss += cross_entropy_loss(y, output)                    
            
        predicted_digits = []
        for row_matrix in pred_label:
            digit = np.argmax(row_matrix[0,:])
            predicted_digits.append(digit)
            
        true_digits = []
        for row_matrix in test_label:
            digit = np.argmax(row_matrix[0,:])
            true_digits.append(digit)
        
        validation_loss /= test_data.shape[0]
        
        conf_mat = confusion_matrix(true_digits,predicted_digits)
        return validation_loss, accuracy_score(true_digits, predicted_digits), f1_score(true_digits, predicted_digits, average='macro'), conf_mat
    
    def predict(self, image):
        pred_label = np.copy(image)
        pred_label = self.batch_normalize(pred_label)
        for layer in self.layers:
            pred_label = layer.forward(pred_label)
        
        prediction = pred_label
        pred_digit = np.argmax(prediction[0,:])
        
        return pred_digit
            
         

def get_numpy_array(image_name):
    image = Image.open(image_name)
    #converting to greyscale
    image = image.convert('L')
    #LENET 
    image = image.resize((32,32))
    num_arr = np.asarray(image)
    return num_arr


def one_hot_encoding(digit):
    if digit < 0 or digit > 9:
       raise ValueError("Input must be in the range of 0-9")
    one_hot = np.zeros((1,10))
    one_hot[:,digit] = 1
    return one_hot


def data_process(filename):
    """
    filename: name of the csv file
    returns X,Y representing the training X and the training Y values
    X.shape = (number_of_images, 32,32)
    Y.shape = (number_of_images, 1,10)
    """
    df = pd.read_csv(filename)
    folder_name = df['database name'][0]
    X = df['filename'].tolist()
    X = ["./Dataset/"+folder_name+"/" + name for name in X]
    X = np.array(list(map(get_numpy_array, X)))
    X = np.reshape(X, (X.shape[0],1,X.shape[1],X.shape[2]))
    
    Y = df['digit'].tolist()
    Y = np.array(list(map(one_hot_encoding,Y)))
    return X,Y

def cross_entropy_loss(y_true,y_pred):
    true_class = np.argmax(y_true, axis=1)
    loss = -np.log(y_pred[0,true_class])
    return np.sum(loss)
    


X,Y = data_process('./Dataset/training-b.csv')  
print(X.shape)

net = network(128,10)

net.build_net(convolution(6,5,0,1))
net.build_net(ReLu())

net.build_net(flatten())

net.build_net(FullyConnected(84, 0.01, name='fc2'))
net.build_net(ReLu())

net.build_net(FullyConnected(10, 0.01, name= 'fc3'))
net.build_net(ReLu())

net.build_net(Softmax())

net.train(X, Y) 

model_file = open('1705065_model.pickle', 'wb')
pickle.dump(net,model_file) 
model_file.close()  