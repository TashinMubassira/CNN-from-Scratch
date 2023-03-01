from layers import *
from data_load import *
import sys
import os
import pickle

model_file = open('1705065_model.pickle', 'rb')
net = pickle.load(model_file) 
model_file.close() 

# X,Y = data_process('./Dataset/training-b.csv')  
# print(X.shape)

# net = network(128,5)

# net.build_net(convolution(6,5,0,1))
# net.build_net(ReLu())
# net.build_net(flatten())

# net.build_net(FullyConnected(84, 0.01, name='fc2'))
# net.build_net(ReLu())

# net.build_net(FullyConnected(10, 0.01, name= 'fc3'))
# net.build_net(ReLu())

# net.build_net(Softmax())

# net.train(X, Y)

test_folder_name = './Dataset/training-b/'

pred_csv = '1705065_prediction.csv'
pred_csv_file = None

#delete previous existing pred_csv
# if os.path.exists(pred_csv):
#     os.remove(pred_csv)
    
# pred_csv_file = open(pred_csv, 'a')

file_content = 'Filename,Digit\n'
# pred_csv_file.write('Filename,Digit')

for file_entry in os.scandir(test_folder_name):
    if file_entry.is_file():
        X_numpy = get_numpy_array(test_folder_name + file_entry.name )
        X = np.reshape(X_numpy, (1,X_numpy.shape[0],X_numpy.shape[1]))
        prediction = net.predict(X)
        file_content += file_entry.name + ',' + str(prediction) + '\n'
        # pred_csv_file.write( file_entry.name + ',' + str(prediction))

pred_csv_file = open(pred_csv, 'w')
pred_csv_file.write(file_content)
pred_csv_file.close()        
