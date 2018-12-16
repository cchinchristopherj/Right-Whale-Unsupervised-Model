'''
Usage:
python training_v2.py
'''
# Create Model 2 architecture and train weights on dataset 

import whale_cnn
import whale_cnn_unsup
import numpy 
from skimage.transform import resize
from sklearn.metrics import roc_auc_score
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction import image
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, normalize
from sklearn.metrics.pairwise import pairwise_distances
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping
from keras import layers
from keras.layers import Input, Lambda, Dense, Reshape, Dropout, Activation, BatchNormalization, Flatten, Conv2D, MaxPooling2D, Concatenate
from keras.models import load_model, Model
from keras.optimizers import Adam
import tensorflow as tf
from matplotlib.pyplot import imshow
import keras
import keras.backend as K
K.set_image_data_format('channels_last')

def create_model(num_filters_0,num_filters_1,reduce_size,kernel_size,input_shape):
    ''' create_model Method
            CNN Model for Right Whale Upcall Recognition with filters learned through K-Means
            
            Args:
                num_filters_0: int number of filters for convolutional layer in Layer 0
                num_filters_1: int number of filters for convolutional layer in Layer 1
                reduce_size: int number of feature maps for the 1x1 convolutional layer 
                             between Layer 0 and Layer 1
                kernel_size: int size of the filters in the Conv2D layers (same kernel size
                             for all convolutional layers)
                input_shape: (Number of Samples, Height, Width, Number of Filters)
            Returns: 
                Compiled Keras CNN Model
                
    '''
    # There are two layers: Layer 0 is the convolution of the raw input layer with the
    # first set of learned filters (each filter is of depth 1 because the raw input layer 
    # is of depth 1). After the output of the convolutional layer is maxpooled, the number of
    # feature maps is reduced via 1x1 convolutions. The number of feature maps becomes 
    # "reduce_size." Layer 1 corresponds to the second set of learned filters that is 
    # convolved with the output of the 1x1 convolutional layer. The output of the 
    # convolutional layer is then maxpooled, flattened, concatenated with the flattened 
    # output of layer 0, then fed into two fully connected layers. 
    # Input Shape: (Number of Samples, Height, Width, Number of Filters)
    X_input = Input(shape=input_shape,name='input')
    # Dropout on the visible layer (1 in 10 probability of dropout) 
    X = Dropout(0.1,name='dropout0')(X_input)
    # BatchNorm on axis=3 (on the axis corresponding to the number of filters)
    X = BatchNormalization(axis=3,name='bn0')(X_input)
    # Layer 0 Convolutional layer. Since the filters are determined through unsupervised
    # learning, they are not updated through backpropagation (i.e. trainable=False)
    X = Conv2D(filters=num_filters_0,kernel_size=kernel_size,use_bias=False,activation='relu',name='conv0',trainable=False)(X)
    # Maxpooling for translation invariance and to halve height and width dimensions 
    X_maps1 = MaxPooling2D(name='maxpool0')(X)
    # 1x1 convolutional layer to reduce number of feature maps to "reduce_size"
    X = Conv2D(filters=reduce_size,kernel_size=1,name='reduce1')(X_maps1)
    # Layer 1 Convolutional Layer.
    X = Conv2D(filters=num_filters_1,kernel_size=kernel_size,use_bias=False,activation='relu',name='conv1',trainable=False)(X)
    # Maxpooling 
    X_maps2 = MaxPooling2D(name='maxpool1')(X)
    # Flatten the output of the maxpooling layer in Layer 0
    X_maps1_f = Flatten(name='flatten1')(X_maps1)
    # Flatten the output of the maxpooling layer in Layer 1
    X_maps2_f = Flatten(name='flatten2')(X_maps2)
    # Concatenate the flattened outputs into one feature vector
    X_maps = keras.layers.concatenate([X_maps1_f,X_maps2_f],name='final3')
    # Feed the concatenated output to two fully connected layers
    X = Dense(200,activation='relu',name='fc1')(X_maps2_f)
    # Dropout on the fully connected layer (1 in 2 probability of dropout) 
    X = Dropout(0.5,name='dropout3')(X)
    X_output = Dense(1,activation='sigmoid',name='fc_final')(X)
    # Use Adam optimizer
    opt = Adam(lr=0.0001,beta_1=0.9,beta_2=0.999,decay=0.01)
    model = Model(inputs=X_input,outputs=X_output)
    # Compile the model for binary classification
    model.compile(optimizer=opt,loss='binary_crossentropy',metrics=['accuracy'])
    return model

# Use the data() method from whale_cnn.py to generate the training and test datasets 
# and labels
X_train, Y_train, X_testV, X_testH, Y_test = data()

# Parameters for Training Model:
# Size of Filters (kernel_size x kernel_size) in Keras model for all convolutional layers
kernel_size = 7
# Number of filters for Layer 0
num_filters_0 = 256
# The output of maxpooling in Layer 0 will have 256 feature maps. Since K-Means performs
# most effectively when the dimensionality of the samples fed into it is kept relatively low
# (100 or 200 features per sample), reduce the number of feature maps in the output via 
# 1x1 convolutions. Reduce the number of feature maps from num_filters_0 to reduce_size
reduce_size = 8
# Number of filters for Layer 1
num_filters_1 = 512
# Input Shape: (Number of Samples, Height, Width, Number of Filters)
input_shape = (X_train.shape[1],X_train.shape[2],X_train.shape[3])
# Instantiate the Keras model 
model = create_model(num_filters_0,num_filters_1,reduce_size,kernel_size,input_shape)
# "k_train" is the set of samples fed to K-Means to learn filters unsupervised
k_train = X_train.transpose((0,3,1,2))
# The "separate_trainlabels" function shuffles the training set (and training labels), and 
# yields the indices to replicate the shuffling operation. The function also generates
# "Y_pos" and "Y_neg," arrays containing indices of the samples in the training set belonging
# to the positive and negative class, respectively. ("Y_pos" and "Y_neg" are each set to 
# contain 450 samples). Equal number of samples from each of these arrays are therefore given 
# to K-Means, so that K-Means is able to learn its dictionary of filters based off features 
# from both the positive and negative class. 
Y_pos,Y_neg,indices = separate_trainlabels(Y_train,num_perclass=450)
# Shuffle k_train, model_train, and model_labels according to the indices specified by 
# "separate_trainlabels"
k_train = k_train[indices]
model_train = X_train[indices]
model_labels = Y_train[indices]
# Feed 50 examples from the positive class and 50 examples from the negative class to 
# K-Means to learn a dictionary of filters. Identify those examples in k_train using the
# indices in the Y_pos and Y_neg arrays 
k_train = numpy.concatenate((k_train[Y_pos[0:50]],k_train[Y_neg[0:50]]),axis=0)
# Use the "MiniBatchKMeansAutoConv" function to learn the dictionary of filters. Extract
# 1/3 of the total number of patches randomly from each example for training, learn
# num_filters_0 filters (centroids), and do not use recursive autoconvolution
centroids_0 = MiniBatchKMeansAutoConv(k_train,(kernel_size,kernel_size),0.33,num_filters_0,[0])
# The learned filters (centroids) will be set as the weights of the "conv0" layer.
layer_toset = model.get_layer('conv0')
# Reshape the learned filters so that they are the same shape as expected by the Keras layer
filters = centroids_0.transpose((2,3,1,0))
filters = filters[numpy.newaxis,...]
layer_toset.set_weights(filters)
# Construct an intermediate model (using model_fp) that generates the output of layer 
# "reduce1." This is the output that the next set of learned filters will be convolved with.
# Therefore, generate the output so that random patches from each example in the output can
# be fed to K-Means to learn a new dictionary of filters.
# For sake of efficiency, continue using only the 450 samples in Y_pos and 450 samples in Y_neg 
# for the processing below (i.e. forward pass only these 900 samples from the training set
# through the Keras model and obtain the output of layer "reduce1" for only these samples).
fp_train = numpy.concatenate((model_train[Y_pos],model_train[Y_neg]),axis=0)
output_0 = model_fp(model,fp_train,'reduce1')
# Transpose the output into shape: (Number of samples, number of filters, height, width)
# for "MiniBatchKMeansAutoConv"
output_0 = output_0.transpose((0,3,1,2))
# Feed all of output_0 to K-Means to learn a new dictionary of filters.
k_train = output_0
# Use the "MiniBatchKMeansAutoConv" function to learn the dictionary of filters. Extract
# 1/2 of the total number of patches randomly from each sample for training, learn
# num_filters_1 filters (centroids), and do not use recursive autoconvolution.
centroids_1 = MiniBatchKMeansAutoConv(k_train,(kernel_size,kernel_size),0.5,num_filters_1,[0])
# The learned filters (centroids) will be set as the weights of the "conv1" layer.
layer_toset = model.get_layer('conv1')
# Reshape the learned filters so that they are the same shape as expected by the Keras layer
filters = centroids_1.transpose((2,3,1,0))
filters = filters[numpy.newaxis,...]
layer_toset.set_weights(filters)

# Now that the model is correctly instantiated with all learned filters, initiate training
# Use the ModelCheckpoint callback to save model weights only when there is an improvement in 
# "roc_auc_val" score. Save the weights to "model_filepath."
model_filepath = 'weights.best.hdf5'
checkpoint = ModelCheckpoint(model_filepath,monitor='roc_auc_val',verbose=1,save_best_only=True,mode='max')
# Stop training early if the "roc_auc_val" score does not improve for 5 epochs using the 
# EarlyStopping callback
early_stopping = EarlyStopping(monitor='roc_auc_val',patience=5,mode='max')
# Use the roc_callback class for training. Every epoch, calculate the "roc_auc_val" 
# score on the "X_testV" dataset using "Y_test" as labels. This score is representative 
# of the "roc_auc_val" score on the test set (actual test "roc_auc_val" score 
# calculated after fitting)
validation_train = X_testV
validation_labels = Y_test
callbacks_list = [roc_callback(validation_data=(validation_train,validation_labels)),checkpoint,early_stopping]
# Fit the model using the callbacks list for 500 epochs and a batch size of 100
model.fit(model_train,model_labels,callbacks=callbacks_list,epochs=500,batch_size=100)
# To calculate the test "roc_auc_val" score: 
# Generate predicted labels for the vertically-enhanced test feature matrix and
# predicted labels for the horizontally-enhanced test feature matrix. 
# The final predicted label is the union of these two predicted labels. 
# For example, if the vertically-enhanced image predicts label 0, but the 
# horizontally-enhanced version of the same image predicts label 1, the label is
# determined to be label 1. If both sets predict label 0, the label is 0. If both sets
# predict 1, the label is 1. The union operation is implemented by adding both
# predicted label vectors and setting the maximum value to 1
Y_predV = model.predict(X_testV)
Y_predH = model.predict(X_testH)
Y_pred = Y_predV + Y_predH 
Y_pred[Y_pred>1] = 1
score = roc_auc_score(Y_test,Y_predV)
print('Test ROC_AUC Score = ' + str(score))
