'''
Usage:
python training_v1.py
'''
# Create Model 1 architecture and train weights on dataset 

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
import keras
import keras.backend as K
K.set_image_data_format('channels_last')

def create_model(num_groups,group_size,kernel_size,input_shape,connections_1f):
    ''' create_model Method
            CNN Model for Right Whale Upcall Recognition with filters learned through K-Means
            and energy-correlated receptive fields
            
            Args:
                num_groups: int number of groups to split the filters of the first and second
                            layers into 
                group_size: int size of the groups created in the first and second layers
                kernel_size: int size of the filters in the Conv2D layers (same kernel size
                             for all three layers)
                input_shape: (Number of Samples, Height, Width, Number of Filters)
                connections_1f: 2-D matrix of connections created by "learn_connections_unsup"
                                comprised of "num_groups" rows and "group_size" columns, for
                                the first layer
            
            Returns: 
                Compiled Keras CNN Model
                
    '''
    # There are three layers: Layer 0 is the convolution of the raw input layer with the
    # first set of learned filters (each filter is of depth 1 because the raw input layer 
    # is of depth 1). Layer 1 corresponds to the second set of learned filters (each filter is
    # of depth "group_size" and the learned filters are applied to smaller groups of the
    # entire set, each group being of depth "group_size"). Layer 2 corresponds to the third set 
    # of learned filters (each filter is of depth "group_size" and the learned filters are also 
    # applied to smaller groups of the entire set, each group being of depth "group_size").
    X_input = Input(shape=input_shape,name='input')
    # Dropout on the visible layer (1 in 5 probability of dropout) 
    X = Dropout(0.2,name='dropout0')(X_input)
    # BatchNorm on axis=3 (on the axis corresponding to the number of filters)
    X = BatchNormalization(axis=3,name='bn0')(X)
    # Convolution of Filters with Input. Since the filters are determined through unsupervised
    # learning, they are not updated through backpropagation (i.e. trainable=False)
    X = Conv2D(filters=num_groups*group_size,kernel_size=kernel_size,use_bias=False,activation='relu',name='conv0',trainable=False)(X)
    # Maxpooling for translation invariance and to halve height and width dimensions 
    X_maps1 = MaxPooling2D(name='maxpool0')(X)
    # The filters learned via K-Means for the next layer are trained on groups of feature maps
    # instead of the entire set. Specifically, the set of feature maps from X_maps1
    # are broken up into "num_groups" groups of "group_size" filters, with the groups
    # determined via energy correlation. (The reduced dimensionality improves the performance 
    # of K-Means). These smaller groups of feature maps are then fed to K-Means to generate
    # new filters. Note, however, that it is not possible to apply the learned filters of 
    # reduced dimensionality immediately to the output of X_maps1 of full dimensionality.
    # In order to account for this discrepancy, the output of X_maps1 is split
    # into the same groups used for K-Means, and all the learned filters applied to each of
    # these smaller groups. For example, suppose X_maps1_masked originally has 64 feature maps, 
    # and these are split into 16 groups of 4 feature maps. After these groups are fed into 
    # K-Means and, say, 128 filters are generated, X_maps1_masked is split into the same 
    # 16 groups of 4 feature maps. Then, the 128 filters are applied to each of these 16 
    # groups (since each group is the same size as the groups originally fed to K-Means). The
    # results are then concatenated along the axis corresponding to the number of filters
    # (the last axis) to represent the output of the convolution. 
    # The dictionaries below are used to implement this grouping mechanism. Note that the 
    # connections_1f array contains all the groups of feature maps determined via 
    # energy-correlation. This array is used to slice the original set of feature maps into
    # the desired groups. 
    layers_1 = dict()
    # connections_1f is composed of "num_groups" rows and "group_size" columns
    # layers_1_lambda, layers_1_reshape, and layers_1_maxpool are dictionaries of 
    # sub-dictionaries, where each sub-dictionary corresponds to one of the "num_groups" groups 
    # in connections_1f
    layers_1_lambda = dict()
    layers_1_reshape = dict()
    layers_1_maxpool = dict()
    for ii in range(num_groups):
        # Instantiate the sub-dictionaries for each main dictionary, for the current group
        # under consideration
        layers_1[ii] = dict()
        layers_1_lambda[ii] = dict()
        layers_1_reshape[ii] = dict()
        layers_1_maxpool[ii] = dict()
        for jj in range(group_size):
            # For the current group under consideration (represented by ii) and the current
            # member of the group under consideration (represented by jj), use Keras'
            # Lambda layer to select that one member of the group (that one feature map) 
            # from the entire main group. Then use Keras' Reshape layer to reshape the 
            # dimensions of the output of the Lambda layer to:
            # (Number of samples, height, width, 1)
            layers_1_lambda[ii]['X_lambda_'+str(ii)+'_'+str(jj)] = Lambda(lambda X: X[:,:,:,connections_1f[ii,jj]],name='lambda1_'+str(ii)+'_'+str(jj))(X_maps1)
            layers_1_reshape[ii]['X_reshape_'+str(ii)+'_'+str(jj)] = Reshape((*K.int_shape(layers_1_lambda[ii]['X_lambda_'+str(ii)+'_'+str(jj)])[1:3],1),name='reshape1_'+str(ii)+'_'+str(jj))(layers_1_lambda[ii]['X_lambda_'+str(ii)+'_'+str(jj)])
        # After the for loop above, layers_1_reshape[ii] contains all the individual feature
        # maps comprising the current group under consideration. Concatenate all of them into
        # one group of "group_size" feature maps using keras.layers.concatenate
        layers_1_concat = [layer for layer in layers_1_reshape[ii].values()]
        layers_1[ii]['X_concat_'+str(ii)] = keras.layers.concatenate(layers_1_concat,name='concat1_'+str(ii))
        # Apply BatchNorm along axis=3 as before
        layers_1[ii]['X_batchnorm_'+str(ii)] = BatchNormalization(axis=3,name='bn1_'+str(ii))(layers_1[ii]['X_concat_'+str(ii)])
        # Apply all the filters learned via K-Means to the current group of feature maps
        # under consideration. (Note that samples from all the groups were fed into K-Means to 
        # learn these filters ("group_size" filters were learned). These
        # "group_size" filters were applied to the current group under consideration.
        layers_1[ii]['X_conv2d_'+str(ii)] = Conv2D(filters=group_size*2,kernel_size=kernel_size,use_bias=False,activation='relu',name='conv1_'+str(ii),trainable=False)(layers_1[ii]['X_batchnorm_'+str(ii)])
        # Maxpooling was applied for translation invariance and to halve the width and height
        # dimensions
        layers_1_maxpool[ii]['X_maxpool2d_'+str(ii)] = MaxPooling2D(pool_size=(2,2),name='maxpool1_'+str(ii))(layers_1[ii]['X_conv2d_'+str(ii)])
    # Concatenate the results for all the groups into one set. This is done because 
    # connections_2f separates this entire set into energy-correlated groups (presumably 
    # different groups than specified by connections_1f).
    layers_1_final = []
    for ii in range(num_groups):
        # The sub-dictionaries in the layers_1_maxpool main dictionary contain all the
        # results to be concatenated
        layers_1_final.extend([layer for layer in layers_1_maxpool[ii].values()])
    X_maps2 = keras.layers.concatenate(layers_1_final,name='final1')
    # Flatten the output from the first, second, and third layers
    X_maps1_f = Flatten(name='flatten1')(X_maps1)
    X_maps2_f = Flatten(name='flatten2')(X_maps2)
    # Concatenate the flattened outputs into one feature vector
    X_maps = keras.layers.concatenate([X_maps1_f,X_maps2_f],name='final_concat')
    # Pass the full feature vector to the first fully connected layer 
    X = Dense(200,activation='relu',name='dense1')(X_maps)
    # Dropout on the fully connected layer (1 in 2 probability of dropout) 
    X = Dropout(0.5,name='dropout3')(X)
    # Pass to the second fully connected layer for binary classification
    X_output = Dense(1,activation='sigmoid',name='dense2')(X)
    # Use Adam optimizer
    opt = Adam(lr=0.0001,beta_1=0.9,beta_2=0.999,decay=0.01)
    model = Model(inputs=X_input,outputs=X_output)
    model.compile(optimizer=opt,loss='binary_crossentropy',metrics=['accuracy'])
    return model

# Use the data() method from whale_cnn.py to generate the training and test datasets 
# and labels
X_train, Y_train, X_testV, X_testH, Y_test = data()

# Parameters for Training Model:
# Size of Filters (kernel_size x kernel_size) in Keras model for all convolutional layers
kernel_size= 7
# Number of groups of filters 
num_groups_f = 32
# Number of feature maps in each group
group_size_f = 8
# Input Shape: (Number of Samples, Height, Width, Number of Filters)
input_shape = (X_train.shape[1],X_train.shape[2],X_train.shape[3])
# Instantiate array for connection matrix as the expected size so that the Keras
# model can be instantiated with the proper architecture. These values will be changed
# once the connections are learned, and the model updated with these learned values
connections_1f = numpy.arange(num_groups_f*group_size_f).reshape(num_groups_f,group_size_f)
connections_1f = numpy.asarray(connections_1f,numpy.int32)
# Instantiate the Keras model with the proper architecture
model = create_model(num_groups_f,group_size_f,kernel_size,input_shape,connections_1f)
# "k_train" is the set of samples fed to K-Means to learn filters unsupervised
k_train = X_train.transpose((0,3,1,2))
# The "separate_trainlabels" function shuffles the training set (and training labels),  
# and yields the indices to replicate the shuffling operation. The function also generates
# "Y_pos" and "Y_neg," arrays containing indices of the samples in the training set belonging
# to the positive and negative class, respectively. ("Y_pos" and "Y_neg" are each set to 
# contain 100 samples). Equal number of samples from each of these arrays are therefore given 
# to K-Means, so that K-Means is able to learn its dictionary of filters based off features 
# from both the positive and negative class. 
Y_pos,Y_neg,indices = separate_trainlabels(Y_train,num_perclass=100)
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
# 1/3 of the total number of patches randomly from each sample for training, learn
# num_groups_f*group_size_f filters (centroids), and do not use recursive autoconvolution 
centroids_0 = MiniBatchKMeansAutoConv(k_train,(kernel_size,kernel_size),0.33,num_groups_f*group_size_f,[0])
# The learned filters (centroids) will be set as the weights of the "conv0" layer.
layer_toset = model.get_layer('conv0')
# Reshape the learned filters so that they are the same shape as expected by the Keras layer
filters = centroids_0.transpose((2,3,1,0))
filters = filters[numpy.newaxis,...]
layer_toset.set_weights(filters)
# Construct an intermediate model (using model_fp) that generates the output of layer 
# "maxpool0." This is the output that the next set of learned filters will be convolved with.
# Therefore, generate the output for further processing (i.e. learning filter connections)
# For sake of efficiency, continue using only the 100 samples in Y_pos and 100 samples in Y_neg 
# for the processing below (i.e. forward pass only these 200 samples from the training set
# through the Keras model and obtain the output of layer "maxpool0" for only these samples).
fp_train = numpy.concatenate((model_train[Y_pos],model_train[Y_neg]),axis=0)
output_0 = model_fp(model,fp_train,'maxpool0')
# Transpose the output into shape: (Number of samples, number of filters, height, width)
# for "learn_connections_unsup" and "MiniBatchKMeansAutoConv"
output_0 = output_0.transpose((0,3,1,2))
# For sake of efficiency, use only the first 10 and last 10 examples from output_0 for
# learning filter connections via "learn_connections_unsup"
output_0_learnf = numpy.concatenate((output_0[0:10],output_0[-10:]),axis=0)
# Use the "learn_connections_unsup" function to find "num_groups_f" groups of feature maps 
# of size "group_size" that are most strongly correlated (using the "energy correlation" 
# pairwise similarity metric). In this case, to calculate the pairwise similaritiy metrics,
# the "samples" are the feature maps (depth dimension) and the "features" are the elements in 
# each feature map (height*width dimension) 
connections_1f = learn_connections_unsup(output_0_learnf,num_groups_f,group_size_f,flag=0)
# The feature maps fed into K-Means to learn the next set of filters are split into groups of 
# reduced dimensionality (to boost performance), with the groups determined by the results
# of connections_1f. Achieve this splitting by repeating output_0 "num_groups_f" times, each
# time using only the filters in one of the "num_groups_f" groups of connections_1f. 
# "output_0_connected" holds all these versions of output_0 with the different sets of 
# filters selected. K-Means will receive samples from each of these different versions of 
# output_0 to learn the next dictionary of filters 
output_0_connected = numpy.zeros((output_0.shape[0]*num_groups_f,group_size_f,output_0.shape[2],output_0.shape[3]))
index = 0
# For sake of efficiency, do not use all the samples in each version of output_0. Only use
# the first and last 10 samples in each version of output_0. "k_train_indices" keeps track
# of what all these indices are, so that the appropriate samples may be selected for feeding
# into K-Means
k_train_indices = []
for ii in range(num_groups_f):
    output_0_connected[index:index+output_0.shape[0],:,:,:] = output_0[:,connections_1f[ii,:],:,:]
    temp1 = numpy.arange(index,index+10)
    k_train_indices.extend(temp1)
    temp2 = numpy.arange((index+output_0.shape[0])-10,(index+output_0.shape[0]))
    k_train_indices.extend(temp2)
    index = index + output_0.shape[0]
k_train = output_0_connected[k_train_indices]
# Use the "MiniBatchKMeansAutoConv" function to learn the dictionary of filters. Extract
# 1/2 of the total number of patches randomly from each sample for training, learn
# group_size_f filters (centroids), and do not use recursive autoconvolution 
# Since group_size_f filters are learned, all the filters convolved with each of
# the "num_groups_f" groups in the Keras model, and the results concatenated, there will be
# a total of num_groups_f*group_size_f filters in the next layer.
centroids_1 = MiniBatchKMeansAutoConv(k_train,(kernel_size,kernel_size),0.5,group_size_f*2,[0])
# Re-instantiate the model with the newly-set connections_1f and connections_1m 
model = create_model(num_groups_f,group_size_f,kernel_size,input_shape,connections_1f)
# Since the model is re-instantiated, re-set the filters for "conv_0" 
layer_toset = model.get_layer('conv0')
filters = centroids_0.transpose((2,3,1,0))
filters = filters[numpy.newaxis,...]
layer_toset.set_weights(filters)
# Since the the new set of learned filters is convolved with each of the smaller groups of
# feature maps of reduced dimensionality, each learned filter is the same shape (height, width,
# depth) as each of these smaller groups. Therefore, there is one "conv" layer for each of 
# these smaller groups in the Keras model that will be set to the newly-learned set of filters.
# Set those filters using the for loop below 
for ii in range(num_groups_f):
    layer_toset = model.get_layer('conv1_'+str(ii))
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
