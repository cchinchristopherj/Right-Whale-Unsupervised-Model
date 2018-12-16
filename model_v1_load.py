'''
Usage:
python model_v1_load.py
'''
# Re-create Model 1 architecture and load in weights to appropriate layers

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
        layers_1[ii]['X_conv2d_'+str(ii)] = Conv2D(filters=group_size,kernel_size=kernel_size,use_bias=False,activation='relu',name='conv1_'+str(ii),trainable=False)(layers_1[ii]['X_batchnorm_'+str(ii)])
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

# Use the data() method from whale_cnn.py to load in the training and test datasets and 
# labels 
X_train,Y_train,X_testV,Y_test = data()
# Parameters for the model: 
kernel_size = 7
num_groups_f = 32
group_size_f = 8
input_shape = (X_train.shape[1],X_train.shape[2],X_train.shape[3])
# Load in the connections_1f matrix I used to train the model (must use this matrix 
# otherwise the model will not have the correct architecture for loading in the 
# weights). 
connections_1f = numpy.loadtxt('connections_1f.txt',delimiter=',')
connections_1f = connections_1f.astype(int)
# Re-create the model with the correct architecture 
model = create_model(num_groups_f,group_size_f,kernel_size,input_shape,connections_1f)
# Load in the weights I obtained from training to the appropriate layers in the model
model.load_weights('model_v1_weights.hdf5',by_name=True)

