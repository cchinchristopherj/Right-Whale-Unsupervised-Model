''' whale_cnn_unsup.py
        Contains classes and functions used by training_v1.py
	and training_v2.py
'''

import numpy 
import matplotlib.pyplot as plt
import glob
from skimage.transform import resize
import aifc
import os
from skimage.transform import resize
from sklearn.metrics import roc_auc_score
from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction import image
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, normalize
from sklearn.metrics.pairwise import pairwise_distances
from matplotlib import mlab
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping
from keras import layers
from keras.layers import Input, Lambda, Dense, Reshape, Dropout, Activation, BatchNormalization, Flatten, Conv2D, MaxPooling2D, Concatenate
from keras.models import load_model, Sequential, Model
from keras.optimizers import Adam
from keras.utils import plot_model
import tensorflow as tf
from matplotlib.pyplot import imshow
import keras
import keras.backend as K
K.set_image_data_format('channels_last')

def MiniBatchKMeansAutoConv(X, patch_size, max_patches, n_clusters, conv_orders, batch_size=20):
    ''' MiniBatchKMeansAutoConv Method 
            Extract patches from the input X, perform all specified orders of recursive
            autoconvolution to generate a richer set of patches, and pass the complete set
            of patches to MiniBatchKMeans to learn a dictionary of filters 
            
            Args:
                X: (Number of samples, Number of filters, Height, Width)
                patch_size: int size of patches to extract from X 
                max_patches: float decimal percentage of maximum number of patches to 
                             extract from X
                n_clusters: int number of centroids (filters) to learn via MiniBatchKMeans
                conv_orders: 1-D array of integers indicating which orders of recursive 
                             autoconvolution to perform, with the set of possible orders 
                             ranging from 0 to 3 inclusive
                batch_size: int size of batches to use in MiniBatchKMeans
            Returns: 
                learned centroids of shape: (Number of samples, Number of filters, Height, Width)
                
    '''
    sz = X.shape
    # Transpose to Shape: (Number of samples, Number of Filters, Height, Width) and extract
    # patches from each sample up to the maximum number of patches using sklearn's
    # PatchExtractor
    X = image.PatchExtractor(patch_size=patch_size,max_patches=max_patches).transform(X.transpose((0,2,3,1))) 
    # For later processing, ensure that X has 4 dimensions (add an additional last axis of
    # size 1 if there are fewer dimensions)
    if(len(X.shape)<=3):
        X = X[...,numpy.newaxis]
    # Local centering by subtracting the mean
    X = X-numpy.reshape(numpy.mean(X, axis=(1,2)),(-1,1,1,X.shape[-1])) 
    # Local scaling by dividing by the standard deviation 
    X = X/(numpy.reshape(numpy.std(X, axis=(1,2)),(-1,1,1,X.shape[-1])) + 1e-10) 
    # Transpose to Shape: (Number of samples, Number of Filters, Height, Width)
    X = X.transpose((0,3,1,2)) 
    # Number of batches determined by number of samples in X and batch size
    n_batches = int(numpy.ceil(len(X)/float(batch_size)))
    # Array to store patches modified by recursive autoconvolution
    autoconv_patches = []
    for batch in range(n_batches):
        # Obtain the samples corresponding to the current batch 
        X_order = X[numpy.arange(batch*batch_size, min(len(X)-1,(batch+1)*batch_size))]
        # conv_orders is an array containing the desired orders of recursive autoconvolution
        # (with 0 corresponding to no recursive autoconvolution and 3 corresponding to 
        # recursive autoconvolution of order 3) 
        for conv_order in conv_orders:
            if conv_order > 0:
                # Perform recursive autoconvolution using "autoconv2d"
                X_order = autoconv2d(X_order)
                # In order to perform recursive autoconvolution, the height and width 
                # dimensions of X were doubled. Therefore, after recursive autoconvolution is
                # performed, reduce the height and width dimensions of X by 2. 
                X_sampled = resize_batch(X_order, [int(numpy.round(s/2.)) for s in X_order.shape[2:]])
                if conv_order > 1:
                    X_order = X_sampled
                # Resize X_sampled to the expected shape for MiniBatchKMeans
                if X_sampled.shape[2] != X.shape[2]:
                    X_sampled = resize_batch(X_sampled, X.shape[2:])
            else:
                X_sampled = X_order
            # Append the output of each order of recursive autoconvolution to "autoconv_patches"
            # in order to provide MiniBatchKMeans with a richer set of patches 
            autoconv_patches.append(X_sampled)
        print('%d/%d ' % (batch,n_batches))
    X = numpy.concatenate(autoconv_patches) 
    # Reshape X into a 2-D array for input into MiniBatchKMeans
    X = numpy.asarray(X.reshape(X.shape[0],-1),dtype=numpy.float32)
    # Convert X into an intensity matrix with values ranging from 0 to 1
    X = mat2gray(X)
    # Use PCA to reduce dimensionality of X
    pca = PCA(whiten=True)
    X = pca.fit_transform(X)
    # Scale input sample vectors individually to unit norm (using sklearn's "normalize")
    X = normalize(X)
    # Use "MiniBatchKMeans" on the extracted patches to find a dictionary of n_clusters 
    # filters (centroids)
    km = MiniBatchKMeans(n_clusters = n_clusters,batch_size=batch_size,init_size=3*n_clusters).fit(X).cluster_centers_
    # Reshape centroids into shape: (Number of samples, Number of filters, Height, Width)
    return km.reshape(-1,sz[1],patch_size[0],patch_size[1])

def mat2gray(X):
    ''' mat2gray Method 
            Convert the input X into an intensity image with values ranging from 0 to 1
            
            Args:
                X: (Number of samples, Number of filters, Height, Width)         
            Returns: 
                X: 2-D intensity image matrix
                
    '''
    # Input Shape: (Number of samples, Number of features)
    # Find the minima of X along axis=1, i.e. the minimum value feature for each sample
    m = numpy.min(X,axis=1,keepdims=True)
    # Find the maxima of X along axis=1 and subtract the minima of X from it to obtain the
    # range of values of X for each sample
    X_range = numpy.max(X,axis=1,keepdims=True)-m
    # Set the values of X so that the maximum corresponds to 1, the minimum corresponds to 0,
    # and values larger than the maximum and smaller than the minimum are set to 1 and 0,
    # respectively
    # For samples where the maximum of the features is equal to the minimum of the features
    # set the feature values to 0
    idx = numpy.squeeze(X_range==0)
    X[idx,:] = 0
    # For samples other than those indexed by idx, subtract the minima of the feature values 
    # from the feature values and divide by X_range
    X[numpy.logical_not(idx),:] = (X[numpy.logical_not(idx),:]-m[numpy.logical_not(idx)])/X_range[numpy.logical_not(idx)]
    return X

def learn_connections_unsup(feature_maps,num_groups,group_size,flag):
    ''' learn_connections_unsup Method 
            Learn which groups of feature maps are most strongly correlated (filter 
            connections), or which feature map elements are most strongly correlated (mask
            connections), depending on the flag

            Args:
                feature_maps: (Number of samples, Number of filters, Height, Width)
                num_groups: int number of groups of feature maps (or feature map elements)
                            to learn
                group_size: int size of groups of feature maps (or feature map elements) to 
                            learn
                flag: int (either 0 or 1), with 0 indicating learning feature map (filter)
                      connections and 1 indicating learning feature map element (mask)
                      connections
            Returns: 
                connections: 2-D numpy array, with "num_groups" rows and "group_size" 
                             columns, where values in each row (each group) indicate 
                             feature maps (or feature map elements) that are strongly 
                             correlated
                
    '''
    # Number of samples
    num_samples = feature_maps.shape[0]
    # Number of filters
    num_filters = feature_maps.shape[1]
    # Height
    height = feature_maps.shape[2]
    # Width
    width = feature_maps.shape[3]
    if flag == 0: 
        # A flag of 0 indicates learning filter connections
        # In this function, which is an implementation of MATLAB's pdist2, the pairwise 
        # distance between samples (rows) is determined using the features (columns) of 
        # each sample. For learning filter connections, reshape the input into a 2-D array 
        # where the rows (samples) are the feature maps and the columns (features) are the 
        # feature map elements. Note that the pairwise distances are computed between
        # the feature maps of all the examples. (For this function, "example" means the 
        # patch under consideration and "sample" means the observation, either feature map
        # or feature map element, that is being compared with all the other observations in 
        # the patch). Feature map 1 of example 1, for example, is compared with all the other 
        # feature maps (samples) of example 1, in addition to the feature maps of all the other 
        # examples. This set-up is not ideal (ideal being comparing the feature maps within one 
        # example separately), but prevents a longer set of computations. 
        # The following line reshapes the feature maps so that the rows are the feature
        # maps and the columns are the feature map elements
        feature_maps = feature_maps.reshape((num_samples*num_filters,height*width))
    else: 
        # A flag of 1 indicates learning mask connections
        # The mask is multiplied by the designated layer in the Keras model (output of 
        # maxpooling) to zero out feature map elements that are not strongly correlated with 
        # other elements, the goal being to obtain more discriminative features. In this 
        # case, reshape the input into a 2-D array where the rows (samples) are the feature
        # map elements and the columns (features) are the feature maps. Once again, the 
        # pairwise distances are computed between feature map elements of all examples.
        # The following lines reshape the feature maps so that the rows are the feature
        # map elements and the columns are the. feature maps 
        feature_maps = feature_maps.reshape((num_samples,num_filters,height*width))
        feature_maps = feature_maps.transpose((0,2,1))
        feature_maps = feature_maps.reshape((num_samples*height*width,num_filters))
    # Ensure that the feature maps have values between 0 and 1
    min_max_scaler = MinMaxScaler()
    feature_maps = min_max_scaler.fit_transform(feature_maps)
    # PCA whiten the feature maps
    pca = PCA(.99,whiten=True)
    feature_maps = pca.fit_transform(feature_maps)
    # Square "feature maps" to obtain squared energy 
    feature_maps = numpy.square(feature_maps)
    # Calculate the pairwise distances between samples using sklearn's "pairwise_distances."
    # Use the "mat2gray" function to conver the result into an intensity image with 
    # values ranging from 0 to 1. The result is a square matrix where the number of rows and
    # columns is equal to the number of samples. Every ith row corresponds to a particular 
    # sample and every jth column corresponds to a particular sample. Each (i,j) element
    # therefore represents the correlation between the ith sample and jth sample. 
    S_all = mat2gray(pairwise_distances(feature_maps,feature_maps,metric='correlation'))
    # In the following section of code, the procedure is as follows: a random sample (a 
    # random row) is selected from the set of all samples. The values of all the elements
    # in the row are sorted and the "group_size" highest values are kept. A group is created 
    # using the indexes of these "group_size" highest values. These indexes indicate which 
    # samples in the set of all samples are most closely correlated with the random sample 
    # currently under consideration (the sample corresponding to the random row that was 
    # selected initially). Note, however, that the indexes will indicate which samples in 
    # which examples are most closely correlated with the sample under consideratio. (This is  
    # because the rows of the matrix are organized such that the samples of each example are 
    # concatenated one after another). It is not important to know information about which
    # example a sample came from, only what the sample is itself. The modulo operation is 
    # therefore used, knowing the number of samples in each example, to isolate the identities
    # of the samples and remove information about the particular examples they came from. More
    # random samples are selected until "num_groups" samples are chosen and this procedure 
    # above repeated for each one. The entire procedure involving "num_groups" samples 
    # represents one iteration 
    # Number of iterations to perform the procedure above (i.e. random selection)
    n_it = 100
    # Instantiate a dictionary to contain all the groups 
    connections_all = dict()
    # Instantiate an array "n_it" elements long called "S_total." For each iteration, the 
    # total sum of distances in the group of most strongly correlated features wil be 
    # computed and stored in the element of S_total corresponding to the current iteration.
    S_total = numpy.zeros(n_it)
    for it in range(n_it):
        S_all_it = S_all;
        # Randomly select "num_groups" samples 
        rand_feats = numpy.random.randint(0,S_all_it.shape[0],size=num_groups)
        # Instantiate a temnporary array to hold the groups created in this iteration
        # The array will have "num_groups" rows and "group_size" columns
        connections_tmp = numpy.zeros((num_groups,group_size))
        for ii in range(num_groups):
            while(True):
                S_tmp = S_all_it
                # Prevent selection of the same sample by setting the (i,i) element to 0.
                # For example, suppose there are 10 samples and one example. Sample 3 is 
                # randomly selected. There will be 10 rows and 10 columns after the 
                # "pairwise_distances" function is executed. The (3,3) element value will
                # most likely be high, since a sample would most likely be strongly correlated
                # with itself. Set the (3,3) element to 0 to prevent the same sample from 
                # being selected.
                S_tmp[rand_feats[ii],rand_feats[ii]] = float('inf')
                # Sort the row under consideration to find the most strongly correlated 
                # samples 
                ids = numpy.argsort(S_tmp[rand_feats[ii],:])
                S_min = S_tmp[rand_feats[ii],:][ids]
                # Find the "group_size" most strongly correlated samples 
                closest = S_min[0:group_size-1]
                # The modulo operation depends on the number of samples per example
                if flag == 0:
                    # For learning filter connections, there are "num_filters" samples
                    # per example, so use the modulo operation with "num_filters" to 
                    # identify the identities of the samples 
                    group = numpy.sort(numpy.append([rand_feats[ii]%num_filters],[ids[0:group_size-1]%num_filters]))
                else:
                    # For learning mask connections, there are (height*width) samples per 
                    # example, so use the modulo operation with (height*width) to identify 
                    # the identities of the samples 
                    group = numpy.sort(numpy.append([rand_feats[ii]%(height*width)],[ids[0:group_size-1]%(height*width)]))
                # Only break the while loop if the number of unique elements is equal to 
                # the length of the group, i.e. if all the elements of the group are unique.
                # This encourages selection of groups that connect more samples 
                if(len(numpy.unique(group)) == len(group)):
                    connections_tmp[ii,:] = group
                    break
                # If not all the members of the group are unique, select a new random
                # sample and perform the procedure again
                rand_feats[ii] = numpy.random.randint(0,S_all_it.shape[0],size=1)
            S_all_it = S_tmp
            # Update the element of S_total corresponding to the current iteration with the
            # total sum of distances in the new group 
            S_total[it] = S_total[it]+numpy.sum(closest)
        connections_tmp = numpy.asarray(connections_tmp,dtype=numpy.int32)
        # Store the matrix of "num_groups" groups of size "group_size" for the current 
        # iteration in the corresponding entry of the connections_all matrix
        connections_all[it] = connections_tmp
    # Sort S_total to determine which iterations contained groups with the lowest sum of
    # pairwise distances 
    ids = numpy.argsort(S_total)
    # Instantiate an array that will hold the number of unique samples in a particular 
    # iteration. For sake of efficiency, only examine n_it/10 iterations, not all n_it.
    n_unique_feats = numpy.zeros(round(n_it/10))
    for it in range(round(n_it/10)):
        # Select n_it/10 of the elements in sorted S_total, i.e. the n_it/10 iterations 
        # with the lowest sum of pairwise distances. For each of these elements, find the 
        # number of unique samples in the groups of the iteration under consideration from 
        # connections_all. 
        n_unique_feats[it] = len(numpy.unique(connections_all[ids[it]].flatten()))
    # Locate the member of n_unique_feats corresponding to the greatest number of unique
    # samples. 
    num_maxind = len(numpy.where(n_unique_feats == numpy.amax(n_unique_feats))[0])
    # If there is only one iteration where the value of n_unique_feats is equal to the 
    # greatest number of unique samples, select that iteration. 
    if num_maxind == 1:
        it = ids[numpy.squeeze(numpy.where(n_unique_feats == numpy.amax(n_unique_feats)))]
    # If there are multiple iterations where the value of n_unique_feats is equal to the 
    # greatest number of unique samples, select the first of these iterations.
    else: 
        it = ids[numpy.squeeze(numpy.where(n_unique_feats == numpy.amax(n_unique_feats)))[0]]
    # Find all the unique groups in the chosen iteration 
    connections = numpy.unique(connections_all[it],axis=0)
    # If the function is being used to learn filter connections, perform the following: 
    if flag == 0:
        # If not all the groups in connections_all[it] are unique, this is an issue because 
        # the Keras model expects num_groups groups. Therefore, create groups of randomly
        # selected samples to add to "connections" until the total number of unique groups 
        # is equal to "num_groups."
        while(connections.shape[0] != num_groups):
            indices = numpy.arange(num_filters)
            numpy.random.shuffle(indices)
            connections = numpy.concatenate((connections,indices[0:group_size].reshape((1,group_size))),axis=0)
    return connections

def model_fp(model,model_train,layer_name):
    ''' model_fp Method 
            Create an intermediate version of the input Keras model, where the output is 
            derived from an earlier layer specified by "layer_name"

            Args:
                model: Compiled Keras model 
                model_train: 4-D array of inputs to pass through the new intermediate model 
                             and generate the desired output from an earlier layer.
                             model_train is of shape: 
                             (Number of smaples, height, width, number of filters)
                layer_name: string name of the layer to produce the ouput from in the Keras
                            model
            Returns: 
                output: 4-D array of outputs from the layer specified by "layer_name"
                        output is of shape:
                        (Number of samples, height, width, number of filters)
                
    '''
    # Create an intermediate version of the input model, where the intput is the same, but 
    # the output is derived from a layer preceding the input model's output. ("layer_name"
    # specifies which layer should be the intermediate model's output)
    intermediate_layer_model = Model(inputs=model.input,outputs=model.get_layer(layer_name).output)  
    # Determine the output of the intermediate model when given an input of model_train
    output = intermediate_layer_model.predict(model_train)
    return output 

def autoconv2d(X):
    ''' autoconv2d Method 
            Perform autoconvolution of the input X 

            Args:
                X: (Number of samples, number of filters, height, width)
            Returns: 
                4-D array representing the autoconvolution of X of shape:
                (Number of samples, number of filters, height, width)
                
    '''
    sz = X.shape 
    # Local centering by subtracting the mean
    X -= numpy.reshape(numpy.mean(X, axis=(2,3)),(-1,sz[-3],1,1)) 
    # Local scaling by dividing by the standard deviation 
    X /= numpy.reshape(numpy.std(X, axis=(2,3)),(-1,sz[1],1,1)) + 1e-10 
    # In order to perform autoconvolution, the height and width dimensions of X 
    # need to be doubled. Achieve this using numpy.pad
    X = numpy.pad(X, ((0,0),(0,0),(0,sz[-2]-1),(0,sz[-1]-1)), 'constant') 
    # Return the autoconvolution of X
    return numpy.real(numpy.fft.ifft2(numpy.fft.fft2(X)**2))

def resize_batch(X, new_size):
    ''' resize_batch Method 
            Resize X into desired shape, specified by "new_size"

            Args:
                X: (Number of samples, number of filters, height, width)
            Returns: 
                4-D reshaped array of X 
                
    '''
    out = []
    # Resize X into the desired shape "new_size"
    for x in X:
        out.append(resize(x.transpose((1,2,0)), new_size, order=1, preserve_range=True,mode='constant').transpose((2,0,1)))
    return numpy.stack(out)

def separate_trainlabels(Y_train,num_perclass):
    ''' separate_trainlabels Method 
            Randomly shuffle the indices of the training set and extract arrays "Y_pos"
            and "Y_neg" holding indices of solely positive and solely negative labels, 
            respectively 

            Args:
                Y_train: Labels for the training set X_train of shape: (Number of samples),
                         with labels being either 0 or 1
		num_perclass: int number of samples to include from the positive class 
			      (and negative class) in "Y_pos" and "Y_neg," respectively
			 
            Returns: 
                Y_pos: 1-D numpy array of indices to positive labels 
                Y_neg: 1-D numpy array of indices to negative labels
                indices: 1-D array of shuffled indices into the original array Y_train
                
    '''
    # Randomly shuffle the indices of Y_train
    indices = numpy.arange(len(Y_train))
    numpy.random.shuffle(indices)
    # Reorder Y_train according to the shuffled indices 
    labels = Y_train[indices]
    # Initialize arrays "Y_pos" and "Y_neg" to hold the indices of the positive examples 
    # and negative examples, respectively. 
    Y_pos = []
    Y_neg = []
    for ii in range(len(labels)):
        # Add the index of a positive label to Y_pos if the length of Y_pos is less than 50.
        # This ensures Y_pos eventually has a total of 50 positive labels. 
        if labels[ii] == 1:
            if len(Y_pos) < num_perclass:
                Y_pos.append(ii)
        # Add the index of a negative label to Y_neg if the length of Y_neg is less than 50.
        # This ensures Y_neg eventually has a total of. 50 negative labels.
        else:
            if len(Y_neg) < num_perclass:
                Y_neg.append(ii)
        # Break the for loop as soon as Y_pos and Y_neg both have 50 elements 
        if len(Y_pos) == num_perclass and len(Y_neg) == num_perclass:
            break
    return Y_pos,Y_neg,indices
