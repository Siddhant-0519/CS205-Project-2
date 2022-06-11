import pandas as pd
import numpy as np
import random
import sys

output_file = r'C:\Users\EndUser\Desktop\UCR Academics\Spring 22\CS205-AI\Output_Large_Backward.txt'
f = open(output_file,'w')

df = pd.read_csv(r'C:\Users\EndUser\Desktop\UCR Academics\Spring 22\CS205-AI\CS205_LargeDataset.csv')
#f.write(df)

# Seperate labels from features
label = df['target']


# Drop target from dataframe
df = df.drop(['target'],axis=1)

#get number of rows(r) and cols(c) for features
r,c = df.shape

def accuracy_stub(x,y,z):
    acc = round(random.uniform(0,100),2)
    return acc

def leave_one_out_cross_validation(data,current_feature,feature_to_add,label):
    correct_classifications = 0
    #initailize copy of data and curr_features
    data1 = data.copy()
    feature_set = current_feature.copy()

     
    # Adding feature_to_add --> current_feature. 
    feature_set.add(feature_to_add)
    feature_set = list(feature_set)
    
    # Select features based on current_feature
    data1 = data1.iloc[:,feature_set]

    

    
    
    # OUTER LOOP: Leave one out
    for i in range(r):
        # select the point to leave out
        point = data1.loc[[i]]
        point = point.to_numpy()
        
        # select label of selected point
        point_label = label[i]
        

        nn_dist = float('inf')
        #nn_loc = float('inf')
        nn_label = None

        for j in range(r):
            if j != i:
                #Euclidean distance function using numpy
                dist = np.linalg.norm( point - data1.loc[[j]] , axis=1 )
                
                if dist < nn_dist:
                    nn_dist = dist
                    #nn_loc = j
                    nn_label = label[j]
        
        
        if point_label == nn_label:
            correct_classifications += 1
    
    accuracy = correct_classifications / r
    
    return accuracy




def forward_feature_search(data,label):
    
    current_features = set()
    global_best_acc = 0
    global_best_features = []

    for i in range(c):
        f.write('On the ' + str((i+1))+" th level of Search Tree:\n")
        
        feature_added = -1
        local_best_acc = 0

        for j in range(c):
            if j not in current_features:
                f.write("---Considering adding the feature " + str(j) +"\n ")
                accuracy = leave_one_out_cross_validation(data,current_features,j,label)
                f.write("Accuracy of feature " + str(j) +" is : " + str(accuracy) + "\n ")

                if accuracy > local_best_acc:
                    local_best_acc = accuracy
                    feature_added = j


        
        current_features.add(feature_added)
        
        
        f.write("On Level " + str((i+1)) + " feature added was " + str(feature_added) +" to the current set \n")
        f.write("Current Feature Set: "+str(current_features)+"Accuracy : " + str(local_best_acc) + "\n")

        if local_best_acc > global_best_acc:
            global_best_acc = local_best_acc
            global_best_features = current_features
    return global_best_features,global_best_acc

# features_set,acc=forward_feature_search(df,label)
# f.write("Best set of Features are : " + str(features_set) +" with accuracy: "+ str(acc))

def leave_one_out_cross_validation_backward(data,current_feature,feature_to_drop,label):
    correct_classifications = 0
    #initailize copy of data and curr_features
    data1 = data.copy()
    feature_set = current_feature.copy()

     
    # Adding feature_to_add --> current_feature. 
    #print(feature_set,feature_to_drop)
    feature_set.remove(feature_to_drop)
    feature_set = list(feature_set)
    #print(feature_set)
    # Select features based on current_feature
    data1 = data1.iloc[:,feature_set]

    

    
    
    # OUTER LOOP: Leave one out
    for i in range(r):
        # select the point to leave out
        point = data1.loc[[i]]
        point = point.to_numpy()
        
        # select label of selected point
        point_label = label[i]
        

        nn_dist = float('inf')
        #nn_loc = float('inf')
        nn_label = None

        for j in range(r):
            if j != i:
                #Euclidean distance function using numpy
                dist = np.linalg.norm( point - data1.loc[[j]] , axis=1 )
                
                if dist < nn_dist:
                    nn_dist = dist
                    #nn_loc = j
                    nn_label = label[j]
        
        
        if point_label == nn_label:
            correct_classifications += 1
    
    accuracy = correct_classifications / r
    #print(accuracy)
    
    return accuracy


def backward_feature_search(data,label):
    
    current_features = set([i for i in range(c)])
    #print(current_features)
    global_best_acc = 0
    global_best_features = []
    dropped = set()

    for i in range(c):
        f.write('On the ' + str((i+1))+" th level of Search Tree:\n")
        
        feature_dropped = -1
        local_best_acc = 0

        for j in range(c):
          if j not in dropped:
              f.write("---Considering dropping the feature " + str(j) +"\n ")
              accuracy = leave_one_out_cross_validation_backward(data,current_features,j,label)
              f.write("Accuracy after dropping feature " + str(j) +" is : " + str(accuracy) + "\n ")

              if accuracy > local_best_acc:
                  local_best_acc = accuracy
                  feature_dropped = j

        #print(current_features,feature_dropped)
        dropped.add(feature_dropped)
        #print(dropped)
        current_features.remove(feature_dropped)
        
        
        f.write("On Level " + str((i+1)) + " feature dropped was " + str(feature_dropped) +" from the current set \n")
        f.write("Current Feature Set: "+str(current_features)+" Accuracy : " + str(local_best_acc) + "\n")

        if local_best_acc > global_best_acc:
            global_best_acc = local_best_acc
            global_best_features = current_features
    return global_best_features,global_best_acc


features_set_back ,accuracy = backward_feature_search(df,label)
f.write("Best set of Features are : " + str(features_set_back) +" with accuracy: "+ str(accuracy))
f.close()


