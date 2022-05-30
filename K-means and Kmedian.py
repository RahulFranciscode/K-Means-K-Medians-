# Import relevant libraries
import numpy as np
import pandas as pd
from copy import deepcopy
import matplotlib.pyplot as plt
import os



##############################################################################
# Open using pandas
a = pd.read_csv( "animals.csv", sep = " ", header = None)
c = pd.read_csv("countries.csv", sep = " ", header = None)
f = pd.read_csv( "fruits.csv", sep = " ", header = None)
v = pd.read_csv("veggies.csv", sep = " ", header = None)

# Add category in cluster
a['Category'] = 'animals'
c['Category'] = 'countries'
f['Category'] = 'fruits'
v['Category'] = 'veggies'

# concatenate
data = pd.concat([a, c, f, v], ignore_index = True)

# Change number to all class  0=animals, 1=countries, 2=fruits, 3=veggies
labels = (pd.factorize(data.Category)[0]+1) - 1 
x = data.drop([0, 'Category'], axis = 1).values
#  each category for  Save the maximum index for the P/R/F
animal_max = data.index[data['Category'] == 'animals'][-1]
country_max = data.index[data['Category'] == 'countries'][-1]
fruit_max = data.index[data['Category'] == 'fruits'][-1]
veg_max = data.index[data['Category'] == 'veggies'][-1]


def kmeans_clustering(x, k, distance,trueOrFalsePRF):
    np.random.seed(10)
 


    
    #  centroids initialize randomly
    centroids = []
    loc = np.random.randint(x.shape[0], size = k)
    while (len(loc) > len(set(loc))):
        loc = np.random.randint(x.shape[0], size = k)
    for i in loc:
        centroids.append(x[i])
    # Create old and new centroids
    previous_centroids = np.zeros(np.shape(centroids))
    latest_centroids = deepcopy(centroids)

    # assign cluster to zeros
    clust = np.zeros(x.shape[0])
    # Create an error object
    error = np.linalg.norm(latest_centroids - previous_centroids)
    num_errors = 0

    # assign while to find updates:
    while error != 0:
        #print(error)
        dist = np.zeros([x.shape[0], k])
        # Add one update
        num_errors += 1
        # Calculate the Euclidean distance    
        if distance == "Euclidean":
            for j in range(len(centroids)):
                dist[:, j] = np.linalg.norm(x - latest_centroids[j], axis=1)
        # Calculate the Manhattan distance  
        elif distance == "Manhattan":
            for j in range(len(centroids)):
                dist[:, j] = np.sum(np.abs(x - latest_centroids[j]), axis=1)
     
        #  cluster assignment Calculate 
        clust = np.argmin(dist, axis = 1)

        # Assign the new copy of centroids 
        previous_centroids = deepcopy(latest_centroids)

        # Calculate the mean to re-adjust the cluster centroids
        if distance == "Euclidean":
            # Calculate the mean to re-adjust the cluster centroids
            for m in range(k):
                latest_centroids[m] = np.mean(x[clust == m], axis = 0)
        else:
            for m in range(k):
                latest_centroids[m] = np.median(x[clust == m], axis = 0)

        # update the error
        error = np.linalg.norm(np.array(latest_centroids) - np.array(previous_centroids))

    # predict predicted_cluster and centroids
    predicted_clust = clust
    centroids_predicted = np.array(latest_centroids)
    #print("centroids_predicted",centroids_predicted)
    if trueOrFalsePRF == False:
        print("\nFinal Results of K-Means/k-median Clustering without regularization", distance, 
              "Distance Measurement\n")
        print("\t Number of clusters :", k)
        print("\t Number of Updates:", num_errors)
        print ("\t predicted  Clusters:\n", predicted_clust)
        print ("\t predicted Centroid Locations:\n", centroids_predicted)
        print("-----------------------------------------------------------------------------")
    
    # to calculate P, R and F 
    else:
        # Create objects of the index positioning of the different classes
        index_animal = predicted_clust[:animal_max+1]
        index_countries = predicted_clust[animal_max+1:country_max+1]
        index_fruit = predicted_clust[country_max+1:fruit_max+1]
        index_veggies = predicted_clust[fruit_max+1:veg_max+1]
        
        # Create objects for contingency calculations
        # True Positives
        TRUE_POSITIVE = 0
        # False Negatives
        FALSE_NEGATIVE = 0
        # True Negatives
        True_negative = 0
        # False Positives
        FALSE_POSITIVE = 0
        #--------------------------------------------------------
        #  index_animal For every row in
        for i in range(len(index_animal)):
            # index_animal  For every row in
            for j in range(len(index_animal)):
                # If i and j are not the same, and j > i
                if (i != j & j>i):
                    #  add 1 to TRUE_POSITIVE If i is equal to j then
                    if(index_animal[i] == index_animal[j]):
                        TRUE_POSITIVE += 1
                    # Otherwise  FALSE_NEGATIVE
                    else:
                        FALSE_NEGATIVE += 1
            # index_countries For every row in               
            for j in range(len(index_countries)):
                #  add 1 to FALSE_POSITIVE If i is equal to j then
                if(index_animal[i] == index_countries[j]):
                    FALSE_POSITIVE += 1
                # Otherwise True_negative
                else:
                    True_negative += 1
            #  index_fruit For every row in
            for j in range(len(index_fruit)):
                #  add 1 to FALSE_POSITIVE If i is equal to j then
                if(index_animal[i]==index_fruit[j]):
                    FALSE_POSITIVE += 1
                # Otherwise True_negative
                else:
                    True_negative += 1
            #  index_veggies For every row in
            for j in range(len(index_veggies)):
                # If i is equal to j then add 1 to FALSE_POSITIVE
                if(index_animal[i] == index_veggies[j]):
                    FALSE_POSITIVE += 1
                # Otherwise True_negative
                else:
                    True_negative += 1
        #--------------------------------------------------------    
        #index_countries For every row in
        for i in range(len(index_countries)):
            # index_countries For every row in
            for j in range(len(index_countries)):
                # If i and j are not the same, and j > i
                if (i != j & j>i):
                    # If i is equal to j then add 1 to TRUE_POSITIVE
                    if(index_countries[i] == index_countries[j]):
                        TRUE_POSITIVE += 1
                    # Otherwise FALSE_NEGATIVE
                    else:
                        FALSE_NEGATIVE += 1
            #  index_fruit For every row in
            for j in range(len(index_fruit)):
                # If i is equal to j then add 1 to FALSE_POSITIVE
                if(index_countries[i] == index_fruit[j]):
                    FALSE_POSITIVE += 1
                # Otherwise add 1 toTrue_negative
                else:
                    True_negative += 1
            # For every row in index_veggies
            for j in range(len(index_veggies)):
                # If i is equal to j then add 1 to FALSE_POSITIVE
                if(index_countries[i] == index_veggies[j]):
                    FALSE_POSITIVE += 1
                # Otherwise True_negative
                else:
                    True_negative += 1     
        #--------------------------------------------------------
        #  index_fruit For every row in
        for i in range(len(index_fruit)):
            #  index_fruit For every row in
            for j in range(len(index_fruit)):
                # If i and j are not the same, and j > i
                if (i != j & j>i):
                    # If i is equal to j then add 1 to TRUE_POSITIVE
                    if(index_fruit[i] == index_fruit[j]):
                        TRUE_POSITIVE += 1
                    # Otherwise FALSE_NEGATIVE
                    else:
                        FALSE_NEGATIVE += 1
            #  index_veggies For every row in
            for j in range(len(index_veggies)):
                # If i is equal to j then add 1 to FALSE_POSITIVE
                if(index_fruit[i] == index_veggies[j]):
                    FALSE_POSITIVE += 1
                # Otherwise True_negative
                else:
                    True_negative += 1    
        #--------------------------------------------------------
        # For every row in index_veggies
        for i in range(len(index_veggies)):       
            # For every row in index_veggies
            for j in range(len(index_veggies)):
                # If i and j are not the same, and j > i
                if (i != j & j>i):
                    # If i is equal to j then add 1 to TRUE_POSITIVE
                    if(index_veggies[i] == index_veggies[j]):
                        TRUE_POSITIVE += 1
                    # Otherwise add 1 to FALSE_NEGATIVE
                    else:
                        FALSE_NEGATIVE += 1       
        # Calculate the Precision (P), Recall (R), and F-Score (F) and round
        # to 2 decimal places
        P = round((TRUE_POSITIVE / (TRUE_POSITIVE + FALSE_POSITIVE)), 2)
        R = round((TRUE_POSITIVE / (TRUE_POSITIVE + FALSE_NEGATIVE)), 2)
        F = round((2 * (P * R) / (P + R)), 2)
    
    
    
   

    
     
    
        
    
        print("\nFinal Results of K-Means/kmedian Clustering without regularization", distance, 
              "Distance Measurement")
        # Print the results
        print("\tNumber of Clusters:", k)
        print("\tNumber of Updates:", num_errors)
        print("\t results for:", P, ", R:", R, ", F:", F)
        
        # Return the P, R and F values for plotting4
        return P, R, F
    

def kmeans_clustering1(x, k, distance):
    np.random.seed(10)
    x = x / np.linalg.norm(x)
 


    
    #  initialise centroids Randomly
    centroids = []
    loc = np.random.randint(x.shape[0], size = k)
    while (len(loc) > len(set(loc))):
        loc = np.random.randint(x.shape[0], size = k)
    for i in loc:
        centroids.append(x[i])
    # Create  centroids copies  for updating
    previous_centroids = np.zeros(np.shape(centroids))
    latest_centroids = deepcopy(centroids)

    # assign clusters to zero
    clust = np.zeros(x.shape[0])
    # find error
    error = np.linalg.norm(latest_centroids - previous_centroids)
    num_errors = 0

    # assign while:
    while error != 0:
        #print(error)
        dist = np.zeros([x.shape[0], k])
        # update 
        num_errors += 1
        # Calculate the Euclidean distance     
        if distance == "Euclidean":
            for j in range(len(centroids)):
                dist[:, j] = np.linalg.norm(x - latest_centroids[j], axis=1)
        # Calculate the Manhattan distance    
        elif distance == "Manhattan":
            for j in range(len(centroids)):
                dist[:, j] = np.sum(np.abs(x - latest_centroids[j]), axis=1)
               
            

        # Calculate the cluster 
        clust = np.argmin(dist, axis = 1)

        # Assign the new copy of centroids 
        previous_centroids = deepcopy(latest_centroids)

        # Calculate the mean  cluster centroids
        if distance == "Euclidean":
            # Calculate the mean  cluster centroids
            for m in range(k):
                latest_centroids[m] = np.mean(x[clust == m], axis = 0)
        else:
            for m in range(k):
                latest_centroids[m] = np.median(x[clust == m], axis = 0)

        # Re-calculate the error
        error = np.linalg.norm(np.array(latest_centroids) - np.array(previous_centroids))

    #find the predicted clusters and centroids
    predicted_clust = clust
    centroids_predicted = np.array(latest_centroids)
    
    
    
   

    
        # Create objects of the index positioning of the different classes
    index_animal = predicted_clust[:animal_max+1]
    index_countries = predicted_clust[animal_max+1:country_max+1]
    index_fruit = predicted_clust[country_max+1:fruit_max+1]
    index_veggies = predicted_clust[fruit_max+1:veg_max+1]
        
        # Create objects for contingency calculations
        # True Positives
    TRUE_POSITIVE = 0
        # False Negatives
    FALSE_NEGATIVE = 0
        # True Negatives
    True_negative = 0
        # False Positives
    FALSE_POSITIVE = 0
        #--------------------------------------------------------
        #  index_animal For every row in
    for i in range(len(index_animal)):
            #  index_animal For every row in
        for j in range(len(index_animal)):
                # If i and j are not the same, and j > i
            if (i != j & j>i):
                    # If i is equal to j then add 1 to TRUE_POSITIVE
                if(index_animal[i] == index_animal[j]):
                        TRUE_POSITIVE += 1
                    # Otherwise  FALSE_NEGATIVE
                else:
                    FALSE_NEGATIVE += 1
            # index_countries For every row in               
        for j in range(len(index_countries)):
                # If i is equal to j then add 1 to FALSE_POSITIVE
            if(index_animal[i] == index_countries[j]):
                FALSE_POSITIVE += 1
                # Otherwise True_negative
            else:
                True_negative += 1
            # index_fruit For every row in 
        for j in range(len(index_fruit)):
                # If i is equal to j then add 1 to FALSE_POSITIVE
            if(index_animal[i]==index_fruit[j]):
                FALSE_POSITIVE += 1
                # Otherwise True_negative
            else:
                True_negative += 1
            #  index_veggies For every row in
        for j in range(len(index_veggies)):
                # If i is equal to j then add 1 to FALSE_POSITIVE
            if(index_animal[i] == index_veggies[j]):
                    FALSE_POSITIVE += 1
                # Otherwise aTrue_negative
            else:
                True_negative += 1
        #--------------------------------------------------------    
        # index_countries For every row in
    for i in range(len(index_countries)):
            # index_countries For every row in 
        for j in range(len(index_countries)):
                # If i and j are not the same, and j > i
            if (i != j & j>i):
                    # If i is equal to j then add 1 to TRUE_POSITIVE
                if(index_countries[i] == index_countries[j]):
                    TRUE_POSITIVE += 1
                    # Otherwise  FALSE_NEGATIVE
            else:
                    FALSE_NEGATIVE += 1
            #  index_fruit For every row in
        for j in range(len(index_fruit)):
                # If i is equal to j then add 1 to FALSE_POSITIVE
            if(index_countries[i] == index_fruit[j]):
                FALSE_POSITIVE += 1
                # Otherwise True_negative
            else:
                True_negative += 1
            #  index_veggies For every row in
        for j in range(len(index_veggies)):
                # If i is equal to j then add 1 to FALSE_POSITIVE
            if(index_countries[i] == index_veggies[j]):
                FALSE_POSITIVE += 1
                # Otherwise True_negative
            else:
                True_negative += 1     
        #--------------------------------------------------------
        #  index_fruit For every row in
    for i in range(len(index_fruit)):
            #  index_fruit For every row in
        for j in range(len(index_fruit)):
                # If i and j are not the same, and j > i
            if (i != j & j>i):
                    # If i is equal to j then add 1 to TRUE_POSITIVE
                if(index_fruit[i] == index_fruit[j]):
                    TRUE_POSITIVE += 1
                    # Otherwise  FALSE_NEGATIVE
                else:
                    FALSE_NEGATIVE += 1
            #  index_veggies For every row in
        for j in range(len(index_veggies)):
                # If i is equal to j then add 1 to FALSE_POSITIVE
            if(index_fruit[i] == index_veggies[j]):
                FALSE_POSITIVE += 1
                # Otherwise  True_negative
            else:
                True_negative += 1    
        #--------------------------------------------------------
        #  index_veggies For every row in
    for i in range(len(index_veggies)):       
            # index_veggies  For every row in
        for j in range(len(index_veggies)):
                # If i and j are not the same, and j > i
            if (i != j & j>i):
                    # If i is equal to j then add 1 to TRUE_POSITIVE
                if(index_veggies[i] == index_veggies[j]):
                    TRUE_POSITIVE += 1
                    # Otherwise  FALSE_NEGATIVE
                else:
                    FALSE_NEGATIVE += 1       
        # Calculate the Precision (P), Recall (R), and F-Score (F) and round
        # to 2 decimal places
    P1 = round((TRUE_POSITIVE / (TRUE_POSITIVE + FALSE_POSITIVE)), 2)
    R1 = round((TRUE_POSITIVE / (TRUE_POSITIVE + FALSE_NEGATIVE)), 2)
    F1 = round((2 * (P1 * R1) / (P1 + R1)), 2)
        
    
    print("\nFinal Results of K-Means/k-median Clustering with Regularization", distance, 
              "Distance Measurement")
    
        # Print the results
    print("\tNumber of Clust:", k)
    print("\tNumber of Updates:", num_errors)
    print("\t results:", P1, ", R:", R1, ", F:", F1)
        
        # Return the P, R and F values for plotting4
    return P1, R1, F1

def plotting(k, P, R, F, distance):
    # Plot K against P
    plt.plot(K_list, P_list, label="Precision")
    # Plot K against R
    plt.plot(K_list, R_list, label="Recall")
    # Plot K against F
    plt.plot(K_list, F_list, label="F-Score")
    # Plot the title
    plt.title(" without regularization" + distance, loc="left")
    # Plot the x and y axis labels
    plt.xlabel('Number of Clust')
    plt.ylabel("Score")
    # Display the legend
    plt.legend()
    # Display the plot
    plt.show()   
print("with regularization")
def plotting1(k, P1, R1, F1, distance):
    # Plot K against P
    plt.plot(K_list, P_liste, label="Precision")
    # Plot K against R
    plt.plot(K_list, R_liste, label="Recall")
    # Plot K against F
    plt.plot(K_list, F_liste, label="F-Score")
    # Plot the title
    plt.title(" Regularizarion" + distance, loc="left")
    # Plot the x and y axis labels
    plt.xlabel('Number of Clust')
    plt.ylabel("Score")
    # Display the legend
    plt.legend()
    # Display the plot
    plt.show()  

#print("predicted_clust",predicted_clust)


#kmeans_clustering(x, 4, "Euclidean",False)

#print("Kmedians for k=4,without regularization")

#kmeans_clustering(x, 4, "Manhattan",False)
for question in range(3,5):
    #Create an empty list for P, R, F and K
    P_list = []
    R_list = []
    F_list = []
    K_list = []
    P_liste = []
    R_liste = []
    F_liste = []
    
    # Create an empty string for the distance method
    distance = ""
    
    # Question 2
    if question == 3:
        distance = "Euclidean"
        
    # Question 3
    elif question == 4:
        distance = "Manhattan"
        

        
    
    
    # Define k between 1 - 10
    for k in range(1,10):
        # Append k to a list for plotting
        K_list.append(k)
        # Save the Precision, Recall and F-Scores
        P,R,F = kmeans_clustering(x, k, distance,True)
        P1,R1,F1= kmeans_clustering1(x, k, distance)
        # Append the Precision, Recall and F-Score to each list for plotting
        P_list.append(P)
        R_list.append(R)
        F_list.append(F)
        P_liste.append(P1)
        R_liste.append(R1)
        F_liste.append(F1)
    plotting(K_list, P_list, R_list, F_list, distance)
   
    plotting1(K_list, P_liste, R_liste, F_liste, distance)




    

  
   
    
############################################################################### 
