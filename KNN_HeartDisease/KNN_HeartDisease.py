import pandas as pd
import numpy as np

#Applying z-score to standardize the dataframe
def standardize_df(df):
    
    for col in df.columns:
        datatype = df[col].dtypes

        if (datatype == 'float64') or (datatype == 'int64'):
            std = df[col].std()
            mean = df[col].mean()
            df[col] = (df[col] - mean) / std

    return df

#Calculating euclidian distance between two given datapoints
def e_dist(e1, e2):
    sqrd_dist = 0
    
    #Points in this dataset have 13 dimensions
    for i in range(13):
        sqrd_dist += (e1[i] - e2[i]) ** 2

    return sqrd_dist ** (1/2)


#Those next few lines are responsible for applying the whole algorithm
def apply_KNN(k, x): 
    
    dist=[]
    #This array will contain all distances from the given test point to the training points, as well as the training
    #points' outcomes
    for i in traindf.index:
        dist.append([e_dist(traindf.loc[i], x), traindf.target[i]])
    
    #Sorting the array so we can have the smaller distances (nearest neighbours) on its early positions
    dist.sort()

    #This little loop will count how many outcomes of the k nearest neighbours are Positive
    pos = 0
    for i in range (k):
        if dist[i][1] == 'Positive':
            pos += 1
    
    #Checking the predominant outcome, and then classifying our training point
    if (k-pos < k/2):
        return 'Positive'
    else:
        return 'Negative'

#Using pandas to read the dataset's csv
df = pd.read_csv("https://bit.ly/2YKHhQi")

#Changing the outcome to a string result so we don't affect the data standardzation and have a more clear view of what's
#Going on
df['target'] = df['target'].replace(0, 'Negative')
df['target'] = df['target'].replace(1, 'Positive')

#Organazing dataset, appying standardize function and sppliting the original dataframe into "train" and "test"
df = standardize_df(df) 
traindf = df.iloc[:700] 
testdf = df.iloc[700:850].reset_index(drop=True)

#Choosing "k" for the algorithm, k=1 was found to be the best one in this ocasion, showing that this dataframe is really
#Well balanced
k = 1

#Counting answers and printing the results out
correct = 0 
wrong = 0 
falseneg = 0 
falsepos = 0
trueneg = 0
truepos = 0
for i in range(len(testdf.index)): 
    algorithm_outcome = apply_KNN(k, testdf.loc[i]) 
    real_outcome = testdf.iloc[i]['target']
    
    if(algorithm_outcome == real_outcome):
        if(real_outcome == 'Positive'):
            truepos += 1
        else:
            trueneg += 1
        correct += 1
    else:
        if (real_outcome == 'Negative'):
            falsepos += 1
        else:
            falseneg += 1
        wrong += 1
    
print('K = '+str(k)+'\n')
print('Total: '+str(correct+wrong))
print('Correct: '+str(correct))
print('Wrong: '+str(wrong)+'\n')
print('True Positives: '+str(truepos))
print('True Negatives: '+str(trueneg)+'\n')
print('Fake Positives: '+str(falsepos))
print('Fake Negatives: '+str(falseneg)+'\n')
print('Accuracy: '+str(round(100*(correct/(correct+wrong))))+'%')
print('Precision: '+str(round(100*(truepos/(truepos + falsepos))))+'%')
print('Recall: '+str(round(100*(truepos/(truepos + falseneg)))))