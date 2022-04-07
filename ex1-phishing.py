from random import randrange
import pandas as pd
import matplotlib.pyplot as plt

#Config
evaluation_method = "holdout" #holdout or cross
holdout_split = 0.8
cross_validations = 10

# Load data
df_raw = pd.read_csv("data/phpV5QYya.csv")

# Preprocess
# Data is already preprocessed - just shuffle to make sure
df_shuffeled = df_raw.sample(frac=1,random_state=193520285)

#Split data
subsets = []

if(evaluation_method == "holdout"):
    num_learning = round(df_shuffeled.shape[0] * holdout_split)
    subsets.append(df_shuffeled[0:num_learning-1, :])
    subsets.append(df_shuffeled[num_learning:,:])

if(evaluation_method == "cross"):
    num_per_subset = (int)(df_shuffeled.shape[0]/cross_validations)
    residual = df_shuffeled[0]%cross_validations

    offset = 0
    for i in range(cross_validations-1):
        begin = num_per_subset*i+offset
        if(offset<residual):
            offset = offset+1
        end = num_per_subset*(i+1)+offset-1

        subsets.append(df_shuffeled[begin:end, :])

print(subsets[0].shape())
print(subsets[0].head())

# Classification:
    # Obvious choice: Bayes
    # k-NN: Calculate distance by 
        #L1
        #L2
    # Decision tree: information

# Evalutaion
# We rather have False Positives than False Negatives -> we want high sensitivity
# Two evaluation measures: Accuaracy & Sensitivity