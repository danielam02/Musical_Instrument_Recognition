# IMPORTING

import joblib
import pandas as pd
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier  
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


#%%
# MAIN 

# Load dataset
data = pd.read_csv('features_training.csv', header=None)


# Extract the features (all columns except the first one) and labels (first column)
X = data.iloc[:, 1:].values  
y = data.iloc[:, 0].values


# Encode the categorical labels into numeric values
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)  


# Initialize Leave-One-Out Cross-Validation
loo = LeaveOneOut()


# Initialize lists to store predictions and actual labels
y_true = []
y_pred = []


# Choose the classifier (options: 'neural_network', 'naive_bayes', 'svm', 'random_forest')
classifier_choice = 'naive_bayes' 


# Loop through each train-test split
for train_index, test_index in loo.split(X):
    # Split the data
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # If using any model that is not naive bayes use scalling
    if classifier_choice != 'naive_bayes':
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    # Initialize the classifier based on user choice
    if classifier_choice == 'neural_network':
        clf = MLPClassifier(
            hidden_layer_sizes=(100,),   # Single hidden layer with 100 neurons
            activation='relu',           # ReLU activation function
            solver='adam',               # Adam solver for optimization
            alpha=0.0001,                # Regularization parameter
            max_iter=200,                # Maximum iterations
            random_state=42)             # For replicable training
        
    elif classifier_choice == 'naive_bayes':
        clf = GaussianNB()
        
    elif classifier_choice == 'svm':
        clf = SVC(
            kernel='rbf',                # Radial basis function kernel
            C=1.0,                       # Regularization parameter
            random_state=42)             # For replicable training
        
    elif classifier_choice == 'random_forest':
        clf = RandomForestClassifier(
            n_estimators=10,             # Number of trees in the forest
            max_depth=None,              # Maximum depth of the tree
            random_state=42)             # For replicable training
        
    else:
        raise ValueError("Invalid classifier choice. Choose from: 'neural_network', 'naive_bayes', 'svm', 'random_forest'.")

    clf.fit(X_train, y_train)

    # Make predictions for the test set
    y_pred.append(clf.predict(X_test)[0])
    y_true.append(y_test[0])


# Evaluate the model
accuracy = accuracy_score(y_true, y_pred)
print(f"LOOCV Accuracy: {accuracy:.2f}")


# F1 Score
f1 = f1_score(y_true, y_pred, average='weighted')  
print(f"F1 Score: {f1:.2f}")


# Plot the Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
cm_df = pd.DataFrame(cm, index=label_encoder.classes_, columns=label_encoder.classes_)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_df, annot=True, fmt='g', cmap='Blues', cbar=False, annot_kws={"size": 14})
plt.title('Confusion Matrix', fontsize=16)
plt.xlabel('Predicted', fontsize=14)
plt.ylabel('True', fontsize=14)
plt.xticks(rotation=45)
plt.yticks(rotation=45)
plt.tight_layout()
plt.show()


# Save the trained model
joblib.dump(clf, 'model.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')
if classifier_choice != 'naive_bayes':
    joblib.dump(scaler, 'scaler.pkl') 
