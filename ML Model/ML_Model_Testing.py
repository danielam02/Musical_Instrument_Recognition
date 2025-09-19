# IMPORTING

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


#%%
# MAIN 

# Load the saved model and label encoder
clf = joblib.load('model.pkl')
label_encoder = joblib.load('label_encoder.pkl')
#scaler = joblib.load('scaler.pkl') # If using model diferent from naive bayes uncomment


# Load the test dataset
test_data = pd.read_csv('features_test.csv', header=None)


# Extract features and true labels from the test data
X_test = test_data.iloc[:, 1:].values  
y_true = test_data.iloc[:, 0].values  


# Encode the test labels using the same label encoder used for training
y_true = label_encoder.transform(y_true)  


# Predict the labels using the trained model
y_pred = clf.predict(X_test)


# Decode the numeric predictions back to the original labels
y_pred_decoded = label_encoder.inverse_transform(y_pred)
y_true_decoded = label_encoder.inverse_transform(y_true)


# Evaluate the model on the test data
accuracy = accuracy_score(y_true_decoded, y_pred_decoded)
print(f"Test Accuracy: {accuracy:.2f}")


# F1 Score
f1 = f1_score(y_true, y_pred, average='weighted')  
print(f"F1 Score: {f1:.2f}")


# Confusion Matrix
cm = confusion_matrix(y_true_decoded, y_pred_decoded)
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
