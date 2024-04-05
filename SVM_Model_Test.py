import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score


# Read and prepare the training data
df = pd.read_csv(r"C:\Users\Vishal\Desktop\Fluid Research\Journal_Paper_Optimization_Codes_And_Data\train3.csv")
obj = df['m'].values
r = df["r"].values
x = df["x"].values
y = df["y"].values

# Normalizing 
d_normalized = 2 * (r/909)
x_normalized = x/909
y_normalized = y/909

X_train = np.column_stack((d_normalized, x_normalized, y_normalized))

# Read and prepare the test data
df2 = pd.read_csv(r"C:\Users\Vishal\Desktop\Fluid Research\Journal_Paper_Optimization_Codes_And_Data\test3.csv")
obj_test = df2['m'] 
X_test = df2.drop(columns=["m"]).values

# Normalize the test data using the same scaling factors as the training data
d_test_normalized =  2 * (X_test[:, 0]/909)
x_test_normalized = X_test[:, 1]/909
y_test_normalized = X_test[:, 2]/909

# Combine the normalized features into a single array
X_test_normalized = np.column_stack((d_test_normalized, x_test_normalized, y_test_normalized))

num_test_samples = X_test_normalized.shape[0]

svm_model = svm.SVR(kernel='sigmoid')
svm_model.fit(X_train, obj) 
predictions = svm_model.predict(X_test_normalized)


# Print predictions and ground truth
print("Predictions:")
print(predictions)

# Ground truth values
ground_truth = df2['m'].values
print("Ground Truth:")
print(ground_truth)

errors = (abs((predictions - df2['m']) / df2['m']) * 100)
print("Error = "+ "\n" + str(errors))

mse = mean_squared_error(obj_test, predictions)
print(f"Mean Squared Error: {mse}")

r2 = r2_score(obj_test, predictions)
print(f"R-squared score: {r2}")

def r2_score(y_true, y_pred):
    """
    Calculate the R2 score given the true and predicted values.
    
    Parameters:
    y_true : array-like
        The true values of the dependent variable.
    y_pred : array-like
        The predicted values of the dependent variable.
        
    Returns:
    r2 : float
        The R2 score.
    """
    # Calculate the mean of the true values
    mean_y_true = np.mean(y_true)
    
    # Calculate the total sum of squares (SST)
    sst = np.sum((y_true - mean_y_true) ** 2)
    
    # Calculate the residual sum of squares (SSE)
    sse = np.sum((y_true - y_pred) ** 2)
    
    # Calculate R2 score
    r2 = 1 - (sse / sst)
    
    return r2

print("R2 Score: ", r2_score(obj_test, predictions))