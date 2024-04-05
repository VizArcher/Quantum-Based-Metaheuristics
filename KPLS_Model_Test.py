import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import qmc
from smt.surrogate_models import RBF, IDW, QP
from smt.surrogate_models import KRG
from smt.surrogate_models import KPLS
import pandas as pd
from sklearn.metrics import r2_score
from mpl_toolkits.mplot3d import axes3d

# Read and prepare the training data
df = pd.read_csv(r"C:\Users\Vishal\Desktop\Fluid Research\Journal_Paper_Optimization_Codes_And_Data\train3.csv")
obj = df['m']
obj = np.array(obj)
r = df["r"].values
x = df["x"].values
y = df["y"].values

# Normalize the training data between 0 and 1
'''
r_normalized = (r - np.min(r)) / (np.max(r) - np.min(r))
x_normalized = (x - np.min(x)) / (np.max(x) - np.min(x))
y_normalized = (y - np.min(y)) / (np.max(y) - np.min(y))
'''
#r_normalized = r/909

d_normalized = 2 * (r/909)
x_normalized = x/909
y_normalized = y/909

x_normalized = np.column_stack((d_normalized, x_normalized, y_normalized))

# Print the normalized data
print(x_normalized)
print(obj)

# Read and prepare the test data
df2 = pd.read_csv(r"C:\Users\Vishal\Desktop\Fluid Research\Journal_Paper_Optimization_Codes_And_Data\test3.csv")
X = df2.drop(columns=["m"]).values

# Normalize the test data using the same scaling factors as the training data
'''
r_test_normalized = (X[:, 0] - np.min(r)) / (np.max(r) - np.min(r))
x_test_normalized = (X[:, 1] - np.min(x)) / (np.max(x) - np.min(x))
y_test_normalized = (X[:, 2] - np.min(y)) / (np.max(y) - np.min(y))
'''
#r_test_normalized = X[:, 0]/909

d_test_normalized =  2 * (X[:, 0]/909)
x_test_normalized = X[:, 1]/909
y_test_normalized = X[:, 2]/909

X_normalized = np.column_stack((d_test_normalized, x_test_normalized, y_test_normalized))

# Print the normalized test data
print(X_normalized)

num_test_samples = X_normalized.shape[0]

errors_df = pd.DataFrame()

R2 = [] 

for j in range(-100 ,100 , 10) : 

    # Create and train the KRG model
    sm = KPLS(theta0= [j])
    sm.set_training_values(x_normalized, obj)
    sm.train()

    l = []
    errors = []

    # Make predictions on the normalized test data
    for i in range(num_test_samples):
        pred = sm.predict_values(np.array([X_normalized[i]]))
        print(pred)
        l.append(pred[0][0])
        error = (abs((pred - df2['m'][i]) / df2['m'][i]) * 100)
        print("Error = " + str(error))
        error_str = str(error[0][0])  # Convert the list to a string
        errors.append(error_str)  # Append the string to your list

    for i in range(num_test_samples):
        print(str(errors[i]))

    errors_df[f'Error_{j}'] = errors
    output_rsm_csv_file = r'C:\Users\Vishal\Desktop\Fluid Research\Journal_Paper_Optimization_Codes_And_Data\KPLS\Kriging_KPLS_New_Data_Error_values.csv'

    truth = [i for i in df2["m"]]
    print("ground truth:" + str(truth))
    print("predictions:" + str(l))

    r2 = r2_score(l, truth)
    r2 = round(r2,3)
    R2.append(r2)

errors_df.to_csv(output_rsm_csv_file, index=False)

r2_df = pd.DataFrame({"R2": R2})
output_rsm_csv_file = r'C:\Users\Vishal\Desktop\Fluid Research\Journal_Paper_Optimization_Codes_And_Data\KPLS\Kringing_KPLS_New_Data_R2_values.csv'
r2_df.to_csv(output_rsm_csv_file, index=False)


def obj_original(X):
    obj = []
    for i in range(25):
        l = []
        l.append(list(X[i]))
        pred = sm.predict_values(np.array(l))
        #print(np.array(l).shape)
        obj.append(pred[0][0])
    obj = np.array(obj)
    return obj

def obj(X):
    pred = sm.predict_values(np.array(X))
    obj = pred[0][0]
    return obj

#--------------------------------------------------------------Validation Plot --------------------------------------------------------------------------
plt.figure(figsize=(6.4,4.8),dpi=100)
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.linewidth"] = 2

a, b = np.polyfit(l, truth, 1)

t = np.linspace(-1,3,20)
r2 = r2_score(l, truth)
r2 = round(r2,3)
plt.text(0.11, 0.35, '$R^2$ score = ' + str(r2), fontsize = 14)

plt.plot(t, a*t+b,c = "black",lw = "3" )

plt.scatter(l,truth,c = "red",s=50)
plt.xlim(-1, 3)
plt.ylim(-1, 3)

plt.xlabel("Predicted $C_{p}$ [-]",fontsize=16)
plt.ylabel("Actual $C_{p}$ [-]",fontsize=16)
plt.tick_params(axis='both',size=8,labelsize=14,direction='inout')

plt.show()
