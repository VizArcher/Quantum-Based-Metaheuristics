import numpy as np
import matplotlib.pyplot as plt
from smt.surrogate_models import KRG, KPLS, KPLSK
import pandas as pd
from sklearn.metrics import r2_score

# Read and prepare the training data
df = pd.read_csv(r"C:\Users\Vishal\Desktop\Fluid Research\Journal_Paper_Optimization_Codes_And_Data\train5.csv")
obj = df['m']
obj = np.array(obj)
r = df["r"].values
x = df["x"].values
y = df["y"].values

# Normalize the training data
d_normalized = 2 * (r/909)
x_normalized = x/909
y_normalized = y/909

x_normalized = np.column_stack((d_normalized, x_normalized, y_normalized))

# Print the normalized data
print(x_normalized)
print(obj)

# Create and train the KRG model
sm = KRG(theta0 = [10], print_global = False)
sm.set_training_values(x_normalized, obj)
sm.train()

# Read and prepare the test data
df2 = pd.read_csv(r"C:\Users\Vishal\Desktop\Fluid Research\Journal_Paper_Optimization_Codes_And_Data\test5.csv")
X = df2.drop(columns=["m"]).values

# Normalize the test data using the same scaling factors as the training data
d_test_normalized =  2 * (X[:, 0]/909)
x_test_normalized = X[:, 1]/909
y_test_normalized = X[:, 2]/909

X_normalized = np.column_stack((d_test_normalized, x_test_normalized, y_test_normalized))

# Print the normalized test data
print(X_normalized)

num_test_samples = X_normalized.shape[0]

l = []
errors = []

# Make predictions on the normalized test data
for i in range(num_test_samples):
    pred = sm.predict_values(np.array([X_normalized[i]]))
    print(pred)
    l.append(pred[0][0])
    error = (abs((pred - df2['m'][i]) / df2['m'][i]) * 100)
    print("Error = %s" %(error))
    errors.append(error)

for i in range(num_test_samples):
    print(errors[i])

truth = [i for i in df2["m"]]
print("ground truth:" + str(truth))
print("predictions:" + str(l))

r2 = r2_score(l, truth)
r2 = round(r2,3)
print(r2) 

def obj(X):
    pred = sm.predict_values(np.array([X]))
    obj = pred[0][0]
    return obj

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


#--------------------------------------------------------------Validation Plot --------------------------------------------------------------------------

plt.figure(figsize=(6.4,4.8),dpi=125)

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.linewidth"] = 2

a, b = np.polyfit(l, truth, 1)

t = np.linspace(0,1,20)
plt.text(0.11, 0.33, '$R^2$ score = ' + str(r2), fontsize = 17)

plt.plot(t, a*t+b,c = "black",lw = "3" )

plt.scatter(l,truth,c = "red",s=50)
plt.xlim(0.1, 0.35)
plt.ylim(0.1, 0.35)

plt.xlabel("Predicted $C_{p}$ [-]",fontsize=20)
plt.ylabel("Actual $C_{p}$ [-]",fontsize=20)
plt.tick_params(axis='both',size=8,labelsize=16, direction='inout')
plt.show()

#--------------------------------------------------------------3D Response Surface-------------------------------------------------------------------

# Create a 3D contour plot
# Generate a grid of r_test_normalized, x_test_normalized, and y_test_normalized values
r_values = np.linspace(np.min(d_test_normalized), np.max(d_test_normalized), 100)
x_values = np.linspace(np.min(x_test_normalized), np.max(x_test_normalized), 100)
y_values = np.linspace(np.min(y_test_normalized), np.max(y_test_normalized), 100)
r_grid, x_grid, y_grid = np.meshgrid(r_values, x_values, y_values)
X_contour = np.column_stack((r_grid.ravel(), x_grid.ravel(), y_grid.ravel()))

# Make predictions on the contour grid
contour_pred = sm.predict_values(X_contour)

# Reshape the predictions for contour plotting
contour_pred = contour_pred.reshape(r_grid.shape)

# Create a 3D plot
plt.rcParams["font.family"] = "Times New Roman"
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
ax.zaxis.set_tick_params(labelsize=12)
plt.tick_params(axis='both',size=12,direction='inout')

# Plot the 3D surface with the predicted Cp values as the colormap
#surf = ax.plot_trisurf(y_grid.ravel(), r_grid.ravel(), contour_pred.ravel(), cmap='turbo', edgecolor='none', linewidth=6)
surf = ax.plot_trisurf(x_grid.ravel(), r_grid.ravel(), contour_pred.ravel(), cmap='turbo', edgecolor='none')
#surf = ax.plot_trisurf(x_grid.ravel(), y_grid.ravel(), contour_pred.ravel(), cmap='turbo', edgecolor='none')

# Create a separate axes for the colorbar and position it to the left
cbar_ax = fig.add_axes([0.15, 0.2, 0.02, 0.6])  # [left, bottom, width, height]

# Add the colorbar to the separate axes
fig.colorbar(surf, cax=cbar_ax)

# Scatter plot of the data points
#ax.set_xlabel('Ly/D' ,fontsize=12,labelpad=12)
ax.set_ylabel('Dc/D' ,fontsize=12,labelpad=12)
#ax.set_ylabel('Ly/D' ,fontsize=12,labelpad=12)
ax.set_xlabel('Lx/D' ,fontsize=12, labelpad=12)
ax.set_zlabel('Objective Function Cp [-]' ,fontsize=12,labelpad=12)
#ax.set_title('3D Contour Plot of Predicted Cp')

plt.show()