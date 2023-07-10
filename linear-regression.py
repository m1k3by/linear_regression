import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the dataset
data = pd.read_csv("C:/Users/StoeckerM/PycharmProjects/linear-regression-MS/CCPP_data.csv")

# Select the input features and target variable
features = ['AT - Temperature', 'V - Exhaust Vacuum', 'AP - Ambient Pressure', 'RH - Relative Humidity']
target = 'PE - Net Hourly Electrical Energy Output'

# Split the dataset into input features (X) and target variable (y)
X = data[features]
y = data[target]

# Split the data into training, validation, and test sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.5, random_state=42)

# ColumnTransformer to preserve feature names
preprocessor = ColumnTransformer(transformers=[('num_features', 'passthrough', features)])

# Select features using SelectKBest
fs = SelectKBest(score_func=f_regression, k=4)

# Fit the ColumnTransformer and SelectKBest on training data
X_train_fs = preprocessor.fit_transform(X_train, y_train)
fs.fit(X_train_fs, y_train)

# Transform validation and test data
X_val_fs = preprocessor.transform(X_val)
X_test_fs = preprocessor.transform(X_test)

# Print feature scores
for i in range(len(fs.scores_)):
    print('Feature %d: %f' % (i, fs.scores_[i]))

# Create an instance of the linear regression model
model = LinearRegression()

# Train the model on the training data
model.fit(X_train_fs, y_train)

# Make predictions on the validation set
y_pred = model.predict(X_val_fs)

# Calculate the mean squared error (MSE) on the validation set
mse = mean_squared_error(y_val, y_pred)
print("Mean Squared Error:", mse)

# Uncomment the following lines to visualize the feature scores
plt.bar(range(len(fs.scores_)), fs.scores_)
plt.show()