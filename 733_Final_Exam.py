#!/usr/bin/env python
# coding: utf-8

# In[ ]:


Name - Monil Darshan Mehta          Email - monil.mehta@umbc.edu


# In[ ]:


Data preprocessing involves several steps, as outlined below:

Loading data from a CSV file: The numpy.loadtxt() function is utilized to import data from a 
CSV file and store it in a NumPy array.
Indexing and slicing: NumPy array indexing and slicing operations enable the extraction of 
specific rows and columns from the data array.
Array arithmetic: Various arithmetic operations, such as addition, subtraction, multiplication, 
and division, can be performed on the NumPy array to manipulate the data.
Aggregation functions: NumPy provides aggregation functions like numpy.mean(), numpy.max(), 
numpy.min(), numpy.std(), and numpy.sum() to compute statistical quantities 
across the data array or along specific axes.
Reshaping arrays: The numpy.reshape() function allows for modifying the shape of the array, such as 
converting a 1D array into a 2D array.
Combining arrays: NumPy offers functions like numpy.vstack() and numpy.hstack() for vertically or 
horizontally stacking multiple arrays, respectively.
Transposing arrays: The numpy.transpose() function or the .T attribute can be employed to interchange 
the dimensions of an array.
Boolean indexing: Boolean indexing is used to select specific elements or subsets of the array based 
on certain conditions.
Masking arrays: Masking involves applying a Boolean condition to an array and using that mask to extract 
or manipulate specific elements of the array.
Iterating over arrays: NumPy arrays can be iterated over using loops to perform operations on individual 
elements or subsets of the array.
        
These steps are performed to ensure that the data is in a suitable format for analysis or modeling, and to address issues like missing values, outliers, inconsistent data, and other data quality concerns. Each step serves a specific purpose in preparing the data for further analysis.


# In[6]:


import numpy as np
import pandas as pd


# In[7]:


data = pd.read_csv(r'C:\Users\monil\Downloads\python-novice-inflammation-data\data\inflammation-01.csv')


# In[9]:


# Basic operations on the data
print("Shape of the data array:", data.shape)
print("Sum of all elements in the data array:", np.sum(data))
print("Product of all elements in the data array:", np.prod(data))
print("Mean of each row in the data array:", np.mean(data, axis=1))
print("Median of each column in the data array:", np.median(data, axis=0))

# Reshaping the array
reshaped_data = np.reshape(data, (12, 10))
print("Shape of the reshaped data array:", reshaped_data.shape)

# Broadcasting to perform arithmetic operations
subtracted_data = data - 5
print("First element of the subtracted data:", subtracted_data[0, 0])
multiplied_data = data * np.array([1, 2, 3])
print("First row of the multiplied data:", multiplied_data[0, :])

# Sorting and finding unique values
sorted_data = np.sort(data, axis=0)
print("Sorted data along columns:")
print(sorted_data)
unique_values = np.unique(data)
print("Unique values in the data array:")
print(unique_values)

# Statistical functions on specific subsets
column_std_dev = np.std(data[:, 1:4], axis=0)
print("Standard deviation of columns 1 to 3:", column_std_dev)
row_min = np.min(data[2:5, :], axis=1)
print("Minimum values of rows 3 to 5:", row_min)


# In[ ]:


Feature Engineering:

The tutorial primarily focuses on data manipulation and analysis rather than traditional feature engineering techniques. 
However, within the data preprocessing section, there are a few steps that can be considered as basic feature engineering 
operations. These steps include:

Indexing and slicing: Extracting specific rows or columns from the data array allows you to work with relevant subsets 
    of the data. By selecting and manipulating specific features/columns, you can engineer new features or perform calculations 
    based on those selected columns.

Array arithmetic: Applying arithmetic operations to the data array enables the derivation of new features or calculations 
    based on existing features. For instance, doubling the data values (double_data = data * 2) can be seen as a form of 
    feature engineering by scaling the values or emphasizing their magnitude.

Boolean indexing and masking arrays: These operations enable data filtering based on specific conditions. 
    By applying conditions to the data, you can create binary variables or flags that represent particular 
    properties or patterns in the dataset.

While these operations can be considered as basic feature engineering techniques, it's important to note that the primary 
focus of the NumPy tutorial in the provided link lies in data manipulation and analysis rather than advanced feature engineering. 
Advanced feature engineering typically involves more intricate transformations, creation of new variables, handling missing data, 
dealing with categorical variables.


# In[17]:


import numpy as np
data = np.loadtxt(fname=r'C:\Users\monil\Downloads\python-novice-inflammation-data\data\inflammation-01.csv', delimiter=',')
# Indexing and slicing to extract specific rows and columns
subset_data = data[:, 1:4]  # Extract columns 1, 2, and 3
print("Subset of data:")
print(subset_data)

# Array arithmetic to manipulate the data
double_data = data * 2
print("Doubled data:")
print(double_data)

# Boolean indexing and masking arrays
positive_values = data[data > 0]  # Extract positive values
print("Positive values:")
print(positive_values)

negative_mask = data < 0  # Create a mask for negative values
masked_data = np.ma.masked_array(data, negative_mask)  # Apply the mask
print("Masked data:")
print(masked_data)


# In[ ]:


Model Selection - 
In the given tutorial, there is no explicit model selection step mentioned. The tutorial primarily focuses on data 
manipulation and analysis rather than model selection and training. Additionally, the steps of training and validation 
mentioned in the question do not apply to the tutorial.
However, to explore and analyze the dataset further, I have implemented a Linear Regression algorithm. The purpose of 
this implementation is to group and identify possible relationships among the data points. Linear Regression is a suitable 
algorithm for this task as it helps in understanding the linear relationship between the input features and the target variable
By applying the Linear Regression algorithm to the dataset, we can analyze the data points and identify any potential patterns
or correlations. This can provide insights into how the input features contribute to the target variable and help in making 
predictions or understanding the underlying relationships within the data.


# In[23]:


import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load the dataset from CSV using NumPy
data = np.loadtxt(fname=r'C:\Users\monil\Downloads\python-novice-inflammation-data\data\inflammation-01.csv', delimiter=',')

# Extract the feature columns from the dataset
X = data[:, 1:-1]  # Use all columns except the last one as features
y = data[:, -1]   # Use the last column as the target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Calculate the Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)


# In[30]:


import matplotlib.pyplot as plt
# Visualize the actual and predicted values
plt.scatter(y_test, y_pred)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')  # Plotting the diagonal line
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Linear Regression - Actual vs. Predicted')
plt.show()


# In[ ]:


Finally, we create a scatter plot to visualize the Linear Regression model's predictions. The scatter plot shows
the predicted values (y_pred) on the y-axis and the actual values (y_test) on the x-axis. The red dashed line represents 
the diagonal line, indicating perfect predictions. The closer the scatter points are to the diagonal line, the better the
model fits the data.

By visualizing the predicted values against the actual values, we can assess the performance and accuracy of the Linear 
Regression model on the test set.

