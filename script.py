   
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import datetime as dt
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from statsmodels.tsa.deterministic import DeterministicProcess
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
import os


# define a function to read csv file
def reading_function(filename):
    """
    Read the CSV file from the given file path and return the pandas dataframe.
    """
    # get the current working directory
    PATH = os.getcwd()

    # construct the file path
    file_path = os.path.join(PATH, filename)

    # read the csv file into a pandas dataframe
    global temperatures
    temperatures = pd.read_csv(file_path)

    # return the dataframe
    return temperatures

# call the function and pass the filename 'GlobalTemperatures.csv'
df = reading_function('GlobalTemperatures.csv')
    
# This code is used to plot temperature data and analyze it. 

# The first set of code displays the first few rows of the temperature data.
print(temperatures.head())

# This line of code finds the maximum date in the 'dt' column of the temperature data.
print(max(temperatures.dt))

# This line of code displays the data types of each column in the temperature data.
print(temperatures.dtypes)

# This line of code provides a summary of the temperature data, including count, mean, std deviation, min and max.
print(temperatures.describe())

# The following set of code creates a scatter plot of temperature data, where x-axis is date and y-axis is land average temperature.
plt.figure(figsize=(18,10))
plt.scatter(data = temperatures, x = 'dt',y = 'LandAverageTemperature')
plt.xlabel("Year")
plt.ylabel("Temprature")
plt.title("Temperature Trends")
plt.show()

# The next set of code converts the 'dt' column in the temperature data to a datetime format.
# It then extracts the year from the date and maps the date to the ordinal value of the datetime format.
temperatures['Date'] = pd.to_datetime(temperatures.dt, format='%Y-%d-%m')
temperatures['Year'] = temperatures['Date'].dt.year
temperatures['Date'] = temperatures['Date'].map(dt.datetime.toordinal)

# This line of code groups the temperature data by year and calculates the mean land average temperature for each year.
# The results are stored in a new dataframe called 'df'.
df = temperatures.groupby('Year')['LandAverageTemperature'].mean().reset_index()

# This set of code creates a scatter plot of average temperature data by year, where x-axis is year and y-axis is average temperature.
plt.figure(figsize=(18,10))
plt.scatter(data = df, x = 'Year',y = 'LandAverageTemperature')
plt.xlabel("Year")
plt.ylabel("Average Temprature")
plt.title("Average Land Temperature Trends")
plt.show()

# This set of code creates a rolling 10-year average of the land average temperature data.
# It first selects the 'LandAverageTemperature' column of the dataframe 'df' and assigns it to the variable 'temperature_px'.
# It then calculates the rolling 10-year average of the temperature data and adds it as a new column to the dataframe 'df'.
temperature_px = df['LandAverageTemperature']
df['10'] = temperature_px.rolling(window=10).mean()

# The following set of code creates a plot of the land average temperature data and its 10-year rolling average.
# It sets the x-axis tick marks and labels, as well as the x and y axis labels and title.
# It also adds a legend to the plot and displays it.
plt.figure(figsize=(18,10))
ax = plt.subplot()
ax.plot(df['LandAverageTemperature'], alpha=0.8, label='land average temperature')
ax.plot(df['10'], color="orange", label='10-year land average temperature')
ax.set_xticks([0,50,100,150,200,250])
ax.set_xticklabels([1750,1800,1850,1900,1950,2000])
plt.xlabel('Years')
plt.ylabel('Temperature (in °C)')
plt.grid()
plt.legend()
plt.show()
plt.clf()

# This set of code creates a line plot of the land average temperature data over time.
# It sets the x-axis tick marks and labels, as well as the x and y axis labels and title.
# It then displays the plot.
ax = df['LandAverageTemperature'].plot()
ax.set(title="Land Average Temperature per Year in the last 250 years", ylabel="Land Average Temperature")
ax.set_xticks([0,50,100,150,200,250])
ax.set_xticklabels([1750,1800,1850,1900,1950,2000])
plt.show()

# This set of code calculates the 10-year rolling mean of the land average temperature data with some parameters specified.
# The calculated rolling mean values are then assigned to a new variable 'trend'.
trend = df['LandAverageTemperature'].rolling(
    window=10,
    center=True,
    min_periods=6,
).mean()

# The following set of code creates a plot of the land average temperature data with the calculated trend line overlaid.
# It sets the x-axis tick marks and labels, as well as the x and y axis labels and title.
# It then displays the plot.
ax = df['LandAverageTemperature'].plot(alpha=0.5)
ax = trend.plot(ax=ax, linewidth=3)
ax.set(title="Land Average Temperature in the last 250 years", ylabel="Land Average Temperature")
ax.set_xticks([0,50,100,150,200,250])
ax.set_xticklabels([1750,1800,1850,1900,1950,2000])
plt.show()

# The following code creates a scatter plot with linear regression line for the 'Year' and 'LandAverageTemperature' variables.
# It first fits a linear regression model to the variables, then plots the scatter plot with the fitted line.
lr = LinearRegression()

X = df['Year']
y = df['LandAverageTemperature']

X = X.values.reshape(-1,1)

lr.fit(X, y)

y_pred = lr.predict(X)

years = pd.DataFrame(X)

plt.figure(figsize=(18,10))
plt.scatter(X, y, alpha=0.6)
plt.plot(X, y_pred, color="orange")
plt.xlabel('Years')
plt.ylabel('Temperature (in °C)')
plt.show()
plt.clf()

# printing coefficients of the linear regression model
print(lr.coef_)

# multiplying the coefficients by 10 and printing the result
print(10 * lr.coef_)

# predicting the temperature for the years 2030 and 2050
print(lr.predict(np.array([2030, 2050]).reshape(-1,1)))

# calculating Pearson correlation coefficient and p-value for the relationship between Year and LandAverageTemperature
corr, p = pearsonr(df[df['Year'] >= 1850]['Year'], df[df['Year'] >= 1850]['LandAverageTemperature'])

# printing the Pearson correlation coefficient between Year and LandAverageTemperature
print('Pearson correlation of Year and Land Average Temperature: ' + str(corr))

# initializing a new LinearRegression object
lr = LinearRegression()

# extracting relevant data and reshaping it
X = df[df['Year'] >= 1850]['Year']
y = df[df['Year'] >= 1850]['LandAverageTemperature']
X = X.values.reshape(-1,1)

# fitting the linear regression model on the data
lr.fit(X, y)

# making predictions using the fitted model
y_pred = lr.predict(X)

# plotting the data and the predicted values
plt.figure(figsize=(18,10))
plt.scatter(X, y, alpha=0.6)
plt.plot(X, y_pred, color="orange")
plt.xlabel('Years')
plt.ylabel('Temperature (in °C)')
plt.show()
plt.clf()

# printing coefficients of the linear regression model
print(lr.coef_)

# multiplying the coefficients by 10 and printing the result
print(10 * lr.coef_)

# predicting the temperature for the years 2030 and 2050
print(lr.predict(np.array([2030, 2050]).reshape(-1,1)))

# calculating Pearson correlation coefficient and p-value for the relationship between Year and LandAverageTemperature
corr, p = pearsonr(df[df['Year'] >= 1950]['Year'], df[df['Year'] >= 1950]['LandAverageTemperature'])

# printing the Pearson correlation coefficient between Year and LandAverageTemperature
print('Pearson correlation of Year and Land Average Temperature: ' + str(corr))

# initializing a new LinearRegression object
lr = LinearRegression()

# extracting relevant data and reshaping it
X = df[df['Year'] >= 1950]['Year']
y = df[df['Year'] >= 1950]['LandAverageTemperature']
X = X.values.reshape(-1,1)

# fitting the linear regression model on the data
lr.fit(X, y)

# making predictions using the fitted model
y_pred = lr.predict(X)

# plotting the data and the predicted values
plt.figure(figsize=(18,10))
plt.scatter(X, y, alpha=0.6)
plt.plot(X, y_pred, color="orange")
plt.xlabel('Years')
plt.ylabel('Temperature (in °C)')
plt.show()
plt.clf()

# printing coefficients of the linear regression model
print(lr.coef_)

# multiplying the coefficients by 10 and printing the result
print(lr.coef_ * 10)

# predicting the temperature for the years 2030 and 2050
print(lr.predict(np.array([2030, 2050]).reshape(-1,1)))