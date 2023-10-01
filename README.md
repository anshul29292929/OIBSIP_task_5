# OIBSIP_task_5

# Advertising Sales Prediction

This Python script is designed to perform advertising sales prediction using a linear regression model. It analyzes a dataset containing information about advertising spending on TV, radio, and newspapers and predicts the sales based on this data.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Data](#data)
- [Data Preprocessing](#data-preprocessing)
- [Feature Selection](#feature-selection)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Prerequisites

To run this script, you need to have Python installed on your machine, along with the following libraries:
- NumPy
- Pandas
- Seaborn
- Scikit-Learn
- Statsmodels

You can install these libraries using the following command:

```bash
pip install numpy pandas seaborn scikit-learn statsmodels
Installation
Clone the repository or download the script.
Make sure you have the necessary prerequisites installed.
Run the script using a Python interpreter.
Usage
Modify the path to your dataset by changing the csv=pd.read_csv("C:\\Users\\anshu\\Downloads\\archive (5)\\Advertising.csv") line in the script.
Run the script using a Python interpreter.
Data
The dataset used for this project is called "Advertising.csv." It contains the following columns:

TV: Advertising spending on TV.
Radio: Advertising spending on radio.
Newspaper: Advertising spending on newspapers.
Sales: The sales generated as a result of advertising.
Data Preprocessing
The script first loads the dataset and performs initial data exploration using Pandas and Seaborn.
It checks for missing values and drops the "Unnamed: 0" column if present.
It scales the features using StandardScaler to prepare them for modeling.
Feature Selection
The script identifies highly correlated features and removes them iteratively to reduce multicollinearity.
Model Training
The linear regression model is trained on the preprocessed data.
Evaluation
The model's performance is evaluated using metrics like Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R-squared (R²).
The results are printed to the console and saved in a DataFrame.
Results
The script provides insights into the advertising sales prediction model's performance.
Contributing
If you have suggestions or find issues with the code, please feel free to open an issue or submit a pull request.
