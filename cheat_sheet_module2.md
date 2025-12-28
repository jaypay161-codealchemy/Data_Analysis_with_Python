# Cheatsheet: Data Cleaning and Preprocessing

## Overview
This cheatsheet provides a quick reference for data cleaning and preprocessing techniques in Python. It includes key concepts, tools, and practical tips for preparing data for analysis.

## Key Concepts

### 1. Data Cleaning
- **Missing Values**: Handle missing data using techniques like imputation or deletion.
- **Outliers**: Identify and handle outliers using statistical methods or visualization.
- **Duplicates**: Remove duplicate entries to ensure data integrity.
- **Data Types**: Convert data to appropriate types (e.g., datetime, boolean).

### 2. Data Preprocessing
- **Normalization/Standardization**: Scale numerical features to a standard range or distribution.
- **Encoding Categorical Variables**: Convert categorical variables into numerical form using techniques like one-hot encoding or label encoding.
- **Feature Selection**: Select the most relevant features to improve model performance and reduce overfitting.
- **Data Splitting**: Split data into training and testing sets to evaluate model performance.

## Practical Tips

### 1. Handling Missing Values
- Use `df.dropna()` to remove rows with missing values.
- Use `df.fillna()` to fill missing values with a specific value or method (e.g., mean, median).

### 2. Handling Outliers
- Use `df.describe()` to identify outliers based on quartiles.
- Use visualization techniques like box plots to detect outliers.
- Use `df.clip()` to cap outliers at a certain threshold.

### 3. Removing Duplicates
- Use `df.drop_duplicates()` to remove duplicate rows.

### 4. Data Type Conversion
- Use `df.astype()` to convert data types.
- Use `pd.to_datetime()` to convert date strings to datetime objects.

### 5. Encoding Categorical Variables
- Use `pd.get_dummies()` for one-hot encoding.
- Use `LabelEncoder` from `sklearn.preprocessing` for label encoding.

### 6. Feature Scaling
- Use `StandardScaler` from `sklearn.preprocessing` for standardization.
- Use `MinMaxScaler` for normalization.

### 7. Data Splitting
- Use `train_test_split` from `sklearn.model_selection` to split data into training and testing sets.

## Tools and Libraries
- **Pandas**: For data manipulation and analysis.
- **NumPy**: For numerical computing.
- **Scikit-learn**: For preprocessing and model evaluation.
- **Matplotlib and Seaborn**: For data visualization.

## References
- [Pandas Documentation](https://pandas.pydata.org/pandas-docs/stable/)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)
- [Seaborn Documentation](https://seaborn.pydata.org/)

## Conclusion
Data cleaning and preprocessing are essential steps in the data analysis pipeline. By understanding the key concepts, tools, and practical tips, you can effectively prepare your data for analysis and modeling.