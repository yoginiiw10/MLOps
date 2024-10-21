import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import pickle

# Load the CSV file into a DataFrame
df = pd.read_csv('C:\\Users\\yogin\\Desktop\\MLOps\\GiveMeSomeCredit-training.csv')

# Display the first few rows of the dataset
print(df)

print(df.isnull().sum())

#------------------------------------------------------------------------------------------------------------------


if df is not None:
    # Check if 'MONTHLY INCOME' column exists
    if 'MonthlyIncome' in df.columns:
        # Calculate the median of the 'MONTHLY INCOME' column
        median_income = df['MonthlyIncome'].median()

        # Replace NA values with the median income
        df['MonthlyIncome'].fillna(median_income, inplace=True)

        # Print the updated DataFrame
        print("DataFrame after replacing NA with median income:")
        print(df)
    else:
        print("'MonthlyIncome' column does not exist in the DataFrame.")
else:
    print("DataFrame 'df' is not loaded correctly.")

print(df.isnull().sum())

#-------------------------------------------------------------------------------------------------------------------
# Replace NA values in the 'NumberOfDependents' column with 0
df['NumberOfDependents'].fillna(0, inplace=True)

# Print the updated DataFrame to verify changes
print("DataFrame after replacing NA in 'NumberOfDependents' with 0:")
print(df)
print(df.isnull().sum())

#-------------------------------------------------------------------------------------------------------------------

# List of columns to scale
scale_columns = ['RevolvingUtilizationOfUnsecuredLines', 'DebtRatio']

# Apply MinMaxScaler
scaler = MinMaxScaler()
df[scale_columns] = scaler.fit_transform(df[scale_columns])

#-------------------------------------------------------------------------------------------------------------------

# Separate features (X) and the target (y)
X = df.drop(columns=['SeriousDlqin2yrs'])
y = df['SeriousDlqin2yrs']

# Perform train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#-------------------------------------------------------------------------------------------------------------------

# Initialize the model
model = LogisticRegression(max_iter=1000)

# Train the model
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

#-------------------------------------------------------------------------------------------------------------------

# Evaluate the model
print(classification_report(y_test, y_pred))

#-------------------------------------------------------------------------------------------------------------------

# Assuming the DataFrame is named 'df'
new_column_names = {
    'SeriousDlqin2yrs': 'SeriousDelinquencyInLast2Years',
    'RevolvingUtilizationOfUnsecuredLines': 'CreditUtilizationPercentage',
    'age': 'BorrowerAge',
    'NumberOfTime30-59DaysPastDueNotWorse': 'Times30-59DaysLate',
    'DebtRatio': 'DebtToIncomeRatio',
    'MonthlyIncome': 'IncomeMonthly',
    'NumberOfOpenCreditLinesAndLoans': 'OpenCreditLinesAndLoansCount',
    'NumberOfTimes90DaysLate': 'Times90DaysLate',
    'NumberRealEstateLoansOrLines': 'RealEstateLoansCount',
    'NumberOfTime60-89DaysPastDueNotWorse': 'Times60-89DaysLate',
    'NumberOfDependents': 'DependentsCount'
}

# Renaming columns
df = df.rename(columns=new_column_names)

# Checking the renamed columns
print(df.head())  # Display first few rows to verify

print(df.isnull().sum())

#-------------------------------------------------------------------------------------------------------------------

with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)

#-------------------------------------------------------------------------------------------------------------------