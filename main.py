from common_import import *

def dataset_overview(df, name):
    print(f'{name.upper()} dataset overview')
    print(f'Structure of {name} dataset')
    print(df.info())
    print('-----------------------------')
    print(f'Statistical of {name} dataset')
    print(df.describe())
    print('-----------------------------')
    print(f'Checking missing value')
    print(df.isnull().sum())
    print()

def show_correlations(df):
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, annot = True)
    plt.show()

def encoder(df):
    df['Sex'] = LabelEncoder().fit_transform(df['Sex'])
    return df

def replace_null_value(df):
    age_mean_value = df['Age'].mean()
    df['Age'] = df['Age'].replace(np.nan, age_mean_value, inplace=True)
    print(df.isnull().sum())

def class_report(y_val, train_predictions):
    class_report = classification_report(train_predictions, y_val)
    print(class_report)

# Dataset Overview
df_train = pd.read_csv('data/train.csv')
df_test = pd.read_csv('data/test.csv')

dataset_overview(df_train, 'Train')
dataset_overview(df_test, 'Test')

# Dataset Preprocessing
encoder(df_train)
encoder(df_test)
replace_null_value(df_train)
replace_null_value(df_test)

# Features selection
X = df_train[['Age', 'Sex']]
y = df_train['Survived']

# Spliting training dataset into train and validation set
X_train, X_val, y_train, y_val = train_test_split(X,y, test_size=0.2, random_state=42)

# Model Training
logistic_regression_model = LogisticRegression(
    C=0.1,
    solver='liblinear',
    penalty='l2',
    random_state=42,
)
logistic_regression_model.fit(X_train, y_train)
joblib.dump(logistic_regression_model, 'models/logistic_regression1.pkl')
lg_predictions = logistic_regression_model.predict(X_val)

random_forest_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
random_forest_model.fit(X_train, y_train)
joblib.dump(logistic_regression_model, 'models/random_forest1.pkl')
rf_predictions = random_forest_model.predict(X_val)

# Classification report
class_report(y_val, lg_predictions)
class_report(y_val, rf_predictions)
# Confusion matrix
cm_lr = confusion_matrix(
    y_val,
    lg_predictions
)

cm_rf = confusion_matrix(
    y_val,
    rf_predictions
)
    # Create 2 panels
fig, axes = plt.subplots(
    1, 2,
    figsize=(12, 5)
)
    # Logistic Regression
sns.heatmap(
    cm_lr,
    annot=True,
    fmt='d',
    cmap='Blues',
    ax=axes[0],
    xticklabels=['Died', 'Survived'],
    yticklabels=['Died', 'Survived']
)

axes[0].set_title('Logistic Regression')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')

# Random Forest
sns.heatmap(
    cm_rf,
    annot=True,
    fmt='d',
    cmap='Blues',
    ax=axes[1],
    xticklabels=['Died', 'Survived'],
    yticklabels=['Died', 'Survived']
)

axes[1].set_title('Random Forest')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')

plt.tight_layout()
plt.show()

from sklearn.model_selection import GridSearchCV
param_grid = {
    'C': [0.01, 0.1, 1, 10],
}

# Model Testing
final_prediction = logistic_regression_model.predict(X_test)

# To submission file
submission_df = pd.DataFrame({
    "PassengerId" : df_test['PassengerId'],
    "Survived": final_prediction
})

submission_df.to_csv('tests/final_result.csv', index= False)
print(submission_df)


