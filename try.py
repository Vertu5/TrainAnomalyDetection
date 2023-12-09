# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.utils import resample
from sklearn.decomposition import PCA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import plotly.express as px
import dash
from dash import dcc, html
from dash.dependencies import Input, Output

# Load the dataset
file_path = "ar41_for_ulb_mini.csv"  
df = pd.read_csv(file_path, sep=';')

# Display the first few rows of the dataset
print("Dataset Overview:")
print(df.head())

# Display basic information about the dataset
print("\nDataset Information:")
print(df.info())

# Summary statistics for numerical columns
print("\nSummary Statistics:")
print(df.describe())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Explore unique values in categorical columns
print("\nUnique Values:")
for column in df.select_dtypes(include='object').columns:
    print(f"{column}: {df[column].unique()}")

# Visualize the distribution of numerical features
plt.figure(figsize=(10, 6))
sns.histplot(df['RS_E_InAirTemp_PC1'], bins=20, kde=True)
plt.title('Distribution of RS_E_InAirTemp_PC1')
plt.xlabel('RS_E_InAirTemp_PC1')
plt.ylabel('Frequency')
plt.show()

# Correlation heatmap for numerical features
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()

# Pair plot for selected numerical features
numerical_features = ['lat', 'lon', 'RS_E_InAirTemp_PC1', 'RS_E_OilPress_PC1', 'RS_E_RPM_PC1']
sns.pairplot(df[numerical_features])
plt.suptitle('Pair Plot of Selected Numerical Features', y=1.02)
plt.show()

# Time series plot for selected features
plt.figure(figsize=(14, 6))
plt.plot(df['timestamps_UTC'], df['RS_E_InAirTemp_PC1'], label='RS_E_InAirTemp_PC1')
plt.plot(df['timestamps_UTC'], df['RS_E_OilPress_PC1'], label='RS_E_OilPress_PC1')
plt.title('Time Series Plot of Selected Features')
plt.xlabel('Timestamp')
plt.ylabel('Values')
plt.legend()
plt.show()

# Convert 'timestamps_UTC' to datetime format
df['timestamps_UTC'] = pd.to_datetime(df['timestamps_UTC'])

# Handle missing values (fill with mean for demonstration)
df.fillna(df.mean(), inplace=True)

# Handling outliers using Z-score
from scipy.stats import zscore
z_scores = zscore(df['RS_E_InAirTemp_PC1'])
outliers = (z_scores > 3) | (z_scores < -3)
df = df[~outliers]

# Feature Engineering
df['hour_of_day'] = df['timestamps_UTC'].dt.hour
df.interpolate(method='linear', inplace=True)

# Encoding categorical variables
label_encoder = LabelEncoder()
df['mapped_veh_id'] = label_encoder.fit_transform(df['mapped_veh_id'])

# Scaling numerical features
scaler = StandardScaler()
numerical_columns = ['lat', 'lon', 'RS_E_InAirTemp_PC1', 'RS_E_OilPress_PC1', 'RS_E_RPM_PC1']
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

# Extract time-related features from the timestamp
df['day_of_week'] = df['timestamps_UTC'].dt.dayofweek
df['day_of_month'] = df['timestamps_UTC'].dt.day
df['month'] = df['timestamps_UTC'].dt.month
df['year'] = df['timestamps_UTC'].dt.year

# Lag features for time series analysis
for feature in ['RS_E_InAirTemp_PC1', 'RS_E_OilPress_PC1', 'RS_E_RPM_PC1']:
    df[f'{feature}_lag_1'] = df[feature].shift(1)
    df[f'{feature}_rolling_mean'] = df[feature].rolling(window=3).mean()

# Resample to balance the number of normal and anomaly instances
normal_instances = df[df['label'] == 0]
anomaly_instances = df[df['label'] == 1]
anomaly_upsampled = resample(anomaly_instances, replace=True, n_samples=len(normal_instances), random_state=42)
df_balanced = pd.concat([normal_instances, anomaly_upsampled])

# Use dimensionality reduction techniques if needed (e.g., PCA)
pca = PCA(n_components=2)
df_pca = pd.DataFrame(pca.fit_transform(df_balanced.drop(['label', 'timestamps_UTC'], axis=1)), columns=['PCA1', 'PCA2'])
df_balanced_pca = pd.concat([df_pca, df_balanced[['label']]], axis=1)

# Split the dataset into training and testing sets
X = df_balanced.drop(['label', 'timestamps_UTC'], axis=1)
y = df_balanced['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Pipeline for preprocessing and model training
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features)
    ])

# Parameter grid for hyperparameter tuning
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_samples': ['auto', 100, 200],
    'contamination': [0.05, 0.1, 0.2]
}

# Grid search for Isolation Forest
if_model = GridSearchCV(IsolationForest(random_state=42), param_grid=param_grid, scoring='roc_auc', cv=3)
if_model.fit(X_train, y_train)

# Best parameters for Isolation Forest
best_params_if = if_model.best_params_

# Predictions using the best Isolation Forest model
y_pred_if = if_model.predict(X_test)
y_pred_if = np.where(y_pred_if == -1, 1, 0)

# Evaluate Isolation Forest
print("\nIsolation Forest Metrics:")
print(confusion_matrix(y_test, y_pred_if))
print(classification_report(y_test, y_pred_if))
print("ROC AUC Score:", roc_auc_score(y_test, y_pred_if))

# Visualize feature importances from the best Isolation Forest model
feature_importances_if = if_model.best_estimator_.feature_importances_
features_if = X_train.columns
plt.figure(figsize=(10, 6))
plt.bar(features_if, feature_importances_if)
plt.title('Feature Importances - Isolation Forest')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.xticks(rotation=45)
plt.show()

# Grid search for Local Outlier Factor
lof_model = GridSearchCV(LocalOutlierFactor(novelty=True), param_grid={'contamination': [0.05, 0.1, 0.2]}, scoring='roc_auc', cv=3)
lof_model.fit(X_train, y_train)

# Best parameters for Local Outlier Factor
best_params_lof = lof_model.best_params_

# Predictions using the best Local Outlier Factor model
y_pred_lof = lof_model.predict(X_test)
y_pred_lof = np.where(y_pred_lof == -1, 1, 0)

# Evaluate Local Outlier Factor
print("\nLocal Outlier Factor Metrics:")
print(confusion_matrix(y_test, y_pred_lof))
print(classification_report(y_test, y_pred_lof))
print("ROC AUC Score:", roc_auc_score(y_test, y_pred_lof))

# Grid search for One-Class SVM
svm_model = GridSearchCV(OneClassSVM(), param_grid={'nu': [0.05, 0.1, 0.2]}, scoring='roc_auc', cv=3)
svm_model.fit(X_train, y_train)

# Best parameters for One-Class SVM
best_params_svm = svm_model.best_params_

# Predictions using the best One-Class SVM model
y_pred_svm = svm_model.predict(X_test)
y_pred_svm = np.where(y_pred_svm == -1, 1, 0)

# Evaluate One-Class SVM
print("\nOne-Class SVM Metrics:")
print(confusion_matrix(y_test, y_pred_svm))
print(classification_report(y_test, y_pred_svm))
print("ROC AUC Score:", roc_auc_score(y_test, y_pred_svm))

# Neural Network Autoencoder
input_dim = X_train.shape[1]
ae_model = Sequential([
    Dense(32, activation='relu', input_dim=input_dim),
    Dense(16, activation='relu'),
    Dense(8, activation='relu'),
    Dense(16, activation='relu'),
    Dense(32, activation='relu'),
    Dense(input_dim, activation='linear')
])
ae_model.compile(optimizer='adam', loss='mean_squared_error')
ae_model.fit(X_train, X_train, epochs=20, batch_size=32, shuffle=True, validation_data=(X_test, X_test))

# Evaluate Autoencoder on the test set
X_pred_ae = ae_model.predict(X_test)
mse_ae = np.mean(np.power(X_test - X_pred_ae, 2), axis=1)
y_pred_ae = np.where(mse_ae > mse_ae.mean() + 3 * mse_ae.std(), 1, 0)

# Evaluate Autoencoder
print("\nAutoencoder Metrics:")
print(confusion_matrix(y_test, y_pred_ae))
print(classification_report(y_test, y_pred_ae))
print("ROC AUC Score:", roc_auc_score(y_test, y_pred_ae))

# Dashboard Development with Dash and Plotly
app = dash.Dash(__name__)

# Define layout and callback functions
app.layout = html.Div([
    html.H1("Anomaly Detection Dashboard"),
    
    dcc.Graph(
        id='time-series-plot',
        figure={
            'data': [
                {'x': df['timestamps_UTC'], 'y': df['RS_E_InAirTemp_PC1'], 'type': 'line', 'name': 'RS_E_InAirTemp_PC1'},
                {'x': df['timestamps_UTC'], 'y': df['RS_E_OilPress_PC1'], 'type': 'line', 'name': 'RS_E_OilPress_PC1'}
            ],
            'layout': {
                'title': 'Time Series Plot of Selected Features',
                'xaxis': {'title': 'Timestamp'},
                'yaxis': {'title': 'Values'},
            }
        }
    ),
    
    dcc.Graph(
        id='correlation-heatmap',
        figure={
            'data': [
                {
                    'z': df.corr().values,
                    'x': df.corr().columns,
                    'y': df.corr().index,
                    'type': 'heatmap',
                    'colorscale': 'Viridis'
                }
            ],
            'layout': {
                'title': 'Correlation Heatmap',
                'xaxis': {'title': 'Features'},
                'yaxis': {'title': 'Features'},
            }
        }
    ),
    
    dcc.Graph(
        id='scatter-plot-pca',
        figure=px.scatter(df_balanced_pca, x='PCA1', y='PCA2', color='label', title='PCA Plot')
    ),
    
    dcc.Graph(
        id='feature-importance-if',
        figure={
            'data': [
                {'x': features_if, 'y': feature_importances_if, 'type': 'bar', 'name': 'Feature Importances - Isolation Forest'}
            ],
            'layout': {
                'title': 'Feature Importances - Isolation Forest',
                'xaxis': {'title': 'Features'},
                'yaxis': {'title': 'Importance'},
            }
        }
    ),
    
    html.Div([
        html.H2("Model Evaluation Metrics"),
        html.Pre(f"Isolation Forest Metrics:\n{confusion_matrix(y_test, y_pred_if)}\n{classification_report(y_test, y_pred_if)}\nROC AUC Score: {roc_auc_score(y_test, y_pred_if)}"),
        html.Pre(f"Local Outlier Factor Metrics:\n{confusion_matrix(y_test, y_pred_lof)}\n{classification_report(y_test, y_pred_lof)}\nROC AUC Score: {roc_auc_score(y_test, y_pred_lof)}"),
        html.Pre(f"One-Class SVM Metrics:\n{confusion_matrix(y_test, y_pred_svm)}\n{classification_report(y_test, y_pred_svm)}\nROC AUC Score: {roc_auc_score(y_test, y_pred_svm)}"),
        html.Pre(f"Autoencoder Metrics:\n{confusion_matrix(y_test, y_pred_ae)}\n{classification_report(y_test, y_pred_ae)}\nROC AUC Score: {roc_auc_score(y_test, y_pred_ae)}"),
    ]),
])

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
