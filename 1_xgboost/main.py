import os
import time
import logging
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from datetime import datetime
import psutil
import json
import warnings
warnings.filterwarnings('ignore')

# Try to import GPUtil, but handle the case if it's not available
try:
    import GPUtil
    HAS_GPUTIL = True
except ImportError:
    HAS_GPUTIL = False
    logging.warning("GPUtil not available. GPU information will not be displayed.")

# Setup directories
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), "playground-series-s5e6")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
LOG_DIR = os.path.join(BASE_DIR, "logs")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Create directories if they don't exist
for directory in [OUTPUT_DIR, LOG_DIR, MODEL_DIR]:
    os.makedirs(directory, exist_ok=True)

# Setup logging
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(LOG_DIR, f"xgboost_training_{current_time}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('xgboost_fertilizer')

def log_system_info():
    """Log system information including CPU, memory, and GPU usage."""
    # CPU Information
    cpu_percent = psutil.cpu_percent(interval=1)
    cpu_count = psutil.cpu_count()
    cpu_freq = psutil.cpu_freq()
    
    # Memory Information
    memory = psutil.virtual_memory()
    memory_total = memory.total / (1024 ** 3)  # Convert to GB
    memory_used = memory.used / (1024 ** 3)    # Convert to GB
    memory_percent = memory.percent
    
    # GPU Information
    if HAS_GPUTIL:
        try:
            gpus = GPUtil.getGPUs()
            gpu_info = []
            for gpu in gpus:
                gpu_info.append({
                    'id': gpu.id,
                    'name': gpu.name,
                    'memory_total': gpu.memoryTotal,
                    'memory_used': gpu.memoryUsed,
                    'load': gpu.load * 100
                })
        except Exception as e:
            gpu_info = f"Error getting GPU info: {str(e)}"
    else:
        gpu_info = "GPUtil not available. Install setuptools or use Python 3.11 or earlier for GPU monitoring."
    
    # Log everything
    logger.info("System Information:")
    logger.info(f"CPU Cores: {cpu_count}")
    if cpu_freq:
        logger.info(f"CPU Frequency: {cpu_freq.current:.2f} MHz")
    logger.info(f"CPU Usage: {cpu_percent:.2f}%")
    logger.info(f"Memory: {memory_used:.2f}GB / {memory_total:.2f}GB ({memory_percent:.2f}%)")
    logger.info(f"GPU Info: {json.dumps(gpu_info, indent=2) if isinstance(gpu_info, list) else gpu_info}")

def load_data():
    """Load and preprocess the data."""
    logger.info("Loading data...")
    start_time = time.time()
    
    # Load train and test data
    train_path = os.path.join(DATA_DIR, "train.csv")
    test_path = os.path.join(DATA_DIR, "test.csv")
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    logger.info(f"Train data shape: {train_df.shape}")
    logger.info(f"Test data shape: {test_df.shape}")
    
    # Print column names for debugging
    logger.info(f"Train columns: {train_df.columns.tolist()}")
    logger.info(f"Test columns: {test_df.columns.tolist()}")
    
    logger.info(f"Data loaded in {time.time() - start_time:.2f} seconds")
    return train_df, test_df

def preprocess_data(train_df, test_df):
    """Preprocess the data for modeling."""
    logger.info("Preprocessing data...")
    start_time = time.time()
    
    # Make a copy to avoid modifying the original dataframes
    train = train_df.copy()
    test = test_df.copy()
    
    # Check for missing values
    logger.info(f"Missing values in train: {train.isnull().sum().sum()}")
    logger.info(f"Missing values in test: {test.isnull().sum().sum()}")
    
    # Fix column names with extra spaces
    train.columns = train.columns.str.strip()
    test.columns = test.columns.str.strip()
    
    # Log column names after stripping whitespace
    logger.info(f"Train columns after strip: {train.columns.tolist()}")
    logger.info(f"Test columns after strip: {test.columns.tolist()}")
    
    # Log column dtypes
    logger.info(f"Train dtypes: {train.dtypes}")
    logger.info(f"Test dtypes: {test.dtypes}")
    
    # Save the target variable
    if 'Fertilizer Name' in train.columns:
        y = train['Fertilizer Name']
    else:
        raise ValueError("'Fertilizer Name' column not found in training data")
    
    # Encode all object type columns
    encoders = {}
    
    # Identify and encode all categorical columns (any column of type 'object')
    categorical_cols = []
    for col in train.columns:
        if train[col].dtype == 'object' and col != 'Fertilizer Name' and col != 'id':
            categorical_cols.append(col)
            
    logger.info(f"Categorical columns to encode: {categorical_cols}")
    
    # Ensure test has the same categorical columns
    for col in categorical_cols:
        if col not in test.columns:
            logger.error(f"Column {col} in train but not in test!")
            raise ValueError(f"Column {col} in train but not in test!")
    
    # Encode categorical variables
    for col in categorical_cols:
        encoders[col] = LabelEncoder()
        train[col] = encoders[col].fit_transform(train[col])
        test[col] = encoders[col].transform(test[col])
    
    # Separate features and target
    feature_cols = [col for col in train.columns if col not in ['id', 'Fertilizer Name']]
    X = train[feature_cols]
    X_test = test[feature_cols]
    
    logger.info(f"Features: {feature_cols}")
    
    # Check datatypes after encoding
    logger.info(f"X dtypes after encoding: {X.dtypes}")
    logger.info(f"X_test dtypes after encoding: {X_test.dtypes}")
    
    # Encode the target variable
    target_encoder = LabelEncoder()
    y_encoded = target_encoder.fit_transform(y)
    
    # Save encoders for later use
    encoders['target'] = target_encoder
    
    logger.info(f"Data preprocessed in {time.time() - start_time:.2f} seconds")
    logger.info(f"Features used: {feature_cols}")
    logger.info(f"Number of classes: {len(target_encoder.classes_)}")
    
    return X, y_encoded, X_test, encoders

def train_xgboost_model(X, y, use_gpu=True, n_folds=5):
    """Train an XGBoost model with cross-validation."""
    logger.info("Training XGBoost model...")
    start_time = time.time()
    
    # Determine number of classes
    n_classes = len(np.unique(y))
    logger.info(f"Number of classes: {n_classes}")
    
    # Check and convert any remaining object columns to numeric
    for col in X.columns:
        if X[col].dtype == 'object':
            logger.warning(f"Column {col} is still object type. Converting to numeric.")
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
    
    # Set parameters
    params = {
        'objective': 'multi:softprob',
        'num_class': n_classes,
        'learning_rate': 0.1,
        'max_depth': 6,
        'min_child_weight': 1,
        'gamma': 0,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0,
        'reg_lambda': 1,
        'seed': 42,
        'n_jobs': -1,  # Use all CPU cores
    }
    
    # Use GPU if available and requested
    if use_gpu:
        if HAS_GPUTIL:
            try:
                # Check if GPU is available
                gpus = GPUtil.getGPUs()
                if gpus:
                    logger.info(f"Using GPU for training: {gpus[0].name}")
                    params['tree_method'] = 'gpu_hist'
                    params['gpu_id'] = 0
                else:
                    logger.info("No GPU found, using CPU for training")
            except Exception as e:
                logger.warning(f"Error checking GPU availability: {str(e)}")
                logger.info("Defaulting to CPU training")
        else:
            # Try to use GPU directly without checking availability
            try:
                logger.info("Attempting to use GPU without GPUtil")
                params['tree_method'] = 'gpu_hist'
                params['gpu_id'] = 0
            except Exception as e:
                logger.warning(f"Error setting GPU parameters: {str(e)}")
                logger.info("Defaulting to CPU training")
    
    # Setup cross-validation
    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # For storing models and predictions
    models = []
    oof_preds = np.zeros((X.shape[0], n_classes))
    
    # Train on each fold
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        logger.info(f"Training fold {fold+1}/{n_folds}")
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Create DMatrix for XGBoost
        try:
            dtrain = xgb.DMatrix(X_train, label=y_train)
            dval = xgb.DMatrix(X_val, label=y_val)
        except Exception as e:
            logger.error(f"Error creating DMatrix: {str(e)}")
            logger.error(f"X_train dtypes: {X_train.dtypes}")
            raise
        
        # Setup watchlist for early stopping
        watchlist = [(dtrain, 'train'), (dval, 'eval')]
        
        # Train model
        fold_start_time = time.time()
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=10000,
            evals=watchlist,
            early_stopping_rounds=100,
            verbose_eval=100
        )
        
        # Save the model
        models.append(model)
        model_path = os.path.join(MODEL_DIR, f"xgboost_fold_{fold}.model")
        model.save_model(model_path)
        
        # Make predictions on validation set
        val_preds = model.predict(dval)
        oof_preds[val_idx] = val_preds
        
        # Evaluate fold performance
        val_pred_labels = np.argmax(val_preds, axis=1)
        fold_accuracy = accuracy_score(y_val, val_pred_labels)
        
        logger.info(f"Fold {fold+1} accuracy: {fold_accuracy:.4f}")
        logger.info(f"Fold {fold+1} training time: {time.time() - fold_start_time:.2f} seconds")
    
    # Calculate overall validation performance
    oof_pred_labels = np.argmax(oof_preds, axis=1)
    overall_accuracy = accuracy_score(y, oof_pred_labels)
    logger.info(f"Overall CV accuracy: {overall_accuracy:.4f}")
    logger.info(f"Total training time: {time.time() - start_time:.2f} seconds")
    
    # Return the trained models
    return models

def predict_and_submit(models, X_test, encoders, test_df):
    """Make predictions on the test set and create a submission file."""
    logger.info("Making predictions on test data...")
    start_time = time.time()
    
    # Check and convert any remaining object columns to numeric
    for col in X_test.columns:
        if X_test[col].dtype == 'object':
            logger.warning(f"Column {col} in test is still object type. Converting to numeric.")
            # If we have an encoder for this column, use it
            if col in encoders:
                X_test[col] = encoders[col].transform(X_test[col])
            else:
                # Otherwise, create a new encoder
                le = LabelEncoder()
                X_test[col] = le.fit_transform(X_test[col])
    
    # Convert test data to DMatrix
    try:
        dtest = xgb.DMatrix(X_test)
    except Exception as e:
        logger.error(f"Error creating DMatrix for test: {str(e)}")
        logger.error(f"X_test dtypes: {X_test.dtypes}")
        raise
    
    # Make predictions with each model and average them
    n_classes = len(encoders['target'].classes_)
    test_preds = np.zeros((X_test.shape[0], n_classes))
    
    for i, model in enumerate(models):
        logger.info(f"Predicting with model {i+1}/{len(models)}")
        test_preds += model.predict(dtest)
    
    # Average predictions
    test_preds /= len(models)
    
    # Convert predictions to class labels
    test_pred_labels = np.argmax(test_preds, axis=1)
    test_pred_classes = encoders['target'].inverse_transform(test_pred_labels)
    
    # Create submission DataFrame
    submission = pd.DataFrame({
        'id': test_df['id'],
        'Fertilizer Name': test_pred_classes
    })
    
    # Save submission file
    submission_path = os.path.join(OUTPUT_DIR, "1_xgboost_submission.csv")
    submission.to_csv(submission_path, index=False)
    
    # Copy to parent directory for submission
    parent_submission_path = os.path.join(os.path.dirname(BASE_DIR), "1_xgboost_submission.csv")
    submission.to_csv(parent_submission_path, index=False)
    
    logger.info(f"Predictions made in {time.time() - start_time:.2f} seconds")
    logger.info(f"Submission saved to {submission_path} and {parent_submission_path}")
    
    return submission

def generate_visualizations(X, y, models, encoders, X_test):
    """Generate and save visualizations for model analysis."""
    logger.info("Generating visualizations...")
    start_time = time.time()
    
    # Feature importance
    fig, ax = plt.subplots(figsize=(12, 8))
    xgb.plot_importance(models[0], ax=ax, height=0.8)
    plt.title('XGBoost Feature Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'feature_importance.png'))
    plt.close()
    
    # SHAP values for feature impact
    try:
        explainer = shap.TreeExplainer(models[0])
        shap_values = explainer.shap_values(X)
        
        # Summary plot
        plt.figure(figsize=(12, 10))
        shap.summary_plot(shap_values, X, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'shap_summary.png'))
        plt.close()
        
        # Dependence plots for top features
        feature_importance = models[0].get_score(importance_type='gain')
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:3]
        
        for feature_name, _ in top_features:
            feature_idx = X.columns.get_loc(feature_name)
            plt.figure(figsize=(12, 8))
            shap.dependence_plot(feature_idx, shap_values[0], X, show=False)
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, f'shap_dependence_{feature_name}.png'))
            plt.close()
            
    except Exception as e:
        logger.warning(f"Error generating SHAP plots: {str(e)}")
    
    # Distribution of target classes
    plt.figure(figsize=(12, 8))
    class_names = encoders['target'].classes_
    class_counts = pd.Series(y).map(lambda x: encoders['target'].inverse_transform([x])[0]).value_counts()
    sns.barplot(x=class_counts.index, y=class_counts.values)
    plt.title('Distribution of Fertilizer Classes')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'target_distribution.png'))
    plt.close()
    
    logger.info(f"Visualizations generated in {time.time() - start_time:.2f} seconds")

def main():
    """Main function to run the full pipeline."""
    logger.info("=== Starting XGBoost Fertilizer Recommendation Pipeline ===")
    
    # Log system information
    log_system_info()
    
    # Start timer for total execution
    total_start_time = time.time()
    
    # Load the data
    train_df, test_df = load_data()
    
    # Preprocess the data
    X, y, X_test, encoders = preprocess_data(train_df, test_df)
    
    # Train the model
    models = train_xgboost_model(X, y, use_gpu=True)
    
    # Make predictions and create submission file
    submission = predict_and_submit(models, X_test, encoders, test_df)
    
    # Generate visualizations
    generate_visualizations(X, y, models, encoders, X_test)
    
    # Log total execution time
    logger.info(f"Total execution time: {time.time() - total_start_time:.2f} seconds")
    logger.info("=== XGBoost Fertilizer Recommendation Pipeline Completed ===")

if __name__ == "__main__":
    main()