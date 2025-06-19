import os
import time
import logging
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, make_scorer
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import psutil
import GPUtil
import json
import warnings
warnings.filterwarnings('ignore')

# Setup directories
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), "playground-series-s5e6")
LOG_DIR = os.path.join(BASE_DIR, "logs")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# Create directories if they don't exist
for directory in [LOG_DIR, OUTPUT_DIR]:
    os.makedirs(directory, exist_ok=True)

# Setup logging
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(LOG_DIR, f"hyperparameter_tuning_{current_time}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('xgboost_hyperparameter_tuning')

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
    
    # Load train data
    train_path = os.path.join(DATA_DIR, "train.csv")
    train_df = pd.read_csv(train_path)
    
    logger.info(f"Train data shape: {train_df.shape}")
    logger.info(f"Data loaded in {time.time() - start_time:.2f} seconds")
    return train_df

def preprocess_data(train_df):
    """Preprocess the data for modeling."""
    logger.info("Preprocessing data...")
    start_time = time.time()
    
    # Make a copy to avoid modifying the original dataframe
    train = train_df.copy()
    
    # Check for missing values
    logger.info(f"Missing values in train: {train.isnull().sum().sum()}")
    
    # Fix column names with extra spaces
    train.columns = train.columns.str.strip()
    
    # Save the target variable
    y = train['Fertilizer Name']
    
    # Encode categorical variables
    categorical_cols = ['Soil Type', 'CropType']
    encoders = {}
    
    for col in categorical_cols:
        encoders[col] = LabelEncoder()
        train[col] = encoders[col].fit_transform(train[col])
    
    # Separate features and target
    feature_cols = [col for col in train.columns if col not in ['id', 'Fertilizer Name']]
    X = train[feature_cols]
    
    # Encode the target variable
    target_encoder = LabelEncoder()
    y_encoded = target_encoder.fit_transform(y)
    
    # Save encoders for later use
    encoders['target'] = target_encoder
    
    logger.info(f"Data preprocessed in {time.time() - start_time:.2f} seconds")
    logger.info(f"Features used: {feature_cols}")
    logger.info(f"Number of classes: {len(target_encoder.classes_)}")
    
    return X, y_encoded, encoders

def tune_hyperparameters(X, y, use_gpu=True):
    """Tune XGBoost hyperparameters using RandomizedSearchCV."""
    logger.info("Tuning hyperparameters...")
    start_time = time.time()
    
    # Determine number of classes
    n_classes = len(np.unique(y))
    logger.info(f"Number of classes: {n_classes}")
    
    # Define the parameter grid
    param_grid = {
        'max_depth': [3, 4, 5, 6, 7, 8],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'n_estimators': [100, 200, 300, 500],
        'min_child_weight': [1, 3, 5, 7],
        'gamma': [0, 0.1, 0.2, 0.3],
        'subsample': [0.6, 0.7, 0.8, 0.9],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
        'reg_alpha': [0, 0.001, 0.01, 0.1, 1],
        'reg_lambda': [0, 0.001, 0.01, 0.1, 1]
    }
    
    # Set up XGBoost classifier
    xgb_model = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=n_classes,
        n_jobs=-1,  # Use all CPU cores
        random_state=42
    )
    
    # Use GPU if available and requested
    if use_gpu:
        try:
            # Check if GPU is available
            gpus = GPUtil.getGPUs()
            if gpus:
                logger.info(f"Using GPU for training: {gpus[0].name}")
                xgb_model.set_params(tree_method='gpu_hist', gpu_id=0)
            else:
                logger.info("No GPU found, using CPU for training")
        except Exception as e:
            logger.warning(f"Error checking GPU availability: {str(e)}")
            logger.info("Defaulting to CPU training")
    
    # Set up cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Set up RandomizedSearchCV
    search = RandomizedSearchCV(
        xgb_model,
        param_grid,
        n_iter=50,  # Number of parameter combinations to try
        scoring='accuracy',
        cv=cv,
        verbose=1,
        n_jobs=-1,  # Use all CPU cores
        random_state=42
    )
    
    # Perform hyperparameter tuning
    search.fit(X, y)
    
    # Log results
    logger.info(f"Best parameters: {search.best_params_}")
    logger.info(f"Best cross-validation score: {search.best_score_:.4f}")
    
    # Save best parameters to a JSON file
    best_params = search.best_params_
    best_params_path = os.path.join(OUTPUT_DIR, "best_parameters.json")
    with open(best_params_path, 'w') as f:
        json.dump(best_params, f, indent=4)
    
    logger.info(f"Best parameters saved to {best_params_path}")
    logger.info(f"Hyperparameter tuning completed in {time.time() - start_time:.2f} seconds")
    
    # Create visualization of parameter importance
    # Get feature importances from the best estimator
    results = pd.DataFrame(search.cv_results_)
    
    # Plot the relationship between important hyperparameters and score
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()
    
    params_to_plot = [
        'param_max_depth', 'param_learning_rate', 'param_n_estimators',
        'param_min_child_weight', 'param_gamma', 'param_subsample',
        'param_colsample_bytree', 'param_reg_alpha', 'param_reg_lambda'
    ]
    
    for i, param in enumerate(params_to_plot):
        if i < len(axes):
            if param in results.columns:
                # Convert parameter values to numeric if possible
                try:
                    param_values = results[param].astype(float)
                    sns.scatterplot(x=param_values, y=results['mean_test_score'], ax=axes[i])
                    axes[i].set_title(f'Score vs {param.replace("param_", "")}')
                except:
                    logger.warning(f"Could not convert {param} to numeric for plotting")
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'hyperparameter_tuning_results.png'))
    plt.close()
    
    return best_params

def main():
    """Main function to run the hyperparameter tuning pipeline."""
    logger.info("=== Starting XGBoost Hyperparameter Tuning ===")
    
    # Log system information
    log_system_info()
    
    # Start timer for total execution
    total_start_time = time.time()
    
    # Load the data
    train_df = load_data()
    
    # Preprocess the data
    X, y, encoders = preprocess_data(train_df)
    
    # Tune hyperparameters
    best_params = tune_hyperparameters(X, y, use_gpu=True)
    
    # Log total execution time
    logger.info(f"Total execution time: {time.time() - total_start_time:.2f} seconds")
    logger.info("=== XGBoost Hyperparameter Tuning Completed ===")
    
    # Return best parameters
    return best_params

if __name__ == "__main__":
    main()