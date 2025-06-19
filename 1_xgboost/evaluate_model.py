import os
import time
import logging
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

# Setup directories
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), "playground-series-s5e6")
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
LOG_DIR = os.path.join(BASE_DIR, "logs")

# Create directories if they don't exist
for directory in [MODEL_DIR, OUTPUT_DIR, LOG_DIR]:
    os.makedirs(directory, exist_ok=True)

# Setup logging
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(LOG_DIR, f"model_evaluation_{current_time}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('xgboost_evaluation')

def load_data():
    """Load the data."""
    logger.info("Loading data...")
    start_time = time.time()
    
    # Load train data
    train_path = os.path.join(DATA_DIR, "train.csv")
    train_df = pd.read_csv(train_path)
    
    logger.info(f"Train data shape: {train_df.shape}")
    logger.info(f"Data loaded in {time.time() - start_time:.2f} seconds")
    return train_df

def preprocess_data(train_df):
    """Preprocess the data for evaluation."""
    logger.info("Preprocessing data...")
    start_time = time.time()
    
    # Make a copy to avoid modifying the original dataframe
    train = train_df.copy()
    
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
    
    return X, y_encoded, encoders, feature_cols

def load_models():
    """Load the trained XGBoost models."""
    logger.info("Loading models...")
    start_time = time.time()
    
    models = []
    model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith('.model')]
    
    if not model_files:
        logger.error("No model files found in the models directory!")
        return None
    
    for model_file in model_files:
        model_path = os.path.join(MODEL_DIR, model_file)
        model = xgb.Booster()
        model.load_model(model_path)
        models.append(model)
        logger.info(f"Loaded model from {model_path}")
    
    logger.info(f"Loaded {len(models)} models in {time.time() - start_time:.2f} seconds")
    return models

def evaluate_models(X, y, models, encoders, feature_cols):
    """Evaluate the models using cross-validation."""
    logger.info("Evaluating models...")
    start_time = time.time()
    
    if not models:
        logger.error("No models to evaluate!")
        return
    
    # Determine number of classes
    n_classes = len(encoders['target'].classes_)
    
    # Setup cross-validation
    kf = StratifiedKFold(n_splits=len(models), shuffle=True, random_state=42)
    
    # For storing predictions
    oof_preds = np.zeros((X.shape[0], n_classes))
    
    # Evaluate on each fold
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        if fold >= len(models):
            break
            
        logger.info(f"Evaluating fold {fold+1}/{len(models)}")
        X_val = X.iloc[val_idx]
        y_val = y[val_idx]
        
        # Create DMatrix for XGBoost
        dval = xgb.DMatrix(X_val)
        
        # Make predictions
        val_preds = models[fold].predict(dval)
        oof_preds[val_idx] = val_preds
        
        # Evaluate fold performance
        val_pred_labels = np.argmax(val_preds, axis=1)
        fold_accuracy = accuracy_score(y_val, val_pred_labels)
        
        logger.info(f"Fold {fold+1} accuracy: {fold_accuracy:.4f}")
        
        # Generate classification report
        class_names = encoders['target'].classes_
        class_report = classification_report(y_val, val_pred_labels, target_names=class_names)
        logger.info(f"Fold {fold+1} classification report:\n{class_report}")
    
    # Calculate overall validation performance
    oof_pred_labels = np.argmax(oof_preds, axis=1)
    overall_accuracy = accuracy_score(y, oof_pred_labels)
    logger.info(f"Overall CV accuracy: {overall_accuracy:.4f}")
    
    # Generate overall classification report
    class_names = encoders['target'].classes_
    overall_report = classification_report(y, oof_pred_labels, target_names=class_names)
    logger.info(f"Overall classification report:\n{overall_report}")
    
    # Generate confusion matrix
    cm = confusion_matrix(y, oof_pred_labels)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'))
    plt.close()
    
    # Calculate mean average precision (MAP)
    def calculate_map(y_true, y_pred_probs, k=5):
        """Calculate Mean Average Precision @ k."""
        n_samples = len(y_true)
        n_classes = y_pred_probs.shape[1]
        
        # For each sample, get the top k predictions
        top_k_indices = np.argsort(-y_pred_probs, axis=1)[:, :k]
        
        # Calculate average precision for each sample
        avg_precision = []
        for i in range(n_samples):
            true_class = y_true[i]
            top_preds = top_k_indices[i]
            
            # Check if true class is in top k predictions
            if true_class in top_preds:
                # Find position of true class in top k predictions (0-indexed)
                position = np.where(top_preds == true_class)[0][0]
                # Calculate precision at that position
                precision = 1.0 / (position + 1)
                avg_precision.append(precision)
            else:
                avg_precision.append(0)
        
        # Calculate mean average precision
        map_score = np.mean(avg_precision)
        return map_score
    
    # Calculate MAP for k=1 to k=5
    for k in range(1, 6):
        map_score = calculate_map(y, oof_preds, k=k)
        logger.info(f"Mean Average Precision @ {k}: {map_score:.4f}")
    
    logger.info(f"Model evaluation completed in {time.time() - start_time:.2f} seconds")

def generate_evaluation_visualizations(X, y, models, encoders, feature_cols):
    """Generate additional visualizations for model evaluation."""
    logger.info("Generating evaluation visualizations...")
    start_time = time.time()
    
    # Class distribution
    class_names = encoders['target'].classes_
    class_counts = pd.Series(y).map(lambda x: class_names[x]).value_counts()
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x=class_counts.index, y=class_counts.values)
    plt.title('Distribution of Fertilizer Classes')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'class_distribution.png'))
    plt.close()
    
    # Feature distributions by class
    n_features = len(feature_cols)
    n_rows = (n_features + 1) // 2
    
    plt.figure(figsize=(16, 4 * n_rows))
    for i, feature in enumerate(feature_cols):
        plt.subplot(n_rows, 2, i + 1)
        for class_idx, class_name in enumerate(class_names):
            class_data = X[y == class_idx][feature]
            sns.kdeplot(class_data, label=class_name)
        plt.title(f'Distribution of {feature} by Class')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'feature_distributions_by_class.png'))
    plt.close()
    
    # Correlation matrix
    plt.figure(figsize=(12, 10))
    corr = X.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, cmap='coolwarm', annot=True, fmt='.2f', square=True)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'feature_correlation.png'))
    plt.close()
    
    logger.info(f"Evaluation visualizations generated in {time.time() - start_time:.2f} seconds")

def main():
    """Main function to run the model evaluation pipeline."""
    logger.info("=== Starting XGBoost Model Evaluation ===")
    
    # Start timer for total execution
    total_start_time = time.time()
    
    # Load the data
    train_df = load_data()
    
    # Preprocess the data
    X, y, encoders, feature_cols = preprocess_data(train_df)
    
    # Load the models
    models = load_models()
    
    if models:
        # Evaluate the models
        evaluate_models(X, y, models, encoders, feature_cols)
        
        # Generate evaluation visualizations
        generate_evaluation_visualizations(X, y, models, encoders, feature_cols)
    
    # Log total execution time
    logger.info(f"Total execution time: {time.time() - total_start_time:.2f} seconds")
    logger.info("=== XGBoost Model Evaluation Completed ===")

if __name__ == "__main__":
    main()