import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import logging
from datetime import datetime

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, VotingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             confusion_matrix, classification_report, roc_curve, auc,
                             precision_recall_curve)

from utils import add_derived_features

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("FakeProfileTrainer")


class FakeProfileDetector:
    """
    Advanced Machine Learning Pipeline for Social Media Fake Profile Detection.
    Features: Automated Preprocessing, Hyperparameter Tuning, and Ensemble Learning.
    """
    
    def __init__(self, data_path_train, data_path_test):
        self.data_path_train = data_path_train
        self.data_path_test = data_path_test
        self.output_dir = 'model'
        self.plot_dir = 'plots'
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.plot_dir, exist_ok=True)
        
        self.categorical_features = ['profile_pic', 'extern_url', 'private', 'sim_name_username']
        self.numerical_features = [
            'ratio_numlen_username', 'len_fullname', 'ratio_numlen_fullname', 
            'len_desc', 'num_posts', 'num_followers', 'num_following'
        ]
        self.target = 'fake'
        self.pipeline = None
        
    def load_and_preprocess(self):
        logger.info("Loading datasets...")
        train_df = pd.read_csv(self.data_path_train)
        test_df = pd.read_csv(self.data_path_test)
        
        # Drop dummy index columns if they exist
        for df in [train_df, test_df]:
            if df.columns[0] == 'Unnamed: 0' or df.columns[0] == '':
                df.drop(df.columns[0], axis=1, inplace=True)
        
        self.feature_engineering_step = FunctionTransformer(add_derived_features)
        
        X_train = train_df.drop(self.target, axis=1)
        y_train = train_df[self.target]
        X_test = test_df.drop(self.target, axis=1)
        y_test = test_df[self.target]
        
        logger.info(f"Training shape: {X_train.shape}, Testing shape: {X_test.shape}")
        return X_train, y_train, X_test, y_test

    def build_pipeline(self):
        logger.info("Building ColumnTransformer and Pipeline...")
        
        # Preprocessing for numerical data: Scaling
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])

        # Preprocessing for categorical data: One-Hot Encoding
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        # Bundle preprocessing for numerical and categorical data
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numerical_features + ['follower_following_ratio']),
                ('cat', categorical_transformer, self.categorical_features)
            ])

        # Full Pipeline with Feature Engineering
        self.pipeline = Pipeline(steps=[
            ('engineering', self.feature_engineering_step),
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(random_state=42))
        ])
        
        return self.pipeline

    def train_with_tuning(self, X_train, y_train):
        logger.info("Starting hyperparameter tuning with GridSearchCV...")
        
        param_grid = {
            'classifier__n_estimators': [50, 100, 200],
            'classifier__max_depth': [None, 10, 20],
            'classifier__min_samples_split': [2, 5]
        }
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        grid_search = GridSearchCV(self.pipeline, param_grid, cv=cv, scoring='f1', n_jobs=-1, verbose=1)
        
        grid_search.fit(X_train, y_train)
        
        logger.info(f"Best Parameters: {grid_search.best_params_}")
        logger.info(f"Best Cross-Validation F1-Score: {grid_search.best_score_:.4f}")
        
        self.best_model = grid_search.best_estimator_
        return self.best_model

    def evaluate(self, X_test, y_test):
        logger.info("Evaluating model on test set...")
        y_pred = self.best_model.predict(X_test)
        y_proba = self.best_model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1 Score': f1_score(y_test, y_pred),
            'ROC AUC': auc(*roc_curve(y_test, y_proba)[:2])
        }
        
        logger.info("\nClassification Report:\n" + classification_report(y_test, y_pred))
        logger.info(f"Model Metrics: {metrics}")
        
        self.save_plots(y_test, y_pred, y_proba)
        return metrics

    def save_plots(self, y_test, y_pred, y_proba):
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
        plt.title('Confusion Matrix - Optimized Ensemble')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.savefig(f'{self.plot_dir}/cm_optimized.png')
        plt.close()

        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc="lower right")
        plt.savefig(f'{self.plot_dir}/roc_curve.png')
        plt.close()
        
        # Feature Importance (Extract from Pipeline)
        try:
            ohe_features = self.best_model.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(self.categorical_features)
            all_features = self.numerical_features + ['follower_following_ratio'] + list(ohe_features)
            importances = self.best_model.named_steps['classifier'].feature_importances_
            
            feat_imp = pd.Series(importances, index=all_features).sort_values(ascending=False).head(10)
            plt.figure(figsize=(10, 8))
            sns.barplot(x=feat_imp.values, y=feat_imp.index, palette='magma')
            plt.title('Top 10 Important Features')
            plt.savefig(f'{self.plot_dir}/feature_importance_optimized.png')
            plt.close()
        except Exception as e:
            logger.warning(f"Could not plot feature importance: {e}")

    def save_model(self):
        model_filename = f'{self.output_dir}/advanced_model.joblib'
        joblib.dump(self.best_model, model_filename)
        # Save feature names for verification
        joblib.dump(self.numerical_features + self.categorical_features, f'{self.output_dir}/feature_list.joblib')
        logger.info(f"Pipeline saved to {model_filename}")

def main():
    detector = FakeProfileDetector(
        data_path_train='data/social_media_train.csv',
        data_path_test='data/social_media_test.csv'
    )
    
    X_train, y_train, X_test, y_test = detector.load_and_preprocess()
    detector.build_pipeline()
    detector.train_with_tuning(X_train, y_train)
    detector.evaluate(X_test, y_test)
    detector.save_model()
    
    logger.info("Advanced Machine Learning Training Complete!")

if __name__ == "__main__":
    main()
