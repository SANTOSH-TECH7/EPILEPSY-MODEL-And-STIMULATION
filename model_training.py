import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from xgboost import XGBClassifier
import pickle
import joblib
import os

# Define the correct paths
BASE_DIR = r"D:\EPILEPSY_PROJECT_DETAILS\EpilepsyTest"
DATA_PATH = os.path.join(BASE_DIR, "Epileptic Seizure Recognition.csv")
MODEL_PICKLE_PATH = os.path.join(BASE_DIR, "epilepsy_seizure_model.pkl")
MODEL_JOBLIB_PATH = os.path.join(BASE_DIR, "epilepsy_seizure_model_joblib.pkl")

print("=" * 60)
print("EPILEPSY SEIZURE DETECTION - Model Training")
print("=" * 60)
print(f"Base Directory: {BASE_DIR}")
print(f"Data Path: {DATA_PATH}")
print(f"Model Pickle Path: {MODEL_PICKLE_PATH}")
print(f"Model Joblib Path: {MODEL_JOBLIB_PATH}")
print("-" * 60)

# Check if base directory exists
if not os.path.exists(BASE_DIR):
    print(f"❌ Base directory does not exist: {BASE_DIR}")
    exit(1)

# Load dataset
try:
    print("📊 Loading dataset...")
    data = pd.read_csv(DATA_PATH)
    print(f"✅ Dataset loaded successfully! Shape: {data.shape}")
except FileNotFoundError:
    print(f"❌ Dataset not found at: {DATA_PATH}")
    print("Please ensure the CSV file exists in the correct location.")
    exit(1)
except Exception as e:
    print(f"❌ Error loading dataset: {e}")
    exit(1)

# Drop unnecessary columns if they exist
columns_to_drop = [col for col in data.columns if 'Unnamed' in col]
if columns_to_drop:
    data.drop(columns_to_drop, axis=1, inplace=True)
    print(f"🗑️  Dropped unnecessary columns: {columns_to_drop}")

# Separate features and target variable
X = data.iloc[:, :-1]
y = data['y']

print(f"📈 Features shape: {X.shape}")
print(f"🎯 Target classes: {sorted(y.unique())}")
print(f"📊 Class distribution:")
print(y.value_counts().sort_index())

# Initialize and fit the scaler
print("\n🔧 Preprocessing data...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Initialize and fit the label encoder
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)

print(f"✅ Train set: {X_train.shape[0]} samples")
print(f"✅ Test set: {X_test.shape[0]} samples")

# Train XGBoost model
print("\n🤖 Training XGBoost model...")
xgb_model = XGBClassifier(
    use_label_encoder=False, 
    eval_metric='mlogloss',
    random_state=42,
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1
)

xgb_model.fit(X_train, y_train_encoded)
print("✅ Model training completed!")

# Make predictions on the test set
print("\n📊 Evaluating model performance...")
y_pred = xgb_model.predict(X_test)

# Evaluation Metrics
acc = accuracy_score(y_test_encoded, y_pred)
f1 = f1_score(y_test_encoded, y_pred, average='weighted')
conf_matrix = confusion_matrix(y_test_encoded, y_pred)

print("-" * 40)
print("MODEL PERFORMANCE METRICS")
print("-" * 40)
print(f"Accuracy: {acc:.4f}")
print(f"F1 Score: {f1:.4f}")
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(classification_report(y_test_encoded, y_pred))

# Create a model package with all necessary components
print("\n📦 Creating model package...")
model_package = {
    'model': xgb_model,
    'scaler': scaler,
    'label_encoder': le,
    'feature_names': list(X.columns),
    'class_names': le.classes_,
    'model_metrics': {
        'accuracy': float(acc),
        'f1_score': float(f1),
        'confusion_matrix': conf_matrix.tolist()
    },
    'training_info': {
        'train_samples': X_train.shape[0],
        'test_samples': X_test.shape[0],
        'feature_count': X.shape[1],
        'class_count': len(le.classes_),
        'model_type': 'XGBClassifier'
    }
}

# Save the complete model package
print("\n💾 Saving model files...")
try:
    # Save using pickle
    with open(MODEL_PICKLE_PATH, 'wb') as f:
        pickle.dump(model_package, f)
    print(f"✅ Model saved (pickle): {MODEL_PICKLE_PATH}")
    
    # Save using joblib (more efficient for scikit-learn models)
    joblib.dump(model_package, MODEL_JOBLIB_PATH)
    print(f"✅ Model saved (joblib): {MODEL_JOBLIB_PATH}")
    
except Exception as e:
    print(f"❌ Error saving model: {e}")
    exit(1)

# Test loading the model
print("\n🧪 Testing model loading...")
try:
    # Test joblib loading
    loaded_model = joblib.load(MODEL_JOBLIB_PATH)
    print("✅ Joblib model loaded successfully!")
    
    # Test prediction with a sample
    sample_data = X_test[:1]
    
    # Make prediction using loaded model
    sample_scaled = loaded_model['scaler'].transform(sample_data)
    prediction = loaded_model['model'].predict(sample_scaled)
    prediction_proba = loaded_model['model'].predict_proba(sample_scaled)
    predicted_class = loaded_model['label_encoder'].inverse_transform(prediction)[0]
    
    print(f"✅ Test prediction successful!")
    print(f"   Predicted class: {predicted_class}")
    print(f"   Confidence: {max(prediction_proba[0]):.4f}")
    
except Exception as e:
    print(f"❌ Error testing model loading: {e}")

# Create visualizations (optional - can be commented out for production)
print("\n📊 Creating visualizations...")

try:
    # Set up the plot style
    plt.style.use('default')
    
    # Plot Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='coolwarm', 
               linewidths=1, linecolor='black')
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix - XGBoost Model")
    
    # Save the plot
    plot_path = os.path.join(BASE_DIR, "confusion_matrix.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✅ Confusion matrix saved: {plot_path}")
    plt.show()

    # Feature Importance
    plt.figure(figsize=(12, 8))
    importances = xgb_model.feature_importances_
    feature_names = X.columns
    
    # Get top 20 most important features
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False).head(20)
    
    sns.barplot(data=feature_importance_df, x='importance', y='feature', palette="viridis")
    plt.xlabel("Feature Importance Score")
    plt.ylabel("Features")
    plt.title("Top 20 Feature Importance from XGBoost")
    
    # Save the plot
    plot_path = os.path.join(BASE_DIR, "feature_importance.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✅ Feature importance plot saved: {plot_path}")
    plt.show()

    # Accuracy and F1 Score Bar Graph
    plt.figure(figsize=(8, 6))
    metrics = ["Accuracy", "F1 Score"]
    values = [acc, f1]
    
    bars = plt.bar(metrics, values, palette="viridis", color=['#1f77b4', '#ff7f0e'])
    plt.ylim(0, 1)
    plt.title("Model Performance Metrics")
    plt.ylabel("Score")
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # Save the plot
    plot_path = os.path.join(BASE_DIR, "performance_metrics.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✅ Performance metrics plot saved: {plot_path}")
    plt.show()

except Exception as e:
    print(f"⚠️  Warning: Could not create visualizations: {e}")

# Summary
print("\n" + "=" * 60)
print("TRAINING SUMMARY")
print("=" * 60)
print(f"✅ Model training completed successfully!")
print(f"✅ Model accuracy: {acc:.4f}")
print(f"✅ Model F1-score: {f1:.4f}")
print(f"✅ Model files saved to: {BASE_DIR}")
print(f"   - Pickle format: epilepsy_seizure_model.pkl")
print(f"   - Joblib format: epilepsy_seizure_model_joblib.pkl")
print(f"✅ Visualizations saved (if matplotlib available)")
print(f"\n🚀 Ready to run Flask application!")
print(f"   Execute: python {os.path.join(BASE_DIR, 'app.py')}")
print("=" * 60)