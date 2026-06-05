import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import mlflow
import mlflow.sklearn
import dagshub

def train_advanced_model():
    dagshub.init(repo_owner='FlorenzaJ', repo_name='Eksperimen_SML_Florenza', mlflow=True)
    mlflow.set_experiment("Advanced_Fraud_Detection")

    df = pd.read_csv("clean_dataset.csv")
    X = df.drop('Fraud_Flag', axis=1).select_dtypes(include=['number'])
    y = df['Fraud_Flag']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    param_grid = {'n_estimators': [50, 100], 'max_depth': [10, 20]}
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    y_pred = best_model.predict(X_test)
    
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0)
    }

    with mlflow.start_run(run_name="tuned_rf_model"):
        mlflow.log_params(best_params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(best_model, "model")
        
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        cm_path = "confusion_matrix.png"
        plt.savefig(cm_path)
        mlflow.log_artifact(cm_path)
        
        with open("tuning_details.txt", "w") as f:
            f.write(f"Best Parameters: {best_params}\n")
        mlflow.log_artifact("tuning_details.txt")

if __name__ == "__main__":
    train_advanced_model()