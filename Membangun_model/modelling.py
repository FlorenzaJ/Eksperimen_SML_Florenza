import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import mlflow

def train_basic_model():
    mlflow.set_tracking_uri("sqlite:///mlruns.db")
    mlflow.set_experiment("Basic_Fraud_Detection")
    mlflow.autolog()

    print("Loading dataset...")
    df = pd.read_csv("clean_dataset.csv")
    
    X = df.drop('Fraud_Flag', axis=1)
    X = X.select_dtypes(include=['number'])
    y = df['Fraud_Flag']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training basic model...")
    with mlflow.start_run(run_name="basic_rf_model"):
        rf = RandomForestClassifier(random_state=42)
        rf.fit(X_train, y_train)
        print("Training complete. Local artifacts generated.")

if __name__ == "__main__":
    train_basic_model()