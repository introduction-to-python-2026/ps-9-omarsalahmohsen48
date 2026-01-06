import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib
import yaml

par = pd.read_csv("parkinsons.csv")

X = pd.DataFrame(par[['PPE',"RPDE", "HNR", 'DFA', "D2"]])
y = par.status

scaler = MinMaxScaler()
X = scaler.fit_transform(X)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

svm = SVC(kernel='rbf', C=1000, gamma='auto')
svm.fit(X_train, y_train)

accuracy = accuracy_score(y_val, svm.predict(X_val))
print("SVM Accuracy:", accuracy)


joblib.dump(svm, 'svm.joblib')
print("Model 'my_model.joblib' saved successfully.")


selected_features = ['PPE', 'RPDE', 'HNR', 'DFA', 'D2']
model_path = 'my_model.joblib'

config_data = {'selected_features': selected_features,'path': model_path}
