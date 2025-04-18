import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import pickle

data = pd.read_csv("traffic_data.csv")

label_map = {"CLEAR": 0, "BUSY": 1, "JAM": 2}
data['traffic_status'] = data['traffic_status'].map(label_map)

X = data[['vehicle_count']]
y = data['traffic_status']

model = DecisionTreeClassifier()
model.fit(X, y)

with open("traffic_model.pkl", "wb") as f:
    pickle.dump(model, f)
