import os, json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
import xgboost as xgb

df = pd.read_csv('data/bank-additional-full.csv', sep=';')
df['y'] = (df['y'] == 'yes').astype(int)
if 'duration' in df.columns:
    df = df.drop(columns=['duration'])
cat_cols = df.select_dtypes(include='object').columns.tolist()
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = {v: int(i) for i, v in enumerate(le.classes_)}
X = df.drop(columns=['y'])
y = df['y']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model = xgb.XGBClassifier(n_estimators=349, max_depth=5, learning_rate=0.0124,
    subsample=0.7244, colsample_bytree=0.7301, min_child_weight=8,
    scale_pos_weight=10, eval_metric='logloss', verbosity=0, random_state=42)
model.fit(X_train, y_train)
auc = roc_auc_score(y_test, model.predict_proba(X_test)[:,1])
os.makedirs('artifacts', exist_ok=True)
model.save_model('artifacts/champion_model.json')
json.dump(list(X.columns), open('artifacts/feature_order.json','w'))
json.dump(encoders, open('artifacts/encoders.json','w'))
json.dump({'model_type':'XGBoost','auc':round(auc,4),'f1':0.0,
    'n_features':len(X.columns),'features':list(X.columns),
    'baseline_auc':0.8174,'project':'mlops-model-serving',
    'data_source':'bank-additional-full.csv'},
    open('artifacts/model_info.json','w'), indent=2)
print(f'Done! AUC={auc:.4f} | Artifacts saved to artifacts/')