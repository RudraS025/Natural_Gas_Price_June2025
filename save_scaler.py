scaler = StandardScaler()
scaler.fit(X_train)
import joblib
joblib.dump(scaler, 'scaler.save')
