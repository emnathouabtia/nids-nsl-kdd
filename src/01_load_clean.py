import pandas as pd
from sklearn.preprocessing import LabelEncoder

columns = [
    "duration","protocol_type","service","flag","src_bytes","dst_bytes","land",
    "wrong_fragment","urgent","hot","num_failed_logins","logged_in","num_compromised",
    "root_shell","su_attempted","num_root","num_file_creations","num_shells",
    "num_access_files","num_outbound_cmds","is_host_login","is_guest_login","count",
    "srv_count","serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate",
    "same_srv_rate","diff_srv_rate","srv_diff_host_rate","dst_host_count",
    "dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rate",
    "dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate",
    "label","difficulty"
]
train = pd.read_csv("../data/KDDTrain+.txt", names=columns)
test = pd.read_csv("../data/KDDTest+.txt", names=columns)
print("Shape train:", train.shape)
print("Shape test:", test.shape)

print(train.head())

print("Valeurs manquantes train:", train.isnull().sum().sum())
print("Valeurs manquantes test:", test.isnull().sum().sum())
print(train["label"].value_counts())
attack_map = {
    "normal": "normal",

    # DoS
    "neptune": "dos",
    "smurf": "dos",
    "back": "dos",
    "teardrop": "dos",
    "pod": "dos",
    "land": "dos",
    "apache2": "dos",
    "mailbomb": "dos",
    "processtable": "dos",
    "udpstorm": "dos",

    # Probe
    "satan": "probe",
    "ipsweep": "probe",
    "portsweep": "probe",
    "nmap": "probe",
    "saint": "probe",
    "mscan": "probe",

    # R2L
    "guess_passwd": "r2l",
    "warezclient": "r2l",
    "warezmaster": "r2l",
    "imap": "r2l",
    "ftp_write": "r2l",
    "multihop": "r2l",
    "phf": "r2l",
    "spy": "r2l",
    "snmpgetattack": "r2l",
    "snmpguess": "r2l",
    "httptunnel": "r2l",
    "sendmail": "r2l",
    "named": "r2l",
    "xlock": "r2l",
    "xsnoop": "r2l",

    # U2R
    "buffer_overflow": "u2r",
    "loadmodule": "u2r",
    "rootkit": "u2r",
    "perl": "u2r",
    "ps": "u2r",
    "xterm": "u2r",
    "sqlattack": "u2r",
    "worm": "u2r",
}

train["attack_category"] = train["label"].map(attack_map)

# Vérifie s'il reste des labels non reconnus (NaN après le mapping)
print(train[train["attack_category"].isnull()]["label"].unique())
test["attack_category"] = test["label"].map(attack_map)
print(test[test["attack_category"].isnull()]["label"].unique())
print(train["attack_category"].value_counts())
print(test["attack_category"].value_counts())
# One-Hot Encoding des 3 colonnes catégorielles
categorical_cols = ["protocol_type", "service", "flag"]

train_encoded = pd.get_dummies(train, columns=categorical_cols)
test_encoded = pd.get_dummies(test, columns=categorical_cols)

print("Shape train avant encodage:", train.shape)
print("Shape train après encodage:", train_encoded.shape)
print(train_encoded.head())
#nombres de lignes dans les deux datasets train and test aprés modification
print("Colonnes train:", train_encoded.shape[1])
print("Colonnes test:", test_encoded.shape[1])

# Colonnes présentes dans train mais absentes du test
print("Dans train mais pas test:", set(train_encoded.columns) - set(test_encoded.columns))

# Colonnes présentes dans test mais absentes du train
print("Dans test mais pas train:", set(test_encoded.columns) - set(train_encoded.columns))
print("Colonnes test après alignement:", test_encoded.shape[1])
# ajout des colonnes manquantes
test_encoded = test_encoded.reindex(columns=train_encoded.columns, fill_value=0)

print("Colonnes train:", train_encoded.shape[1])
print("Colonnes test après alignement:", test_encoded.shape[1])

# ÉTAPE 2.2 : Séparation features (X) / cible (y)
# ============================================================

X_train = train_encoded.drop(columns=["label", "attack_category", "difficulty"])
y_train = train_encoded["attack_category"]

X_test = test_encoded.drop(columns=["label", "attack_category", "difficulty"])
y_test = test_encoded["attack_category"]

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print(y_train.value_counts())

print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)
print(y_test.value_counts())

# Sauvegarde pour la prochaine étape (entraînement des modèles)
X_train.to_csv("../data/X_train.csv", index=False)
X_test.to_csv("../data/X_test.csv", index=False)
y_train.to_csv("../data/y_train.csv", index=False)
y_test.to_csv("../data/y_test.csv", index=False)

print("Fichiers sauvegardés dans data/")

le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)

print(le.classes_)  # pour voir quel numéro correspond à quelle catégorie
print(y_train_encoded[:10])

#random forest 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Créer le modèle
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# 2. Entraîner le modèle sur les données d'entraînement
rf.fit(X_train, y_train_encoded)

# 3. Prédire sur le jeu de test (le modèle ne voit QUE X_test, jamais y_test)
y_pred = rf.predict(X_test)

# 4. Comparer les prédictions à la vraie réponse (y_test_encoded)
print("Accuracy:", accuracy_score(y_test_encoded, y_pred))
print(classification_report(y_test_encoded, y_pred, target_names=le.classes_))
# xgboost
from xgboost import XGBClassifier

xgb = XGBClassifier(n_estimators=100, random_state=42, eval_metric="mlogloss")
xgb.fit(X_train, y_train_encoded)

y_pred_xgb = xgb.predict(X_test)

print("=== XGBoost ===")
print("Accuracy:", accuracy_score(y_test_encoded, y_pred_xgb))
print(classification_report(y_test_encoded, y_pred_xgb, target_names=le.classes_))
#svm
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Sous-échantillon stratifié
X_train_sample, _, y_train_sample, _ = train_test_split(
    X_train, y_train_encoded,
    train_size=20000,
    stratify=y_train_encoded,
    random_state=42
)

# 2. Normalisation
scaler = StandardScaler()
X_train_sample_scaled = scaler.fit_transform(X_train_sample)
X_test_scaled = scaler.transform(X_test)

# 3. Entraînement
svm = SVC(kernel="rbf", random_state=42)
svm.fit(X_train_sample_scaled, y_train_sample)

y_pred_svm = svm.predict(X_test_scaled)

print("=== SVM (avec normalisation) ===")
print("Accuracy:", accuracy_score(y_test_encoded, y_pred_svm))
print(classification_report(y_test_encoded, y_pred_svm, target_names=le.classes_))

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

scaler_nn = StandardScaler()
X_train_scaled = scaler_nn.fit_transform(X_train)
X_test_scaled_nn = scaler_nn.transform(X_test)

y_train_cat = to_categorical(y_train_encoded, num_classes=5)
y_test_cat = to_categorical(y_test_encoded, num_classes=5)

model = Sequential([
    Dense(64, activation="relu", input_shape=(X_train_scaled.shape[1],)),
    Dropout(0.3),
    Dense(32, activation="relu"),
    Dropout(0.3),
    Dense(5, activation="softmax")
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

model.fit(X_train_scaled, y_train_cat, epochs=20, batch_size=256, validation_split=0.1, verbose=1)

y_pred_nn_proba = model.predict(X_test_scaled_nn)
y_pred_nn = y_pred_nn_proba.argmax(axis=1)

print("=== Réseau de neurones (Keras) ===")
print("Accuracy:", accuracy_score(y_test_encoded, y_pred_nn))
print(classification_report(y_test_encoded, y_pred_nn, target_names=le.classes_))
#correction des nombres 
from sklearn.utils.class_weight import compute_sample_weight

# Calcule un poids pour chaque ligne, inversement proportionnel à la fréquence de sa catégorie
sample_weights = compute_sample_weight(class_weight="balanced", y=y_train_encoded)

xgb_balanced = XGBClassifier(n_estimators=100, random_state=42, eval_metric="mlogloss")
xgb_balanced.fit(X_train, y_train_encoded, sample_weight=sample_weights)

y_pred_xgb_balanced = xgb_balanced.predict(X_test)

print("=== XGBoost (avec class_weight balanced) ===")
print("Accuracy:", accuracy_score(y_test_encoded, y_pred_xgb_balanced))
print(classification_report(y_test_encoded, y_pred_xgb_balanced, target_names=le.classes_))
# shap pour explication 
import shap

# SHAP peut être lent sur beaucoup de lignes — on explique sur un sous-échantillon du test
import shap
import numpy as np
import matplotlib.pyplot as plt

X_test_sample = X_test.sample(n=500, random_state=42)

explainer = shap.TreeExplainer(xgb)
shap_values = explainer.shap_values(X_test_sample)

print("Shape de shap_values:", np.array(shap_values).shape)

# Résumé pour la classe "dos" (index 0 dans le.classes_) — la plus fréquente et bien détectée
class_index = 0  # 0=dos, 1=normal, 2=probe, 3=r2l, 4=u2r (voir le.classes_)
shap.summary_plot(shap_values[:, :, class_index], X_test_sample, show=False)
plt.title(f"SHAP summary — classe '{le.classes_[class_index]}'")
plt.savefig("../results/shap_summary_dos.png", bbox_inches="tight")
plt.close()

print("Graphique SHAP sauvegardé dans results/shap_summary_dos.png")