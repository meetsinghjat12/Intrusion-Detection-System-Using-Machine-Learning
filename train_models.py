import os
import sys
import pandas as pd
import numpy as np

# Use 'Agg' backend for matplotlib to run headlessly without GUI issues
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Standard 41 feature column names for the NSL-KDD dataset
# plus 'attack_type' (target) and 'difficulty_level' (helper metadata)
NSL_KDD_COLUMNS = [
    # 1. Basic features
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", 
    "land", "wrong_fragment", "urgent",
    # 2. Content features
    "hot", "num_failed_logins", "logged_in", "num_compromised", "root_shell", 
    "su_attempted", "num_root", "num_file_creations", "num_shells", 
    "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login",
    # 3. Time-based traffic features
    "count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", 
    "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
    # 4. Host-based traffic features
    "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate", 
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate", 
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", 
    "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
    # 5. Target labels and metadata
    "attack_type",
    "difficulty_level"
]

# Mapping of granular attack types to 4 core wireless threat vectors + normal
ATTACK_MAPPING = {
    # DoS (Denial of Service / Jamming)
    'neptune': 'dos', 'smurf': 'dos', 'back': 'dos', 'teardrop': 'dos', 
    'pod': 'dos', 'land': 'dos', 'apache2': 'dos', 'mailbomb': 'dos', 
    'processtable': 'dos', 'udpstorm': 'dos', 'worm': 'dos',
    
    # Probe (Reconnaissance / Scanning)
    'satan': 'probe', 'ipsweep': 'probe', 'portsweep': 'probe', 'nmap': 'probe', 
    'mscan': 'probe', 'saint': 'probe',
    
    # R2L (Unauthorized Wireless Access)
    'warezclient': 'r2l', 'guess_passwd': 'r2l', 'warezmaster': 'r2l', 
    'imap': 'r2l', 'ftp_write': 'r2l', 'multihop': 'r2l', 'phf': 'r2l', 
    'spy': 'r2l', 'snmpgetattack': 'r2l', 'httptunnel': 'r2l', 'snmpguess': 'r2l', 
    'mailbomb': 'r2l', 'named': 'r2l', 'sendmail': 'r2l', 'xlock': 'r2l', 
    'xsnoop': 'r2l',
    
    # U2R (Privilege Escalation)
    'buffer_overflow': 'u2r', 'loadmodule': 'u2r', 'rootkit': 'u2r', 
    'perl': 'u2r', 'ps': 'u2r', 'xterm': 'u2r', 'sqlattack': 'u2r',
    
    # Normal Traffic
    'normal': 'normal'
}

def clean_and_map_attack(label):
    """Clean the label string and map it to a core threat category."""
    label_clean = str(label).strip().lower()
    # Handle optional trailing dots (common in KDD-style files)
    if label_clean.endswith('.'):
        label_clean = label_clean[:-1]
    
    category = ATTACK_MAPPING.get(label_clean)
    if category is None:
        print(f"[WARN] Unrecognized attack class '{label}' encountered! Mapping to 'normal'.")
        return 'normal'
    return category

def main():
    print("=========================================================================")
    print("    AI-Driven Intrusion Detection System - Training & Evaluation Pipeline")
    print("=========================================================================\n")
    
    # ----------------------------------------------------
    # 1. DATASET LOADING
    # ----------------------------------------------------
    print("[1/4] Loading NSL-KDD dataset...")
    train_path = os.path.join("data", "KDDTrain+.txt")
    test_path = os.path.join("data", "KDDTest+.txt")
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print(f"[ERROR] Dataset files not found. Run verify_setup.py first or download the files.")
        sys.exit(1)
        
    df_train = pd.read_csv(train_path, header=None, names=NSL_KDD_COLUMNS)
    df_test = pd.read_csv(test_path, header=None, names=NSL_KDD_COLUMNS)
    
    print(f"      Train Dataset Shape: {df_train.shape}")
    print(f"      Test Dataset Shape:  {df_test.shape}")
    
    # Discard difficulty_level column
    print("      Discarding 'difficulty_level' column...")
    df_train = df_train.drop(columns=["difficulty_level"])
    df_test = df_test.drop(columns=["difficulty_level"])
    
    # Map raw attack types to core next-gen threat vectors
    print("      Mapping granular attack types to 4 core threat vectors...")
    df_train["attack_category"] = df_train["attack_type"].apply(clean_and_map_attack)
    df_test["attack_category"] = df_test["attack_type"].apply(clean_and_map_attack)
    
    # Print threat vector distribution
    print("\n--- Training Set Threat Category Distribution ---")
    print(df_train["attack_category"].value_counts())
    print("\n--- Testing Set Threat Category Distribution ---")
    print(df_test["attack_category"].value_counts())
    print("-------------------------------------------------\n")

    # ----------------------------------------------------
    # 2. PREPROCESSING PIPELINE
    # ----------------------------------------------------
    print("[2/4] Preprocessing features and labels...")
    
    # Categorical input features
    categorical_cols = ["protocol_type", "service", "flag"]
    
    # Safe categorical label encoding
    print("      Encoding categorical attributes (safely handling unseen classes)...")
    for col in categorical_cols:
        le = LabelEncoder()
        df_train[col] = le.fit_transform(df_train[col])
        
        # Safely map unseen classes in the test set to the first seen class of the training set
        df_test[col] = df_test[col].apply(lambda val: val if val in le.classes_ else le.classes_[0])
        df_test[col] = le.transform(df_test[col])
        
    # Target encoding using a separate LabelEncoder
    print("      Encoding target labels...")
    target_le = LabelEncoder()
    df_train["label"] = target_le.fit_transform(df_train["attack_category"])
    df_test["label"] = target_le.transform(df_test["attack_category"])
    
    # Define features list (excluding raw text attack_type, attack_category, and labeled target column)
    features_list = [col for col in df_train.columns if col not in ["attack_type", "attack_category", "label"]]
    numerical_cols = [col for col in features_list if col not in categorical_cols]
    
    # Normalize numerical features using MinMaxScaler
    print("      Normalizing numerical features with MinMaxScaler...")
    scaler = MinMaxScaler()
    df_train[numerical_cols] = scaler.fit_transform(df_train[numerical_cols])
    df_test[numerical_cols] = scaler.transform(df_test[numerical_cols])
    
    # Extract features and targets
    X_train = df_train[features_list].values
    y_train = df_train["label"].values
    X_test = df_test[features_list].values
    y_test = df_test["label"].values
    
    # ----------------------------------------------------
    # 3. MODEL TRAINING
    # ----------------------------------------------------
    print("\n[3/4] Initializing and training models...")
    
    # Model A: Decision Tree
    print("      Training Decision Tree Classifier...")
    dt_model = DecisionTreeClassifier(random_state=42)
    dt_model.fit(X_train, y_train)
    print("      [OK] Decision Tree Classifier completed training.")
    
    # Model B: Random Forest
    print("      Training Random Forest Classifier (n_estimators=100)...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    print("      [OK] Random Forest Classifier completed training.")
    
    # ----------------------------------------------------
    # 4. ACADEMIC EVALUATION & ARTIFACT GENERATION
    # ----------------------------------------------------
    print("\n[4/4] Evaluating models and generating artifacts...")
    
    # Evaluate Decision Tree
    dt_y_pred = dt_model.predict(X_test)
    dt_acc = accuracy_score(y_test, dt_y_pred)
    
    # Evaluate Random Forest
    rf_y_pred = rf_model.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_y_pred)
    
    # Report performance comparisons
    print("\n==============================================")
    print(f"      COMPARATIVE PERFORMANCE ANALYSIS        ")
    print("==============================================")
    print(f"Decision Tree Accuracy : {dt_acc:.6f}")
    print(f"Random Forest Accuracy : {rf_acc:.6f}")
    print("==============================================")
    
    # Print full classification reports
    target_names = target_le.classes_
    
    print("\n>>> DECISION TREE CLASSIFICATION REPORT <<<")
    print(classification_report(y_test, dt_y_pred, target_names=target_names))
    
    print(">>> RANDOM FOREST CLASSIFICATION REPORT <<<")
    print(classification_report(y_test, rf_y_pred, target_names=target_names))
    
    # Determine the highest-performing model
    if dt_acc >= rf_acc:
        best_model_name = "Decision Tree"
        best_y_pred = dt_y_pred
        best_acc = dt_acc
    else:
        best_model_name = "Random Forest"
        best_y_pred = rf_y_pred
        best_acc = rf_acc
        
    print(f"Highest-performing model: {best_model_name} ({best_acc:.6f})")
    print("Generating Confusion Matrix visualization for the highest-performing model...")
    
    # Compute Confusion Matrix
    cm = confusion_matrix(y_test, best_y_pred)
    
    # Create the high-resolution visualization
    sns.set_theme(style="white")
    plt.figure(figsize=(10, 8))
    
    # Plot using a modern, visually striking color palette ('mako')
    ax = sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='mako',
        xticklabels=target_names,
        yticklabels=target_names,
        cbar=True,
        annot_kws={"size": 12, "weight": "bold"},
        linewidths=0.8,
        linecolor='white'
    )
    
    plt.title(f'Confusion Matrix: {best_model_name}\n(Overall Test Accuracy: {best_acc:.4%})', fontsize=14, weight='bold', pad=15)
    plt.xlabel('Predicted Threat Class', fontsize=12, labelpad=10)
    plt.ylabel('True Threat Class', fontsize=12, labelpad=10)
    
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10, rotation=0)
    
    # Automatically adjust layout and save file
    plt.tight_layout()
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    print(f"      [OK] Confusion Matrix visualization saved successfully to: '{output_path}'")
    print("\nPipeline execution completed successfully.")
    print("=========================================================================")

if __name__ == "__main__":
    main()
