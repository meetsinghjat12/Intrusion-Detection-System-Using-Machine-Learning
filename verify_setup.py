import os
import sys

def verify_libraries():
    """Verify that the essential libraries are installed and print their versions."""
    print("=== Checking Essential Libraries ===")
    libraries = {
        "pandas": "pandas",
        "numpy": "numpy",
        "scikit-learn": "sklearn",
        "matplotlib": "matplotlib",
        "seaborn": "seaborn"
    }
    
    all_ok = True
    for name, import_name in libraries.items():
        try:
            module = __import__(import_name)
            version = getattr(module, "__version__", "Version info not available")
            print(f"[OK] {name:<15}: Installed ({version})")
        except ImportError:
            print(f"[FAIL] {name:<15}: NOT Installed")
            all_ok = False
            
    if all_ok:
        print("\n[SUCCESS] All essential libraries are installed and ready to go!\n")
    else:
        print("\n[WARN] Some libraries are missing. Please verify your virtual environment configuration.\n")
    return all_ok

def create_directory_structure():
    """Create the folders for the structured ML pipeline."""
    print("=== Setting Up Directory Structure ===")
    directories = ["data", "notebooks", "outputs"]
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            # Create a simple placeholder/readme file to keep the directory tracked
            readme_path = os.path.join(directory, "README.md")
            with open(readme_path, "w", encoding="utf-8") as f:
                f.write(f"# {directory.capitalize()} Directory\n\nThis directory holds the {directory} files for the Intrusion Detection System project.\n")
            print(f"[DIR] Created folder: '{directory}' (with README.md)")
        else:
            print(f"[DIR] Directory already exists: '{directory}'")
    print("\n[OK] Directory structure is verified.\n")

# Standard 41 feature column names for the NSL-KDD dataset
# plus 'attack_type' (classification target) and 'difficulty_level' (helper metadata)
NSL_KDD_COLUMNS = [
    # 1. Basic features of individual TCP connections
    "duration",
    "protocol_type",
    "service",
    "flag",
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
    # 2. Content features suggested by domain knowledge
    "hot",
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "num_outbound_cmds",
    "is_host_login",
    "is_guest_login",
    # 3. Time-based traffic features
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
    # 4. Host-based traffic features
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
    # 5. Label and difficulty level (42nd and 43rd columns)
    "attack_type",       # Target Label (e.g., normal, neptune, teardrop, etc.)
    "difficulty_level"   # Difficulty score in raw NSL-KDD files
]

def check_dataset_files():
    """Check if 'KDDTrain+.txt' and 'KDDTest+.txt' are present in the 'data' directory."""
    print("=== Checking NSL-KDD Dataset Files ===")
    
    # We expect pandas is installed at this point to run this function
    try:
        import pandas as pd
    except ImportError:
        print("[FAIL] Cannot check datasets with Pandas because pandas is not installed in the active environment.")
        return
        
    data_dir = "data"
    train_file = os.path.join(data_dir, "KDDTrain+.txt")
    test_file = os.path.join(data_dir, "KDDTest+.txt")
    
    files_to_check = {
        "Train Set (KDDTrain+.txt)": train_file,
        "Test Set (KDDTest+.txt)": test_file
    }
    
    all_present = True
    for label, path in files_to_check.items():
        if os.path.exists(path):
            print(f"[OK] Found {label} at '{path}'")
            try:
                # Load a small sample to verify format matches NSL-KDD column schema
                df_sample = pd.read_csv(path, header=None, names=NSL_KDD_COLUMNS, nrows=5)
                print(f"     Successfully parsed a preview of {label} ({len(df_sample.columns)} columns found).")
                print(f"     Target Label field: 'attack_type', Metadata field: 'difficulty_level'")
            except Exception as e:
                print(f"     [WARN] Found the file, but encountered error reading it: {e}")
        else:
            print(f"[FAIL] Missing {label} at '{path}'")
            all_present = False
            
    if all_present:
        print("\n[SUCCESS] Both dataset files are ready to be used in the preprocessing pipeline!\n")
    else:
        print("\n[INFO] NSL-KDD files are missing from the 'data' directory.")
        print("       Please download 'KDDTrain+.txt' and 'KDDTest+.txt' and place them inside the 'data/' folder.")
        print("       You can find the NSL-KDD dataset on academic repositories like Kaggle or University of New Brunswick (UNB) web archives.\n")

if __name__ == "__main__":
    # Configure stdout to use UTF-8 representation safely or stick to ASCII to avoid terminal encoding bugs
    print("=====================================================")
    print("      Intrusion Detection System Workspace Setup      ")
    print("=====================================================\n")
    
    # 1. Verify/create directory structure
    create_directory_structure()
    
    # 2. Check installed packages
    libs_ok = verify_libraries()
    
    # 3. Check for dataset files (conditional on libraries being available)
    if libs_ok:
        check_dataset_files()
        
    print("=====================================================")
