"""
Script to set up data and model files.
Run this once to copy/link necessary files from the old project location.
"""
import shutil
import os

OLD_PROJECT = r"d:\loan_prediction_project"
NEW_PROJECT = r"d:\Doan_vayvon"

def setup():
    # Copy model files
    models_src = os.path.join(OLD_PROJECT, "models")
    models_dst = os.path.join(NEW_PROJECT, "models")
    os.makedirs(models_dst, exist_ok=True)
    
    for f in ['loan_model.pkl', 'label_encoders.pkl', 'feature_columns.pkl']:
        src = os.path.join(models_src, f)
        dst = os.path.join(models_dst, f)
        if os.path.exists(src) and not os.path.exists(dst):
            print(f"Copying {f}...")
            shutil.copy2(src, dst)
        elif os.path.exists(dst):
            print(f"  {f} already exists, skipping.")
        else:
            print(f"  WARNING: {src} not found!")

    # Copy data files
    data_src = os.path.join(OLD_PROJECT, "data")
    data_dst = os.path.join(NEW_PROJECT, "data")
    os.makedirs(data_dst, exist_ok=True)
    
    for f in ['train_processed.csv', 'application_train.csv']:
        src = os.path.join(data_src, f)
        dst = os.path.join(data_dst, f)
        if os.path.exists(src) and not os.path.exists(dst):
            size_mb = os.path.getsize(src) / (1024 * 1024)
            print(f"Copying {f} ({size_mb:.0f} MB)...")
            shutil.copy2(src, dst)
        elif os.path.exists(dst):
            print(f"  {f} already exists, skipping.")
        else:
            print(f"  WARNING: {src} not found!")

    # Copy feature importance image
    fi_src = os.path.join(OLD_PROJECT, "static", "feature_importance.png")
    fi_dst = os.path.join(NEW_PROJECT, "static", "feature_importance.png")
    if os.path.exists(fi_src) and not os.path.exists(fi_dst):
        print("Copying feature_importance.png...")
        shutil.copy2(fi_src, fi_dst)

    print("\n✅ Setup complete! You can now run: python app.py")

if __name__ == "__main__":
    setup()
