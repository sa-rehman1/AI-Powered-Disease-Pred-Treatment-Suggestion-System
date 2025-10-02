# Just run this code to clear all the data
# clear_data.py
import os
import shutil
import pandas as pd

def clear_all_data():
    """Clear all application data"""
    
    # Directories to clear
    data_dirs = ["data/patients", "data"]
    files_to_remove = ["data/users.csv", "data/doctors.csv", "data/reminders.csv"]
    
    print("🧹 Clearing all application data...")
    
    # Remove patient data directory
    for dir_path in data_dirs:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
            print(f"✅ Removed directory: {dir_path}")
    
    # Remove individual files
    for file_path in files_to_remove:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"✅ Removed file: {file_path}")
    
    # Recreate necessary directories
    os.makedirs("data/patients", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    
    # Create empty user files with headers
    pd.DataFrame(columns=["username", "password"]).to_csv("data/users.csv", index=False)
    pd.DataFrame(columns=["username", "password"]).to_csv("data/doctors.csv", index=False)
    pd.DataFrame(columns=["username", "reminder_date", "note", "created_at"]).to_csv("data/reminders.csv", index=False)
    
    print("✅ All data cleared successfully!")
    print("✅ New empty databases created!")
    print("🎯 You can now register new users with clean data.")

if __name__ == "__main__":
    clear_all_data()