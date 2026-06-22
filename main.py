import pandas as pd
import numpy as np
import joblib
import os
import sys
from utils import add_derived_features

# Paths to artifacts
MODEL_PATH = 'model/advanced_model.joblib'

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def load_advanced_pipeline():
    """Load the trained Scikit-Learn Pipeline from disk."""
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file '{MODEL_PATH}' not found. Please run 'trainer.py' first.")
        sys.exit(1)
        
    pipeline = joblib.load(MODEL_PATH)
    return pipeline

def get_user_input_from_cli():
    """Interactively get profile data from the user."""
    print("\n--- Social Media Profile Data Input ---")
    print("Please provide the following profile information:")
    
    try:
        profile_pic = input("1. Profile picture (Yes/No): ").strip().capitalize()
        ratio_numlen_username = float(input("2. Ratio of digits in username (0.0 to 1.0): "))
        len_fullname = int(input("3. Length of full name: "))
        ratio_numlen_fullname = float(input("4. Ratio of digits in full name (0.0 to 1.0): "))
        sim_name_username = input("5. Similarity between name and username (No match/Partial match/Full match): ").strip().capitalize()
        if "Partial" in sim_name_username: sim_name_username = "Partial match"
        elif "Full" in sim_name_username: sim_name_username = "Full match"
        else: sim_name_username = "No match"
            
        len_desc = int(input("6. Length of biography description: "))
        extern_url = input("7. External URL in bio (Yes/No): ").strip().capitalize()
        private = input("8. Account is private (Yes/No): ").strip().capitalize()
        num_posts = int(input("9. Number of posts: "))
        num_followers = int(input("10. Number of followers: "))
        num_following = int(input("11. Number of following: "))
        
        # Note: 'follower_following_ratio' is handled by the pipeline internally via FunctionTransformer
        # We just need to provide the raw features in a DataFrame.
        
        data = {
            'profile_pic': profile_pic,
            'ratio_numlen_username': ratio_numlen_username,
            'len_fullname': len_fullname,
            'ratio_numlen_fullname': ratio_numlen_fullname,
            'sim_name_username': sim_name_username,
            'len_desc': len_desc,
            'extern_url': extern_url,
            'private': private,
            'num_posts': num_posts,
            'num_followers': num_followers,
            'num_following': num_following
        }
        
        return pd.DataFrame([data])
    
    except Exception as e:
        print(f"Invalid input: {e}")
        return None

def main():
    clear_screen()
    print("====================================================")
    print("    FAKE PROFILE DETECTION SYSTEM (ADVANCED ML)")
    print("====================================================")
    
    pipeline = load_advanced_pipeline()
    
    while True:
        df_input = get_user_input_from_cli()
        
        if df_input is not None:
            # Predict using the loaded pipeline (deals with scaling and OHE automatically)
            prediction = pipeline.predict(df_input)[0]
            probability = pipeline.predict_proba(df_input)[0]
            
            print("\n----------------------------------------------------")
            print("        CLASSIFICATION RESULT")
            print("----------------------------------------------------")
            
            if prediction == 1:
                print(f"RESULT: !!! FAKE PROFILE DETECTED !!!")
                confidence = probability[1]
            else:
                print(f"RESULT: >>> REAL PROFILE <<<")
                confidence = probability[0]
                
            print(f"Confidence Level: {confidence*100:.2f}%")
            print("----------------------------------------------------")
            
        cont = input("\nDo you want to test another profile? (y/n): ").lower()
        if cont != 'y':
            break

    print("\nThank you for using the Fake Profile Detection System!")

if __name__ == "__main__":
    main()
