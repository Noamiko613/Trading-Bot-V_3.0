"""
Test script for Gemini reviewer
Run this to test the review generation without waiting for scheduled times.
"""

import os
import sys
from utils.gemini_reviewer import ReviewScheduler

if __name__ == "__main__":
    # Use the API key from environment or default
    api_key = os.getenv("GEMINI_API_KEY", "AIzaSyAtxfgJJQT4FnfhTQ5S-5iBKmVOaDy1eQc")
    
    scheduler = ReviewScheduler(
        gemini_api_key=api_key,
        model_dir="models/rl_models",
        state_file="models/rl_models/rl_training_state.json",
        db_path="sim_results/trades.db",
        to_email="noamiko613@gmail.com",
    )
    
    print("Generating test review...")
    review = scheduler.generate_review_now("daily")
    
    print("\n" + "="*60)
    print("REVIEW GENERATED:")
    print("="*60)
    print(review)
    print("="*60)
    print("\nCheck your email at noamiko613@gmail.com for the full review.")
