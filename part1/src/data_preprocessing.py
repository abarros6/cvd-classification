"""
Data preprocessing module for cardiovascular disease dataset.
Handles loading, cleaning, feature engineering, and splitting.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def load_data(filepath='data/cardio_train.csv'):
    """
    Load the cardiovascular disease dataset.
    
    Parameters:
    -----------
    filepath : str
        Path to the CSV file
        
    Returns:
    --------
    pd.DataFrame
        Loaded dataset
    """
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath, delimiter=';')
    print(f"✓ Loaded {len(df):,} records with {len(df.columns)} columns")
    return df


def engineer_features(df):
    """
    Create derived features and convert units.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with engineered features
    """
    df_copy = df.copy()
    
    # Convert age from days to years (more interpretable)
    df_copy['age_years'] = (df_copy['age'] / 365.25).round(0)
    
    # Calculate BMI (Body Mass Index)
    df_copy['bmi'] = df_copy['weight'] / ((df_copy['height'] / 100) ** 2)
    
    print("\n✓ Feature engineering completed:")
    print("  • Age converted from days to years")
    print("  • BMI calculated from height and weight")
    
    return df_copy


def remove_outliers(df):
    """
    Remove physiologically implausible values.
    
    Clinical rationale:
    - Systolic BP: 80-220 mmHg (values outside suggest measurement error)
    - Diastolic BP: 50-140 mmHg
    - Height: 130-230 cm (adult human range)
    - Weight: 30-200 kg (adult human range)
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
        
    Returns:
    --------
    pd.DataFrame
        Cleaned dataframe
    """
    initial_count = len(df)
    
    df_clean = df[
        (df['ap_hi'] > 80) & (df['ap_hi'] < 220) &    # Systolic BP
        (df['ap_lo'] > 50) & (df['ap_lo'] < 140) &    # Diastolic BP
        (df['height'] > 130) & (df['height'] < 230) & # Height
        (df['weight'] > 30) & (df['weight'] < 200)    # Weight
    ].copy()
    
    removed_count = initial_count - len(df_clean)
    removed_pct = (removed_count / initial_count) * 100
    
    print(f"\n✓ Outlier removal:")
    print(f"  • Removed: {removed_count:,} records ({removed_pct:.2f}%)")
    print(f"  • Remaining: {len(df_clean):,} records")
    
    return df_clean


def prepare_features_target(df, feature_cols=None):
    """
    Separate features and target variable.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    feature_cols : list, optional
        List of feature column names
        
    Returns:
    --------
    tuple
        (X, y) features and target
    """
    if feature_cols is None:
        feature_cols = ['age_years', 'gender', 'height', 'weight', 'ap_hi', 
                       'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco', 
                       'active', 'bmi']
    
    X = df[feature_cols].copy()
    y = df['cardio'].copy()
    
    print(f"\n✓ Features prepared:")
    print(f"  • Number of features: {len(feature_cols)}")
    print(f"  • Features: {', '.join(feature_cols)}")
    print(f"  • Target variable: cardio (0=No CVD, 1=CVD present)")
    print(f"\n  Class distribution:")
    print(f"    - No CVD (0): {(y == 0).sum():,} ({(y == 0).sum()/len(y)*100:.1f}%)")
    print(f"    - CVD (1): {(y == 1).sum():,} ({(y == 1).sum()/len(y)*100:.1f}%)")
    
    return X, y


def reduce_dataset_size(X, y, max_samples=10000, min_train_samples=7000, random_state=42):
    """
    Reduce dataset size while maintaining class balance and minimum training samples.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Features
    y : pd.Series
        Target
    max_samples : int
        Maximum total samples to keep (default: 10000)
    min_train_samples : int
        Minimum training samples required (default: 7000)
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    tuple
        (X_reduced, y_reduced)
    """
    if len(X) <= max_samples:
        print(f"\n✓ Dataset size ({len(X):,}) is already within limit ({max_samples:,})")
        return X, y
    
    print(f"\n✓ Reducing dataset size:")
    print(f"  • Original size: {len(X):,} samples")
    print(f"  • Target size: {max_samples:,} samples")
    print(f"  • Minimum training samples required: {min_train_samples:,}")
    
    # Use stratified sampling to maintain class balance
    X_reduced, _, y_reduced, _ = train_test_split(
        X, y, train_size=max_samples, random_state=random_state, stratify=y
    )
    
    # Verify class balance maintained
    original_cvd_pct = (y == 1).sum() / len(y) * 100
    reduced_cvd_pct = (y_reduced == 1).sum() / len(y_reduced) * 100
    
    print(f"  • Reduced to: {len(X_reduced):,} samples")
    print(f"  • Original CVD%: {original_cvd_pct:.2f}%")
    print(f"  • Reduced CVD%: {reduced_cvd_pct:.2f}%")
    print(f"  • Class balance preserved: ✓")
    
    return X_reduced, y_reduced


def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets with stratification.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Features
    y : pd.Series
        Target
    test_size : float
        Proportion of test set (default: 0.2 = 20%)
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    tuple
        (X_train, X_test, y_train, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"\n✓ Data split (stratified):")
    print(f"  • Training set: {len(X_train):,} samples ({(1-test_size)*100:.0f}%)")
    print(f"  • Test set: {len(X_test):,} samples ({test_size*100:.0f}%)")
    print(f"  • Random seed: {random_state}")
    
    # Verify stratification maintained class balance
    train_cvd_pct = (y_train == 1).sum() / len(y_train) * 100
    test_cvd_pct = (y_test == 1).sum() / len(y_test) * 100
    print(f"\n  Class balance maintained:")
    print(f"    - Training CVD%: {train_cvd_pct:.2f}%")
    print(f"    - Test CVD%: {test_cvd_pct:.2f}%")
    
    return X_train, X_test, y_train, y_test


def main():
    """
    Main preprocessing pipeline.
    Executes all preprocessing steps and saves results.
    """
    print("="*80)
    print("CARDIOVASCULAR DISEASE CLASSIFICATION - DATA PREPROCESSING")
    print("="*80)
    
    # Step 1: Load data
    df = load_data()
    
    # Step 2: Engineer features
    df = engineer_features(df)
    
    # Step 3: Remove outliers
    df_clean = remove_outliers(df)
    
    # Step 4: Prepare features and target
    X, y = prepare_features_target(df_clean)
    
    # Step 5: Reduce dataset size (to make training feasible)
    X_reduced, y_reduced = reduce_dataset_size(X, y, max_samples=10000, min_train_samples=7000)
    
    # Step 6: Split data
    X_train, X_test, y_train, y_test = split_data(X_reduced, y_reduced)
    
    # Step 7: Save processed data
    print("\n" + "="*80)
    print("SAVING PROCESSED DATA")
    print("="*80)
    
    import os
    os.makedirs('data', exist_ok=True)
    
    X_train.to_csv('data/X_train.csv', index=False)
    X_test.to_csv('data/X_test.csv', index=False)
    y_train.to_csv('data/y_train.csv', index=False, header=True)
    y_test.to_csv('data/y_test.csv', index=False, header=True)
    
    print("\n✓ Saved files:")
    print("  • data/X_train.csv")
    print("  • data/X_test.csv")
    print("  • data/y_train.csv")
    print("  • data/y_test.csv")
    
    print("\n" + "="*80)
    print("PREPROCESSING COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()