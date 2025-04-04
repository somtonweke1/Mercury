from sklearn.metrics import f1_score
import xgboost as xgb
import pandas as pd

def test_improvement(raw_data, test_data, test_labels):
    """
    Compare baseline mean imputation vs Mercury hybrid imputation.
    
    Args:
        raw_data (pd.DataFrame): Training data with missing values
        test_data (pd.DataFrame): Test data
        test_labels (np.array): Ground truth labels
        
    Returns:
        dict: Performance metrics comparison
    """
    # Baseline (mean imputation)
    base_data = raw_data.fillna(raw_data.mean())
    base_model = xgb.XGBClassifier()
    base_model.fit(base_data)
    base_f1 = f1_score(test_labels, base_model.predict(test_data))
    
    # Mercury imputation
    mercury_imputer = MercuryImputer()
    mercury_data = mercury_imputer.process(raw_data)
    mercury_model = xgb.XGBClassifier()
    mercury_model.fit(mercury_data)
    mercury_f1 = f1_score(test_labels, mercury_model.predict(test_data))
    
    improvement = (mercury_f1 - base_f1) / base_f1
    
    return {
        "baseline_f1": base_f1,
        "mercury_f1": mercury_f1,
        "improvement": f"{improvement:.0%}"
    } 