import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

class AirlineDataPreprocessor:
    def __init__(self):
        self.label_encoders = {}
        self.categorical_cols = ['airline', 'source_city', 'destination_city', 
                                 'departure_time', 'class', 'stops', 'route']

    def engineer_features(self, df):
        """Creates derived features."""
        df = df.copy()
        # Route combination
        df['route'] = df['source_city'] + '-' + df['destination_city']
        
        # Demand indicator (binary) - high demand for flights within a week
        df['high_demand'] = (df['days_left'] <= 7).astype(int)
        
        # Time bucket encoding
        time_mapping = {'Morning': 1, 'Afternoon': 2, 'Evening': 3, 'Night': 0}
        df['time_bucket'] = df['departure_time'].map(time_mapping)
        
        # Duration category (fixed bins for single row inference)
        df['duration_cat'] = pd.cut(df['duration'], bins=[0, 120, 240, 480, float('inf')], labels=[0, 1, 2, 3]).astype(int)
        
        return df

    def fit_transform(self, df):
        df = self.engineer_features(df)
        
        for col in self.categorical_cols:
            le = LabelEncoder()
            # Convert to string to avoid NaNs breaking it
            df[col] = le.fit_transform(df[col].astype(str))
            self.label_encoders[col] = le
            
        return df

    def transform(self, df):
        df = self.engineer_features(df)
        
        for col in self.categorical_cols:
            if col in self.label_encoders:
                le = self.label_encoders[col]
                # Handle unseen categories by filling with a known category or modal value (simplification)
                # For a robust production app, unseen labels should map to an "unknown" bucket
                df[col] = df[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
                df[col] = le.transform(df[col].astype(str))
                
        return df
        
    def save(self, filepath='models/preprocessor.pkl'):
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.label_encoders, filepath)
        
    def load(self, filepath='models/preprocessor.pkl'):
        self.label_encoders = joblib.load(filepath)

def load_and_split_data(filepath='data/airline_data.csv'):
    df = pd.read_csv(filepath)
    
    # Split features and targets
    X = df.drop(['price', 'delay'], axis=1)
    y_price = df['price']
    y_delay = df['delay']
    
    X_train, X_test, yp_train, yp_test, yd_train, yd_test = train_test_split(
        X, y_price, y_delay, test_size=0.2, random_state=42
    )
    
    return X_train, X_test, yp_train, yp_test, yd_train, yd_test

if __name__ == "__main__":
    print("Testing preprocessor...")
    df = pd.read_csv('data/airline_data.csv').head(100)
    prep = AirlineDataPreprocessor()
    df_transformed = prep.fit_transform(df)
    print("Transformed Data Shapes:", df_transformed.shape)
    print("Derived Features Successfully Created.")
