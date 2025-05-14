"""
Smart Resource Prioritizer (SRP)
--------------------------------
An ML-powered tool that helps decision-makers allocate limited resources during crisis situations
by analyzing multiple factors and recommending optimal distribution strategies.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


class SmartResourcePrioritizer:
    """
    Main class for the Smart Resource Prioritizer application.
    Handles data generation, model training, and resource allocation recommendations.
    """

    def __init__(self):
        """Initialize the SRP system with empty data and model attributes."""
        self.data = None
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False

    def generate_sample_data(self, num_locations=50):
        """
        Generate synthetic crisis data for demonstration purposes.

        Parameters:
        -----------
        num_locations : int
            Number of affected locations to generate

        Returns:
        --------
        DataFrame containing the generated crisis data
        """
        np.random.seed(42)  # For reproducibility

        # Generate random data
        locations = [f"Location_{i}" for i in range(1, num_locations + 1)]
        population = np.random.randint(1000, 50000, size=num_locations)
        severity = np.random.uniform(1, 10, size=num_locations)  # 1-10 scale
        infrastructure_damage = np.random.uniform(0, 1, size=num_locations)  # 0-1 scale
        medical_needs = np.random.uniform(0, 1, size=num_locations)  # 0-1 scale
        food_needs = np.random.uniform(0, 1, size=num_locations)  # 0-1 scale
        accessibility = np.random.uniform(0, 1, size=num_locations)  # 0-1 scale (higher is more accessible)

        # Calculate "true" optimal resource allocation (our target variable)
        # This formula creates a realistic but complex relationship between features
        resource_allocation = (
                population / 10000 * 0.3 +
                severity * 0.25 +
                infrastructure_damage * 10000 * 0.15 +
                medical_needs * 10000 * 0.15 +
                food_needs * 10000 * 0.1 +
                (1 - accessibility) * 5000 * 0.05  # Less accessible areas need more resources
        )

        # Create DataFrame
        self.data = pd.DataFrame({
            'location': locations,
            'population': population,
            'severity': severity,
            'infrastructure_damage': infrastructure_damage,
            'medical_needs': medical_needs,
            'food_needs': food_needs,
            'accessibility': accessibility,
            'optimal_allocation': resource_allocation
        })

        print(f"Generated sample crisis data for {num_locations} locations.")
        return self.data

    def train_model(self):
        """
        Train the machine learning model on the generated or loaded data.

        Returns:
        --------
        Model accuracy metrics
        """
        if self.data is None:
            print("No data available. Generating sample data...")
            self.generate_sample_data()

        # Prepare features and target
        X = self.data[['population', 'severity', 'infrastructure_damage',
                       'medical_needs', 'food_needs', 'accessibility']]
        y = self.data['optimal_allocation']

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train the model
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X_train_scaled, y_train)

        # Evaluate the model
        y_pred = self.model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)

        self.is_trained = True
        print(f"Model trained successfully. Root Mean Squared Error: {rmse:.2f}")
        return {'rmse': rmse}

    def predict_allocation(self, new_data=None):
        """
        Predict optimal resource allocation for new crisis data.

        Parameters:
        -----------
        new_data : DataFrame, optional
            New crisis data to make predictions on. If None, uses existing data.

        Returns:
        --------
        DataFrame with predictions
        """
        if not self.is_trained:
            print("Model not trained yet. Training now...")
            self.train_model()

        # Use existing data if no new data is provided
        if new_data is None:
            data_to_predict = self.data
        else:
            data_to_predict = new_data

        # Extract features
        X = data_to_predict[['population', 'severity', 'infrastructure_damage',
                             'medical_needs', 'food_needs', 'accessibility']]

        # Scale features
        X_scaled = self.scaler.transform(X)

        # Make predictions
        predictions = self.model.predict(X_scaled)

        # Add predictions to the data
        result = data_to_predict.copy()
        result['predicted_allocation'] = predictions

        return result

    def visualize_allocation(self, data=None):
        """
        Visualize the resource allocation recommendations.

        Parameters:
        -----------
        data : DataFrame, optional
            Data with allocation predictions to visualize. If None, makes predictions on existing data.
        """
        if data is None:
            data = self.predict_allocation()

        # Sort by predicted allocation for better visualization
        data_sorted = data.sort_values('predicted_allocation', ascending=False).head(10)

        # Create a horizontal bar chart
        plt.figure(figsize=(12, 8))
        plt.barh(data_sorted['location'], data_sorted['predicted_allocation'], color='skyblue')
        plt.xlabel('Recommended Resource Allocation')
        plt.ylabel('Location')
        plt.title('Top 10 Locations by Resource Allocation Priority')
        plt.tight_layout()

        # Create output directory if it doesn't exist
        if not os.path.exists('output'):
            os.makedirs('output')

        # Save the figure
        plt.savefig('output/resource_allocation.png')
        plt.close()

        print("Visualization saved to 'output/resource_allocation.png'")

        # Also print the top 5 locations that need resources
        print("\nTop 5 Priority Locations:")
        for i, (_, row) in enumerate(data_sorted.head(5).iterrows(), 1):
            print(f"{i}. {row['location']}: {row['predicted_allocation']:.2f} units")

    def feature_importance(self):
        """
        Analyze and visualize feature importance from the trained model.

        Returns:
        --------
        DataFrame with feature importance scores
        """
        if not self.is_trained:
            print("Model not trained yet. Training now...")
            self.train_model()

        # Get feature importance
        features = ['population', 'severity', 'infrastructure_damage',
                    'medical_needs', 'food_needs', 'accessibility']
        importance = self.model.feature_importances_

        # Create DataFrame for feature importance
        feature_importance = pd.DataFrame({
            'Feature': features,
            'Importance': importance
        }).sort_values('Importance', ascending=False)

        # Visualize feature importance
        plt.figure(figsize=(10, 6))
        plt.barh(feature_importance['Feature'], feature_importance['Importance'], color='lightgreen')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title('Feature Importance in Resource Allocation Decision')
        plt.tight_layout()

        # Save the figure
        if not os.path.exists('output'):
            os.makedirs('output')
        plt.savefig('output/feature_importance.png')
        plt.close()

        print("Feature importance visualization saved to 'output/feature_importance.png'")
        return feature_importance

    def add_new_location(self, location_name, population, severity, infrastructure_damage,
                         medical_needs, food_needs, accessibility):
        """
        Add a new crisis location and predict its resource allocation.

        Parameters:
        -----------
        location_name : str
            Name or identifier of the location
        population : int
            Population affected
        severity : float
            Crisis severity (1-10)
        infrastructure_damage : float
            Level of infrastructure damage (0-1)
        medical_needs : float
            Level of medical needs (0-1)
        food_needs : float
            Level of food/water needs (0-1)
        accessibility : float
            Accessibility of the location (0-1), higher is more accessible

        Returns:
        --------
        DataFrame with the new location and its predicted allocation
        """
        # Create DataFrame for new location
        new_location = pd.DataFrame({
            'location': [location_name],
            'population': [population],
            'severity': [severity],
            'infrastructure_damage': [infrastructure_damage],
            'medical_needs': [medical_needs],
            'food_needs': [food_needs],
            'accessibility': [accessibility]
        })

        # If model exists, predict allocation
        if self.is_trained:
            # Extract features
            X = new_location[['population', 'severity', 'infrastructure_damage',
                              'medical_needs', 'food_needs', 'accessibility']]

            # Scale features
            X_scaled = self.scaler.transform(X)

            # Make prediction
            prediction = self.model.predict(X_scaled)
            new_location['predicted_allocation'] = prediction[0]

            print(f"Predicted allocation for {location_name}: {prediction[0]:.2f} units")
        else:
            print("Model not trained yet. Cannot make predictions.")

        return new_location


def run_demo():
    """Run a demonstration of the Smart Resource Prioritizer system."""
    print("=" * 50)
    print("SMART RESOURCE PRIORITIZER (SRP) - DEMONSTRATION")
    print("=" * 50)

    # Create SRP instance
    srp = SmartResourcePrioritizer()

    # Generate sample data
    print("\n[1] Generating sample crisis data...")
    data = srp.generate_sample_data(num_locations=50)
    print(f"Sample of the generated data:")
    print(data[['location', 'population', 'severity']].head())

    # Train the model
    print("\n[2] Training the resource allocation model...")
    metrics = srp.train_model()

    # Make predictions
    print("\n[3] Making resource allocation predictions...")
    predictions = srp.predict_allocation()
    print("Predictions sample:")
    print(predictions[['location', 'predicted_allocation']].head())

    # Visualize allocations
    print("\n[4] Visualizing resource allocation priorities...")
    srp.visualize_allocation(predictions)

    # Analyze feature importance
    print("\n[5] Analyzing factors influencing resource allocation...")
    importance = srp.feature_importance()
    print("Feature importance:")
    print(importance)

    # Add a new crisis location
    print("\n[6] Adding a new crisis location...")
    new_location = srp.add_new_location(
        location_name="Emergency_Zone_A",
        population=25000,
        severity=8.7,
        infrastructure_damage=0.75,
        medical_needs=0.9,
        food_needs=0.8,
        accessibility=0.4
    )

    print("\n" + "=" * 50)
    print("DEMONSTRATION COMPLETE")
    print("=" * 50)
    print("\nThe Smart Resource Prioritizer (SRP) has successfully:")
    print("1. Generated and analyzed crisis data across 50 locations")
    print("2. Trained a machine learning model to predict optimal resource allocation")
    print("3. Identified the highest-priority locations needing assistance")
    print("4. Determined which factors have the greatest impact on resource needs")
    print("5. Demonstrated the ability to assess new crisis locations in real-time")
    print("\nThis project demonstrates application of machine learning to solve real-world")
    print("humanitarian challenges through data analysis and predictive modeling.")


if __name__ == "__main__":
    run_demo()