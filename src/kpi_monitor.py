import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta
import azure.storage.blob as blob

class ConstructionKPIMonitor:
    def __init__(self, connection_string):
        self.blob_client = blob.BlobServiceClient.from_connection_string(connection_string)
        self.rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        
    def extract_project_data(self, container_name, blob_name):
        """Extract project data from Azure Blob Storage"""
        blob_client = self.blob_client.get_blob_client(
            container=container_name, 
            blob=blob_name
        )
        
        # Download and parse the data
        data = blob_client.download_blob().readall()
        df = pd.read_csv(io.StringIO(data.decode('utf-8')))
        return df
    
    def calculate_kpis(self, df):
        """Calculate key performance indicators"""
        kpis = {}
        
        # Schedule Performance Index (SPI)
        kpis['spi'] = df['actual_progress'] / df['planned_progress']
        
        # Cost Performance Index (CPI)
        kpis['cpi'] = df['earned_value'] / df['actual_cost']
        
        # Delay Risk Score
        kpis['delay_risk'] = self.predict_delay_risk(df)
        
        # Resource Utilization
        kpis['resource_utilization'] = df['resources_used'] / df['resources_allocated']
        
        return kpis
    
    def predict_delay_risk(self, df):
        """Predict delay risk using machine learning"""
        features = ['weather_days', 'resource_availability', 'complexity_score', 
                   'supplier_delays', 'change_orders']
        
        X = df[features].fillna(0)
        
        # Train model if not already trained
        if not hasattr(self, 'model_trained'):
            # Historical data for training (simplified example)
            y_train = df['actual_delay_days'].fillna(0)
            self.rf_model.fit(X, y_train)
            self.model_trained = True
        
        delay_predictions = self.rf_model.predict(X)
        return np.clip(delay_predictions, 0, 100)  # Risk score 0-100
    
    def generate_alerts(self, kpis, thresholds):
        """Generate automated alerts based on KPI thresholds"""
        alerts = []
        
        if np.mean(kpis['spi']) < thresholds['spi_min']:
            alerts.append({
                'type': 'warning',
                'message': f"Schedule performance below threshold: {np.mean(kpis['spi']):.2f}",
                'priority': 'high'
            })
        
        if np.mean(kpis['delay_risk']) > thresholds['delay_risk_max']:
            alerts.append({
                'type': 'critical',
                'message': f"High delay risk detected: {np.mean(kpis['delay_risk']):.1f}%",
                'priority': 'critical'
            })
        
        return alerts

# Usage example
monitor = ConstructionKPIMonitor(connection_string="your_azure_connection_string")
project_data = monitor.extract_project_data("construction-data", "project_metrics.csv")
kpis = monitor.calculate_kpis(project_data)
alerts = monitor.generate_alerts(kpis, {
    'spi_min': 0.9,
    'delay_risk_max': 70
})
