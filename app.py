
from flask import Flask, request, render_template
import torch
import pandas as pd
import numpy as np


from Model import FraudDetectionGNN, load_dataset, create_graph, train_model, predict_transaction

app = Flask(__name__)

# Global variables
model = None
graph_data = None
encoder = None
scaler = None

def initialize_model():
    global model, graph_data, encoder, scaler
    csv_file_path = "dataset.csv"
    try:
        print("Starting model training...")
        model, graph_data, encoder, scaler = train_model(csv_file_path)
        print("Model trained and loaded successfully.")
    except Exception as e:
        print(f"Error initializing model: {str(e)}")
        model = None

initialize_model()

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or graph_data is None or encoder is None or scaler is None:
        return render_template('result.html', error="Model not initialized. Please check server logs.")

    try:
        # Collect form data
        transaction = {
            'step': request.form.get('step'),
            'type': request.form.get('type'),
            'amount': request.form.get('amount'),
            'nameOrig': request.form.get('nameOrig'),
            'oldbalanceOrg': request.form.get('oldbalanceOrg'),
            'newbalanceOrig': request.form.get('newbalanceOrig'),
            'nameDest': request.form.get('nameDest'),
            'oldbalanceDest': request.form.get('oldbalanceDest'),
            'newbalanceDest': request.form.get('newbalanceDest')
        }
        print("Received features:", transaction)

        # Validate inputs
        required_fields = list(transaction.keys())
        if not all(transaction[field] for field in required_fields):
            return render_template('result.html', error="All fields are required.")

        numeric_fields = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
        for field in numeric_fields:
            try:
                transaction[field] = float(transaction[field])
                if transaction[field] < 0 or (field == 'step' and transaction[field] < 1):
                    raise ValueError
            except (ValueError, TypeError):
                return render_template('result.html', error=f"{field} must be a valid positive number.")

        # Predict using modified predict_transaction logic
        new_tx_df = pd.DataFrame([transaction])
        orig_df = pd.read_csv("dataset.csv")
        full_df = pd.concat([orig_df, new_tx_df], ignore_index=True)

        full_features, _, _, _ = load_dataset(encoder=encoder, scaler=scaler, fit=False, df=full_df)
        
        device = next(model.parameters()).device
        graph_data.x = torch.tensor(full_features, dtype=torch.float).to(device)

        senders = full_df['nameOrig'].astype('category').cat.codes
        new_node_index = len(full_df) - 1
        sender_id = senders.iloc[new_node_index]
        
        edges = []
        for i, sender in enumerate(senders[:-1]):
            if sender == sender_id:
                edges.append([i, new_node_index])
                edges.append([new_node_index, i])
        
        if edges:
            new_edges = torch.tensor(edges, dtype=torch.long).t().contiguous().to(device)
            graph_data.edge_index = torch.cat([graph_data.edge_index.to(device), new_edges], dim=1)
        
        model.eval()
        with torch.no_grad():
            output = model(graph_data.x, graph_data.edge_index)
            prob = output[new_node_index].item()

        print(f"Prediction successful, probability: {prob}")
        result = {
            'probability': prob,
            'is_fraud': prob > 0.5,
            'message': '🔴 Likely FRAUDULENT Transaction' if prob > 0.5 else '🟢 Likely LEGITIMATE Transaction'
        }
        return render_template('result.html', result=result)
    
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return render_template('result.html', error=str(e))

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
