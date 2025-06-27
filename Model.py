import torch
import torch.nn.functional as F
import pandas as pd
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
from imblearn.over_sampling import SMOTE

# Define the GNN model
class FraudDetectionGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=3):
        super(FraudDetectionGNN, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, concat=True)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False)
        self.prob_layer = torch.nn.Linear(out_channels, 1)

    def forward(self, x, edge_index):
        x = F.elu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        prob = torch.sigmoid(self.prob_layer(x))
        return prob

# Load and preprocess dataset
def load_dataset(csv_path=None, encoder=None, scaler=None, fit=True, df=None):
    if df is None:
        if csv_path is None:
            raise ValueError("Must provide either csv_path or df.")
        df = pd.read_csv(csv_path)

    df.columns = df.columns.astype(str)

    if encoder is None:
        encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
        type_encoded = encoder.fit_transform(df[['type']])
    else:
        type_encoded = encoder.transform(df[['type']])

    features = df.drop(columns=['nameOrig', 'nameDest', 'isFraud', 'isFlaggedFraud', 'type'], errors='ignore')
    features = features.apply(pd.to_numeric, errors='coerce').fillna(0)
    features = pd.concat([pd.DataFrame(type_encoded), features.reset_index(drop=True)], axis=1)
    features = features.to_numpy()

    if scaler is None:
        scaler = StandardScaler()
        features = scaler.fit_transform(features)
    elif fit:
        features = scaler.fit_transform(features)
    else:
        features = scaler.transform(features)

    labels = df['isFraud'].values.astype(float) if 'isFraud' in df.columns else np.zeros(len(df))
    return features, labels, encoder, scaler

# Create graph structure
def create_graph(csv_path):
    df = pd.read_csv(csv_path)
    features, labels, _, _ = load_dataset(csv_path=csv_path)
    senders = df['nameOrig'].astype('category').cat.codes.values
    edge_index = []
    sender_dict = {}

    for idx, sender in enumerate(senders):
        if sender in sender_dict:
            edge_index.append((sender_dict[sender], idx))
            edge_index.append((idx, sender_dict[sender]))
        sender_dict[sender] = idx

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    return Data(x=torch.tensor(features, dtype=torch.float), edge_index=edge_index, y=torch.tensor(labels, dtype=torch.float).unsqueeze(1))

# Oversample fraud transactions
def oversample_fraud_transactions(features, labels):
    smote = SMOTE(sampling_strategy='minority', random_state=42)
    features_resampled, labels_resampled = smote.fit_resample(features, labels)
    return features_resampled, labels_resampled

# Train the GNN model
def train_model(csv_path, patience=5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    features, labels, encoder, scaler = load_dataset(csv_path=csv_path)
    features_resampled, labels_resampled = oversample_fraud_transactions(features, labels)

    data = create_graph(csv_path)
    data.x = torch.tensor(features_resampled, dtype=torch.float)
    data.y = torch.tensor(labels_resampled, dtype=torch.float).unsqueeze(1)

    model = FraudDetectionGNN(in_channels=data.x.shape[1], hidden_channels=32, out_channels=16).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    best_loss = float('inf')
    best_model_state = None
    patience_counter = 0

    for epoch in range(40):  # Train for max 100 epochs
        model.train()
        optimizer.zero_grad()
        prob = model(data.x.to(device), data.edge_index.to(device)).squeeze()
        loss = F.binary_cross_entropy(prob, data.y.squeeze().to(device))
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

        # Early stopping logic
        if loss.item() < best_loss - 1e-4:
            best_loss = loss.item()
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1  

        if patience_counter >= patience:
            print(f"⏹️ Early stopping at epoch {epoch}. Best Loss: {best_loss:.4f}")
            break

    # Load best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Evaluate final model
    model.eval()
    with torch.no_grad():
        probs = model(data.x.to(device), data.edge_index.to(device)).squeeze().cpu().numpy()
        predicted = (probs > 0.5).astype(int)
        acc = accuracy_score(data.y.squeeze().cpu().numpy(), predicted)
        print(f"\n✅ Final Training Accuracy: {acc:.4f}")
        print("\nClassification Report:")
        print(classification_report(data.y.squeeze().cpu().numpy(), predicted, target_names=['Legitimate', 'Fraudulent']))

    return model, data, encoder, scaler

# Predict a new transaction
def predict_transaction(model, data, transaction_dict, csv_path, encoder, scaler):
    new_tx_df = pd.DataFrame([transaction_dict])
    orig_df = pd.read_csv(csv_path)
    full_df = pd.concat([orig_df, new_tx_df], ignore_index=True)

    # Reuse encoder/scaler to preprocess full features
    full_features, _, _, _ = load_dataset(encoder=encoder, scaler=scaler, fit=False, df=full_df)

    # Update graph features
    device = next(model.parameters()).device
    data.x = torch.tensor(full_features, dtype=torch.float).to(device)

    # Add new edges for the new transaction (simple heuristic: link to same sender)
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
        data.edge_index = torch.cat([data.edge_index.to(device), new_edges], dim=1)
    
    model.eval()
    with torch.no_grad():
        output = model(data.x, data.edge_index)
        prob = output[new_node_index].item()
        print(f"\nSmoothed Fraud Probability: {prob:.4f}")
        if prob > 0.5:
            print("🔴 Likely FRAUDULENT Transaction.")
        else:
            print("🟢 Likely LEGITIMATE Transaction.")

# Main entry point
if __name__ == "__main__":
    csv_file_path = "dataset.csv"
    trained_model, graph_data, enc, sc = train_model(csv_file_path)
#1,TRANSFER,181,C1305486145,181,0,C553264065,0,0,1,0
    new_transaction = {
    'step': 1,
    'type': 'TRANSFER',
    'amount': 1000,  # Amount is equal to the old balance of the origin account
    'nameOrig': 'C1234567890',  # Origin account (random ID)
    'oldbalanceOrg': 1000,  # Origin account balance before the transfer
    'newbalanceOrig': 0,  # Origin account balance after the transfer
    'nameDest': 'C9876543210',  # Destination account (random ID)
    'oldbalanceDest': 122, 
    'newbalanceDest': 457  
}
    predict_transaction(trained_model, graph_data, new_transaction, csv_file_path, encoder=enc, scaler=sc)

   