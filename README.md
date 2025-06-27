# Detecting-Fraud-transactions-GNN
E.	Fraud Prediction for New Transactions
To evaluate new or unseen transactions, the model integrates them into the existing graph structure by:
1)	Placing the transaction at the end of the dataset.
2)	Using the same encoder and scaler to recalculate the graph’s features. 
3)	Join the new node with previous transactions that came from the same sender.
4)	Estimating the possibility of fraud with the help of the trained GNN network.
The architecture of the Graph Attention Network (GAT)  is just a simple version designed for fraud detection. The model works with the features of each node and the connections between them shown by edges. Input data go through the first GAT convolution layer which pays attention to the edges to merge neighborhood data.The presence of ELU activation in a model leads to a non-linear curve. Using the second GAT convolution layer, the system aims to make the node embeddings more clear. At the last step, a sigmoid function is applied to the product’s output representation to estimate the possibility that it is fraud.
