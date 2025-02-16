import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import gpytorch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# Import from GP.py
from GP import (
    load_support_dataset,
    GPUncertainty as GPModel,
    train_survival_model,
    estimate_uncertainty,
    DiscreteFailureTimeNLL
)

# Load SUPPORT dataset (same as in GP.py)
def load_support_dataset(random_state=0):
    """Load and preprocess the SUPPORT dataset."""
    FILL_VALUES = {
        'alb': 3.5,
        'pafi': 333.3,
        'bili': 1.01,
        'crea': 1.01,
        'bun': 6.51,
        'wblc': 9.,
        'urine': 2502.
    }

    COLUMNS_TO_DROP = [
        'aps', 'sps', 'surv2m', 'surv6m', 'prg2m',
        'prg6m', 'dnr', 'dnrday', 'sfdm2', 'hospdead'
    ]

    df = (
        pd.read_csv('data/support2.csv')
        .drop(COLUMNS_TO_DROP,axis=1)
        .fillna(value=FILL_VALUES)
        .sample(frac=1, random_state=random_state)
    )

    # one-hot encode categorical variables
    df = pd.get_dummies(df, dummy_na=True)
    df = df.fillna(df.median())

    # standardize numeric columns
    numeric_cols = df.dtypes == 'float64'
    df.loc[:, numeric_cols] = df.loc[:, numeric_cols].transform(
        lambda x: (x - x.mean()) / x.std()
    )
    
    features = df.drop(['death', 'd.time'], axis=1).values.astype(float)
    event_indicator = df['death'].values.astype(float)
    event_time = df['d.time'].values.astype(float)

    return features, event_indicator, event_time

# Define a simple GP model using GPyTorch
class GPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        # 修改核函数，使用 RBF kernel with ARD
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=train_x.size(-1))
        )
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def get_bin_boundaries(train_y):
    # Create bin boundaries based on the training data
    min_time = train_y.min().item()
    max_time = train_y.max().item()
    return torch.linspace(min_time, max_time, 11)  # 10 bins

# Define a Monte Carlo Noise Injection (MCNI) model
class MCNIModel(nn.Module):
    def __init__(self, input_dim, num_bins=10, hidden_dim=64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_bins)
        )
    
    def forward(self, x):
        logits = self.network(x)
        return F.softmax(logits, dim=-1)

def train_mcni(train_X, train_y, bin_boundaries, event_indicators, num_epochs=100, lr=0.01):
    model = MCNIModel(input_dim=train_X.shape[1])
    criterion = DiscreteFailureTimeNLL(bin_boundaries)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        
        probs = model(train_X)
        
        loss = criterion(
            predictions=probs, 
            event_times=train_y,
            event_indicators=event_indicators
        )
        
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
    return model

def predict_mcni(model, x, bin_boundaries):
    model.eval()
    with torch.no_grad():
        probs = model(x)
        # Convert probabilities to time predictions using bin centers
        bin_centers = (bin_boundaries[1:] + bin_boundaries[:-1]) / 2
        predictions = torch.sum(probs * bin_centers.unsqueeze(0), dim=1)
        return predictions

# Train GP and MCNI models
def train_models():
    # Load data using existing function
    X, s, t = load_support_dataset()
    
    # Split data into train and test sets
    test_idx = len(X) * 4 // 5
    train_X, train_t = X[:test_idx], t[:test_idx]
    train_s = s[:test_idx]
    test_X, test_t = X[test_idx:], t[test_idx:]
    
    # Convert to tensors
    train_x = torch.tensor(train_X, dtype=torch.float32)
    train_y = torch.tensor(train_t, dtype=torch.float32)
    train_s = torch.tensor(train_s, dtype=torch.float32)
    test_x = torch.tensor(test_X, dtype=torch.float32)
    test_y = torch.tensor(test_t, dtype=torch.float32)
    
    # Get bin boundaries
    bin_boundaries = get_bin_boundaries(train_y)
    
    # Train GP model with modified training loop
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model_gp = GPModel(train_x, train_y, likelihood)
    
    # Use standard GP training objective
    model_gp.train()
    likelihood.train()
    
    # 修复参数重复问题
    model_params = []
    likelihood_params = []
    
    # 分别收集模型和似然函数的参数
    for param in model_gp.parameters():
        if not any(id(param) == id(p) for p in likelihood.parameters()):
            model_params.append(param)
    
    for param in likelihood.parameters():
        likelihood_params.append(param)
    
    # 创建优化器，确保参数不重复
    optimizer = torch.optim.Adam([
        {'params': model_params},
        {'params': likelihood_params}
    ], lr=0.01)
    
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model_gp)
    
    best_loss = float('inf')
    best_state = None
    patience = 5
    no_improve = 0
    
    for i in range(200):  # 增加训练轮数
        optimizer.zero_grad()
        output = model_gp(train_x)
        loss = -mll(output, train_y)  # 使用负对数似然
        loss.backward()
        optimizer.step()
        
        # Early stopping
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = {
                'model': model_gp.state_dict(),
                'likelihood': likelihood.state_dict()
            }
            no_improve = 0
        else:
            no_improve += 1
        
        if no_improve >= patience:
            print(f'Early stopping at iteration {i+1}')
            break
            
        if (i + 1) % 10 == 0:
            print(f'GP Iter {i+1}/200 - Loss: {loss.item():.3f}')
    
    # Load best model
    if best_state is not None:
        model_gp.load_state_dict(best_state['model'])
        likelihood.load_state_dict(best_state['likelihood'])
    
    # Train MCNI model
    model_mcni = train_mcni(train_x, train_y, bin_boundaries, train_s)
    
    return model_gp, likelihood, model_mcni, test_x, test_y

# Evaluate models
def evaluate():
    def get_timestamp():
        from datetime import datetime
        return datetime.now().strftime('%H%M%S')
    
    model_gp, likelihood, model_mcni, test_x, test_y = train_models()
    
    # Get bin boundaries for prediction
    bin_boundaries = get_bin_boundaries(test_y)
    timestamp = get_timestamp()
    
    # Evaluate models
    model_gp.eval()
    likelihood.eval()
    model_mcni.eval()
    
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        # GP predictions
        f_pred = model_gp(test_x)
        y_pred_gp = likelihood(f_pred)
        mean_gp = y_pred_gp.mean
        std_gp = y_pred_gp.stddev
        
        # MCNI predictions
        probs = model_mcni(test_x)
        bin_centers = (bin_boundaries[1:] + bin_boundaries[:-1]) / 2
        mean_mcni = torch.sum(probs * bin_centers.unsqueeze(0), dim=1)
        std_mcni = torch.sqrt(torch.sum(probs * (bin_centers.unsqueeze(0) - mean_mcni.unsqueeze(1))**2, dim=1))
        
        # Print results
        mse_gp = F.mse_loss(mean_gp, test_y)
        mse_mcni = F.mse_loss(mean_mcni, test_y)
        
        print(f'GP MSE: {mse_gp.item():.4f}')
        print(f'MCNI MSE: {mse_mcni.item():.4f}')
        print(f'GP Uncertainty (mean std): {std_gp.mean().item():.4f}')
        print(f'MCNI Uncertainty (mean std): {std_mcni.mean().item():.4f}')
        
        # Plot distributions
        plot_distributions_together(
            pred_means=[mean_gp, mean_mcni],
            labels=['GP', 'MCNI'],
            true_values=test_y,
            method=f"zscore_{timestamp}"
        )
        
        # Plot uncertainties as separate scatter plots
        plot_uncertainty_scatter(
            predictions=[mean_gp, mean_mcni],
            uncertainties=[std_gp, std_mcni],
            labels=['GP', 'MCNI'],
            method=f"zscore_{timestamp}"
        )

def plot_uncertainty_scatter(predictions, uncertainties, labels, method="zscore"):
    # 为每个模型创建单独的图
    for pred, unc, label in zip(predictions, uncertainties, labels):
        plt.figure(figsize=(12, 6))
        
        if isinstance(pred, torch.Tensor):
            pred = pred.detach().cpu().numpy()
        if isinstance(unc, torch.Tensor):
            unc = unc.detach().cpu().numpy()
            
        plt.scatter(pred, unc, alpha=0.5, label=label, 
                   color='blue' if label == 'GP' else 'red', s=20)
    
        plt.title(f'Uncertainty Comparison - {label} ({method})')
        plt.xlabel('Predicted Time to Event')
        plt.ylabel('Uncertainty (Std)')
        plt.legend()
        
        # 保存每个模型的图
        plt.savefig(f'uncertainty_comparison_{label}_{method}.png')
        plt.close()
        print(f"✅ Uncertainty comparison plot saved to uncertainty_comparison_{label}_{method}.png")

if __name__ == "__main__":
    from EpiU_basic import plot_distributions_together
    evaluate()
