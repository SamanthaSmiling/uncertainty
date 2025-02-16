# Standard imports
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-darkgrid')

# ML imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import gpytorch
import joblib
from torch.utils.data import DataLoader, TensorDataset

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
        lambda x: (x - x.mean()) / x.std())
    
    features = df.drop(['death', 'd.time'], axis=1).values.astype(float)
    event_indicator = df['death'].values.astype(float)
    event_time = df['d.time'].values.astype(float)

    return features, event_indicator, event_time

class DiscreteTimeNeuralSurvival(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, n_intervals=50):
        super().__init__()
        self.n_intervals = n_intervals
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, n_intervals)
        )
        
    def forward(self, x):
        logits = self.network(x)
        hazard = F.softplus(logits)
        return hazard
    
    def get_features(self, x):
        """Get intermediate features for GP uncertainty estimation"""
        for layer in list(self.network.children())[:-1]:
            x = layer(x)
        return x

def discretize(time, n_intervals):
    """Discretize continuous time into intervals"""
    max_time = time.max()
    boundaries = np.linspace(0, max_time, n_intervals + 1)
    discretized = np.digitize(time.numpy(), boundaries) - 1
    discretized = np.clip(discretized, 0, n_intervals - 1)
    return torch.tensor(discretized, dtype=torch.long), boundaries

class DiscreteFailureTimeNLL(torch.nn.Module):
    def __init__(self, bin_boundaries, tolerance=1e-7):
        super(DiscreteFailureTimeNLL,self).__init__()
        
        self.bin_starts = torch.tensor(bin_boundaries[:-1], dtype=torch.float32)
        self.bin_ends = torch.tensor(bin_boundaries[1:], dtype=torch.float32)
        self.bin_lengths = self.bin_ends - self.bin_starts
        self.tolerance = tolerance
        
    def _discretize_times(self, times):
        return (
            (times[:, None] > self.bin_starts[None, :])
            & (times[:, None] <= self.bin_ends[None, :])
        ).float()

    def _get_proportion_of_bins_completed(self, times):
        return torch.clamp(
            (times[:, None] - self.bin_starts[None, :]) / self.bin_lengths[None, :],
            min=0.0,
            max=1.0
        )
    
    def forward(self, predictions, event_indicators, event_times):
        predictions = torch.clamp(predictions, min=self.tolerance, max=1-self.tolerance)
        
        event_likelihood = torch.sum(
            self._discretize_times(event_times) * predictions,
            dim=-1
        ) + self.tolerance

        nonevent_likelihood = 1 - torch.sum(
            self._get_proportion_of_bins_completed(event_times) * predictions,
            dim=-1
        ) + self.tolerance
        
        event_likelihood = torch.clamp(event_likelihood, min=self.tolerance, max=1-self.tolerance)
        nonevent_likelihood = torch.clamp(nonevent_likelihood, min=self.tolerance, max=1-self.tolerance)
        
        log_likelihood = event_indicators * torch.log(event_likelihood)
        log_likelihood += (1 - event_indicators) * torch.log(nonevent_likelihood)
        
        return -1. * torch.mean(log_likelihood)

def train_survival_model(model, train_X, train_t, train_s, batch_size=32, n_epochs=100):
    """Train the discrete time neural survival model"""
    # Create data loader
    train_dataset = TensorDataset(
        torch.tensor(train_X, dtype=torch.float32),
        torch.tensor(train_t, dtype=torch.float32),
        torch.tensor(train_s, dtype=torch.float32)
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize loss function with proper bin boundaries
    max_time = np.max(train_t)
    bin_boundaries = np.linspace(0, max_time, model.n_intervals + 1)
    loss_fn = DiscreteFailureTimeNLL(bin_boundaries)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        
        for batch_x, batch_t, batch_s in train_loader:
            optimizer.zero_grad()
            hazard = model(batch_x)
            loss = loss_fn(hazard, batch_s, batch_t)
            
            if not torch.isnan(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item()
            else:
                print("Warning: NaN loss detected!")
        
        avg_loss = total_loss/len(train_loader)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.4f}")
        
        scheduler.step(avg_loss)
    
    return model, loss_fn

class GPUncertainty(gpytorch.models.ExactGP):
    """GP model for uncertainty estimation"""
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel()
        )
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def estimate_uncertainty(survival_model, train_X, train_t, test_X):
    """Estimate uncertainty using GP on survival model predictions"""
    survival_model.eval()
    with torch.no_grad():
        # Keep using float32
        train_pred = survival_model(torch.tensor(train_X, dtype=torch.float32))
        test_pred = survival_model(torch.tensor(test_X, dtype=torch.float32))
    
    # Initialize and train GP model with float32
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    likelihood = likelihood.float()  # Convert to float32
    
    gp_model = GPUncertainty(
        train_pred, 
        torch.tensor(train_t, dtype=torch.float32),
        likelihood
    )
    gp_model = gp_model.float()  # Convert to float32
    
    # Train GP
    gp_model.train()
    likelihood.train()
    optimizer = torch.optim.Adam(gp_model.parameters(), lr=0.1)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, gp_model)
    
    for i in range(100):
        optimizer.zero_grad()
        output = gp_model(train_pred)
        loss = -mll(output, torch.tensor(train_t, dtype=torch.float32))
        loss.backward()
        optimizer.step()
    
    # Get uncertainty estimates
    gp_model.eval()
    likelihood.eval()
    with torch.no_grad():
        pred_distribution = likelihood(gp_model(test_pred))
        uncertainty = pred_distribution.stddev
    
    return uncertainty

def calculate_nll(predictions, uncertainties, true_values):
    """Calculate negative log-likelihood with uncertainty"""
    # Assume Gaussian distribution
    nll = -torch.distributions.Normal(
        predictions, uncertainties
    ).log_prob(torch.tensor(true_values, dtype=torch.float32))
    
    return nll.mean().item(), nll.std().item()

def plot_prediction_histogram(predictions, title="Prediction Distribution"):
    """Plot histogram of predictions"""
    plt.figure(figsize=(8, 5))
    plt.hist(predictions.numpy(), bins=50, density=True)
    plt.xlabel("Predicted Values")
    plt.ylabel("Density")
    plt.title(title)
    plt.show()

def plot_uncertainty(test_t, predictions, uncertainties):
    """Plot predictions with uncertainty bands"""
    plt.figure(figsize=(10, 5))

    # Convert to numpy if needed
    mean_1d = predictions.mean(dim=-1).detach().numpy()
    std_1d = uncertainties.detach().numpy()
    test_t_np = test_t if isinstance(test_t, np.ndarray) else test_t.numpy()

    # Smooth uncertainty
    smooth_std_1d = gaussian_filter1d(std_1d, sigma=5)

    # Sort by time for better visualization
    sort_idx = np.argsort(test_t_np)
    test_t_np = test_t_np[sort_idx]
    mean_1d = mean_1d[sort_idx]
    std_1d = std_1d[sort_idx]

    # Plot mean prediction
    plt.plot(test_t_np, mean_1d, label="Mean Prediction", color="y")

    # Plot uncertainty bands
    plt.fill_between(
        test_t_np,
        mean_1d - smooth_std_1d,
        mean_1d + smooth_std_1d,
        color="blue",
        alpha=0.3,
        label="Smoothed Uncertainty Range (± 1 std)"
    )

    plt.xlabel("Time")
    plt.ylabel("Predicted Probability")
    plt.title("GP Uncertainty Estimation (Smoothed)")
    plt.legend()
    plt.show()

def save_data_and_model(X, s, t, model, likelihood, filename_prefix='gp_model'):
    """Save the dataset and trained model."""
    # Save the data
    np.savez(f'{filename_prefix}_data.npz', X=X, s=s, t=t)
    
    # Save the model
    torch.save({
        'model_state_dict': model.state_dict(),
        'likelihood_state_dict': likelihood.state_dict(),
    }, f'{filename_prefix}.pth')

def load_data_and_model(train_x, train_y, filename_prefix='gp_model'):
    """Load the saved dataset and model."""
    # Load the data
    data = np.load(f'{filename_prefix}_data.npz')
    X, s, t = data['X'], data['s'], data['t']
    
    # Initialize likelihood and model with actual training data
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = GPUncertainty(train_x, train_y, likelihood)
    
    # Load the saved state
    checkpoint = torch.load(f'{filename_prefix}.pth', weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    likelihood.load_state_dict(checkpoint['likelihood_state_dict'])
    
    # Make sure to put model in evaluation mode
    model.eval()
    likelihood.eval()
    
    return X, s, t, model, likelihood

if __name__ == "__main__":
    # Load data
    X, s, t = load_support_dataset()
    test_idx = len(X) * 4 // 5
    train_X, train_s, train_t = (arr[:test_idx] for arr in (X, s, t))
    test_X, test_s, test_t = (arr[test_idx:] for arr in (X, s, t))

    # Train survival model
    model = DiscreteTimeNeuralSurvival(input_dim=train_X.shape[1])
    model, loss_fn = train_survival_model(model, train_X, train_t, train_s)
    
    # Make predictions
    model.eval()
    with torch.no_grad():
        predictions = model(torch.tensor(test_X, dtype=torch.float32))
        # Calculate NLL on test set
        test_nll = loss_fn(
            predictions,
            torch.tensor(test_s, dtype=torch.float32),
            torch.tensor(test_t, dtype=torch.float32)
        ).item()
    
    # Estimate uncertainty
    uncertainties = estimate_uncertainty(model, train_X, train_t, test_X)
    
    # Print metrics
    print(f"Test NLL: {test_nll:.4f}")
    print(f"GP Uncertainty (mean std): {uncertainties.mean().item():.4f}")
    
    # Plot prediction histogram
    plot_prediction_histogram(predictions.mean(dim=-1), "Survival Model Predictions")
    
    # Plot predictions with uncertainty
    plot_uncertainty(test_t, predictions, uncertainties)
