import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import gpytorch
import torchbnn as bnn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import os
import seaborn as sns
from scipy.stats import ks_2samp  # Kolmogorov-Smirnov test

# 定义模型保存路
MODEL_DIR = 'trained'
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

MC_MODEL_PATH = os.path.join(MODEL_DIR, 'mc_dropout.pt')
BNN_MODEL_PATH = os.path.join(MODEL_DIR, 'bnn.pt')
GP_MODEL_PATH = os.path.join(MODEL_DIR, 'gp.pt')
DKL_MODEL_PATH = os.path.join(MODEL_DIR, 'dkl.pt')

def load_support_dataset(random_state=0):
    # data loading (same as before)
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
        .sample(frac=1, random_state= 0)
    )

    # one-hot encode categorical variables
    df = pd.get_dummies(df, dummy_na=True)

    # fill missing values to the median
    df = df.fillna(df.median())

    # standardize numeric columns
    numeric_cols = df.dtypes == 'float64'
    df.loc[:, numeric_cols] = df.loc[:, numeric_cols].transform(
        lambda x: (x - x.mean()) / x.std())
     
    
    
    features = (
        df
        .drop(['death', 'd.time'], axis=1)
        .values
        .astype(float)
    )
    
    event_indicator = df['death'].values.astype(float)
    event_time = df['d.time'].values.astype(float)

    return features, event_indicator, event_time


class DiscreteFailureTimeNLL(nn.Module):
    def __init__(self, boundaries):
        super().__init__()
        self.boundaries = boundaries
    
    def forward(self, predictions, targets):
        # 将目标时间离散化到区间中
        discretized_times = torch.zeros_like(predictions)
        for i, (left, right) in enumerate(zip(self.boundaries[:-1], self.boundaries[1:])):
            mask = (targets >= left) & (targets < right)
            discretized_times[:, i] = mask.float()
        
        # 计算负对数似然
        nll = -torch.sum(discretized_times * torch.log(predictions + 1e-7), dim=1)
        return torch.mean(nll)


# MC Dropout Model
class MCDropoutModel(nn.Module):
    def __init__(self, input_dim, num_bins, hidden_dim=64, dropout_rate=0.1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), 
            nn.ReLU(),
            nn.Dropout(dropout_rate), 
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(), 
            nn.Dropout(dropout_rate), 
            nn.Linear(hidden_dim, num_bins)
        )

    def forward(self, x):
        logits = self.network(x)
        # print(f"Model output shape: {logits.shape}, Expected bins: {num_bins}")  # Debugging
        return F.softmax(logits, dim=-1)

# Bayesian Neural Network Model
class BNNModel(nn.Module):
    def __init__(self, input_dim, num_bins, hidden_dim=64):
        super().__init__()
        self.network = nn.Sequential(
            bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=input_dim, out_features=hidden_dim),
            nn.ReLU(),
            bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=hidden_dim, out_features=hidden_dim),
            nn.ReLU(),
            bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=hidden_dim, out_features=num_bins)
        )

    def forward(self, x):
        logits = self.network(x)
        return F.softmax(logits, dim=-1)

# Gaussian Process Model
class GPModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            num_inducing_points=inducing_points.size(0)
        )
        
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution,
            learn_inducing_locations=True
        )
        
        super().__init__(variational_strategy)
        
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(lengthscale_prior=gpytorch.priors.GammaPrior(3.0, 6.0))
        )
    
    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)

def train_mc_dropout(train_X, train_s, train_t, bin_boundaries):
    if os.path.exists(MC_MODEL_PATH):
        model = MCDropoutModel(input_dim=train_X.shape[1], num_bins=num_bins)
        model.load_state_dict(torch.load(MC_MODEL_PATH))
        print("🔄 MC Dropout model loaded")
    else:
        model = MCDropoutModel(input_dim=train_X.shape[1], num_bins=num_bins)
        train_model(model, train_X, train_s, train_t, bin_boundaries)
        torch.save(model.state_dict(), MC_MODEL_PATH)
        print(f"✅ MC Dropout model saved to {MC_MODEL_PATH}")
    return model

def train_bnn(train_X, train_s, train_t, bin_boundaries):
    if os.path.exists(BNN_MODEL_PATH):
        model = BNNModel(input_dim=train_X.shape[1], num_bins=num_bins)
        model.load_state_dict(torch.load(BNN_MODEL_PATH))
        print("🔄 BNN model loaded")
    else:
        model = BNNModel(input_dim=train_X.shape[1], num_bins=num_bins)
        train_model(model, train_X, train_s, train_t, bin_boundaries)
        torch.save(model.state_dict(), BNN_MODEL_PATH)
        print(f"✅ BNN model saved to {BNN_MODEL_PATH}")
    return model

def train_gp(train_X, train_y, num_iter=100, lr=0.01):
    print("Training GP model...")
    
    # create new model
    inducing_points = train_X[:500].clone()
    model = GPModel(inducing_points)
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    
    # train mode
    model.train()
    likelihood.train()
    
    # use ELBO loss
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_X.size(0))
    
    # optimizer
    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
        {'params': likelihood.parameters()},
    ], lr=lr)
    
    # train loop
    for i in range(num_iter):
        optimizer.zero_grad()
        output = model(train_X)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()
        
        if (i+1) % 10 == 0:
            print(f'Iter {i+1}/{num_iter} - Loss: {loss.item():.3f}')
    
    # save model
    torch.save({
        'model': model.state_dict(),
        'likelihood': likelihood.state_dict()
    }, GP_MODEL_PATH)
    print("✅ GP model saved")
    
    return model, likelihood

def predict_with_gp(model, likelihood, test_X):
    model.eval()
    likelihood.eval()
    
    with torch.no_grad():
        # get prediction distribution
        f_pred = model(test_X)
        y_pred = likelihood(f_pred)
        
        # get mean and std
        mean = y_pred.mean
        std = y_pred.stddev
        
        # ensure prediction is non-negative
        mean = torch.clamp(mean, min=1.0)
        
        return mean, std

class DKLFeatureExtractor(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        # use the same structure as MCDropoutModel, but without dropout
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.network(x)

class DKLModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            num_inducing_points=inducing_points.size(0)
        )
        
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution,
            learn_inducing_locations=True
        )
        
        super().__init__(variational_strategy)
        
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
    
    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)

class DKLGPModel(nn.Module):
    def __init__(self, input_dim, num_bins, hidden_dim=64):
        super().__init__()
        self.num_bins = num_bins
        
        # feature extractor
        self.feature_extractor = DKLFeatureExtractor(input_dim, hidden_dim)
        
        # GP layer
        inducing_points = torch.randn(100, hidden_dim)  # use the same dimension as the hidden layer
        self.gp_layer = DKLModel(inducing_points)
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        
        # final layer to target dimension (same as MCDropoutModel)
        self.final_layer = nn.Linear(hidden_dim, num_bins)
    
    def forward(self, x):
        # feature extraction
        features = self.feature_extractor(x)
        
        # GP prediction
        gp_output = self.gp_layer(features)
        gp_mean = gp_output.mean
        gp_var = gp_output.variance
        
        # final prediction
        logits = self.final_layer(features + torch.randn_like(features) * torch.sqrt(gp_var.unsqueeze(-1)))
        probs = F.softmax(logits, dim=-1)
        
        return probs, gp_var

def train_dkl(train_X, train_y, num_iter=100, lr=0.01):
    print("Training DKL model...")
    input_dim = train_X.shape[1]
    num_bins = len(bin_boundaries) - 1
    
    # create model
    model = DKLGPModel(input_dim, num_bins)
    
    # create loss function
    criterion = DiscreteFailureTimeNLL(bin_boundaries)
    
    # train mode
    model.train()
    
    # optimizer    
    optimizer = torch.optim.Adam([
        {'params': model.feature_extractor.parameters(), 'lr': lr},
        {'params': model.gp_layer.parameters(), 'lr': lr * 0.1},
        {'params': model.likelihood.parameters(), 'lr': lr * 0.1},
        {'params': model.final_layer.parameters(), 'lr': lr}
    ])
    
    # train loop
    best_loss = float('inf')
    best_state = None
    patience = 5
    no_improve = 0
    
    for i in range(num_iter):
        optimizer.zero_grad()
        
        # forward propagation (multiple sampling to increase stability)
        n_samples = 5
        total_loss = 0
        
        for _ in range(n_samples):
            probs, _ = model(train_X)
            loss = criterion(probs, train_y)
            total_loss += loss
        
        avg_loss = total_loss / n_samples
        avg_loss.backward()
        
        # gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # early stopping
        if avg_loss.item() < best_loss:
            best_loss = avg_loss.item()
            best_state = {
                'model': model.state_dict()
            }
            no_improve = 0
        else:
            no_improve += 1
            
        if no_improve >= patience:
            print(f"Early stopping at iteration {i+1}")
            break
            
        if (i+1) % 10 == 0:
            print(f'Iter {i+1}/{num_iter} - Loss: {avg_loss.item():.3f}')
    
    # load best model
    if best_state is not None:
        model.load_state_dict(best_state['model'])
        torch.save(best_state, DKL_MODEL_PATH)
        print("✅ DKL model saved")
    
    return model

def predict_with_dkl(model, test_X):
    model.eval()
    
    with torch.no_grad():
        # multiple sampling to get more stable predictions
        n_samples = 10
        all_predictions = []
        
        for _ in range(n_samples):
            probs, _ = model(test_X)
            # calculate mean of predictions
            bin_centers = torch.tensor([(bin_boundaries[i] + bin_boundaries[i+1])/2 
                                      for i in range(len(bin_boundaries)-1)])
            pred = torch.sum(probs * bin_centers.unsqueeze(0), dim=1)
            all_predictions.append(pred)
        
        # calculate mean and std of multiple sampling
        predictions = torch.stack(all_predictions)  # [n_samples, batch_size]
        mean = predictions.mean(dim=0)
        std = predictions.std(dim=0)
        
        # ensure uncertainty estimation is within reasonable range
        std = torch.clamp(std, min=mean * 0.1, max=mean * 0.5)  # limit to 10% to 50% of predictions
        
        return mean, std

# Training function for Discrete Failure Time NLL
def train_model(model, train_X, train_s, train_t, bin_boundaries, n_epochs=100, batch_size=32, lr=0.001):
    train_dataset = TensorDataset(
        torch.tensor(train_X, dtype=torch.float32),
        torch.tensor(train_s, dtype=torch.float32),
        torch.tensor(train_t, dtype=torch.float32)
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = DiscreteFailureTimeNLL(bin_boundaries=bin_boundaries)

    model.train()
    for epoch in range(n_epochs):
        total_loss = 0
        for batch_X, batch_s, batch_t in train_loader:
            optimizer.zero_grad()
            preds = model(batch_X)
            loss = loss_fn(preds, batch_t)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}, Loss: {total_loss/len(train_loader):.4f}")

# Prediction function for Discrete Failure Time NLL
def estimate_uncertainty(model, test_X, bin_midpoints, n_samples=100):
    model.train()
    pred_probs = torch.stack([model(test_X) for _ in range(n_samples)])  # (n_samples, batch, bins)
    pred_times = torch.einsum('sbi,i->sb', pred_probs, bin_midpoints)
    return pred_times.mean(dim=0), pred_times.std(dim=0)

from scipy.stats import ks_2samp
from sklearn.preprocessing import QuantileTransformer

def normalize_predictions(pred, reference, method="zscore"):
    """
    Normalize predictions to match the reference distribution.
    
    :param pred: Predicted values (NumPy array)
    :param reference: Reference True Y values (NumPy array)
    :param method: "zscore" (Z-score mapping) or "quantile" (Quantile Mapping)
    :return: Normalized predicted values
    """
    if method == "zscore":
        mu_y, sigma_y = np.mean(reference), np.std(reference)
        mu_pred, sigma_pred = np.mean(pred), np.std(pred)
        return mu_y + ((pred - mu_pred) / sigma_pred) * sigma_y  # Z-score Scaling

    elif method == "quantile":
        qt = QuantileTransformer(output_distribution="normal", n_quantiles=len(reference), random_state=42)
        qt.fit(reference.reshape(-1, 1))  # Learn the quantile mapping from True Y
        return qt.transform(pred.reshape(-1, 1)).flatten()  # Apply transformation to predictions

    else:
        raise ValueError("Unsupported normalization method. Use 'zscore' or 'quantile'.")
    
def save_predictions_to_excel(true_values, means, stds, model_names, method="zscore"):
    #  Fucntions for converting tensor to numpy
    def process_tensor(tensor):
        # 确保是tensor
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.tensor(tensor)
        
        # detach and move to CPU
        array = tensor.detach().cpu()
        
        # if multi-dimensional, ensure the last dimension is only one
        if array.dim() > 1:
            # if two-dimensional, select the correct dimension
            if array.dim() == 2:
                if array.shape[0] == 1:
                    array = array.squeeze(0)
                elif array.shape[1] == 1:
                    array = array.squeeze(1)
                else:
                    # if neither dimension is 1, select the first dimension
                    array = array[:, 0]
            else:
                # if dimension is higher, flatten
                array = array.flatten()
        
        return array.numpy()

    # process true values
    true_values_array = process_tensor(true_values)
    expected_length = len(true_values_array)
    
    # initialize data dictionary
    data = {
        "True Values": true_values_array
    }
    
    # process each model's predictions
    for name, mean, std in zip(model_names, means, stds):
        # process mean
        mean_array = process_tensor(mean)
        if len(mean_array) != expected_length:
            print(f"Warning: {name} mean length mismatch. Expected {expected_length}, got {len(mean_array)}")
            # if length mismatch, truncate or pad
            if len(mean_array) > expected_length:
                mean_array = mean_array[:expected_length]
            else:
                # use the last value to pad
                mean_array = np.pad(mean_array, 
                                  (0, expected_length - len(mean_array)),
                                  'edge')
        
        # process std
        std_array = process_tensor(std)
        if len(std_array) != expected_length:
            print(f"Warning: {name} std length mismatch. Expected {expected_length}, got {len(std_array)}")
            # if length mismatch, truncate or pad
            if len(std_array) > expected_length:
                std_array = std_array[:expected_length]
            else:
                # use the last value to pad
                std_array = np.pad(std_array,
                                 (0, expected_length - len(std_array)),
                                 'edge')
        
        # add to data dictionary
        data[f"{name} Mean"] = mean_array
        data[f"{name} Std"] = std_array
    
    # verify all array lengths are consistent
    lengths = [len(arr) for arr in data.values()]
    if len(set(lengths)) != 1:
        raise ValueError(f"Array length mismatch after processing: {dict(zip(data.keys(), lengths))}")
    
    # create DataFrame and save
    df = pd.DataFrame(data)
    output_file = f"predictions_{method}.xlsx"
    df.to_excel(output_file, index=False)
    print(f"✅ Predictions saved to {output_file}")

def plot_distributions_together(pred_means, labels, true_values, method="zscore"):
    #  Fucntions for ensuring array is 1D and length is correct
    def process_array(tensor, target_length):
        # convert to numpy array
        if isinstance(tensor, torch.Tensor):
            array = tensor.detach().cpu().numpy()
        else:
            array = np.array(tensor)
        
        # ensure 1D
        if array.ndim > 1:
            print(f"Warning: Converting {array.shape} array to 1D")
            array = array.flatten()
        
        # ensure length matches
        if len(array) != target_length:
            print(f"Warning: Array length mismatch. Expected {target_length}, got {len(array)}")
            if len(array) > target_length:
                array = array[:target_length]
            else:
                array = np.pad(array, (0, target_length - len(array)), 'edge')
        
        return array
    
    # get target length (use the length of true values as standard)
    true_values = process_array(true_values, len(true_values))
    target_length = len(true_values)
    
    # process all predictions
    pred_means_normalized = [process_array(mean, target_length) for mean in pred_means]
    
    # print debug information
    for label, mean in zip(labels, pred_means_normalized):
        print(f"{label} predictions shape: {mean.shape}")
    print(f"True values shape: {true_values.shape}")
    
    # create DataFrame
    data = {label: mean for mean, label in zip(pred_means_normalized, labels)}
    data['True Values'] = true_values
    df = pd.DataFrame(data)
    
    # plot
    plt.figure(figsize=(12, 6))
    
    # plot kde for each prediction distribution
    for label in labels:
        sns.kdeplot(data=df[label], label=f'{label} Predictions')
    
    # plot the distribution of true values
    sns.kdeplot(data=df['True Values'], label='True Values', linestyle='--')
    
    plt.title(f'Distribution of Predictions vs True Values ({method} normalization)')
    plt.xlabel('Normalized Time to Event')
    plt.ylabel('Density')
    plt.legend()
    
    # save plot
    plt.savefig(f'distribution_comparison_{method}.png')
    plt.close()
    print(f"✅ Distribution plot saved to distribution_comparison_{method}.png")

def plot_uncertainty_together(true_values, pred_means, pred_stds, labels, method="zscore"):
    #  Fucntions for processing array
    def process_array(tensor, target_length):
        if isinstance(tensor, torch.Tensor):
            array = tensor.detach().cpu().numpy()
        else:
            array = np.array(tensor)
        
        if array.ndim > 1:
            array = array.flatten()
        
        if len(array) != target_length:
            if len(array) > target_length:
                array = array[:target_length]
            else:
                array = np.pad(array, (0, target_length - len(array)), 'edge')
        
        return array
    
    # get target length
    true_values = process_array(true_values, len(true_values))
    target_length = len(true_values)
    
    # process all predictions and stds
    pred_means = [process_array(mean, target_length) for mean in pred_means]
    pred_stds = [process_array(std, target_length) for std in pred_stds]
    
    # sort by true values
    sort_indices = np.argsort(true_values)
    true_values_sorted = true_values[sort_indices]
    pred_means_sorted = [mean[sort_indices] for mean in pred_means]
    pred_stds_sorted = [std[sort_indices] for std in pred_stds]
    
    # create x-axis data
    x = np.arange(target_length)
    
    # set color scheme
    colors = ['#82CAFF', '#FDBD01', '#967BB6', '#EB7070'] 
    
    # plot
    plt.figure(figsize=(15, 8))
    
    # first plot true values (gray dashed line)
    plt.plot(x, true_values_sorted, color='gray', linestyle='--', label='True Values', alpha=0.5)
    
    # plot each model's prediction and uncertainty
    for mean, std, label, color in zip(pred_means_sorted, pred_stds_sorted, labels, colors):
        # plot prediction mean
        plt.plot(x, mean, color=color, label=f'{label}', alpha=0.8)
        
        # plot uncertainty interval
        plt.fill_between(
            x,
            mean - 2*std,  # 2 standard deviations confidence interval
            mean + 2*std,
            color=color,
            alpha=0.2
        )
    
    plt.title(f'Predictions with Uncertainty ({method} normalization)\nSorted by True Values')
    plt.xlabel('Sample Index (Sorted)')
    plt.ylabel('Normalized Time to Event')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # save plot
    plt.savefig(f'uncertainty_comparison_{method}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Uncertainty plot saved to uncertainty_comparison_{method}.png")

def compare_distributions(true_values, predictions, model_names, method="zscore"):
    def process_array(tensor, target_length, name="unknown"):
        # Convert tensor to numpy array
        if isinstance(tensor, torch.Tensor):
            array = tensor.detach().cpu().numpy()
        else:
            array = np.array(tensor)
        
        # Ensure array is 1D
        if array.ndim > 1:
            array = array.flatten()
        
        # Match target length
        if len(array) != target_length:
            if len(array) > target_length:
                array = array[:target_length]
            else:
                array = np.pad(array, (0, target_length - len(array)), 'edge')
        
        return array
    
    target_length = len(true_values)
    true_values = process_array(true_values, target_length, "True Values")
    
    # Process predictions for non-GP models
    processed_predictions = []
    processed_names = []
    
    for pred, name in zip(predictions, model_names):
        if name != "GP":  # Skip GP model
            processed_pred = process_array(pred, target_length, name)
            processed_predictions.append(processed_pred)
            processed_names.append(name)
    
    # Create data dictionary
    data = {
        "True Values": true_values,
        **{name: pred for name, pred in zip(processed_names, processed_predictions)}
    }
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Create plot
    plt.figure(figsize=(12, 6))
    
    # Plot distribution for each model
    for name in processed_names:
        sns.kdeplot(data=df[name], label=f'{name}')
    
    # Plot true value distribution
    sns.kdeplot(data=df["True Values"], label='True Values', 
                linestyle='--', color='gray')
    
    plt.title(f'Distribution Comparison ({method} normalization)')
    plt.xlabel('Normalized Time to Event')
    plt.ylabel('Density')
    plt.legend()
    
    # Save plot
    plt.savefig(f'distribution_comparison_{method}.png')
    plt.close()
    print(f"✅ Distribution comparison plot saved to distribution_comparison_{method}.png")

def compare_uncertainties(predictions, uncertainties, model_names, method="zscore"):
    # create data dictionary
    data = {}
    for pred, unc, name in zip(predictions, uncertainties, model_names):
        if isinstance(pred, torch.Tensor):
            pred = pred.detach().cpu().numpy()
        if isinstance(unc, torch.Tensor):
            unc = unc.detach().cpu().numpy()
        
        data[name] = {
            'predictions': pred.flatten(),
            'uncertainties': unc.flatten()
        }
    
    # create plot
    plt.figure(figsize=(12, 6))
    
    # plot each model's scatter plot
    colors = ['#82CAFF', '#FDBD01', '#967BB6', '#EB7070']  # set different colors for different models
    for (name, values), color in zip(data.items(), colors):
        plt.scatter(values['predictions'], 
                   values['uncertainties'],
                   alpha=0.5,  # set transparency
                   label=name,
                   color=color,
                   s=20)  # set point size
    
    plt.title(f'Uncertainty Comparison ({method} normalization)')
    plt.xlabel('Predicted Time to Event')
    plt.ylabel('Uncertainty (Std)')
    plt.legend()
    
    # save plot
    plt.savefig(f'uncertainty_comparison_{method}.png')
    plt.close()
    print(f"✅ Uncertainty comparison plot saved to uncertainty_comparison_{method}.png")

# Main function
if __name__ == "__main__":
    X, s, t = load_support_dataset()
    test_size = len(X) // 5
    train_X, train_s, train_t = X[:-test_size], s[:-test_size], t[:-test_size]
    test_X, test_s, test_t = X[-test_size:], s[-test_size:], t[-test_size:]
    test_X_tensor = torch.tensor(test_X, dtype=torch.float32)
    test_Y_tensor = torch.tensor(test_t, dtype=torch.float32)

    # Generate bin boundaries and midpoints
    bin_boundaries = np.linspace(train_t.min(), train_t.max(), num=12) 
    num_bins = len(bin_boundaries) - 1  

    bin_midpoints = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
    bin_midpoints = torch.tensor(bin_midpoints, dtype=torch.float32)

    # Train models
    mc_model = train_mc_dropout(train_X, train_s, train_t, bin_boundaries)
    bnn_model = train_bnn(train_X, train_s, train_t, bin_boundaries)

    # GP model remains the same 
    gp_model, gp_likelihood = train_gp(
    torch.tensor(train_X, dtype=torch.float32),
    torch.tensor(train_t, dtype=torch.float32)
)

    # Get predictions
    mc_mean, mc_std = estimate_uncertainty(mc_model, test_X_tensor, bin_midpoints)
    bnn_mean, bnn_std = estimate_uncertainty(bnn_model, test_X_tensor, bin_midpoints)

    # GP prediction (keep original logic)
    gp_model.eval()
    with torch.no_grad():
        gp_pred = gp_likelihood(gp_model(test_X_tensor))
        gp_mean, gp_std = gp_pred.mean, gp_pred.stddev

    # Train DKL (use training set)
    train_X_tensor = torch.tensor(train_X, dtype=torch.float32)
    train_t_tensor = torch.tensor(train_t, dtype=torch.float32)

    dkl_model = train_dkl(train_X_tensor, train_t_tensor)

    # Predict & calculate uncertainty (use test set)
    test_X_tensor = torch.tensor(test_X, dtype=torch.float32)
    dkl_mean, dkl_std = predict_with_dkl(dkl_model, test_X_tensor)

    # Plot (ensure consistent prediction format)
    save_predictions_to_excel(test_Y_tensor, [bnn_mean, mc_mean, gp_mean, dkl_mean], [bnn_std, mc_std, gp_std, dkl_std], ["BNN", "MC Dropout", "GP", "DKL"], method="zscore")
    plot_distributions_together([bnn_mean, mc_mean, gp_mean, dkl_mean], ["BNN", "MC Dropout", "GP", "DKL"], test_Y_tensor, method="zscore")

    plot_uncertainty_together(test_Y_tensor, [bnn_mean, mc_mean, gp_mean, dkl_mean], [bnn_std, mc_std, gp_std, dkl_std], ["BNN", "MC Dropout", "GP", "DKL"], method="zscore")
    compare_distributions(test_Y_tensor, [bnn_mean, mc_mean, gp_mean, dkl_mean], ["BNN", "MC Dropout", "GP", "DKL"], method="zscore")

    compare_uncertainties([bnn_mean, mc_mean, gp_mean, dkl_mean], [bnn_std, mc_std, gp_std, dkl_std], ["BNN", "MC Dropout", "GP", "DKL"], method="zscore")

    
