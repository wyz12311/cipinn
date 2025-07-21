import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import os
import time
from datetime import timedelta
import networkx as nx
from scipy import stats
import itertools
import copy

# Import only available functions from pgmpy
try:
    from pgmpy.estimators import PC
    from pgmpy.estimators.CITests import chi_square
except ImportError:
    print("Warning: pgmpy not fully available, will use alternative methods")

# Configuration
MODEL_PATH = r'./models9/checkpoints'
os.makedirs(MODEL_PATH, exist_ok=True)
MODEL_FILE = os.path.join(MODEL_PATH, 'best_model.pth')

# Create plots directory
PLOTS_PATH = r'./plots9'
os.makedirs(PLOTS_PATH, exist_ok=True)

# Create causal analysis directory
CAUSAL_PATH = os.path.join(PLOTS_PATH, 'causal')
os.makedirs(CAUSAL_PATH, exist_ok=True)

# Create sensitivity analysis directory
SENSITIVITY_PATH = os.path.join(PLOTS_PATH, 'sensitivity')
os.makedirs(SENSITIVITY_PATH, exist_ok=True)

# Create correlation analysis directory
CORRELATION_PATH = os.path.join(PLOTS_PATH, 'correlation')
os.makedirs(CORRELATION_PATH, exist_ok=True)

# Create interactions analysis directory
INTERACTIONS_PATH = os.path.join(PLOTS_PATH, 'interactions')
os.makedirs(INTERACTIONS_PATH, exist_ok=True)


# Custom implementation of pearson correlation CI test
def pearson_correlation_ci_test(X, Y, Z, data, **kwargs):
    """
    Pearson Correlation based conditional independence test.

    Parameters
    ----------
    X: int
        First variable index
    Y: int
        Second variable index
    Z: list
        Conditioning variable indices
    data: pandas.DataFrame
        Dataset on which to test the independence condition
    kwargs: dict
        Additional parameters

    Returns
    -------
    p_value: float
        The p-value of the test
    """
    significance_level = kwargs.get("significance_level", 0.05)

    if len(Z) == 0:
        # If no conditioning variables, perform standard correlation test
        r, p_value = stats.pearsonr(data.iloc[:, X], data.iloc[:, Y])
    else:
        # Partial correlation with conditioning variables
        # First, regress out Z from X and Y
        X_data = data.iloc[:, X].values
        Y_data = data.iloc[:, Y].values
        Z_data = data.iloc[:, Z].values

        # Residualize X with respect to Z
        if Z_data.ndim == 1:
            Z_data = Z_data.reshape(-1, 1)

        # Add intercept to Z
        Z_data = np.hstack((np.ones((Z_data.shape[0], 1)), Z_data))

        # Calculate least squares coefficients for X
        beta_X = np.linalg.lstsq(Z_data, X_data, rcond=None)[0]
        X_resid = X_data - Z_data @ beta_X

        # Calculate least squares coefficients for Y
        beta_Y = np.linalg.lstsq(Z_data, Y_data, rcond=None)[0]
        Y_resid = Y_data - Z_data @ beta_Y

        # Calculate correlation between residuals
        r, p_value = stats.pearsonr(X_resid, Y_resid)

    # Return p-value for the test
    return p_value


class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout_rate, activation):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            activation,
            nn.Dropout(dropout_rate),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim)
        )
        self.activation = activation

    def forward(self, x):
        return self.activation(x + self.block(x))


class DynamicPINN(nn.Module):
    def __init__(self, input_dim=3, hidden_dims=[256, 512, 1024, 2048, 1024, 512, 256], output_dim=1,
                 dropout_rate=0.331047, activation='relu', causal_edges=None, edge_weights=None,
                 interaction_order=1, param_init=None):
        """
        Enhanced Physics-Informed Neural Network with dynamic physical constraint based on causal structure.

        Parameters:
        -----------
        input_dim: int
            Number of input features
        hidden_dims: list
            Dimensions of hidden layers
        output_dim: int
            Number of output features
        dropout_rate: float
            Dropout probability
        activation: str
            Activation function to use
        causal_edges: list
            List of (source, target) tuples representing causal edges
        edge_weights: dict
            Dictionary mapping (source, target) to edge weight/strength
        interaction_order: int
            Maximum order of interaction terms to include (1 = main effects only, 2 = up to pairwise, etc.)
        param_init: dict
            Initial values for physics parameters
        """
        super(DynamicPINN, self).__init__()

        # Setup basic network architecture
        layers = []

        if activation == 'relu':
            act_fn = nn.ReLU()
        elif activation == 'leaky_relu':
            act_fn = nn.LeakyReLU(0.2)
        elif activation == 'elu':
            act_fn = nn.ELU()
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

        layers.extend([
            nn.Linear(input_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            act_fn,
            nn.Dropout(dropout_rate)
        ])

        for i in range(len(hidden_dims) - 1):
            if hidden_dims[i] == hidden_dims[i + 1]:
                layers.append(ResidualBlock(hidden_dims[i], dropout_rate, act_fn))
            else:
                layers.extend([
                    nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                    nn.LayerNorm(hidden_dims[i + 1]),
                    act_fn,
                    nn.Dropout(dropout_rate)
                ])

        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        self.net = nn.Sequential(*layers)
        self.apply(self._init_weights)

        # Store feature names for reference
        self.feature_names = ['Vf', 'f', 'Vc']
        self.target_name = 'Ra'

        # Default parameter initialization
        default_init = {
            'A': 15.3965,
            'B_Vc': -5.5333,
            'B_Vf': 5.3660,
            'B_f': -5.5043
        }

        # Use provided init values if available, otherwise use defaults
        if param_init is None:
            param_init = default_init

        # Initialize constant term
        self.A = nn.Parameter(torch.tensor(param_init.get('A', default_init['A']), dtype=torch.float32))

        # Process causal structure information
        self.causal_edges = causal_edges if causal_edges is not None else []
        self.edge_weights = edge_weights if edge_weights is not None else {}
        self.interaction_order = interaction_order

        # Dynamically create parameters based on causal structure
        self.create_dynamic_parameters(param_init, default_init)

    def create_dynamic_parameters(self, param_init, default_init):
        """Create model parameters dynamically based on causal structure"""
        # Create main effect parameters based on causal edges
        self.main_effect_params = nn.ParameterDict()

        for source, target in self.causal_edges:
            if target == 3:  # If target is Ra (index 3)
                param_name = f'B_{self.feature_names[source]}'
                default_value = default_init.get(param_name, 0.0)
                init_value = param_init.get(param_name, default_value)
                # Scale parameter importance by edge weight if available
                weight = self.edge_weights.get((source, target), 1.0)
                # If weight is very small, we could initialize the parameter close to zero
                if weight < 0.1:
                    init_value *= weight * 10  # Reduce initial value for weak edges
                self.main_effect_params[param_name] = nn.Parameter(
                    torch.tensor(init_value, dtype=torch.float32)
                )

        # Create interaction effect parameters if interaction_order > 1
        if self.interaction_order > 1:
            self.interaction_params = nn.ParameterDict()

            # For all input features that have a causal edge to the target
            causal_features = [source for source, target in self.causal_edges if target == 3]

            # Generate all possible interactions up to the specified order
            for order in range(2, self.interaction_order + 1):
                for combo in itertools.combinations(causal_features, order):
                    # Create parameter name for this interaction
                    param_name = 'I_' + '_'.join(self.feature_names[i] for i in combo)

                    # Check if all features in combo have strong edges to target
                    all_strong = all(self.edge_weights.get((i, 3), 0.0) > 0.1 for i in combo)

                    # Initialize interaction term close to zero if not all edges are strong
                    if all_strong:
                        init_value = param_init.get(param_name, 0.01)
                    else:
                        init_value = param_init.get(param_name, 0.001)

                    self.interaction_params[param_name] = nn.Parameter(
                        torch.tensor(init_value, dtype=torch.float32)
                    )

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.net(x)

    def physical_equation(self, original_Vf, original_f, original_Vc):
        """
        Compute the physics-based prediction dynamically using the causal structure
        """
        # Initialize with constant term
        physical_pred = self.A.clone()

        # Dictionary mapping feature index to its value
        feature_values = {
            0: original_Vf,
            1: original_f,
            2: original_Vc
        }

        # Add main effects
        for source, target in self.causal_edges:
            if target == 3:  # If target is Ra (index 3)
                param_name = f'B_{self.feature_names[source]}'
                if param_name in self.main_effect_params:
                    physical_pred = physical_pred + self.main_effect_params[param_name] * feature_values[source]

        # Add interaction effects if applicable
        if self.interaction_order > 1 and hasattr(self, 'interaction_params'):
            causal_features = [source for source, target in self.causal_edges if target == 3]

            for order in range(2, self.interaction_order + 1):
                for combo in itertools.combinations(causal_features, order):
                    param_name = 'I_' + '_'.join(self.feature_names[i] for i in combo)

                    if param_name in self.interaction_params:
                        # Multiply the features involved in this interaction
                        interaction_term = self.interaction_params[param_name]
                        for idx in combo:
                            interaction_term = interaction_term * feature_values[idx]

                        physical_pred = physical_pred + interaction_term

        return physical_pred


def physical_constraint(model, inputs, Ra, y_scaler, x_scaler):
    """
    Dynamic physical constraint function based on the causal structure encoded in the model.
    Now y_scaler scales Ra and x_scaler scales inputs.
    """
    # Unscale the inputs and outputs
    batch_size = inputs.shape[0]
    unscaled_inputs = {}

    for i in range(len(model.feature_names)):
        unscaled_inputs[i] = inputs[:, i] * x_scaler.scale_[i] + x_scaler.mean_[i]

    original_Ra = Ra * y_scaler.scale_[0] + y_scaler.mean_[0]

    # Calculate the physics-based prediction using the model's physical equation
    Ra_pred = model.physical_equation(
        unscaled_inputs[0],  # original_Vf
        unscaled_inputs[1],  # original_f
        unscaled_inputs[2]  # original_Vc
    )

    # Return the mean absolute error between the physics-based prediction and actual Ra
    return torch.mean(torch.abs(Ra_pred - original_Ra))


def discover_causal_structure(data, var_names=None, significance_level=0.01, use_pc_algorithm=True):
    """
    Discover causal structure from data using PC algorithm or alternative method.

    Parameters:
    -----------
    data: pandas.DataFrame
        Data to analyze
    var_names: list
        List of variable names (optional)
    significance_level: float
        Significance level for conditional independence tests
    use_pc_algorithm: bool
        Whether to use PC algorithm or alternative method

    Returns:
    --------
    causal_edges: list
        List of (source, target) tuples representing causal edges
    edge_weights: dict
        Dictionary mapping (source, target) to edge weight/strength
    causal_graph: networkx.DiGraph
        Directed graph representing causal structure
    """
    if var_names is None:
        var_names = data.columns.tolist()

    print(f"\nDiscovering causal structure with significance level {significance_level}...")

    # Prepare data
    data_with_names = data.copy()
    if data_with_names.columns.tolist() != var_names:
        data_with_names.columns = var_names

    causal_edges = []
    edge_weights = {}
    G = nx.DiGraph()

    for name in var_names:
        G.add_node(name)

    try:
        if use_pc_algorithm and 'PC' in globals():
            # Use PC algorithm
            pc = PC(data=data_with_names)
            causal_model = pc.estimate(variant="stable", ci_test=pearson_correlation_ci_test,
                                       significance_level=significance_level)

            # Extract edges from causal model
            for i, j in causal_model.edges():
                source_idx = var_names.index(i)
                target_idx = var_names.index(j)
                causal_edges.append((source_idx, target_idx))

                # Add edge to graph
                G.add_edge(i, j)

            # Calculate edge weights based on partial correlations
            for source_idx, target_idx in causal_edges:
                source = var_names[source_idx]
                target = var_names[target_idx]

                # Find conditioning set (all other variables)
                conditioning_vars = [k for k, name in enumerate(var_names)
                                     if k != source_idx and k != target_idx]

                # Calculate partial correlation
                if not conditioning_vars:
                    r, _ = stats.pearsonr(data.iloc[:, source_idx], data.iloc[:, target_idx])
                    edge_weights[(source_idx, target_idx)] = abs(r)
                else:
                    # Use partial correlation with conditioning set
                    X_data = data.iloc[:, source_idx].values
                    Y_data = data.iloc[:, target_idx].values
                    Z_data = data.iloc[:, conditioning_vars].values

                    if Z_data.ndim == 1:
                        Z_data = Z_data.reshape(-1, 1)

                    # Add intercept to Z
                    Z_data = np.hstack((np.ones((Z_data.shape[0], 1)), Z_data))

                    # Calculate least squares coefficients
                    beta_X = np.linalg.lstsq(Z_data, X_data, rcond=None)[0]
                    X_resid = X_data - Z_data @ beta_X

                    beta_Y = np.linalg.lstsq(Z_data, Y_data, rcond=None)[0]
                    Y_resid = Y_data - Z_data @ beta_Y

                    # Calculate correlation between residuals
                    r, _ = stats.pearsonr(X_resid, Y_resid)
                    edge_weights[(source_idx, target_idx)] = abs(r)

            print(f"Discovered {len(causal_edges)} causal edges using PC algorithm")

            return causal_edges, edge_weights, G

        else:
            raise ImportError("PC algorithm not available")

    except Exception as e:
        print(f"Error in causal discovery with PC algorithm: {str(e)}")
        print("Falling back to alternative causal analysis method...")

    # Alternative: Use pairwise correlations and significance tests
    print("Using correlation-based approach to discover causal structure")
    n_vars = len(var_names)

    # Calculate correlation matrix
    corr_matrix = data.corr().values

    # Perform significance tests on correlations
    for i in range(n_vars):
        for j in range(n_vars):
            if i != j:
                r, p_value = stats.pearsonr(data.iloc[:, i], data.iloc[:, j])

                # If correlation is significant, add edge
                if p_value < significance_level:
                    # For simplicity, assume direction from lower index to higher index
                    # In a real implementation, direction would be determined by more sophisticated methods
                    if i < j:
                        causal_edges.append((i, j))
                        edge_weights[(i, j)] = abs(r)
                        G.add_edge(var_names[i], var_names[j], weight=abs(r))

    print(f"Discovered {len(causal_edges)} causal edges using correlation-based approach")

    return causal_edges, edge_weights, G


def visualize_causal_structure(G, edge_weights=None, var_names=None, threshold=0.0,
                               save_path=None, title="Causal Graph"):
    """
    Visualize discovered causal structure with edge weights.

    Parameters:
    -----------
    G: networkx.DiGraph
        Directed graph representing causal structure
    edge_weights: dict
        Dictionary mapping (source_idx, target_idx) to edge weight
    var_names: list
        List of variable names
    threshold: float
        Minimum edge weight to display
    save_path: str
        Path to save the figure
    title: str
        Title for the plot
    """
    plt.figure(figsize=(12, 10), dpi=300)

    # Create a copy of the graph with only edges above threshold
    if edge_weights is not None and var_names is not None:
        G_filtered = nx.DiGraph()
        for node in G.nodes():
            G_filtered.add_node(node)

        # Add edges with weights above threshold
        for (i, j), weight in edge_weights.items():
            if weight > threshold:
                source = var_names[i]
                target = var_names[j]
                if source in G.nodes() and target in G.nodes():
                    G_filtered.add_edge(source, target, weight=weight)

        # Use the filtered graph
        plot_G = G_filtered
    else:
        # If no weights or names provided, use original graph
        plot_G = G

    # Create a better layout
    pos = nx.spring_layout(plot_G, seed=42, k=0.5)

    # Get edge weights for width and color mapping
    edges = plot_G.edges(data=True)
    if edge_weights is not None and var_names is not None:
        widths = []
        for u, v, data in edges:
            try:
                source_idx = var_names.index(u)
                target_idx = var_names.index(v)
                widths.append(edge_weights.get((source_idx, target_idx), 1.0) * 5)
            except ValueError:
                widths.append(1.0)
    else:
        widths = [data.get('weight', 1.0) * 5 for u, v, data in edges]

    # Draw nodes with enhanced appearance
    node_colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral']
    nx.draw_networkx_nodes(plot_G, pos,
                           node_color=[node_colors[i % len(node_colors)] for i in range(len(plot_G.nodes()))],
                           node_size=4000, edgecolors='black', linewidths=2.5)

    # Draw edges with varying width based on weight
    nx.draw_networkx_edges(plot_G, pos, width=widths, arrows=True, arrowsize=25,
                           edge_color='navy', connectionstyle='arc3,rad=0.1', alpha=0.8)

    # Create edge labels with weights
    if edge_weights is not None and var_names is not None:
        edge_labels = {}
        for u, v, data in edges:
            try:
                source_idx = var_names.index(u)
                target_idx = var_names.index(v)
                weight = edge_weights.get((source_idx, target_idx), 0.0)
                edge_labels[(u, v)] = f"{weight:.3f}"
            except ValueError:
                edge_labels[(u, v)] = ""
    else:
        edge_labels = {(u, v): f"{data.get('weight', 1.0):.3f}" for u, v, data in edges}

    # Draw edge labels
    nx.draw_networkx_edge_labels(plot_G, pos, edge_labels=edge_labels, font_size=12,
                                 font_color='darkred', font_weight='bold')

    # Draw node labels
    nx.draw_networkx_labels(plot_G, pos, font_size=16, font_weight='bold')

    plt.axis('off')
    plt.title(title, fontsize=20, fontweight='bold', pad=20)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Causal graph saved to: {save_path}")
    else:
        plt.show()

    plt.close()


def analyze_correlations(data, var_names=None, save_dir=None):
    """
    Analyze correlations in both log and original scales with enhanced visualizations.
    """
    if var_names is None:
        var_names = data.columns.tolist()

    print("\nAnalyzing correlations in both log and original scales...")

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # Create data with proper column names
    data_with_names = data.copy()
    if data_with_names.columns.tolist() != var_names:
        data_with_names.columns = var_names

    # 1. Correlation Analysis for log-transformed data
    correlation_matrix = data_with_names.corr()

    # Visualize correlation matrix for log-transformed data
    plt.figure(figsize=(10, 8), dpi=300)
    plt.style.use('seaborn-v0_8-pastel')

    im = plt.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar(im, label='Correlation Coefficient (Log Scale)')
    plt.title('Correlation Matrix of Manufacturing Parameters (Log Scale)',
              fontsize=16, fontweight='bold', pad=20)
    plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns,
               rotation=45, fontsize=12)
    plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns,
               fontsize=12)

    # Add correlation values to the heatmap
    for i in range(len(correlation_matrix.columns)):
        for j in range(len(correlation_matrix.columns)):
            text_color = "black" if abs(correlation_matrix.iloc[i, j]) < 0.7 else "white"
            plt.text(j, i, f'{correlation_matrix.iloc[i, j]:.3f}',
                     ha="center", va="center", fontsize=11, fontweight='bold',
                     color=text_color)

    plt.tight_layout()

    if save_dir:
        corr_path = os.path.join(save_dir, 'correlation_matrix_log_scale.png')
        plt.savefig(corr_path, dpi=300, bbox_inches='tight')
        print(f"Log-scale correlation matrix saved to: {corr_path}")
    plt.close()

    # 2. Correlation Analysis for original scale data
    data_exp = data_with_names.apply(np.exp)  # Transform back to original scale
    correlation_matrix_exp = data_exp.corr()

    # Visualize correlation matrix for original scale data
    plt.figure(figsize=(10, 8), dpi=300)
    plt.style.use('seaborn-v0_8-pastel')

    im = plt.imshow(correlation_matrix_exp, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar(im, label='Correlation Coefficient (Original Scale)')
    plt.title('Correlation Matrix of Manufacturing Parameters (Original Scale)',
              fontsize=16, fontweight='bold', pad=20)
    plt.xticks(range(len(correlation_matrix_exp.columns)), correlation_matrix_exp.columns,
               rotation=45, fontsize=12)
    plt.yticks(range(len(correlation_matrix_exp.columns)), correlation_matrix_exp.columns,
               fontsize=12)

    # Add correlation values to the heatmap
    for i in range(len(correlation_matrix_exp.columns)):
        for j in range(len(correlation_matrix_exp.columns)):
            text_color = "black" if abs(correlation_matrix_exp.iloc[i, j]) < 0.7 else "white"
            plt.text(j, i, f'{correlation_matrix_exp.iloc[i, j]:.3f}',
                     ha="center", va="center", fontsize=11, fontweight='bold',
                     color=text_color)

    plt.tight_layout()

    if save_dir:
        corr_path_exp = os.path.join(save_dir, 'correlation_matrix_original_scale.png')
        plt.savefig(corr_path_exp, dpi=300, bbox_inches='tight')
        print(f"Original-scale correlation matrix saved to: {corr_path_exp}")
    plt.close()

    # 3. Create comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8), dpi=300)

    # Log scale subplot
    im1 = ax1.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    ax1.set_title('Log Scale Correlations', fontsize=14, fontweight='bold')
    ax1.set_xticks(range(len(correlation_matrix.columns)))
    ax1.set_xticklabels(correlation_matrix.columns, rotation=45)
    ax1.set_yticks(range(len(correlation_matrix.columns)))
    ax1.set_yticklabels(correlation_matrix.columns)

    for i in range(len(correlation_matrix.columns)):
        for j in range(len(correlation_matrix.columns)):
            text_color = "black" if abs(correlation_matrix.iloc[i, j]) < 0.7 else "white"
            ax1.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                     ha="center", va="center", fontsize=10, fontweight='bold',
                     color=text_color)

    # Original scale subplot
    im2 = ax2.imshow(correlation_matrix_exp, cmap='coolwarm', vmin=-1, vmax=1)
    ax2.set_title('Original Scale Correlations', fontsize=14, fontweight='bold')
    ax2.set_xticks(range(len(correlation_matrix_exp.columns)))
    ax2.set_xticklabels(correlation_matrix_exp.columns, rotation=45)
    ax2.set_yticks(range(len(correlation_matrix_exp.columns)))
    ax2.set_yticklabels(correlation_matrix_exp.columns)

    for i in range(len(correlation_matrix_exp.columns)):
        for j in range(len(correlation_matrix_exp.columns)):
            text_color = "black" if abs(correlation_matrix_exp.iloc[i, j]) < 0.7 else "white"
            ax2.text(j, i, f'{correlation_matrix_exp.iloc[i, j]:.2f}',
                     ha="center", va="center", fontsize=10, fontweight='bold',
                     color=text_color)

    # Add shared colorbar
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.87, 0.15, 0.03, 0.7])
    fig.colorbar(im1, cax=cbar_ax, label='Correlation Coefficient')

    plt.suptitle('Correlation Analysis: Log Scale vs Original Scale',
                 fontsize=16, fontweight='bold')

    if save_dir:
        comparison_path = os.path.join(save_dir, 'correlation_comparison.png')
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
        print(f"Correlation comparison plot saved to: {comparison_path}")
    plt.close()

    return correlation_matrix, correlation_matrix_exp


def perform_sensitivity_analysis(model, x_scaler, y_scaler, data, var_names=None, save_dir=None):
    """
    Perform comprehensive sensitivity analysis showing how parameter changes affect surface roughness.
    """
    if var_names is None:
        var_names = ['Vf', 'f', 'Vc']

    print("\nPerforming sensitivity analysis...")

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    device = next(model.parameters()).device
    model.eval()

    # Get parameter ranges (in log scale from the data)
    param_ranges = {}
    for i, param in enumerate(var_names):
        param_ranges[param] = {
            'min': data.iloc[:, i].min(),
            'max': data.iloc[:, i].max(),
            'median': data.iloc[:, i].median()
        }

    # Create sensitivity plots for each parameter
    for i, param in enumerate(var_names):
        print(f"Analyzing sensitivity for parameter: {param}")

        # Get parameter range (in log scale)
        param_min = param_ranges[param]['min']
        param_max = param_ranges[param]['max']
        param_values = np.linspace(param_min, param_max, 100)

        # Convert to original scale for visualization
        param_values_exp = np.exp(param_values)

        # Set other parameters to their median values (in log scale)
        other_params = [param_ranges[var_names[j]]['median'] for j in range(3)]

        # Create input array (in log scale)
        X_sensitivity = np.tile(other_params, (100, 1))
        X_sensitivity[:, i] = param_values

        # Normalize inputs
        X_sensitivity_norm = x_scaler.transform(X_sensitivity)

        # Predict Ra values
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_sensitivity_norm).to(device)
            predictions = model(X_tensor).cpu().numpy().flatten()
            Ra_pred = y_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()

        # Convert predicted Ra to original scale
        Ra_pred_exp = np.exp(Ra_pred)

        # Calculate sensitivity in log scale
        sensitivity_log = np.gradient(Ra_pred) / np.gradient(param_values)

        # Calculate sensitivity in original scale
        # For log-transformed data: d(exp(y))/d(exp(x)) = (exp(y)/exp(x))*(dy/dx)
        sensitivity_exp = (Ra_pred_exp / param_values_exp) * sensitivity_log

        # Plot 1: Original scale sensitivity
        fig, ax1 = plt.subplots(figsize=(12, 8), dpi=300)
        plt.style.use('seaborn-v0_8-pastel')

        color1 = '#3498db'  # Blue
        ax1.set_xlabel(f'{param} (Original Scale)', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Surface Roughness (Ra) - Original Scale', fontsize=14, fontweight='bold', color=color1)
        line1 = ax1.plot(param_values_exp, Ra_pred_exp, color=color1, linewidth=3, label='Ra')
        ax1.tick_params(axis='y', labelcolor=color1, labelsize=12)
        ax1.tick_params(axis='x', labelsize=12)

        # Create second y-axis for sensitivity
        ax2 = ax1.twinx()
        color2 = '#e74c3c'  # Red
        ax2.set_ylabel(f'Sensitivity (∂Ra/∂{param}) - Original Scale', fontsize=14, fontweight='bold', color=color2)
        line2 = ax2.plot(param_values_exp, sensitivity_exp, color=color2, linestyle='--', linewidth=2.5,
                         label='Sensitivity')
        ax2.tick_params(axis='y', labelcolor=color2, labelsize=12)

        plt.title(f'Sensitivity Analysis: {param} vs Surface Roughness (Original Scale)',
                  fontsize=16, fontweight='bold', pad=20)

        ax1.grid(False)

        # Add combined legend
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='best', fontsize=12, frameon=True)

        ax1.set_facecolor('#f8f9fa')
        for spine in ax1.spines.values():
            spine.set_linewidth(1.5)

        fig.tight_layout()

        if save_dir:
            plot_path = os.path.join(save_dir, f'{param}_sensitivity_original_scale.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Original scale sensitivity plot for {param} saved to: {plot_path}")
        plt.close()

        # Plot 2: Log scale sensitivity
        fig, ax1 = plt.subplots(figsize=(12, 8), dpi=300)
        plt.style.use('seaborn-v0_8-pastel')

        color1 = '#3498db'  # Blue
        ax1.set_xlabel(f'{param} (Log Scale)', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Surface Roughness (Ra) - Log Scale', fontsize=14, fontweight='bold', color=color1)
        line1 = ax1.plot(param_values, Ra_pred, color=color1, linewidth=3, label='Ra')
        ax1.tick_params(axis='y', labelcolor=color1, labelsize=12)
        ax1.tick_params(axis='x', labelsize=12)

        # Create second y-axis for sensitivity
        ax2 = ax1.twinx()
        color2 = '#e74c3c'  # Red
        ax2.set_ylabel(f'Sensitivity (∂Ra/∂{param}) - Log Scale', fontsize=14, fontweight='bold', color=color2)
        line2 = ax2.plot(param_values, sensitivity_log, color=color2, linestyle='--', linewidth=2.5,
                         label='Sensitivity')
        ax2.tick_params(axis='y', labelcolor=color2, labelsize=12)

        plt.title(f'Sensitivity Analysis: {param} vs Surface Roughness (Log Scale)',
                  fontsize=16, fontweight='bold', pad=20)

        ax1.grid(False)

        # Add combined legend
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='best', fontsize=12, frameon=True)

        ax1.set_facecolor('#f8f9fa')
        for spine in ax1.spines.values():
            spine.set_linewidth(1.5)

        fig.tight_layout()

        if save_dir:
            plot_path = os.path.join(save_dir, f'{param}_sensitivity_log_scale.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Log scale sensitivity plot for {param} saved to: {plot_path}")
        plt.close()

    # Create summary sensitivity comparison plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 12), dpi=300)
    fig.suptitle('Comprehensive Sensitivity Analysis Summary', fontsize=20, fontweight='bold')

    for i, param in enumerate(var_names):
        # Get parameter range
        param_min = param_ranges[param]['min']
        param_max = param_ranges[param]['max']
        param_values = np.linspace(param_min, param_max, 100)
        param_values_exp = np.exp(param_values)

        # Set other parameters to their median values
        other_params = [param_ranges[var_names[j]]['median'] for j in range(3)]
        X_sensitivity = np.tile(other_params, (100, 1))
        X_sensitivity[:, i] = param_values

        # Normalize and predict
        X_sensitivity_norm = x_scaler.transform(X_sensitivity)
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_sensitivity_norm).to(device)
            predictions = model(X_tensor).cpu().numpy().flatten()
            Ra_pred = y_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()

        Ra_pred_exp = np.exp(Ra_pred)
        sensitivity_log = np.gradient(Ra_pred) / np.gradient(param_values)
        sensitivity_exp = (Ra_pred_exp / param_values_exp) * sensitivity_log

        # Original scale plot
        ax1 = axes[0, i]
        color = ['#3498db', '#2ecc71', '#9b59b6'][i]
        ax1.plot(param_values_exp, Ra_pred_exp, color=color, linewidth=2.5)
        ax1.set_title(f'{param} - Original Scale', fontsize=12, fontweight='bold')
        ax1.set_xlabel(f'{param}', fontsize=10)
        ax1.set_ylabel('Ra', fontsize=10)
        ax1.grid(True, alpha=0.3)

        # Log scale plot
        ax2 = axes[1, i]
        ax2.plot(param_values, Ra_pred, color=color, linewidth=2.5)
        ax2.set_title(f'{param} - Log Scale', fontsize=12, fontweight='bold')
        ax2.set_xlabel(f'log({param})', fontsize=10)
        ax2.set_ylabel('log(Ra)', fontsize=10)
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_dir:
        summary_path = os.path.join(save_dir, 'sensitivity_analysis_summary.png')
        plt.savefig(summary_path, dpi=300, bbox_inches='tight')
        print(f"Sensitivity analysis summary saved to: {summary_path}")
    plt.close()

    print("Sensitivity analysis completed successfully!")


def analyze_parameter_interactions(model, x_scaler, y_scaler, data, var_names=None, save_dir=None):
    """
    Analyze how parameters interact to affect the surface roughness (Ra) with comprehensive visualizations.
    """
    if var_names is None:
        var_names = ['Vf', 'f', 'Vc']

    print("\nAnalyzing parameter interactions...")

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    device = next(model.parameters()).device
    model.eval()

    # Get parameter ranges (in log scale from the data)
    param_ranges = {}
    for i, param in enumerate(var_names):
        param_ranges[param] = {
            'min': data.iloc[:, i].min(),
            'max': data.iloc[:, i].max(),
            'median': data.iloc[:, i].median()
        }

    n_points = 50  # Resolution for interaction plots

    # Create 2D contour plots for each pair of parameters
    for i in range(len(var_names)):
        for j in range(i + 1, len(var_names)):
            param1 = var_names[i]
            param2 = var_names[j]

            print(f"Analyzing interaction between {param1} and {param2}...")

            # Create parameter meshgrid (in log scale)
            p1_range = np.linspace(param_ranges[param1]['min'], param_ranges[param1]['max'], n_points)
            p2_range = np.linspace(param_ranges[param2]['min'], param_ranges[param2]['max'], n_points)
            P1, P2 = np.meshgrid(p1_range, p2_range)

            # Fixed value for the third parameter (median in log scale)
            remaining_param_idx = [k for k in range(3) if k != i and k != j][0]
            remaining_param = var_names[remaining_param_idx]
            fixed_value = param_ranges[remaining_param]['median']

            # Prepare input data for model prediction
            grid_shape = P1.shape
            X_grid = np.zeros((grid_shape[0] * grid_shape[1], 3))

            for k in range(3):
                if k == i:
                    X_grid[:, k] = P1.flatten()
                elif k == j:
                    X_grid[:, k] = P2.flatten()
                else:
                    X_grid[:, k] = fixed_value

            # Normalize inputs
            X_grid_norm = x_scaler.transform(X_grid)

            # Predict Ra values using the model
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X_grid_norm).to(device)
                predictions = model(X_tensor).cpu().numpy().flatten()
                Ra_pred = y_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()

            # Reshape predictions to grid
            Ra_grid = Ra_pred.reshape(grid_shape)

            # Convert to original scale for visualization
            P1_exp = np.exp(P1)
            P2_exp = np.exp(P2)
            Ra_grid_exp = np.exp(Ra_grid)
            fixed_value_exp = np.exp(fixed_value)

            # Create contour plot in original scale
            plt.figure(figsize=(12, 10), dpi=300)
            plt.style.use('seaborn-v0_8-pastel')

            contour = plt.contourf(P1_exp, P2_exp, Ra_grid_exp, 50, cmap='viridis')

            # Add colorbar
            cbar = plt.colorbar(contour, pad=0.02)
            cbar.set_label('Surface Roughness (Ra) - Original Scale', fontsize=14, fontweight='bold')
            cbar.ax.tick_params(labelsize=12)

            plt.title(
                f'Interaction Effect: {param1} × {param2} on Surface Roughness\n({remaining_param} = {fixed_value_exp:.3f} - Original Scale)',
                fontsize=16, fontweight='bold', pad=20)
            plt.xlabel(f'{param1} (Original Scale)', fontsize=14, fontweight='bold')
            plt.ylabel(f'{param2} (Original Scale)', fontsize=14, fontweight='bold')
            plt.tick_params(axis='both', which='major', labelsize=12)

            # Add contour lines for clarity
            contour_lines = plt.contour(P1_exp, P2_exp, Ra_grid_exp, 10, colors='white', linewidths=0.7, alpha=0.8)
            plt.clabel(contour_lines, inline=True, fontsize=10, fmt='%.2f')

            plt.grid(False)
            plt.tight_layout()

            # Save interaction plot in original scale
            if save_dir:
                plot_path = os.path.join(save_dir, f'{param1}_{param2}_interaction_original.png')
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                print(f"Original scale interaction plot for {param1} and {param2} saved to: {plot_path}")
            plt.close()

            # Also create log-scale plot for comparison
            plt.figure(figsize=(12, 10), dpi=300)
            plt.style.use('seaborn-v0_8-pastel')

            contour = plt.contourf(P1, P2, Ra_grid, 50, cmap='viridis')
            cbar = plt.colorbar(contour, pad=0.02)
            cbar.set_label('Surface Roughness (Ra) - Log Scale', fontsize=14, fontweight='bold')
            cbar.ax.tick_params(labelsize=12)

            plt.title(
                f'Interaction Effect: {param1} × {param2} on Surface Roughness\n({remaining_param} = {fixed_value:.3f} - Log Scale)',
                fontsize=16, fontweight='bold', pad=20)
            plt.xlabel(f'{param1} (Log Scale)', fontsize=14, fontweight='bold')
            plt.ylabel(f'{param2} (Log Scale)', fontsize=14, fontweight='bold')
            plt.tick_params(axis='both', which='major', labelsize=12)

            # Add contour lines for clarity
            contour_lines = plt.contour(P1, P2, Ra_grid, 10, colors='white', linewidths=0.7, alpha=0.8)
            plt.clabel(contour_lines, inline=True, fontsize=10, fmt='%.2f')

            plt.grid(False)
            plt.tight_layout()

            # Save interaction plot in log scale
            if save_dir:
                plot_path = os.path.join(save_dir, f'{param1}_{param2}_interaction_log.png')
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                print(f"Log scale interaction plot for {param1} and {param2} saved to: {plot_path}")
            plt.close()

    # Create 3D surface plots for each parameter combination
    for i in range(len(var_names)):
        for j in range(i + 1, len(var_names)):
            param1 = var_names[i]
            param2 = var_names[j]

            print(f"Creating 3D surface plot for {param1} and {param2}...")

            # Create parameter meshgrid (in log scale)
            p1_range = np.linspace(param_ranges[param1]['min'], param_ranges[param1]['max'], n_points)
            p2_range = np.linspace(param_ranges[param2]['min'], param_ranges[param2]['max'], n_points)
            P1, P2 = np.meshgrid(p1_range, p2_range)

            # Fixed value for the third parameter (median in log scale)
            remaining_param_idx = [k for k in range(3) if k != i and k != j][0]
            remaining_param = var_names[remaining_param_idx]
            fixed_value = param_ranges[remaining_param]['median']

            # Prepare input data for model prediction
            grid_shape = P1.shape
            X_grid = np.zeros((grid_shape[0] * grid_shape[1], 3))

            for k in range(3):
                if k == i:
                    X_grid[:, k] = P1.flatten()
                elif k == j:
                    X_grid[:, k] = P2.flatten()
                else:
                    X_grid[:, k] = fixed_value

            # Normalize inputs
            X_grid_norm = x_scaler.transform(X_grid)

            # Predict Ra values using the model
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X_grid_norm).to(device)
                predictions = model(X_tensor).cpu().numpy().flatten()
                Ra_pred = y_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()

            # Reshape predictions to grid
            Ra_grid = Ra_pred.reshape(grid_shape)

            # Convert to original scale for visualization
            P1_exp = np.exp(P1)
            P2_exp = np.exp(P2)
            Ra_grid_exp = np.exp(Ra_grid)
            fixed_value_exp = np.exp(fixed_value)

            # Create 3D surface plot
            fig = plt.figure(figsize=(14, 12), dpi=300)
            ax = fig.add_subplot(111, projection='3d')

            # Create surface plot using original scale values
            surf = ax.plot_surface(P1_exp, P2_exp, Ra_grid_exp, cmap='viridis', edgecolor='none',
                                   alpha=0.8, linewidth=0, antialiased=True)

            # Add colorbar
            cbar = fig.colorbar(surf, ax=ax, pad=0.05, shrink=0.8)
            cbar.set_label('Surface Roughness (Ra)', fontsize=14, fontweight='bold')
            cbar.ax.tick_params(labelsize=12)

            # Set labels
            ax.set_xlabel(f'{param1} (Original Scale)', fontsize=14, fontweight='bold')
            ax.set_ylabel(f'{param2} (Original Scale)', fontsize=14, fontweight='bold')
            ax.set_zlabel('Surface Roughness (Ra)', fontsize=14, fontweight='bold')

            # Set title
            plt.title(
                f'3D Surface Plot: Ra vs {param1} and {param2}\n({remaining_param} = {fixed_value_exp:.3f})',
                fontsize=16, fontweight='bold', pad=20)

            # Improve tick labels
            ax.tick_params(axis='both', which='major', labelsize=12)
            ax.grid(False)
            ax.set_facecolor('white')

            plt.tight_layout()

            # Save 3D surface plot
            if save_dir:
                plot_path = os.path.join(save_dir, f'{param1}_{param2}_3d_surface.png')
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                print(f"3D surface plot for {param1} and {param2} saved to: {plot_path}")
            plt.close()

    # Create 4D visualization (3D space + color for Ra)
    print("Creating 4D visualization...")

    n_grid_3d = 20  # Lower resolution for 3D grid
    Vf_range = np.linspace(param_ranges['Vf']['min'], param_ranges['Vf']['max'], n_grid_3d)
    f_range = np.linspace(param_ranges['f']['min'], param_ranges['f']['max'], n_grid_3d)
    Vc_range = np.linspace(param_ranges['Vc']['min'], param_ranges['Vc']['max'], n_grid_3d)

    # Create 3D grid of points
    points = []
    for Vf_val in Vf_range:
        for f_val in f_range:
            for Vc_val in Vc_range:
                points.append([Vf_val, f_val, Vc_val])

    points = np.array(points)

    # Normalize inputs for model prediction
    points_norm = x_scaler.transform(points)

    # Predict Ra values
    with torch.no_grad():
        points_tensor = torch.FloatTensor(points_norm).to(device)
        predictions = model(points_tensor).cpu().numpy().flatten()
        Ra_pred = y_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()

    # Convert to original scale
    points_exp = np.exp(points)
    Ra_pred_exp = np.exp(Ra_pred)

    # Sample points for visualization (stratified sampling)
    n_samples = 1000
    Ra_min = np.min(Ra_pred_exp)
    Ra_max = np.max(Ra_pred_exp)
    Ra_bins = np.linspace(Ra_min, Ra_max, 10)

    sampled_indices = []
    for i in range(len(Ra_bins) - 1):
        bin_indices = np.where((Ra_pred_exp >= Ra_bins[i]) & (Ra_pred_exp < Ra_bins[i + 1]))[0]
        if len(bin_indices) > 0:
            n_bin_samples = max(10, int(n_samples * len(bin_indices) / len(Ra_pred_exp)))
            sampled_bin_indices = np.random.choice(bin_indices, size=min(n_bin_samples, len(bin_indices)),
                                                   replace=False)
            sampled_indices.extend(sampled_bin_indices)

    # Ensure we have enough samples
    if len(sampled_indices) < n_samples:
        remaining_indices = np.setdiff1d(np.arange(len(Ra_pred_exp)), sampled_indices)
        if len(remaining_indices) > 0:
            additional_samples = np.random.choice(
                remaining_indices,
                size=min(n_samples - len(sampled_indices), len(remaining_indices)),
                replace=False
            )
            sampled_indices.extend(additional_samples)

    # Get sampled points
    sampled_points = points_exp[sampled_indices]
    sampled_Ra = Ra_pred_exp[sampled_indices]

    # Create 4D plot (3D scatter with color as Ra)
    fig = plt.figure(figsize=(14, 12), dpi=300)
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot with points colored by Ra value
    scatter = ax.scatter(
        sampled_points[:, 0],  # Vf
        sampled_points[:, 1],  # f
        sampled_points[:, 2],  # Vc
        c=sampled_Ra,  # Ra (color)
        cmap='viridis',
        s=50,
        alpha=0.7,
        edgecolors='k',
        linewidth=0.5
    )

    # Add colorbar
    cbar = fig.colorbar(scatter, ax=ax, pad=0.05, shrink=0.8)
    cbar.set_label('Surface Roughness (Ra)', fontsize=14, fontweight='bold')
    cbar.ax.tick_params(labelsize=12)

    # Set labels
    ax.set_xlabel('Vf (Original Scale)', fontsize=14, fontweight='bold')
    ax.set_ylabel('f (Original Scale)', fontsize=14, fontweight='bold')
    ax.set_zlabel('Vc (Original Scale)', fontsize=14, fontweight='bold')

    # Set title
    plt.title('4D Visualization: Surface Roughness (Ra)\nwith respect to Vf, f, and Vc',
              fontsize=16, fontweight='bold', pad=20)

    # Improve appearance
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.grid(False)

    # Add annotation
    ax.text2D(0.02, 0.02, "Color represents Ra value", transform=ax.transAxes,
              fontsize=12, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    plt.tight_layout()

    # Save 4D visualization
    if save_dir:
        plot_path = os.path.join(save_dir, '4d_visualization.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"4D visualization saved to: {plot_path}")
    plt.close()

    # Create comprehensive interaction summary
    fig, axes = plt.subplots(2, 3, figsize=(20, 12), dpi=300)
    fig.suptitle('Parameter Interaction Summary (Original Scale)', fontsize=20, fontweight='bold')

    interaction_pairs = [(0, 1), (0, 2), (1, 2)]
    pair_names = [('Vf', 'f'), ('Vf', 'Vc'), ('f', 'Vc')]

    for idx, ((i, j), (param1, param2)) in enumerate(zip(interaction_pairs, pair_names)):
        # Create parameter meshgrid for summary (lower resolution)
        p1_range = np.linspace(param_ranges[param1]['min'], param_ranges[param1]['max'], 30)
        p2_range = np.linspace(param_ranges[param2]['min'], param_ranges[param2]['max'], 30)
        P1, P2 = np.meshgrid(p1_range, p2_range)

        # Fixed value for the third parameter
        remaining_param_idx = [k for k in range(3) if k != i and k != j][0]
        remaining_param = var_names[remaining_param_idx]
        fixed_value = param_ranges[remaining_param]['median']

        # Prepare input data
        grid_shape = P1.shape
        X_grid = np.zeros((grid_shape[0] * grid_shape[1], 3))

        for k in range(3):
            if k == i:
                X_grid[:, k] = P1.flatten()
            elif k == j:
                X_grid[:, k] = P2.flatten()
            else:
                X_grid[:, k] = fixed_value

        # Normalize and predict
        X_grid_norm = x_scaler.transform(X_grid)
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_grid_norm).to(device)
            predictions = model(X_tensor).cpu().numpy().flatten()
            Ra_pred = y_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()

        Ra_grid = Ra_pred.reshape(grid_shape)
        P1_exp = np.exp(P1)
        P2_exp = np.exp(P2)
        Ra_grid_exp = np.exp(Ra_grid)

        # Top row: Contour plots
        ax1 = axes[0, idx]
        contour = ax1.contourf(P1_exp, P2_exp, Ra_grid_exp, 20, cmap='viridis')
        ax1.set_title(f'{param1} × {param2}', fontsize=14, fontweight='bold')
        ax1.set_xlabel(f'{param1}', fontsize=12)
        ax1.set_ylabel(f'{param2}', fontsize=12)

        # Bottom row: Show sensitivity along one dimension
        ax2 = axes[1, idx]

        # Take a slice through the middle of the second parameter
        middle_idx = Ra_grid_exp.shape[0] // 2
        p1_slice = P1_exp[middle_idx, :]
        Ra_slice = Ra_grid_exp[middle_idx, :]

        ax2.plot(p1_slice, Ra_slice, linewidth=2.5, color='#3498db')
        ax2.set_title(f'Ra vs {param1} (fixed {param2})', fontsize=14, fontweight='bold')
        ax2.set_xlabel(f'{param1}', fontsize=12)
        ax2.set_ylabel('Ra', fontsize=12)
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_dir:
        summary_path = os.path.join(save_dir, 'interaction_summary.png')
        plt.savefig(summary_path, dpi=300, bbox_inches='tight')
        print(f"Interaction summary plot saved to: {summary_path}")
    plt.close()

    print("Parameter interaction analysis completed successfully!")


def format_time(seconds):
    return str(timedelta(seconds=int(seconds)))


def plot_training_process(train_losses, val_losses, save_path=None):
    plt.figure(figsize=(12, 8), dpi=300)
    epochs = range(1, len(train_losses) + 1)

    plt.style.use('seaborn-v0_8-pastel')

    plt.plot(epochs, train_losses, '-', color='#3498db', label='Training Loss',
             linewidth=2.5, alpha=0.8)
    plt.plot(epochs, val_losses, '-', color='#e74c3c', label='Validation Loss',
             linewidth=2.5, alpha=0.8)

    plt.title('Training and Validation Loss over Epochs', fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('Epoch', fontsize=14, fontweight='bold')
    plt.ylabel('Loss', fontsize=14, fontweight='bold')

    plt.grid(False)
    plt.legend(fontsize=12, frameon=True, facecolor='white', edgecolor='black')

    ax = plt.gca()
    ax.set_facecolor('#f8f9fa')
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Training plot saved to: {save_path}")
    else:
        plt.show()


def plot_prediction_scatter(true_values, predictions, dataset_name, param_name='Ra', save_path=None):
    # Apply exponential transformation to true values and predictions since data was log-transformed
    true_values_exp = np.exp(true_values)
    predictions_exp = np.exp(predictions)

    # Calculate metrics on exponential scale
    r2 = r2_score(true_values_exp, predictions_exp)
    rmse = np.sqrt(mean_squared_error(true_values_exp, predictions_exp))

    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)

    plt.style.use('seaborn-v0_8-pastel')

    ax.scatter(true_values_exp, predictions_exp, alpha=0.7, color='#3498db',
               s=80, edgecolors='navy', linewidths=0.8)

    min_val = min(true_values_exp.min(), predictions_exp.min())
    max_val = max(true_values_exp.max(), predictions_exp.max())
    perfect_line = np.linspace(min_val, max_val, 100)

    ax.plot(perfect_line, perfect_line, '--', color='#e74c3c', linewidth=2.5,
            label='Perfect Prediction')

    ax.set_title(f'{param_name.upper()} - {dataset_name} (Actual Scale)\nR² = {r2:.4f}, RMSE = {rmse:.4f}',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Actual Values (Exponential Scale)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Predicted Values (Exponential Scale)', fontsize=14, fontweight='bold')

    ax.grid(False)
    ax.legend(fontsize=12, frameon=True, facecolor='white', edgecolor='black')

    ax.set_facecolor('#f8f9fa')
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Scatter plot for {dataset_name} saved to: {save_path}")
    else:
        plt.show()

    return r2, rmse


def train_dynamic_pinn(model, train_loader, X_val, y_val, optimizer, scheduler, device,
                       x_scaler, y_scaler, physics_weight=0.000137742, num_epochs=1000,
                       adaptive_physics_weight=False):
    """
    Train the dynamic PINN model with adaptive physics weight and early stopping
    """
    mse_loss = nn.MSELoss()
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience = 30
    patience_counter = 0

    total_start_time = time.time()

    print("\nStarting training...")
    for epoch in range(num_epochs):
        epoch_start_time = time.time()

        model.train()
        epoch_train_loss = 0.0
        epoch_mse_loss = 0.0
        epoch_physics_loss = 0.0

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            pred_Ra = model(batch_X)
            mse = mse_loss(pred_Ra, batch_y)

            Ra = pred_Ra.squeeze()
            physics_loss = physical_constraint(model, batch_X, Ra, y_scaler, x_scaler)

            current_physics_weight = physics_weight
            if adaptive_physics_weight:
                progress = epoch / num_epochs
                current_physics_weight = physics_weight * (1 + 9 * progress)

            loss = mse + current_physics_weight * physics_loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_train_loss += loss.item()
            epoch_mse_loss += mse.item()
            epoch_physics_loss += physics_loss.item()

        # Validation phase
        model.eval()
        with torch.no_grad():
            X_val_tensor = torch.FloatTensor(X_val).to(device)
            y_val_tensor = torch.FloatTensor(y_val).to(device)
            val_pred = model(X_val_tensor)
            val_loss = mse_loss(val_pred, y_val_tensor)

            train_rmse = np.sqrt(epoch_mse_loss / len(train_loader))
            val_rmse = np.sqrt(val_loss.item())
            train_losses.append(train_rmse)
            val_losses.append(val_rmse)

            epoch_time = time.time() - epoch_start_time
            total_time = time.time() - total_start_time

            print(
                f"Epoch {epoch + 1}/{num_epochs} - Time: {format_time(epoch_time)} - Total Time: {format_time(total_time)}")
            print(f"Training RMSE: {train_rmse:.6f}, Validation RMSE: {val_rmse:.6f}")
            print(
                f"MSE Loss: {epoch_mse_loss / len(train_loader):.6f}, Physics Loss: {epoch_physics_loss / len(train_loader):.6f}")

            if adaptive_physics_weight:
                print(f"Current Physics Weight: {current_physics_weight:.8f}")

            scheduler.step(val_loss.item())

            if val_loss.item() < best_val_loss:
                best_val_loss = val_loss.item()
                patience_counter = 0

                checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'causal_edges': model.causal_edges,
                    'edge_weights': model.edge_weights,
                    'interaction_order': model.interaction_order
                }
                torch.save(checkpoint, MODEL_FILE)
                print(f"Model saved with validation loss: {val_loss.item():.6f}")
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print("Early stopping triggered")
                break

    total_training_time = time.time() - total_start_time
    print(f"\nTotal training time: {format_time(total_training_time)}")

    return train_losses, val_losses


def evaluate_dynamic_pinn(model, X, y, y_scaler, dataset_name="", save_dir=None):
    """
    Evaluate the dynamic PINN model and create prediction plots
    """
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X).to(device)
        predictions = model(X_tensor).cpu().numpy().flatten()
        predictions_transformed = y_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plot_path = os.path.join(save_dir, f'prediction_scatter_{dataset_name.replace(" ", "_").lower()}.png')
        else:
            plot_path = None

        r2, rmse = plot_prediction_scatter(y, predictions_transformed, dataset_name, save_path=plot_path)

        metrics = {
            "Ra_r2": r2,
            "Ra_rmse": rmse
        }

        print(f"{dataset_name} Ra - R²: {r2:.4f}, RMSE: {rmse:.4f}")

        return predictions_transformed, metrics


def extract_equation_terms(model):
    """
    Extract the learned equation terms from the dynamic PINN model

    Returns a dictionary with coefficient names and values
    """
    equation_terms = {}

    # Add constant term
    equation_terms['Constant (A)'] = model.A.item()

    # Add main effect terms
    for name, param in model.main_effect_params.items():
        equation_terms[name] = param.item()

    # Add interaction terms if available
    if hasattr(model, 'interaction_params'):
        for name, param in model.interaction_params.items():
            equation_terms[name] = param.item()

    return equation_terms


def format_equation(equation_terms, is_log_scale=True, threshold=0.001):
    """
    Format the equation for display, filtering out terms with coefficients below threshold

    Parameters:
    -----------
    equation_terms: dict
        Dictionary with coefficient names and values
    is_log_scale: bool
        Whether to format for log scale or original scale
    threshold: float
        Minimum absolute coefficient value to include in the equation

    Returns:
    --------
    str: Formatted equation
    """
    if is_log_scale:
        # Format log-scale equation
        equation = f"log(Ra) = {equation_terms['Constant (A)']:.6f}"

        # Add main effects
        for name, value in equation_terms.items():
            if 'B_' in name and abs(value) >= threshold:
                feature = name.split('_')[1]
                sign = '+' if value >= 0 else '-'
                equation += f" {sign} {abs(value):.6f}*log({feature})"

        # Add interaction effects
        for name, value in equation_terms.items():
            if 'I_' in name and abs(value) >= threshold:
                features = name.split('_')[1:]
                sign = '+' if value >= 0 else '-'
                terms = '*'.join([f"log({f})" for f in features])
                equation += f" {sign} {abs(value):.6f}*{terms}"
    else:
        # Format original scale equation (exponential form)
        constant = np.exp(equation_terms['Constant (A)'])
        equation = f"Ra = {constant:.6f}"

        # Add main effects as power terms
        for name, value in equation_terms.items():
            if 'B_' in name and abs(value) >= threshold:
                feature = name.split('_')[1]
                equation += f" × {feature}^{value:.6f}"

        # Add interaction effects
        interaction_terms = []
        for name, value in equation_terms.items():
            if 'I_' in name and abs(value) >= threshold:
                features = name.split('_')[1:]
                term = " × ".join([f"{f}" for f in features])
                interaction_terms.append(f"({term})^{value:.6f}")

        if interaction_terms:
            equation += " × " + " × ".join(interaction_terms)

    return equation


def main():
    # Record total runtime
    total_start_time = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    try:
        # Update this path to your data file location
        data_path = r'C:/Users/Administrator.DESKTOP-JC66FHP/Desktop/cipinn/logsjzq2000-1.csv'
        full_data = pd.read_csv(data_path)

        print("\nNOTE: The dataset contains log-transformed values of manufacturing parameters and surface roughness.")
        print("All analyses will be performed in both log scale and original scale (after exponential transformation).")

        # Modified data split
        train_data = full_data.iloc[4:1500]  # 10-2000 rows
        test_data1 = full_data.iloc[:3]  # 1-9 rows
        test_data2 = full_data.iloc[1500:]  # 2000-2200 rows

        print(f"Data loaded successfully:")
        print(f"Train data shape: {train_data.shape}")
        print(f"Test data 1 shape: {test_data1.shape}")
        print(f"Test data 2 shape: {test_data2.shape}")
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return

    # Variable names
    var_names = ['Vf', 'f', 'Vc', 'Ra']

    # Create directory for dynamic PINN results
    DYNAMIC_PINN_PATH = os.path.join(PLOTS_PATH, 'dynamic_pinn')
    os.makedirs(DYNAMIC_PINN_PATH, exist_ok=True)

    # Step 1: Analyze correlations in both scales
    print("\nStep 1: Analyzing correlations in log and original scales...")
    correlation_matrix_log, correlation_matrix_exp = analyze_correlations(
        full_data, var_names=var_names, save_dir=CORRELATION_PATH
    )

    # Step 2: Discover causal structure
    print("\nStep 2: Discovering causal structure from data...")
    causal_edges, edge_weights, causal_graph = discover_causal_structure(
        full_data, var_names=var_names, significance_level=0.01
    )

    # Visualize the discovered causal structure
    causal_graph_path = os.path.join(DYNAMIC_PINN_PATH, 'causal_structure.png')
    visualize_causal_structure(
        causal_graph, edge_weights, var_names,
        save_path=causal_graph_path,
        title="Discovered Causal Structure"
    )

    # Data preprocessing
    train_data = train_data.dropna()
    test_data1 = test_data1.dropna()
    test_data2 = test_data2.dropna()

    # Combine test sets for simplicity in this example
    test_data_combined = pd.concat([test_data1, test_data2])

    # Prepare training data
    X_train_full = train_data.iloc[:, :3].values  # Vf, f, Vc
    y_train_full = train_data.iloc[:, 3].values.reshape(-1, 1)  # Ra

    # Split training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42
    )

    # Standardization
    X_scaler = StandardScaler()
    y_scaler = StandardScaler()

    # Standardize training data
    X_train_norm = X_scaler.fit_transform(X_train)
    X_val_norm = X_scaler.transform(X_val)

    # Standardize test data
    X_test_norm = X_scaler.transform(test_data_combined.iloc[:, :3].values)
    y_test = test_data_combined.iloc[:, 3].values.reshape(-1, 1)

    y_train_norm = y_scaler.fit_transform(y_train)
    y_val_norm = y_scaler.transform(y_val)

    # Step 3: Train dynamic PINN
    print("\nStep 3: Training dynamic PINN model...")

    # Find the index of 'Ra' in var_names
    Ra_idx = var_names.index('Ra')

    # Filter causal edges to include only those that point to Ra
    Ra_causal_edges = [(source, target) for source, target in causal_edges if target == Ra_idx]

    # If no edges to Ra are found, create default edges (all features to Ra)
    if not Ra_causal_edges:
        print("No causal edges to Ra found. Using all features as causal inputs.")
        Ra_causal_edges = [(i, Ra_idx) for i in range(len(var_names)) if i != Ra_idx]

    # Initialize model
    model = DynamicPINN(
        input_dim=3,
        output_dim=1,
        causal_edges=Ra_causal_edges,
        edge_weights=edge_weights,
        interaction_order=2
    ).to(device)

    # Initialize optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=0.00631744, weight_decay=0.000551604)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.154922, patience=20, verbose=True
    )

    # Build DataLoader
    train_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_train_norm),
        torch.FloatTensor(y_train_norm)
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=64, shuffle=True
    )

    # Train the model
    train_losses, val_losses = train_dynamic_pinn(
        model, train_loader, X_val_norm, y_val_norm,
        optimizer, scheduler, device,
        x_scaler=X_scaler, y_scaler=y_scaler,
        num_epochs=300, adaptive_physics_weight=True
    )

    # Plot training process
    plot_training_process(
        train_losses, val_losses,
        save_path=os.path.join(DYNAMIC_PINN_PATH, 'training_losses.png')
    )

    # Load best model
    checkpoint = torch.load(MODEL_FILE)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Evaluate model
    predictions, metrics = evaluate_dynamic_pinn(
        model, X_test_norm, y_test.flatten(), y_scaler,
        dataset_name="Test",
        save_dir=DYNAMIC_PINN_PATH
    )

    # Extract and print learned equations
    equation_terms = extract_equation_terms(model)
    log_equation = format_equation(equation_terms, is_log_scale=True)
    original_equation = format_equation(equation_terms, is_log_scale=False)

    print(f"\nLearned Physical Equation (Log Scale):")
    print(log_equation)
    print(f"\nLearned Physical Equation (Original Scale):")
    print(original_equation)

    # Step 4: Perform sensitivity analysis
    print("\nStep 4: Performing comprehensive sensitivity analysis...")
    perform_sensitivity_analysis(
        model, X_scaler, y_scaler, full_data,
        var_names=['Vf', 'f', 'Vc'], save_dir=SENSITIVITY_PATH
    )

    # Step 5: Analyze parameter interactions
    print("\nStep 5: Analyzing parameter interactions and their effects on surface roughness...")
    analyze_parameter_interactions(
        model, X_scaler, y_scaler, full_data,
        var_names=['Vf', 'f', 'Vc'], save_dir=INTERACTIONS_PATH
    )

    # Calculate and output total runtime
    total_run_time = time.time() - total_start_time
    print(f"\nTotal program runtime: {format_time(total_run_time)}")
    print("\nEnhanced Dynamic PINN analysis completed successfully!")
    print(f"\nResults saved to:")
    print(f"- Main plots: {os.path.abspath(DYNAMIC_PINN_PATH)}")
    print(f"- Correlation analysis: {os.path.abspath(CORRELATION_PATH)}")
    print(f"- Sensitivity analysis: {os.path.abspath(SENSITIVITY_PATH)}")
    print(f"- Parameter interactions: {os.path.abspath(INTERACTIONS_PATH)}")

    print("\nAnalysis Features Completed:")
    print("✓ Causal structure discovery and visualization")
    print("✓ Multi-scale correlation analysis (log and original scales)")
    print("✓ Dynamic PINN training with physics constraints")
    print("✓ Comprehensive sensitivity analysis")
    print("✓ Advanced parameter interaction visualization")
    print("✓ 2D contour plots for parameter interactions")
    print("✓ 3D surface plots for parameter relationships")
    print("✓ 4D visualization (3D scatter + color mapping)")
    print("✓ Equation extraction and interpretation")


if __name__ == "__main__":
    main()