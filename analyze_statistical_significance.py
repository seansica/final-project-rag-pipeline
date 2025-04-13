import json
import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

# Parse command line arguments
parser = argparse.ArgumentParser(description='Analyze statistical significance of RAG experiment results')
parser.add_argument('--results_path', type=str, default='results/phase1_ragas_20250411_174100',
                    help='Path to the results folder (default: results/phase1_ragas_20250411_174100)')
parser.add_argument('--phase', type=int, choices=[1, 2, 3], default=None,
                    help='Experiment phase: 1=embedding models, 2=chunk size/overlap, 3=retrieval methods')
args = parser.parse_args()

results_path = args.results_path

# Auto-detect phase from results path if not explicitly provided
if args.phase is None:
    if 'phase1' in results_path:
        phase = 1
    elif 'phase2' in results_path:
        phase = 2
    elif 'phase3' in results_path:
        phase = 3
    else:
        # Default to phase 2 if can't determine
        phase = 2
        print(f"Could not determine phase from path '{results_path}'. Defaulting to phase 2.")
else:
    phase = args.phase

print(f"Analyzing phase {phase} experiments from {results_path}")

# Load the results data
with open(os.path.join(results_path, 'results.json'), 'r') as f:
    results = json.load(f)

# Extract metrics into a structured format
experiments = []
experiments_config = {}  # Store configs separately keyed by experiment ID
metrics_of_interest = [
    'correctness', 'groundedness', 'relevance', 'retrieval_relevance',
    'ragas_answer_accuracy', 'ragas_context_relevance', 'ragas_faithfulness',
    'ragas_response_relevancy', 'deepeval_faithfulness', 'deepeval_geval',
    'bertscore_evaluator'
]

for experiment in results:
    # Handle different formats of experiment ID
    if 'experiment_id' in experiment:
        exp_id = experiment['experiment_id']
    elif 'id' in experiment:
        exp_id = experiment['id']
    else:
        # Skip experiments with no ID
        print(f"Warning: Skipping experiment with no ID: {experiment.keys()}")
        continue
    
    metrics = {}
    
    # Handle different formats of metrics structure
    if 'metrics' in experiment and 'feedback' in experiment['metrics']:
        metrics_source = experiment['metrics']['feedback']
    elif 'feedback' in experiment:
        metrics_source = experiment['feedback']
    else:
        # Try to find metrics at the top level
        metrics_source = experiment
    
    for metric in metrics_of_interest:
        if metric in metrics_source:
            # Make sure we have the required statistics
            if all(k in metrics_source[metric] for k in ('mean', 'std', 'count')):
                metrics[metric] = {
                    'mean': metrics_source[metric]['mean'],
                    'std': metrics_source[metric]['std'],
                    'count': metrics_source[metric]['count']
                }
    
    # Store the experiment
    experiments.append({
        'id': exp_id,
        'metrics': metrics
    })
    
    # Store the full config if available
    if 'config' in experiment:
        experiments_config[exp_id] = experiment['config']

# Print experiment IDs for debugging
print(f"\nFound {len(experiments)} experiments with {len(experiments_config)} configs")
for i, exp in enumerate(experiments):
    print(f"Experiment {i+1}: ID={exp['id']}")
    if exp['id'] in experiments_config:
        config = experiments_config[exp['id']]
        if 'embedding_model' in config:
            print(f"  Embedding model: {config['embedding_model']}")
        if 'chunk_size' in config:
            print(f"  Chunk size: {config['chunk_size']}")
        if 'chunk_overlap' in config:
            print(f"  Chunk overlap: {config['chunk_overlap']}")

# Function to perform ANOVA test
def perform_anova(metric_name):
    groups = []
    for exp in experiments:
        if metric_name in exp['metrics']:
            mean = exp['metrics'][metric_name]['mean']
            std = exp['metrics'][metric_name]['std']
            count = exp['metrics'][metric_name]['count']
            
            # Reconstruct approximate data points using mean and std
            # This is a rough approximation since we don't have the raw data
            # Assuming normal distribution centered at mean with given std
            simulated_data = np.random.normal(mean, std, count)
            groups.append(simulated_data)
    
    # Check if we have at least 2 groups to compare
    if len(groups) < 2:
        return 0.0, 1.0  # Return values indicating no significance
    
    # Perform one-way ANOVA
    f_val, p_val = stats.f_oneway(*groups)
    return f_val, p_val

# Function to perform t-tests between experiments
def perform_pairwise_ttests(metric_name):
    results = []
    
    for i in range(len(experiments)):
        for j in range(i+1, len(experiments)):
            exp1 = experiments[i]
            exp2 = experiments[j]
            
            # Skip if either experiment doesn't have this metric
            if metric_name not in exp1['metrics'] or metric_name not in exp2['metrics']:
                continue
            
            mean1 = exp1['metrics'][metric_name]['mean']
            std1 = exp1['metrics'][metric_name]['std']
            count1 = exp1['metrics'][metric_name]['count']
            
            mean2 = exp2['metrics'][metric_name]['mean']
            std2 = exp2['metrics'][metric_name]['std']
            count2 = exp2['metrics'][metric_name]['count']
            
            # Calculate t-statistic and p-value
            # Using Welch's t-test (unequal variances)
            t_stat, p_val = stats.ttest_ind_from_stats(
                mean1=mean1, std1=std1, nobs1=count1, 
                mean2=mean2, std2=std2, nobs2=count2,
                equal_var=False
            )
            
            results.append({
                'exp1': exp1['id'],
                'exp2': exp2['id'],
                'metric': metric_name,
                't_stat': float(t_stat),  # Convert to Python float
                'p_value': float(p_val),  # Convert to Python float
                'significant': bool(p_val < 0.05)
            })
    
    return results

# Run ANOVA for each metric
print("=== ANOVA Results ===")
anova_results = {}
for metric in metrics_of_interest:
    f_val, p_val = perform_anova(metric)
    anova_results[metric] = {
        'f_value': float(f_val),  # Convert to Python float
        'p_value': float(p_val),  # Convert to Python float
        'significant': bool(p_val < 0.05)
    }
    print(f"{metric}: F={f_val:.4f}, p={p_val:.4f}, significant={p_val < 0.05}")

# Run pairwise t-tests for all metrics
all_pairwise_results = []
for metric in metrics_of_interest:
    results = perform_pairwise_ttests(metric)
    all_pairwise_results.extend(results)

# Convert to DataFrame for easier analysis
df_results = pd.DataFrame(all_pairwise_results)

# Count how many significant differences we found
significant_count = df_results['significant'].sum()
total_tests = len(df_results)
print(f"\nFound {significant_count} significant differences out of {total_tests} comparisons ({significant_count/total_tests:.2%})")

# Calculate the means and standard errors across experiments for each metric
mean_values = {metric: [] for metric in metrics_of_interest}
se_values = {metric: [] for metric in metrics_of_interest}
exp_names = []
exp_indices = {metric: [] for metric in metrics_of_interest}  # Track which experiments have each metric

for i, exp in enumerate(experiments):
    # Get experiment ID
    exp_id = exp['id']
    
    # Get config if available from our pre-built dictionary
    if exp_id in experiments_config:
        config = experiments_config[exp_id]
        
        # Create display name based on phase
        if phase == 1:  # Embedding model comparison
            # For Phase 1, use the embedding model and team type
            emb_model = config.get('embedding_model', 'unknown')
            team_type = config.get('team_type', '')
            
            # Shorten long embedding model names
            if isinstance(emb_model, str) and len(emb_model) > 12:
                short_name = emb_model[:10] + '...'
            else:
                short_name = emb_model
                
            display_name = f"{team_type}-{short_name}"
            
        elif phase == 3:  # Retriever method comparison
            # For Phase 3, use the retriever type
            ret_type = config.get('retriever_type', 'unknown')
            
            # Add threshold for similarity_score_threshold
            if ret_type == 'similarity_score_threshold' and 'retriever_kwargs' in config:
                threshold = config['retriever_kwargs'].get('similarity_threshold', 0.0)
                display_name = f"{ret_type}-{threshold}"
            else:
                display_name = ret_type
                
        else:  # Phase 2 (default) - chunk size and overlap
            # For Phase 2, use chunk size and overlap
            chunk_size = config.get('chunk_size', 'N/A')
            chunk_overlap = config.get('chunk_overlap', 'N/A')
            display_name = f"cs{chunk_size}-co{chunk_overlap}"
    else:
        # Extract from ID if config not available
        id_parts = exp_id.split('-')
        
        # Try to guess based on ID and phase
        if phase == 1 and 'emb-' in exp_id:  # Embedding model
            # Try to extract embedding model name
            emb_idx = id_parts.index('emb') if 'emb' in id_parts else -1
            if emb_idx >= 0 and emb_idx + 1 < len(id_parts):
                emb_name = id_parts[emb_idx + 1]
                display_name = f"unknown-{emb_name}"
            else:
                display_name = exp_id
                
        elif phase == 2:  # Chunk size and overlap
            # Look for cs and co patterns
            cs_part = next((p for p in id_parts if p.startswith('cs')), None)
            co_part = next((p for p in id_parts if p.startswith('co')), None)
            
            if cs_part and co_part:
                display_name = f"{cs_part}-{co_part}"
            else:
                display_name = exp_id
                
        elif phase == 3:  # Retriever method
            # Look for retriever type
            if 'mmr' in id_parts:
                display_name = 'mmr'
            elif 'similarity' in id_parts:
                display_name = 'similarity'
            else:
                display_name = exp_id
        else:
            display_name = exp_id
    
    # Debug output to verify display names
    print(f"Experiment {i+1}: ID={exp_id}, Display={display_name}")
    
    exp_names.append(display_name)
    
    for metric in metrics_of_interest:
        if metric in exp['metrics']:
            mean_values[metric].append(exp['metrics'][metric]['mean'])
            # Standard error = std / sqrt(n)
            se = exp['metrics'][metric]['std'] / np.sqrt(exp['metrics'][metric]['count'])
            se_values[metric].append(se)
            exp_indices[metric].append(i)  # Store experiment index for this metric

# Function to create a plot showing mean and standard error for a metric with regression analysis
def plot_metric_comparison(metric, exp_names, means, standard_errors):
    if not means:  # Skip if no data for this metric
        print(f"Skipping plot for {metric} - no data available")
        return
    
    # We need to use exp_names corresponding to the indices in exp_indices[metric]
    selected_names = [exp_names[i] for i in exp_indices[metric]]
    
    # Extract parameters for visualization and regression
    param_values = []   # Primary parameter (embedding model, chunk size, retriever type)
    param2_values = []  # Secondary parameter (chunk overlap for phase 2)
    display_names = []  # Display names for the plot
    
    # Get experiment IDs and experiment objects matching the selected names
    selected_exps = [exp for exp in experiments if any(metric in exp['metrics'] for metric in metrics_of_interest)]
    selected_ids = [exp['id'] for exp in selected_exps]
    
    # Extract parameters based on phase
    for i, exp_id in enumerate(selected_ids):
        # Get config from our pre-populated dictionary
        config = experiments_config.get(exp_id, {})
        
        if phase == 1:  # Embedding Model Comparison
            # Get embedding model name (primary parameter for Phase 1)
            emb_model = config.get('embedding_model', 'unknown')
            team_type = config.get('team_type', '')
            
            # For regression analysis, we'll use a numeric index for the embedding model
            # This is just for visualization purposes since regression on categorical variables isn't meaningful
            if emb_model == 'multi-qa-mpnet-base-dot-v1':
                param_index = 1
            elif emb_model == 'all-mpnet-base-v2':
                param_index = 2
            elif emb_model == 'all-MiniLM-L6-v2':
                param_index = 3
            elif emb_model == 'all-distilroberta-v1':
                param_index = 4
            elif emb_model == 'multi-qa-mpnet-base-cos-v1':
                param_index = 5
            else:
                param_index = 0
                
            param_values.append(param_index)  # Use numeric index for plotting
            param2_values.append(0)  # No secondary parameter for Phase 1
            
            # Create display name for Phase 1
            if len(emb_model) > 10:
                short_name = emb_model[:8] + '...'
            else:
                short_name = emb_model
                
            display_name = f"{team_type}-{short_name}"
            display_names.append(display_name)
            
        elif phase == 2:  # Chunk Size and Overlap Comparison
            # Get chunk size and overlap (primary and secondary parameters for Phase 2)
            chunk_size = config.get('chunk_size', 0)
            chunk_overlap = config.get('chunk_overlap', 0)
            
            param_values.append(chunk_size)  
            param2_values.append(chunk_overlap)
            
            # Create display name for Phase 2
            display_name = f"CS={chunk_size}, CO={chunk_overlap}"
            display_names.append(display_name)
            
        elif phase == 3:  # Retriever Method Comparison
            # Get retriever type (primary parameter for Phase 3)
            retriever_type = config.get('retriever_type', 'unknown')
            
            # For regression analysis, we'll use a numeric index for the retriever type
            if retriever_type == 'similarity':
                param_index = 1
            elif retriever_type == 'mmr':
                param_index = 2
            elif 'similarity_score_threshold' in retriever_type:
                param_index = 3
            else:
                param_index = 0
                
            param_values.append(param_index)  # Use numeric index for plotting
            param2_values.append(0)  # No secondary parameter for Phase 3
            
            # Create display name for Phase 3
            display_name = f"Retriever: {retriever_type}"
            display_names.append(display_name)
            
        else:  # Fallback for unknown phase
            param_values.append(0)
            param2_values.append(0)
            display_names.append(f"Unknown-{i+1}")
    
    # Debug the extracted parameters
    print("\nExtracted parameters for plotting:")
    for i, (name, param, display) in enumerate(zip(selected_names, param_values, display_names)):
        print(f"Experiment {i+1}: ID={name}, Param={param}, Display={display}")
        
    # Sort based on the primary parameter type
    if phase == 1 or phase == 3:
        # For phase 1 and 3, we're dealing with string values (embedding models/retrieval methods)
        # Sort alphabetically
        sort_indices = np.argsort(display_names)
    else:
        # For phase 2, sort by chunk size
        sort_indices = np.argsort(param_values)
    
    # Sort all arrays using the sort indices
    sorted_param_values = np.array(param_values)[sort_indices]
    sorted_param2_values = np.array(param2_values)[sort_indices]
    sorted_means = np.array(means)[sort_indices]
    sorted_errors = np.array(standard_errors)[sort_indices]
    sorted_display_names = np.array(display_names)[sort_indices]
    
    # Create letter labels for x-axis (A, B, C, etc.)
    import string
    letter_labels = list(string.ascii_uppercase[:len(sorted_means)])
    
    # Create a mapping between letter labels and experiment configs for the legend
    exp_legend_entries = []
    for i, (letter, config) in enumerate(zip(letter_labels, sorted_display_names)):
        exp_legend_entries.append(f"{letter}: {config}")
    
    # Create color mapping based on the phase
    if phase == 2:
        # For phase 2, we use a color gradient for chunk sizes
        if len(set(sorted_param_values)) > 1:
            norm = plt.Normalize(min(sorted_param_values), max(sorted_param_values))
            colors = plt.cm.viridis(norm(sorted_param_values))
        else:
            # If all chunk sizes are the same, use a default color
            colors = ['steelblue'] * len(sorted_means)
    else:
        # For phase 1 and 3, use a categorical color map
        # Get unique parameter values
        unique_params = list(dict.fromkeys(sorted_param_values))
        
        # Create a mapping from parameter values to colors
        color_map = {}
        # Use pyplot's get_cmap for newer matplotlib versions
        try:
            # For newer matplotlib versions
            cmap = plt.colormaps['tab10']
        except:
            # Fallback for older versions
            cmap = plt.cm.get_cmap('tab10')
            
        # Create evenly spaced colors
        for i, param in enumerate(unique_params):
            color_val = i / max(1, len(unique_params) - 1)  # Normalized between 0 and 1
            color_map[param] = cmap(color_val)
        
        # Create color array
        colors = [color_map[param] for param in sorted_param_values]

    # Create a new figure with a gridspec layout
    # This will create a clean 2-column layout
    fig = plt.figure(figsize=(16, 10))
    
    # Create a GridSpec layout with 1 row and 2 columns
    # The left column (for the main plot) is 70% of the width
    # The right column (for legends) is 30% of the width
    gs = fig.add_gridspec(1, 2, width_ratios=[0.7, 0.3])
    
    # Create the main plot axes
    ax_main = fig.add_subplot(gs[0, 0])
    
    # Plot the bars
    x = np.arange(len(sorted_means))
    bars = ax_main.bar(x, sorted_means, yerr=sorted_errors, align='center',
                      alpha=0.7, capsize=5, color=colors, width=0.7)
    
    # Add data labels above each bar
    for i, v in enumerate(sorted_means):
        ax_main.text(i, v + sorted_errors[i] + 0.01, letter_labels[i],
                    ha='center', va='bottom', fontweight='bold')
    
    # Set simple letter labels on x-axis
    ax_main.set_xticks(x)
    ax_main.set_xticklabels(letter_labels, fontsize=12, fontweight='bold')
    
    # Set labels and title for main plot
    ax_main.set_title(f'Comparison of {metric} across experiments', fontsize=14)
    ax_main.set_ylabel(f'{metric} score', fontsize=12)
    ax_main.set_xlabel('Experiment Configuration (see legend →)', fontsize=12)
    
    # Add grid for better readability
    ax_main.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Create a parameter-specific legend based on the phase
    param_handles = []
    param_labels = []
    
    # Get phase-specific legend title and labels
    if phase == 1:
        legend_title = "Embedding Models"
        param_name = "Embedding Model"
    elif phase == 2:
        legend_title = "Chunk Sizes"
        param_name = "Chunk Size"
    elif phase == 3:
        legend_title = "Retrieval Methods"
        param_name = "Retrieval Method"
    else:
        legend_title = "Parameters"
        param_name = "Parameter"
    
    # Create the parameter legend entries
    unique_params = sorted(set(sorted_param_values))
    
    for param in unique_params:
        if phase == 2:
            # For phase 2, use the gradient color for chunk sizes
            if len(unique_params) > 1:
                color = plt.cm.viridis(norm(param))
            else:
                color = 'steelblue'
            label = f"{param_name}: {param}"
        else:
            # For phase 1 and 3, use the categorical colors
            color = color_map[param]
            # For embedding models, create shorter label if needed
            if phase == 1 and isinstance(param, str) and len(param) > 15:
                label = f"{param_name}: {param[:12]}..."
            else:
                label = f"{param_name}: {param}"
        
        # Create colored patch for the legend
        patch = plt.Rectangle((0, 0), 1, 1, facecolor=color, alpha=0.7)
        param_handles.append(patch)
        param_labels.append(label)
    
    # Add the parameter legend to the top-left corner of the main plot
    param_legend = ax_main.legend(param_handles, param_labels, 
                               loc='upper left', title=legend_title, 
                               framealpha=0.7)

    # Regression analysis for all phases
    has_regression = False
    
    # For Phase 1 and 3, we use numeric indices for categorical variables
    # For Phase 2, we use actual numeric values (chunk size and overlap)
    if len(means) > 1 and len(set(sorted_param_values)) > 1:
        # Prepare data for regression
        if phase == 2:
            # Use both primary and secondary parameters for Phase 2
            X = np.array([sorted_param_values, sorted_param2_values]).T
        else:
            # For Phase 1 and 3, just use the primary parameter
            X = np.array(sorted_param_values).reshape(-1, 1)
            
        # Add constant term for intercept
        X = np.column_stack((np.ones(len(X)), X))
        y = np.array(sorted_means)
        
        # Fit regression model
        try:
            # Ordinary Least Squares regression
            beta, resid, _, _ = np.linalg.lstsq(X, y, rcond=None)
            has_regression = True
            
            # Calculate R-squared
            y_mean = np.mean(y)
            ss_total = np.sum((y - y_mean) ** 2)
            ss_residual = np.sum(resid) if resid.size > 0 else np.sum((y - np.dot(X, beta)) ** 2)
            r_squared = 1 - (ss_residual / ss_total) if ss_total != 0 else 0
            
            # Model coefficients - different handling based on phase
            if phase == 2:
                # For Phase 2, we have intercept, chunk_size coef, and chunk_overlap coef
                if len(beta) >= 3:
                    intercept, cs_coef, co_coef = beta
                else:
                    # Handle case with fewer coefficients
                    intercept = beta[0]
                    cs_coef = beta[1] if len(beta) > 1 else 0
                    co_coef = 0
            else:
                # For Phase 1 and 3, we just have intercept and one coefficient
                if len(beta) >= 2:
                    intercept, cs_coef = beta
                    co_coef = 0  # No secondary parameter
                else:
                    # Fallback if beta has unexpected shape
                    intercept = beta[0]
                    cs_coef = 0
                    co_coef = 0
            
            # Plot regression line
            ax_reg = ax_main.twinx().twiny()  # Create twin axes for regression
            
            # Plot regression points directly on top of the bars
            scatter_x = sorted_param_values
            
            # Calculate predicted values for each data point
            if phase == 2:
                # For Phase 2, use both coefficients
                scatter_y = [intercept + cs_coef * p1 + co_coef * p2 
                          for p1, p2 in zip(sorted_param_values, sorted_param2_values)]
            else:
                # For Phase 1 and 3, just use the primary coefficient
                scatter_y = [intercept + cs_coef * p1 for p1 in sorted_param_values]
            
            # Plot regression line as a function of the primary parameter
            x_min = min(sorted_param_values)
            x_max = max(sorted_param_values)
            if x_min == x_max:
                # Handle case where all values are the same
                x_reg = np.array([x_min - 0.5, x_min, x_min + 0.5])
            else:
                # Generate evenly spaced x values
                x_reg = np.linspace(x_min * 0.9, x_max * 1.1, 100)
            
            # Calculate y values for the regression line
            if phase == 2:
                # For Phase 2, use average chunk overlap (secondary parameter)
                mean_overlap = np.mean(sorted_param2_values)
                y_reg = intercept + cs_coef * x_reg + co_coef * mean_overlap
            else:
                # For Phase 1 and 3, just use the primary parameter
                y_reg = intercept + cs_coef * x_reg
            
            # Add regression line
            ax_reg.plot(x_reg, y_reg, 'r-', linewidth=3)
            ax_reg.scatter(scatter_x, scatter_y, color='red', marker='x', s=60, zorder=5)
            
            # We'll add both legends at once later
            
            # Set limits and hide the ticks of the regression axis
            ax_reg.set_xlim(min(sorted_param_values) * 0.9, max(sorted_param_values) * 1.1)
            ax_reg.set_xticks([])
            ax_reg.set_yticks([])
            
            # Match y-axis limits
            y_min, y_max = ax_main.get_ylim()
            ax_reg.set_ylim(y_min, y_max)
            
            # Add a small legend for the regression line
            from matplotlib.lines import Line2D
            reg_line_legend = ax_main.legend([Line2D([0], [0], color='r', lw=2, marker='x')],
                               ['Regression Line'],
                               loc='upper right', framealpha=0.7)
            
            # Make sure the parameter legend is also visible
            ax_main.add_artist(param_legend)
            
            # Save regression data with phase-appropriate parameter names
            if phase == 1:
                primary_name = 'embedding_model'
                secondary_name = 'N/A'
            elif phase == 2:
                primary_name = 'chunk_size'
                secondary_name = 'chunk_overlap'
            elif phase == 3:
                primary_name = 'retriever_type'
                secondary_name = 'N/A'
            else:
                primary_name = 'parameter'
                secondary_name = 'parameter2'
            
            regression_data = {
                'metric': metric,
                'intercept': float(intercept),
                'primary_param_coefficient': float(cs_coef),
                'secondary_param_coefficient': float(co_coef),
                'primary_param_name': primary_name,
                'secondary_param_name': secondary_name,
                'r_squared': float(r_squared)
            }
            
            # Update regression results dictionary
            if 'regression_results' not in globals():
                global regression_results
                regression_results = []
            
            regression_results.append(regression_data)
            
        except np.linalg.LinAlgError:
            print(f"Could not perform regression for {metric} - singular matrix")
            has_regression = False

    # ---- Right column for legends and information ----
    # Create a subplot for the right side
    ax_right = fig.add_subplot(gs[0, 1])
    # Make the right subplot invisible (we'll add custom content)
    ax_right.axis('off')
    
    # Create a simpler right panel with just experiment configurations
    # and minimal regression info if needed
    right_gs = gs[0, 1].subgridspec(2, 1, height_ratios=[0.3, 0.7])
    
    # Experiment Configuration Legend (takes most of the space)
    ax_legend = fig.add_subplot(right_gs[1, 0])
    ax_legend.axis('off')
    ax_legend.set_title("Experiment Configurations", fontsize=12, fontweight='bold')
    
    # Create legend entries with colored patches - give more vertical space
    max_entries_per_column = min(8, len(letter_labels))
    columns = 1
    
    # Calculate how many entries to show per column
    if len(letter_labels) > max_entries_per_column:
        columns = 2
        entries_per_column = (len(letter_labels) + 1) // 2
    else:
        entries_per_column = len(letter_labels)
    
    # Draw the legend entries
    for i, (label, entry) in enumerate(zip(letter_labels, exp_legend_entries)):
        # Determine which column this entry belongs in
        col = i // entries_per_column
        row = i % entries_per_column
        
        # Calculate position based on column and row
        x_start = 0.05 + (col * 0.5)  # 0.5 is the width of a column
        y_pos = 0.95 - (row * 1.0 / entries_per_column)
        
        # Add colored rectangle
        ax_legend.add_patch(plt.Rectangle((x_start, y_pos), 0.05, 0.04, 
                                      facecolor=colors[i], alpha=0.7))
        # Add text (with more space and larger font)
        ax_legend.text(x_start + 0.07, y_pos + 0.02, entry, fontsize=11, 
                     verticalalignment='center')
    
    # Add minimal regression info if available
    if has_regression:
        ax_regstats = fig.add_subplot(right_gs[0, 0])
        ax_regstats.axis('off')
        
        # Phase-specific regression info text
        if r_squared > 0.3:  # Only show details if regression is somewhat meaningful
            if phase == 1:
                info_text = (f"Regression R² = {r_squared:.2f}\n"
                           f"Embedding Model coefficient: {cs_coef:.4f}")
            elif phase == 2:
                info_text = (f"Regression R² = {r_squared:.2f}\n"
                           f"Chunk Size coefficient: {cs_coef:.4f}\n"
                           f"Chunk Overlap coefficient: {co_coef:.4f}")
            elif phase == 3:
                info_text = (f"Regression R² = {r_squared:.2f}\n"
                           f"Retriever Type coefficient: {cs_coef:.4f}")
            else:
                info_text = (f"Regression R² = {r_squared:.2f}\n"
                           f"Parameter coefficient: {cs_coef:.4f}")
        else:
            info_text = f"Regression R² = {r_squared:.2f}\nWeak relationship"
            
        ax_regstats.text(0.5, 0.5, info_text, 
                      ha='center', va='center', fontsize=11,
                      bbox=dict(facecolor='white', alpha=0.8))
        
    # Adjust layout
    fig.tight_layout()
    
    # Create analysis_output directory if it doesn't exist
    os.makedirs(os.path.join(results_path, 'analysis_output'), exist_ok=True)
    
    # Save with higher DPI for better quality
    plt.savefig(os.path.join(results_path, f'analysis_output/{metric}_comparison.png'), dpi=150)
    plt.close()

# Create metric comparison plots (moved to after initializing regression_results)

# Calculate effect sizes (Cohen's d) for significant differences
def cohens_d_from_stats(mean1, std1, n1, mean2, std2, n2):
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
    # Cohen's d
    d = abs(mean1 - mean2) / pooled_std
    return d

# Add effect sizes to the results
df_results['effect_size'] = 0.0
for idx, row in df_results.iterrows():
    if row['significant']:
        metric = row['metric']
        try:
            exp1_idx = next(i for i, exp in enumerate(experiments) if exp['id'] == row['exp1'])
            exp2_idx = next(i for i, exp in enumerate(experiments) if exp['id'] == row['exp2'])
            
            # Make sure both experiments have this metric
            if metric in experiments[exp1_idx]['metrics'] and metric in experiments[exp2_idx]['metrics']:
                mean1 = experiments[exp1_idx]['metrics'][metric]['mean']
                std1 = experiments[exp1_idx]['metrics'][metric]['std']
                count1 = experiments[exp1_idx]['metrics'][metric]['count']
                
                mean2 = experiments[exp2_idx]['metrics'][metric]['mean']
                std2 = experiments[exp2_idx]['metrics'][metric]['std']
                count2 = experiments[exp2_idx]['metrics'][metric]['count']
                
                effect_size = cohens_d_from_stats(mean1, std1, count1, mean2, std2, count2)
                df_results.at[idx, 'effect_size'] = float(effect_size)  # Convert to Python float
        except (StopIteration, KeyError):
            # If any error occurs, just keep the default effect_size of 0.0
            pass

# Filter to only significant results and sort by effect size
significant_results = df_results[df_results['significant']].sort_values(by='effect_size', ascending=False)

# Output top significant differences
print("\n=== Top Significant Differences (by Effect Size) ===")
for idx, row in significant_results.head(10).iterrows():
    print(f"Metric: {row['metric']}")
    print(f"  {row['exp1']} vs {row['exp2']}")
    print(f"  p-value: {row['p_value']:.4f}, effect size: {row['effect_size']:.2f}")

# Initialize regression results list
regression_results = []

# Create metric comparison plots
for metric in metrics_of_interest:
    if mean_values[metric]:  # Only plot if we have data for this metric
        plot_metric_comparison(
            metric, 
            exp_names,  # Pass all exp_names, our function will select the right ones
            mean_values[metric], 
            se_values[metric]
        )

# Save regression results to file
print("\n=== Regression Analysis Results ===")
for result in regression_results:
    metric = result['metric']
    r_squared = result['r_squared']
    primary_coef = result['primary_param_coefficient']
    secondary_coef = result['secondary_param_coefficient']
    primary_name = result['primary_param_name']
    secondary_name = result['secondary_param_name']
    print(f"{metric}: R² = {r_squared:.3f}, {primary_name} coef = {primary_coef:.4f}, {secondary_name} coef = {secondary_coef:.4f}")

# Save results to file
with open(os.path.join(results_path, 'analysis_output/statistical_analysis.json'), 'w') as f:
    json.dump({
        'anova_results': anova_results,
        'significant_tests': int(significant_count),
        'total_tests': int(total_tests),
        'significant_percentage': float(significant_count/total_tests),
        'regression_analysis': regression_results
    }, f, indent=2)

# Convert all numpy types to Python types to avoid serialization issues
for col in df_results.columns:
    if df_results[col].dtype.kind in 'iuf':  # integer, unsigned int, or float
        df_results[col] = df_results[col].astype(float)
    elif df_results[col].dtype.kind == 'b':  # boolean
        df_results[col] = df_results[col].astype(bool)

significant_results.to_csv(os.path.join(results_path, 'analysis_output/significant_differences.csv'), index=False)

print(f"\nAnalysis complete. Results saved to {os.path.join(results_path, 'analysis_output')} directory.")