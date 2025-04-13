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
args = parser.parse_args()

results_path = args.results_path

# Load the results data
with open(os.path.join(results_path, 'results.json'), 'r') as f:
    results = json.load(f)

# Extract metrics into a structured format
experiments = []
metrics_of_interest = [
    'correctness', 'groundedness', 'relevance', 'retrieval_relevance',
    'ragas_answer_accuracy', 'ragas_context_relevance', 'ragas_faithfulness',
    'ragas_response_relevancy', 'deepeval_faithfulness', 'deepeval_geval',
    'bertscore_evaluator'
]

for experiment in results:
    exp_id = experiment['experiment_id']
    metrics = {}
    
    for metric in metrics_of_interest:
        if metric in experiment['metrics']['feedback']:
            metrics[metric] = {
                'mean': experiment['metrics']['feedback'][metric]['mean'],
                'std': experiment['metrics']['feedback'][metric]['std'],
                'count': experiment['metrics']['feedback'][metric]['count']
            }
    
    experiments.append({
        'id': exp_id,
        'metrics': metrics
    })

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
    # Extract configuration from experiment ID
    exp_id_parts = exp['id'].split('-')
    # Extract relevant experiment parameters for display
    experiment_config = None
    # Look for results file to extract full config
    if 'results.json' in os.listdir(results_path):
        with open(os.path.join(results_path, 'results.json'), 'r') as f:
            results_data = json.load(f)
            for result in results_data:
                if result['experiment_id'] == exp['id'] and 'config' in result:
                    experiment_config = result['config']
                    break
    
    if experiment_config:
        # Use the actual config values from the results file
        chunk_size = experiment_config.get('chunk_size', 'N/A')
        chunk_overlap = experiment_config.get('chunk_overlap', 'N/A')
        display_name = f"cs{chunk_size}-co{chunk_overlap}"
    else:
        # Fallback to parsing from ID if config not found
        display_name = '-'.join(exp_id_parts[-6:])
    
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
    
    # Extract chunk size and chunk overlap for regression analysis
    chunk_sizes = []
    chunk_overlaps = []
    display_names = []  # Better formatted names for display
    
    for name in selected_names:
        # Extract values from name format like "cs512-co50"
        try:
            cs_part = name.split('-')[0]
            co_part = name.split('-')[1]
            
            # Extract numeric values
            chunk_size = int(cs_part.replace('cs', ''))
            chunk_overlap = int(co_part.replace('co', ''))
            
            chunk_sizes.append(chunk_size)
            chunk_overlaps.append(chunk_overlap)
            
            # Create cleaner display name
            display_names.append(f"CS={chunk_size}, CO={chunk_overlap}")
        except (ValueError, IndexError):
            # If we can't parse the values, use original name
            chunk_sizes.append(0)
            chunk_overlaps.append(0)
            display_names.append(name)
    
    # Sort everything by chunk size for better visualization
    sort_indices = np.argsort(chunk_sizes)
    sorted_chunk_sizes = np.array(chunk_sizes)[sort_indices]
    sorted_chunk_overlaps = np.array(chunk_overlaps)[sort_indices]
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
    
    # Use color gradient based on chunk size for better visual correlation
    norm = plt.Normalize(min(sorted_chunk_sizes), max(sorted_chunk_sizes))
    colors = plt.cm.viridis(norm(sorted_chunk_sizes))

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
    
    # Create a discrete legend for chunk sizes instead of a continuous colorbar
    # Find unique chunk sizes
    unique_chunk_sizes = sorted(set(sorted_chunk_sizes))
    
    # Create a custom legend for chunk sizes
    chunk_handles = []
    chunk_labels = []
    
    for cs in unique_chunk_sizes:
        # Get the color for this chunk size from our normalized colormap
        color = plt.cm.viridis(norm(cs))
        # Create a colored patch for the legend
        patch = plt.Rectangle((0, 0), 1, 1, facecolor=color, alpha=0.7)
        chunk_handles.append(patch)
        chunk_labels.append(f"Chunk Size: {cs}")
    
    # Add the chunk size legend to the top-left corner of the main plot
    # chunk_legend = ax_main.legend(chunk_handles, chunk_labels, 
    #                            loc='upper left', title="Chunk Sizes", 
    #                            framealpha=0.7)

    # Regression analysis
    has_regression = False
    if len(means) > 1 and len(set(chunk_sizes)) > 1:
        # Prepare data for regression
        X = np.array([chunk_sizes, chunk_overlaps]).T
        # Add constant term for intercept
        X = np.column_stack((np.ones(len(X)), X))
        y = np.array(means)
        
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
            
            # Model coefficients
            intercept, cs_coef, co_coef = beta
            
            # Plot regression line
            ax_reg = ax_main.twinx().twiny()  # Create twin axes for regression
            
            # Plot regression points directly on top of the bars
            scatter_x = sorted_chunk_sizes
            scatter_y = [intercept + cs_coef * cs + co_coef * co 
                       for cs, co in zip(sorted_chunk_sizes, sorted_chunk_overlaps)]
            
            # Plot regression line as a function of chunk size
            x_reg = np.linspace(min(sorted_chunk_sizes) * 0.9, max(sorted_chunk_sizes) * 1.1, 100)
            mean_overlap = np.mean(sorted_chunk_overlaps)
            y_reg = intercept + cs_coef * x_reg + co_coef * mean_overlap
            
            # Add regression line
            ax_reg.plot(x_reg, y_reg, 'r-', linewidth=3)
            ax_reg.scatter(scatter_x, scatter_y, color='red', marker='x', s=60, zorder=5)
            
            # We'll add both legends at once later
            
            # Set limits and hide the ticks of the regression axis
            ax_reg.set_xlim(min(sorted_chunk_sizes) * 0.9, max(sorted_chunk_sizes) * 1.1)
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
            
            # Make sure both legends are visible
            # ax_main.add_artist(chunk_legend)
            
            # Save regression data
            regression_data = {
                'metric': metric,
                'intercept': float(intercept),
                'chunk_size_coefficient': float(cs_coef),
                'chunk_overlap_coefficient': float(co_coef),
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
        
        # Simple clean text without any overlapping elements
        if r_squared > 0.5:  # Only show if regression is meaningful
            info_text = (f"Regression R² = {r_squared:.2f}\n"
                        f"CS coefficient: {cs_coef:.4f}\n"
                        f"CO coefficient: {co_coef:.4f}")
        else:
            info_text = "Regression analysis shows\nweak relationship (R² < 0.5)"
            
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
    cs_coef = result['chunk_size_coefficient']
    co_coef = result['chunk_overlap_coefficient']
    print(f"{metric}: R² = {r_squared:.3f}, CS coef = {cs_coef:.4f}, CO coef = {co_coef:.4f}")

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