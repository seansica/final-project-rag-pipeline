import json
import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt

# Load the results data
with open('results/phase1_ragas_20250411_174100/results.json', 'r') as f:
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
        mean = exp['metrics'][metric_name]['mean']
        std = exp['metrics'][metric_name]['std']
        count = exp['metrics'][metric_name]['count']
        
        # Reconstruct approximate data points using mean and std
        # This is a rough approximation since we don't have the raw data
        # Assuming normal distribution centered at mean with given std
        simulated_data = np.random.normal(mean, std, count)
        groups.append(simulated_data)
    
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

for exp in experiments:
    exp_names.append(exp['id'].split('-')[-6:])  # Simplify names for display
    for metric in metrics_of_interest:
        mean_values[metric].append(exp['metrics'][metric]['mean'])
        # Standard error = std / sqrt(n)
        se = exp['metrics'][metric]['std'] / np.sqrt(exp['metrics'][metric]['count'])
        se_values[metric].append(se)

# Function to create a plot showing mean and standard error for a metric
def plot_metric_comparison(metric, exp_names, means, standard_errors):
    plt.figure(figsize=(12, 6))
    x = np.arange(len(exp_names))
    plt.bar(x, means, yerr=standard_errors, align='center', alpha=0.7, capsize=5)
    plt.xticks(x, exp_names, rotation=45, ha='right')
    plt.title(f'Comparison of {metric} across experiments')
    plt.ylabel(f'{metric} score')
    plt.tight_layout()
    plt.savefig(f'results/phase1_ragas_20250411_174100/analysis_output/{metric}_comparison.png')
    plt.close()

# Create metric comparison plots
for metric in metrics_of_interest:
    plot_metric_comparison(
        metric, 
        [name[-2:] for name in exp_names],  # Further simplify names 
        mean_values[metric], 
        se_values[metric]
    )

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
        exp1_idx = next(i for i, exp in enumerate(experiments) if exp['id'] == row['exp1'])
        exp2_idx = next(i for i, exp in enumerate(experiments) if exp['id'] == row['exp2'])
        
        mean1 = experiments[exp1_idx]['metrics'][metric]['mean']
        std1 = experiments[exp1_idx]['metrics'][metric]['std']
        count1 = experiments[exp1_idx]['metrics'][metric]['count']
        
        mean2 = experiments[exp2_idx]['metrics'][metric]['mean']
        std2 = experiments[exp2_idx]['metrics'][metric]['std']
        count2 = experiments[exp2_idx]['metrics'][metric]['count']
        
        effect_size = cohens_d_from_stats(mean1, std1, count1, mean2, std2, count2)
        df_results.at[idx, 'effect_size'] = float(effect_size)  # Convert to Python float

# Filter to only significant results and sort by effect size
significant_results = df_results[df_results['significant']].sort_values(by='effect_size', ascending=False)

# Output top significant differences
print("\n=== Top Significant Differences (by Effect Size) ===")
for idx, row in significant_results.head(10).iterrows():
    print(f"Metric: {row['metric']}")
    print(f"  {row['exp1']} vs {row['exp2']}")
    print(f"  p-value: {row['p_value']:.4f}, effect size: {row['effect_size']:.2f}")

# Save results to file
with open('results/phase1_ragas_20250411_174100/analysis_output/statistical_analysis.json', 'w') as f:
    json.dump({
        'anova_results': anova_results,
        'significant_tests': int(significant_count),
        'total_tests': int(total_tests),
        'significant_percentage': float(significant_count/total_tests)
    }, f, indent=2)

# Convert all numpy types to Python types to avoid serialization issues
for col in df_results.columns:
    if df_results[col].dtype.kind in 'iuf':  # integer, unsigned int, or float
        df_results[col] = df_results[col].astype(float)
    elif df_results[col].dtype.kind == 'b':  # boolean
        df_results[col] = df_results[col].astype(bool)

significant_results.to_csv('results/phase1_ragas_20250411_174100/analysis_output/significant_differences.csv', index=False)

print("\nAnalysis complete. Results saved to analysis_output directory.")