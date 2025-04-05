#!/usr/bin/env python
"""
Script to determine the top performing RAG system in a given results.json file.
Reports the top performer for each metric category.
"""
import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional


def load_results(results_file: str) -> List[Dict[str, Any]]:
    """
    Load results from a results.json file.
    
    Args:
        results_file: Path to the results.json file
        
    Returns:
        List of experiment results
    """
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Filter out unsuccessful experiments
    successful_results = [r for r in results if r.get('success', False)]
    
    if not successful_results:
        raise ValueError("No successful experiments found in the results file")
    
    return successful_results


def analyze_metrics(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Extract and analyze metrics from the results.
    
    Args:
        results: List of experiment results
        
    Returns:
        DataFrame with metrics for each experiment
    """
    rows = []
    
    for result in results:
        # Basic experiment info
        row = {
            'experiment_id': result.get('experiment_id', ''),
            'rag_type': result.get('config', {}).get('rag_type', ''),
            'team_type': result.get('config', {}).get('team_type', ''),
            'embedding_model': result.get('config', {}).get('embedding_model', ''),
            'chunk_size': result.get('config', {}).get('chunk_size', 0),
            'chunk_overlap': result.get('config', {}).get('chunk_overlap', 0),
            'top_k': result.get('config', {}).get('top_k', 0),
            'retriever_type': result.get('config', {}).get('retriever_type', ''),
            'evaluation_type': result.get('evaluation_type', '')
        }
        
        # Extract feedback metrics
        metrics = result.get('metrics', {})
        feedback = metrics.get('feedback', {})
        
        for metric_name, metric_data in feedback.items():
            # For each metric, extract the mean value
            mean_value = metric_data.get('mean', 0.0)
            row[f'{metric_name}_mean'] = mean_value
        
        # Extract text comparison metrics
        text_comparison = metrics.get('text_comparison', {})
        
        # For character length and word count, compute how close the output is to the reference
        # We want ratio to be as close to 1.0 as possible
        for metric_name, metric_data in text_comparison.items():
            mean_output = metric_data.get('mean_output', 0.0)
            mean_reference = metric_data.get('mean_reference', 0.0)
            mean_ratio = metric_data.get('mean_ratio', 0.0)
            
            # Store original values
            row[f'{metric_name}_output'] = mean_output
            row[f'{metric_name}_reference'] = mean_reference
            row[f'{metric_name}_ratio'] = mean_ratio
            
            # Calculate closeness score (how close ratio is to 1.0, higher is better)
            closeness = 1.0 / (abs(mean_ratio - 1.0) + 0.01)  # Add small epsilon to avoid division by zero
            row[f'{metric_name}_closeness'] = closeness
        
        rows.append(row)
    
    # Convert to DataFrame
    df = pd.DataFrame(rows)
    
    return df


def find_top_performers(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Determine top performing systems for each metric.
    
    Args:
        df: DataFrame with metrics for each experiment
        
    Returns:
        Dictionary with top performers for each metric
    """
    top_performers = {}
    
    # Get RAGAS feedback metrics (columns ending with _mean)
    feedback_metrics = [col for col in df.columns if col.endswith('_mean')]
    
    # For each feedback metric, find the experiment with the highest score
    for metric in feedback_metrics:
        metric_name = metric.replace('_mean', '')
        
        # Sort by the metric in descending order and get the top row
        top_experiment = df.sort_values(by=metric, ascending=False).iloc[0]
        
        top_performers[metric_name] = {
            'experiment_id': top_experiment['experiment_id'],
            'score': top_experiment[metric],
            'config': {
                'rag_type': top_experiment['rag_type'],
                'team_type': top_experiment['team_type'],
                'embedding_model': top_experiment['embedding_model'],
                'chunk_size': top_experiment['chunk_size'],
                'chunk_overlap': top_experiment['chunk_overlap'],
                'top_k': top_experiment['top_k'],
                'retriever_type': top_experiment['retriever_type']
            }
        }
    
    # Get text comparison closeness metrics
    closeness_metrics = [col for col in df.columns if col.endswith('_closeness')]
    
    # For each closeness metric, find the experiment with the highest score
    for metric in closeness_metrics:
        metric_name = metric.replace('_closeness', '')
        
        # Sort by the metric in descending order and get the top row
        top_experiment = df.sort_values(by=metric, ascending=False).iloc[0]
        
        top_performers[metric_name] = {
            'experiment_id': top_experiment['experiment_id'],
            'original_ratio': top_experiment[f'{metric_name}_ratio'],
            'closeness_score': top_experiment[metric],
            'config': {
                'rag_type': top_experiment['rag_type'],
                'team_type': top_experiment['team_type'],
                'embedding_model': top_experiment['embedding_model'],
                'chunk_size': top_experiment['chunk_size'],
                'chunk_overlap': top_experiment['chunk_overlap'],
                'top_k': top_experiment['top_k'],
                'retriever_type': top_experiment['retriever_type']
            }
        }
    
    # Calculate overall ranking based on average normalized rank across all metrics
    # First, calculate normalized ranks for each metric (0-1 scale, 1 is best)
    rank_columns = []
    
    # For feedback metrics, higher values are better
    for metric in feedback_metrics:
        rank_col = f"{metric.replace('_mean', '')}_rank"
        df[rank_col] = df[metric].rank(method='min', ascending=False) / len(df)
        df[rank_col] = 1 - df[rank_col]  # Invert so 1 is best
        rank_columns.append(rank_col)
    
    # For closeness metrics, higher values are better
    for metric in closeness_metrics:
        rank_col = f"{metric.replace('_closeness', '')}_rank"
        df[rank_col] = df[metric].rank(method='min', ascending=False) / len(df)
        df[rank_col] = 1 - df[rank_col]  # Invert so 1 is best
        rank_columns.append(rank_col)
    
    # Calculate overall score as average of normalized ranks
    df['overall_score'] = df[rank_columns].mean(axis=1)
    
    # Get top performer overall
    top_overall = df.sort_values(by='overall_score', ascending=False).iloc[0]
    
    top_performers['overall'] = {
        'experiment_id': top_overall['experiment_id'],
        'overall_score': top_overall['overall_score'],
        'config': {
            'rag_type': top_overall['rag_type'],
            'team_type': top_overall['team_type'],
            'embedding_model': top_overall['embedding_model'],
            'chunk_size': top_overall['chunk_size'],
            'chunk_overlap': top_overall['chunk_overlap'],
            'top_k': top_overall['top_k'],
            'retriever_type': top_overall['retriever_type']
        }
    }
    
    return top_performers, df


def visualize_results(df: pd.DataFrame, output_dir: Optional[str] = None):
    """
    Create visualizations of the results.
    
    Args:
        df: DataFrame with metrics and rankings
        output_dir: Directory to save visualizations, if None, visualizations won't be saved
    """
    # Create output directory if specified
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    
    try:
        # Get feedback metrics
        feedback_metrics = [col for col in df.columns if col.endswith('_mean')]
        
        if feedback_metrics:
            # Create a heatmap of normalized metric scores
            plt.figure(figsize=(14, 10))
            
            # Extract experiment IDs and metrics for heatmap
            heatmap_data = df.set_index('experiment_id')[feedback_metrics].copy()
            
            # Convert any NumPy data types to Python native types
            for col in heatmap_data.columns:
                heatmap_data[col] = heatmap_data[col].astype(float)
            
            # Rename columns for better display
            heatmap_data = heatmap_data.rename(columns={col: col.replace('_mean', '') for col in feedback_metrics})
            
            # Create heatmap
            sns.heatmap(heatmap_data, annot=True, cmap='viridis', fmt=".3f")
            plt.title('Feedback Metrics Scores by Experiment')
            plt.tight_layout()
            
            if output_dir:
                plt.savefig(output_path / 'feedback_metrics_heatmap.png', dpi=300, bbox_inches='tight')
                plt.close()
            
        # Create bar chart of overall scores
        if 'overall_score' in df.columns:
            plt.figure(figsize=(12, 8))
            
            # Sort by overall score
            top_experiments = df.sort_values(by='overall_score', ascending=False).copy()
            
            # Ensure data types are compatible with plotting
            top_experiments['overall_score'] = top_experiments['overall_score'].astype(float)
            
            # Create bar chart
            sns.barplot(x='overall_score', y='experiment_id', data=top_experiments)
            plt.title('Overall Ranking of Experiments')
            plt.tight_layout()
            
            if output_dir:
                plt.savefig(output_path / 'overall_ranking.png', dpi=300, bbox_inches='tight')
                plt.close()
                
        # Create a radar chart for the top performer
        if 'overall_score' in df.columns and '_rank' in ''.join(df.columns):
            top_exp_id = df.sort_values(by='overall_score', ascending=False).iloc[0]['experiment_id']
            top_exp_data = df[df['experiment_id'] == top_exp_id]
            
            # Get normalized scores (ranks) for radar chart
            radar_metrics = [col for col in df.columns if col.endswith('_rank')]
            
            if radar_metrics:
                radar_values = top_exp_data[radar_metrics].iloc[0].values.astype(float)
                radar_labels = [col.replace('_rank', '') for col in radar_metrics]
                
                # Create radar chart
                fig = plt.figure(figsize=(10, 10))
                ax = fig.add_subplot(111, polar=True)
                
                # Plot the radar chart
                angles = np.linspace(0, 2*np.pi, len(radar_labels), endpoint=False)
                radar_values = np.concatenate((radar_values, [radar_values[0]]))  # Close the loop
                angles = np.concatenate((angles, [angles[0]]))  # Close the loop
                radar_labels = np.concatenate((radar_labels, [radar_labels[0]]))  # Close the loop
                
                ax.plot(angles, radar_values, 'o-', linewidth=2)
                ax.fill(angles, radar_values, alpha=0.25)
                ax.set_thetagrids(angles * 180/np.pi, radar_labels)
                ax.set_ylim(0, 1)
                plt.title(f'Performance Profile of Top Experiment: {top_exp_id}')
                
                if output_dir:
                    plt.savefig(output_path / 'top_performer_radar.png', dpi=300, bbox_inches='tight')
                    plt.close()
    
    except Exception as e:
        print(f"Error generating visualizations: {e}")
        if output_dir:
            # Write the error to a log file
            with open(output_path / 'visualization_error.log', 'w') as f:
                f.write(f"Error generating visualizations: {e}\n")


def print_top_performers(top_performers: Dict[str, Any]):
    """
    Print the top performers in a human-readable format.
    
    Args:
        top_performers: Dictionary with top performers for each metric
    """
    print("\n" + "="*80)
    print("TOP PERFORMING RAG SYSTEMS BY METRIC")
    print("="*80)
    
    # Print RAGAS metrics
    print("\nRAGAS METRICS (higher is better):")
    print("-" * 50)
    
    ragas_metrics = [k for k in top_performers.keys() if k.startswith('ragas_')]
    for metric in ragas_metrics:
        performer = top_performers[metric]
        config = performer['config']
        
        print(f"\n{metric.upper()}:")
        print(f"  Score: {performer['score']:.4f}")
        print(f"  Experiment: {performer['experiment_id']}")
        print(f"  Configuration:")
        print(f"    - RAG Type: {config['rag_type']}")
        print(f"    - Team Type: {config['team_type']}")
        print(f"    - Embedding Model: {config['embedding_model']}")
        print(f"    - Chunk Size: {config['chunk_size']}")
        print(f"    - Chunk Overlap: {config['chunk_overlap']}")
        print(f"    - Top K: {config['top_k']}")
        print(f"    - Retriever Type: {config['retriever_type']}")
    
    # Print text comparison metrics
    print("\nTEXT COMPARISON METRICS (closer to reference length is better):")
    print("-" * 50)
    
    text_metrics = [k for k in top_performers.keys() if k in ['character_length', 'word_count']]
    for metric in text_metrics:
        performer = top_performers[metric]
        config = performer['config']
        
        print(f"\n{metric.upper()}:")
        print(f"  Original Ratio (output/reference): {performer['original_ratio']:.4f}")
        print(f"  Closeness Score: {performer['closeness_score']:.4f}")
        print(f"  Experiment: {performer['experiment_id']}")
        print(f"  Configuration:")
        print(f"    - RAG Type: {config['rag_type']}")
        print(f"    - Team Type: {config['team_type']}")
        print(f"    - Embedding Model: {config['embedding_model']}")
        print(f"    - Chunk Size: {config['chunk_size']}")
        print(f"    - Chunk Overlap: {config['chunk_overlap']}")
        print(f"    - Top K: {config['top_k']}")
        print(f"    - Retriever Type: {config['retriever_type']}")
    
    # Print overall top performer
    print("\nOVERALL TOP PERFORMER (based on average ranking across all metrics):")
    print("-" * 50)
    
    performer = top_performers['overall']
    config = performer['config']
    
    print(f"  Overall Score: {performer['overall_score']:.4f}")
    print(f"  Experiment: {performer['experiment_id']}")
    print(f"  Configuration:")
    print(f"    - RAG Type: {config['rag_type']}")
    print(f"    - Team Type: {config['team_type']}")
    print(f"    - Embedding Model: {config['embedding_model']}")
    print(f"    - Chunk Size: {config['chunk_size']}")
    print(f"    - Chunk Overlap: {config['chunk_overlap']}")
    print(f"    - Top K: {config['top_k']}")
    print(f"    - Retriever Type: {config['retriever_type']}")


def save_analysis(top_performers: Dict[str, Any], df: pd.DataFrame, output_file: str):
    """
    Save the analysis results to a JSON file.
    
    Args:
        top_performers: Dictionary with top performers for each metric
        df: DataFrame with all metrics
        output_file: Path to the output file
    """
    # Custom JSON serializer to handle NumPy types
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)
    
    # Process top_performers to convert NumPy types to Python types
    processed_top_performers = {}
    for metric, data in top_performers.items():
        if isinstance(data, dict):
            processed_data = {}
            for k, v in data.items():
                if isinstance(v, (np.integer, np.floating)):
                    processed_data[k] = float(v) if isinstance(v, np.floating) else int(v)
                elif isinstance(v, np.ndarray):
                    processed_data[k] = v.tolist()
                elif isinstance(v, dict):
                    # Handle nested dictionaries (like config)
                    processed_nested = {}
                    for nk, nv in v.items():
                        if isinstance(nv, (np.integer, np.floating)):
                            processed_nested[nk] = float(nv) if isinstance(nv, np.floating) else int(nv)
                        else:
                            processed_nested[nk] = nv
                    processed_data[k] = processed_nested
                else:
                    processed_data[k] = v
            processed_top_performers[metric] = processed_data
        else:
            processed_top_performers[metric] = data
    
    # Convert DataFrame to records (list of dicts) and ensure all values are JSON serializable
    df_dict = []
    for record in df.to_dict(orient='records'):
        processed_record = {}
        for k, v in record.items():
            if isinstance(v, (np.integer, np.floating)):
                processed_record[k] = float(v) if isinstance(v, np.floating) else int(v)
            elif isinstance(v, np.ndarray):
                processed_record[k] = v.tolist()
            else:
                processed_record[k] = v
        df_dict.append(processed_record)
    
    # Create analysis results
    analysis = {
        'top_performers': processed_top_performers,
        'all_metrics': df_dict
    }
    
    # Save to file
    with open(output_file, 'w') as f:
        json.dump(analysis, f, indent=2, cls=NumpyEncoder)


def main():
    parser = argparse.ArgumentParser(description="Analyze results and determine top performing RAG system")
    parser.add_argument("results_file", type=str, help="Path to the results.json file")
    parser.add_argument("--output_dir", type=str, help="Directory to save analysis outputs")
    parser.add_argument("--visualize", action="store_true", help="Create visualizations")
    
    args = parser.parse_args()
    
    # Load results
    results = load_results(args.results_file)
    
    # Analyze metrics
    df = analyze_metrics(results)
    
    # Find top performers
    top_performers, df_with_ranks = find_top_performers(df)
    
    # Print top performers
    print_top_performers(top_performers)
    
    # Save analysis if output directory is specified
    if args.output_dir:
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save analysis
        save_analysis(top_performers, df_with_ranks, output_dir / 'analysis.json')
        
        # Save metrics DataFrame
        df_with_ranks.to_csv(output_dir / 'all_metrics.csv', index=False)
        
        print(f"\nAnalysis saved to {output_dir}")
    
    # Create visualizations if requested
    if args.visualize:
        visualize_results(df_with_ranks, args.output_dir if args.output_dir else None)
        if args.output_dir:
            print(f"Visualizations saved to {args.output_dir}")


if __name__ == "__main__":
    main()