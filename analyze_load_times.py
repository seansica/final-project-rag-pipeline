#!/usr/bin/env python
"""
Script to analyze load times and performance metrics from RAG system experiments in LangSmith.
Extracts and reports query, generation, and total latency statistics.
"""
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Optional
from langsmith import Client
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables
load_dotenv()

def fetch_experiment_runs(project_name: str, filter_tags: Optional[List[str]] = None):
    """
    Fetch runs from a specific LangSmith project with optional filtering.
    
    Args:
        project_name: The name of the LangSmith project
        filter_tags: Optional list of tags to filter runs
    
    Returns:
        List of run objects from LangSmith
    """
    client = Client()
    
    # Just fetch all runs for the project
    runs = list(client.list_runs(project_name=project_name))
    
    # Manually filter by tags if provided
    if filter_tags and filter_tags[0]:
        filtered_runs = []
        for run in runs:
            # Check if run has tags attribute and it contains any of the specified tags
            if hasattr(run, 'tags') and run.tags:
                if any(tag in run.tags for tag in filter_tags):
                    filtered_runs.append(run)
        runs = filtered_runs
        print(f"Found {len(runs)} runs in project '{project_name}' with matching tags")
    else:
        print(f"Found {len(runs)} runs in project '{project_name}'")
    
    return runs

def analyze_run_performance(runs: List[Any]):
    """
    Analyze performance metrics for a list of LangSmith runs.
    
    Args:
        runs: List of run objects from LangSmith
    
    Returns:
        DataFrame with performance metrics for each run
    """
    performance_data = []
    
    for run in runs:
        try:
            # Get the basic run info
            run_data = {
                'run_id': run.id,
                'experiment_name': getattr(run, 'name', 'Unknown'),
                'start_time': getattr(run, 'start_time', None),
                'end_time': getattr(run, 'end_time', None),
                'status': getattr(run, 'status', 'unknown'),
                'error': getattr(run, 'error', None),
            }
            
            # Calculate latency if start_time and end_time exist
            if run_data['start_time'] and run_data['end_time']:
                run_data['latency'] = (run_data['end_time'] - run_data['start_time']).total_seconds()
            elif hasattr(run, 'latency'):
                run_data['latency'] = run.latency
            else:
                run_data['latency'] = None
            
            # Extract experiment config from metadata if available
            if hasattr(run, 'metadata') and run.metadata:
                for key, value in run.metadata.items():
                    if isinstance(value, (str, int, float, bool)):
                        run_data[f'config_{key}'] = value
                
                # Try to extract nested configs in metadata if they exist
                for top_key in ['llm', 'retriever', 'vectorstore']:
                    if top_key in run.metadata:
                        nested_config = run.metadata[top_key]
                        if isinstance(nested_config, dict):
                            for key, value in nested_config.items():
                                if isinstance(value, (str, int, float, bool)):
                                    run_data[f'config_{top_key}_{key}'] = value
            
            # Add to results
            performance_data.append(run_data)
        except Exception as e:
            print(f"Error processing run {getattr(run, 'id', 'unknown')}: {e}")
    
    # Convert to DataFrame
    df = pd.DataFrame(performance_data)
    
    # Clean up the DataFrame - drop rows with missing latency
    if 'latency' in df.columns:
        df = df.dropna(subset=['latency'])
    
    return df

def calculate_statistics(df):
    """
    Calculate performance statistics from run data.
    
    Args:
        df: DataFrame with run performance data
    
    Returns:
        Dictionary with statistics
    """
    # Make sure we have data
    if len(df) == 0:
        return {
            'total_runs': 0,
            'successful_runs': 0,
            'failed_runs': 0,
            'error': 'No runs with latency data found'
        }
    
    # Filter by status if the column exists
    if 'status' in df.columns:
        successful_runs = df[df['status'] == 'succeeded']
        if len(successful_runs) == 0:
            successful_runs = df  # If no 'succeeded' runs, use all runs
    else:
        successful_runs = df
    
    # Calculate statistics
    latency_stats = {
        'total_runs': len(df),
        'successful_runs': len(successful_runs),
        'failed_runs': len(df) - len(successful_runs),
        'latency': {
            'mean': float(successful_runs['latency'].mean()),
            'median': float(successful_runs['latency'].median()),
            'min': float(successful_runs['latency'].min()),
            'max': float(successful_runs['latency'].max()),
            'std': float(successful_runs['latency'].std()),
            'p90': float(successful_runs['latency'].quantile(0.9)),
            'p95': float(successful_runs['latency'].quantile(0.95)),
            'p99': float(successful_runs['latency'].quantile(0.99)),
        }
    }
    
    # Identify common config columns
    config_cols = [col for col in df.columns if col.startswith('config_')]
    
    # Find all retriever type and chunk size columns with different naming patterns
    retriever_cols = [col for col in config_cols if 'retriever' in col.lower() and ('type' in col.lower() or 'kind' in col.lower())]
    chunk_size_cols = [col for col in config_cols if 'chunk' in col.lower() and 'size' in col.lower()]
    
    # If we found these columns, do groupby analysis
    if retriever_cols and chunk_size_cols:
        retriever_col = retriever_cols[0]
        chunk_size_col = chunk_size_cols[0]
        
        # Only include rows that have both values
        group_data = successful_runs.dropna(subset=[retriever_col, chunk_size_col])
        
        if len(group_data) > 0:
            grouped = group_data.groupby([retriever_col, chunk_size_col])
            group_stats = grouped['latency'].agg(['mean', 'median', 'min', 'max']).reset_index()
            latency_stats['by_config'] = group_stats.to_dict(orient='records')
    
    # Add other interesting groupings if we have the data
    if 'config_team_type' in config_cols:
        team_groups = successful_runs.groupby('config_team_type')['latency'].agg(['mean', 'median', 'min', 'max']).reset_index()
        latency_stats['by_team'] = team_groups.to_dict(orient='records')
    
    return latency_stats

def visualize_performance(df, output_dir=None):
    """
    Create visualizations of performance metrics.
    
    Args:
        df: DataFrame with run performance data
        output_dir: Directory to save visualizations
    """
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Check if we have data to visualize
    if len(df) == 0:
        print("No runs with latency data to visualize")
        return
    
    # Filter successful runs if status column exists
    if 'status' in df.columns:
        successful_runs = df[df['status'] == 'succeeded']
        if len(successful_runs) == 0:
            successful_runs = df  # If no 'succeeded' runs, use all runs
    else:
        successful_runs = df
    
    try:
        # Histogram of latencies
        plt.figure(figsize=(12, 6))
        sns.histplot(successful_runs['latency'], kde=True, bins=20)
        plt.title('Distribution of Latency Times')
        plt.xlabel('Latency (seconds)')
        plt.ylabel('Count')
        plt.tight_layout()
        
        if output_dir:
            plt.savefig(Path(output_dir) / 'latency_distribution.png', dpi=300)
            plt.close()
        
        # Identify config columns for retriever type and chunk size
        config_cols = [col for col in df.columns if col.startswith('config_')]
        retriever_cols = [col for col in config_cols if 'retriever' in col.lower() and ('type' in col.lower() or 'kind' in col.lower())]
        chunk_size_cols = [col for col in config_cols if 'chunk' in col.lower() and 'size' in col.lower()]
        
        # Box plot of latencies by retriever type if available
        if retriever_cols:
            retriever_col = retriever_cols[0]
            valid_data = successful_runs.dropna(subset=[retriever_col])
            
            if len(valid_data) > 0:
                plt.figure(figsize=(14, 8))
                sns.boxplot(x=retriever_col, y='latency', data=valid_data)
                plt.title('Latency by Retriever Type')
                plt.ylabel('Latency (seconds)')
                plt.xlabel('Retriever Type')
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                if output_dir:
                    plt.savefig(Path(output_dir) / 'latency_by_retriever.png', dpi=300)
                    plt.close()
        
        # Box plot by chunk size if available
        if chunk_size_cols:
            chunk_size_col = chunk_size_cols[0]
            valid_data = successful_runs.dropna(subset=[chunk_size_col])
            
            if len(valid_data) > 0:
                plt.figure(figsize=(12, 6))
                try:
                    # Convert to numeric if not already
                    valid_data[chunk_size_col] = pd.to_numeric(valid_data[chunk_size_col])
                    # Sort by chunk size
                    valid_data = valid_data.sort_values(by=chunk_size_col)
                except:
                    pass  # If conversion fails, proceed with original data
                
                sns.boxplot(x=chunk_size_col, y='latency', data=valid_data)
                plt.title('Latency by Chunk Size')
                plt.ylabel('Latency (seconds)')
                plt.xlabel('Chunk Size')
                plt.tight_layout()
                
                if output_dir:
                    plt.savefig(Path(output_dir) / 'latency_by_chunk_size.png', dpi=300)
                    plt.close()
        
        # Heat map of latency by configuration combinations
        if retriever_cols and chunk_size_cols:
            retriever_col = retriever_cols[0]
            chunk_size_col = chunk_size_cols[0]
            
            valid_data = successful_runs.dropna(subset=[retriever_col, chunk_size_col])
            
            if len(valid_data) > 0:
                try:
                    # Create pivot table
                    pivot_data = valid_data.pivot_table(
                        values='latency',
                        index=retriever_col,
                        columns=chunk_size_col,
                        aggfunc='mean'
                    )
                    
                    if not pivot_data.empty:
                        plt.figure(figsize=(12, 8))
                        sns.heatmap(pivot_data, annot=True, cmap='viridis', fmt=".2f")
                        plt.title('Mean Latency by Retriever Type and Chunk Size')
                        plt.tight_layout()
                        
                        if output_dir:
                            plt.savefig(Path(output_dir) / 'latency_heatmap.png', dpi=300)
                            plt.close()
                except Exception as e:
                    print(f"Error creating heatmap: {e}")
        
        # Create bar chart showing mean latency and 95% CI by experiment name
        if 'experiment_name' in successful_runs.columns:
            try:
                exp_groups = successful_runs.groupby('experiment_name')['latency'].agg(['mean', 'count', 'std']).reset_index()
                
                # Calculate 95% confidence interval
                z = 1.96  # z-score for 95% confidence
                exp_groups['ci'] = z * exp_groups['std'] / np.sqrt(exp_groups['count'])
                
                if not exp_groups.empty:
                    # Sort by mean latency
                    exp_groups = exp_groups.sort_values('mean')
                    
                    plt.figure(figsize=(14, 10))
                    # Bar chart with error bars
                    plt.barh(y=exp_groups['experiment_name'], width=exp_groups['mean'], 
                            xerr=exp_groups['ci'], capsize=5)
                    plt.title('Mean Latency by Experiment (with 95% CI)')
                    plt.xlabel('Latency (seconds)')
                    plt.ylabel('Experiment')
                    plt.tight_layout()
                    
                    if output_dir:
                        plt.savefig(Path(output_dir) / 'latency_by_experiment.png', dpi=300)
                        plt.close()
            except Exception as e:
                print(f"Error creating experiment comparison chart: {e}")
    
    except Exception as e:
        print(f"Error in visualization: {e}")
        import traceback
        traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description="Analyze load times from RAG system experiments in LangSmith")
    parser.add_argument("--project", type=str, default="w267-final-project-rag-pipeline",
                        help="LangSmith project name (default: w267-final-project-rag-pipeline)")
    parser.add_argument("--filter_tags", type=str, nargs="+", help="Filter by experiment tags")
    parser.add_argument("--output_dir", type=str, help="Directory to save analysis outputs")
    parser.add_argument("--visualize", action="store_true", help="Create visualizations")
    
    args = parser.parse_args()
    
    # Create timestamp for output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Set output directory
    output_dir = args.output_dir
    if not output_dir:
        output_dir = f"results/performance_analysis_{timestamp}"
    
    # Fetch experiment runs
    runs = fetch_experiment_runs(args.project, args.filter_tags)
    
    if not runs:
        print("No runs found. Check your project name and filters.")
        return
    
    # Analyze performance
    df = analyze_run_performance(runs)
    
    # Calculate statistics
    stats = calculate_statistics(df)
    
    # Print summary statistics
    print("\n" + "="*80)
    print("PERFORMANCE STATISTICS SUMMARY")
    print("="*80)
    print(f"Total Runs: {stats['total_runs']}")
    print(f"Successful Runs: {stats['successful_runs']}")
    print(f"Failed Runs: {stats['failed_runs']}")
    
    if stats['successful_runs'] > 0:
        print("\nLATENCY STATISTICS (seconds):")
        print(f"  Mean: {stats['latency']['mean']:.2f}")
        print(f"  Median: {stats['latency']['median']:.2f}")
        print(f"  Min: {stats['latency']['min']:.2f}")
        print(f"  Max: {stats['latency']['max']:.2f}")
        print(f"  Std Dev: {stats['latency']['std']:.2f}")
        print(f"  90th Percentile: {stats['latency']['p90']:.2f}")
        print(f"  95th Percentile: {stats['latency']['p95']:.2f}")
        print(f"  99th Percentile: {stats['latency']['p99']:.2f}")
    
    # Create output directory
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Save raw data
        df.to_csv(Path(output_dir) / 'run_performance.csv', index=False)
        
        # Save statistics
        import json
        with open(Path(output_dir) / 'performance_stats.json', 'w') as f:
            json.dump(stats, f, indent=2, default=str)
        
        print(f"\nAnalysis saved to {output_dir}")
    
    # Create visualizations
    if args.visualize:
        visualize_performance(df, output_dir)
        if output_dir:
            print(f"Visualizations saved to {output_dir}")

if __name__ == "__main__":
    main()