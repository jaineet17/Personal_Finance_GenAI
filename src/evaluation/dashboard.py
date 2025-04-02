"""
Dashboard for visualizing evaluation results for the Finance RAG system.
"""
import os
import json
import argparse
import glob
from pathlib import Path
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import numpy as np
import pandas as pd

def load_results(results_dir: str = "evaluation_results") -> List[Dict[str, Any]]:
    """Load all evaluation results from a directory.
    
    Args:
        results_dir (str): Directory containing evaluation result JSON files
        
    Returns:
        List[Dict[str, Any]]: List of evaluation results
    """
    results = []
    
    # Find all evaluation result JSON files
    json_files = glob.glob(os.path.join(results_dir, "evaluation_results_*.json"))
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                # Add filename to data for reference
                data["filename"] = os.path.basename(json_file)
                # Parse timestamp from filename if not in the data
                if "timestamp" not in data:
                    timestamp_str = data["filename"].replace("evaluation_results_", "").replace(".json", "")
                    try:
                        timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S").timestamp()
                        data["timestamp"] = timestamp
                    except:
                        pass
                results.append(data)
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
    
    # Sort by timestamp
    results.sort(key=lambda x: x.get("timestamp", 0))
    
    return results

def create_performance_charts(results: List[Dict[str, Any]], output_dir: str = "evaluation_results"):
    """Create performance charts from evaluation results.
    
    Args:
        results (List[Dict[str, Any]]): List of evaluation results
        output_dir (str): Directory to save charts
    """
    if not results:
        print("No results to create charts from")
        return
    
    # Extract summary metrics
    timestamps = []
    avg_response_times = []
    avg_relevance_scores = []
    query_counts = []
    
    for result in results:
        # Extract timestamp
        timestamp = result.get("timestamp", 0)
        if timestamp:
            dt = datetime.fromtimestamp(timestamp)
            timestamps.append(dt)
            
            # Extract summary metrics
            summary = result.get("summary", {})
            avg_response_times.append(summary.get("avg_response_time_ms", 0))
            avg_relevance_scores.append(summary.get("avg_relevance", 0))
            query_counts.append(summary.get("total_queries", 0))
    
    if not timestamps:
        print("No valid timestamps found in results")
        return
    
    # Create figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    # Plot average response time
    ax1.plot(timestamps, avg_response_times, 'o-', color='blue')
    ax1.set_ylabel('Avg Response Time (ms)')
    ax1.set_title('Average Response Time Over Time')
    ax1.grid(True)
    
    # Plot average relevance score
    ax2.plot(timestamps, avg_relevance_scores, 'o-', color='green')
    ax2.set_ylabel('Avg Relevance Score')
    ax2.set_title('Average Relevance Score Over Time')
    ax2.grid(True)
    
    # Plot query count
    ax3.plot(timestamps, query_counts, 'o-', color='purple')
    ax3.set_ylabel('Total Queries')
    ax3.set_title('Total Queries Evaluated Over Time')
    ax3.grid(True)
    
    # Format x-axis to show dates nicely
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    plt.gcf().autofmt_xdate()
    
    # Set common title
    plt.suptitle('Finance RAG System Performance Metrics', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    # Save figure
    charts_file = os.path.join(output_dir, "performance_charts.png")
    plt.savefig(charts_file)
    plt.close()
    
    print(f"Performance charts saved to {charts_file}")
    
    return charts_file

def create_query_performance_chart(results: List[Dict[str, Any]], output_dir: str = "evaluation_results"):
    """Create a chart showing performance by query type.
    
    Args:
        results (List[Dict[str, Any]]): List of evaluation results
        output_dir (str): Directory to save charts
    """
    if not results:
        print("No results to create charts from")
        return
    
    # Get the latest result for analysis
    latest_result = results[-1]
    
    # Extract query metrics
    queries = []
    response_times = []
    relevance_scores = []
    
    for metric in latest_result.get("metrics", []):
        query = metric.get("query", "")
        if len(query) > 30:
            query = query[:27] + "..."
        queries.append(query)
        response_times.append(metric.get("response_time_ms", 0))
        relevance_scores.append(metric.get("relevance_score", 0))
    
    if not queries:
        print("No queries found in latest result")
        return
    
    # Create a dataframe for easier plotting
    df = pd.DataFrame({
        'Query': queries,
        'Response Time (ms)': response_times,
        'Relevance Score': relevance_scores
    })
    
    # Sort by response time
    df = df.sort_values('Response Time (ms)', ascending=False)
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot response time by query
    ax1.barh(df['Query'], df['Response Time (ms)'], color='blue')
    ax1.set_xlabel('Response Time (ms)')
    ax1.set_title('Response Time by Query')
    ax1.grid(True, axis='x')
    
    # Plot relevance score by query
    ax2.barh(df['Query'], df['Relevance Score'], color='green')
    ax2.set_xlabel('Relevance Score')
    ax2.set_title('Relevance Score by Query')
    ax2.grid(True, axis='x')
    
    # Set common title
    plt.suptitle('Query Performance Metrics', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    # Save figure
    charts_file = os.path.join(output_dir, "query_performance_chart.png")
    plt.savefig(charts_file)
    plt.close()
    
    print(f"Query performance chart saved to {charts_file}")
    
    return charts_file

def create_breakdown_chart(results: List[Dict[str, Any]], output_dir: str = "evaluation_results"):
    """Create a chart showing the breakdown of response time.
    
    Args:
        results (List[Dict[str, Any]]): List of evaluation results
        output_dir (str): Directory to save charts
    """
    if not results:
        print("No results to create charts from")
        return
    
    # Get the latest result for analysis
    latest_result = results[-1]
    
    # Extract metrics
    llm_times = []
    retrieval_times = []
    other_times = []
    queries = []
    
    for metric in latest_result.get("metrics", []):
        query = metric.get("query", "")
        if len(query) > 30:
            query = query[:27] + "..."
        queries.append(query)
        
        response_time = metric.get("response_time_ms", 0)
        llm_time = metric.get("llm_time_ms", 0)
        retrieval_time = metric.get("retrieval_time_ms", 0)
        
        # Calculate other time (overhead)
        other_time = max(0, response_time - llm_time - retrieval_time)
        
        llm_times.append(llm_time)
        retrieval_times.append(retrieval_time)
        other_times.append(other_time)
    
    if not queries:
        print("No queries found in latest result")
        return
    
    # Create a stacked bar chart
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot stacked bars
    y_pos = np.arange(len(queries))
    bar_width = 0.75
    
    ax.barh(y_pos, retrieval_times, bar_width, label='Retrieval Time', color='gold')
    ax.barh(y_pos, llm_times, bar_width, left=retrieval_times, label='LLM Time', color='orangered')
    ax.barh(y_pos, other_times, bar_width, left=np.array(retrieval_times) + np.array(llm_times), 
            label='Other Processing Time', color='royalblue')
    
    # Add labels and title
    ax.set_yticks(y_pos)
    ax.set_yticklabels(queries)
    ax.set_xlabel('Time (ms)')
    ax.set_title('Response Time Breakdown by Component')
    ax.legend()
    
    plt.tight_layout()
    
    # Save figure
    charts_file = os.path.join(output_dir, "time_breakdown_chart.png")
    plt.savefig(charts_file)
    plt.close()
    
    print(f"Time breakdown chart saved to {charts_file}")
    
    return charts_file

def create_html_dashboard(results: List[Dict[str, Any]], output_dir: str = "evaluation_results"):
    """Create an HTML dashboard from evaluation results.
    
    Args:
        results (List[Dict[str, Any]]): List of evaluation results
        output_dir (str): Directory to save dashboard
    """
    if not results:
        print("No results to create dashboard from")
        return
    
    # Generate charts
    performance_chart = create_performance_charts(results, output_dir)
    query_chart = create_query_performance_chart(results, output_dir)
    breakdown_chart = create_breakdown_chart(results, output_dir)
    
    # Get the latest result for summary
    latest_result = results[-1]
    summary = latest_result.get("summary", {})
    
    # Create HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Finance RAG Evaluation Dashboard</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; }}
            .dashboard {{ max-width: 1200px; margin: 0 auto; }}
            .header {{ background-color: #f5f5f5; padding: 20px; border-radius: 5px; margin-bottom: 20px; }}
            .chart-container {{ margin-bottom: 30px; background-color: white; padding: 20px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            h1, h2 {{ color: #333; }}
            .metric {{ display: inline-block; margin-right: 30px; margin-bottom: 20px; }}
            .metric .value {{ font-size: 24px; font-weight: bold; color: #0066cc; }}
            .metric .label {{ font-size: 14px; color: #666; }}
            table {{ width: 100%; border-collapse: collapse; }}
            th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
            th {{ background-color: #f2f2f2; }}
            tr:hover {{ background-color: #f5f5f5; }}
        </style>
    </head>
    <body>
        <div class="dashboard">
            <div class="header">
                <h1>Finance RAG Evaluation Dashboard</h1>
                <p>Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                
                <div class="metrics">
                    <div class="metric">
                        <div class="value">{summary.get('avg_response_time_ms', 0):.2f} ms</div>
                        <div class="label">Avg Response Time</div>
                    </div>
                    <div class="metric">
                        <div class="value">{summary.get('avg_relevance', 0):.2f}</div>
                        <div class="label">Avg Relevance Score</div>
                    </div>
                    <div class="metric">
                        <div class="value">{summary.get('total_queries', 0)}</div>
                        <div class="label">Total Queries</div>
                    </div>
                </div>
            </div>
            
            <div class="chart-container">
                <h2>Performance Trends</h2>
                <img src="performance_charts.png" alt="Performance Charts" style="width: 100%;">
            </div>
            
            <div class="chart-container">
                <h2>Query Performance</h2>
                <img src="query_performance_chart.png" alt="Query Performance" style="width: 100%;">
            </div>
            
            <div class="chart-container">
                <h2>Response Time Breakdown</h2>
                <img src="time_breakdown_chart.png" alt="Time Breakdown" style="width: 100%;">
            </div>
            
            <div class="chart-container">
                <h2>Recent Evaluation Results</h2>
                <table>
                    <tr>
                        <th>Query</th>
                        <th>Response Time (ms)</th>
                        <th>Relevance Score</th>
                        <th>LLM Time (ms)</th>
                    </tr>
    """
    
    # Add rows for each query in the latest result
    for metric in latest_result.get("metrics", []):
        query = metric.get("query", "")
        response_time = metric.get("response_time_ms", 0)
        relevance_score = metric.get("relevance_score", 0)
        llm_time = metric.get("llm_time_ms", 0)
        
        html_content += f"""
                    <tr>
                        <td>{query}</td>
                        <td>{response_time:.2f}</td>
                        <td>{relevance_score:.2f}</td>
                        <td>{llm_time:.2f}</td>
                    </tr>
        """
    
    html_content += """
                </table>
            </div>
        </div>
    </body>
    </html>
    """
    
    # Write HTML to file
    dashboard_file = os.path.join(output_dir, "dashboard.html")
    with open(dashboard_file, 'w') as f:
        f.write(html_content)
    
    print(f"Dashboard saved to {dashboard_file}")
    
    return dashboard_file

def main():
    """Main entry point for the dashboard script."""
    parser = argparse.ArgumentParser(description="Generate dashboard for evaluation results")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="evaluation_results",
        help="Directory containing evaluation results"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation_results",
        help="Directory to save dashboard"
    )
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load results
    results = load_results(args.results_dir)
    
    if not results:
        print(f"No evaluation results found in {args.results_dir}")
        return 1
    
    # Create dashboard
    dashboard_file = create_html_dashboard(results, args.output_dir)
    
    print(f"Dashboard created at {dashboard_file}")
    print(f"Open this file in a web browser to view the dashboard")
    
    return 0

if __name__ == "__main__":
    main() 