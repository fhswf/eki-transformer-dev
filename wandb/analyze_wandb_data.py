"""
Example script showing how to query and analyze the wandb data 
stored in the local SQLite database.
"""

import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from PIL import Image
import json

class WandbDataAnalyzer:
    def __init__(self, db_path="wandb_data.db"):
        self.db_path = db_path
    
    def query(self, sql):
        """Execute SQL query and return DataFrame"""
        conn = sqlite3.connect(self.db_path)
        try:
            return pd.read_sql_query(sql, conn)
        finally:
            conn.close()
    
    def get_best_runs(self, metric="loss", ascending=True, limit=5):
        """Get best runs based on final metric value"""
        order = "ASC" if ascending else "DESC"
        return self.query(f"""
            SELECT run_id, name, state, 
                   json_extract(summary, '$.{metric}') as final_{metric},
                   created_at
            FROM runs 
            WHERE json_extract(summary, '$.{metric}') IS NOT NULL
            ORDER BY json_extract(summary, '$.{metric}') {order}
            LIMIT {limit}
        """)
    
    def compare_runs(self, run_ids, metric="loss"):
        """Compare specific runs for a given metric"""
        placeholders = ','.join(['?' for _ in run_ids])
        return self.query(f"""
            SELECT r.name, m.step, m.metric_value, m.timestamp
            FROM metrics m
            JOIN runs r ON m.run_id = r.run_id
            WHERE m.run_id IN ({placeholders}) AND m.metric_name = ?
            ORDER BY m.step
        """, run_ids + [metric])
    
    def plot_metric_comparison(self, run_ids, metric="loss", save_path=None):
        """Plot metric comparison for multiple runs"""
        df = self.compare_runs(run_ids, metric)
        
        plt.figure(figsize=(12, 6))
        for name in df['name'].unique():
            run_data = df[df['name'] == name]
            plt.plot(run_data['step'], run_data['metric_value'], 
                    label=name, marker='o', markersize=2)
        
        plt.xlabel('Step')
        plt.ylabel(metric.title())
        plt.title(f'{metric.title()} Comparison Across Runs')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    
    def get_run_config_analysis(self):
        """Analyze hyperparameter configurations"""
        runs_df = self.query("SELECT run_id, name, config, summary FROM runs")
        
        configs = []
        for _, row in runs_df.iterrows():
            config = json.loads(row['config'])
            summary = json.loads(row['summary'])
            
            config_data = {
                'run_id': row['run_id'],
                'name': row['name'],
                'final_loss': summary.get('loss'),
                'final_accuracy': summary.get('accuracy'),
                **config
            }
            configs.append(config_data)
        
        return pd.DataFrame(configs)
    
    def display_stored_plot(self, run_id, plot_name):
        """Display a plot stored in the database"""
        result = self.query(f"""
            SELECT image_data FROM plots 
            WHERE run_id = '{run_id}' AND plot_name = '{plot_name}'
        """)
        
        if not result.empty:
            image_data = base64.b64decode(result.iloc[0]['image_data'])
            image = Image.open(BytesIO(image_data))
            plt.figure(figsize=(10, 6))
            plt.imshow(image)
            plt.axis('off')
            plt.title(f'Plot: {plot_name} (Run: {run_id})')
            plt.show()
        else:
            print(f"Plot {plot_name} not found for run {run_id}")
    
    def get_training_efficiency(self):
        """Analyze training efficiency across runs"""
        return self.query("""
            SELECT 
                run_id,
                name,
                runtime_seconds,
                json_extract(summary, '$.loss') as final_loss,
                json_extract(summary, '$.accuracy') as final_accuracy,
                ROUND(runtime_seconds / 3600.0, 2) as runtime_hours,
                CASE 
                    WHEN json_extract(summary, '$.loss') IS NOT NULL 
                    THEN ROUND(runtime_seconds / json_extract(summary, '$.loss'), 2)
                    ELSE NULL 
                END as seconds_per_loss_unit
            FROM runs 
            WHERE runtime_seconds IS NOT NULL
            ORDER BY final_loss ASC
        """)
    
    def export_run_comparison(self, run_ids, output_file="run_comparison.html"):
        """Export detailed run comparison to HTML"""
        html_content = ["<html><head><title>Run Comparison</title></head><body>"]
        html_content.append("<h1>Wandb Run Comparison</h1>")
        
        for run_id in run_ids:
            # Get run info
            run_info = self.query(f"SELECT * FROM runs WHERE run_id = '{run_id}'")
            if run_info.empty:
                continue
            
            run_data = run_info.iloc[0]
            html_content.append(f"<h2>Run: {run_data['name']} ({run_id})</h2>")
            html_content.append(f"<p><strong>State:</strong> {run_data['state']}</p>")
            html_content.append(f"<p><strong>Created:</strong> {run_data['created_at']}</p>")
            
            # Add config
            config = json.loads(run_data['config'])
            html_content.append("<h3>Configuration</h3><ul>")
            for key, value in config.items():
                html_content.append(f"<li><strong>{key}:</strong> {value}</li>")
            html_content.append("</ul>")
            
            # Add final metrics
            summary = json.loads(run_data['summary'])
            html_content.append("<h3>Final Metrics</h3><ul>")
            for key, value in summary.items():
                if isinstance(value, (int, float)):
                    html_content.append(f"<li><strong>{key}:</strong> {value:.4f}</li>")
            html_content.append("</ul>")
            
            # Add plots if available
            plots = self.query(f"SELECT plot_name, image_data FROM plots WHERE run_id = '{run_id}'")
            if not plots.empty:
                html_content.append("<h3>Plots</h3>")
                for _, plot in plots.iterrows():
                    html_content.append(f'<h4>{plot["plot_name"]}</h4>')
                    html_content.append(f'<img src="data:image/png;base64,{plot["image_data"]}" style="max-width:800px;"/>')
            
            html_content.append("<hr>")
        
        html_content.append("</body></html>")
        
        with open(output_file, 'w') as f:
            f.write('\n'.join(html_content))
        
        print(f"Run comparison exported to {output_file}")


def main():
    analyzer = WandbDataAnalyzer()
    
    # Example analyses
    print("=== Best Runs by Loss ===")
    best_runs = analyzer.get_best_runs("loss", ascending=True, limit=3)
    print(best_runs)
    
    print("\n=== Training Efficiency Analysis ===")
    efficiency = analyzer.get_training_efficiency()
    print(efficiency)
    
    print("\n=== Configuration Analysis ===")
    config_analysis = analyzer.get_run_config_analysis()
    print(config_analysis.describe())
    
    # Plot comparison for best runs
    if not best_runs.empty:
        run_ids = best_runs['run_id'].tolist()[:3]  # Top 3 runs
        print(f"\n=== Plotting comparison for runs: {run_ids} ===")
        analyzer.plot_metric_comparison(run_ids, "loss")
        
        # Export comparison report
        analyzer.export_run_comparison(run_ids, "top_runs_comparison.html")


if __name__ == "__main__":
    main()
