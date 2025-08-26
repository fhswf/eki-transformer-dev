"""
Comprehensive script to extract wandb training logs, metrics, and graphs 
and store them in a local SQLite database.

This script:
1. Fetches all runs from a wandb project
2. Downloads complete training history (metrics over time)
3. Downloads and stores graph/plot images
4. Stores everything in a local SQLite database
"""

import os
import json
import sqlite3
import pandas as pd
import wandb
from datetime import datetime
import base64
import requests
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

class WandbToDatabase:
    def __init__(self, entity, project, db_path="wandb_data.db", download_dir="wandb_downloads"):
        self.entity = entity
        self.project = project
        self.db_path = db_path
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(exist_ok=True)
        
        # Initialize wandb API
        self.api = wandb.Api()
        
        # Initialize database
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Table for run metadata
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                name TEXT,
                state TEXT,
                created_at TEXT,
                updated_at TEXT,
                runtime_seconds REAL,
                config TEXT,
                summary TEXT,
                tags TEXT,
                notes TEXT,
                project TEXT,
                entity TEXT
            )
        """)
        
        # Table for training history/metrics
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT,
                step INTEGER,
                epoch REAL,
                timestamp TEXT,
                metric_name TEXT,
                metric_value REAL,
                FOREIGN KEY (run_id) REFERENCES runs (run_id)
            )
        """)
        
        # Table for files and artifacts
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT,
                file_name TEXT,
                file_type TEXT,
                file_size INTEGER,
                local_path TEXT,
                download_url TEXT,
                created_at TEXT,
                FOREIGN KEY (run_id) REFERENCES runs (run_id)
            )
        """)
        
        # Table for plots/graphs (stored as base64 encoded images)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS plots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT,
                plot_name TEXT,
                plot_type TEXT,
                image_data TEXT,
                local_path TEXT,
                created_at TEXT,
                FOREIGN KEY (run_id) REFERENCES runs (run_id)
            )
        """)
        
        conn.commit()
        conn.close()
    
    def fetch_and_store_runs(self, filters=None, limit=None):
        """Fetch runs from wandb and store in database"""
        print(f"Fetching runs from {self.entity}/{self.project}...")
        
        # Get runs with optional filters
        if filters:
            runs = self.api.runs(f"{self.entity}/{self.project}", filters)
        else:
            runs = self.api.runs(f"{self.entity}/{self.project}")
        
        if limit:
            runs = list(runs)[:limit]
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for i, run in enumerate(runs):
            print(f"Processing run {i+1}: {run.name} ({run.id})")
            
            # Store run metadata
            cursor.execute("""
                INSERT OR REPLACE INTO runs 
                (run_id, name, state, created_at, updated_at, runtime_seconds, 
                 config, summary, tags, notes, project, entity)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                run.id,
                run.name,
                run.state,
                str(getattr(run, 'created_at', '')) if getattr(run, 'created_at', None) else None,
                str(getattr(run, 'updated_at', '')) if getattr(run, 'updated_at', None) else None,
                getattr(run, 'runtime', None),
                json.dumps(dict(run.config)),
                json.dumps(run.summary._json_dict),
                json.dumps(run.tags),
                getattr(run, 'notes', ''),
                self.project,
                self.entity
            ))
            
            # Store complete training history
            self.store_training_history(run, cursor)
            
            # Download and store files/artifacts
            self.download_files(run, cursor)
            
            # Generate and store plots
            self.generate_plots(run, cursor)
        
        conn.commit()
        conn.close()
        print("All runs processed and stored in database!")
    
    def store_training_history(self, run, cursor):
        """Store complete training metrics history"""
        print(f"  Downloading training history for {run.name}...")
        
        try:
            # Get all history data (not sampled)
            history = run.scan_history()
            
            for row in history:
                step = row.get('_step', None)
                epoch = row.get('epoch', None)
                timestamp = row.get('_timestamp', None)
                
                # Convert timestamp to ISO format if it exists
                if timestamp:
                    timestamp = datetime.fromtimestamp(timestamp).isoformat()
                
                # Store each metric
                for key, value in row.items():
                    if key.startswith('_'):
                        continue  # Skip internal wandb keys
                    
                    if isinstance(value, (int, float)):
                        cursor.execute("""
                            INSERT INTO metrics 
                            (run_id, step, epoch, timestamp, metric_name, metric_value)
                            VALUES (?, ?, ?, ?, ?, ?)
                        """, (run.id, step, epoch, timestamp, key, value))
        
        except Exception as e:
            print(f"    Error downloading history: {e}")
    
    def download_files(self, run, cursor):
        """Download files and artifacts from the run"""
        print(f"  Downloading files for {run.name}...")
        
        try:
            # Create run-specific directory
            run_dir = self.download_dir / run.id
            run_dir.mkdir(exist_ok=True)
            
            # Download files
            for file in run.files():
                try:
                    local_path = run_dir / file.name
                    file.download(root=str(run_dir), replace=True)
                    
                    # Store file info in database
                    cursor.execute("""
                        INSERT OR REPLACE INTO files 
                        (run_id, file_name, file_type, file_size, local_path, download_url, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        run.id,
                        file.name,
                        file.name.split('.')[-1] if '.' in file.name else 'unknown',
                        file.size,
                        str(local_path),
                        file.url,
                        datetime.now().isoformat()
                    ))
                    
                except Exception as e:
                    print(f"    Error downloading {file.name}: {e}")
        
        except Exception as e:
            print(f"    Error accessing files: {e}")
    
    def generate_plots(self, run, cursor):
        """Generate and store plots from training data"""
        print(f"  Generating plots for {run.name}...")
        
        try:
            # Get history for plotting
            history_df = run.history()
            
            if history_df.empty:
                return
            
            run_dir = self.download_dir / run.id / "plots"
            run_dir.mkdir(exist_ok=True)
            
            # Common metrics to plot
            metrics_to_plot = []
            for col in history_df.columns:
                if col.startswith('_'):
                    continue
                if col in ['loss', 'accuracy', 'val_loss', 'val_accuracy', 'train_loss', 'learning_rate']:
                    metrics_to_plot.append(col)
                elif 'loss' in col.lower() or 'acc' in col.lower() or 'lr' in col.lower():
                    metrics_to_plot.append(col)
            
            # Generate individual metric plots
            for metric in metrics_to_plot:
                if metric in history_df.columns:
                    plt.figure(figsize=(10, 6))
                    plt.plot(history_df.index, history_df[metric])
                    plt.title(f'{metric} over time - {run.name}')
                    plt.xlabel('Step')
                    plt.ylabel(metric)
                    plt.grid(True)
                    
                    # Clean metric name for filename
                    clean_metric = metric.replace('/', '_').replace(' ', '_')
                    plot_path = run_dir / f"{clean_metric}_plot.png"
                    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                    plt.close()
                    
                    # Store plot in database as base64
                    with open(plot_path, 'rb') as f:
                        image_data = base64.b64encode(f.read()).decode()
                    
                    cursor.execute("""
                        INSERT OR REPLACE INTO plots 
                        (run_id, plot_name, plot_type, image_data, local_path, created_at)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        run.id,
                        f"{clean_metric}_plot",
                        "line_plot",
                        image_data,
                        str(plot_path),
                        datetime.now().isoformat()
                    ))
            
            # Generate combined loss plot if multiple loss metrics exist
            loss_cols = [col for col in metrics_to_plot if 'loss' in col.lower()]
            if len(loss_cols) > 1:
                plt.figure(figsize=(12, 6))
                for loss_col in loss_cols:
                    plt.plot(history_df.index, history_df[loss_col], label=loss_col)
                plt.title(f'Loss Comparison - {run.name}')
                plt.xlabel('Step')
                plt.ylabel('Loss')
                plt.legend()
                plt.grid(True)
                
                plot_path = run_dir / "loss_comparison.png"
                plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                plt.close()
                
                with open(plot_path, 'rb') as f:
                    image_data = base64.b64encode(f.read()).decode()
                
                cursor.execute("""
                    INSERT OR REPLACE INTO plots 
                    (run_id, plot_name, plot_type, image_data, local_path, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    run.id,
                    "loss_comparison",
                    "comparison_plot",
                    image_data,
                    str(plot_path),
                    datetime.now().isoformat()
                ))
        
        except Exception as e:
            print(f"    Error generating plots: {e}")
    
    def query_database(self, query):
        """Execute SQL query on the database"""
        conn = sqlite3.connect(self.db_path)
        try:
            return pd.read_sql_query(query, conn)
        finally:
            conn.close()
    
    def get_run_summary(self):
        """Get summary of all runs in database"""
        return self.query_database("""
            SELECT run_id, name, state, created_at, 
                   json_extract(summary, '$.loss') as final_loss,
                   json_extract(summary, '$.accuracy') as final_accuracy
            FROM runs 
            ORDER BY created_at DESC
        """)
    
    def get_metrics_for_run(self, run_id):
        """Get all metrics for a specific run"""
        return self.query_database(f"""
            SELECT * FROM metrics 
            WHERE run_id = '{run_id}' 
            ORDER BY step
        """)
    
    def export_to_csv(self, output_dir="exported_data"):
        """Export all data to CSV files"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Export runs
        runs_df = self.query_database("SELECT * FROM runs")
        runs_df.to_csv(output_path / "runs.csv", index=False)
        
        # Export metrics
        metrics_df = self.query_database("SELECT * FROM metrics")
        metrics_df.to_csv(output_path / "metrics.csv", index=False)
        
        # Export files info
        files_df = self.query_database("SELECT * FROM files")
        files_df.to_csv(output_path / "files.csv", index=False)
        
        print(f"Data exported to {output_dir}/")


def main():
    print("Starting wandb data extraction...")
    
    # Configuration
    entity = "eki-fhswf"
    project = "qtransform"
    
    try:
        # Initialize the wandb to database converter
        print(f"Connecting to wandb project: {entity}/{project}")
        converter = WandbToDatabase(entity, project)
        
        # Fetch and store all runs (increase limit to get more data)
        print("Fetching runs from wandb...")
        converter.fetch_and_store_runs(limit=504)
        
        # Example queries
        print("\n=== Run Summary ===")
        summary = converter.get_run_summary()
        print(summary)
        
        # Get metrics for the first run
        if not summary.empty:
            first_run_id = summary.iloc[0]['run_id']
            print(f"\n=== Metrics for run {first_run_id} ===")
            metrics = converter.get_metrics_for_run(first_run_id)
            print(metrics.head())
        
        # Export everything to CSV
        print("\nExporting to CSV...")
        converter.export_to_csv()
        
        print("\n✅ Successfully completed wandb data extraction!")
        
    except Exception as e:
        print(f"❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
