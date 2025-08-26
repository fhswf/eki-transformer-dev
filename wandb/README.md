# Wandb to Local Database

This directory contains scripts to extract training logs, metrics, and graphs from Weights & Biases (wandb) and store them in a local SQLite database for offline analysis.

## Features

- **Complete data extraction**: Downloads run metadata, training metrics, configuration, and files
- **Plot generation**: Creates and stores training curve plots locally
- **Local database storage**: Uses SQLite for efficient querying and analysis
- **Data analysis tools**: Provides utilities to analyze and compare runs offline
- **Export capabilities**: Can export data to CSV or HTML reports

## Files

- `wandb_to_database.py` - Main script to extract wandb data and store in local database
- `analyze_wandb_data.py` - Analysis utilities for querying and visualizing stored data
- `test.py` - Your original wandb API exploration script
- `requirements.txt` - Required Python packages

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure you're logged into wandb:
```bash
wandb login
```

## Usage

### 1. Extract Data from Wandb

Edit the configuration in `wandb_to_database.py`:
```python
entity = "your-entity"  # Currently set to "eki-fhswf"
project = "your-project"  # Currently set to "qtransform"
```

Then run:
```bash
python wandb_to_database.py
```

This will:
- Create a SQLite database (`wandb_data.db`)
- Download all run data including:
  - Run metadata (config, summary, tags, etc.)
  - Complete training history (all logged metrics over time)
  - Files and artifacts
  - Generated training plots
- Store everything locally for offline access

### 2. Analyze Stored Data

```bash
python analyze_wandb_data.py
```

This provides examples of:
- Finding best performing runs
- Comparing runs across metrics
- Analyzing hyperparameter configurations
- Generating comparison plots
- Exporting HTML reports

## How to Use - CLI Commands

Here are the actual commands we used to set up and run the system:

### Initial Setup
```bash
# Navigate to the wandb directory
cd /home/kuhmichel/git/eki-transformer-dev/wandb

# Install required packages
pip install -r requirements.txt
```

### Extract Data from Wandb
```bash
# Run the main extraction script
python wandb_to_database.py
```

### Analyze Extracted Data
```bash
# Run the demo analysis
python demo_analysis.py

# Check what files were created
ls -la

# View downloaded files structure
ls -la wandb_downloads/
ls -la wandb_downloads/32x1gxxz/  # Example run directory

# Check exported CSV data
ls -la exported_data/
```

### Database Inspection Commands
```bash
# Check available metrics using SQLite CLI
sqlite3 wandb_data.db "SELECT DISTINCT metric_name FROM metrics LIMIT 10;"

# Count data points per metric and run
sqlite3 wandb_data.db "SELECT run_id, metric_name, COUNT(*) as count FROM metrics GROUP BY run_id, metric_name;"

# View run summary
sqlite3 wandb_data.db "SELECT run_id, name, state FROM runs;"

# Get training loss data
sqlite3 wandb_data.db "SELECT run_id, step, metric_value FROM metrics WHERE metric_name='train/loss' LIMIT 10;"

# params for each run:
sqlite3 wandb_data.db "SELECT DISTINCT json_extract(config, '$') FROM runs LIMIT 1;" | python -m json.tool
```

### Python Commands for Quick Analysis
```bash
# Check database contents programmatically
python -c "
import sqlite3
import pandas as pd

conn = sqlite3.connect('wandb_data.db')
print('=== Available metrics ===')
metrics = pd.read_sql_query('SELECT DISTINCT metric_name FROM metrics', conn)
print(metrics)

print('\n=== Run summary ===')
runs = pd.read_sql_query('SELECT run_id, name, state FROM runs', conn)
print(runs)
conn.close()
"

# Test wandb connection
python -c "
import wandb
api = wandb.Api()
runs = api.runs('eki-fhswf/qtransform')
print(f'Found {len(list(runs))} runs')
"
```

### File Operations
```bash
# View database size and structure
ls -lh wandb_data.db
file wandb_data.db

# Check CSV exports
head exported_data/metrics.csv
wc -l exported_data/*.csv

# View generated plots
ls -la wandb_downloads/*/plots/
```

### Backup and Maintenance
```bash
# Create backup of database
cp wandb_data.db wandb_data_backup_$(date +%Y%m%d).db

# Clean up old downloads (if needed)
rm -rf wandb_downloads/

# Re-run extraction with different limits
python -c "
from wandb_to_database import WandbToDatabase
converter = WandbToDatabase('eki-fhswf', 'qtransform')
converter.fetch_and_store_runs(limit=10)  # Get 10 runs instead of 5
"
```

### Advanced Queries
```bash
# Complex SQL queries
sqlite3 wandb_data.db "
SELECT r.name, 
       MIN(m.metric_value) as min_loss,
       MAX(m.metric_value) as max_loss,
       COUNT(m.step) as training_steps
FROM runs r 
JOIN metrics m ON r.run_id = m.run_id 
WHERE m.metric_name = 'train/loss'
GROUP BY r.run_id;
"

# Export specific data to CSV
sqlite3 wandb_data.db -header -csv "
SELECT r.name, m.step, m.metric_value 
FROM runs r 
JOIN metrics m ON r.run_id = m.run_id 
WHERE m.metric_name = 'train/loss'
ORDER BY r.name, m.step;
" > training_loss_data.csv
```

## Database Schema

The SQLite database contains four main tables:

### `runs` table
- `run_id`: Unique wandb run identifier
- `name`: Human-readable run name
- `state`: Run state (finished, failed, running, etc.)
- `config`: JSON string of hyperparameters
- `summary`: JSON string of final metrics
- `created_at`, `updated_at`: Timestamps
- `runtime_seconds`: Total runtime
- `tags`, `notes`: Additional metadata

### `metrics` table
- `run_id`: Foreign key to runs table
- `step`: Training step number
- `epoch`: Training epoch (if logged)
- `timestamp`: When metric was logged
- `metric_name`: Name of the metric (loss, accuracy, etc.)
- `metric_value`: Numerical value

### `files` table
- `run_id`: Foreign key to runs table
- `file_name`: Original filename
- `file_type`: File extension
- `local_path`: Where file is stored locally
- `download_url`: Original wandb URL

### `plots` table
- `run_id`: Foreign key to runs table
- `plot_name`: Name of the generated plot
- `plot_type`: Type of plot (line_plot, comparison_plot, etc.)
- `image_data`: Base64 encoded PNG image
- `local_path`: Path to saved image file

## Example Queries

### Find best runs by final loss:
```python
from wandb_to_database import WandbToDatabase

converter = WandbToDatabase("entity", "project")
best_runs = converter.query_database("""
    SELECT run_id, name, json_extract(summary, '$.loss') as final_loss
    FROM runs 
    WHERE json_extract(summary, '$.loss') IS NOT NULL
    ORDER BY json_extract(summary, '$.loss') ASC
    LIMIT 5
""")
```

### Get training curve for a specific run:
```python
metrics = converter.query_database("""
    SELECT step, metric_value 
    FROM metrics 
    WHERE run_id = 'your_run_id' AND metric_name = 'loss'
    ORDER BY step
""")
```

### Compare hyperparameters of top runs:
```python
analysis = converter.query_database("""
    SELECT 
        name,
        json_extract(config, '$.learning_rate') as lr,
        json_extract(config, '$.batch_size') as batch_size,
        json_extract(summary, '$.loss') as final_loss
    FROM runs 
    ORDER BY json_extract(summary, '$.loss') ASC
""")
```

## Customization

### Adding Custom Plots
Edit the `generate_plots()` method in `WandbToDatabase` to create additional visualizations:

```python
# Add custom plot generation
if 'custom_metric' in history_df.columns:
    plt.figure(figsize=(10, 6))
    plt.plot(history_df.index, history_df['custom_metric'])
    # ... save and store plot
```

### Filtering Runs
You can filter which runs to download:

```python
# Only download finished runs from last week
filters = {
    "$and": [
        {"state": "finished"},
        {"created_at": {"$gte": "2023-01-01T00:00:00"}}
    ]
}
converter.fetch_and_store_runs(filters=filters)
```

### Adding Custom Analysis
Extend the `WandbDataAnalyzer` class with your own analysis methods:

```python
def analyze_convergence(self, run_id):
    # Custom analysis for training convergence
    metrics = self.query(f"""
        SELECT step, metric_value FROM metrics 
        WHERE run_id = '{run_id}' AND metric_name = 'loss'
        ORDER BY step
    """)
    # Your analysis logic here
```

## Benefits of Local Storage

1. **Offline access**: Analyze data without internet connection
2. **Fast queries**: SQLite provides efficient local querying
3. **Data preservation**: Keep data even if wandb runs are deleted
4. **Custom analysis**: Run complex SQL queries across all runs
5. **Backup**: Local backup of training results
6. **Integration**: Easy to integrate with other local tools and scripts

## Troubleshooting

- **Authentication**: Make sure you're logged into wandb with `wandb login`
- **Permissions**: Ensure you have access to the specified entity/project
- **Storage space**: Large projects may require significant disk space
- **Rate limits**: The script includes error handling for API limits
- **Missing metrics**: Not all runs may have the same logged metrics
