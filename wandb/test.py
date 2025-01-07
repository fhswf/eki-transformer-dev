
"""
    You can filters by using the MongoDB Query Language.
Date

runs = api.runs(
    "<entity>/<project>",
    {"$and": [{"created_at": {"$lt": "YYYY-MM-DDT##", "$gt": "YYYY-MM-DDT##"}}]},
)

# select
if run.state == "finished":
    for i, row in run.history(keys=["accuracy"]).iterrows():
        print(row["_timestamp"], row["accuracy"])
        
# rename
run.summary["new_name"] = run.summary["old_name"]
del run.summary["old_name"]
run.summary.update()

When you pull data from history, by default it's sampled to 500 points. Get all the logged data points using run.scan_history(). Here's an example downloading all the loss data points logged in history.
history = run.scan_history()
losses = [row["loss"] for row in history]
paginated run.scan_history(keys=sorted(cols), page_size=100)

run meta data:
meta = json.load(run.file("wandb-metadata.json").download())
program = ["python"] + [meta["program"]] + meta["args"]
"""

import pandas as pd
import wandb

api = wandb.Api()
entity, project = "eki-fhswf", "qtransform"
# 50 runs at a time in sequence as required
# kwarg per_page to change
# kwarg order "+/-created_at"

runs = api.runs(entity + "/" + project)
# call .update() on any object in run to upload changed data

summary_list, config_list, name_list = [], [], []
for run in runs:
    # .summary contains output keys/values for
    # metrics such as accuracy.
    #  We call ._json_dict to omit large files
    summary_list.append(run.summary._json_dict)

    # .config contains the hyperparameters.
    #  We remove special values that start with _.
    config_list.append({k: v for k, v in run.config.items() if not k.startswith("_")})

    # .name is the human-readable name of the run.
    name_list.append(run.name)

runs_df = pd.DataFrame(
    {"summary": summary_list, "config": config_list, "name": name_list}
)

runs_df.to_csv("project.csv")