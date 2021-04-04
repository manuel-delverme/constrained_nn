import wandb
api = wandb.Api()

sweep = api.sweep("delvermm/constrained_nn/w5ysu1sm")
# runs = api.runs("delvermm/constrained_nn")
for run in sweep.runs:
    if run.state == 'running':
        run
    # run.summary are the output key/values like accuracy.  We call ._json_dict to omit large files
    summary_list.append(run.summary._json_dict)

    # run.config is the input metrics.  We remove special values that start with _.
    config_list.append({k:v for k,v in run.config.items() if not k.startswith('_')})

    # run.name is the name of the run.
    name_list.append(run.name)

import pandas as pd
summary_df = pd.DataFrame.from_records(summary_list)
config_df = pd.DataFrame.from_records(config_list)
name_df = pd.DataFrame({'name': name_list})
all_df = pd.concat([name_df, config_df,summary_df], axis=1)

all_df.to_csv("project.csv")
