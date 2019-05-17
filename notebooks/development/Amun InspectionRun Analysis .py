# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:hydrogen
#     text_representation:
#       extension: .py
#       format_name: hydrogen
#       format_version: '1.2'
#       jupytext_version: 1.1.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown] {"toc": true}
# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Retrieve-inspection-jobs-from-Ceph" data-toc-modified-id="Retrieve-inspection-jobs-from-Ceph-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Retrieve inspection jobs from Ceph</a></span></li><li><span><a href="#Describe-the-structure-of-an-inspection-job-result" data-toc-modified-id="Describe-the-structure-of-an-inspection-job-result-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Describe the structure of an inspection job result</a></span></li><li><span><a href="#Mapping-InspectionRun-JSON-to-pandas-DataFrame" data-toc-modified-id="Mapping-InspectionRun-JSON-to-pandas-DataFrame-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Mapping InspectionRun JSON to pandas DataFrame</a></span><ul class="toc-item"><li><span><a href="#Feature-importance-analysis" data-toc-modified-id="Feature-importance-analysis-3.1"><span class="toc-item-num">3.1&nbsp;&nbsp;</span>Feature importance analysis</a></span><ul class="toc-item"><li><span><a href="#Status" data-toc-modified-id="Status-3.1.1"><span class="toc-item-num">3.1.1&nbsp;&nbsp;</span>Status</a></span></li><li><span><a href="#Specification" data-toc-modified-id="Specification-3.1.2"><span class="toc-item-num">3.1.2&nbsp;&nbsp;</span>Specification</a></span></li><li><span><a href="#Job-log" data-toc-modified-id="Job-log-3.1.3"><span class="toc-item-num">3.1.3&nbsp;&nbsp;</span>Job log</a></span></li></ul></li></ul></li><li><span><a href="#Profile-InspectionRun-duration" data-toc-modified-id="Profile-InspectionRun-duration-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Profile InspectionRun duration</a></span></li><li><span><a href="#Plot-InspectionRun-duration" data-toc-modified-id="Plot-InspectionRun-duration-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Plot InspectionRun duration</a></span></li><li><span><a href="#Library-usage" data-toc-modified-id="Library-usage-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>Library usage</a></span></li></ul></div>

# %% [markdown]
# # Amun InspectionRun Analysis

# %% [markdown]
# **Introduction**
#
# The goal of this notebook is to show the behaviour of micro-benchmarks tested on the selected software/hardware architecture. This notebook will evaluate the feasibility and accuracy of the tests in order to prove or discard the possibility of running benchmarks tests for ML applications. 
#
# The analysis consider ~300 inspection jobs obtained considering:
#
# - same libraries
# - same versions
# - same environment
#
# The inputs for test are shown in section 2.

# %% [markdown]
# ---

# %% [markdown]
# ## Retrieve inspection jobs from Ceph

# %%
%env THOTH_DEPLOYMENT_NAME     thoth-core-upshift-stage
%env THOTH_CEPH_BUCKET         thoth
%env THOTH_CEPH_BUCKET_PREFIX  data/thoth
%env THOTH_S3_ENDPOINT_URL     https://s3.upshift.redhat.com/

# %%
from thoth.storages import InspectionResultsStore

inspection_store = InspectionResultsStore()
inspection_store.connect()

# %%
doc_id, doc = next(inspection_store.iterate_results())  # sample

# build log is unnecessary for our purposes and it is demanding to display it
doc['build_log'] = None
doc

# %% [markdown]
# ---

# %% [markdown]
# ## Describe the structure of an inspection job result

# %%
import pandas as pd
pd.set_option('max_colwidth', 800)


# %%
def extract_structure_json(input_json, upper_key: str, level: int, json_structure):
    """Convert a json file structure into a list with rows showing tree depths, keys and values"""
    level += 1
    for key in input_json.keys():
        if type(input_json[key]) is dict:
            json_structure.append([level, upper_key, key, [k for k in input_json[key].keys()]])
            
            extract_structure_json(input_json[key], f"{upper_key}__{key}", level, json_structure)
        else:
            json_structure.append([level, upper_key, key, input_json[key]])
            
    return json_structure

def filter_dfs(df_s, filter_df):
    """Filter the specific dataframe created for a certain key, combination of keys or for a tree depth"""
    if type(filter_df) is str:
        available_keys = set(df_s["Current_key"].values)
        available_combined_keys = set(df_s["Upper_keys"].values)
        
        if filter_df in available_keys:
            ndf = df_s[df_s["Current_key"].str.contains(f"^{filter_df}$", regex=True)]
            
        elif filter_df in available_combined_keys:
            ndf = df_s[df_s["Upper_keys"].str.contains(f"{filter_df}$", regex=True)]
        else:
            print("The key is not in the json")
            ndf = "". join([f"The available keys are (WARNING: Some of the keys have no leafs):{available_keys} ", f"The available combined keys are: {available_combined_keys}"])
    elif type(filter_df) is int:
        max_depth = df_s["Tree_depth"].max()
        if filter_df <= max_depth:
            ndf = df_s[df_s["Tree_depth"] == filter_df]
        else:
            ndf = f"The maximum tree depth available is: {max_depth}"
    return ndf


# %% [markdown]
# We can take a look at the inspection job structure from the point of view of the tree depth, considering a key or a combination of keys in order to understand the common inputs for all inspections.

# %%
df_structure = pd.DataFrame(extract_structure_json(doc, "", 0, []))
df_structure.columns = ["Tree_depth", "Upper_keys", "Current_key", "Value"]

# %%
filter_dfs(df_structure, 1)

# %% [markdown]
# Check hardware info

# %%
filter_dfs(df_structure, "__job_log__hwinfo")

# %% [markdown]
# Check CPU information

# %%
filter_dfs(df_structure, "__job_log__hwinfo__cpu")

# %% [markdown]
# Check Platform information

# %%
filter_dfs(df_structure, "__job_log__hwinfo__platform")

# %% [markdown]
# Check which libraries are used.

# %%
for package in filter_dfs(df_structure, "__specification__python__requirements_locked__default")["Current_key"].values:
    dfp = filter_dfs(df_structure, f"__specification__python__requirements_locked__default__{package}")
    print("{:15}  {}".format(package, dfp[dfp["Current_key"].str.contains("version")]["Value"].values[0]))

# %% [markdown]
# ---

# %% [markdown]
# ## Mapping InspectionRun JSON to pandas DataFrame

# %% {"require": ["notebook/js/codecell"]}
import pandas as pd

from pandas_profiling import ProfileReport as profile
from pandas.io.json import json_normalize

# %%
inspection_results = []

for document_id, document in inspection_store.iterate_results():
    # pop build logs to save some memory (not necessary for now)
    document['build_log'] = None
    
    inspection_results.append(document)

# %%
df = json_normalize(inspection_results, sep = ".")  # each row resembles InspectionResult

# %% [markdown]
# ### Feature importance analysis
#
# For the purposes of the performance analysis we take under consideration the impact of a variable on the performance score, the variance of the features is therefore an important indicator. We can assume that the more variance feature evinces, the higher is its impact on the performance measure stability.
#
# We can perform profiling as the first stage of this analysis to identify constants which won't affect the prediction.

# %%
f"The original DataFrame contains  {len(df.columns)}  columns"

# %% [markdown]
# These are the top-level keys:

# %%
inspection_results[0].keys()

# %% [markdown]
# #### Status

# %% {"require": ["base/js/events", "datatables.net", "d3", "jupyter-datatables"]}
df_status = df.filter(regex='status')

date_columns = df_status.filter(regex="started_at|finished_at").columns
for col in date_columns:
    df_status[col] = df[col].apply(pd.to_datetime)

# %%
p = profile(df_status)
p

# %% [markdown]
# According to the profiling, we can drop the values with the constant value:

# %%
rejected = p.description_set['variables'].query("distinct_count <= 1 & type != 'UNSUPPORTED'")
rejected

# %%
df.drop(rejected.index, axis=1, inplace=True)

# %% [markdown]
# #### Specification

# %%
df_spec = df.filter(regex='specification')

# %%
p = profile(df_spec)
p

# %%
rejected = p.description_set['variables'].query("distinct_count <= 1 & type != 'UNSUPPORTED'")
rejected

# %% [markdown]
# exclude versions, we might wanna use them later on

# %%
rejected = rejected.filter(regex="^((?!version).)*$", axis=0)
rejected

# %%
df.drop(rejected.index, axis=1, inplace=True)

# %% [markdown]
# #### Job log

# %%
df_job = df.filter(regex='job_log')

# %%
p = profile(df_job)
p

# %%
rejected = p.description_set['variables'].query("distinct_count <= 1 & type != 'UNSUPPORTED'")
    
rejected

# %%
df.drop(rejected.index, axis=1, inplace=True)

# %% [markdown]
# ---

# %%
from typing import Callable, Dict, List, Tuple, Union

def process_inspection_results(
    inspection_results: List[dict],
    exclude: Union[list, set] = None,
    apply: List[Tuple] = None,
    verbose: bool = False
) -> pd.DataFrame:
    """Process inspection result into pd.DataFrame."""
    if not inspection_results:
        return ValueError("Empty iterable provided.")
    
    exclude = exclude or []
    apply = apply or ()
    
    df = json_normalize(inspection_results, sep = "__")  # each row resembles InspectionResult
    
    if len(df) <= 1:
        return df
    
    for regex, func in apply:
        for col in df.filter(regex=regex).columns:
            df[col] = df[col].apply(func)
    
    keys = [k for k in inspection_results[0] if not k in exclude]
    for k in keys:
        if k in exclude:
            continue
        d = df.filter(regex=k)
        p = profile(d)
        
        rejected = p.description_set['variables'] \
            .query("distinct_count <= 1 & type != 'UNSUPPORTED'") \
            .filter(regex="^((?!version).)*$", axis=0)  # explicitly include versions
        
        if verbose:
            print("Rejected columns: ", rejected.index)
        
        df.drop(rejected.index, axis=1, inplace=True)
        
    df = df \
        .eval("status__job__duration   = status__job__finished_at   - status__job__started_at", engine='python') \
        .eval("status__build__duration = status__build__finished_at - status__build__started_at", engine='python')
        
    return df

# %%
df = process_inspection_results(
    inspection_results,
    exclude=['build_log', 'created', 'inspection_id'],
    apply=[
        ("created|started_at|finished_at", pd.to_datetime)
    ]
)

# %% [markdown]
# ---

# %% [markdown]
# ## Profile InspectionRun duration

# %%
df_duration = df.filter(like='duration') \
    .rename(columns=lambda s: s.replace("status__", "").replace("__", "_")) \
    .apply(lambda ts: pd.to_timedelta(ts).dt.total_seconds())

# %%
p = profile(df_duration)
p

# %%
stats = p.description_set['variables'].drop(["histogram", "mini_histogram"], axis=1)
stats

# %% [markdown]
# ## Plot InspectionRun duration

# %%
import numpy as np
import plotly

from plotly import graph_objs as go
from plotly.offline import iplot, init_notebook_mode

init_notebook_mode()

# %%
import cufflinks as cf

cf.go_offline()

# %% [markdown]
# Make sure that the versions are constant

# %%
df.filter(regex='python.*version').drop_duplicates()

# %% [markdown]
# Visualize statistics

# %%
fig = df_duration.iplot(
    kind='box',
    title="InspectionRun duration",
    yTitle="duration [s]",
    asFigure=True,
)

# %%
df_duration \
    .iplot(
    fill='tonexty',
    kind='scatter',
    title="InspectionRun duration",
    yTitle="duration [s]"
)

# %%
df_duration = df_duration \
    .eval("job_duration_mean           = job_duration.mean()", engine='python') \
    .eval("build_duration_mean         = build_duration.mean()", engine='python') \
    .eval("job_duration_upper_bound    = job_duration + job_duration.std()", engine='python') \
    .eval("job_duration_lower_bound    = job_duration - job_duration.std()", engine='python') \
    .eval("build_duration_upper_bound  = build_duration + build_duration.std()", engine='python') \
    .eval("build_duration_lower_bound  = build_duration - build_duration.std()", engine='python')

upper_bound = go.Scatter(
    name='Upper Bound',
    x=df_duration.index,
    y=df_duration.job_duration_upper_bound,
    mode='lines',
    marker=dict(color="lightgray"),
    line=dict(width=0),
    fillcolor='rgba(68, 68, 68, 0.3)',
    fill='tonexty' )

trace = go.Scatter(
    name='Duration',
    x=df_duration.index,
    y=df_duration.job_duration,
    mode='lines',
    line=dict(color='rgb(31, 119, 180)'),
    fillcolor='rgba(68, 68, 68, 0.3)',
    fill='tonexty' )

lower_bound = go.Scatter(
    name='Lower Bound',
    x=df_duration.index,
    y=df_duration.job_duration_lower_bound,
    marker=dict(color="lightgray"),
    line=dict(width=0),
    mode='lines')

data = [lower_bound, trace, upper_bound]

m = stats.loc['job_duration']['mean']
layout = go.Layout(
    yaxis=dict(title='duration [s]'),
    shapes=[
        {
            'type': 'line',
            'x0': 0,
            'x1': len(df_duration.index),
            'y0': m,
            'y1': m,
            'line': {
                'color': 'red',
                'dash': 'longdash'
            }
        }
    ],
    title='InspectionRun job duration',
    showlegend = False)

fig = go.Figure(data=data, layout=layout)

iplot(fig, filename='pandas-time-series-error-bars')

# %%
bins = np.lib.histograms._hist_bin_auto(df_duration.job_duration.values, None)

df_duration.job_duration.iplot(
    title="InspectionRun job distribution",
    xTitle="duration [s]",
    yTitle="count",
    kind='hist',
    bins=int(np.ceil(bins))
)

# %% [markdown]
# ---

# %% [markdown]
# ## Library usage


# %%
# -*- coding: utf-8 -*-

import plotly.graph_objs as go
import plotly.offline as py
import cufflinks as cf

import numpy as np
import pandas as pd

from pandas_profiling import ProfileReport as profile

from typing import Any, List, Tuple, Union

"""Thoth InspectionRun dashboard app."""

cf.go_offline()
pd.set_option('precision', 4)
pd.set_option('colheader_justify', 'center')


def create_duration_dataframe(inspection_df: pd.DataFrame):
    """Compute statistics and duration DataFrame."""
    if len(inspection_df) <= 0:
        raise ValueError("Empty DataFrame provided")

    try:
        inspection_df.drop("build_log", axis=1, inplace=True)
    except KeyError:
        pass

    data = (
        inspection_df.filter(like="duration")
        .rename(columns=lambda s: s.replace("status__", "").replace("__", "_"))
        .apply(lambda ts: pd.to_timedelta(ts).dt.total_seconds())
    )
    data = (
        data
        .eval("job_duration_mean           = job_duration.mean()", engine="python")
        .eval("job_duration_upper_bound    = job_duration + job_duration.std()", engine="python")
        .eval("job_duration_lower_bound    = job_duration - job_duration.std()", engine="python")
        .eval("build_duration_mean         = build_duration.mean()", engine="python")
        .eval("build_duration_upper_bound  = build_duration + build_duration.std()", engine="python")
        .eval("build_duration_lower_bound  = build_duration - build_duration.std()", engine="python")
    )

    return data.round(4)


def create_duration_box(data: pd.DataFrame, columns: List[str] = None, **kwargs):
    """Create duration Box plot."""
    columns = columns or []

    figure = data[columns].iplot(
        kind="box", title=kwargs.pop("title", "InspectionRun duration"), yTitle="duration [s]", asFigure=True
    )

    return figure


def create_duration_scatter(data: pd.DataFrame, col: str, **kwargs):
    """Create duration Scatter plot."""
    df_duration = (
        data[[col]]
        .eval(f"upper_bound = {col} + {col}.std()", engine="python")
        .eval(f"lower_bound = {col} - {col}.std()", engine="python")
    )

    upper_bound = go.Scatter(
        name="Upper Bound",
        x=df_duration.index,
        y=df_duration.upper_bound,
        mode="lines",
        marker=dict(color="lightgray"),
        line=dict(width=0),
        fillcolor="rgba(68, 68, 68, 0.3)",
        fill="tonexty",
    )

    trace = go.Scatter(
        name="Duration",
        x=df_duration.index,
        y=df_duration[col],
        mode="lines",
        line=dict(color="rgb(31, 119, 180)"),
        fillcolor="rgba(68, 68, 68, 0.3)",
        fill="tonexty",
    )

    lower_bound = go.Scatter(
        name="Lower Bound",
        x=df_duration.index,
        y=df_duration.lower_bound,
        marker=dict(color="lightgray"),
        line=dict(width=0),
        mode="lines",
    )

    data = [lower_bound, trace, upper_bound]
    m = df_duration[col].mean()

    layout = go.Layout(
        yaxis=dict(title="duration [s]"),
        shapes=[
            {
                "type": "line",
                "x0": 0,
                "x1": len(df_duration.index),
                "y0": m,
                "y1": m,
                "line": {"color": "red", "dash": "longdash"},
            }
        ],
        title=kwargs.pop("title", "InspectionRun duration"),
        showlegend=False,
    )

    fig = go.Figure(data=data, layout=layout)

    return fig


def create_duration_histogram(data: pd.DataFrame, columns: List[str] = None, bins: int = None, **kwargs):
    """Create duration histogram."""
    columns = columns or data.columns

    if not bins:
        bins = np.max([np.lib.histograms._hist_bin_auto(data[col].values, None) for col in columns])

    figure = data[columns].iplot(
        title=kwargs.pop("title", "InspectionRun distribution"),
        xTitle="duration [s]",
        yTitle="count",
        kind="hist",
        bins=int(np.ceil(bins)),
        asFigure=True,
    )

    return figure



# %%
df_duration = create_duration_dataframe(df)

# %%
fig = create_duration_box(df_duration, ['build_duration', 'job_duration'])

py.iplot(fig)

# %%
fig = create_duration_scatter(df_duration, 'job_duration', title="InspectionRun job duration")

py.iplot(fig)

# %%
fig = create_duration_scatter(df_duration, 'build_duration', title="InspectionRun build duration")

py.iplot(fig)

# %%
fig = create_duration_histogram(df_duration, ['job_duration'])

py.iplot(fig)
