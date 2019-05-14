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
# <div class="toc"><ul class="toc-item"><li><span><a href="#Retrieve-inspection-jobs-from-Ceph" data-toc-modified-id="Retrieve-inspection-jobs-from-Ceph-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Retrieve inspection jobs from Ceph</a></span></li><li><span><a href="#Describe-the-structure-of-an-inspection-job-result" data-toc-modified-id="Describe-the-structure-of-an-inspection-job-result-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Describe the structure of an inspection job result</a></span></li><li><span><a href="#Mapping-InspectionRun-JSON-to-pandas-DataFrame" data-toc-modified-id="Mapping-InspectionRun-JSON-to-pandas-DataFrame-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Mapping InspectionRun JSON to pandas DataFrame</a></span></li><li><span><a href="#Plot-InspectionRun" data-toc-modified-id="Plot-InspectionRun-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Plot InspectionRun</a></span></li><li><span><a href="#Wrapper" data-toc-modified-id="Wrapper-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Wrapper</a></span></li></ul></div>

# %% [markdown]
# # Amun InspectionRun Analysis

# %% [markdown]
# **Introduction**
#
# The goal of this notebook is to...

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
#Create a dataframe from the json as it is provided
dft = pd.DataFrame(doc)
dft


# %%
def extract_structure_json(input_json, upper_key: str, level: int, json_structure):
    """Convert a json file structure into a nested list showing keys depths"""
    level += 1
    for key in input_json.keys():
        if type(input_json[key]) is dict:
            json_structure.append([level, upper_key, key, [k for k in input_json[key].keys()]])
            
            extract_structure_json(input_json[key], f"{upper_key}__{key}", level, json_structure)
        else:
            json_structure.append([level, upper_key, key, input_json[key]])
    return json_structure

def filter_dataframe(json_pandas, filter_df):
    """Filter the dataframe for a certain key, combination of keys or for a tree depth"""
    if type(filter_df) is str:
        available_keys = set(df["Current_key"].values)
        available_combined_keys = set(df["Upper_keys"].values)
        
        if filter_df in available_keys:
            ndf = df[df["Current_key"].str.contains(filter_df)]
            
        elif filter_df in available_combined_keys:
            ndf = df[df["Upper_keys"].str.contains(filter_df)]
        else:
            print("The key is not in the json")
            ndf = "". join([f"The available keys are (WARNING: Some of the keys have no leafs): {available_keys} ", f"The available combined keys are: {available_combined_keys}"])
            
    elif type(filter_df) is int:
        max_depth = df["Tree_depth"].max()
        if filter_df <= max_depth:
            ndf = df[df["Tree_depth"] == filter_df]
        else:
            ndf = f"The maximum tree depth available is: {max_depth}"
    return ndf


# %%
#Create the dataframe with the pre-processing function
df = pd.DataFrame(extract_structure_json(doc,"", 0, []))
df.columns = ["Tree_depth", "Upper_keys", "Current_key", "Value"]

# %% [markdown]
# We can take a look at the inspection job structure from the point of view of the tree depth, considering a key or a combination of keys.

# %%
filter_dataframe(df, 1)

# %%
filter_dataframe(df, 2)

# %%
filter_dataframe(df, 3)

# %%
filter_dataframe(df, 4)

# %%
filter_dataframe(df, 5)

# %%
filter_dataframe(df, 6)

# %% [markdown]
# ---

# %% [markdown]
# ## Mapping InspectionRun JSON to pandas DataFrame

# %%

# %% [markdown]
# ---

# %% [markdown]
# ## Plot InspectionRun

# %%

# %% [markdown]
# ---

# %% [markdown]
# ## Wrapper

