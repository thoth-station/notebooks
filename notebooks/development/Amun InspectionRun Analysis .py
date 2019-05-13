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
#     display_name: thoth-notebooks
#     language: python
#     name: thoth-notebooks
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

# %%
from thoth.storages import InspectionResultsStore

inspection_store = InspectionResultsStore()
inspection_store.connect()

# %% [markdown]
# ---

# %% [markdown]
# ## Describe the structure of an inspection job result

# %%

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
