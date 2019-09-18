Jupyter notebooks templates for Thoth
--------------------------------------

RUN THOTH NOTEBOOK TEMPLATE
==============================

1. Install `papermill <https://github.com/nteract/papermill/>`_:

.. code-block:: console

  $ pipenv install papermill


2. Select TEMPLATE inputs

Using .yaml for example

.. code-block:: console

    cat input_notebook.yaml 

.. code-block:: console

    identifier_inspection:
        - "64-matrix-size"
        - "128-matrix-size"
        - "256-matrix-size"
        - "test-ms"  
        - "1024-matrix-size"
        - "2048-matrix-size"
        - "4096-matrix-size"  
    static_figure: True
    limit_results: True


More details on how to give inputs in `papermill <https://github.com/nteract/papermill/>`_.


3. Run the notebook

.. code-block:: console

  $ pipenv run papermill 'path/to/input.ipynb' 'path/to/output.ipynb' -f input_notebook.yaml


THOTH NOTEBOOK TEMPLATE INPUTS
==============================

--> Inspection_jobs_analysis_TEMPLATE.ipynb 

    :identifier_inspection: List of inspection identifiers to filter the inspection document ids
        - "64-matrix-size"
        - "128-matrix-size"
    :static_figure: Bool to produce static or interactive plots
    :limit_results: Bool to limit the analysis to few inspection ids (used for testing notebook outputs)