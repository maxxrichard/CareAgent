# Databricks notebook source
# MAGIC %md
# MAGIC # CareAgent – evaluation dashboard
# MAGIC Runs the labelled query set across LLM backends, logs to MLflow, appends per-case rows
# MAGIC to a Delta table and renders the SQL analytics. Attach to a cluster with the repo
# MAGIC installed (`%pip install -e .[databricks]`) and provider keys in secret scopes.

# COMMAND ----------

# MAGIC %pip install -e .[databricks,embeddings] -q

# COMMAND ----------

import os

os.environ["ANTHROPIC_API_KEY"] = dbutils.secrets.get("careagent", "anthropic_api_key")  # noqa: F821
os.environ["OPENAI_API_KEY"] = dbutils.secrets.get("careagent", "openai_api_key")  # noqa: F821
os.environ["MLFLOW_TRACKING_URI"] = "databricks"
os.environ["MLFLOW_EXPERIMENT_NAME"] = "/Shared/careagent"
os.environ["CAREAGENT_DELTA_TABLE"] = "main.careagent.eval_results"

# COMMAND ----------

from careagent.evaluation.runner import run_evaluation  # noqa: E402

summary = run_evaluation(backends=["anthropic", "openai", "ollama"], use_judge=True)
display(summary["backends"])  # noqa: F821

# COMMAND ----------

from careagent.evaluation.databricks import analytics_queries  # noqa: E402

for name, query in analytics_queries(os.environ["CAREAGENT_DELTA_TABLE"]).items():
    print(f"== {name}")
    display(spark.sql(query))  # noqa: F821
