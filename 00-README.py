# Databricks notebook source
# MAGIC %md
# MAGIC ### Patient Cohort Building with NLP and Knowledge Graphs
# MAGIC 
# MAGIC Cohort building is an essential part of patient analytics. Defining which patients belong to a cohort, testing the sensitivity of various inclusion and exclusion criteria on sample size, building a control cohort with propensity score matching techniques: These are just some of the processes that healthcare and life sciences researchers live day in and day out, and that's unlikely to change anytime soon. What is changing is the underlying data, the complexity of clinical criteria, and the dynamism demanded by the industry.
# MAGIC <br></br>
# MAGIC <img src="https://raw.githubusercontent.com/iamvarol/blogposts/main/databricks/images/db_viz.png" width=60%>
# MAGIC <br></br>
# MAGIC While tools exist for building patient cohorts based on structured data from EHR data or claims, their practical utility is limited. More and more, cohort building in healthcare and life sciences requires criteria extracted from unstructured and semi-structured clinical documentation with Natural Language Processing (NLP) pipelines. Making this a reality requires a seamless combination of three technologies:
# MAGIC 
# MAGIC 1. a platform that scales for computationally-intensive calculations of massive real world datasets,
# MAGIC 2. an accurate NLP library & healthcare-specific models to extract and relate entities from medical documents, and
# MAGIC 3. a knowledge graph toolset, able to represent the relationships between a network of entities.
# MAGIC 
# MAGIC In this solution accelerator from John Snow Labs and Databricks, we bring all of this together in the Lakehouse.
# MAGIC Read the blog [here](https://www.databricks.com/blog/2022/09/30/building-patient-cohorts-nlp-and-knowledge-graphs.html)

# COMMAND ----------

# DBTITLE 1,Cluster Setup
# MAGIC %md
# MAGIC **SparkNLP for healthcare**
# MAGIC To create a cluster with access to SparkNLP for healthcare models, follow [these steps](https://nlp.johnsnowlabs.com/docs/en/licensed_install#install-on-databricks) or run the `RUNME` notebook in this repository to create a cluster with the models.
# MAGIC 
# MAGIC **Neo4j**
# MAGIC To build the knowledge graph, we use [neo4j](https://neo4j.com/product/graph-data-science/?utm_program=na-prospecting&utm_source=google&utm_medium=cpc&utm_campaign=na-search-offers&utm_adgroup=dynamic&utm_content=dynamic&utm_placement=&utm_network=g&gclid=Cj0KCQiAmaibBhCAARIsAKUlaKRcpYyWONbT5eDtjpKLAGisE6vI6CEMaDkbpoS_khm5L2BrqPzVnmoaArHiEALw_wcB) graph database. 
# MAGIC 
# MAGIC To run this solution accelerator, you can use [neo4j community edition](https://neo4j.com/cloud/platform/aura-graph-database/?ref=nav-get-started-cta) to stand up a sandbox environment and connect to it using your credentials. 
# MAGIC  
# MAGIC To learn more on how to stand up a graph database within your cloud environment, see the relevant depplyment guide:
# MAGIC   - [AWS](https://neo4j.com/docs/operations-manual/current/cloud-deployments/neo4j-aws/)
# MAGIC   - [Azure](https://neo4j.com/docs/operations-manual/current/cloud-deployments/neo4j-azure/)
# MAGIC   - [GCP](https://neo4j.com/docs/operations-manual/current/cloud-deployments/neo4j-gcp/)
# MAGIC   
# MAGIC And to connect to your `neo4j` graph database from databricks, see [Neo4j on databricks](https://docs.databricks.com/external-data/neo4j.html)

# COMMAND ----------

# MAGIC %md
# MAGIC ## License
# MAGIC Copyright / License info of the notebook. Copyright [2021] the Notebook Authors.  The source in this notebook is provided subject to the [Apache 2.0 License](https://spdx.org/licenses/Apache-2.0.html).  All included or referenced third party libraries are subject to the licenses set forth below.
# MAGIC 
# MAGIC |Library Name|Library License|Library License URL|Library Source URL|
# MAGIC | :-: | :-:| :-: | :-:|
# MAGIC |Pandas |BSD 3-Clause License| https://github.com/pandas-dev/pandas/blob/master/LICENSE | https://github.com/pandas-dev/pandas|
# MAGIC |Numpy |BSD 3-Clause License| https://github.com/numpy/numpy/blob/main/LICENSE.txt | https://github.com/numpy/numpy|
# MAGIC |Apache Spark |Apache License 2.0| https://github.com/apache/spark/blob/master/LICENSE | https://github.com/apache/spark/tree/master/python/pyspark|
# MAGIC |Requests|Apache License 2.0|https://github.com/psf/requests/blob/main/LICENSE|https://github.com/psf/requests|
# MAGIC |Spark NLP Display|Apache License 2.0|https://github.com/JohnSnowLabs/spark-nlp-display/blob/main/LICENSE|https://github.com/JohnSnowLabs/spark-nlp-display|
# MAGIC |Spark NLP |Apache License 2.0| https://github.com/JohnSnowLabs/spark-nlp/blob/master/LICENSE | https://github.com/JohnSnowLabs/spark-nlp|
# MAGIC |Spark NLP for Healthcare|[Proprietary license - John Snow Labs Inc.](https://www.johnsnowlabs.com/spark-nlp-health/) |NA|NA|
# MAGIC |Neo4j|GPL v3.|https://github.com/neo4j/neo4j/blob/5.1/LICENSE.txt|https://github.com/neo4j/neo4j|
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC |Author|
# MAGIC |-|
# MAGIC |Databricks Inc.|
# MAGIC |John Snow Labs Inc.|

# COMMAND ----------

# MAGIC %md
# MAGIC ## Disclaimers
# MAGIC Databricks Inc. (“Databricks”) does not dispense medical, diagnosis, or treatment advice. This Solution Accelerator (“tool”) is for informational purposes only and may not be used as a substitute for professional medical advice, treatment, or diagnosis. This tool may not be used within Databricks to process Protected Health Information (“PHI”) as defined in the Health Insurance Portability and Accountability Act of 1996, unless you have executed with Databricks a contract that allows for processing PHI, an accompanying Business Associate Agreement (BAA), and are running this notebook within a HIPAA Account.  Please note that if you run this notebook within Azure Databricks, your contract with Microsoft applies.
