

### Patient Cohort Building with NLP and Knowledge Graphs
Cohort building is an essential part of patient analytics. Defining which patients belong to a cohort, testing the sensitivity of various inclusion and exclusion criteria on sample size, building a control cohort with propensity score matching techniques: These are just some of the processes that healthcare and life sciences researchers live day in and day out, and that's unlikely to change anytime soon. What is changing is the underlying data, the complexity of clinical criteria, and the dynamism demanded by the industry.
<img src="https://raw.githubusercontent.com/iamvarol/blogposts/main/databricks/images/db_viz.png">
While tools exist for building patient cohorts based on structured data from EHR data or claims, their practical utility is limited. More and more, cohort building in healthcare and life sciences requires criteria extracted from unstructured and semi-structured clinical documentation with Natural Language Processing (NLP) pipelines. Making this a reality requires a seamless combination of three technologies:
(1) a platform that scales for computationally-intensive calculations of massive real world datasets,
(2) an accurate NLP library & healthcare-specific models to extract and relate entities from medical documents, and
(3) a knowledge graph toolset, able to represent the relationships between a network of entities.
In this solution accelerator from John Snow Labs and Databricks, we bring all of this together in the Lakehouse.
Read the blog [here](https://www.databricks.com/blog/2022/09/30/building-patient-cohorts-nlp-and-knowledge-graphs.html)



## Cluster Setup

**SparkNLP for healthcare**
To create a cluster with access to SparkNLP for healthcare models, follow [these steps](https://nlp.johnsnowlabs.com/docs/en/licensed_install#install-on-databricks) or run the `RUNME` notebook in this repository to create a cluster with the models.
**Neo4j**
To build the knowledge graph, we use [neo4j](https://neo4j.com/product/graph-data-science/?utm_program=na-prospecting&utm_source=google&utm_medium=cpc&utm_campaign=na-search-offers&utm_adgroup=dynamic&utm_content=dynamic&utm_placement=&utm_network=g&gclid=Cj0KCQiAmaibBhCAARIsAKUlaKRcpYyWONbT5eDtjpKLAGisE6vI6CEMaDkbpoS_khm5L2BrqPzVnmoaArHiEALw_wcB) graph database. To learn more on how to stand up a graph database within your cloud environment, see the relevant depplyment guide:
  - [AWS](https://neo4j.com/docs/operations-manual/current/cloud-deployments/neo4j-aws/)
  - [Azure](https://neo4j.com/docs/operations-manual/current/cloud-deployments/neo4j-azure/)
  - [GCP](https://neo4j.com/docs/operations-manual/current/cloud-deployments/neo4j-gcp/)
  
To connect to your `neo4j` graph database from databricks, see [Neo4j on databricks](https://docs.databricks.com/external-data/neo4j.html)




## License
Copyright / License info of the notebook. Copyright [2021] the Notebook Authors.  The source in this notebook is provided subject to the [Apache 2.0 License](https://spdx.org/licenses/Apache-2.0.html).  All included or referenced third party libraries are subject to the licenses set forth below.
|Library Name|Library License|Library License URL|Library Source URL|
| :-: | :-:| :-: | :-:|
|Pandas |BSD 3-Clause License| https://github.com/pandas-dev/pandas/blob/master/LICENSE | https://github.com/pandas-dev/pandas|
|Numpy |BSD 3-Clause License| https://github.com/numpy/numpy/blob/main/LICENSE.txt | https://github.com/numpy/numpy|
|Apache Spark |Apache License 2.0| https://github.com/apache/spark/blob/master/LICENSE | https://github.com/apache/spark/tree/master/python/pyspark|
|BeautifulSoup|MIT License|https://www.crummy.com/software/BeautifulSoup/#Download|https://www.crummy.com/software/BeautifulSoup/bs4/download/|
|Requests|Apache License 2.0|https://github.com/psf/requests/blob/main/LICENSE|https://github.com/psf/requests|
|Spark NLP Display|Apache License 2.0|https://github.com/JohnSnowLabs/spark-nlp-display/blob/main/LICENSE|https://github.com/JohnSnowLabs/spark-nlp-display|
|Spark NLP |Apache License 2.0| https://github.com/JohnSnowLabs/spark-nlp/blob/master/LICENSE | https://github.com/JohnSnowLabs/spark-nlp|
|Spark NLP for Healthcare|[Proprietary license - John Snow Labs Inc.](https://www.johnsnowlabs.com/spark-nlp-health/) |NA|NA|
|Author|
|-|
|Databricks Inc.|
|John Snow Labs Inc.|




## Disclaimers
Databricks Inc. (“Databricks”) does not dispense medical, diagnosis, or treatment advice. This Solution Accelerator (“tool”) is for informational purposes only and may not be used as a substitute for professional medical advice, treatment, or diagnosis. This tool may not be used within Databricks to process Protected Health Information (“PHI”) as defined in the Health Insurance Portability and Accountability Act of 1996, unless you have executed with Databricks a contract that allows for processing PHI, an accompanying Business Associate Agreement (BAA), and are running this notebook within a HIPAA Account.  Please note that if you run this notebook within Azure Databricks, your contract with Microsoft applies.
