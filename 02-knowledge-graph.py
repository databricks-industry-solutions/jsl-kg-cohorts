# Databricks notebook source
# MAGIC %md
# MAGIC # Patient Cohort Building with NLP and Knowledge Graphs

# COMMAND ----------

# MAGIC %md
# MAGIC In this notebook, we will build a Neo4j Clinical Knowledge Graph (KG) from the output of a Spark NLP pipeline that contains NER (named entity recognition) and RE (relation extraction) pretrained models. After creating the knowledge graph, we will query the KG to get some insightful results.
# MAGIC 
# MAGIC To obtain the visualizations included below, run the provided queries (just the query, without quotation marks) in Neo4j Browser. This can be accessed from the Instances screen in Neo4j via the `>_ Query` tab. The password for the instance is required to access the graph database. 
# MAGIC 
# MAGIC Please notice that the nodes and the relations color codes might differ from the ones provided here, as they depend on the individual browser settings. Also, you may obtain slightly different outputs, these are related to the versions of the pretrained models used in the Spark NLP pipelines.  

# COMMAND ----------

# MAGIC %md
# MAGIC [Cluster Setup](https://nlp.johnsnowlabs.com/docs/en/licensed_install#install-on-databricks)

# COMMAND ----------

# MAGIC %pip install neo4j tqdm

# COMMAND ----------

# MAGIC %md
# MAGIC ## Creation of the Knowledge Graph

# COMMAND ----------

# MAGIC %md
# MAGIC ### Neo4j Connection

# COMMAND ----------

from neo4j import GraphDatabase

import time
from tqdm import tqdm
import pandas as pd
import json

# COMMAND ----------

notes_path='/FileStore/HLS/jsl_kg/data/'
patient_df = pd.read_csv(f'/dbfs{notes_path}data.csv', sep=';')
patient_df

# COMMAND ----------

patient_demographics = patient_df[['subject_id', 'gender', 'dateOfBirth']].drop_duplicates().reset_index(drop=True)
patient_demographics

# COMMAND ----------

filename = 'posology_RE_rxnorm_w_drug_resolutions.csv'
folderdir = f'/FileStore/HLS/jsl_kg/data/'
pos_RE_result =  pd.read_csv(f'/dbfs{folderdir+filename}')
pos_RE_result.head()

# COMMAND ----------

filename = 'ner_jsl_slim_results.csv'
folderdir = f'/FileStore/HLS/jsl_kg/data/'
ner_DF_result = pd.read_csv(f'/dbfs{folderdir+filename}')
ner_DF_result.head()

# COMMAND ----------

class Neo4jConnection:
    def __init__(self, uri, user, pwd):
        self.__uri = uri
        self.__user = user
        self.__pwd = pwd
        self.__driver = None
        try:
            self.__driver = GraphDatabase.driver(self.__uri, auth=(self.__user, self.__pwd))
        except Exception as e:
            print("Failed to create the driver:", e)
        
    def close(self):
        if self.__driver is not None:
            self.__driver.close()
        
    def query(self, query, parameters=None, db=None):
        assert self.__driver is not None, "Driver not initialized!"
        session = None
        response = None
        try: 
            session = self.__driver.session(database=db) if db is not None else self.__driver.session() 
            response = list(session.run(query, parameters))
        except Exception as e:
            print("Query failed:", e)
        finally: 
            if session is not None:
                session.close()
        return response

# COMMAND ----------

# Credentials for Neo4j graph database

uri = dbutils.secrets.get("solution-accelerator-cicd","neo4j-uri") # replace with '<Neo4j Aura instance uri>' or set up this secret in your own workspace
pwd = dbutils.secrets.get("solution-accelerator-cicd","neo4j-password") # replace with '<Neo4j Aura instance password>' or set up this secret in your own workspace
user = dbutils.secrets.get("solution-accelerator-cicd","neo4j-user") # replace with '<Neo4j Aura instance user>' or set up this secret in your own workspace

# Establish the connection with Neo4j GDB
conn = Neo4jConnection(uri=uri, user=user , pwd=pwd)

# COMMAND ----------

# MAGIC %md
# MAGIC **Creating constraints:**

# COMMAND ----------

conn.query('CREATE CONSTRAINT patients IF NOT EXISTS FOR (p:Patient) REQUIRE p.name IS UNIQUE;')
conn.query('CREATE CONSTRAINT rx_norm_codes IF NOT EXISTS FOR (rx:RxNorm) REQUIRE rx.code IS UNIQUE;')
conn.query('CREATE CONSTRAINT drugs IF NOT EXISTS FOR (drug:Drug) REQUIRE drug.name IS UNIQUE;')
conn.query('CREATE CONSTRAINT ners IF NOT EXISTS FOR (n:NER) REQUIRE n.name IS UNIQUE;')
conn.query('CREATE CONSTRAINT symptoms IF NOT EXISTS FOR (s:Symptom) REQUIRE s.name IS UNIQUE;')
conn.query('CREATE CONSTRAINT bodyParts IF NOT EXISTS FOR (bp:BodyPart) REQUIRE bp.name IS UNIQUE;')
conn.query('CREATE CONSTRAINT procedures IF NOT EXISTS FOR (p:Procedure) REQUIRE p.name IS UNIQUE;')
conn.query('CREATE CONSTRAINT tests IF NOT EXISTS FOR (t:Test) REQUIRE t.name IS UNIQUE;')
conn.query('CREATE CONSTRAINT dsds IF NOT EXISTS FOR (dsd:DSD) REQUIRE dsd.name IS UNIQUE;')

# COMMAND ----------

# MAGIC %md
# MAGIC **defining helper functions:**

# COMMAND ----------

def update_data(query, rows, batch_size = 10000):
    total = 0
    batch = 0
    start = time.time()
    result = None
    while batch * batch_size < len(rows):
        res = conn.query(query, parameters={'rows': rows[batch*batch_size:(batch+1)*batch_size].to_dict('records')})
        total += res[0]['total']
        batch += 1
        result = {"total":total, "batches":batch, "time":time.time()-start}
        print(result)
    return result

# COMMAND ----------

def add_patients(rows, batch_size=10000):
    query = '''
    UNWIND $rows as row
    MERGE(p:Patient{name:row.subject_id}) 
    ON CREATE SET p.gender      = row.gender,
                  p.dateOfBirth = row.dateOfBirth

    WITH p
    MATCH (p)
    RETURN count(*) as total
    '''
    return update_data(query, rows, batch_size)

add_patients(patient_demographics)

# COMMAND ----------

def add_drugs_ners(rows, batch_size=1000):
    query = '''
    UNWIND $rows as row
    
    MERGE(p:Patient{name:row.subject_id}) 
    MERGE(rx:RxNorm{code:row.rxnorm_code})
    MERGE (p)-[:RXNORM_CODE{date:date(row.date)}]->(rx)
    
    MERGE (d:Drug{name:row.drug_resolution})
    MERGE (rx)-[:DRUG_GENERIC{date:date(row.date), patient_name:row.subject_id}]->(d)
    
    MERGE(n1:NER{name:row.chunk1}) ON CREATE SET n1.type=row.entity1
    MERGE(n2:NER{name:row.chunk2}) ON CREATE SET n2.type=row.entity2
    
    WITH *
    MATCH (d:Drug{name:row.drug_resolution}), (n1:NER{name:row.chunk1}), (n2:NER{name:row.chunk2})
    CALL apoc.create.relationship (d,row.entity1, {patient_name:row.subject_id, date:date(row.date)}, n1) YIELD rel as relx
    CALL apoc.create.relationship (d,row.entity2, {patient_name:row.subject_id, date:date(row.date)}, n2) YIELD rel as rely
    
    WITH d
    MATCH (d)
    RETURN count(*) as total  
    '''
    return update_data(query, rows, batch_size)
  
add_drugs_ners(pos_RE_result)

# COMMAND ----------

# MAGIC %md
# MAGIC **splitting dataframe into multiple dataframes by ner_label and creating nodes and relationships**

# COMMAND ----------

# spliting dataframe into multiple dataframe by ner_label
grouped        = ner_DF_result.groupby('ner_label')
df_symptom     = grouped.get_group('Symptom')
df_dsd         = grouped.get_group('Disease_Syndrome_Disorder')
df_test        = grouped.get_group('Test')
df_bodyPart    = grouped.get_group('Body_Part')
df_procedure   = grouped.get_group('Procedure')

# COMMAND ----------

def add_symptoms(rows, batch_size=500):
    query = '''
    UNWIND $rows as row
    MATCH(p:Patient{name:row.subject_id})
    MERGE(n:Symptom {name:row.chunk})
    MERGE (p)-[:IS_SYMPTOM{date:date(row.date)}]->(n)

    WITH n
    MATCH (n)
    RETURN count(*) as total  
    '''
    return update_data(query, rows, batch_size)
  
add_symptoms(df_symptom)

# COMMAND ----------

def add_dsds(rows, batch_size=500):
    query = '''
    UNWIND $rows as row
    MATCH(p:Patient{name:row.subject_id})
    MERGE(n:DSD {name:row.chunk})
    MERGE (p)-[:IS_DSD{date:date(row.date)}]->(n)

    WITH n
    MATCH (n)
    RETURN count(*) as total  
    '''
    return update_data(query, rows, batch_size)
  
add_dsds(df_dsd)

# COMMAND ----------

def add_tests(rows, batch_size=500):
    query = '''
    UNWIND $rows as row
    MATCH(p:Patient{name:row.subject_id})
    MERGE(n:Test {name:row.chunk})
    MERGE (p)-[:IS_TEST{date:date(row.date)}]->(n)

    WITH n
    MATCH (n)
    RETURN count(*) as total  
    '''
    return update_data(query, rows, batch_size)
  
add_tests(df_test)

# COMMAND ----------

def add_bodyParts(rows, batch_size=500):
    query = '''
    UNWIND $rows as row
    MATCH(p:Patient{name:row.subject_id})
    MERGE(n:BodyPart {name:row.chunk})
    MERGE (p)-[:IS_BODYPART{date:date(row.date)}]->(n)

    WITH n
    MATCH (n)
    RETURN count(*) as total
    '''
    return update_data(query, rows, batch_size)
  
add_bodyParts(df_bodyPart)

# COMMAND ----------

def add_procedures(rows, batch_size=500):
    query = '''
    UNWIND $rows as row
    MATCH(p:Patient{name:row.subject_id})
    MERGE(n:Procedure {name:row.chunk})
    MERGE (p)-[:IS_PROCEDURE{date:date(row.date)}]->(n)

    WITH n
    MATCH (n)
    RETURN count(*) as total  
    '''
    return update_data(query, rows, batch_size)
  
add_procedures(df_procedure)

# COMMAND ----------

query_string = '''
CALL db.labels() YIELD label
CALL apoc.cypher.run('MATCH (:`'+label+'`) RETURN count(*) as count',{}) YIELD value
RETURN label, value.count as size
'''
df_nodes = pd.DataFrame([dict(_) for _ in conn.query(query_string)])
df_nodes

# COMMAND ----------

query_string = '''
CALL db.relationshipTypes() YIELD relationshipType as type
CALL apoc.cypher.run('MATCH ()-[:`'+type+'`]->() RETURN count(*) as count',{}) YIELD value
RETURN type, value.count as size
'''
df_relationships = pd.DataFrame([dict(_) for _ in conn.query(query_string)])
df_relationships

# COMMAND ----------

# MAGIC %md
# MAGIC **database schema visualization**

# COMMAND ----------

# To get the following visualization run the query in the Neo4j Browser
query_string = '''
CALL db.schema.visualization()
'''


# COMMAND ----------

# MAGIC %md
# MAGIC <img src="https://raw.githubusercontent.com/iamvarol/blogposts/main/databricks/images/db_viz.png">

# COMMAND ----------

# MAGIC %md
# MAGIC ## Queries

# COMMAND ----------

# MAGIC %md
# MAGIC **patient 21153's prescriptions:**

# COMMAND ----------

patient_name = '21153'
query_part1 = 'MATCH (p:Patient)-[rel_rx]->(rx:RxNorm)-[rel_d]->(d:Drug)-[rel_n]->(n:NER) ' #  
query_part2 = f'WHERE p.name ={patient_name} AND rel_n.date=rel_rx.date AND rel_n.patient_name=p.name ' # 
query_part3 = '''RETURN DISTINCT
                 p.name as patient_name,
                 rel_rx.date as date,
                 d.name as drug_generic_name,  
                 rx.code as rxnorm_code,
                 COALESCE(n.name,'') +  "(" + COALESCE (type(rel_n), "") + ")" as details
                 '''
query_string = query_part1 + query_part2 + query_part3

df = pd.DataFrame([dict(_) for _ in conn.query(query_string)])
df = df.drop_duplicates(subset= ['patient_name', 'date', 'drug_generic_name'])
df = df.groupby(['patient_name', 'date', 'drug_generic_name', 'rxnorm_code']).agg(lambda x: ' '.join(x)).reset_index()
df

# COMMAND ----------

# MAGIC %md
# MAGIC **patient 21153's journey in medical records:symptoms, procedures, disease-syndrome-disorders, test, drugs & rxnorms**

# COMMAND ----------

patient_name = '21153'

query_part1 = f'MATCH (p:Patient)-[r1:IS_SYMPTOM]->(s:Symptom) WHERE p.name = {patient_name} '
query_part2 = '''
WITH DISTINCT p.name as patients, r1.date as dates, COLLECT(DISTINCT s.name) as symptoms, COUNT(DISTINCT s.name) as num_symptoms

MATCH (p:Patient)-[r2:IS_PROCEDURE]->(pr:Procedure)
WHERE p.name=patients AND r2.date = dates

WITH DISTINCT p.name as patients, r2.date as dates, COLLECT(DISTINCT pr.name) as procedures, COUNT(DISTINCT pr.name) as num_procedures, symptoms, num_symptoms
MATCH (p:Patient)-[r3:IS_DSD]->(_d:DSD) 
WHERE p.name=patients AND r3.date = dates

WITH DISTINCT p.name as patients, r3.date as dates, symptoms, num_symptoms, procedures, num_procedures,  COLLECT(DISTINCT _d.name) as dsds, COUNT(DISTINCT _d.name) as num_dsds
MATCH (p:Patient)-[r4:IS_TEST]->(_t:Test) 
WHERE p.name=patients AND r4.date = dates

WITH DISTINCT p.name as patients, r4.date as dates, symptoms, num_symptoms, procedures, num_procedures, dsds, num_dsds, COLLECT(_t.name) as tests, COUNT(_t.name) as num_tests
MATCH (p:Patient)-[r5:RXNORM_CODE]->(rx:RxNorm)-[r6]->(_d:Drug)
WHERE p.name=patients AND r5.date = dates
RETURN DISTINCT p.name as patients, r5.date as dates, symptoms, num_symptoms, procedures, num_procedures, dsds, num_dsds, tests, num_tests, COLLECT(DISTINCT toLower(_d.name)) as drugs, COUNT(DISTINCT toLower(_d.name)) as num_drugs, COLLECT(DISTINCT rx.code) as rxnorms, COUNT(DISTINCT rx.code) as num_rxnorm
ORDER BY dates;
'''
query_string = query_part1 + query_part2
df = pd.DataFrame([dict(_) for _ in conn.query(query_string)])
df

# COMMAND ----------

# To get the following visualization, run the query in the Neo4j Browser

query_string = '''
  MATCH (p:Patient)
  WHERE p.name = 21153

  CALL apoc.path.subgraphAll(p, {
      relationshipFilter: "RXNORM_CODE>|DRUG_GENERIC",
                minLevel: 0,
                maxLevel: 3
                })
   YIELD nodes, relationships
   RETURN nodes, relationships;
   '''

# COMMAND ----------

# MAGIC %md
# MAGIC <img src="https://raw.githubusercontent.com/iamvarol/blogposts/main/databricks/images/patients_journey.png">

# COMMAND ----------

# MAGIC %md
# MAGIC **which patient used isosorbide:**

# COMMAND ----------

# drug based query
drug_generic_name = 'isosorbide' 

query_part1 = 'MATCH (p:Patient)-[rel_rx]->(rx:RxNorm)-[rel_d]->(d:Drug)-[rel_n]->(n:NER) '
query_part2 = f'WHERE d.name CONTAINS "{drug_generic_name}" AND rel_n.date=rel_rx.date AND rel_n.patient_name=p.name '
query_part3 = '''RETURN DISTINCT
                 d.name as drug_generic_name, 
                 p.name as patient_name, 
                 rel_rx.date as date, 
                 rx.code as rxnorm_code, 
                 COALESCE(n.name,'') +  "(" + COALESCE (type(rel_n), "") + ")" as details'''

query_string = query_part1 + query_part2 + query_part3
df = pd.DataFrame([dict(_) for _ in conn.query(query_string)])
df = df.groupby(['patient_name', 'date', 'rxnorm_code','drug_generic_name']).agg(lambda x : ' '.join(x)).reset_index()
df

# COMMAND ----------

# MAGIC %md
# MAGIC **patients who are prescribed Lasix between May 2060 and May 2125:**

# COMMAND ----------

query_string ='''
MATCH (p:Patient)-[rel_rx]->(rx:RxNorm)-[rel_d]->(d:Drug)-[rel_n:DRUG]->(n:NER)
WHERE d.name IN ['lasix']
      AND rel_n.patient_name=p.name
      AND rel_n.date=rel_rx.date 
      AND rel_rx.date >= date("2060-05-01")
      AND rel_n.date >= date("2060-05-01")
      AND rel_rx.date < date("2125-05-01")
      AND rel_n.date < date("2125-05-01")
RETURN DISTINCT
      d.name as drug_generic_name, 
      p.name as patient_name, 
      rel_rx.date as date
ORDER BY date ASC
'''

df = pd.DataFrame([dict(_) for _ in conn.query(query_string)])
df

# COMMAND ----------

# To obtain the visualization below run the query in Neo4j Browser:

query_string = '''
  MATCH (p:Patient)-[rel_rx]->(rx:RxNorm)-[rel_d]->(d:Drug)-[rel_n:DRUG]->(n:NER)
  WHERE d.name IN ['lasix']
      AND rel_n.patient_name=p.name
      AND rel_n.date=rel_rx.date 
      AND rel_rx.date >= date("2060-05-01")
      AND rel_n.date >= date("2060-05-01")
      AND rel_rx.date < date("2125-05-01")
      AND rel_n.date < date("2125-05-01")
  RETURN d, rel_rx, rx, rel_d, rel_n, p, n;
  '''

# COMMAND ----------

# MAGIC %md
# MAGIC <img src="https://raw.githubusercontent.com/iamvarol/blogposts/main/databricks/images/lasix.png">

# COMMAND ----------

# MAGIC %md
# MAGIC **patients using warfarin 2mg and up:**

# COMMAND ----------

query_string ='''
MATCH (p:Patient)-[rel_rx]->(rx:RxNorm)-[rel_d]->(d:Drug)-[rel_n:STRENGTH]->(n:NER)
WHERE toLower(d.name) CONTAINS 'warfarin'
      AND rel_n.patient_name=p.name
      AND rel_n.date=rel_rx.date 
      AND toInteger(left(n.name,1)) >=2
RETURN  DISTINCT
      d.name as drug_generic_name,
      rx.code as rxnorm_code,
      p.name as patient_name,
      n.name as strength,
      rel_rx.date as date
'''
df = pd.DataFrame([dict(_) for _ in conn.query(query_string)])
df

# COMMAND ----------

# MAGIC %md
# MAGIC **dangerous drug combinations:**

# COMMAND ----------

query_string ='''
WITH ["ibuprofen", "naproxen", "diclofenac", "indometacin", "ketorolac", "aspirin", "ketoprofen", "dexketoprofen", "meloxicam"] AS nsaids
MATCH (p:Patient)-[r1:RXNORM_CODE]->(rx:RxNorm)-[r2]->(d:Drug)
WHERE any(word IN nsaids WHERE d.name CONTAINS word) 
WITH DISTINCT p.name as patients, COLLECT(DISTINCT d.name) as nsaid_drugs, COUNT(DISTINCT d.name) as num_nsaids
MATCH (p:Patient)-[r1:RXNORM_CODE]->(rx:RxNorm)-[r2]->(d:Drug)
WHERE p.name=patients AND d.name CONTAINS 'warfarin'
RETURN DISTINCT patients, 
                nsaid_drugs, 
                num_nsaids, 
                d.name as warfarin_drug, 
                r1.date as date
'''

df = pd.DataFrame([dict(_) for _ in conn.query(query_string)])
df

# COMMAND ----------

# To obtain the visualization below, run the following query in Neo4j Browser:

query_string = '''
    WITH ["ibuprofen", "naproxen", "diclofenac", "indometacin", "ketorolac", "aspirin", "ketoprofen", "dexketoprofen", "meloxicam"] AS nsaids
    MATCH (p:Patient)-[r1:RXNORM_CODE]->(rx:RxNorm)-[r2]->(d:Drug)
    WHERE any(word IN nsaids WHERE d.name CONTAINS word) 
    WITH DISTINCT p.name as patients, COLLECT(DISTINCT d.name) as nsaid_drugs, COUNT(DISTINCT d.name) as num_nsaids
    MATCH (p:Patient)-[r1:RXNORM_CODE]->(rx:RxNorm)-[r2]->(d:Drug)
    WHERE p.name=patients AND d.name CONTAINS 'warfarin'
    RETURN p, rx, d, r1, r2;
    '''

# COMMAND ----------

# MAGIC %md
# MAGIC <img src="https://raw.githubusercontent.com/iamvarol/blogposts/main/databricks/images/ddc.png">

# COMMAND ----------

# MAGIC %md
# MAGIC **patients who underwent a hernia repair or appendectomy, or cholecystectomy:**

# COMMAND ----------

query_string = """
MATCH (pcd1:Procedure)-[rel1:IS_PROCEDURE]-(pati1:Patient)
WHERE pcd1.name CONTAINS 'hernia repair' OR pcd1.name CONTAINS 'appendectomy' OR pcd1.name CONTAINS 'cholecystectomy'
RETURN DISTINCT pati1.name as patients, 
                COLLECT(DISTINCT toLower(pcd1.name)) as procedures
"""

df = pd.DataFrame([dict(_) for _ in conn.query(query_string)])
df

# COMMAND ----------

# patients with chest pain and shortness of breath
query_string = """
MATCH (p1:Patient)-[r1:IS_SYMPTOM]->(s1:Symptom),
(p2:Patient)-[r2:IS_SYMPTOM]->(s2:Symptom)
WHERE s1.name CONTAINS "chest pain" AND s2.name CONTAINS "shortness of breath"
    AND p2.name=p1.name AND r2.date = r1.date
RETURN DISTINCT p1.name as patient, r1.date as date,s1.name as symptom1, s2.name as symptom2
ORDER BY patient
"""
df = pd.DataFrame([dict(_) for _ in conn.query(query_string)])
df

# COMMAND ----------

# MAGIC %md
# MAGIC **patients with hypertension or diabetes with chest pain:**

# COMMAND ----------

query_string = """
MATCH (p:Patient)-[r:IS_SYMPTOM]->(s:Symptom),
(p1:Patient)-[r2:IS_DSD]->(_dsd:DSD)
WHERE s.name CONTAINS "chest pain" AND p1.name=p.name AND _dsd.name IN ['hypertension', 'diabetes'] AND r2.date=r.date
RETURN DISTINCT p.name as patient, r.date as date, _dsd.name as dsd, s.name as symptom
"""
df = pd.DataFrame([dict(_) for _ in conn.query(query_string)])
df

# COMMAND ----------

# To obtain the visualization below, run the following query in Neo4j Browser:

query_string = '''
  MATCH (p:Patient)-[r:IS_SYMPTOM]->(s:Symptom),
  (p1:Patient)-[r2:IS_DSD]->(_dsd:DSD)
  WHERE s.name CONTAINS "chest pain" AND p1.name=p.name AND _dsd.name IN ['hypertension', 'diabetes'] AND r2.date=r.date
  RETURN DISTINCT p, r, _dsd, s;
  '''

# COMMAND ----------

# MAGIC %md
# MAGIC <img src="https://raw.githubusercontent.com/iamvarol/blogposts/main/databricks/images/chest_pain.png">

# COMMAND ----------

# MAGIC %md
# MAGIC ## License
# MAGIC Copyright / License info of the notebook. Copyright [2021] the Notebook Authors.  The source in this notebook is provided subject to the [Apache 2.0 License](https://spdx.org/licenses/Apache-2.0.html).  All included or referenced third party libraries are subject to the licenses set forth below.
# MAGIC 
# MAGIC |Library Name|Library License|Library License URL|Library Source URL|
# MAGIC | :-: | :-:| :-: | :-:|
# MAGIC |Pandas |BSD 3-Clause License| https://github.com/pandas-dev/pandas/blob/master/LICENSE | https://github.com/pandas-dev/pandas|
# MAGIC |Numpy |BSD 3-Clause License| https://github.com/numpy/numpy/blob/main/LICENSE.txt | https://github.com/numpy/numpy|
# MAGIC |Neo4j |Apache License 2.0|https://github.com/neo4j/neo4j/blob/4.4/LICENSE.txt|https://github.com/neo4j/neo4j|
# MAGIC |Apache Spark |Apache License 2.0| https://github.com/apache/spark/blob/master/LICENSE | https://github.com/apache/spark/tree/master/python/pyspark|
# MAGIC |BeautifulSoup|MIT License|https://www.crummy.com/software/BeautifulSoup/#Download|https://www.crummy.com/software/BeautifulSoup/bs4/download/|
# MAGIC |Requests|Apache License 2.0|https://github.com/psf/requests/blob/main/LICENSE|https://github.com/psf/requests|
# MAGIC |Spark NLP Display|Apache License 2.0|https://github.com/JohnSnowLabs/spark-nlp-display/blob/main/LICENSE|https://github.com/JohnSnowLabs/spark-nlp-display|
# MAGIC |Spark NLP |Apache License 2.0| https://github.com/JohnSnowLabs/spark-nlp/blob/master/LICENSE | https://github.com/JohnSnowLabs/spark-nlp|
# MAGIC |Spark NLP for Healthcare|[Proprietary license - John Snow Labs Inc.](https://www.johnsnowlabs.com/spark-nlp-health/) |NA|NA|
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
