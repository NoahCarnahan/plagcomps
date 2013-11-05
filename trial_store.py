# Some code from:
#   http://wiki.postgresql.org/wiki/Using_psycopg2_with_PostgreSQL
#   http://initd.org/psycopg/docs/usage.html
#   http://www.hydrogen18.com/blog/unpickling-buffers.html

import db_constants
import psycopg2
import sys
import cPickle
import StringIO

def _get_connection():
    conn_string = "host='localhost' dbname='"+db_constants.db+"' user='"+db_constants.user+"' password='"+db_constants.pw+"'"
    return psycopg2.connect(conn_string)

def _feature_list_to_cell_value(l):
    return str(sorted(l)).replace("'", "").replace('"','')

def store_trial(trial):
    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO trials (document, features, atom_type, cluster_type, pickle) VALUES (%s, %s, %s, %s, %s)",
        (trial.doc_name, _feature_list_to_cell_value(trial.features), trial.atom_type, trial.cluster_strategy, cPickle.dumps(trial)))
    conn.commit()
    cursor.close()
    conn.close()
    
def load_trial(doc_name, features, atom_type, cluster_strategy):
    conn = _get_connection()
    cursor = conn.cursor()   
    cursor.execute("SELECT (pickle) FROM trials WHERE document = '"+doc_name+"' AND features = '"+_feature_list_to_cell_value(features)+"' AND atom_type = '"+atom_type+"' AND cluster_type = '"+cluster_strategy+"'")
    
    records = cursor.fetchall()
    
    cursor.close()
    conn.close()
    
    if len(records) == 0:
        return None
        
    assert(len(records) == 1)
    return cPickle.loads(records[0][0])
    
    