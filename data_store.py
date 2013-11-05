# Some code from:
#   http://wiki.postgresql.org/wiki/Using_psycopg2_with_PostgreSQL
#   http://initd.org/psycopg/docs/usage.html
#   http://www.hydrogen18.com/blog/unpickling-buffers.html

import db_constants
import psycopg2
import sys
import cPickle

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

def _trials_to_id_list(trials):
    trial_ids = []
    for t in trials:
        trial_ids.append(_get_id(t))
    return sorted(trial_ids)

def store_roc(trials, path, auc):
    trial_ids = _trials_to_id_list(trials)
    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO figures (path, trials, type, auc) VALUES (%s, %s, %s, %s)", (path, trial_ids, 'roc', auc))
    conn.commit()
    cursor.close()
    conn.close()

def load_roc(trials):
    '''
    Returns None or (roc_path, auc)
    '''
    # Query table by type and trial ids
    trial_ids = _trials_to_id_list(trials)
    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT (path, auc) FROM figures WHERE type = 'roc' AND trials = %s", (trial_ids,))
    records = cursor.fetchall()
    cursor.close()
    conn.close()
    if len(records) == 0:
        return None
    assert(len(records) == 1)
    return records[0][0]
    
def _get_id(trial):
    '''
    '''
    # NOTE: If trial objects had ids we wouldn't have to do this shit... going for easiest
    #       possible thing right now though.
    
    conn = _get_connection()
    cursor = conn.cursor()   
    cursor.execute("SELECT (id) FROM trials WHERE document = '"+trial.doc_name+"' AND features = '"+_feature_list_to_cell_value(trial.features)+"' AND atom_type = '"+trial.atom_type+"' AND cluster_type = '"+trial.cluster_strategy+"'")
    
    records = cursor.fetchall()

    cursor.close()
    conn.close()
    
    if len(records) == 0:
        return None
        
    assert(len(records) == 1)
    return records[0]