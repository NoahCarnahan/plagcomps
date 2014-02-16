'''
Tables created as follows (also a dev version of each, ex. dev_documents:

CREATE TABLE documents (
    did         serial,
    name        text,
    path        text,
    xml_path    text,
    is_source   boolean
);

CREATE TABLE passages (
    pid         serial,
    did         integer,
    atom_num    integer,
    atom_type   text
);

CREATE TABLE methods (
    mid         serial,
    atom_type   text,
    method_name text,
    n           integer,
    k           integer,
    hash_size   integer
);

CREATE TABLE hashes (
    is_source   boolean,
    pid         integer,
    mid         integer,
    hash_value  integer
);
'''

import psycopg2
import os

import fingerprint_extraction
from ..dbconstants import username, password, dbname
from ..shared.util import ExtrinsicUtility
from ..tokenization import tokenize


TMP_LOAD_FILE = "/tmp/dbimport.txt"
DEBUG = True
DEV_MODE = False

def populate_database(files, method_name, n, k, atom_type, hash_size, check_for_duplicate=True):
    '''
    Example usages:
    srs, sus = ExtrinsicUtility().get_training_files(n=10)
    populate_database(srs+sus, "kth_in_sent", n, k, "paragraph", 10000)
    
    If <check_for_duplicate>, a query for a duplicate row is made before creating a new row 
    and fingerprinting a given document/set of parameters. In other words, we check for the 
    exact row before creating it (and all of its hashes). This doubles the number of queries
    made to the passage_fingerprints table -- if we are running a population script for the first
    time, we can set this flag to be False to speed things up. BUT we must be careful later on
    not to create duplicates (which is why its default is to make the extra check)
    
    '''
    with psycopg2.connect(user = username, password = password, database = dbname.split("/")[1], host="localhost", port = 5432) \
            as conn:
        conn.autocommit = True
                
        # Get the mid (method id) for the given parameters. Create it if it does not already exist.
        with conn.cursor() as cur:
            if DEV_MODE:
                query = "SELECT mid FROM dev_methods WHERE method_name = %s AND n = %s AND k = %s AND atom_type = %s AND hash_size = %s;"
            else:
                query = "SELECT mid FROM methods WHERE method_name = %s AND n = %s AND k = %s AND atom_type = %s AND hash_size = %s;"
            cur.execute(query, (method_name, n, k, atom_type, hash_size))
            result = cur.fetchone()
            if result != None:
                mid = result[0] 
            else:
                if DEV_MODE:
                    query = "INSERT INTO dev_methods (atom_type, method_name, n, k, hash_size) VALUES (%s, %s, %s, %s, %s) RETURNING mid;"
                else:
                    query = "INSERT INTO methods (atom_type, method_name, n, k, hash_size) VALUES (%s, %s, %s, %s, %s) RETURNING mid;"
                cur.execute(query, (atom_type, method_name, n, k, hash_size))
                mid = cur.fetchone()[0]
        
        for doc_num, abs_path in enumerate(files):
            print 'Started working on doc %i of %i' % (doc_num, len(files))
            
            # Get doc_name, doc_path, doc_xml_path, and is_source
            doc_name, doc_path, doc_xml_path, is_source =  _get_doc_metadata(abs_path)
            
            is_source_string = "TRUE" if is_source else "FALSE"
            
            # Get the did (document id) for the docuemnt. Create it if it does not already exist.
            with conn.cursor() as cur:
                if DEV_MODE:
                    query = "SELECT did FROM dev_documents WHERE name = %s AND path = %s AND xml_path = %s AND is_source = %s;"
                else:
                    query = "SELECT did FROM documents WHERE name = %s AND path = %s AND xml_path = %s AND is_source = %s;"
                cur.execute(query, (doc_name, doc_path, doc_xml_path, is_source))
                result = cur.fetchone()
                if result != None:
                    did = result[0]
                else:
                    if DEV_MODE:
                        query = "INSERT INTO dev_documents (name, path, xml_path, is_source) VALUES (%s, %s, %s, %s) RETURNING did;"
                    else:
                        query = "INSERT INTO documents (name, path, xml_path, is_source) VALUES (%s, %s, %s, %s) RETURNING did;"
                    cur.execute(query, (doc_name, doc_path, doc_xml_path, is_source))
                    did = cur.fetchone()[0]
            
            # open the file, and tokenize it.
            # Build atom_texts list
            f = open(doc_path, "r")
            text = f.read()
            f.close()
            if atom_type == "full":
                atom_texts = [text]
            else:
                atom_spans = tokenize(text, atom_type, n=5000)
                atom_texts = [text[start:end] for start, end in atom_spans]
            
            # Initialize a string to write to the copy file.
            copy_file_string = ""
            
            with conn.cursor() as cur:
                for atom_number in xrange(len(atom_texts)):
                    
                    # Get the pid (passage id) for the passage. Create it if it does not already exist.
                    if DEV_MODE:
                        query = "SELECT pid FROM dev_passages WHERE did = %s AND atom_num = %s AND atom_type = %s;"
                    else:
                        query = "SELECT pid FROM passages WHERE did = %s AND atom_num = %s AND atom_type = %s;"
                    cur.execute(query, (did, atom_number, atom_type))
                    result = cur.fetchone()
                    if result != None:
                        pid = result[0]
                    else:
                        if DEV_MODE:
                            query = "INSERT INTO dev_passages (did, atom_num, atom_type) VALUES (%s, %s, %s) RETURNING pid;"
                        else:
                            query = "INSERT INTO passages (did, atom_num, atom_type) VALUES (%s, %s, %s) RETURNING pid;"
                        cur.execute(query, (did, atom_number, atom_type))
                        pid = cur.fetchone()[0]
                    
                    # Check if this passage has been populated for this method.
                    if check_for_duplicate and _row_already_exists(pid, mid):
                        print (doc_path, atom_number, atom_type, method_name, n, k, hash_size), 'already existed! Skipped it.'
                        continue 
                    
                    # Fingerprint the passage                        
                    extractor = fingerprint_extraction.FingerprintExtractor(hash_size)
                    hash_values = set(extractor.get_fingerprint(atom_texts[atom_number], n, method_name, k))
                    
                    # Add rows to the copy_file_string
                    for hash_value in hash_values:
                        copy_file_string += is_source_string + "\t" + str(pid) + "\t" + str(mid) + "\t" + str(hash_value) + "\n"

                # Number of lines about to be inserted into DB
                num_rows_to_insert = copy_file_string.count('\n')
                _display_one_doc_debug(doc_name, atom_type, len(atom_texts), num_rows_to_insert)

                # Writes all hashes to file, then bulk inserts to DB
                _copy_and_write_to_hash_table(copy_file_string, cur)



def get_passage_fingerprint(full_path, passage_num, atom_type, mid):
    '''
    Return the fingerprint (list of hash values) of the <passage_num>th passage in <file_name>
    using the parameters that correspond to the given <mid> (method id).
    '''
    # With statement cleans up the connection
    with psycopg2.connect(user = username, password = password, database = dbname.split("/")[1], host="localhost", port = 5432) \
            as conn:
        conn.autocommit = True
        
        if DEV_MODE:
            fp_query = '''SELECT hash_value FROM dev_hashes, dev_documents, dev_passages WHERE
                        dev_documents.path = %s AND
                        
                        dev_passages.atom_type = %s AND
                        dev_passages.did = dev_documents.did AND
                        dev_passages.atom_num = %s AND
                        
                        dev_hashes.mid = %s AND
                        dev_hashes.pid = dev_passages.pid;'''
        else:
            fp_query = '''SELECT hash_value FROM hashes, documents, passages WHERE
                        documents.path = %s AND
                        
                        passages.atom_type = %s AND
                        passages.did = documents.did AND
                        passages.atom_num = %s AND
                        
                        hashes.mid = %s AND
                        hashes.pid = passages.pid;'''
                        
        fp_args = (full_path, atom_type, passage_num, mid)
        
        # With statement cleans up the cursor no matter what
        with conn.cursor() as cur:
            cur.execute(fp_query, fp_args)
            fingerprint = cur.fetchall()

        # Format into nicer list
        clean_fp = [hsh[0] for hsh in fingerprint]
        
        return clean_fp
        
def get_passage_fingerprint_by_id(pid, mid, conn):
    '''
    Return the fingerprint for the passage designated by pid and for the fingerprinting method
    designated by mid.
    '''

    if DEV_MODE:
        fp_query = "SELECT dev_hashes.hash_value FROM dev_hashes WHERE dev_hashes.pid = %s AND dev_hashes.mid = %s;"
    else:
        fp_query = "SELECT hashes.hash_value FROM hashes WHERE hashes.pid = %s AND hashes.mid = %s;"
    with conn.cursor() as cur:
        cur.execute(fp_query, (pid, mid))
        fingerprint = cur.fetchall()
        return [row[0] for row in fingerprint]

    
def get_mid(method, n, k, atom_type, hash_size):
    '''
    Return the method id (mid) for the given method name, n, k, atom_type, and hash_size.
    Return None if no mid exists for these parameters.
    '''
    with psycopg2.connect(user = username, password = password, database = dbname.split("/")[1], host="localhost", port = 5432) \
            as conn:
        with conn.cursor() as cur:
            if DEV_MODE:
                query = "SELECT mid FROM dev_methods WHERE atom_type = %s AND method_name = %s AND n = %s AND k = %s AND hash_size = %s;"
            else:
                query = "SELECT mid FROM methods WHERE atom_type = %s AND method_name = %s AND n = %s AND k = %s AND hash_size = %s;"
            cur.execute(query, (atom_type, method, n, k, hash_size))
            result = cur.fetchone()
    if result == None:
        return result
    else:
        return result[0]

def get_matching_passages(target_hash_value, mid, conn):
    '''
    Returns a list of dictionaries like this:
    {"pid":1, "doc_name":"foo", "atom_number":34}, ...]
    
    Only returns dictionaries where <target_hash_value> is part of its fingerprint.
    '''
        
    # Get passages (and their atom_numbers and doc_names) with target_hash_value and mid            
    if DEV_MODE:
        reverse_query = '''SELECT dev_hashes.pid, dev_documents.name, dev_passages.atom_num FROM dev_hashes, dev_documents, dev_passages WHERE
                        dev_documents.did = dev_passages.did AND
                        dev_passages.pid = dev_hashes.pid AND
                        dev_hashes.hash_value = %s AND
                        dev_hashes.mid = %s AND
                        dev_hashes.is_source = 't';'''
    else:
        reverse_query = '''SELECT hashes.pid, documents.name, passages.atom_num FROM hashes,  documents, passages WHERE
                        documents.did = passages.did AND
                        passages.pid = hashes.pid AND
                        hashes.hash_value = %s AND
                        hashes.mid = %s AND
                        hashes.is_source = 't';'''
                        
    reverse_args = (target_hash_value, mid)

    # With statement cleans up the cursor no matter what
    with conn.cursor() as cur:
        cur.execute(reverse_query, reverse_args)
        matching_passages = cur.fetchall()

    passages = [{"pid":passage[0], "doc_name":passage[1], "atom_number":passage[2]} for passage in matching_passages]

    return passages

def get_matching_passages_with_fingerprints(target_hash_value, mid):
    '''
    Returns a list of dictionaries like this:
    [{"pid":1, "doc_name":"foo", "atom_number":34, "fingerprint":[23453, 114, ...]}, ...]
    
    Only returns dictionaries where <target_hash_value> is part of its fingerprint.
    '''
    
    with psycopg2.connect(user = username, password = password, database = dbname.split("/")[1], host="localhost", port = 5432) \
        as conn:
        
        passages = get_matching_passages(target_hash_value, mid, conn)    
        for i in range(len(passages)):
            passages[i]["fingerprint"] = get_passage_fingerprint_by_id(passages[i]["pid"], mid, conn)
    
    return passages
    
    """
    with psycopg2.connect(user = username, password = password, database = dbname.split("/")[1], host="localhost", port = 5432) \
            as conn:
        
        # Get passages (and their atom_numbers and doc_names) with target_hash_value and mid            
        if DEV_MODE:
            reverse_query = '''SELECT dev_hashes.pid, dev_documents.name, dev_passages.atom_num FROM dev_hashes, dev_documents, dev_passages WHERE
                            dev_documents.did = dev_passages.did AND
                            dev_passages.pid = dev_hashes.pid AND
                            dev_hashes.hash_value = %s AND
                            dev_hashes.mid = %s AND
                            dev_hashes.is_source = 't';'''
        else:
            reverse_query = '''SELECT hashes.pid, documents.name, passages.atom_num FROM hashes,  documents, passages WHERE
                            documents.did = passages.did AND
                            passages.pid = hashes.pid AND
                            hashes.hash_value = %s AND
                            hashes.mid = %s AND
                            hashes.is_source = 't';'''
                            
        reverse_args = (target_hash_value, mid)

        # With statement cleans up the cursor no matter what
        with conn.cursor() as cur:
            cur.execute(reverse_query, reverse_args)
            matching_passages = cur.fetchall()

        passages = [{"pid":passage[0], "doc_name":passage[1], "atom_number":passage[2]} for passage in matching_passages]
        
        # Now get the fingerprints for each passage:
        for i in range(len(passages)):
            if DEV_MODE:
                fp_query = "SELECT dev_hashes.hash_value FROM dev_hashes WHERE dev_hashes.pid = %s AND dev_hashes.mid = %s;"
            else:
                fp_query = "SELECT hashes.hash_value FROM hashes WHERE hashes.pid = %s AND hashes.mid = %s;"
            with conn.cursor() as cur:
                cur.execute(fp_query, (passages[i]["pid"], mid))
                fingerprint = cur.fetchall()
                passages[i]["fingerprint"] = [row[0] for row in fingerprint]

    return passages
    """

def get_passage_ids_by_hash(target_hash_value, mid):
    '''
    Return a list of passaged ids (pid) corresponding to passages that contain <target_hash_value>
    as part of its fingerprint. NOTE: only returns pids for source passages. 
    '''

    with psycopg2.connect(user = username, password = password, database = dbname.split("/")[1], host="localhost", port = 5432) \
            as conn:
        if DEV_MODE:
            query = '''SELECT pid FROM dev_hashes WHERE hash_value = %s AND mid = %s AND is_source = 't';'''
        else:
            query = '''SELECT pid FROM dev_hashes WHERE hash_value = %s AND mid = %s AND is_source = 't';'''
            
        args = (target_hash_value, mid)

        # With statement cleans up the cursor no matter what
        with conn.cursor() as cur:
            cur.execute(query, args)
            matching_passage_ids = cur.fetchall()

    clean_ids = [passid[0] for passid in matching_passage_ids]

    return clean_ids



def _copy_and_write_to_hash_table(copy_file_string, curs):
    '''
    Writes <copy_file_string> to a temporary file. DB then reads from the temp file
    and performs a bulk insert of all data stored in <copy_file_string>.
    '''
    f = open(TMP_LOAD_FILE, "w")
    f.write(copy_file_string)
    f.close()
    
    # Table names aren't permitted to be parameterized -- have to pass them explicitly
    if DEV_MODE:
        copy_query = "COPY dev_hashes FROM %s" 
    else:
        copy_query = "COPY hashes FROM %s" 
    copy_args = (TMP_LOAD_FILE, )
    curs.execute(copy_query, copy_args)

    os.remove(TMP_LOAD_FILE)

def _get_doc_metadata(abs_path):
    '''
    Returns metadata about file stored in <abs_path>. Specifically returns
    doc_name, doc_path, doc_xml_path, is_source
    '''
    if 'source' in abs_path:
        base_path = ExtrinsicUtility().CORPUS_SRC_LOC
        is_source = True
    else:
        base_path = ExtrinsicUtility().CORPUS_SUSPECT_LOC
        is_source = False

    doc_path = abs_path
    doc_xml_path = abs_path.replace(".txt", ".xml")
    doc_name = abs_path.replace(base_path, "").replace(".txt", "")

    return doc_name, doc_path, doc_xml_path, is_source      
            
def _row_already_exists(pid, mid):
    '''
    Checks if a row with the provided parameters already exists by querying the DB. Used to make 
    sure we don't overwrite/re-fingerprint documents
    '''
    with psycopg2.connect(user = username, password = password, database = dbname.split("/")[1], host="localhost", port = 5432) \
            as conn:
        if DEV_MODE:
            existence_query = "SELECT * FROM dev_hashes WHERE pid = %s AND mid = %s LIMIT 1;"
        else:
            existence_query = "SELECT * FROM hashes WHERE pid = %s AND mid = %s LIMIT 1;"
        existence_args = (pid, mid)

        with conn.cursor() as cur:
            cur.execute(existence_query, existence_args)
            exists = True if cur.fetchone() else False
            
    return exists

def _display_one_doc_debug(doc_name, atom_type, num_atoms, num_rows_to_insert):
    '''
    Print data about document being processed
    '''
    print '%s with atom_type = %s. Had %i atoms. Inserting %i rows...' % \
            (doc_name, atom_type, num_atoms, num_rows_to_insert)

def _display_verbose_debug_info(doc_name, atom_type, atom_number, num_atom_texts, text_len, num_hashes):
    '''
    Call this function for each atom in populate_database() to see a lot of information about what's happening
    '''
    if DEBUG:
        print '%s with atom_type = %s. Finished atom %i of %i. Length = %i chars, had %i hashes.' % \
              (doc_name, atom_type, atom_number, num_atom_texts, text_len, num_hashes)

def _test():
    method = 'kth_in_sent'
    n = 4
    k = 3
    atom_type = 'paragraph'
    hash_len = 10000

    srs, sus = ExtrinsicUtility().get_training_files(n=12)
    populate_database(srs+sus, method, n, k, atom_type, hash_len)

def _test_get_fp_query():
    '''
    TODO just grab all passages associated with a given doc, don't read it again!
    '''
    method = 'kth_in_sent'
    n = 4
    k = 3
    atom_type = 'paragraph'
    hash_len = 10000
    srcs, sus = ExtrinsicUtility().get_training_files(n=1)
    mid = get_mid(method, n, k, atom_type, hash_len)

    for doc in srcs + sus:
        print 'Fingerprint for', doc
        f = open(doc, "r")
        text = f.read()
        f.close()
        num_atoms = 1 if atom_type == 'full' else len(tokenize(text, atom_type))
        
        for atom_num in xrange(num_atoms):
            print 'atom_num =', atom_num, get_passage_fingerprint(doc, atom_num, atom_type, mid)
        print '-'*40

def _test_reverse_lookup():
    method = 'kth_in_sent'
    n = 4
    k = 3
    atom_type = 'paragraph'
    hash_len = 10000
    srcs, sus = ExtrinsicUtility().get_training_files(n=1)
    mid = get_mid(method, n, k, atom_type, hash_len)
    
    for s in sus:
        print 'Fingerprint for', s
        f = open(s, "r")
        text = f.read()
        f.close()
        num_atoms = 1 if atom_type == 'full' else len(tokenize(text, atom_type))
        
        for atom_num in xrange(num_atoms):
            passage_fp = set(get_passage_fingerprint(s, atom_num, atom_type, mid))
            for single_hash in passage_fp:
                matching_pass_ids = get_passage_ids_by_hash(single_hash, mid)
                print 'Hash %i found in:' % (single_hash)
                print matching_pass_ids
        print '-'*40

def _populate_variety_of_params():
    srs, sus = ExtrinsicUtility().get_training_files(n=400)
    # After you populate the database, leave its line here, but commented out. This way we can easily see what
    # is already in it. Yes I am aware this is an extremely error prone and stupid way to
    # keep track of this.
    #populate_database(sus+srs, "kth_in_sent", 5, 3, "full", 10000000, check_for_duplicate=False)
    #populate_database(sus+srs, "kth_in_sent", 5, 3, "nchars", 10000000, check_for_duplicate=False)
    #populate_database(sus+srs, "kth_in_sent", 5, 3, "paragraph", 10000000, check_for_duplicate=False)
    #populate_database(sus+srs, "kth_in_sent", 3, 3, "paragraph", 10000000, check_for_duplicate=False)
    #populate_database(sus+srs, "anchor", 5, 0, "paragraph", 10000000, check_for_duplicate=False)
    #populate_database(sus+srs, "anchor", 3, 0, "paragraph", 10000000, check_for_duplicate=False)
    populate_database(sus+srs, "anchor", 5, 0, "full", 10000000, check_for_duplicate=False)
    populate_database(sus+srs, "anchor", 3, 0, "full", 10000000, check_for_duplicate=False)
    

if __name__ == "__main__":
    print 'DEV_MODE is set to', DEV_MODE
    _populate_variety_of_params()
    #_test()
    #_test_get_fp_query()
    #_test_reverse_lookup()
    