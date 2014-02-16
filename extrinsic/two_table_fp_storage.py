import psycopg2
import os

import fingerprint_extraction
from ..dbconstants import username, password, dbname
from ..shared.util import ExtrinsicUtility
from ..tokenization import tokenize


TMP_LOAD_FILE = "/tmp/dbimport.txt"
DEBUG = True
DEV_MODE = False

'''
Tables created with:
CREATE TABLE passage_fingerprints (
    id serial,
    doc_name text,
    doc_path text,
    doc_xml_path text,
    atom_number integer,
    is_source boolean,

    atom_type text,
    method text,
    n integer,
    k integer,
    hash_len integer
);

CREATE TABLE hash_table (
    fingerprint_id integer,
    hash_value integer
);
'''

def populate_database(files, method, n, k, atom_type, hash_len, check_for_duplicate=True):
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
        
        for doc_num, abs_path in enumerate(files):
        
            print 'Started working on doc %i of %i' % (doc_num, len(files))
            # Get doc_name, doc_path, doc_xml_path, and is_source
            doc_name, doc_path, doc_xml_path, is_source =  _get_doc_metadata(abs_path)
            # open the file, and tokenize it.
            # Build atom_texts list
            f = open(doc_path, "r")
            text = f.read()
            f.close()
            if atom_type == "full":
                atom_texts = [text]
            elif atom_type == "paragraph" or atom_type == "nchars":
                atom_spans = tokenize(text, atom_type, n=5000)
                atom_texts = [text[start:end] for start, end in atom_spans]
            else:
                raise ValueError("Invalid atom_type! Only 'full', 'nchars', and 'paragraph' are allowed.")
            
            # Initialize a string to write to the copy file.
            copy_file_string = ""
            
            with conn.cursor() as cur:
                for atom_number in xrange(len(atom_texts)):
                    
                    # Only make the duplicate-check query if checking for duplicates
                    # I'd love to use a Ruby `unless` statement here. Oh well...
                    if check_for_duplicate and _row_already_exists(doc_path, atom_number, atom_type, method, n, k, hash_len):
                        print (doc_path, atom_number, atom_type, method, n, k, hash_len), 'already existed! Skipped it.'
                        continue
                    else:
                        # Table names aren't permitted to be parameterized -- have to pass them explicitly
                        # Set DEV_MODE = True to use a development version of the fingerprint/hash table
                        if DEV_MODE:
                            passage_query = """INSERT INTO dev_passage_fingerprints
                                            (doc_name, doc_path, doc_xml_path, atom_number, is_source, atom_type, method, n, k, hash_len)
                                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id;"""
                        else:
                            passage_query = """INSERT INTO passage_fingerprints
                                            (doc_name, doc_path, doc_xml_path, atom_number, is_source, atom_type, method, n, k, hash_len)
                                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id;"""

                        passage_args = (doc_name, doc_path, doc_xml_path, atom_number, is_source, atom_type, method, n, k, hash_len)
                        # Insert the passage into the database and get its id
                        # TODO is any executemany() call faster?
                        cur.execute(passage_query, passage_args)
                        passage_id = str(cur.fetchone()[0])
                        
                        # Fingerprint the passage
                        extractor = fingerprint_extraction.FingerprintExtractor(hash_len)
                        hash_values = set(extractor.get_fingerprint(atom_texts[atom_number], n, method, k))
                        
                        # Add rows to the copy_file_string
                        for hash_value in hash_values:
                            copy_file_string += passage_id + "\t" + str(hash_value) + "\n"

                # Number of lines about to be inserted into DB
                num_rows_to_insert = copy_file_string.count('\n')
                _display_one_doc_debug(doc_name, atom_type, len(atom_texts), num_rows_to_insert)

                # Writes all hashes to file, then bulk inserts to DB
                _copy_and_write_to_hash_table(copy_file_string, cur)

def get_passage_fingerprint(full_path, passage_num, method, n, k, atom_type, hash_len):
    '''
    Return the fingerprint (list of hash values) of the <passage_num>th passage in <file_name>
    using parameters <method>, <n>, <k>
    '''
    # With statement cleans up the connection
    with psycopg2.connect(user = username, password = password, database = dbname.split("/")[1], host="localhost", port = 5432) \
            as conn:
        conn.autocommit = True
        # Can't parameterize table name, so this is gross -- see 
        # http://initd.org/psycopg/docs/usage.html#the-problem-with-the-query-parameters
        if DEV_MODE:
            fp_query = '''SELECT dev_hash_table.hash_value FROM dev_hash_table, dev_passage_fingerprints
                        WHERE
                        dev_passage_fingerprints.id = dev_hash_table.fingerprint_id AND
                        dev_passage_fingerprints.doc_path = %s AND
                        dev_passage_fingerprints.atom_number = %s AND
                        dev_passage_fingerprints.method = %s AND
                        dev_passage_fingerprints.n = %s AND
                        dev_passage_fingerprints.k = %s AND
                        dev_passage_fingerprints.atom_type = %s AND
                        dev_passage_fingerprints.hash_len = %s;'''
        else:
            fp_query = '''SELECT hash_table.hash_value FROM hash_table, passage_fingerprints
                        WHERE
                        passage_fingerprints.id = hash_table.fingerprint_id AND
                        passage_fingerprints.doc_path = %s AND
                        passage_fingerprints.atom_number = %s AND
                        passage_fingerprints.method = %s AND
                        passage_fingerprints.n = %s AND
                        passage_fingerprints.k = %s AND
                        passage_fingerprints.atom_type = %s AND
                        passage_fingerprints.hash_len = %s;'''
        
        fp_args = (full_path, passage_num, method, n, k, atom_type, hash_len)

        # With statement cleans up the cursor no matter what
        with conn.cursor() as cur:
            cur.execute(fp_query, fp_args)
            fingerprint = cur.fetchall()

        # Format into nicer list
        clean_fp = [hsh[0] for hsh in fingerprint]
        
        return clean_fp

def get_passages_with_fingerprints(target_hash_value, method, n, k, atom_type, hash_len):
    '''
    Returns a list of dictionaries like this:
    [{"id":1, "doc_name":"foo", "atom_number":34, "fingerprint":[23453, 114, ...]}, ...]
    
    Only returns dictionaries where <target_hash_value> is part of its fingerprint.
    '''
    
    with psycopg2.connect(user = username, password = password, database = dbname.split("/")[1], host="localhost", port = 5432) \
            as conn:
        if DEV_MODE:
            reverse_query = '''SELECT dev_passage_fingerprints.id, dev_passage_fingerprints.doc_name, dev_passage_fingerprints.atom_number
                            FROM dev_passage_fingerprints, dev_hash_table WHERE
                            dev_hash_table.hash_value = %s AND
                            dev_hash_table.fingerprint_id = dev_passage_fingerprints.id AND
                            dev_passage_fingerprints.method = %s AND
                            dev_passage_fingerprints.n = %s AND
                            dev_passage_fingerprints.k = %s AND
                            dev_passage_fingerprints.atom_type = %s AND
                            dev_passage_fingerprints.hash_len = %s AND
                            dev_passage_fingerprints.is_source = 't';'''
        else:
            reverse_query = '''SELECT passage_fingerprints.id, passage_fingerprints.doc_name, passage_fingerprints.atom_number
                            FROM passage_fingerprints, hash_table WHERE
                            hash_table.hash_value = %s AND
                            hash_table.fingerprint_id = passage_fingerprints.id AND
                            passage_fingerprints.method = %s AND
                            passage_fingerprints.n = %s AND
                            passage_fingerprints.k = %s AND
                            passage_fingerprints.atom_type = %s AND
                            passage_fingerprints.hash_len = %s AND
                            passage_fingerprints.is_source = 't';'''
        reverse_args = (target_hash_value, method, n, k, atom_type, hash_len)

        # With statement cleans up the cursor no matter what
        with conn.cursor() as cur:
            cur.execute(reverse_query, reverse_args)
            matching_passages = cur.fetchall()

        passages = [{"id":passage[0], "doc_name":passage[1], "atom_number":passage[2]} for passage in matching_passages]
        
        # Now get the fingerprints for each passage:
        for i in range(len(passages)):
            if DEV_MODE:
                fp_query = "SELECT dev_hash_table.hash_value FROM dev_hash_table WHERE dev_hash_table.fingerprint_id = %s"
            else:
                fp_query = "SELECT hash_table.hash_value FROM hash_table WHERE hash_table.fingerprint_id = %s"
            with conn.cursor() as cur:
                cur.execute(fp_query, (passages[i]["id"],))
                fingerprint = cur.fetchall()
                passages[i]["fingerprint"] = [row[0] for row in fingerprint]

    return passages


def get_passage_ids_with_hash(target_hash_value, method, n, k, atom_type, hash_len):
    '''
    Returns a list of IDs corresponding to rows in the <passage_fingerprints> table which
    contain <target_hash_value> as part of its fingerprint. 

    TODO this returns all IDs, including the one used to query. Don't forget to filter this out!
    '''

    with psycopg2.connect(user = username, password = password, database = dbname.split("/")[1], host="localhost", port = 5432) \
            as conn:
        if DEV_MODE:
            reverse_query = '''SELECT dev_passage_fingerprints.id FROM dev_passage_fingerprints, dev_hash_table 
                            WHERE
                            dev_hash_table.hash_value = %s AND
                            dev_hash_table.fingerprint_id = dev_passage_fingerprints.id AND
                            dev_passage_fingerprints.method = %s AND
                            dev_passage_fingerprints.n = %s AND
                            dev_passage_fingerprints.k = %s AND
                            dev_passage_fingerprints.atom_type = %s AND
                            dev_passage_fingerprints.hash_len = %s;'''
        else:
            reverse_query = '''SELECT passage_fingerprints.id FROM passage_fingerprints, hash_table 
                            WHERE
                            hash_table.hash_value = %s AND
                            hash_table.fingerprint_id = passage_fingerprints.id AND
                            passage_fingerprints.method = %s AND
                            passage_fingerprints.n = %s AND
                            passage_fingerprints.k = %s AND
                            passage_fingerprints.atom_type = %s AND
                            passage_fingerprints.hash_len = %s;'''
        reverse_args = (target_hash_value, method, n, k, atom_type, hash_len)

        # With statement cleans up the cursor no matter what
        with conn.cursor() as cur:
            cur.execute(reverse_query, reverse_args)
            matching_passage_ids = cur.fetchall()

    clean_ids = [passid[0] for passid in matching_passage_ids]

    return clean_ids

def _copy_and_write_to_hash_table(copy_file_string, curs):
    '''
    Writes <copy_file_string> to a temporary file. DB then reads from the temp file
    and performs a bulk insert of all data stored in <copy_file_string>
    '''
    f = open(TMP_LOAD_FILE, "w")
    f.write(copy_file_string)
    f.close()
    
    # Table names aren't permitted to be parameterized -- have to pass them explicitly
    if DEV_MODE:
        copy_query = "COPY dev_hash_table FROM %s" 
    else:
        copy_query = "COPY hash_table FROM %s" 
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
            
def _row_already_exists(doc_path, atom_number, atom_type, method, n, k, hash_len):
    '''
    Checks if a row with the provided parameters already exists by querying the DB. Used to make 
    sure we don't overwrite/re=fingerprint documents
    '''
    with psycopg2.connect(user = username, password = password, database = dbname.split("/")[1], host="localhost", port = 5432) \
            as conn:
        if DEV_MODE:
            existence_query = '''SELECT COUNT(*) FROM dev_passage_fingerprints
                                WHERE
                                doc_path = %s AND
                                atom_number = %s AND
                                method = %s AND
                                n = %s AND
                                k = %s AND
                                atom_type = %s AND
                                hash_len = %s;'''
        else:
            existence_query = '''SELECT COUNT(*) FROM passage_fingerprints
                                WHERE
                                doc_path = %s AND
                                atom_number = %s AND
                                method = %s AND
                                n = %s AND
                                k = %s AND
                                atom_type = %s AND
                                hash_len = %s;'''
        existence_args = (doc_path, atom_number, method, n, k, atom_type, hash_len)

        with conn.cursor() as cur:
            cur.execute(existence_query, existence_args)
            num_existing_rows = cur.fetchall()

    return num_existing_rows[0][0] > 0

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
    srcs, sus = ExtrinsicUtility().get_training_files(n=3)

    for doc in srcs + sus:
        print 'Fingerprint for', doc
        f = open(doc, "r")
        text = f.read()
        f.close()
        num_atoms = 1 if atom_type == 'full' else len(tokenize(text, atom_type))
        
        for atom_num in xrange(num_atoms):
            print 'atom_num =', atom_num, get_passage_fingerprint(doc, atom_num, method, n, k, atom_type, hash_len)
        print '-'*40

def _test_reverse_lookup():
    method = 'kth_in_sent'
    n = 4
    k = 3
    atom_type = 'paragraph'
    hash_len = 10000
    srcs, sus = ExtrinsicUtility().get_training_files(n=1)

    for s in sus:
        print 'Fingerprint for', s
        f = open(s, "r")
        text = f.read()
        f.close()
        num_atoms = 1 if atom_type == 'full' else len(tokenize(text, atom_type))
        
        for atom_num in xrange(num_atoms):
            passage_fp = set(get_passage_fingerprint(s, atom_num, method, n, k, atom_type, hash_len))
            for single_hash in passage_fp:
                matching_pass_ids = get_passage_ids_with_hash(single_hash, method, n, k, atom_type, hash_len)
                print 'Hash %i found in:' % (single_hash)
                print matching_pass_ids
        print '-'*40

def _populate_variety_of_params():
    srs, sus = ExtrinsicUtility().get_training_files(n=400)
    #populate_database(sus+srs, "kth_in_sent", 5, 0, "full", 10000000, check_for_duplicate=True)
    populate_database(sus+srs, "kth_in_sent", 5, 0, "nchars", 10000000, check_for_duplicate=True)
    
    #hash_len = 10000
    # method, n, k, atom_type
    
    #populate_database(sus+srs, "anchor", 5, 0, "paragraph", hash_len, check_for_duplicate=False)
    #populate_database(sus+srs, "anchor", 3, 0, "paragraph", hash_len, check_for_duplicate=False)

    #populate_database(sus+srs, "kth_in_sent", 5, 5, "paragraph", hash_len, check_for_duplicate=False)
    #populate_database(sus+srs, "kth_in_sent", 5, 3, "paragraph", hash_len, check_for_duplicate=False)
    #populate_database(sus+srs, "kth_in_sent", 3, 5, "paragraph", hash_len, check_for_duplicate=False)
    #populate_database(sus+srs, "kth_in_sent", 3, 3, "paragraph", hash_len, check_for_duplicate=False)

    #populate_database(sus+srs, "full", 5, 0, "paragraph", hash_len, check_for_duplicate=False)
    #populate_database(sus+srs, "full", 3, 0, "paragraph", hash_len, check_for_duplicate=False)


if __name__ == "__main__":
    print 'DEV_MODE is set to', DEV_MODE
    _populate_variety_of_params()
    #_test_get_fp_query()
    #_test()
