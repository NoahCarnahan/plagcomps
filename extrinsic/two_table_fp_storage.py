import psycopg2
import os
import fingerprint_extraction
from ..dbconstants import username, password, dbname
from ..shared.util import ExtrinsicUtility
from ..tokenization import tokenize


TMP_LOAD_FILE = "/tmp/dbimport.txt"
DEBUG = True
DEV_MODE = True

'''
Tables created with:
CREATE TABLE dev_passage_fingerprints (
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

CREATE TABLE dev_hash_table (
    fingerprint_id integer,
    hash_value integer
);
'''

def populate_database(files, method, n, k, atom_type, hash_len):
    '''
    Example usages:
    srs, sus = ExtrinsicUtility().get_training_files(n=10)
    populate_database(srs+sus, "kth_in_sent", n, k, "paragraph", 10000)
    TODO be smart and don't overwrite fingerprints we've already calculated!
    
    '''
        
    conn = psycopg2.connect(user = username, password = password, database = dbname.split("/")[1], host="localhost", port = 5432)
    conn.autocommit = True
    
    for abs_path in files:
    
        # Get doc_name, doc_path, doc_xml_path, and is_source
        doc_name, doc_path, doc_xml_path, is_source =  _get_doc_metadata(abs_path)
        # open the file, and tokenize it.
        # Build atom_texts list
        f = open(doc_path, "r")
        text = f.read()
        f.close()
        if atom_type == "full":
            atom_texts = [text]
        elif atom_type == "paragraph":
            paragraph_spans = tokenize(text, atom_type)
            atom_texts = [text[start:end] for start, end in paragraph_spans]
        else:
            raise ValueError("Invalid atom_type! Only 'full' and 'paragraph' are allowed.")
        
        # Initialize a string to write to the copy file.
        copy_file_string = ""
        
        cur = conn.cursor()
        for atom_number in xrange(len(atom_texts)):
            
            
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
            cur.execute(passage_query, passage_args)
            passage_id = str(cur.fetchone()[0])
            
            # Fingerprint the passage
            extractor = fingerprint_extraction.FingerprintExtractor(hash_len)
            hash_values = extractor.get_fingerprint(atom_texts[atom_number], n, method, k)
            
            # Add rows to the copy_file_string
            for hash_value in hash_values:
                copy_file_string += passage_id + "\t" + str(hash_value) + "\n"

        # Number of lines about to be inserted into DB
        num_rows_to_insert = copy_file_string.count('\n')
        _display_one_doc_debug(doc_name, atom_type, len(atom_texts), num_rows_to_insert)

        # Writes all hashes to file, then bulk inserts to DB
        # NOTE THAT the cursor is NOT closed by the below function. It's cleaned up at the 
        # end of the populate_database() function
        _copy_and_write_to_hash_table(copy_file_string, cur)

    cur.close()
    conn.close()


def get_passage_fingerprint(full_path, passage_num, method, n, k, atom_type, hash_len):
    '''
    Return the fingerprint (list of hash values) of the <passage_num>th passage in <file_name>
    using parameters <method>, <n>, <k>

    TODO clean up output. Returning a list like this:
    [(3160,), (2080,), (584,), (1941,) ...]
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
        
        fp_args = (full_path, passage_num, method, n, k, atom_type, hash_len)

        # With statement cleans up the cursor no matter what
        with conn.cursor() as cur:
            cur.execute(fp_query, fp_args)
            fingerprint = cur.fetchall()

        return fingerprint

def get_passage_ids_with_hash(target_hash_value, method, n, k, atom_type, hash_len):
    '''
    Returns a list of IDs corresponding to rows in the <passage_fingerprints> table which
    contain <target_hash_value> as part of its fingerprint. 

    TODO clean up output. Returning a list like this:
    [(7,), (103,), (3774,), (4083,)...]

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

    return matching_passage_ids

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

    srs, sus = ExtrinsicUtility().get_training_files(n=10)
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
    srcs, sus = ExtrinsicUtility().get_training_files(n=2)

    for s in sus:
        print 'Fingerprint for', s
        f = open(s, "r")
        text = f.read()
        f.close()
        num_atoms = 1 if atom_type == 'full' else len(tokenize(text, atom_type))
        
        for atom_num in xrange(num_atoms):
            passage_fp = [hsh[0] for hsh in set(get_passage_fingerprint(s, atom_num, method, n, k, atom_type, hash_len))]
            for single_hash in passage_fp:
                matching_pass_ids = get_passage_ids_with_hash(single_hash, method, n, k, atom_type, hash_len)
                print single_hash, matching_pass_ids
        print '-'*40

if __name__ == "__main__":
    _test_get_fp_query()
    #_test_reverse_lookup()
    #_test()
