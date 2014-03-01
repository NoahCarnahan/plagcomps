from ..shared.util import ExtrinsicUtility
from ..extrinsic.fingerprintstorage import populate_database, _get_connection
from ..extrinsic.extrinsic_testing import test

create_tables_query = '''
    CREATE TABLE crisp_documents (
        did         serial,
        name        text,
        path        text,
        xml_path    text,
        is_source   boolean
    );

    CREATE TABLE crisp_passages (
        pid         serial,
        did         integer,
        atom_num    integer,
        atom_type   text
    );

    CREATE TABLE crisp_methods (
        mid         serial,
        atom_type   text,
        method_name text,
        n           integer,
        k           integer,
        hash_size   integer
    );

    CREATE TABLE crisp_hashes (
        is_source   boolean,
        pid         integer,
        mid         integer,
        hash_value  integer
    );
    CREATE INDEX idx_crisp_did ON crisp_passages(did);
    CREATE INDEX idx_crisp_atom_num ON crisp_passages(atom_num);
'''
create_indices_query = '''
    CREATE INDEX idx_crisp_path ON          crisp_documents(path);
    CREATE INDEX idx_crisp_atom_type ON     crisp_passages(atom_type);
    CREATE INDEX idx_crisp_pid2 ON          crisp_passages(pid);
    CREATE INDEX idx_crisp_hash_value2 ON   crisp_hashes(hash_value);
    CREATE INDEX idx_crisp_is_source ON     crisp_hashes(is_source);
    CREATE INDEX idx_crisp_mid ON           crisp_hashes(mid);
    CREATE INDEX idx_crisp_pid ON           crisp_hashes(pid);
    CREATE INDEX idx_crisp_foo ON           crisp_hashes(pid, mid);
'''
drop_tables_query = '''
    DROP TABLE IF EXISTS crisp_documents, crisp_passages, crisp_methods, crisp_hashes;
'''

#def get_connection(autocommit=False):
#    conn = psycopg2.connect(user = username, password = password, database = extrinsicdbname2, host="localhost", port = 5432)
#    conn.autocommit = autocommit
#    return conn

def exec_q(query):
    with _get_connection(autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(query)

def main():
    num_sus = 16
    print "Getting documents"
    srs, sus = ExtrinsicUtility().get_corpus_files(n=num_sus)

    for atom_type in ["nchars", "paragraph", "full"]:
        for method in ["kth_in_sent", "anchor", "winnow-k", "full"]:
            if method == "kth_in_sent":
                nks = [(5,3), (3,3), (5,5), (3,5), (5, 0), (5,8)]
            elif method == "anchor":
                nks = [(3,0), (4,0), (5,0)]
            elif method == "full":
                nks = [(3,0), (4,0), (5,0)]
            elif method == "winnow-k":
                nks = [(8,13),(8,15),(6,13),(6,15)]
            for n,k in nks:
                print atom_type, method,n,k
                # Drop tables
                print "Dropping tables..."
                exec_q(drop_tables_query)
                # Create tables
                print "Creating tables..."
                exec_q(create_tables_query)
                # Populate
                print "Popuating db..."
                populate_database(sus+srs, method, n, k, atom_type, 10000000, check_for_duplicate=False)
                # Create indices
                print "Creating indices..."
                exec_q(create_indices_query)
                # Test
                print "Running tests..."
                for compare_method in ["containment", "jaccard"]:
                    test(method, n, k, atom_type, 10000000, compare_method, num_files=num_sus, search_method='normal', search_n=1, save_to_db=True, ignore_high_obfuscation=False, show_false_negpos_info=False)

if __name__ == "__main__":
    #main()
    num_sus = 64
    srs, sus = ExtrinsicUtility().get_corpus_files(n=num_sus)
    docs = srs + sus

    exec_q(drop_tables_query)
    exec_q(create_tables_query)
    populate_database(sus+srs, "full", 5, 0, "nchars", 10000000, check_for_duplicate=False)
    populate_database(sus+srs, "kth_in_sent", 5, 5, "nchars", 10000000, check_for_duplicate=False)
    exec_q(create_indices_query)

