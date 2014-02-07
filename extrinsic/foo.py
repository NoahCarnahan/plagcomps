import psycopg2
import os
import fingerprint_extraction
from ..dbconstants import username, password, dbname
from ..shared.util import ExtrinsicUtility
from ..tokenization import tokenize


def populate_database(files, method, n, k, atom_type, hash_len):
    '''
    Example usages:
    srs, sus = ExtrinsicUtility().get_training_files(n=10)
    populate_database(srs+sus, "kth_in_sent", n, k, "paragraph", 10000)
    
    '''
        
    conn = psycopg2.connect(user = username, password = password, database = dbname.split("/")[1], host="localhost", port = 5432)
    conn.autocommit = True
    
    for abs_path in files:
    
        # Get doc_name, doc_path, doc_xml_path, and is_source
        try:
            abs_path.index("source")
            base_path = ExtrinsicUtility().CORPUS_SRC_LOC
            is_source = True
        except ValueError, e:
            base_path = ExtrinsicUtility().CORPUS_SUSPECT_LOC
            is_source = False
        doc_path = abs_path
        doc_xml_path = abs_path.replace(".txt", ".xml")
        doc_name = abs_path.replace(base_path, "").replace(".txt", "")
        
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
        for atom_number in range(len(atom_texts)):
            
            # Insert the passage into the database and get its id
            cur.execute("INSERT INTO passage_fingerprints (doc_name, doc_path, doc_xml_path, atom_number, is_source, atom_type, method, n, k, hash_len) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id;",
                (doc_name, doc_path, doc_xml_path, atom_number, is_source, atom_type, method, n, k, hash_len))
            passage_id = str(cur.fetchone()[0])
            
            # Fingerprint the passage
            extractor = fingerprint_extraction.FingerprintExtractor(hash_len)
            hash_values = extractor.get_fingerprint(atom_texts[atom_number], n, method, k)
            
            # Add rows to the copy_file_string
            for hash_value in hash_values:
                copy_file_string += passage_id + "\t" + str(hash_value) + "\n"
        
        # write copy_file_string to a file named foo.txt
        f = open("/tmp/dbimport.txt", "w")
        f.write(copy_file_string)
        f.close()
        
        cur.execute("COPY hash_table FROM '/tmp/dbimport.txt'")
        cur.close()
        
        os.remove("/tmp/dbimport.txt")
    
    conn.close()
            
def _test():
    srs, sus = ExtrinsicUtility().get_training_files(n=1)
    populate_database(srs+sus, "kth_in_sent", 4, 3, "paragraph", 10000)

if __name__ == "__main__":
    _test()
