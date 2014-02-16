import datetime

from ..shared.util import ExtrinsicUtility
import fingerprint_extraction
from ..tokenization import tokenize
from ..dbconstants import username, password, dbname

from sqlalchemy import create_engine, Table, Column, Integer, String, MetaData, ForeignKey, DateTime, Sequence, Boolean
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.sql import select, and_

url = "postgresql://%s:%s@%s" % (username, password, dbname)
engine = create_engine(url)
metadata = MetaData()

fingerprints_table = Table("stcore_fingerprints", metadata,
    Column("id", Integer, Sequence("stcore_fingerprint_id_seq"), primary_key=True, index=True),
    Column("timestamp", DateTime),
    Column("doc_name", String, index = True),
    Column("doc_path", String),
    Column("doc_xml_path", String),
    Column("method", String), # The fingerprinting method used to create this fingerprint
    Column("n", Integer),
    Column("k", Integer),
    Column("atom_type", String),
    Column("atom_number", Integer, index = True), # If this is the 1th paragraph in the document, then atom_number = 1
    Column("hash_indexed", Boolean), # If true, the hash_values of this fingerprint will be put into the hash index table.
    Column("hash_values", ARRAY(Integer))
)

hash_index_table = Table("stcore_hash_index", metadata,
    Column("id", Integer, Sequence("stcore_hash_index_id_seq"), primary_key = True),
    Column("hash_value", Integer, index=True),
    Column("method", String),
    Column("n", Integer),
    Column("k", Integer),
    Column("atom_type", String),
    Column("fingerprint_ids", ARRAY(Integer))    
)

#metadata.drop_all(engine)
metadata.create_all(engine)

class Fingerprint:
    def __init__(self, id, timestamp, doc_name, doc_path, doc_xml_path, method, n, k, atom_type, atom_number, hash_indexed, hash_values):
        self.id = id
        self.timestamp = timestamp
        self.doc_name = doc_name
        self._doc_path = doc_path
        self._doc_xml_path = doc_xml_path
        self.method = method
        self.n = n
        self.k = k
        self.atom_type = atom_type
        self.atom_number = atom_number
        self.hash_indexed = hash_indexed
        self.hash_values = hash_values

class HashIndex:
    def __init__(self, id, hash_value, method, n, k, atom_type, fingerprint_ids):
        self.id = id
        self.hash_value = hash_value
        self.method = method
        self.n = n
        self.k = k
        self.atom_type = atom_type
        self.fingerprint_ids = fingerprint_ids  

def get_fingerprints(doc_name, base_path, method, n, k, atom_type, hash_indexed, create = True):
    '''
    Retrieve the Fingerprint objects from the database with the given parameters. This function will
    return a Fingerprint for each atom in this document. They are NOT necessarily in order.
    New Fingerprint objects will be created if they do not already exist in the database.
    '''
    
    # WILL THIS BREAK IF ATOM_TYPE IS FULL? (tokenize might not know what to do)
    abs_path = base_path + doc_name + ".txt"
    f = open(abs_path, "r")
    text = f.read()
    f.close()
    atom_spans = tokenize(text, atom_type)
    
    print "atoms =", len(atom_spans)

    fingerprints = []
    for i in range(len(atom_spans)):
        atom_text = text[atom_spans[i][0]:atom_spans[i][1]]
        fp = get_fingerprint(doc_name, base_path, method, n, k, atom_type, i, hash_indexed, create, atom_text)
        fingerprints.append(fp)
    return fingerprints

def get_fingerprint(doc_name, base_path, method, n, k, atom_type, atom_number, hash_indexed, create=True, atom_text = None):
    '''
    Retrieve the Fingerprint object from the database with the given parameters. If create=True,
    then a new Fingerprint will be created and returned in the event that it does not already
    exist in the database.
    '''
    
    # Sanitize the input
    if method not in ["kth_in_sent", "winnow-k"]:
        k = 0
    
    conn = engine.connect()
    
    # See if the fingerprint is in the database
    fpt = fingerprints_table
    s = select([fpt]).where(and_(fpt.c.doc_name == doc_name, fpt.c.method == method, fpt.c.n == n, fpt.c.k == k, fpt.c.atom_type == atom_type, fpt.c.atom_number == atom_number))
    result = conn.execute(s)
    row = result.fetchone()
    if row != None:
        conn.close()
        return Fingerprint(*row)
    else:
        if create:
            # Fingerprint is not in the database, so create it!
            fp_dic = build_fingerprint_row(doc_name, base_path, method, n, k, atom_type, atom_number, hash_indexed, atom_text = atom_text)
            ins = fingerprints_table.insert().values(fp_dic)
            result = conn.execute(ins)
            key = result.inserted_primary_key[0]
            fp_dic["id"] = key
            fp = Fingerprint(*fp_dict_to_tup(fp_dic))
            
            # UPDATE HASH INDEX!!
            _update_hash_index(fp)
            
            conn.close()
            return fp
        else:
            conn.close()
            return None

def build_fingerprint_row(doc_name, base_path, method, n, k, atom_type, atom_number, hash_indexed, atom_text = None):
    '''
    Given the arguments that would have been passed to a Fingerprint object constructor in previous versions,
    return a dictionary that can be insrterd into the db.
    '''
    if method not in ["kth_in_sent", "winnow-k"]:
        k = 0
        
    dict = {"doc_name":doc_name, "method":method, "n":n, "k":k, "atom_type":atom_type, "atom_number":atom_number, "hash_indexed":hash_indexed, "timestamp":datetime.datetime.now()}
    
    if base_path in doc_name:
        doc_path = doc_name + ".txt"
        doc_xml_path = doc_name + ".xml"
    else:
        doc_path = base_path + doc_name + ".txt"
        doc_xml_path = base_path + doc_name + ".xml"
        
    dict["doc_path"] = doc_path
    dict["doc_xml_path"] = doc_xml_path
    
    if atom_text == None:
        f = open(doc_path, "r")
        text = f.read()
        f.close()
        if atom_type == "full":
            atom_text = text
        elif atom_type == "paragraph":
            paragraph_spans = tokenize(text, atom_type)
            atom_start, atom_end = paragraph_spans[atom_number]
            atom_text = text[atom_start:atom_end]
        else:
            raise ValueError("Invalid atom_type! Only 'full' and 'paragraph' are allowed.")

    extractor = fingerprint_extraction.FingerprintExtractor()
    hash_values = extractor.get_fingerprint(atom_text, n, method, k)
    dict["hash_values"] = hash_values
    
    return dict

def fp_dict_to_tup(d):
    '''
    Turns a dictionary into a tuple that can be used by the Fingerprint constructor
    '''
    return (d["id"], d["timestamp"], d["doc_name"], d["doc_path"], d["doc_xml_path"], d["method"], d["n"], d["k"], d["atom_type"], d["atom_number"], d["hash_indexed"], d["hash_values"])
    
def _update_hash_index(fp):
    if fp.hash_indexed == True:
        conn = engine.connect()
        for hash in set(fp.hash_values):
            s = select([hash_index_table]).where(and_(hash_index_table.c.hash_value == hash, hash_index_table.c.method == fp.method, hash_index_table.c.n == fp.n, hash_index_table.c.k == fp.k, hash_index_table.c.atom_type == fp.atom_type))
            result = conn.execute(s)
            hash_index_tup = result.fetchone()
            if hash_index_tup != None:
                new_id_list = hash_index_tup[hash_index_table.c.fingerprint_ids]
                new_id_list.append(fp.id)
                # UPDATE THE ROW!
                stmt = hash_index_table.update().where(hash_index_table.c.id == hash_index_tup["id"]).values(fingerprint_ids = new_id_list)
                #stmt = hash_index_table.update().values(new_tup)
                conn.execute(stmt)
            else:
                stmt = hash_index_table.insert().values({"hash_value":hash, "method":fp.method, "n":fp.n, "k":fp.k, "atom_type":fp.atom_type, "fingerprint_ids":[fp.id]})
                conn.execute(stmt)
        conn.close()

def get_fingerprints_by_hash(hash, method, n, k, atom_type):
    if method not in ["winnow-k", "kth_in_sent"]:
        k = 0
    conn = engine.connect()
    s = select([hash_index_table]).where(and_(hash_index_table.c.hash_value == hash, hash_index_table.c.method == method, hash_index_table.c.n == n, hash_index_table.c.k == k, hash_index_table.c.atom_type == atom_type))
    result = conn.execute(s)
    row = result.fetchone()
    if row == None:
        conn.close()
        return []
    else:
        fps = []
        fp_ids = row["fingerprint_ids"]
        # select them
        s = select([fingerprints_table]).where(fingerprints_table.c.id.in_(fp_ids))
        result = conn.execute(s)
        for row in result:
            fps.append(Fingerprint(*row))
        conn.close()
        return fps

def populate_db(absolute_paths, method, n, k, atom_type):
    '''
    Populate the database with Fingerprints and HashIndexs for each document in the
    absolute_paths list. Use the fingerprint method, n, and k given. If hash_indexed is
    True then the hash index will be populated for these documents as well.
    '''
        
    num_populated = 0
    for abs_path in absolute_paths:
        try:
            abs_path.index("source")
            base_path = ExtrinsicUtility().CORPUS_SRC_LOC
            hash_index = True
        except ValueError, e:
            base_path = ExtrinsicUtility().CORPUS_SUSPECT_LOC
            hash_index = False
            
        filename = abs_path.replace(base_path, "").replace(".txt", "")
        print "Populating doc", num_populated+1, "of", len(absolute_paths), ":", filename, "-", str(datetime.datetime.now()), "-", "(", method, n, k, atom_type,")"
        num_populated += 1
        get_fingerprints(filename, base_path, method, n, k, atom_type, hash_index)

def main():
    srs, sus = ExtrinsicUtility().get_training_files(n=10)
    
    populate_db(sus+srs, "anchor", 5, 0, "paragraph")
    #populate_db(sus+srs, "anchor", 3, 0, "paragraph")
    #populate_db(sus+srs, "full", 5, 0, "paragraph")
    #populate_db(sus+srs, "full", 3, 0, "paragraph")
    #populate_db(sus+srs, "kth_in_sent", 5, 5, "paragraph")
    #populate_db(sus+srs, "kth_in_sent", 5, 3, "paragraph")
    #populate_db(sus+srs, "kth_in_sent", 3, 5, "paragraph")
    #populate_db(sus+srs, "kth_in_sent", 3, 3, "paragraph")
    #populate_db(sus+srs, "winnow-k", 5, 5, "paragraph")
    #populate_db(sus+srs, "winnow-k", 5, 3, "paragraph")
    #populate_db(sus+srs, "winnow-k", 3, 5, "paragraph")
    #populate_db(sus+srs, "winnow-k", 3, 3, "paragraph")
       
def _test():
    '''
    Retrieve fingerprints from two documents, twice. Then do a hash index look up.
    '''
    # Fingerprints
    base_path = "/copyCats/itty-bitty-corpus/source"
    for j in range(2): # Do it twice, once where the objects are created, and a second where they are retrieved.
        for path in ["/source-born","/source-feynman"]:
            f = open(base_path+path+".txt", "r")
            text = f.read()
            f.close()
            for i in range(len(tokenize(text, "paragraph"))):
                fp = get_fingerprint(path, base_path, "kth_in_sent", 5, 5, "paragraph", i, True)
                fp = get_fingerprint(path, base_path, "anchor", 5, 5, "paragraph", i, True)
                print fp.hash_values
    
    # HashIndex
    print fp.hash_values[0]
    print get_fingerprints_by_hash(fp.hash_values[0], "anchor", 5, 5, "paragraph")



if __name__ == "__main__":
    main()