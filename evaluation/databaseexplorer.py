from ..dbconstants import username
from ..dbconstants import password
from ..dbconstants import dbname
from ..shared.util import IntrinsicUtility

import sqlalchemy
from sqlalchemy import and_
from sqlalchemy.orm import sessionmaker

from intrinsic import ReducedDoc

# an Engine, which the Session will use for connection resources
url = "postgresql://%s:%s@%s" % (username, password, dbname)
engine = sqlalchemy.create_engine(url)
# create a configured "Session" class
Session = sessionmaker(bind=engine)

def retreive_feat_vect(doc_name, atom_type, feature, version):
    '''
    Queries the database for the feature vector associated with the given parameters. Returns False
    if it does not exist.
    
    '''
    # NOTE NOTE NOTE: THIS FUNCTION DOES NOT WORK (for some unknown reason)
    session = Session()
    q = session.query(ReducedDoc).filter(and_(ReducedDoc.full_path == doc_name, ReducedDoc.atom_type == atom_type, ReducedDoc.version_number == version))
    r = q.one()
    print r.id
    feature_values = r._get_feature_values(feature, session, populate = False)
    
    return feature_values
    
#def delete_feat_vect(

if __name__ == "__main__":
    for name in IntrinsicUtility().get_n_training_files(1):
        print name
        print retreive_feat_vect(name, "paragraph", "stopword_percentage", 5)