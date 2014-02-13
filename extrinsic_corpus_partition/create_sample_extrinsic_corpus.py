from plagcomps.shared.util import ExtrinsicUtility
import os.path

def create(suspect_output_filename, source_output_filename, n=1000):
    '''
    Creates a smaller corpus from the corpus defined in ExtrinsicUtility's
    TRAINING_SUSPECT_LOC and TRAINING_SRC_LOC paths.  The number of suspect
    documents in the generated corpus is <n>.
    '''
    util = ExtrinsicUtility()
    print 'Getting ' + str(n) + ' suspect files from ' + str(util.TRAINING_SUSPECT_LOC) + '...'
    n_suspects = util.get_n_training_files(n, 'suspect', include_txt_extension=False)

    all_sources = set()
    for suspect in n_suspects:
        xml_path = suspect + '.xml'
        all_sources.update(util.get_source_plag(xml_path))

    write_src_listing(all_sources, source_output_filename)
    write_suspect_listing(n_suspects, suspect_output_filename)

    
def write_src_listing(srcs, source_output_filename):
    loc = os.path.join(os.path.dirname(__file__), source_output_filename)

    f =  file(loc, 'wb')
    for src in srcs:
        f.write(src + '\n')
    f.close()

    print 'Wrote ' + str(len(srcs)) + ' source listings to ' + loc


def write_suspect_listing(suspects, suspect_output_filename):
    loc = os.path.join(os.path.dirname(__file__), suspect_output_filename)

    f =  file(loc, 'wb')
    for suspect in suspects:
        f.write(suspect + '\n')
    f.close()

    print 'Wrote ' + str(len(suspects)) + ' suspect listings to ' + loc


if __name__ == '__main__':
    create('small_crisp_corpus_suspect_files.txt', 'small_crisp_corpus_source_files.txt', n=50)
