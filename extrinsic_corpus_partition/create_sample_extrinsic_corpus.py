from plagcomps.shared.util import ExtrinsicUtility
import os.path

def create(n=1000):
    util = ExtrinsicUtility()
    n_suspects = util.get_n_training_files(n, 'suspect', include_txt_extension=False)

    all_sources = set()
    for suspect in n_suspects:
        xml_path = suspect + '.xml'
        all_sources.update(util.get_source_plag(xml_path))

    write_src_listing(all_sources)
    write_suspect_listing(n_suspects)

    
def write_src_listing(srcs):
    loc = os.path.join(os.path.dirname(__file__), 'small_sample_corpus/sample_source_listing.txt')

    f =  file(loc, 'wb')
    for src in srcs:
        f.write(src + '\n')
    f.close()

    print 'Wrote src listing to', loc


def write_suspect_listing(suspects):
    loc = os.path.join(os.path.dirname(__file__), 'small_sample_corpus/sample_suspect_listing.txt')

    f =  file(loc, 'wb')
    for suspect in suspects:
        f.write(suspect + '\n')
    f.close()

    print 'Wrote suspects listing to', loc

if __name__ == '__main__':
    create()
