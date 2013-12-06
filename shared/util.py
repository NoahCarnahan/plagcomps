class BaseUtility:
    '''
    Utility functions common to both extrinsic and intrinsic
    '''

    def read_file_list(self, file_name, base_location_path):
        '''
        Return list of absolute paths to files in <file_name> whose
        location is relative to <base_location_path>
        '''
        f = file(file_name, 'r')
        relative_paths = [l.strip() + '.txt' for l in f.readlines()]
        training_list = [base_location_path + r for r in relative_paths]

        return training_list

class IntrinsicUtility(BaseUtility):

    TRAINING_LOC = 'corpus_partition/training_set_files.txt'
    CORPUS_LOC = '/copyCats/pan-plagiarism-corpus-2009/intrinsic-detection-corpus/suspicious-documents'

    def get_n_training_files(self, n=None):
        '''
        Returns first <n> training files, or all of them if <n> is not specified
        '''
        all_training_files = self.read_file_list(IntrinsicUtility.TRAINING_LOC, 
                                                 IntrinsicUtility.CORPUS_LOC)

        # Default to using all training files if <n> isn't specified
        n = len(all_training_files) if n is None else n

        return all_training_files[:n]

class ExtrinsicUtility(BaseUtility):

    TRAINING_SRC_LOC = 'extrinsic_corpus_partition/extrinsic_training_source_files.txt'
    TRAINING_SUSPECT_LOC = 'extrinsic_corpus_partition/extrinsic_training_suspect_files.txt'

    CORPUS_SRC_LOC = '/copyCats/pan-plagiarism-corpus-2009/external-detection-corpus/source-documents'
    CORPUS_SUSPECT_LOC = '/copyCats/pan-plagiarism-corpus-2009/external-detection-corpus/suspicious-documents'

    def get_n_training_files(self, n=None, file_type='both'):
        '''
        Returns first <n> training files, or all of them if <n> is not specified
        <file_type> should be 'source', 'suspect', or 'both'.
        If 'both', return both lists (source files first, suspect files second)

        NOTE that all source documents are returned, regardless of the size
        of <n>. We always need to know all source documents in order
        to detect plagiarism in any of the suspicious ones!
        '''
        all_src_files = self.read_file_list(ExtrinsicUtility.TRAINING_SRC_LOC,
                                            ExtrinsicUtility.CORPUS_SRC_LOC)

        all_suspect_files = self.read_file_list(ExtrinsicUtility.TRAINING_SUSPECT_LOC,
                                                ExtrinsicUtility.CORPUS_SUSPECT_LOC)

        
        n = len(all_suspect_files) if n is None else n

        if file_type == 'source':
            return all_src_files
        elif file_type == 'suspect':
            return all_suspect_files[:n]
        elif file_type == 'both':
            return all_src_files, all_suspect_files[:n]
        else:
            raise Exception("Argument <file_type> must be 'source', 'suspect' or 'both'")




