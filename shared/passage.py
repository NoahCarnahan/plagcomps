class Passage:
	'''
	A class to store/query information about a passage of text, including
	what chunk it refers to (text from char_index_start (inclusive) to 
	char_index_end (exclusive))

	Should be subclassed to do more specialized work for intrinsic/extrinsic passages
	'''
	
	def __init__(self, char_index_start, char_index_end, text):
		self.char_index_start = char_index_start
		self.char_index_end = char_index_end
		self.text = text

class PassageWithGroundTruth(Passage):
	'''
	'''
	def __init__(self, char_index_start, char_index_end, text, plag_span):
		Passage.__init__(self, char_index_start, char_index_end, text)
		self.plag_span = plag_span

	def to_html(self):
		'''
		Returns an HTML representation of this passage's features, for use 
		by the plagapp front-end
		'''
		html = '<p>Plag. Span</p>'
		html += '<p>%s</p>' % str(self.plag_span) 
		html += '<hr size = "10"'

		return html


class IntrinsicPassage(Passage):
	'''
	Stores additional fields for an intrinsic passage, namely 
	<atom_type> ('word', 'sentence', or 'paragraph')
	and <features> (and dictionary mapping feature -> observed value for that feature)
	'''
	def __init__(self, char_index_start, char_index_end, text, atom_type, features={}):
		Passage.__init__(self, char_index_start, char_index_end, text)
		self.atom_type = atom_type
		self.features = features
		self.plag_confidence = None

	def __str__(self):
		return 'Text: ' + self.text + \
		'\nFeatures: ' + str(self.features) +  \
		'\nPlag Confidence: ' + str(self.plag_confidence) + \
		'\n------------'

	def to_json(self):
		'''
		Returns a dictionary representation of the passage
		'''
		json_rep = {
			'atom_type' : self.atom_type,
			'char_index_start' : self.char_index_start,
			'char_index_end' : self.char_index_end,
			'text' : self.text,
			'features' : self.features
		}
		
		return json_rep


	def to_html(self):
		'''
		Returns an HTML representation of this passage's features, for use 
		by the plagapp front-end
		'''
		html = '<p>Plag. conf.</p>'
		html += '<p>%s</p>' % str(self.plag_confidence) 
		html += '<hr size = "10"'

		for feat, val in self.features.iteritems():
			html += '<p>%s</p>' % feat
			html += '<p>%.4f</p>' % val
			html += '<hr size = "10"'

		return html

	def set_plag_confidence(self, confidence):
		'''
		Sets how confident we are this passage was plagiarized
		'''
		self.plag_confidence = confidence

class ExtrinsicPassage(Passage):
    '''
    Stores additional fields for an extrinsic passage, namely 
    <atom_type> ('paragraph')
    and <features> (and dictionary mapping feature -> observed value for that feature)
    '''
    def __init__(self, char_index_start, char_index_end, text, atom_type, fingerprint):
        Passage.__init__(self, char_index_start, char_index_end, text)
        self.atom_type = atom_type
        self.fingerprint = fingerprint
        self.plag_confidence = None

    def __str__(self):
        return 'Text: ' + self.text + \
        '\nFingerprint: ' + str(self.features) +  \
        '\nPlag Confidence: ' + str(self.plag_confidence) + \
        '\n------------'

    def to_json(self):
        '''
        Returns a dictionary representation of the passage
        '''
        json_rep = {
            'atom_type' : self.atom_type,
            'char_index_start' : self.char_index_start,
            'char_index_end' : self.char_index_end,
            'text' : self.text,
            'fingerprint' : self.fingerprint
        }
        return json_rep

    def to_html(self):
        '''
        Returns an HTML representation of this passage's fingerprint, for use 
        by the plagapp front-end
        '''
        html = '<p>Plag. conf.</p>'
        html += '<p>%s</p>' % str(self.plag_confidence) 
        html += '<hr size = "10"'

        for minutia in self.fingerprint:
            html += '<p>%d</p>' % minutia
            html += '<hr size = "10"'
        return html
