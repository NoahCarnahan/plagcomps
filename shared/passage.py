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
		self.plag_spans = []

	def add_plag_span(self, span):
		self.plag_spans.append(span)


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

	@staticmethod
	def serialization_header(feature_names):
		return ['start_index','end_index'] + \
			   feature_names + ['plag_confidence', 'contains_plag']


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

	def to_list(self, feature_names):
		'''
		Returns a list representation of this passage. Follows the 
		template outlined in <IntrinsicPassage.serialization_header> 
		i.e.
		start_index, end_index, feat1, feat2, ..., plag_confidence, contains_plag

		Note that feat1, feat2, ... are returned in the order of the features
		passed in from <feature_names>
		'''
		feature_vals = [self.features[name] for name in feature_names]

		plag_conf = self.plag_confidence if self.plag_confidence else 0
		contains_plag = 1 if len(self.plag_spans) > 0 else 0
		serialized = [self.char_index_start, self.char_index_end] + \
					 feature_vals + [plag_conf, contains_plag]
		
		return serialized


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


	def get_relative_plag_spans(self):
		'''
		Returns the span, relative to the text stored in <self.text>,
		of the plagiarized span 
		'''
		rel_spans = []

		for span in self.plag_spans:
			start = span[0] - self.char_index_start
			end = span[1] - self.char_index_start
			
			rel_spans.append((start, end))

		# Sorts by first element by default
		rel_spans.sort()

		return rel_spans

	def format_text_with_underlines(self):
		'''
		Returns HTML for <self.text> including markup to underline 
		plagiarized passages
		'''
		rel_spans = self.get_relative_plag_spans()
		html = ''
		last_end = 0

		for cur_span in rel_spans:
			cur_start, cur_end = cur_span

			html += self.text[last_end : cur_start]
			html += '<span style="text-decoration: underline;">'
			html += self.text[cur_start : cur_end]
			html += '</span>'
			last_end = cur_end

		html += self.text[last_end:]
	
		return html
	
	def to_html(self):
		'''
		Returns an HTML representation of this passage's features, for use 
		by the plagapp front-end
		'''
		
		html = '<p>Span</p>'
		html += '<p>%i, %i</p>' % (self.char_index_start, self.char_index_end)
		html += '<hr size = "10"'

		html += '<p>Plag. conf.</p>'
		html += '<p>%s</p>' % str(self.plag_confidence) 
		html += '<hr size = "10"'

		html += '<p>PLAG SPAN</p>'
		if len(self.plag_spans) > 0:
			for sp in self.plag_spans:
				html += '<p>(%i, %i)</p>' % sp
		else:
			html += '<p>No plag!</p>'
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
