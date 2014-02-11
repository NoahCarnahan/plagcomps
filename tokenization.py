import nltk
import re
import string

def tokenize(text, atom_type, n=5000, return_spans=True):
    '''
    By default, return a list of spans designating the location of 
    each passage in the given text. atom_type determines the type of passages.
    (output is of the form [(0, 20), (22, 50), ...])

    If <return_spans> is False, return a list of the actual tokens
    '''
    if atom_type == "word":
        return _tokenize_by_word(text, return_spans)
    elif atom_type == "sentence":
        return _tokenize_by_sentence(text, return_spans)
    elif atom_type == "paragraph":
        return _tokenize_by_paragraph(text, return_spans)
    elif atom_type == "full":
    	return _tokenize_by_full(text, return_spans)
    elif atom_type == "nchars":
        return _tokenize_by_n_chars(text, n, return_spans)
    else:
        raise ValueError("Unacceptable atom_type")
    
def _tokenize_by_word(text, return_spans):
    tokenizer = _CopyCatPunktWordTokenizer()

    if return_spans:
        return tokenizer.span_tokenize(text)
    else:
        return tokenizer.tokenize(text)
    
def _tokenize_by_sentence(text, return_spans):
    tokenizer = nltk.PunktSentenceTokenizer()

    if return_spans:
        return tokenizer.span_tokenize(text)
    else:
        return tokenizer.tokenize(text)

def _tokenize_by_paragraph(text, return_spans):
    # Idea from this gem: http://stackoverflow.com/a/4664889/3083983
    # boundaries[i][0] == start of <ith> newline sequence (i.e. 2+ newlines)
    # boundaries[i][1] == end of <ith> newline sequence (i.e. 2+ newlines)
    PARAGRAPH_RE = r'\s*\n{2,}\s*'
    boundaries = [(m.start(), m.end()) for m in re.finditer(PARAGRAPH_RE, text)]

    if len(boundaries) == 0:
        spans = [(0, len(text))]
        tokens = [text[spans[0][0] : spans[0][1]]]
    else:
        spans = [(0, boundaries[0][0])]
        tokens = [text[spans[0][0] : spans[0][1]]]
        for i in range(len(boundaries) - 1):
            cur_span = (boundaries[i][1], boundaries[i + 1][0])
            cur_token = text[cur_span[0] : cur_span[1]]
            spans.append(cur_span)
            tokens.append(cur_token)

        # NOTE could be an edge-case if there's no new-line at the end of the text
        if boundaries[-1][1] != len(text):
            cur_span = (boundaries[-1][1], len(text))
            cur_token = text[cur_span[0] : cur_span[1]]
            spans.append(cur_span)
            tokens.append(cur_token)
    
    if return_spans:
        return spans
    else:
        return tokens

def _tokenize_by_n_chars(text, n, return_spans=True):
    '''
    Returns spans of length <n> from text. The last spans "snaps out" to include
    all characters meaning that the last span is likely to be longer than <n> characters. 
    For example, if <text> is of length
    65 and n=20, then we get spans:
    [(0, 20), (20, 40), (40, 65)]
    '''
    spans = []
    text_spans = []

    if len(text) < n:
        spans.append((0, len(text)))
        text_spans.append(text)
    else:
        start_index = 0
        end_index = n

        while end_index < len(text) - n + 1:
            spans.append((start_index, end_index))
            text_spans.append(text[start_index : end_index])
            start_index = end_index
            end_index = min(end_index + n, len(text))

        # Set the last span's ending index to be the end of document
        spans.append((start_index, len(text)))
        text_spans.append(text[start_index :])

    if return_spans:
        return spans
    else:
        return text_spans


def strip_punctuation(words):
    '''
    Returns all w in <words> such that w is not a punctuation character
    If w starts or ends with punctuation, the leading/trailing punctuation
    is stripped in the returned list
    '''
    # See http://stackoverflow.com/a/266162/3083983 -- translate is fast!
    # TODO do we want to strip all punctuation? 
    return [w.translate(None, string.punctuation) \
            for w in words if w not in string.punctuation]


def _tokenize_by_full(text, return_spans):
    if return_spans:
	   return [(0, len(text))]
    else:
        return [text]


class _CopyCatPunktWordTokenizer(nltk.tokenize.punkt.PunktBaseClass,nltk.tokenize.punkt.TokenizerI):
    '''
    Class that extends the nltk PunktWordTokenizer. Unfortunately, PunktWordTokenizer doesn't 
    implement the span_tokenize() method, so we implemented it here.
    '''
    def __init__(self, train_text=None, verbose=False, lang_vars=nltk.tokenize.punkt.PunktLanguageVars(), token_cls=nltk.tokenize.punkt.PunktToken):
        nltk.tokenize.punkt.PunktBaseClass.__init__(self, lang_vars=lang_vars, token_cls=token_cls)

    def tokenize(self, text):
        '''Returns a list of strings that are the individual words of the given text.'''
        return self._lang_vars.word_tokenize(text)
    
    def span_tokenize(self, text):
        '''Returns a list of tuples, each containing the start and end index for the respective
            words returned by tokenize().'''
        return [(sl[0], sl[1]) for sl in self._slices_from_text(text)]

    def _slices_from_text(self, text):
        last_break = 0
        indices = []
        for match in self._lang_vars._word_tokenizer_re().finditer(text):
            context = match.group()
            indices.append((match.start(), match.end()))
        return indices

def _test():

    text = "When in the Course of human events, it becomes necessary for one people to dissolve the political bands which have connected them with another, and to assume among the powers of the earth, the separate and equal station to which the Laws of Nature and of Nature's God entitle them, a decent respect to the opinions of mankind requires that they should declare the causes which impel them to the separation.\n\nWe hold these truths to be self-evident, that all men are created equal, that they are endowed by their Creator with certain unalienable Rights, that among these are Life, Liberty and the pursuit of Happiness.--That to secure these rights, Governments are instituted among Men, deriving their just powers from the consent of the governed, --That whenever any Form of Government becomes destructive of these ends, it is the Right of the People to alter or to abolish it, and to institute new Government, laying its foundation on such principles and organizing its powers in such form, as to them shall seem most likely to effect their Safety and Happiness. Prudence, indeed, will dictate that Governments long established should not be changed for light and transient causes; and accordingly all experience hath shewn, that mankind are more disposed to suffer, while evils are sufferable, than to right themselves by abolishing the forms to which they are accustomed. But when a long train of abuses and usurpations, pursuing invariably the same Object evinces a design to reduce them under absolute Despotism, it is their right, it is their duty, to throw off such Government, and to provide new Guards for their future security.--Such has been the patient sufferance of these Colonies; and such is now the necessity which constrains them to alter their former Systems of Government. The history of the present King of Great Britain is a history of repeated injuries and usurpations, all having in direct object the establishment of an absolute Tyranny over these States. To prove this, let Facts be submitted to a candid world."
    print text
    
    print tokenize(text, "word")
    print tokenize(text, "sentence")
    print tokenize(text, "paragraph")
    
    print text[0:4] # test a word
    print text[408:1063] #test a sentence
    print text[0:406] # test the paragraphs
    print text[408:2036]
    
    text = "The brown fox ran."
    print text[tokenize(text, "word")[-1][0]:tokenize(text, "word")[-1][1]]
    #TODO: Words include punctuation. Is it like this in the original version !?
    
if __name__ == "__main__":
    _test()