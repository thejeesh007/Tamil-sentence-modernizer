from indicnlp.tokenize.sentence_tokenize import sentence_split
from indicnlp.tokenize.indic_tokenize import trivial_tokenize

def tokenize_text(text, lang='hi'):
    return trivial_tokenize(text, lang)

def split_sentences(text, lang='hi'):
    return sentence_split(text, lang)
