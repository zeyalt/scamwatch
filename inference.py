# -*- coding: utf-8 -*-

# =============================================================================
# Load necessary libraries
# =============================================================================

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import spacy
from spacy.matcher import Matcher
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import re
import math
import ast
# import plotly.express as px
import networkx as nx
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import warnings
warnings.simplefilter("ignore")



nlp = spacy.load('en_core_web_sm')

# =============================================================================
# Define functions
# =============================================================================

def read_csv(path):
    
    """This function reads a CSV file from a specified filepath, while preserving the data types of each variable.
    Source: https://stackoverflow.com/questions/50047237/how-to-preserve-dtypes-of-dataframes-when-using-to-csv/50051542#50051542"""
    
    # Read types first line of csv
    dtypes = {key:value for (key,value) in pd.read_csv(path, nrows=1).iloc[0].to_dict().items() if 'date' not in value}

    parse_dates = [key for (key,value) in pd.read_csv(path, 
                   nrows=1).iloc[0].to_dict().items() if 'date' in value]
    
    # Read the rest of the lines with the types from above
    return pd.read_csv(path, dtype=dtypes, parse_dates=parse_dates, skiprows=[1])


def get_noun_phrases(sent):
    
    """This function takes a text string as input, and returns a list of noun phrases extracted from the text string"""
    
    doc = nlp(sent)
    matcher = Matcher(nlp.vocab)  
    pattern_1 = [{"POS": "NOUN", "OP": "+"}]
    pattern_2 = [{"POS": "ADJ"}, {"POS": "NOUN", "OP": "+"}]
    matcher.add("Noun_phrases", None, pattern_1, pattern_2)
    matches = matcher(doc)

    span = []
    for match_id, start, end in matches:
        span.append(doc[start:end].text)

    return span


def get_jaccard_sim(str1, str2): 
    
    """This function takes two lists of tokens as inputs and computes the Jaccard similarity between them."""
    
    a = set(str1) 
    b = set(str2)
    c = a.intersection(b)
    
    return float(len(c)) / (len(a) + len(b) - len(c) + 0.0001)


def find_similar_docs_cosine_jaccard(model, corpus, data, tag_id=None, new_doc=None):    
    
    
    """This function takes a Doc2Vec model, corpus and tag ID or text string as inputs and 
    returns a dataframe with the most similar documents and their cosine and 
    Jaccard similarity scores."""
            
    if tag_id is not None:
        infer_vector = model.infer_vector(corpus[tag_id].words)
        candidate_sentence = ' '.join(corpus[tag_id].words).replace('< ', '<').replace(' >', '>').replace(' ,', ',').replace(' .', '.')
        candidate_noun_phrases = get_noun_phrases(candidate_sentence)

    elif new_doc is not None:
        infer_vector = model.infer_vector(word_tokenize(new_doc.lower()))
        candidate_noun_phrases = get_noun_phrases(new_doc.lower())

    similar_documents = model.docvecs.most_similar([infer_vector], topn = len(model.docvecs))
    cosine_df = pd.DataFrame(similar_documents, columns=['tag', 'cosine'])
    
    similarity_scores_list = []
    for row, _ in data.iterrows():
        tag = similar_documents[row][0]
        text = data.loc[tag, 'incident_description']
        pp_text = data.loc[tag, 'preprocessed_text']
        scam_type = data.loc[tag, 'scam_type']
        noun_phrases = ast.literal_eval(data.loc[tag, 'noun_phrases'])
        jaccard = get_jaccard_sim(candidate_noun_phrases, noun_phrases)
        similarity_scores_list.append((tag, jaccard, text, pp_text, scam_type))

    jaccard_df = pd.DataFrame(similarity_scores_list, columns=['tag', 'jaccard', 'text', 'pp_text', 'scam_type'])
    cosine_jaccard_df = pd.merge(cosine_df, jaccard_df, how="inner", on="tag")
    cosine_jaccard_df['cosine'] = cosine_jaccard_df['cosine'].round(3)
    cosine_jaccard_df['jaccard'] = cosine_jaccard_df['jaccard'].round(3)
    
#     return cosine_jaccard_df.head(n)
    return cosine_jaccard_df


def documents_without_stopwords(doc_list):

    """This function takes a list of documents as input and returns a list of documents without stop words."""
    
    # Define a list of additional stopwords to add
    new_stopwords = ['ask', 'said', 'say', 'asked', 'claimed', 'told', 'got', 'tell', 'get']

    # Add new stopwords into existing list
    stop_words = stopwords.words('english')
    for word in new_stopwords:
        stop_words.append(word)
    
    without_stopwords = []
    for i in range(len(doc_list)):
        sentence_without_stopwords = []
        for word in nltk.word_tokenize(doc_list[i]):
            if not word in set(stop_words):
                sentence_without_stopwords.append(word)
        without_stopwords.append(' '.join(sentence_without_stopwords).replace('< ', '<').replace(' >', '>').replace(' ,', ',').replace(' .', '.'))

    return without_stopwords


def extract_n_grams(doc_list, n_min=1, n_max=1, top_n=20):
    
    """This function takes a list of documents as input, and generates top_n n-grams based on TF-IDF scores. 
    1. For unigrams, select n_min = 1 and n_max = 1;
    2. For bigrams, select n_min = 2 and n_max = 2; 
    3. For trigrams, select n_min = 3 and n_max = 3;
    4. For combination, e.g. unigrams and bigrams, select n_min = 1 and n_max = 2."""
    
    vectorizer = TfidfVectorizer(ngram_range=(n_min, n_max))
    vectorizer_vectors = vectorizer.fit_transform(doc_list)

    features = vectorizer.get_feature_names()

    term_doc_df = pd.DataFrame(vectorizer_vectors.toarray())
    term_doc_df.columns = features

    data1 = []
    for index, term in enumerate(features): 
        data1.append((term, term_doc_df[term].sum()))

    ranking_df = pd.DataFrame(data1, columns=['term', 'tfidf_score']).nlargest(top_n, 'tfidf_score')
    
    return term_doc_df, ranking_df


def arrange_n_grams_in_sequence(ranking_df, without_stopwords):
    
    """This function takes a ranked n-gram dataframe as input and arranges them in sequence 
    according to the mean index positions in the corpus of documents."""

    # Define a list of additional stopwords to add
    new_stopwords = ['ask', 'said', 'say', 'asked', 'claimed', 'told', 'got', 'tell', 'get']

    # Add new stopwords into existing list
    stop_words = stopwords.words('english')
    for word in new_stopwords:
        stop_words.append(word)

    ranking_df = ranking_df.assign(sequence="")
    for term_idx, col in ranking_df.iterrows():
        term = col['term']
        start_list = []
        for sent in without_stopwords:
            if re.search(term, sent) != None: 
                start = re.search(term, sent).span()[0]
                start_list.append(start)
        ranking_df.loc[term_idx, 'sequence'] = round(np.median(start_list),2)
    
    return ranking_df.sort_values('sequence')


def create_next_term_column(df):
    
    """This function creates a new column in a dataframe containing the next terms."""
    
    df = df.assign(next_term="")
    for idx, col in df.iterrows():
        try:
            df.loc[idx, 'next_term'] = df.loc[idx+1, 'term']
        except KeyError:
            continue
    return df[['index', 'term', 'next_term', 'tfidf_score', 'sequence']]


def visualise_n_grams(df):
    
    """This function takes a dataframe as input and produces a network graph visualisation."""
    
    score_last = df.loc[len(df)-1, 'tfidf_score']

    # Drop the last row 
    df = df[:len(df)-1]

    # Create network plot as a directed graph
    G = nx.DiGraph()

    # Create connections between nodes
    for idx, col in df.iterrows():
        G.add_edge(col['term'], col['next_term'], weight=(col['tfidf_score']))
    
    weight_last = 100*math.exp(score_last)
    node_size = [100*math.exp(G[u][v]['weight']) for u, v in G.edges()] 
    node_size.append(weight_last)

    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Specify type of network graph
    pos = nx.circular_layout(G)
#     pos = nx.spring_layout(G)
#     pos = nx.kamada_kawai_layout(G)
#     pos = nx.spectral_layout(G)

    # Plot networks
    nx.draw_networkx(G, pos, font_size=16, width=2, edge_color='#000000', node_color='#CCCCCC',
                     node_size = node_size, with_labels = False, ax=ax, arrowstyle='-|>', arrowsize=30)

    # Set background colour
    ax.set_facecolor("#FFFFFF")
    ax.set_frame_on(False)
    
    # Create offset labels
    for key, value in pos.items():
        x, y = value[0]+0.12, value[1]+0.1
        ax.text(x, y, s=key, bbox=dict(facecolor='#FFCC99', alpha=0.95),
                horizontalalignment='center', fontsize=13)

    plt.tight_layout()
    plt.show()
    
    
    
def find_similar_words(model, corpus, word, top_n):
    
    """This function finds the top_n words that are similar to an input word, given a doc2vec model and a corpus."""
    
    similar_words = model.wv.most_similar(word, topn = 100)
    for n in range(top_n):
        print("Word: %s" % similar_words[n][0])
        print("Similarity Score: %f" % similar_words[n][1], "\n")
    
    
    
def insert_break(text, n):
    
    """This function inserts <br> at every n words in a text."""
    
    a = text.split()
    a = [' '.join(a[n * i: n * i + n]) for i in range(0, int(len(a) / n))]    
    
    return '<br> '.join(a)


def predict_label(model, text):
    
    label_to_idx = {0: 'Home/Room Rental Scam',
                    1: 'Impersonation Scam', 
                    2: 'Internet Love Scam', 
                    3: 'Investment Scam', 
                    4: 'Online Purchase Scam'}
    
    text = [pre_process(text)]
    text = tokenizer.texts_to_sequences(text)
    text = pad_sequences(text, maxlen=maxlen)
    pred_probabilities = model.predict(text)
    Y_pred = [np.argmax(x) for x in pred_probabilities][0]
    
    for k, v in label_to_idx.items():
        if k == Y_pred:
            predicted = v
            break
        
    return predicted

def unabbreviate(text):

    """This function takes a text string as input, finds acronyms as defined in a specified Python dictionary,
    and replaces those acronyms with their unabbreviated forms."""
    
    # Define a Python dictionary to match acronym to their non-abbreviated forms
    acronym_dict = {'ICA': 'immigration and checkpoints authority', 
                    'ID': 'identity',
                    'DBS': 'dbs bank', 
                    'FB': 'facebook',
                    'SG': 'singapore',
                    'UK': 'united kingdom',
                    'NRIC': 'identity number',
                    'IC': 'identity number',
                    'I/C': 'identity number',
                    'HQ': 'headquarters',
                    'MOM': 'ministry of manpower',
                    'POSB': 'posb bank',
                    'MOH': 'ministry of health',
                    'OCBC': 'ocbc bank',
                    'CMB': 'cmb bank',
                    'SPF': 'singapore police force',
                    'IRAS': 'inland revenue authority of singapore',
                    'UOB': 'uob bank',
                    'IG': 'instagram',
                    'HP': 'handphone',
                    'HK': 'hong kong',
                    'KL': 'kuala lumpur',
                    'PM': 'private message',
                    'MRT': 'mass rapid transit train',
                    'DOB': 'date of birth',
                    'ATM': 'automated teller machine',
                    'MAS': 'monetary authority of singapore',
                    'PRC': 'people republic of china',
                    'USS': 'universal studios singapore',
                    'MIA': 'missing in action',
                    'GST': 'goods and services tax',
                    'CIMB': 'cimb bank',
                    'HSBC': 'hsbc bank',
                    'MBS': 'marina_bay_sands',
                    'LTD': 'limited',
                    'ASAP': 'as soon as possible',
                    'IBAN': 'international bank account number',
                    'HR': 'human resource',
                    'AMK': 'ang mo kio',
                    'CID': 'criminal investigation department',
                    'PTE': 'private',
                    'OTP': 'one time password',
                    'WA': 'whatsapp',
                    'PC': 'personal computer',
                    'ACRA': 'accounting and corporate regulatory authority',
                    'CPF': 'central provident fund',
                    'ISD': 'internal security department', 
                    'WP': 'work permit',
                    'OKC': 'okcupid', 
                    'HDB': 'housing development board', 
                    'NPC': 'neighbourhood police centre',
                    'MOP': 'member of public',
                    'MOPS': 'members of public', 
                    'IMO': 'in my opinion',
                    'ISP': 'internet service provider', 
                    'IMDA': 'infocomm media development authority', 
                    'CB': 'circuit breaker',
                    'MINLAW': 'ministry of law',
                    'LMAO': 'laugh my ass off',
                    'AKA': 'also known as',
                    'BF': 'boyfriend', 
                    'W/O': 'without',
                    'MOF': 'ministry of finance'}
    
    # Tokenize the text
    x = nltk.word_tokenize(text)

    # Replace acronyms (both upper-case and lower-case) with their unabbreviated forms
    for index, token in enumerate(x):
        for k, v in acronym_dict.items():
            if token == k or token == k.lower():
                x[index] = v
                break

    return ' '.join(x).replace(" .", ".").replace(" ,", ",")

def remove_url(text): 

    """This function takes a text string as input and replaces URL links with a <url> token."""

    regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))" 
    text = re.sub(regex, "url_link", text)   
    
    return text

def decontract(phrase):
    
    """This function takes a phrase, finds contracted words and expands them.
    Source: https://stackoverflow.com/questions/43018030/replace-apostrophe-short-words-in-python"""
    
    # Specific
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)
    phrase = re.sub(r"let\'s", "let us", phrase)
    
    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    
    # Specific
    phrase = re.sub(r"won\’t", "will not", phrase)
    phrase = re.sub(r"can\’t", "can not", phrase)
    phrase = re.sub(r"let\’s", "let us", phrase)

    # general
    phrase = re.sub(r"n\’t", " not", phrase)
    phrase = re.sub(r"\’re", " are", phrase)
    phrase = re.sub(r"\’s", " is", phrase)
    phrase = re.sub(r"\’d", " would", phrase)
    phrase = re.sub(r"\’ll", " will", phrase)
    phrase = re.sub(r"\’t", " not", phrase)
    phrase = re.sub(r"\’ve", " have", phrase)
    phrase = re.sub(r"\’m", " am", phrase)
    
    return phrase


def remove_punct(text):

    """This function takes a text string as input and returns the same text string without specified punctuation marks."""
    
    # Specify punctuation marks to remove from text string
    punctuation = "``-±!@#$%^&*()+?:;”“’<>" 

    # Loop through the text to remove specified punctuations
    for c in text:
        if c in punctuation:
            text = text.replace(c, "").replace('/', ' ').replace('`', "").replace('"', '')

    return text

def correct_misspelled_words(text):
    
    # Define a Python dictionary to match misspelled words to their correctly-spelled forms
    spellcheck_dict = {'acct': 'account', 
                       'acc': 'account',
                       'a/c': 'account',
                       'blk': 'block',
                       'alot': 'a lot', 
                       'abit': 'a bit',
                       'watsapp': 'whatsapp',
                       'whatapps': 'whatsapp',
                       'whatapp': 'whatsapp',
                       'wadsapp': 'whatsapp',
                       'watapps': 'whatsapp',
                       'whatsapps': 'whatsapp',
                       'whats app': 'whatsapp',
                       'whatsaap': 'whatsapp',
                       'whatsap': 'whatsapp',
                       'whattsapp': 'whatsapp',
                       'whattapp': 'whatsapp',
                       'whatsap': 'whatsapp',
                       'whataspp': 'whatsapp',
                       'whatapps': 'whatsapp',
                       'whastapp': 'whatsapp',
                       'whatsapphe': 'whatsapp',
                       'whattapp': 'whatsapp',
                       'abt': 'about',
                       'recieved': 'received',
                       'recieve': 'receive',
                       'hv': 'have',
                       'amt': 'amount',
                       'mths': 'months',
                       'gf': 'girlfriend',
                       'msia': 'malaysia',
                       'tranfer': 'transfer',
                       'trans': 'transfer',
                       'trf': 'transfer',
                       'becareful': 'be careful',
                       'frm': 'from',
                       'msgs': 'messages',
                       'msg': 'message',
                       'plz': 'please',
                       'pls': 'please',
                       'harrass': 'harass',
                       'sintel': 'singtel',
                       'ard': 'around',
                       'wk': 'week', 
                       'fyi': 'for your information',
                       'govt': 'government',
                       'gov': 'government',
                       'thru': 'through',
                       'assent': 'accent', 
                       'dun': 'do not',
                       'nv': 'never', 
                       'sing-tel': 'singtel', 
                       'sintel': 'singtel',
                       'insta': 'instagram', 
                       'sg': 'singapore', 
                       'payapl': 'paypal', 
                       'carousel': 'carousell',
                       'tix': 'tickets', 
                       'mandrain': 'mandarin', 
                       'admin': 'administrative',
                       'bz': 'busy',
                       'daugter': 'daughter',
                       'cos': 'because',
                       'bcos': 'because',
                       'I-banking': 'internet banking',
                       'intl': 'international',
                       'shoppe': 'shopee',
                       'tis': 'this',
                       'docs': 'documents',
                       'doc': 'document',
                       'ytd': 'yesterday', 
                       'tmr': 'tomorrow', 
                       'mon': 'monday',
                       'tue': 'tuesday', 
                       'tues': 'tuesday', 
                       'wed': 'wednesday',
                       'thu': 'thursday',
                       'thur': 'thursday', 
                       'thurs': 'thursday',
                       'fri': 'friday',
                       'wikipeida': 'wikipedia',
                       'juz': 'just',
                       'impt': 'important',
                       'transger': 'transfer',
                       'suspicios': 'suspicious',
                       'suspicius': 'suspicious',
                       'suspicous': 'suspicious',
                       'suspecious': 'suspicious',
                       'suspision': 'suspicion',
                       'nvr': 'never', 
                       'instagam': 'instagram', 
                       'instagramm': 'instagram',
                       "s'pore": "singapore", 
                       'polive': 'police',
                       'linkein': 'linkedin',
                       'messanger': 'messenger', 
                       'scammmer': 'scammer',
                       'laywer': 'lawyer',
                       'dunno': 'do not know',
                       'tidner': 'tinder',
                       'rcvd': 'received',
                       'infomed': 'informed',
                       'informaing': 'informing', 
                       'knowldge': 'knowledge'}

    # Tokenize the text
    x = nltk.word_tokenize(text)    
    
    for index, token in enumerate(x):
        for k, v in spellcheck_dict.items():
            if token == k:
                x[index] = v
                break
        
    return ' '.join(x).replace(' .', '.').replace(' ,', ',').replace('< ', '<').replace(' >', '>')


def remove_stopwords(text_string):

    """This function takes a text string as input, tokenises it and returns a list of tokens without stopwords."""
    
    word_list = [word for word in nltk.word_tokenize(text_string) if not word in set(stopwords.words('english'))]
    text = ' '.join(word_list).replace(' .', '').replace(' ,', '').replace('< ', '<').replace(' >', '>')

    return text

def lemmatise(text_string):

    """This function takes a tokenised text string as input and returns another tokenised text string after lemmatisation."""

    list_of_tokens = [token.lemma_ for token in nlp(text_string)]
    text = ' '.join(list_of_tokens).replace('< ', '<').replace(' >', '>')
    
    return text

def pre_process(text):

    """This function takes a text string as input and pre-processes them as follow: 
    1. Ignore ASCII encodings if any, and decode them as UTF-8
    2. Replaces any URL link with 'url_link' 
    3. Add space after comma, full-stop, question and exclamation mark
    4. Remove digits
    5. Expands out contracted words
    6. Convert all words into lower cases and remove white spaces
    7. Remove punctuations except full-stops and commas
    8. Replace acronyms with unabbreviated forms 
    9. Replace misspelled words with correctly-spelled forms"""
    
    # 1. Ignore ASCII encodings if any, and decode them as UTF-8
#     text = text.encode('ascii', 'ignore').decode('utf-8')
    text = re.sub(r'[^\x00-\x7f]',r' ', text)
    
    # 2. Replace any URL link with 'url_link' 
    text = remove_url(text.replace('\n', ' '))

    # 3. Add a space after comma, full-stop, question mark and exclamation mark
    text = re.sub(r'(?<=[.,?!])(?=[^\s])', r' ', text)
    
    # 4. Remove digits
    text = re.sub(r'\d+', '', text)
    
    # 5. Expand contractions
    text = decontract(text)

    # 6. Convert to lower cases and remove white spaces
    text = text.lower().strip().replace('’s', '')

    # 7. Remove punctuations except full-stops and commas.
    text = remove_punct(text)

    # 8. Replace acronyms with their unabbreviated forms
    text = unabbreviate(text)

    # 9. Replace misspelled words with their correctly-spelled forms
    text = correct_misspelled_words(text)

    return text

def predict_label(model, text, tokenizer, label_to_idx, maxlen=66):

    text = [pre_process(text)][0]
    text = remove_stopwords(text)
    text = [lemmatise(text)]
    text = tokenizer.texts_to_sequences(text)
    text = pad_sequences(text, maxlen=maxlen)
    pred_probabilities = model.predict(text)
    
    Y_pred = [np.argmax(x) for x in pred_probabilities][0]

    for k, v in label_to_idx.items():
        if k == Y_pred:
            predicted = v
            break
    
    return pred_probabilities, predicted