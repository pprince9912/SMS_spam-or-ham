import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk import word_tokenize
import string
import pandas as pd

def preprocess_sms(df):
    df = df.sample(frac = 1, ignore_index = True, random_state = 42)
    X = df.text
    Y = df.spam.to_numpy()

    return X, Y

def preprocess_text(X):
    stop = set(stopwords.words('english') + list(string.punctuation))

    if isinstance(X, str):
        X = np.array([X])
        
    X_preprocessed = []

    for i, sms in enumerate(X):
        sms = np.array([i.lower() for i in word_tokenize(sms) if i.lower() not in stop]).astype(X.dtype)
        X_preprocessed.append(sms)
        
    if len(X) == 1:
        return X_preprocessed[0]
    
    return X_preprocessed

def get_word_frequency(X,Y):
    word_dict = {}
    num_sms = len(X)

    for i in range(num_sms):
        sms = X[i] 
        cls = Y[i] 
        sms = set(sms) 

        for word in sms:
            if word not in word_dict.keys():
                word_dict[word] = {"spam": 1, "ham": 1}

            if cls == 0:    
                word_dict[word]["ham"] += 1
            if cls == 1:
                word_dict[word]["spam"] += 1

    return word_dict

def prob_word_given_class(word, cls, word_frequency, class_frequency):
    amount_word_and_class = word_frequency[word][cls]
    p_word_given_class = amount_word_and_class/class_frequency[cls]

    return p_word_given_class

def prob_sms_given_class(treated_sms, cls, word_frequency, class_frequency):
    prob = 1

    for word in treated_sms:
        if word in word_frequency.keys(): 
            prob *= prob_word_given_class(word, cls, word_frequency, class_frequency)

    return prob

def log_prob_sms_given_class(treated_sms, cls, word_frequency, class_frequency):
    prob = 0

    for word in treated_sms: 
        if word in word_frequency.keys(): 
            prob += np.log(prob_word_given_class(word, cls,word_frequency, class_frequency))

    return prob

def log_naive_bayes(treated_sms, word_frequency, class_frequency, return_likelihood = False):    
    log_prob_sms_given_spam = log_prob_sms_given_class(treated_sms, cls = 'spam',word_frequency = word_frequency, class_frequency = class_frequency) 
    log_prob_sms_given_ham = log_prob_sms_given_class(treated_sms, cls = 'ham',word_frequency = word_frequency, class_frequency = class_frequency) 

    p_spam = class_frequency['spam']/(class_frequency['ham'] + class_frequency['spam']) 
    p_ham = class_frequency['ham']/(class_frequency['ham'] + class_frequency['spam']) 

    log_spam_likelihood = np.log(p_spam) + log_prob_sms_given_spam 
    log_ham_likelihood = np.log(p_ham) + log_prob_sms_given_ham 

    if return_likelihood == True:
        return (log_spam_likelihood, log_ham_likelihood)

    if log_spam_likelihood >= log_ham_likelihood:
        return 1
    else:
        return 0


dataframe_sms = pd.read_csv('dataset\sms.csv')
dataframe_sms['spam'] = dataframe_sms['spam'].map({'spam': 1, 'ham': 0})

X, Y = preprocess_sms(dataframe_sms)
X_treated = preprocess_text(X)
TRAIN_SIZE = int(0.80*len(X_treated)) 

X_train = X_treated[:TRAIN_SIZE]
Y_train = Y[:TRAIN_SIZE]
    
word_frequency = get_word_frequency(X_train,Y_train)
class_frequency = {'ham': sum(Y_train == 0), 'spam': sum(Y_train == 1)}
proportion_spam = class_frequency['spam']/(class_frequency['ham'] + class_frequency['spam'])