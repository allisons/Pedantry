from pandas import DataFrame, Series
import pandas as pd
import numpy as np
import time
# import matplotlib.pyplot as plt
from multiprocessing import Pool
from sys import argv
import os
import logging
import time
import re
logging.basicConfig()
logger = logging.getLogger()
handler = logging.StreamHandler()
formatter = logging.Formatter(
        '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
handler.setFormatter(formatter)
logger.setLevel(logging.DEBUG)



def define_FW(probs, cols):
    """
    Returns a panda.DataFrame with additional columns
    with boolean values for whether that word is a "function word"
    Function word is defined as a word that appears in the top 1 percent
    most frequent words fo a given corpus.
     
    """
    for col in cols:
        probs[col+"FW"] = probs[col] > np.percentile(probs[col], 99)
    return probs
    
def fetch_probs(probs, erpa, tokens=False):
    """
    Returns a panda.DataFrame where the rows are each word from the speech sample and that word's
    unigram value based on the probability matrix provided.  Default will only retreive each word
    from a given participant once.
    """
    placeholder = [] #collecting rows in a list and returning in a dataframe                          gives more reliable rows/col alignmnet
    for i, row in erpa.iterrows():
        if tokens:
            tokens = row['text'].split()
        else:
            tokens = list(set(row['text'].split()))
        for tok in tokens:
            stats = probs[probs.word == tok]
            for i, s in stats.iterrows():
                whole_row = Series({"id" : row['ID'], "dx" : row['DX'], 
                "word" : tok, "child" : s["child"], "callhome" : s['callhome'],
                "TAL" :s['TAL'], "WSJ" : s["WSJ"], "childFW":s['childFW'], "callhomeFW":s['callhomeFW'], 
                "TALFW" : s['TALFW'], "WSJFW" : s["WSJFW"]})
                placeholder.append(whole_row)
    return DataFrame(placeholder)   
    
def FW_remove(data, col_1, col_2):
    """
    Returns a panda.DataFrame with words that are function words in both corpora provided
    """
    fw_groups = data.groupby([col_1, col_2])
    return pd.concat([fw_groups.get_group((False,False)),fw_groups.get_group((False,True)),
                      fw_groups.get_group((True,False))])


def euclideanfit(x,y):
    """
    Returns a 2-tuple equal to the slope and intercept of a euclidean-fit line
    """
    
    cov_mat = np.cov(x,y)
    _, eigvec = np.ligalg.ein(cov_mat)
    a = eigvec[2,1]/eigec[1,2]
    b = np.mean(y) - a*np.mean(x)
    return a, b
    
def mapper(x):
    return one_bootstrap(x)

def subject_summary(data, other):
    """
    Returns a pandas.DataFrame summarizing the mean pedantry values
    """
    placeholder = []
    subjects = data.groupby("id")
    subjects_noFW = pd.read_csv("ERPA-stats-types-noFW-"+other+"FW.csv").groupby('id')
    for id in subjects.groups.keys():
        values = {}
        for col in ['child', other]:
            values[col+"_all"] = np.mean(subjects.get_group(id)[col])
            values[col+"_noFW"] = np.mean(subjects_noFW.get_group(id)[col])
            values['id'] = id
        dx = np.random.choice(subjects.get_group(id)['dx'])
        values['dx'] = dx
        values = Series(values)
        placeholder.append(values)
    return DataFrame(placeholder)    

def one_bootstrap(data, function, *args, **kwargs):
    """
    Returns a boostrapped metric (provided as a python method reference in function) 
    """
    subjects = data.groupby(data.columns[0])
    
    placeholder = []
    for name, group in subjects:
        logger.debug("Beginning bootstrap for {0}".format(name))
        handler.flush()    
        values = group.ix[:,1]
        sample_set = draw_sample_report_metric(values, function, *args, **kwargs)
        line = Series([name,sample_set])
        placeholder.append(line)
    return DataFrame(placeholder)
        
    

def run_bootstrap(N, n, data, function, *args, **kwargs):
    """
    Returns a DataFrame with the results of n bootstrap sampling using the provided metric function
    """
    pool = Pool(processes=N)
    X = [(data, function) + tuple([a for a in args if args]) for _ in xrange(n)]
    return DataFrame(pool.map(mapper, X))
    
    

def create_erpaprobs(model_descrip, args):
    """
    Runs data-creation side of work - args could include the file names for the data and the language models
    """
    probs = pd.read_csv(args[0])
    erpa = pd.read_csv(args[1])
    corpora = ['callhome', 'TAL', 'WSJ']
    #Tokens or types?
    tokens = bool(int(args[2]))
    
    if not os.path.isdir("outputfiles"):
        os.mkdir("outputfiles")

    #Mark out which words are in the 99th percentile for frequency
    probs = define_FW(probs,corpora)
    probs.to_csv("outputfiles/probs_defined_FW.csv", index=False)
    erpaprobs = fetch_probs(probs, erpa, tokens=tokens)
    erpaprobsfn = "outputfiles/ERPA-stats_"+model_descrip+".csv"
    erpaprobs.to_csv(erpaprobsfn, index=False)
    if not os.path.exists(erpaprobsfn):
        logger.debug("Failed to save erpa matrix")
    else:
        logger.debug("Saved Erpa matrix")
    handler.flush()
    

def draw_sample_report_metric(data, function, Nsize=100, *args, **kwargs):
    """
    Samples data with a random subset and using a provided function, 
    reports whatever that function (presumably a test statistic) returns.  
    """
    
	arr = np.array(data)
	sample = np.random.choice(arr, size=Nsize)
	return function(sample, *args, **kwargs)



if __name__ == "__main__":
    import EMGMM
    """
    Currently this code does not run - providing the EMGMM bootstrap iterations with the arbitrary
    parameters it needs has proved challening.  
    """
    cols = ['callhome', 'TAL', 'WSJ']
    if len(argv) > 3:
        model_descrip = argv[2]
        model_descrip = re.sub(" ", "_", model_descrip)
        logger.debug("Beginning "+model_descrip+" iteration of model")
        handler.flush()
        if argv[1] == "data":
            create_erpaprobs(model_descrip, argv[3:])
        if argv[1] == "bootstrap":
            erpaprobs = pd.read_csv(argv[3])
            results = run_bootstrap(50, 1000, erpaprobs[['id', 'pedantry_score']], EMGMM.GaussianMixture.fit, -4, 4, .1, 1)
            results.to_csv("EMGMM_raw_results.csv")

            
    elif argv[1] == "test_with_data":
        #Testing module to try to get at least a single iteration working - not     currently passing this test.
        erpaprobs = pd.read_csv(argv[2])
        results = run_bootstrap(5, 2, erpaprobs[['id', 'pedantry_score']], EMGMM.GaussianMixture.fit, -4, 4, .1, 1, num_restarts=2)
        results.to_csv("EMGMM_raw_results.csv")
    else:
        print "Terminating after doing basically nothing, so oops"
    
    
