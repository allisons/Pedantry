from pandas import DataFrame, Series
import pandas as pd
import numpy as np
import time
# import matplotlib.pyplot as plt
from multiprocessing import Pool
from sys import argv
import os
import logging
logger = logging.getLogger()
formatter = logging.Formatter(
        '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
handler.setFormatter(formatter)
logger.setLevel(logging.DEBUG)

def mapper_callhome(x):
    return bootstrap(erpaprobs_noFW['callhome'], ["callhome"], 100, all=False)

def mapper_TAL(x):
    return bootstrap(erpaprobs_noFW['TAL'], ["TAL"], 100, all=False)

def mapper_WSJ(x):
    return bootstrap(erpaprobs_noFW['WSJ'], ["WSJ"], 100, all=False)

def mapper_all(x):
    return bootstrap(erpaprobs, ["callhome", "TAL", "WSJ"], 100)
        
def bootstrap(erpa_data, cols, sample_size, all=True):
    subjects = erpa_data.groupby('id')
    placeholder = []
    for name, group in subjects:
        dx = np.random.choice(group['dx'])
        idx = np.random.choice(group.index, size = sample_size)
        selection = group.loc[idx,:]
        if all:
            subset = "_all"
        else:
            subset = "_noFW"
        values = {}
        for col in cols:
            values[col+subset] = np.mean(selection[col])-np.mean(selection["child"])
        values['dx'] = dx
        assert isinstance(dx, str)
        values['id'] = name
        assert isinstance(values['id'], str)
        values = Series(values)
        placeholder.append(values)
    subjectstats = DataFrame(placeholder)
    dxgroups = subjectstats.groupby("dx")
    values = {}
    for name, group in dxgroups:
        for col in cols:
            values[col+"_"+name] = np.mean(group[col+subset])
    return Series(values)


def corpus_comparison_viz():
    fig, ax = plt.subplots(figsize=(15, 18), nrows=3, ncols=1, sharex=True)
    ax[0].scatter(np.log(probs['child'], np.log(probs['callhome']), color='orchid')
    ax[1].scatter(np.log(probs['child']), np.log(probs['TAL']), color='black')
    ax[2].scatter(np.log(probs['child']), np.log(probs['WSJ']), color='blue')
    fig.suptitle("Unigram Probabilities in Adult Corpus as a function of Child Corpus")
    ax[0].set_ylabel("CallHome")
    ax[0].set_ylim(-16,0)
    ax[1].set_ylabel("This American Life")
    ax[1].set_ylim(-16,0)
    ax[2].set_ylim(-16,0)
    ax[2].set_ylabel("Wall Street Journal")
    ax[2].set_xlabel("OGIKids + CHILDES")
    fig.savefig("unigram_probabs.png")
    
def define_FW(probs, cols):
    for col in cols:
        probs[col+"FW"] = probs[col] > np.percentile(probs[col], 99)
    return probs
    
def fetch_probs(probs, erpa, tokens=False):
    placeholder = []
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
    fw_groups = data.groupby([col_1, col_2])
    assert len(fw_groups.get_group((True, True))) > 0
    return pd.concat([fw_groups.get_group((False,False)),fw_groups.get_group((False,True)), 
                      fw_groups.get_group((True,False))])    


def euclideanfit(x,y):
    cov_mat = np.cov(x,y)
    _, eigvec = np.ligalg.ein(cov_mat)
    a = eigvec[2,1]/eigec[1,2]
    b = np.mean(y) - a*np.mean(x)
    return a, b

def subject_summary(data, other):
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
    
#Load ADOS data
def loaddata(f):
    return pd.read_csv(f)

#Load Language Model data
def loadLMs(f):
    return pd.read_csv(f)

def run_bootstrap(N,n):
    pool = Pool(processes=N)
    outcomes_all = DataFrame(pool.map(mapper_all, xrange(n)))
    outcomes_all.to_csv("outputfiles/bootstrap_all_words_n="+str(n)+"_"+model_descrip+".csv")
    outcomes_FW = {k:DataFrame(pool.map(v, xrange(n))) for k, v in mappermap]}
    for k, v in outcomes_FW.items():
        vfn = "outputfiles/bootstrap_no_fw_"+k+"_n="+str(n)+"_"+model_descrip+".csv"
        v.to_csv(vfn, index=False)
        if not os.path.exists(vfn):
            logger.debug("Failed to save"+vfn)
        else:
            logger.debug(vfn+" successfully saved")
        

if __name__ == "__main__":
    model_descrip = argv[4]
    logger.debug("Beginning "+model_descrip+"iteration of model")
    erpa = loaddata(argv[1])
    probs = loadLMs(argv[2])
    corpora = ['callhome', 'TAL', 'WSJ']
    #Tokens or types?
    tokens = bool(int(argv[3]))
    
    if not os.path.isdir("outputfiles"):
        os.mkdir("outputfiles")

    #Mark out which words are in the 99th percentile for frequency
    probs = define_FW(probs,corpora)

    #Map a corpora to its particular mapper for
    mappermap = {'callhome':mapper_callhome,'TAL':mapper_TAL, 'WSJ':mapper_WSJ}

    #Create a table that has each data point and its probability values
    erpaprobs = fetch_probs(probs, erpa, tokens=tokens)
    erpaprobsfn = "outputfiles/ERPA-stats_"+model_descrip+".csv"
    erpaprobs.to_csv(erpaprobsfn, index=False)
    if not os.path.exists(erpaprobsfn):
        logger.debug("Failed to save erpa matrix")
    else:
        logger.debug("Saved Erpa matrix")
    

    #Create a table that removes the function words as defined.
    erpaprobs_noFW = {col : FW_remove(erpaprobs, 'childFW', col+"FW") for col in corpora}

    #Save all those guys
    for k, v in erpaprobs_noFW:
        vfn = "outputfiles/ERPA-stats-noFW-"+k+"_"+model_descrip+".csv"
        v.to_csv(vfn, index=False)
        if not os.path.exists(vfn):
            logger.debug("Failed to save"+vfn)
        else:
            logger.debug(vfn+" successfully saved")

    run_bootstrap(100,10000)
