# coding=latin-1
# Non nobis, Domine, non nobis, sed Nomini Tuo da gloriam
import sys, re, time, json
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, precision_recall_fscore_support, log_loss, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy
import warnings
###############################################################################


def stringtime(n):
    """
    :param n: integer, representing a number seconds
    :return: string representing time, in format hh:mm:ss
    """
    h = str(int(n / 3600))
    m = str(int((n % 3600) / 60))
    s = str(int((n % 3600) % 60))
    if len(h) == 1: h = '0' + h
    if len(m) == 1: m = '0' + m
    if len(s) == 1: s = '0' + s
    return h + ':' + m + ':' + s


def start(sep=True):
    """
    This function prints and returns the present moment
    :param sep: bool. If True, a separation line is printed on the shell
    :return: the present moment
    """
    stt = time.time()
    now = time.strftime("%Y/%m/%d %H:%M:%S")
    if sep: print('#'*80)
    print('start:', now)
    return stt


def end(stt, sep=True):
    """
    The function takes an hour in the past and prints the present moment and the time elapsed from the input moment
    :param stt: a time.time() variable
    :param sep: bool. If True, a separation line is printed on the shell
    :return: the number of seconds between the input hour and the present moment
    """
    now = time.time()
    dur = now - stt
    str_dur = stringtime(now - stt)
    str_now = time.strftime("%Y/%m/%d %H:%M:%S")
    if sep:
        print('#'*80 + "\nend:", str_now, " - time elapsed:", str_dur + "\n" + '#'*80)
    else:
        print("end:", str_now, " - time elapsed:", str_dur)
    return dur

    
class BColor:
    """
    Class that defines colors for fancy results' printing
    """
    reset     = '\033[0m'
    bold      = '\033[1m'
    underline = '\033[4m'
    reversed  = '\033[7m'

    white     = '\033[38;5;0m'
    cyan      = '\033[38;5;14m'
    magenta   = '\033[38;5;13m'
    blue      = '\033[38;5;12m'
    yellow    = '\033[38;5;11m'
    green     = '\033[38;5;10m'
    red       = '\033[38;5;9m'
    grey      = '\033[38;5;8m'
    black     = '\033[38;5;0m'

    cleargrey  = '\033[38;5;7m'
    darkyellow = '\033[38;5;3m'
    darkred    = '\033[38;5;88m'
    darkcyan   = '\033[38;5;6m'
    pink       = '\033[38;5;207m'
    clearpink  = '\033[38;5;218m'
    cyangreen  = '\033[38;5;85m'
    cleargreen = '\033[38;5;192m'
    olivegreen = '\033[38;5;29m'
    
    CEND      = '\33[0m'
    CBOLD     = '\33[1m'
    CITALIC   = '\33[3m'
    CURL      = '\33[4m'
    CBLINK    = '\33[5m'
    CBLINK2   = '\33[6m'
    CSELECTED = '\33[7m'

    CBLACK    = '\33[30m'
    CRED      = '\33[31m'
    CGREEN    = '\33[32m'
    CYELLOW   = '\33[33m'
    CBLUE     = '\33[34m'
    CVIOLET   = '\33[35m'
    CBEIGE    = '\33[36m'
    CWHITE    = '\33[37m'

    CBLACKBG  = '\33[40m'
    CREDBG    = '\33[41m'
    CGREENBG  = '\33[42m'
    CYELLOWBG = '\33[43m'
    CBLUEBG   = '\33[44m'
    CVIOLETBG = '\33[45m'
    CBEIGEBG  = '\33[46m'
    CWHITEBG  = '\33[47m'

    CGREY     = '\33[90m'
    CRED2     = '\33[91m'
    CGREEN2   = '\33[92m'
    CYELLOW2  = '\33[93m'
    CBLUE2    = '\33[94m'
    CVIOLET2  = '\33[95m'
    CBEIGE2   = '\33[96m'
    CWHITE2   = '\33[97m'

    CGREYBG    = '\33[100m'
    CREDBG2    = '\33[101m'
    CGREENBG2  = '\33[102m'
    CYELLOWBG2 = '\33[103m'
    CBLUEBG2   = '\33[104m'
    CVIOLETBG2 = '\33[105m'
    CBEIGEBG2  = '\33[106m'
    CWHITEBG2  = '\33[107m'

    # red nuances:
    i2red = {0:  ([255/255, 204/255, 204/255]), # almost white
             1:  ([255/255, 153/255, 153/255]),
             2:  ([255/255, 102/255, 102/255]),
             3:  ([255/255, 51/255,  51/255]),
             4:  ([255/255, 0/255,   0/255]), # red
             5:  ([204/255, 0/255,   0/255]),
             6:  ([153/255, 0/255,   0/255]),
             7:  ([102/255, 0/255,   0/255]),
             8:  ([51/255,  0/255,   0/255])} # almost black


###############################################################################


class Bootstrap:
    """
    Class that computes performance metrics for hard and soft labels,
    and the bootstrap sampling significance test.
    I takes experiments' results and it prints performance and significance
    levels, saving both inputs and outputs to different files
    """
    def __init__(self, save_results=True, save_outcomes=True, dir_out=''):
        """
        :param save_results: type: bool; default: True. If True, the results
        (performance and significance levels) are saved as tsv file(s)
        :param save_outcomes: type: bool; default: True. If True, the inputs
        (experiments' outcomes) are saved as json file
        :param dir_out: type: str; default: ''. Dir where to save the files above
        :attribute data:
                defaultdict(lambda: {'exp_idxs': list(), 'preds': list(), 'targs': list(), 'idxs': list(), 'epochs': list(),
                                     'h1': defaultdict(lambda: {'exp_idxs': list(), 'preds': list(), 'targs': list(), 'idxs': list(), 'epochs': list()})}
                Used to store and save to json the class inputs.
        """
        self.dirout = dir_out + '/' if (dir_out != '') and (not re.search('/$', dir_out)) else dir_out
        self.savetsv = save_results
        self.savejson = save_outcomes
        self.data = defaultdict(lambda: {'exp_idxs': list(),
                                         'preds':    list(),
                                         'targs':    list(),
                                         'idxs':     list(),
                                         'epochs':   list(),
                                         'h1': defaultdict(lambda: {'exp_idxs': list(),
                                                                    'preds':    list(),
                                                                    'targs':    list(),
                                                                    'idxs':     list(),
                                                                    'epochs':   list()})})

    def feed(self, h0, h1=None, exp_idx=None, preds=None, targs=None, idxs=None, epochs=()):
        """
        :param h0: type: str. It identifies the control condition.
        :param h1: type: str. It identifies the experimental condition.
        :param exp_idx: type: str. It identifies the experiments' index
        :param preds:  type: list, numpy.ndarray or str. If str, interpreted as a path to a .npy, .csv,
        .tsv or .txt file, containing experiments' outcomes (hard or soft labels)
        :param targs: type: list, numpy.ndarray or str. If str, interpreted as a path to a .npy, .csv,
        .tsv or .txt file, containing experiments' targets (hard or soft labels)
        :param idxs:  type: list, numpy.ndarray or str. If str, interpreted as a path to a .npy, .csv,
        .tsv or .txt file, containing experiments' data point indexes
        :param epochs: type: int. Number of experiment's training epochs.
        :return: 1. The function stores the inputs in a dictionary accessible from other functions,
        and that can be saved as json file
        """
        targs = self.input2ndarray(targs)
        preds = self.input2ndarray(preds)
        idxs = np.arange(len(preds)) if idxs is None else self.input2ndarray(idxs)
        assert len(preds) == len(targs) == len(idxs), 'preds, targs or idxs have different length'
        if h1:
            self.data[h0]['h1'][h1]['exp_idxs'].append(exp_idx)
            self.data[h0]['h1'][h1]['preds'].append(preds)
            self.data[h0]['h1'][h1]['targs'].append(targs)
            self.data[h0]['h1'][h1]['idxs'].append(idxs)
            self.data[h0]['h1'][h1]['epochs'].extend(epochs)
        else:
            self.data[h0]['exp_idxs'].append(exp_idx)
            self.data[h0]['preds'].append(preds)
            self.data[h0]['targs'].append(targs)
            self.data[h0]['idxs'].append(idxs)
            self.data[h0]['epochs'].extend(epochs)
        return 1

    def data2json(self, pathname):
        """
        I takes the path where to save the input data and saves them to json.
        :param pathname: path where to save the json file
        :return: 1
        """
        for h0 in self.data:
            self.data[h0]['targs'] = [targ.tolist() for targ in self.data[h0]['targs']]
            self.data[h0]['preds'] = [pred.tolist() for pred in self.data[h0]['preds']]
            self.data[h0]['idxs']  = [idx.tolist()  for idx  in self.data[h0]['idxs']]
            for h1 in self.data[h0]['h1']:
                self.data[h0]['h1'][h1]['targs'] = [targ.tolist() for targ in self.data[h0]['h1'][h1]['targs']]
                self.data[h0]['h1'][h1]['preds'] = [pred.tolist() for pred in self.data[h0]['h1'][h1]['preds']]
                self.data[h0]['h1'][h1]['idxs']  = [idx.tolist()  for idx  in self.data[h0]['h1'][h1]['idxs']]
        with open(pathname, 'w') as out: json.dump(self.data, out)
        return 1

    def loadjson(self, pathname):
        """
        It takes a path to a json file previously saved and loads it in a dictionary
        accessible from other functions
        :param pathname: path to a json file previously saved.
        :return: 1
        """
        with open(pathname) as f_in: jin = json.load(f_in)
        for h0 in jin:
            self.data[h0]['exp_idxs'] = jin[h0]['exp_idxs']
            self.data[h0]['preds']    = np.array(jin[h0]['preds'])
            self.data[h0]['targs']    = np.array(jin[h0]['targs'])
            self.data[h0]['idxs']     = np.array(jin[h0]['idxs'])
            self.data[h0]['epochs']   = jin[h0]['epochs']
            for h1 in jin[h0]['h1']:
                self.data[h0]['h1'][h1]['exp_idxs'] = jin[h0]['h1'][h1]['exp_idxs']
                self.data[h0]['h1'][h1]['preds']    = np.array(jin[h0]['h1'][h1]['preds'])
                self.data[h0]['h1'][h1]['targs']    = np.array(jin[h0]['h1'][h1]['targs'])
                self.data[h0]['h1'][h1]['idxs']     = np.array(jin[h0]['h1'][h1]['idxs'])
                self.data[h0]['h1'][h1]['epochs']   = jin[h0]['h1'][h1]['epochs']
        
        return 1

    @staticmethod
    def input2ndarray(obj):
        """
        It takes an input object and converts it in a 2D numpy.ndarray
        :param obj: list, numpy.ndarray or str. If str, treated as a path to a .npy, .csv, .tsv
        or .txt file.
        :return: 2D numpy.ndarray
        """
        if isinstance(obj, np.ndarray):
            if len(obj.shape) == 1:
                return obj.reshape(-1, 1)
            else:
                return obj
        elif isinstance(obj, list):
            if  isinstance(obj[0], list):
                return np.array(obj)
            else:
                if type(obj[0]) is int:
                    return np.array(obj).reshape(-1, 1)
                else:
                    sys.exit("The input list should contain integers representing the classes' indexes")
        elif isinstance(obj, str):
            if re.search('npy$', obj):
                return np.load(obj)
            elif re.search('txt|csv$', obj):
                return pd.read_csv(obj, header=None).to_numpy()
            elif re.search('tsv$', obj):
                return pd.read_csv(obj, header=None).to_numpy()
            else:
                sys.exit(f"Please provide a '.npy', '.csv', '.tsv' file, or a '.txt' file containing one integer for row")

    @staticmethod
    def crossentropy_mean(p_targ, q_pred):
        """
        Given 2 lists of probability distributions, computes the mean cross-entropy
        :param p_targ: list of target probability ditributions
        :param q_pred: list of predicted probability ditributions
        :return: mean cross-entropy
        """
        epsilon = 1e-12
        p_targ = np.clip(p_targ, epsilon, 1. - epsilon) # se qualche valore deborda, quello che eccede lo riduco tra 0espiccioli e 1-0espiccioli
        q_pred = np.clip(q_pred, epsilon, 1. - epsilon)
        ce = -np.sum(p_targ * np.log(q_pred), axis=1)
        return ce.mean()

    def metrics(self, targs, h0_preds, h1_preds, h0_name='h0', h1_name='h1', targetclass=None, verbose=False):
        """
        :param targs: type: numpy.ndarray. target hard or soft labels
        :param h0_preds: type: numpy.ndarray. predicted hard or soft labels from a control experiment
        :param h1_preds: type: numpy.ndarray. predicted hard or soft labels from a treatment experiment
        :param h0_name: type: str. it identifies the control condition
        :param h1_name: type: str. it identifies the treatment condition
        :param targetclass: type: int, default: None. If given, performance and significance levels
         are computed wrt the class whose index is given
        :param verbose: type: bool; default: False. If true, it prints performance information.
        :return: two pandas DataFrame: the first contain the overall performance,
        the second the target class performance (empty if not requested)
        """
        if targs.shape[1] == 1:
            rounding_value = 2
            h0_acc  = accuracy_score(targs, h0_preds)
            h0_f1   = f1_score(targs, h0_preds, average='macro')
            h0_prec = precision_score(targs, h0_preds, average='macro')
            h0_rec  = recall_score(targs, h0_preds, average='macro')
            h1_acc  = accuracy_score(targs, h1_preds)
            h1_f1   = f1_score(targs, h1_preds, average='macro')
            h1_prec = precision_score(targs, h1_preds, average='macro')
            h1_rec  = recall_score(targs, h1_preds, average='macro')
            # h0_conf_matrix = confusion_matrix(targs, h0_preds)
            # h1_conf_matrix = confusion_matrix(targs, h1_preds)
            # print(h0_conf_matrix, "\n", h1_conf_matrix)
            diff_acc  = h1_acc  - h0_acc
            diff_f1   = h1_f1   - h0_f1
            diff_prec = h1_prec - h0_prec
            diff_rec  = h1_rec  - h0_rec
            df_tot = pd.DataFrame({ 'f1':    [h0_f1,       h1_f1],       'd_f1':    ['',  diff_f1],       's_f1':    ['', ''],
                                    'acc':   [h0_acc,      h1_acc],      'd_acc':   ['',  diff_acc],      's_acc':   ['', ''],
                                    'prec':  [h0_prec,     h1_prec],     'd_prec':  ['',  diff_prec],     's_prec':  ['', ''],
                                    'rec':   [h0_rec,      h1_rec],      'd_rec':   ['',  diff_rec],      's_rec':   ['', ''],
                                    'mean_epochs': [None, None]
                                    }, index=[h0_name, h1_name])
            df_tot = df_tot['mean_epochs f1 d_f1 s_f1 acc d_acc s_acc prec d_prec s_prec rec d_rec s_rec'.split()]
            if verbose:
                h0_countpreds = Counter(h0_preds.flatten())
                h1_countpreds = Counter(h1_preds.flatten())
                counttargs    = Counter(targs.flatten())
                h0_countpreds = [f"class {tup[0]} freq {tup[1]} perc {tup[1] / len(h0_preds) * 100:.2f}%" for tup in sorted({k: h0_countpreds[k] for k in h0_countpreds}.items(), key=lambda item: item[0])]
                h1_countpreds = [f"class {tup[0]} freq {tup[1]} perc {tup[1] / len(h1_preds) * 100:.2f}%" for tup in sorted({k: h1_countpreds[k] for k in h1_countpreds}.items(), key=lambda item: item[0])]
                counttargs    = [f"class {tup[0]} freq {tup[1]} perc {tup[1] / len(targs) * 100:.2f}%" for tup in sorted({k: counttargs[k] for k in counttargs}.items(), key=lambda item: item[0])]
                print(f"h0: {h0_name} - h1: {h1_name}")
                print(f"{'targs count:':<15} {counttargs}")
                print(f"{'h0 preds count:':<15} {h0_countpreds}")
                print(f"{'h1 preds count:':<15} {h1_countpreds}")
                print(f"{'F-measure':.<15} - h0: {h0_f1:<7.4f} - h1: {h1_f1:<7.4f} - diff: {diff_f1:.4f}")
                print(f"{'precision':.<15} - h0: {h0_prec:<7.4f} - h1: {h1_prec:<7.4f} - diff: {diff_prec:.4f}")
                print(f"{'recall':.<15} - h0: {h0_rec:<7.4f} - h1: {h1_rec:<7.4f} - diff: {diff_rec:.4f}")
                print(f"{'accuracy':.<15} - h0: {h0_acc:<7.4f} - h1: {h1_acc:<7.4f} - diff: {diff_acc:.4f}")

            df_tgt = pd.DataFrame(index=[h0_name, h1_name])
            if targetclass is not None:
                assert targetclass in np.unique(targs), 'targetclass must belong to the classes\' set'
                h0_vals = precision_recall_fscore_support(targs, h0_preds)
                h1_vals = precision_recall_fscore_support(targs, h1_preds)
                h0_tgt_prec   = round(h0_vals[0][targetclass] * 100, rounding_value)
                h0_tgt_rec    = round(h0_vals[1][targetclass] * 100, rounding_value)
                h0_tgt_f1     = round(h0_vals[2][targetclass] * 100, rounding_value)
                h1_tgt_prec   = round(h1_vals[0][targetclass] * 100, rounding_value)
                h1_tgt_rec    = round(h1_vals[1][targetclass] * 100, rounding_value)
                h1_tgt_f1     = round(h1_vals[2][targetclass] * 100, rounding_value)
                diff_tgt_f1   = round(h1_tgt_f1   - h0_tgt_f1,   rounding_value)
                diff_tgt_prec = round(h1_tgt_prec - h0_tgt_prec, rounding_value)
                diff_tgt_rec  = round(h1_tgt_rec  - h0_tgt_rec,  rounding_value)
                df_tgt = pd.DataFrame({ 'tf1':   [h0_tgt_f1,   h1_tgt_f1],   'd_tf1':   ['',  diff_tgt_f1],   's_tf1':   ['', ''],
                                        'tprec': [h0_tgt_prec, h1_tgt_prec], 'd_tprec': ['',  diff_tgt_prec], 's_tprec': ['', ''],
                                        'trec':  [h0_tgt_rec,  h1_tgt_rec],  'd_trec':  ['',  diff_tgt_rec],  's_trec':  ['', ''],
                                        'mean_epochs': [None, None]
                                        }, index=[h0_name, h1_name])
                df_tgt = df_tgt['mean_epochs tf1 d_tf1 s_tf1 tprec d_tprec s_tprec trec d_trec s_trec'.split()]
                if verbose:
                    print(f"{'targetclass F-measure':.<25} - h0: {h0_tgt_f1:<7.4f} - h1: {h1_tgt_f1:<7.4f} - diff: {diff_tgt_f1:.4f}")
                    print(f"{'targetclass precision':.<25} - h0: {h0_tgt_prec:<7.4f} - h1: {h1_tgt_prec:<7.4f} - diff: {diff_tgt_prec:.4f}")
                    print(f"{'targetclass recall':.<25} - h0: {h0_tgt_rec:<7.4f} - h1: {h1_tgt_rec:<7.4f} - diff: {diff_tgt_rec:.4f}")
            return df_tot, df_tgt
        else:
            h0_jsd = np.mean(jensenshannon(targs, h0_preds, axis=1))
            h1_jsd = np.mean(jensenshannon(targs, h1_preds, axis=1))

            h0_ce   = self.crossentropy_mean(targs, h0_preds)
            h1_ce   = self.crossentropy_mean(targs, h1_preds)

            norm_ent = entropy(np.array([1 / targs.shape[1]] * targs.shape[1]))
            tgt_ent  = entropy(targs,    axis=1) / norm_ent
            h0_ent   = entropy(h0_preds, axis=1) / norm_ent
            h1_ent   = entropy(h1_preds, axis=1) / norm_ent
            h0_sim   = cosine_similarity(tgt_ent.reshape(1, -1), h0_ent.reshape(1, -1))[0][0]
            h1_sim   = cosine_similarity(tgt_ent.reshape(1, -1), h1_ent.reshape(1, -1))[0][0]

            h0_cor = np.corrcoef(tgt_ent, h0_ent)[0][1]
            h1_cor = np.corrcoef(tgt_ent, h1_ent)[0][1]

            diff_jsd = h1_jsd - h0_jsd
            diff_ce  = h1_ce  - h0_ce
            diff_sim = h1_sim - h0_sim
            diff_cor = h1_cor - h0_cor

            df_tot = pd.DataFrame({ 'jsd': [h0_jsd, h1_jsd], 'd_jsd': ['',  diff_jsd], 's_jsd': ['', ''],
                                    'ce':  [h0_ce,  h1_ce],  'd_ce':  ['',  diff_ce],  's_ce':  ['', ''],
                                    'sim': [h0_sim, h1_sim], 'd_sim': ['',  diff_sim], 's_sim': ['', ''],
                                    'cor': [h0_cor, h1_cor], 'd_cor': ['',  diff_cor], 's_cor': ['', ''],
                                    'mean_epochs': [None, None]
                                    }, index=[h0_name, h1_name])
            df_tot = df_tot['mean_epochs jsd d_jsd s_jsd ce d_ce s_ce sim d_sim s_sim cor d_cor s_cor'.split()]
            if verbose:
                print(f"h0: {h0_name} - h1: {h1_name}")
                print(f"{'targs distribution:':<26} {np.mean(targs, axis=0)}")
                print(f"{'h0_preds distribution:':<26} {np.mean(h0_preds, axis=0)}")
                print(f"{'h1_preds distribution:':<26} {np.mean(h1_preds, axis=0)}")
                print(f"{'Jensen-Shannon divergence:':<26} - h0: {h0_jsd:<7.4f} - h1: {h1_jsd:<7.4f} - diff: {diff_jsd:.4f}")
                print(f"{'cross-entropy:':<26} - h0: {h0_ce:<7.4f} - h1: {h1_ce:<7.4f} - diff: {diff_ce:.4f}")
                print(f"{'entropy similarity:':<26} - h0: {h0_sim:<7.4f} - h1: {h1_sim:<7.4f} - diff: {diff_sim:.4f}")
                print(f"{'entropy correlation:':<26} - h0: {h0_cor:<7.4f} - h1: {h1_cor:<7.4f} - diff: {diff_cor:.4f}")

            df_tgt = pd.DataFrame(index=[h0_name, h1_name])
            return df_tot, df_tgt

    def test(self, targs, h0_preds, h1_preds, h0_name='h0', h1_name='h1', n_loops=1000, sample_size=.1, targetclass=None, verbose=False):
        """
        :param targs: type: list, numpy.ndarray or str. If str, interpreted as a path to a .npy, .csv,
        .tsv or .txt file, containing experiments' targets (hard or soft labels)
        :param h0_preds: type: list, numpy.ndarray or str. If str, interpreted as a path to a .npy, .csv,
        .tsv or .txt file, containing experiments' predictions (hard or soft labels), from the control
        condition (h0)
        :param h1_preds: type: list, numpy.ndarray or str. If str, interpreted as a path to a .npy, .csv,
        .tsv or .txt file, containing experiments' predictions (hard or soft labels), from the
        experimental condition (h1)
        :param h0_name: type: str. it identifies the control condition
        :param h1_name: type: str. it identifies the treatment condition
        :param n_loops: type: int; default: 1000. number of iterations for bootstrap sampling.
        :param sample_size: type: float; default: .1. Percent sample size wrt the test set, f
        or bootstrap sampling.
        :param targetclass: type: int; default: None. Index of a target class. If given, performance and
        significance levels are computed for that class.
        :param verbose: type: bool; default: False. If true, it prints performance information.
        :return: two pandas DataFrame: the first contain the overall performance,
        the second the target class performance (empty if not requested)
        """
        assert .05 <= sample_size <= .5, 'sample_size must be between .05 and .5'
        targs    = self.input2ndarray(targs)
        h0_preds = self.input2ndarray(h0_preds)
        h1_preds = self.input2ndarray(h1_preds)
        sample_size = int(targs.shape[0] * sample_size)
        print(f"{'data shape:':<12} {targs.shape}\n{'sample size:':<12} {sample_size}")
        df_tot, df_tgt = self.metrics(targs, h0_preds, h1_preds, h0_name=h0_name, h1_name=h1_name, targetclass=targetclass, verbose=verbose)
        if targs.shape[1] == 1:
            diff_acc  = df_tot.d_acc[-1]
            diff_f1   = df_tot.d_f1[-1]
            diff_prec = df_tot.d_prec[-1]
            diff_rec  = df_tot.d_rec[-1]
            twice_diff_acc  = 0
            twice_diff_f1   = 0
            twice_diff_prec = 0
            twice_diff_rec  = 0
            diff_tgt_f1, diff_tgt_prec, diff_tgt_rec, twice_diff_tgt_f1, twice_diff_tgt_prec, twice_diff_tgt_rec = None, None, None, None, None, None
            if targetclass is not None:
                diff_tgt_f1   = df_tgt.d_tf1[-1]
                diff_tgt_prec = df_tgt.d_tprec[-1]
                diff_tgt_rec  = df_tgt.d_trec[-1]
                twice_diff_tgt_f1   = 0
                twice_diff_tgt_prec = 0
                twice_diff_tgt_rec  = 0
            for _ in tqdm(range(n_loops), desc='bootstrap', ncols=80):
                i_sample = np.random.choice(range(targs.shape[0]), size=sample_size, replace=True) # Berg-Kirkpatrick, p. 996: "with replacement"
                sample_targs    = targs[i_sample]
                sample_h0_preds = h0_preds[i_sample]
                sample_h1_preds = h1_preds[i_sample]
                df_sample_tot, df_sample_tgt = self.metrics(sample_targs, sample_h0_preds, sample_h1_preds, targetclass=targetclass)
                if df_sample_tot.d_acc[-1]   > 2 * diff_acc:  twice_diff_acc  += 1
                if df_sample_tot.d_f1[-1]    > 2 * diff_f1:   twice_diff_f1   += 1
                if df_sample_tot.d_prec[-1]  > 2 * diff_prec: twice_diff_prec += 1
                if df_sample_tot.d_rec[-1]   > 2 * diff_rec:  twice_diff_rec  += 1
                # print(round(diff_f1, 4), "***", round(df_sample_tot.d_f1[-1], 4), "***", twice_diff_f1)
                if targetclass is not None:
                    if df_sample_tgt.d_tf1[-1]    > 2 * diff_tgt_f1:   twice_diff_tgt_f1   += 1
                    if df_sample_tgt.d_tprec[-1]  > 2 * diff_tgt_prec: twice_diff_tgt_prec += 1
                    if df_sample_tgt.d_trec[-1]   > 2 * diff_tgt_rec:  twice_diff_tgt_rec  += 1
            col_sign_f1   = f"{BColor.red}**{BColor.reset}" if twice_diff_f1   / n_loops < 0.01 else f"{BColor.red}*{BColor.reset}" if twice_diff_f1   / n_loops < 0.05 else f"{BColor.grey}!{BColor.reset}" if twice_diff_f1   / n_loops > 0.95 else f"{BColor.grey}!!{BColor.reset}" if twice_diff_f1   / n_loops > 0.99 else ''
            col_sign_acc  = f"{BColor.red}**{BColor.reset}" if twice_diff_acc  / n_loops < 0.01 else f"{BColor.red}*{BColor.reset}" if twice_diff_acc  / n_loops < 0.05 else f"{BColor.grey}!{BColor.reset}" if twice_diff_acc  / n_loops > 0.95 else f"{BColor.grey}!!{BColor.reset}" if twice_diff_acc  / n_loops > 0.99 else ''
            col_sign_prec = f"{BColor.red}**{BColor.reset}" if twice_diff_prec / n_loops < 0.01 else f"{BColor.red}*{BColor.reset}" if twice_diff_prec / n_loops < 0.05 else f"{BColor.grey}!{BColor.reset}" if twice_diff_prec / n_loops > 0.95 else f"{BColor.grey}!!{BColor.reset}" if twice_diff_prec / n_loops > 0.99 else ''
            col_sign_rec  = f"{BColor.red}**{BColor.reset}" if twice_diff_rec  / n_loops < 0.01 else f"{BColor.red}*{BColor.reset}" if twice_diff_rec  / n_loops < 0.05 else f"{BColor.grey}!{BColor.reset}" if twice_diff_rec  / n_loops > 0.95 else f"{BColor.grey}!!{BColor.reset}" if twice_diff_rec  / n_loops > 0.99 else ''
            sign_f1   = "**" if twice_diff_f1   / n_loops < 0.01 else "*" if twice_diff_f1   / n_loops < 0.05 else "!" if twice_diff_f1   / n_loops > 0.95 else "!!" if twice_diff_f1   / n_loops > 0.99 else ''
            sign_acc  = "**" if twice_diff_acc  / n_loops < 0.01 else "*" if twice_diff_acc  / n_loops < 0.05 else "!" if twice_diff_acc  / n_loops > 0.95 else "!!" if twice_diff_acc  / n_loops > 0.99 else ''
            sign_prec = "**" if twice_diff_prec / n_loops < 0.01 else "*" if twice_diff_prec / n_loops < 0.05 else "!" if twice_diff_prec / n_loops > 0.95 else "!!" if twice_diff_prec / n_loops > 0.99 else ''
            sign_rec  = "**" if twice_diff_rec  / n_loops < 0.01 else "*" if twice_diff_rec  / n_loops < 0.05 else "!" if twice_diff_rec  / n_loops > 0.95 else "!!" if twice_diff_rec  / n_loops > 0.99 else ''
            str_out = f"\n{'count sample diff f1   is twice tot diff f1':.<50} {twice_diff_f1:<5}/ {n_loops:<8}p < {round((twice_diff_f1 / n_loops), 4):<6} {col_sign_f1}\n" \
                      f"{'count sample diff prec is twice tot diff prec':.<50} {twice_diff_prec:<5}/ {n_loops:<8}p < {round((twice_diff_prec / n_loops), 4):<6} {col_sign_prec}\n" \
                      f"{'count sample diff rec  is twice tot diff rec ':.<50} {twice_diff_rec:<5}/ {n_loops:<8}p < {round((twice_diff_rec / n_loops), 4):<6} {col_sign_rec }\n" \
                      f"{'count sample diff acc  is twice tot diff acc':.<50} {twice_diff_acc:<5}/ {n_loops:<8}p < {round((twice_diff_acc / n_loops), 4):<6} {col_sign_acc }"
            df_tot.s_f1   = ['', sign_f1]
            df_tot.s_acc  = ['', sign_acc]
            df_tot.s_prec = ['', sign_prec]
            df_tot.s_rec  = ['', sign_rec]
            if targetclass is not None:
                col_sign_tgt_f1   = f"{BColor.red}**{BColor.reset}" if twice_diff_tgt_f1   / n_loops < 0.01 else f"{BColor.red}*{BColor.reset}" if twice_diff_tgt_f1   / n_loops < 0.05 else f"{BColor.grey}!{BColor.reset}" if twice_diff_tgt_f1   / n_loops > 0.95 else f"{BColor.grey}!!{BColor.reset}" if twice_diff_tgt_f1   / n_loops > 0.99 else ''
                col_sign_tgt_prec = f"{BColor.red}**{BColor.reset}" if twice_diff_tgt_prec / n_loops < 0.01 else f"{BColor.red}*{BColor.reset}" if twice_diff_tgt_prec / n_loops < 0.05 else f"{BColor.grey}!{BColor.reset}" if twice_diff_tgt_prec / n_loops > 0.95 else f"{BColor.grey}!!{BColor.reset}" if twice_diff_tgt_prec / n_loops > 0.99 else ''
                col_sign_tgt_rec  = f"{BColor.red}**{BColor.reset}" if twice_diff_tgt_rec  / n_loops < 0.01 else f"{BColor.red}*{BColor.reset}" if twice_diff_tgt_rec  / n_loops < 0.05 else f"{BColor.grey}!{BColor.reset}" if twice_diff_tgt_rec  / n_loops > 0.95 else f"{BColor.grey}!!{BColor.reset}" if twice_diff_tgt_rec  / n_loops > 0.99 else ''
                sign_tgt_f1   = "**" if twice_diff_tgt_f1   / n_loops < 0.01 else "*" if twice_diff_tgt_f1   / n_loops < 0.05 else "!" if twice_diff_tgt_f1   / n_loops > 0.95 else "!!" if twice_diff_tgt_f1   / n_loops > 0.99 else ''
                sign_tgt_prec = "**" if twice_diff_tgt_prec / n_loops < 0.01 else "*" if twice_diff_tgt_prec / n_loops < 0.05 else "!" if twice_diff_tgt_prec / n_loops > 0.95 else "!!" if twice_diff_tgt_prec / n_loops > 0.99 else ''
                sign_tgt_rec  = "**" if twice_diff_tgt_rec  / n_loops < 0.01 else "*" if twice_diff_tgt_rec  / n_loops < 0.05 else "!" if twice_diff_tgt_rec  / n_loops > 0.95 else "!!" if twice_diff_tgt_rec  / n_loops > 0.99 else ''
                str_out += f"\ntarget {targetclass} {'count sample diff f1   is twice tot diff f1':.<50} {twice_diff_tgt_f1:<5}/ {n_loops:<8}p < {round((twice_diff_tgt_f1 / n_loops), 4):<6} {col_sign_tgt_f1}\n" \
                             f"target {targetclass} {'count sample diff prec is twice tot diff prec':.<50} {twice_diff_tgt_prec:<5}/ {n_loops:<8}p < {round((twice_diff_tgt_prec / n_loops), 4):<6} {col_sign_tgt_prec}\n" \
                             f"target {targetclass} {'count sample diff rec  is twice tot diff rec ':.<50} {twice_diff_tgt_rec:<5}/ {n_loops:<8}p < {round((twice_diff_tgt_rec / n_loops), 4):<6} {col_sign_tgt_rec }"
                df_tgt.s_tf1   = ['', sign_tgt_f1]
                df_tgt.s_tprec = ['', sign_tgt_prec]
                df_tgt.s_trec  = ['', sign_tgt_rec]
            print(str_out)
            if self.savetsv:
                df_tot.to_csv(f"{self.dirout}results.tsv")
                df_tgt.to_csv(f"{self.dirout}results_targetclass.tsv")
            return df_tot, df_tgt
        else:
            diff_jsd = df_tot.d_jsd[-1]
            diff_ce  = df_tot.d_ce[-1]
            diff_sim = df_tot.d_sim[-1]
            diff_cor = df_tot.d_cor[-1]
            twice_diff_jsd = 0
            twice_diff_ce  = 0
            twice_diff_sim = 0
            twice_diff_cor = 0
            diff_tgt_jsd, diff_tgt_ce, twice_diff_tgt_jsd, twice_diff_tgt_ce = None, None, None, None
            for _ in tqdm(range(n_loops), desc='bootstrap', ncols=80):
                i_sample = np.random.choice(range(targs.shape[0]), size=sample_size, replace=True) # Berg-Kirkpatrick, p. 996: "with replacement"
                sample_targs    = targs[i_sample]
                sample_h0_preds = h0_preds[i_sample]
                sample_h1_preds = h1_preds[i_sample]
                df_sample_tot, df_sample_tgt = self.metrics(sample_targs, sample_h0_preds, sample_h1_preds, targetclass=targetclass)
                if df_sample_tot.d_jsd[-1]  > -2 * diff_jsd: twice_diff_jsd  += 1 # lower is better
                if df_sample_tot.d_ce[-1]   > -2 * diff_ce:  twice_diff_ce   += 1 # lower is better
                if df_sample_tot.d_sim[-1]  > 2 * diff_sim: twice_diff_sim += 1
                if df_sample_tot.d_cor[-1]  > 2 * diff_cor: twice_diff_cor += 1
            col_sign_jsd = f"{BColor.red}**{BColor.reset}" if twice_diff_jsd / n_loops < 0.01 else f"{BColor.red}*{BColor.reset}" if twice_diff_jsd / n_loops < 0.05 else f"{BColor.grey}!{BColor.reset}" if twice_diff_jsd / n_loops > 0.95 else f"{BColor.grey}!!{BColor.reset}" if twice_diff_jsd / n_loops > 0.99 else ''
            col_sign_ce  = f"{BColor.red}**{BColor.reset}" if twice_diff_ce  / n_loops < 0.01 else f"{BColor.red}*{BColor.reset}" if twice_diff_ce  / n_loops < 0.05 else f"{BColor.grey}!{BColor.reset}" if twice_diff_ce  / n_loops > 0.95 else f"{BColor.grey}!!{BColor.reset}" if twice_diff_ce  / n_loops > 0.99 else ''
            col_sign_sim = f"{BColor.red}**{BColor.reset}" if twice_diff_sim / n_loops < 0.01 else f"{BColor.red}*{BColor.reset}" if twice_diff_sim / n_loops < 0.05 else f"{BColor.grey}!{BColor.reset}" if twice_diff_sim / n_loops > 0.95 else f"{BColor.grey}!!{BColor.reset}" if twice_diff_sim / n_loops > 0.99 else ''
            col_sign_cor = f"{BColor.red}**{BColor.reset}" if twice_diff_cor / n_loops < 0.01 else f"{BColor.red}*{BColor.reset}" if twice_diff_cor / n_loops < 0.05 else f"{BColor.grey}!{BColor.reset}" if twice_diff_cor / n_loops > 0.95 else f"{BColor.grey}!!{BColor.reset}" if twice_diff_cor / n_loops > 0.99 else ''
            sign_jsd = "**" if twice_diff_jsd / n_loops < 0.01 else "*" if twice_diff_jsd / n_loops < 0.05 else "!" if twice_diff_jsd / n_loops > 0.95 else "!!" if twice_diff_jsd / n_loops > 0.99 else ''
            sign_ce  = "**" if twice_diff_ce  / n_loops < 0.01 else "*" if twice_diff_ce  / n_loops < 0.05 else "!" if twice_diff_ce  / n_loops > 0.95 else "!!" if twice_diff_ce  / n_loops > 0.99 else ''
            sign_sim = "**" if twice_diff_sim / n_loops < 0.01 else "*" if twice_diff_sim / n_loops < 0.05 else "!" if twice_diff_sim / n_loops > 0.95 else "!!" if twice_diff_sim / n_loops > 0.99 else ''
            sign_cor = "**" if twice_diff_cor / n_loops < 0.01 else "*" if twice_diff_cor / n_loops < 0.05 else "!" if twice_diff_cor / n_loops > 0.95 else "!!" if twice_diff_cor / n_loops > 0.99 else ''
            str_out = f"{'count sample diff jsd is twice tot diff jsd':.<50} {twice_diff_jsd:<5}/ {n_loops:<8}p < {round((twice_diff_jsd / n_loops), 4):<6} {col_sign_jsd}\n" \
                      f"{'count sample diff ce  is twice tot diff ce':.<50} {twice_diff_ce:<5}/ {n_loops:<8}p < {round((twice_diff_ce / n_loops), 4):<6} {col_sign_ce }\n" \
                      f"{'count sample diff sim is twice tot diff sim':.<50} {twice_diff_sim:<5}/ {n_loops:<8}p < {round((twice_diff_sim / n_loops), 4):<6} {col_sign_sim}\n" \
                      f"{'count sample diff cor is twice tot diff cor':.<50} {twice_diff_cor:<5}/ {n_loops:<8}p < {round((twice_diff_cor / n_loops), 4):<6} {col_sign_cor }"
            df_tot.s_jsd = ['', sign_jsd]
            df_tot.s_ce  = ['', sign_ce]
            df_tot.s_sim = ['', sign_sim]
            df_tot.s_cor = ['', sign_cor]
            print(str_out)
            if self.savetsv:
                df_tot.to_csv(f"{self.dirout}results.tsv")
            return df_tot, df_tgt

    def run(self, n_loops=1000, sample_size=.1, targetclass=None, verbose=False):
        """
        :param n_loops: type: int; default: 1000. number of iterations for bootstrap sampling.
        :param sample_size: type: float; default: .1. Percent sample size wrt the test set, f
        or bootstrap sampling.
        :param targetclass: type: int; default: None. Index of a target class. If given, performance and
        significance levels are computed for that class.
        :param verbose: type: bool; default: False. If true, it prints performance information.
        :return: two pandas DataFrame: the first contain the overall performance,
        the second the target class performance (empty if not requested)
        """
        startime = start()

        df_tot, df_tgt = pd.DataFrame(), pd.DataFrame()
        for h0_cond in self.data:
            print('#'*80)
            h0_preds_all, h0_targs_all, h0_idxs_all = np.empty([0, self.data[h0_cond]['targs'][0].shape[1]]), np.empty([0, self.data[h0_cond]['targs'][0].shape[1]]), np.empty([0, 1])
            h0_f1_all, h0_f1tgt_all, h0_jsd_all, h0_jsdtgt_all = list(), list(), list(), list()
            for exp_idx, preds, targs, idxs in zip(self.data[h0_cond]['exp_idxs'], self.data[h0_cond]['preds'], self.data[h0_cond]['targs'], self.data[h0_cond]['idxs']):
                h0_preds_all = np.concatenate([h0_preds_all, preds], axis=0)
                h0_targs_all = np.concatenate([h0_targs_all, targs], axis=0)
                h0_idxs_all  = np.concatenate([h0_idxs_all, idxs], axis=0)
                if targs.shape[1] == 1:
                    f1 = f1_score(targs, preds, average='macro')
                    h0_f1_all.append(f1)
                    if targetclass is not None:
                        h0_f1tgt_all.append(precision_recall_fscore_support(targs, preds)[2][targetclass])
                    if verbose:
                        print(f"{exp_idx:<60} F1 {f1:.4f}")
                else:
                    jsd = jensenshannon(targs, preds, axis=1).mean()
                    h0_jsd_all.append(jsd)
                    if verbose:
                        print(f"{exp_idx:<60} jsd {jsd:.4f}")

            for h1_cond in self.data[h0_cond]['h1']:
                print(f"{'#'*80}\n{h0_cond}   vs   {h1_cond}")
                h1_preds_all, h1_targs_all, h1_idxs_all = np.empty([0, self.data[h0_cond]['targs'][0].shape[1]]), np.empty([0, self.data[h0_cond]['targs'][0].shape[1]]), np.empty([0, 1])
                h1_f1_all, h1_f1tgt_all, h1_jsd_all, h1_jsdtgt_all = list(), list(), list(), list()
                for exp_idx, preds, targs, idxs in zip(self.data[h0_cond]['h1'][h1_cond]['exp_idxs'],
                                                       self.data[h0_cond]['h1'][h1_cond]['preds'],
                                                       self.data[h0_cond]['h1'][h1_cond]['targs'],
                                                       self.data[h0_cond]['h1'][h1_cond]['idxs']):
                    h1_preds_all = np.concatenate([h1_preds_all, preds], axis=0)
                    h1_targs_all = np.concatenate([h1_targs_all, targs], axis=0)
                    h1_idxs_all  = np.concatenate([h1_idxs_all, idxs], axis=0)
                    if targs.shape[1] == 1:
                        f1 = f1_score(targs, preds, average='macro')
                        h1_f1_all.append(f1)
                        if targetclass is not None:
                            h1_f1tgt_all.append(precision_recall_fscore_support(targs, preds)[2][targetclass])
                        if verbose:
                            print(f"{exp_idx:<60} F1 {f1:.4f}")
                    else:
                        jsd = jensenshannon(targs, preds, axis=1).mean()
                        h1_jsd_all.append(jsd)
                        if verbose:
                            print(f"{exp_idx:<60} jsd {jsd:.4f}")
                assert np.array(h0_targs_all == h1_targs_all).all(), 'h0 and h1 targets differ'
                assert np.array(h0_idxs_all == h1_idxs_all).all(), 'h0 and h1 idxs differ'
                targs_all = h0_targs_all
                
                df_tot_cond, df_tgt_cond = self.test(targs_all, h0_preds_all, h1_preds_all, h0_name=h0_cond, h1_name=h1_cond, n_loops=n_loops, sample_size=sample_size, targetclass=targetclass, verbose=True)

                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore') # possible mean on empty list
                    df_tot_cond['mean_epochs'] = [round(np.mean(self.data[h0_cond]['epochs']), 2), round(np.mean(self.data[h0_cond]['h1'][h1_cond]['epochs']), 2)]
                if self.data[h0_cond]['targs'][0].shape[1] == 1:
                    df_tot_cond['std_f1']      = [round(np.std(h0_f1_all), 2), round(np.std(h1_f1_all), 2)]
                else:
                    df_tot_cond['std_jsd']     = [round(np.std(h0_jsd_all), 2), round(np.std(h1_jsd_all), 2)]
                if h0_cond not in df_tot.index:
                    df_tot = df_tot.append(df_tot_cond.iloc[0, :])
                if h1_cond not in df_tot.index:
                    df_tot = df_tot.append(df_tot_cond.iloc[1, :])

                if targetclass is not None:
                    with warnings.catch_warnings():
                        warnings.filterwarnings('ignore') # possible mean on empty list
                        df_tgt_cond['mean_epochs'] = [round(np.mean(self.data[h0_cond]['epochs']), 2), round(np.mean(self.data[h0_cond]['h1'][h1_cond]['epochs']), 2)]
                    if self.data[h0_cond]['targs'][0].shape[1] == 1:
                        df_tgt_cond['std_tf1']  = [round(np.std(h0_f1tgt_all), 2), round(np.std(h1_f1tgt_all), 2)]
                    if h0_cond not in df_tgt.index:
                        df_tgt = df_tgt.append(df_tgt_cond.iloc[0, :])
                    if h1_cond not in df_tgt.index:
                        df_tgt = df_tgt.append(df_tgt_cond.iloc[1, :])

        if self.data[list(self.data.keys())[0]]['targs'][0].shape[1] == 1:
            df_tot = df_tot['mean_epochs f1 d_f1 s_f1 std_f1 acc d_acc s_acc prec d_prec s_prec rec d_rec s_rec'.split()].round(4)
        else:
            df_tot = df_tot['mean_epochs jsd d_jsd s_jsd std_jsd ce d_ce s_ce sim d_sim s_sim cor d_cor s_cor'.split()].round(4)
        if targetclass is not None:
            if self.data[list(self.data.keys())[0]]['targs'][0].shape[1] == 1:
                df_tgt = df_tgt['mean_epochs tf1 d_tf1 s_tf1 std_tf1 tprec d_tprec s_tprec trec d_trec s_trec'.split()].round(4)
            df_both = pd.concat([df_tot, df_tgt.iloc[:, 1:]], axis=1)
            print(df_both.to_string())
            if self.savetsv:
                df_tgt.to_csv(f"{self.dirout}results_targetclass.tsv")
                df_both.to_csv(f"{self.dirout}results_overall.tsv")
        else:
            print(df_tot.to_string())
        if self.savejson:
            self.data2json(f"{self.dirout}outcomes.json")
        if self.savetsv:
            df_tot.to_csv(f"{self.dirout}results.tsv")
        end(startime)
        return df_tot, df_tgt
