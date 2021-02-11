# coding=latin-1
# Non nobis, Domine, non nobis, sed Nomini Tuo da gloriam
import sys, re, time, json
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, precision_recall_fscore_support, log_loss, confusion_matrix


###############################################################################


def stringtime(n):
    h = str(int(n / 3600))
    m = str(int((n % 3600) / 60))
    s = str(int((n % 3600) % 60))
    if len(h) == 1: h = '0' + h
    if len(m) == 1: m = '0' + m
    if len(s) == 1: s = '0' + s
    return h + ':' + m + ':' + s


def start(sep=True):
    start = time.time()
    now = time.strftime("%Y/%m/%d %H:%M:%S")
    if sep: print('#'*80)
    print('start:', now)
    return start


def end(start, sep=True):
    end = time.time()
    dur = end - start
    str_dur = stringtime(end - start)
    now = time.strftime("%Y/%m/%d %H:%M:%S")
    if sep:
        print('#'*80 + "\nend:", now, " - time elapsed:", str_dur + "\n" + '#'*80)
    else:
        print("end:", now, " - time elapsed:", str_dur)
    return dur


###############################################################################


def get_lines(path, maxrow=-1):
    """the function traverses a generator until maxrow or StopIteration.
    the output is the another generator, having the wanted nr of lines"""
    generator = open(path)
    row_counter = 0
    while maxrow != 0:
        try:
            line = next(generator)
            maxrow -= 1
            row_counter += 1
            yield line
        except StopIteration:
            print(f"{'nr rows in generator:':.<40} {row_counter}")
            maxrow = 0


def read_file(filename, code='utf-8'):
    with open(filename, 'r', encoding=code) as f_in:
        out = f_in.read()
        return out


def writejson(data, pathname):
    with open(pathname, 'w') as out: json.dump(data, out)
    return 1


def printjson(data, stop=None):
    for i, k in enumerate(data):
        if i == stop: break
        print(f"{k}: {json.dumps(data[k], indent=4, ensure_ascii=False)}") # ensure ascii false permette di stampare i caratteri utf-8 invece di \usblindo
    return 1

    
###############################################################################


class bcolors:
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

    i2red = {0:  ([255/255, 204/255, 204/255]), # quasi bianco
             1:  ([255/255, 153/255, 153/255]),
             2:  ([255/255, 102/255, 102/255]),
             3:  ([255/255, 51/255,  51/255]),
             4:  ([255/255, 0/255,   0/255]), # rosso
             5:  ([204/255, 0/255,   0/255]),
             6:  ([153/255, 0/255,   0/255]),
             7:  ([102/255, 0/255,   0/255]),
             8:  ([51/255,  0/255,   0/255])} # quasi nero

    def examples(self):
        for i in range(0, 16):
            for j in range(0, 16):
                code = str(i * 16 + j)
                sys.stdout.write(u"\u001b[38;5;" + code + "m " + code.ljust(4))
        for i in range(0, 16):
            for j in range(0, 16):
                code = str(i * 16 + j)
                sys.stdout.write(u"\u001b[48;5;" + code + "m " + code.ljust(4))


###############################################################################

class Bootstrap:
    def __init__(self, save_results=True, save_outcomes=True, dir_out=''):
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

    def feed(self, h0, h1=None, exp_idx=None, preds=None, targs=None, idxs=None, epochs=None):
        targs = self.input2list(targs)
        preds = self.input2list(preds)
        idxs = self.input2list(idxs)
        assert len(preds) == len(targs) == len(idxs), 'preds, targs or idxs have different length'
        if h1:
            self.data[h0]['h1'][h1]['exp_idxs'].append(exp_idx)
            self.data[h0]['h1'][h1]['preds'].append(preds)
            self.data[h0]['h1'][h1]['targs'].append(targs)
            self.data[h0]['h1'][h1]['idxs'].append(idxs)
            self.data[h0]['h1'][h1]['epochs'].append(epochs)
        else:
            self.data[h0]['exp_idxs'].append(exp_idx)
            self.data[h0]['preds'].append(preds)
            self.data[h0]['targs'].append(targs)
            self.data[h0]['idxs'].append(idxs)
            self.data[h0]['epochs'].append(epochs)
        return 1

    def loadjson(self, pathname):
        with open(pathname) as f_in: jin = json.load(f_in)
        for h0 in jin:
            self.data[h0]['exp_idxs'] = jin[h0]['exp_idxs']
            self.data[h0]['preds']    = jin[h0]['preds']
            self.data[h0]['targs']    = jin[h0]['targs']
            self.data[h0]['idxs']     = jin[h0]['idxs']
            self.data[h0]['epochs']   = jin[h0]['epochs']
            for h1 in jin[h0]['h1']:
                self.data[h0]['h1'][h1]['exp_idxs'] = jin[h0]['h1'][h1]['exp_idxs']
                self.data[h0]['h1'][h1]['preds']    = jin[h0]['h1'][h1]['preds']
                self.data[h0]['h1'][h1]['targs']    = jin[h0]['h1'][h1]['targs']
                self.data[h0]['h1'][h1]['idxs']     = jin[h0]['h1'][h1]['idxs']
                self.data[h0]['h1'][h1]['epochs']   = jin[h0]['h1'][h1]['epochs']
        
        return 1

    @staticmethod
    def input2list(object, encoding='utf-8', elemtype=int, sep="\n", emptyend=True):
        if type(object) is list:
            return object
        else:
            with open(object, 'r', encoding=encoding) as input_file: str_file = input_file.read()
            if emptyend: str_file = re.sub("\n+$", '', str_file)
            if elemtype == int:
                out = [float(x) for x in str_file.split(sep)]
                out = [int(x) for x in out]
            elif elemtype == float:
                out = [float(x) for x in str_file.split(sep)]
            else:
                out = [x for x in str_file.split(sep)]
            return out
        
    @staticmethod
    def metrics(targs, h0_preds, h1_preds, h0_name='h0', h1_name='h1', verbose=False):
        rounding_value = 2
        h0_acc  = round(accuracy_score(targs, h0_preds) * 100, rounding_value)
        h0_f1   = round(f1_score(targs, h0_preds, average='macro') * 100, rounding_value)
        h0_prec = round(precision_score(targs, h0_preds, average='macro') * 100, rounding_value)
        h0_rec  = round(recall_score(targs, h0_preds, average='macro') * 100, rounding_value)
        h1_acc  = round(accuracy_score(targs, h1_preds) * 100, rounding_value)
        h1_f1   = round(f1_score(targs, h1_preds, average='macro') * 100, rounding_value)
        h1_prec = round(precision_score(targs, h1_preds, average='macro') * 100, rounding_value)
        h1_rec  = round(recall_score(targs, h1_preds, average='macro') * 100, rounding_value)
        # h0_conf_matrix = confusion_matrix(targs, h0_preds)
        # h1_conf_matrix = confusion_matrix(targs, h1_preds)
        diff_acc  = round(h1_acc - h0_acc, rounding_value)
        diff_f1   = round(h1_f1  - h0_f1, rounding_value)
        diff_prec = round(h1_prec - h0_prec, rounding_value)
        diff_rec  = round(h1_rec  - h0_rec, rounding_value)
        if verbose:
            h0_countpreds = Counter(h0_preds)
            h1_countpreds = Counter(h1_preds)
            counttargs    = Counter(targs)
            h0_countpreds = [f"class {tup[0]} freq {tup[1]} perc {tup[1] / len(h0_preds) * 100:.2f}%" for tup in sorted({k: h0_countpreds[k] for k in h0_countpreds}.items(), key=lambda item: item[0])]
            h1_countpreds = [f"class {tup[0]} freq {tup[1]} perc {tup[1] / len(h1_preds) * 100:.2f}%" for tup in sorted({k: h1_countpreds[k] for k in h1_countpreds}.items(), key=lambda item: item[0])]
            counttargs    = [f"class {tup[0]} freq {tup[1]} perc {tup[1] / len(targs) * 100:.2f}%" for tup in sorted({k: counttargs[k] for k in counttargs}.items(), key=lambda item: item[0])]
            print(f"{'targs count:':<25} {counttargs}")
            print(f"{h0_name + ' preds count:':<25} {h0_countpreds}")
            print(f"{h1_name + ' preds count:':<25} {h1_countpreds}")
            print(f"{h0_name + ' F-measure':.<25} {h0_f1:<7} {h1_name + ' F-measure':.<25} {h1_f1:<7} {'diff':.<7} {diff_f1}")
            print(f"{h0_name + ' accuracy':.<25} {h0_acc:<7} {h1_name + ' accuracy':.<25} {h1_acc:<7} {'diff':.<7} {diff_acc}")
            print(f"{h0_name + ' precision':.<25} {h0_prec:<7} {h1_name + ' precision':.<25} {h1_prec:<7} {'diff':.<7} {diff_prec}")
            print(f"{h0_name + ' recall':.<25} {h0_rec:<7} {h1_name + ' recall':.<25} {h1_rec:<7} {'diff':.<7} {diff_rec}")
        return pd.DataFrame({'f1':   [h0_f1,   h1_f1],   'diff_f1':   ['',  diff_f1],   'sign_f1':   ['', ''],
                             'acc':  [h0_acc,  h1_acc],  'diff_acc':  ['',  diff_acc],  'sign_acc':  ['', ''],
                             'prec': [h0_prec, h1_prec], 'diff_prec': ['',  diff_prec], 'sign_prec': ['', ''],
                             'rec':  [h0_rec,  h1_rec],  'diff_rec':  ['',  diff_rec],  'sign_rec':  ['', ''],
                             }, index=[h0_name, h1_name])

    def test(self, targs, h0_preds, h1_preds, h0_name='h0', h1_name='h1', n_loops=100, sample_size=.1, verbose=False):
        assert .05 <= sample_size <= .5, 'sample_size must be between .05 and .5'
        targs    = self.input2list(targs)
        h0_preds = self.input2list(h0_preds)
        h1_preds = self.input2list(h1_preds)
        overall_size = len(targs)
        sample_size = int(len(targs) * sample_size)
        print(f"{'total size':.<25} {overall_size}\n{'sample size':.<25} {sample_size}")
        targs    = np.array(targs)
        h0_preds = np.array(h0_preds)
        h1_preds = np.array(h1_preds)
        df_tot = self.metrics(targs, h0_preds, h1_preds, h0_name=h0_name, h1_name=h1_name, verbose=verbose)
        diff_acc  = df_tot.diff_acc[-1]
        diff_f1   = df_tot.diff_f1[-1]
        diff_prec = df_tot.diff_prec[-1]
        diff_rec  = df_tot.diff_rec[-1]
        twice_diff_acc  = 0
        twice_diff_f1   = 0
        twice_diff_prec = 0
        twice_diff_rec  = 0
        for _ in tqdm(range(n_loops), desc='bootstrap', ncols=80):
            i_sample = np.random.choice(range(overall_size), size=sample_size, replace=False)
            sample_h0_preds = h0_preds[i_sample]
            sample_h1_preds = h1_preds[i_sample]
            sample_targs    = targs[i_sample]
            df_sample       = self.metrics(sample_targs, sample_h0_preds, sample_h1_preds)
            if df_sample.diff_acc[-1]   > 2 * diff_acc:  twice_diff_acc  += 1
            if df_sample.diff_f1[-1]    > 2 * diff_f1:   twice_diff_f1   += 1
            if df_sample.diff_prec[-1]  > 2 * diff_prec: twice_diff_prec += 1
            if df_sample.diff_rec[-1]   > 2 * diff_rec:  twice_diff_rec  += 1
        sign_f1   = '**' if twice_diff_f1   / n_loops < 0.01 else '*' if twice_diff_f1   / n_loops < 0.05 else ''
        sign_acc  = '**' if twice_diff_acc  / n_loops < 0.01 else '*' if twice_diff_acc  / n_loops < 0.05 else ''
        sign_prec = '**' if twice_diff_prec / n_loops < 0.01 else '*' if twice_diff_prec / n_loops < 0.05 else ''
        sign_rec  = '**' if twice_diff_rec  / n_loops < 0.01 else '*' if twice_diff_rec  / n_loops < 0.05 else ''
        str_out = f"{'count sample diff f1   is twice tot diff f1':.<50} {twice_diff_f1:<5}/ {n_loops:<8}p < {round((twice_diff_f1 / n_loops), 4):<6} {bcolors.red}{sign_f1  }{bcolors.reset}\n" \
                  f"{'count sample diff acc  is twice tot diff acc':.<50} {twice_diff_acc:<5}/ {n_loops:<8}p < {round((twice_diff_acc / n_loops), 4):<6} {bcolors.red}{sign_acc }{bcolors.reset}\n" \
                  f"{'count sample diff prec is twice tot diff prec':.<50} {twice_diff_prec:<5}/ {n_loops:<8}p < {round((twice_diff_prec / n_loops), 4):<6} {bcolors.red}{sign_prec}{bcolors.reset}\n" \
                  f"{'count sample diff rec  is twice tot diff rec ':.<50} {twice_diff_rec:<5}/ {n_loops:<8}p < {round((twice_diff_rec / n_loops), 4):<6} {bcolors.red}{sign_rec }{bcolors.reset}"
        print(str_out)
        df_tot.sign_f1   = ['', sign_f1]
        df_tot.sign_acc  = ['', sign_acc]
        df_tot.sign_prec = ['', sign_prec]
        df_tot.sign_rec  = ['', sign_rec]
        if self.savetsv:
            df_tot.to_csv(f"{self.dirout}results.tsv")
        return df_tot

    def run(self, n_loops=100, sample_size=.1, verbose=False):
        """
        :param data:
                defaultdict(lambda: {'exp_idxs': list(), 'preds': list(), 'targs': list(), 'idxs': list(), 'epochs': list(),
                                     'h1': defaultdict(lambda: {'exp_idxs': list(), 'preds': list(), 'targs': list(), 'idxs': list(), 'epochs': list()})}
        """
        startime = start()

        df = pd.DataFrame(columns="mean_epochs acc diff_acc sign_acc prec diff_prec sign_prec rec diff_rec sign_rec f1 diff_f1 sign_f1".split())
        for h0_cond in self.data:
            print('#'*80)
            h0_preds_all, h0_targs_all, h0_idxs_all = list(), list(), list()
            for exp_idx, preds, targs, idxs in zip(self.data[h0_cond]['exp_idxs'], self.data[h0_cond]['preds'], self.data[h0_cond]['targs'], self.data[h0_cond]['idxs']):
                acc = round(accuracy_score(targs, preds) * 100, 2)
                f1  = round(f1_score(targs, preds, average='macro') * 100, 2)
                h0_preds_all.extend(preds)
                h0_targs_all.extend(targs)
                h0_idxs_all.extend(idxs)
                if verbose:  print(f"{exp_idx:<60} acc {acc:<7} F {f1}")
            for h1_cond in self.data[h0_cond]['h1']:
                print(f"{'#'*80}\n{h0_cond}   vs   {h1_cond}")
                h1_preds_all, h1_targs_all, h1_idxs_all = list(), list(), list()
                for exp_idx, preds, targs, idxs in zip(self.data[h0_cond]['h1'][h1_cond]['exp_idxs'],
                                                       self.data[h0_cond]['h1'][h1_cond]['preds'],
                                                       self.data[h0_cond]['h1'][h1_cond]['targs'],
                                                       self.data[h0_cond]['h1'][h1_cond]['idxs']):
                    acc = round(accuracy_score(targs, preds) * 100, 2)
                    f1  = round(f1_score(targs, preds, average='macro') * 100, 2)
                    h1_preds_all.extend(preds)
                    h1_targs_all.extend(targs)
                    h1_idxs_all.extend(idxs)
                    if verbose: print(f"{exp_idx:<60} acc {acc:<7} F {f1}")
                # print(len(h0_targs_all), len(h1_targs_all), h0_targs_all[:7], h1_targs_all[:7])
                assert h0_targs_all == h1_targs_all, 'h0 and h1 targets differ'
                assert h0_idxs_all == h1_idxs_all, 'h0 and h1 idxs differ'
                targs_all = h0_targs_all
                
                df_out = self.test(targs_all, h0_preds_all, h1_preds_all, h0_name=h0_cond, h1_name=h1_cond, n_loops=n_loops, sample_size=sample_size, verbose=True)
                df_out['mean_epochs'] = [round(np.mean(self.data[h0_cond]['epochs']), 2), round(np.mean(self.data[h0_cond]['h1'][h1_cond]['epochs']), 2)] if ((self.data[h0_cond]['epochs'][0] is not None) and (self.data[h0_cond]['h1'][h1_cond]['epochs'][0] is not None)) else [None, None]

                if h0_cond not in df.index:
                    df = df.append(df_out.iloc[0, :])
                if h1_cond not in df.index:
                    df = df.append(df_out.iloc[1, :])
        if self.savetsv:
            df.to_csv(f"{self.dirout}results.tsv")
        if self.savejson:
            writejson(self.data, f"{self.dirout}outcomes.json")
        print(df.to_string())
        end(startime)
        return df



