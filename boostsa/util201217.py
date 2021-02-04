# coding=latin-1
import sys
import re
import time
import json

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


