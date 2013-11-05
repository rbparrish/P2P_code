import datetime
import re


def parse_employment_years(x):
    m = re.search(r'(\<?)(\d{1,2})\+? years?', x)
    if m:
        if m.group(1) == '<' and m.group(2) == '1':
            return 0
        return int(m.group(2))
    return None



def parse_date(x, format='%Y-m-d'):
    d = datetime.strptime(x, format)


def to_num(x):
    try:
        return float(re.sub(r'[^0-9\.]', '', x))
    except ValueError:
        return "Error: do not proceed"

def binary_true_false(x):
    if x != False:
        return 1
    return 0
    # note, can use '.fillna(False)' before this to have all NaN be 0(False), and everything else be 1(True)