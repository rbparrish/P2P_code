import datetime
import re



def parse_employment_years(x):
    m = re.search(r'(\<?)(\d{1,2})\+? years?', x)
    if m:
        if m.group(1) == '<' and m.group(2) == '1':
            return 0
        return int(m.group(2))
    return None
    	(need to find out )


def parse_date(x, format='%Y-m-d'):
    d = datetime.strptime(x, format)


def strip_non_numeric_and_parse(x):
    try:
        return float(re.sub(r'[^0-9\.]', '', x))
    except ValueError:
        return 0.0


