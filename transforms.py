transforms.py


def strip_non_numeric_and_parse(x):
    try:
        return float(re.sub(r'[^0-9\.]', '', x))
    except ValueError:
        return "Error: do not proceed"

def present_check(x):

