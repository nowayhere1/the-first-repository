import re

def name_change(string):
    pattern = r'(ice|concrete|mud|asphalt|gravel|snow)'
    match = re.search(pattern, string)
    if match:
        return match.group(0)
    else:
        return str("scsc")

