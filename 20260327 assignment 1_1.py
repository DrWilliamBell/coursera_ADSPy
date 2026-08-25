import numpy as np
import re
# Part A
def names():
    simple_string = """Amy is 5 years old, and her sister Mary is 2 years old. 
    Ruth and Peter, their parents, have 3 kids."""
    # YOUR CODE HERE
    pattern = r"[A-Z]\w+"
    return re.findall(pattern, simple_string)
    raise NotImplementedError()

print(names())
assert len(names()) == 4, "There are 4 names here"

# Part B
import re
def grades():
    with open ("grades.txt", "r") as file:
        grades = file.read()
    # YOUR CODE HERE
        return re.findall("(.+)(?=: B)", grades)
    raise NotImplementedError()

print(grades())
assert len(grades()) == 16

# Part C
import re
def logs():
    with open("logdata.txt", "r") as file:
        logdata = file.read()
    # YOUR CODE HERE
    result = []
    pattern = """
    (?P<host>.*)            #IP Address (host)
    (\\ -\\ )               #an indicator of the user_name
    (?P<user_name>\\w*|-)   #user_name (incl. "-")
    (\\ \\[)                #separator for the time
    (?P<time>.*)            #the time
    (\\]\\ ")               #separator for the request
    (?P<request>.*)         #the request
    (?=")                   #final separator"""

    for item in re.finditer(pattern, logdata, re.VERBOSE):
        result.append(item.groupdict())

    return result

    raise NotImplementedError()

print(logs())
assert len(logs()) == 979

one_item={'host': '146.204.224.152',
  'user_name': 'feest6811',
  'time': '21/Jun/2019:15:45:24 -0700',
  'request': 'POST /incentivize HTTP/1.1'}
assert one_item in logs(), "Sorry, this item should be in the log results, check your formating"


