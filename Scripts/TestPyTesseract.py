import pandas as pd
import random

import requests

word_site = "https://www.mit.edu/~ecprice/wordlist.10000"


list1 = []
for i in range(7):
    list2 = []
    for j in range(6):
        response = requests.get(word_site)
        WORDS = response.content.splitlines()[int(random.random()*100)]
        list2.append(WORDS.decode('utf-8'))
    list1.append(list2)

print(list1)
