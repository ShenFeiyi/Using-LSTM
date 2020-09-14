import os
import numpy as np


txtpath = os.path.join('.','Mao')
dirs = os.listdir(txtpath)
txtnames = [name for name in dirs if name.endswith('txt')]

d = {}
ignore = '\u3000'+'\x1a'

for name in txtnames:
    with open(os.path.join(txtpath,name),'r') as txt:
        content = txt.read()
        for word in content:
            if word in ignore:
                continue
            elif word in ['\t']:
                word = 'tab'
                d[word] = d.get(word,0) + 1
            elif word in ['\n']:
                word = 'enter'
                d[word] = d.get(word,0) + 1
            else:
                d[word] = d.get(word,0) + 1

h2l = sorted(d.items(),key=lambda x:x[1],reverse=True)
order = {}
total = 0
for item in h2l:
    for i in range(total,total+item[1]):
        order[i] = item[0]
    total += item[1]

count = 0
with open('word.txt','w') as file:
    for item in h2l:
        file.write(str(count).zfill(4))
        file.write('\t')
        file.write(str(item[0]))
        file.write('\t')
        file.write(str(item[1]))
        file.write('\n')
        count += 1

length = int(1e5)
with open('nonsense.txt','w') as file:
    for _ in range(length):
        rand = np.random.randint(total)
        word = order[rand]
        if word in ['tab']:
            word = '\t'
        if word in ['enter']:
            word = '\n'
        file.write(word)
