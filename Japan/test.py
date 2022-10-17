
import pandas as pd
import numpy as np

file_location = r'C:\Users\cshu\Desktop\shi\Japan'

lines = []
with open(file_location+r'\h1919', mode='r', encoding='utf-8') as fin:         
    with open(file_location+r'\h1919.txt', mode='w+', encoding='utf-8') as fout:
        for line in fin:
            for fh in fin:
                for f in fh:
                    if f != '\n':
                        fout.writelines(' '.join([f+'#']))
                    else:
                        fout.writelines('\n')
