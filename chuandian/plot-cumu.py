
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import config

fonten = config.fonten
file_location = config.file_location

catalog = pd.read_csv(file_location+r'\catalog\filter_catalog.txt', 
    dtype=str, delimiter=' ', header=None)
catalog = np.array(catalog)
catalog = np.around(np.array(catalog[:,-1], dtype=np.float32), 1)

count = []
for i in np.arange(3.0, max(catalog)+0.05, 0.1):       
    names = globals()
    names['count_' + str(i)] = np.sum(i==catalog)
    count.append(names['count_' + str(i)])

fig = plt.figure(figsize=(8, 6))
plt.grid(True, linestyle='--', linewidth=1.5)
plt.plot(np.arange(3.0, max(catalog)+0.05, 0.1), count, color='dodgerblue', linewidth=2)
plt.xlabel('Earthquake Magnitude', fontsize=18)
plt.ylabel('Cumulative frequency of earthquakes', fontsize=18)
plt.title(r'Frequency-M', fontdict=fonten, fontsize=20, color='red')
plt.tight_layout()
plt.xticks(size=14)
plt.yticks(size=14)
ax = plt.gca()
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
ax.spines['top'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)
plt.savefig(r'.\figure\cumulative.pdf')
plt.show()

