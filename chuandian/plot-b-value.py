import config

np = config.np
pd = config.pd
plt = config.plt
catolog_name = config.catolog_name
file_location = config.file_location

factor = pd.read_csv(file_location+r'\factor\factor-'+catolog_name+r'.txt', 
    delimiter=' ', header=None)
factor = np.array(factor)

catalog = pd.read_csv(file_location+r'\catalog\filter_catalog.txt', 
    dtype=str, delimiter=' ', header=None)
catalog = np.array(catalog)

fig = plt.figure(figsize=(6, 6))
plt.scatter(factor[:, 4], factor[:, 5], c='', edgecolor='dodgerblue')
plt.xlabel('b Value (Least Squares)', fontsize=18)
plt.ylabel('b Value (Maximum Likehood Estimate)', fontsize=18)
plt.title('b Value', fontsize=20, color='red')
plt.xticks(size=14)
plt.yticks(size=14)
plt.xlim(-0.1, 2.6)
plt.ylim(-0.1, 2.6)
plt.grid(True, linestyle='--', linewidth=1.5)
ax = plt.gca()
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
ax.spines['top'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)
ax.set_aspect(aspect='equal')
plt.tight_layout()
plt.show()

fig = plt.figure(figsize=(12, 6))
plt.scatter(np.arange(len(factor[:, 0])), factor[:, 0], c='', edgecolor='dodgerblue')
plt.xlabel('Sample Index', fontsize=18)
plt.ylabel('Max. $M_L$ for Next Year', fontsize=18)
plt.title('M-t', fontsize=20, color='red')
plt.xticks(size=14)
plt.yticks(size=14)
plt.grid(True, linestyle='--', linewidth=1.5)
ax = plt.gca()
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
ax.spines['top'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)
plt.tight_layout()
plt.show()

fig = plt.figure(figsize=(14, 6))
plt.subplot(1,1,1) 
plt.scatter(np.arange(len(catalog)), np.array(catalog[:, -1], dtype=np.float32), 
    c='', edgecolor='dodgerblue')
plt.xlabel('Sample Index (1970-2020)', fontsize=18)
plt.ylabel('ML/Ms/mb', fontsize=18)
plt.title(u'M-t', fontsize=20, color='red')
plt.xticks(size=14)
plt.yticks(size=14)
plt.grid(True, linestyle='--', linewidth=1.5)
# plt.ylim(1.6, 8.1)
ax = plt.gca()
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
ax.spines['top'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)
plt.tight_layout()
plt.show()


