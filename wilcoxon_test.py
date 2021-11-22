from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import ranksums

mses_dict = {}

for mses in Path('mses').iterdir():
    if mses.is_file():
        mses_dict[mses.stem] = np.loadtxt(mses)

if len(mses_dict) == 0:
    print("No MSEs files found")
    print("Run `generate_mses.py` file first")

matrix_greater = []
matrix_less = []

for mses1 in mses_dict.values():
    row_greater = []
    row_less = []

    for mses2 in mses_dict.values():
        row_greater.append(
            ranksums(mses1, mses2, alternative="greater").pvalue)
        row_less.append(ranksums(mses1, mses2, alternative="less").pvalue)

    matrix_greater.append(row_greater)
    matrix_less.append(row_less)

df_greater = pd.DataFrame(
    matrix_greater, index=mses_dict.keys(), columns=mses_dict.keys())
df_less = pd.DataFrame(
    matrix_less, index=mses_dict.keys(), columns=mses_dict.keys())

pd.options.display.float_format = "{:.14f}".format
print("Wilcoxon rank sums test with a greater hypothesis")
print(df_greater)
print()

print("Wilcoxon rank sums test with a less hypothesis")
print(df_less)
