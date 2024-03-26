import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("results_mlp.csv", index_col=0)
x = data.index.values
for i, column in enumerate(data.values.T):
    plt.plot(x, column, label=data.columns.values[i])
plt.legend()
plt.xlabel("threshold")
plt.ylabel("F1 score")
plt.savefig("f1_scores_mlp.pdf")
plt.show()
