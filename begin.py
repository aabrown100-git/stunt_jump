import numpy as np
import pandas as pd
temp=np.array([[0, 2.5],[1, 1],[2, 0.25],[3, 0.25],[4, 1]])
df = pd.DataFrame(temp)
df.to_excel("nodelocs.xlsx", index=False)
