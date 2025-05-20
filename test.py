import pandas as pd
df = pd.read_csv("predictions_epoch1.tsv", sep="\t")
df[df["Target"] == df["Predicted"]]
