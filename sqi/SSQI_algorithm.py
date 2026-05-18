import numpy as np
import pandas as pd
import glob


def Ssqi( reading_array):
    mu = reading_array.mean()
    stddev = reading_array.std(ddof=0)
    return (np.mean(((reading_array - mu)/stddev) ** 3))


if __name__ == "__main__":
    csv_files = glob.glob(r"D:\Study\Lab\fingerTests\*.csv")
    for file in csv_files:
        df = pd.read_csv(file)
        data_np = df.to_numpy()
        score = Ssqi(data_np[:,1])
        print(file, score)

