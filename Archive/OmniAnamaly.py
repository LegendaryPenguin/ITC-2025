import pandas as pd

url = "https://raw.githubusercontent.com/numenta/NAB/master/data/artificialWithAnomaly/art_daily_jumpsup.csv"
df = pd.read_csv(url, parse_dates=['timestamp'])
