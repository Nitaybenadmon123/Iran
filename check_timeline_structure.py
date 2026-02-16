import pandas as pd
from pathlib import Path

# Check timeline file structure
timeline_file = Path("Data/Users_Timelines/20rapcom_timeline.csv")
df = pd.read_csv(timeline_file)

print(f"Columns: {list(df.columns)}")
print(f"\nFirst 3 rows:\n{df.head(3)}")
print(f"\nData types:\n{df.dtypes}")
