import pandas as pd

from workload_inference.constants import DATA_DIR

FILEPATH = DATA_DIR / "experiments/experiment_test/ZZZZ/FlyingPractice/drone_data.csv"


def main():
    df = pd.read_csv(FILEPATH)
    # Get a frequency estimate using the timestamp column
    timestamps = df["timestamp"].drop_duplicates()
    time_diffs = timestamps.diff()
    print(time_diffs[time_diffs > 50])
    print(f"Length of data: {len(df)}")


if __name__ == "__main__":
    main()
