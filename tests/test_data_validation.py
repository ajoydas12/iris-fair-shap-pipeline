import pandas as pd
import pytest

def test_iris_data_columns():
    df = pd.read_csv('data/iris.csv')
    expected_columns = {'sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species'}
    assert set(df.columns) == expected_columns, f"Columns mismatch: {df.columns}"

def test_iris_data_no_missing():
    df = pd.read_csv('data/iris.csv')
    assert not df.isnull().values.any(), "Data contains missing values!"


if __name__ == "__main__":
    test_iris_data_columns()
    test_iris_data_no_missing()
    print("Data validation tests passed successfully.")
