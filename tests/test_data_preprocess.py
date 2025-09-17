import pandas as pd
from ml.data import process_data
def test_process_data_shapes():
    df = pd.read_csv("data/census.csv")
    Xtr, ytr, enc, lb = process_data(
        df, categorical_features=[
            "workclass","education","marital-status","occupation",
            "relationship","race","sex","native-country"
        ],
        label="salary", training=True
    )
    assert Xtr.shape[0] == ytr.shape[0] > 0
    assert enc is not None and lb is not None
