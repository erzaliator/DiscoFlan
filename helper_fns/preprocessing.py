import pandas as pd
from datasets import Dataset

def read_df_custom(file):
    header = 'doc     unit1_toks      unit2_toks      unit1_txt       unit2_txt       s1_toks s2_toks unit1_sent      unit2_sent      dir     orig_label      label'
    extracted_columns = ['unit1_txt', 'unit1_sent', 'unit2_txt', 'unit2_sent', 'dir', 'label']
    header = header.split()
    df = pd.DataFrame(columns=extracted_columns)
    file = open(file, 'r')

    rows = []
    count = 0
    for line in file:
        if line=='\n': continue
        line = line[:-1].split('\t')
        count+=1
        if count ==1: continue
        row = {}
        for column in extracted_columns:
            index = header.index(column)
            row[column] = line[index]
        rows.append(row)

    df = pd.concat([df, pd.DataFrame.from_records(rows)])
    return df


def construct_dataset(train_path, test_path, valid_path, logger):
    train_dataset = Dataset.from_pandas(read_df_custom(train_path))
    test_dataset = Dataset.from_pandas(read_df_custom(test_path))
    valid_dataset = Dataset.from_pandas(read_df_custom(valid_path))

    return train_dataset, test_dataset, valid_dataset