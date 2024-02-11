import json
import csv
import os
import pandas as pd


def handle_json():
    module_dir = os.path.dirname(__file__)
    file_path = os.path.join(module_dir, 'stock_market_data.json')
    with open(file_path, 'r') as file:
        data = json.load(file)

    file.close()
    return data


def handle_csv():
    module_dir = os.path.dirname(__file__)
    file_path = os.path.join(module_dir, 'stock_market_data.csv')
    with open(file_path, 'r') as file:
        csvreader = csv.reader(file)
        header = next(csvreader)

        rows = []
        for row in csvreader:
            # print(row)
            rows.append(row)

    return rows


def handle_with_pandas():
    module_dir = os.path.dirname(__file__)
    file_path = os.path.join(module_dir, 'stock_market_data.csv')
    df = pd.read_csv(file_path, index_col=False)

    return df
    # print(df['date'])


if __name__ == '__main__':
    # handle_json()
    # handle_csv()
    handle_with_pandas()
