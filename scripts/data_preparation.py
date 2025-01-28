import argparse
import sys
from pathlib import Path

import pandas as pd

from bank_marketing.data.prep_datasets import Dataset, prepare_binary_classfication_tabular_data

person_info_cols_cat = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'comm_type']
person_info_cols_num = ['age']
num_cols_wo_customer = [
    'curr_n_contact',
    'days_since_last_campaign',
    'last_n_contact',
    'emp.var.rate',
    'cons.price.idx',
    'cons.conf.idx',
    'euribor3m',
    'nr.employed',
]
cat_cols_wo_customer = ['comm_month', 'comm_day', 'last_outcome']
numerical_cols = person_info_cols_num + num_cols_wo_customer
categorical_cols = person_info_cols_cat + cat_cols_wo_customer
predictors = (
    person_info_cols_cat + person_info_cols_num + num_cols_wo_customer + cat_cols_wo_customer
)

PREDICTED = 'curr_outcome'


def split_data(
    raw_df: pd.DataFrame,
    train_size: float = 0.7,
    valid_size: float = 0.2,
    test_size: float = 0.1,
    seed: int = 42,
) -> Dataset:
    dataset = prepare_binary_classfication_tabular_data(
        raw_df,
        predictors,
        PREDICTED,
        pos_neg_pair=('yes', 'no'),
        splits_sizes=[train_size, valid_size, test_size],
        seed=seed,
    )

    return dataset


def persist_dataset_locally(dataset: Dataset, data_dir: Path):
    dataset.train_x.to_csv(data_dir / 'splits' / 'train_x.csv', sep=';', index=False)
    dataset.train_y.to_csv(data_dir / 'splits' / 'train_y.csv', sep=';', index=False)
    dataset.val_x.to_csv(data_dir / 'splits' / 'val_x.csv', sep=';', index=False)
    dataset.val_y.to_csv(data_dir / 'splits' / 'val_y.csv', sep=';', index=False)
    dataset.test_x.to_csv(data_dir / 'splits' / 'test_x.csv', sep=';', index=False)
    dataset.test_y.to_csv(data_dir / 'splits' / 'test_y.csv', sep=';', index=False)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='../data', help='Path to data directory')
    parser.add_argument('--train-size', type=float, default=0.7, help='Train size')
    parser.add_argument('--valid-size', type=float, default=0.2, help='Validation size')
    parser.add_argument('--test-size', type=float, default=0.1, help='Test size')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    if not Path(args.data_dir).exists():
        raise FileNotFoundError(f"Directory '{args.data_dir}' not found")

    return args


if __name__ == '__main__':
    ARGS = parse_args()

    print('Chosen arguments:')
    print(ARGS)

    DATA_DIR = Path(ARGS.data_dir)

    try:
        print('Loading raw data...')
        RAW_DF = pd.read_csv(DATA_DIR / 'raw' / 'extraction.csv', sep=';')
        # Or get from dvc using get_extraction_url_from_dvc(), using also fsspec util
        # RAW_DF_DVC = load_fsspec_locally_temp(get_extraction_url_from_dvc())
        print('Splitting data...')
        DATASET = split_data(
            RAW_DF, train_size=ARGS.train_size, valid_size=ARGS.valid_size, test_size=ARGS.test_size
        )
    except Exception as e:
        print(f'An error occurred: {e}', file=sys.stderr)
    else:
        print('Saving data...')
        persist_dataset_locally(DATASET, DATA_DIR)
        print('Data saved under:', DATA_DIR / 'splits')
        print('Absolute path:', (DATA_DIR / 'splits').resolve())

