# Copyright (c) 2024 Houssem Ben Braiek, Emilio Rivera-Landos, IVADO, SEMLA
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from bank_marketing.data.make_datasets import make_bank_marketing_dataframe
from bank_marketing.helpers.file_loaders import load_fsspec_locally_temp


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'db_name', type=str, choices=('start', 'update1', 'update2'), help='Which database to use'
    )
    # Argument for data directory
    parser.add_argument('--data-dir', type=str, default='../data', help='Path to data directory')
    parser.add_argument(
        '--sed',
        '--socio-economic-data',
        type=str,
        default='../data/external/socio_economic_indices_data.csv',
        help='Path to socio-economic data file',
    )

    args = parser.parse_args()

    if not Path(args.sed).exists():
        raise FileNotFoundError(f"File '{args.sed}' not found")
    if not Path(args.data_dir).exists():
        raise FileNotFoundError(f"Directory '{args.data_dir}' not found")

    return args


if __name__ == '__main__':
    ARGS = parse_args()
    db_file = f's3://bank-marketing/data/{ARGS.db_name}.db'
    temp_file = load_fsspec_locally_temp(db_file)
    output_file = Path(ARGS.data_dir) / 'raw' / 'extraction.csv'

    try:
        print('Creating raw dataset...')
        df = make_bank_marketing_dataframe(Path(temp_file.name), Path(ARGS.sed))
    except Exception as e:
        print(f'An error occurred: {e}', file=sys.stderr)
    else:
        print('Saving raw dataset...')
        df.to_csv(output_file, sep=';', index=False)
        print('Created raw dataset at:', output_file)
        print('Absolute path:', output_file.resolve())
    finally:
        temp_file.close()
