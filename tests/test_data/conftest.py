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
import pytest

from bank_marketing.data.make_datasets import make_bank_marketing_dataframe


def pytest_addoption(parser):
    parser.addoption('--db', action='store')
    parser.addoption('--sed', action='store')


@pytest.fixture(scope='module')
def dataframe(request):
    if request.config.option.db is None or request.config.option.sed is None:
        raise pytest.UsageError(
            'Please provide the path to the database and socio-economic data files.'
        )
    df = make_bank_marketing_dataframe(request.config.option.db, request.config.option.sed)
    return df


@pytest.fixture(scope='module')
def predictors():
    predictors = [
        'age',
        'job',
        'marital',
        'education',
        'comm_month',
        'comm_day',
        'comm_type',
        'curr_n_contact',
        'days_since_last_campaign',
        'last_n_contact',
        'last_outcome',
        'emp.var.rate',
        'cons.price.idx',
        'cons.conf.idx',
        'euribor3m',
        'nr.employed',
        'housing',
        'loan',
        'default',
    ]
    return predictors


@pytest.fixture(scope='module')
def predicted():
    return 'curr_outcome'
