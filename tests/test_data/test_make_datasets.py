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
import pandas as pd
import pytest

from bank_marketing.data.make_datasets import merge_defaults


@pytest.mark.parametrize(
    ('default_x', 'default_y', 'expected'),
    [
        ('yes', 'yes', 'yes'),
        ('yes', 'no', 'yes'),
        ('yes', 'unknown', 'yes'),
        ('no', 'yes', 'yes'),
        ('no', 'no', 'no'),
        ('no', 'unknown', 'unknown'),
        ('unknown', 'yes', 'yes'),
        ('unknown', 'no', 'unknown'),
        ('unknown', 'unknown', 'unknown'),
    ],
)
def test_merge_default(default_x, default_y, expected):
    df = pd.DataFrame({'default_x': [default_x], 'default_y': [default_y]})
    assert merge_defaults(df.iloc[0]) == expected
