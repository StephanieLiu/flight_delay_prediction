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

import os
import shutil
import json
import uuid

def truncated_uuid4():
    return str(uuid.uuid4())[:8]

def create_temporary_dir_if_not_exists(tmp_dir_path:os.PathLike='tmp') -> None:
    """creation of a temporary folder 

    Args:
        tmp_dir_path (os.PathLike, optional): Path of the folder. Defaults to 'tmp'.
    """
    if not os.path.exists(tmp_dir_path):
        os.makedirs(tmp_dir_path)
    return tmp_dir_path

def clean_temporary_dir(tmp_dir_path:os.PathLike='tmp') -> None:
    """delete the temporary folder

    Args:
        tmp_dir_path (os.PathLike, optional): Path of the folder. Defaults to 'tmp'.
    """
    if os.path.exists(tmp_dir_path):
        shutil.rmtree(tmp_dir_path)

def cameltosnake(camel_string: str) -> str:
    # If the input string is empty, return an empty string
    if not camel_string:
        return ""
    # If the first character of the input string is uppercase,
    # add an underscore before it and make it lowercase
    elif camel_string[0].isupper():
        return f"_{camel_string[0].lower()}{cameltosnake(camel_string[1:])}"
    # If the first character of the input string is lowercase,
    # simply return it and call the function recursively on the remaining string
    else:
        return f"{camel_string[0]}{cameltosnake(camel_string[1:])}"
 
def camel_to_snake(s: str) -> str:
    if len(s)<=1:
        return s.lower()
    # Changing the first character of the input string to lowercase
    # and calling the recursive function on the modified string
    return cameltosnake(s[0].lower()+s[1:])

def load_json(fpath):
    # JSON file
    with open(fpath, "r") as f:
        # Reading from file
        data = json.loads(f.read())
    return data