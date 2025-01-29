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

import fsspec
import tempfile
import yaml

def load_fsspec_locally_temp(file_url: str, binary: bool = True, temp_dir: str="") -> tempfile._TemporaryFileWrapper:
    """Downloads an object from file_url locally to a temporary directory

    It is up to the caller to delete file.

    Args:
        file_url: Location of the file to load with fsspec
        binary: Whether to read and write a binary file

    Environment:
        Depends on the environment variable SCRATCH_DIR which, if set,
        will be used as the directory for the temporary file.

    Returns:
        The file handle
    """
    in_mode = 'rb' if binary else 'r'
    out_mode = 'wb' if binary else 'w'
    # Implementation detail: This is necessary if we want to guarantee compatibility across:
    #                        Linux and Windows (inside and outside docker/kubernetes too)
    output_file = tempfile.NamedTemporaryFile(dir=temp_dir or tempfile.gettempdir(), mode=out_mode, delete=False)

    with fsspec.open(file_url, mode=in_mode) as f:
        output_file.write(f.read())

    output_file.close()

    return output_file

def load_yaml_file(file_path):
    """
    Load and parse a YAML file.

    Args:
        file_path (str): The path to the YAML file.

    Returns:
        dict: The parsed YAML content as a dictionary.

    Raises:
        FileNotFoundError: If the file does not exist.
        yaml.YAMLError: If there is an error parsing the YAML content.
    """
    try:
        with open(file_path, 'r') as file:
            return yaml.safe_load(file)
    except FileNotFoundError as e:
        print(f"Error: The file '{file_path}' was not found.")
        raise e
    except yaml.YAMLError as e:
        print(f"Error: Failed to parse YAML file '{file_path}'.")
        raise e