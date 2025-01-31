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

from dotenv import load_dotenv

load_dotenv()
import pandas as pd
from flask import Flask, request
from flask_restful import Api, Resource
from joblib import load

app = Flask(__name__)
api = Api(app)

MODEL_PATH = os.environ.get('MODEL_PATH')
# Load model using the binary file
model = load(MODEL_PATH)


# Function to test if the request contains multiple
def islist(obj):
    return 'list' in str(type(obj))


class Preds(Resource):
    def post(self):
        """Make a prediction with our ML model."""
        json_ = request.json
        # If there are multiple records to be predicted,
        # directly convert the request json file into a pandas dataframe
        if all(islist(json_[feature]) for feature in json_):
            entry = pd.DataFrame(json_)
        # In the case of a single record to be predicted,
        # convert json request data into a list and then to a pandas dataframe
        else:
            entry = pd.DataFrame([json_])
        # Make predictions using data
        prediction = model.predict(entry)
        res = {'prediction': prediction.tolist()}
        return res, 200  # Send the response object


api.add_resource(Preds, '/predict')

if __name__ == '__main__':
    # Dev. note: Use FLASK_DEBUG environment variable if you want to debug
    app.run()
