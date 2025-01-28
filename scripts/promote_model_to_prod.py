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
import logging
import argparse
from dotenv import load_dotenv
load_dotenv()
MLFLOW_URI = os.getenv("MLFLOW_URI")
import bank_marketing.mlflow.registry as registry

logger = logging.getLogger('scripts.promote_model_to_prod')

if __name__ == "__main__":
    # Initialize parser
    parser = argparse.ArgumentParser()
    # Adding arguments
    parser.add_argument("-m", "--model_name", 
                        default="camp-accept-predictor",
                        help="MLFlow Experiment Name")
    # Read arguments from command line
    args = parser.parse_args()
    
    #promote the model named model_name at staging to {model_name}-production at production
    model_version = registry.promote_model_to_production(MLFLOW_URI, args.model_name, f"{args.model_name}-production")
    
    #set the above added {model_name}-production at production to active 
    registry.set_active_production_model(MLFLOW_URI,  f"{args.model_name}-production", model_version.version)


    
    