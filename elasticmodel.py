# The data set used in this example is from http://archive.ics.uci.edu/ml/datasets/Wine+Quality
# P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis.
# Modeling wine preferences by data mining from physicochemical properties. In Decision Support Systems, Elsevier, 47(4):547-553, 2009.

import os
import warnings
import sys
from google.cloud import storage
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
import dvc.api
import logging
from test import get_most_recent_git_tag
import gdrivefs
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

# os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'secrets/mlflowpoc-954453dc1b99.json'


client = storage.Client()
mlflow.set_tracking_uri("postgresql://postgres:mysecretpassword@127.0.0.1:5432/mlflow")


path = 'data/winequality-red.csv'
repo = 'https://github.com/sam2881/mlflowdemo.git'


data_url = dvc.api.get_url(
    path=path,
    )
print(data_url)

tag = get_most_recent_git_tag

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


if __name__ == "__main__":

    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Read the wine-quality csv file from the URL
    # csv_url = (
    #     "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    # )
    # try:
    #     data = pd.read_csv(csv_url, sep=";")
    # except Exception as e:
    #     logger.exception(
    #         "Unable to download training & test CSV, check your internet connection. Error: %s", e
    #     )
    data = pd.read_csv(data_url)
    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)

    # The predicted column is "quality" which is a scalar from [3, 9]
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]

    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

    with mlflow.start_run():
        mlflow.set_experiment(experiment_name="Elastic Net Model ")
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        lr.fit(train_x, train_y)

        predicted_qualities = lr.predict(test_x)

        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

        print("Elasticnet model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)

        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        mlflow.log_param('data_url', data_url)
        mlflow.log_param('tag', tag)
        mlflow.log_param('input_rows', data.shape[0])
        mlflow.log_param('input_cols', data.shape[1])

        print(train_x.columns)
        cols_x = pd.DataFrame(list(train_x.columns))
        cols_x.to_csv('features.csv', header=False, index=False)
        mlflow.log_artifact('features.csv')

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # Model registry does not work with file store
        if tracking_url_type_store != "file":

            # Register the model
            # There are other ways to use the Model Registry, which depends on the use case,
            # please refer to the doc for more information:
            # https://mlflow.org/docs/latest/model-registry.html#api-workflow
            mlflow.sklearn.log_model(lr, "model", registered_model_name="ElasticnetWineModel")
        else:
            mlflow.sklearn.log_model(lr, "model")

