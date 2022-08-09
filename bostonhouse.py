import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import metrics
import mlflow
import os
from urllib.parse import urlparse
import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)




if __name__ == "__main__":
    # enable autologging
    mlflow.set_experiment(experiment_name="Reg Price")

    # Load data
    boston = datasets.load_boston()
    df = pd.DataFrame(boston.data, columns=boston.feature_names)
    df['MEDV'] = boston.target

    # Split Model
    X = df.drop(['MEDV'], axis=1)
    y = df['MEDV']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)

    mlflow.set_tracking_uri("postgresql://postgres:mysecretpassword@127.0.0.1:5432/mlflow")

    with mlflow.start_run():
        # Model Creation
        lm = LinearRegression()
        lm.fit(X_train, y_train)

        # Model prediction
        Y_Pred = lm.predict(X_test)
        RMSE = np.sqrt(metrics.mean_squared_error(y_test, Y_Pred))
        print('RMSE: ', RMSE)

        mlflow.log_metric("RMSE", RMSE)

        print(X_train.columns)
        cols_x = pd.DataFrame(list(X_train.columns))
        cols_x.to_csv('features.csv', header=False, index=False)
        mlflow.log_artifact('features.csv')

        # Model registry does not work with file store
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        if tracking_url_type_store != "file":

            # Register the model
            # There are other ways to use the Model Registry, which depends on the use case,
            # please refer to the doc for more information:
            # https://mlflow.org/docs/latest/model-registry.html#api-workflow
            mlflow.sklearn.log_model(lm, "model", registered_model_name="Boston House model")
        else:
            mlflow.sklearn.log_model(lm, "model")


