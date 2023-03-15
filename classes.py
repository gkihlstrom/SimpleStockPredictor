import os
import typing

import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler

os.chdir("C:\\Users\\Gustav\\Documents\\Python Scripts\\Webstep Case")


class DataFormatter:
    def create_data(self) -> typing.Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Creates a dataset that contains created features, and a Return target
        column. All continuous features are scales and shifted, and all
        categorical features are one-hot-encoded. 80% of the data is set to the
        train set and following 20% (except the first 30 rows) are  set to
        the test set.

        Returns
        -------
        pd.DataFrame
            The training set.
        pd.DataFrame
            The test set.

        """
        self.load_data()
        self.create_features()
        self.df = self.df.iloc[30:-1].reset_index(drop=True)
        self.util_cols = ["Date", "Return"]

        self.train_data = self.df[: int(len(self.df) * 0.8)]
        self.test_data = self.df[int(len(self.df) * 0.8) + 30 :].reset_index(drop=True)

        self.fit_transformer(self.train_data)
        self.train_data = self.transform(self.train_data)
        self.test_data = self.transform(self.test_data)

        return self.train_data, self.test_data

    def load_data(self) -> None:
        """
        Simply reads the csv file that the data is stored in.

        Returns
        -------
        None

        """
        df = pd.read_csv("microsoft_stocks.csv")
        df["Date"] = pd.to_datetime(df["Date"])
        self.df = df

    def create_features(self) -> None:
        """
        Creates features and adds these to the class variable 'df'. The
        features created are of the type differences, date/month, momentum,
        moving averages, and "current value" divided y moving average. The
        'Return', i.e. the target, column is also created by dividing the next
        day's opening wth today's closing.

        Returns
        -------
        None

        """

        def create_moving_average(
            feature_name: str, rolling_period_length: int
        ) -> None:
            new_col_name = feature_name + "_MA" + str(rolling_period_length)
            self.df[new_col_name] = (
                self.df[feature_name].rolling(rolling_period_length).mean()
            )
            self.cont_cols.append(new_col_name)

        def current_close_over_rolling(rolling_feature: str) -> None:
            new_col_name = "Close_Over_" + rolling_feature
            self.df[new_col_name] = self.df["Close"] / self.df[rolling_feature]
            self.cont_cols.append(new_col_name)

        self.df["Return"] = self.df["Open"].shift(-1) / self.df["Close"]

        self.df["PrevReturn"] = self.df["Return"].shift()
        self.df["HL_Delta"] = self.df["High"] - self.df["Low"]
        self.df["CO_Delta"] = self.df["Close"] - self.df["Open"]
        self.df["Momentum"] = self.df["Close"] * self.df["Volume"]
        self.df["PrevReturnMomentum"] = (
            self.df["PrevReturn"] * self.df["Volume"].shift()
        )
        self.df["PredictionWindowLength"] = (
            self.df["Date"].shift(-1) - self.df["Date"]
        ).dt.days

        self.df["DayOfMonth"] = self.df["Date"].dt.day
        self.df["DayOfWeek"] = self.df["Date"].dt.weekday
        self.df["MonthOfYear"] = self.df["Date"].dt.month
        self.df["WeekOfYear"] = self.df["Date"].dt.isocalendar().week

        self.cont_cols = [
            "Open",
            "Close",
            "High",
            "Low",
            "Volume",
            "PrevReturn",
            "HL_Delta",
            "CO_Delta",
            "Momentum",
            "PrevReturnMomentum",
            "PredictionWindowLength",
        ]
        self.cat_cols = ["DayOfMonth", "DayOfWeek", "MonthOfYear", "WeekOfYear"]

        for c in self.cont_cols.copy():
            if "MA" not in c:
                create_moving_average(c, 30)

        for c in self.cont_cols.copy():
            if "MA" in c:
                current_close_over_rolling(c)

    def fit_transformer(self, data: pd.DataFrame) -> None:
        """
        Fits two transformers to the data. A standard scaler is fit to the
        continuous features, and a one-hot encoder is fitted to the categorical
        features. The transformers are saved as class variables.

        Parameters
        ----------
        data : pd.DataFrame
            The dataset used to fit the transformers.

        Returns
        -------
        None

        """
        self.ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
        self.ohe.fit(data[self.cat_cols])

        self.ss = StandardScaler()
        self.ss.fit(data[self.cont_cols])

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Trasforms the input dataset according to the transformers fitted by
        running the 'fit_transformer' function.

        Parameters
        ----------
        data : pd.DataFrame
            Dataset to be transformed.

        Returns
        -------
        df : pd.DataFrame
            Tramsformed dataset.

        """

        df = data[self.util_cols].copy()

        cont_df = pd.DataFrame(self.ss.transform(data[self.cont_cols]))
        cont_df.columns = self.cont_cols

        cat_df = pd.DataFrame(self.ohe.transform(data[self.cat_cols]))
        cat_df.columns = self.ohe.get_feature_names_out()

        df = pd.concat([df, cont_df, cat_df], axis=1)

        return df


class BlockCV:
    def __init__(self, splits: int = 5, margin: int = 31) -> None:
        """
        Initiates the class instance and sets the splits and margin variable,
        to be used later when running cross validations.

        Parameters
        ----------
        splits : int, optional
            The number of splits to use in the cross validations. The default
            is 5.
        margin : int, optional
            The margin to be used between train and test sets in the corss
            validations. The default is 31.

        Returns
        -------
        None

        """
        self.splits = splits
        self.margin = margin

    def calculate_indexes(self, N: int) -> typing.Tuple[np.array, np.array]:
        """
        Calculates the start and end indices for each block to be used.

        Parameters
        ----------
        N : int
            Number of data observations.

        Returns
        -------
        start_idxs : list
            List of the start indicies for each block.
        end_idxs : list
            List of the end indicies for each block.

        """
        start_idxs = np.zeros(self.splits)
        end_idxs = np.zeros(self.splits)
        for i in range(self.splits):
            start_idxs[i] = i * int(N / self.splits)
            end_idxs[i] = (i + 1) * int(N / self.splits)
        end_idxs[-1] = N
        start_idxs = start_idxs.astype(int)
        end_idxs = end_idxs.astype(int)
        return start_idxs, end_idxs

    def CV(
        self, model: RegressorMixin, features: pd.DataFrame, target: pd.Series
    ) -> float:
        """
        Run a block cross-validation scheme on the provided model and data, and
        returns the average RMSE for each block.

        Parameters
        ----------
        model : RegressorMixin
            A sklearn model.
        features : pd.DataFrame
            The feature set.
        target : pd.Series
            The target.

        Returns
        -------
        float
            RMSE of the cross validation.

        """
        features = features.reset_index(drop=True)
        start_idxs, end_idxs = self.calculate_indexes(len(features))

        rmse = 0
        for i in range(self.splits):
            idx_shift = int((end_idxs[i] - start_idxs[i] - self.margin) * 0.8)
            train_end_idx = start_idxs[i] + idx_shift
            test_start_idx = start_idxs[i] + idx_shift + self.margin

            train_X = features.iloc[start_idxs[i] : train_end_idx]
            train_y = target.iloc[start_idxs[i] : train_end_idx]

            test_X = features.iloc[test_start_idx : end_idxs[i]]
            test_y = target.iloc[test_start_idx : end_idxs[i]]

            model.fit(train_X, train_y)
            pred = np.array(model.predict(test_X))
            rmse += np.sqrt(np.mean((pred - test_y.values) ** 2))
        return rmse / self.splits


from itertools import product

from sklearn.linear_model import ElasticNet, LinearRegression


class ModelFactory:
    def create_model(self) -> RegressorMixin:
        """
        Creates the dataset, selects the features to be used, runs a
        hyperparameter search (over a predefined set of parameter and
        parameter space), creates the final model trained on the training set,
        and returns this.

        Returns
        -------
        RegressorMixin
            The final model.

        """
        self.cv = BlockCV()
        self.data_formatter = DataFormatter()
        self.train, self.test = self.data_formatter.create_data()
        self.select_features()
        self.model = self.find_optimal_hyperparameters()
        return self.model

    def select_features(self) -> None:
        """
        Selects the 20 most important features. The importance of the features
        is determined by running a simple linear regression on each, and
        using the resulting RMSEs (by running block CV) to rank them.

        Returns
        -------
        None

        """
        lr = LinearRegression()
        features = []
        performances = []
        for c in self.train.columns:
            if c not in ["Return", "Date"]:
                performances.append(
                    self.cv.CV(lr, self.train[[c]], self.train["Return"])
                )
                features.append(c)
        single_feature_performances = (
            pd.DataFrame({"Feature": features, "Performance": performances})
            .sort_values("Performance")
            .reset_index(drop=True)
        )
        self.selected_features = single_feature_performances["Feature"].iloc[:20].values

    def find_optimal_hyperparameters(self) -> RegressorMixin:
        """
        Runs a hyperparameter search to fit an ElasticNet (sklearn), by
        alternating the alpha and l1_ratio hyperparameters. The search is
        conducted by trying all possible parameter combinations (of the
        predefined search space) by running a block CV on the model created by
        using the parameter pairs. The pair resulting in the lowest RMSE is
        used to create a model that is trained on the entire train set.

        Returns
        -------
        RegressorMixin
            A model trained on the entire train dataset, with the optimal
            hyperparameters.

        """
        alpha_ = [0.5, 1, 1.5]
        l1_ratio_ = [0, 0.25, 0.5, 0.75, 1]

        performances = [None for _ in range(len(alpha_) * len(l1_ratio_))]
        parameters = [None for _ in range(len(alpha_) * len(l1_ratio_))]

        for i, p in enumerate(product(alpha_, l1_ratio_)):
            model = ElasticNet(alpha=p[0], l1_ratio=p[1])
            parameters[i] = p
            performances[i] = self.cv.CV(
                model, self.train[self.selected_features], self.train["Return"]
            )

        opt_p = parameters[np.argmin(performances)]
        model = ElasticNet(alpha=opt_p[0], l1_ratio=opt_p[1])
        model.fit(self.train[self.selected_features], self.train["Return"])
        return model

    def oos_test(self) -> None:
        """
        Tests the out of sample performance of the model. This is done by
        predicting the return of every observation in the test set using the
        class variable model. If the model predicts that the Return will be
        higher than 1, we assume the investor is long in the stock. Conversely,
        if the model predict that the Return will be lower than 1, we assume
        the investor is short in the stock. The resulting portfolio return
        from using this strategy is printed to the console.

        Returns
        -------
        None

        """
        preds = self.model.predict(self.test[self.selected_features])
        prediction_df = pd.DataFrame(
            {"Prediction": preds, "Actual": self.test["Return"]}
        )
        prediction_df["Return"] = prediction_df["Actual"] - 1
        prediction_df["Long"] = 0
        prediction_df.loc[prediction_df["Prediction"] > 1, "Long"] = 1
        prediction_df.loc[prediction_df["Long"] == 0, "Return"] = -prediction_df.loc[
            prediction_df["Long"] == 0, "Return"
        ]
        prediction_df["Return"] = prediction_df["Return"] + 1
        print(
            "The generated return from following going long/short when the predicted stock price increase/decrease was {:.4f}%.".format(
                100 * (prediction_df["Return"].prod() - 1)
            )
        )
