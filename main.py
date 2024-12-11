"""
参考元: https://takaito0423.hatenablog.com/entry/2024/10/23/221608
"""

import os
import gc
import warnings
import random
import numpy as np
import pandas as pd
from pathlib import Path
import pickle

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import lightgbm as lgb

import json
from logging import getLogger, config


with open("./log_config.json", "r") as f:
    log_conf = json.load(f)
config.dictConfig(log_conf)
logger = getLogger(__name__)

warnings.filterwarnings("ignore")


# ====================================================
# Configurations
# ====================================================
class CFG:
    VER = 1
    AUTHOR = "ns"
    COMPETITION = "PCDUA1"
    DATA_PATH = Path("./train/data")
    OOF_DATA_PATH = Path("./train/oof")
    MODEL_DATA_PATH = Path("./train/models")
    SUB_DATA_PATH = Path("./train/submission")
    METHOD_LIST = ["lightgbm"]
    seed = 42
    n_folds = 3
    target_col = "money_room"
    metric = "RMSE"
    metric_maximize_flag = False
    num_boost_round = 2500
    early_stopping_round = 200
    verbose = 100
    regression_lgb_params = {
        "objective": "regression",  # 'regression'
        "metric": "rmse",
        "learning_rate": 0.05,
        "seed": seed,
    }
    model_weight_dict = {"lightgbm": 1.00}


# ====================================================
# Seed everything
# ====================================================
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


logger.info("Start Seeding...")
seed_everything(CFG.seed)
logger.info("Seeding Done.")

# ====================================================
# Loading Data
# ====================================================
logger.info("Loading Data...")
train_df = (
    pd.read_csv(CFG.DATA_PATH / "train.csv")
    .sort_values("target_ym")
    .reset_index(drop=True)
)
test_df = pd.read_csv(CFG.DATA_PATH / "test.csv").drop("index", axis=1)
submit_df = pd.read_csv(CFG.DATA_PATH / "sample_submit.csv", header=None)

numerical_features = []
categorical_features = []
for col in train_df.columns[2:]:
    if train_df[col].dtype == object:
        logger.info(col)
    else:
        numerical_features.append(col)

default_numerical_features = numerical_features
features = default_numerical_features
logger.info("Done.")


def Preprocessing(input_df: pd.DataFrame) -> pd.DataFrame:
    # いろいろ特徴量作成を追加する
    output_df = input_df.copy()
    return output_df


def lightgbm_training(
    x_train: pd.DataFrame,
    y_train: pd.DataFrame,
    x_valid: pd.DataFrame,
    y_valid: pd.DataFrame,
):
    lgb_train = lgb.Dataset(x_train, y_train)
    lgb_valid = lgb.Dataset(x_valid, y_valid)

    model = lgb.train(
        params=CFG.regression_lgb_params,
        train_set=lgb_train,
        num_boost_round=CFG.num_boost_round,
        valid_sets=[lgb_train, lgb_valid],
        callbacks=[
            lgb.early_stopping(
                stopping_rounds=CFG.early_stopping_round, verbose=CFG.verbose
            ),
            lgb.log_evaluation(CFG.verbose),
        ],
    )
    # Predict validation
    valid_pred = model.predict(x_valid)
    return model, valid_pred


def gradient_boosting_model_cv_training(
    method: str, train_df: pd.DataFrame, features: list
):
    # Create a numpy array to store out of folds predictions
    oof_predictions = np.zeros(len(train_df))
    oof_fold = np.zeros(len(train_df))
    kfold = KFold(n_splits=CFG.n_folds)  # , shuffle=True, random_state=CFG.seed)
    for fold, (train_index, valid_index) in enumerate(kfold.split(train_df)):
        logger.info("-" * 50)
        logger.info(f"{method} training fold {fold+1}")

        x_train = train_df[features].iloc[train_index]
        y_train = train_df[CFG.target_col].iloc[train_index]
        x_valid = train_df[features].iloc[valid_index]
        y_valid = train_df[CFG.target_col].iloc[valid_index]
        if method == "lightgbm":
            model, valid_pred = lightgbm_training(x_train, y_train, x_valid, y_valid)

        # Save best model
        pickle.dump(
            model,
            open(
                CFG.MODEL_DATA_PATH
                / f"{method}_fold{fold + 1}_seed{CFG.seed}_ver{CFG.VER}.pkl",
                "wb",
            ),
        )
        # Add to out of folds array
        oof_predictions[valid_index] = valid_pred
        oof_fold[valid_index] = fold + 1
        del x_train, x_valid, y_train, y_valid, model, valid_pred
        gc.collect()

    # Compute out of folds metric
    score = np.sqrt(mean_squared_error(train_df[CFG.target_col], oof_predictions))
    logger.info(f"{method} our out of folds CV rmse is {score}")
    # Create a dataframe to store out of folds predictions
    oof_df = pd.DataFrame(
        {
            CFG.target_col: train_df[CFG.target_col],
            f"{method}_prediction": oof_predictions,
            "fold": oof_fold,
        }
    )
    oof_df.to_csv(
        CFG.OOF_DATA_PATH / f"oof_{method}_seed{CFG.seed}_ver{CFG.VER}.csv", index=False
    )


def Learning(input_df: pd.DataFrame, features: list):
    for method in CFG.METHOD_LIST:
        gradient_boosting_model_cv_training(method, input_df, features)


def lightgbm_inference(x_test: pd.DataFrame):
    test_pred = np.zeros(len(x_test))
    for fold in range(CFG.n_folds):
        model = pickle.load(
            open(
                CFG.MODEL_DATA_PATH
                / f"lightgbm_fold{fold + 1}_seed{CFG.seed}_ver{CFG.VER}.pkl",
                "rb",
            )
        )
        # Predict
        pred = model.predict(x_test)
        test_pred += pred
    return test_pred / CFG.n_folds


def gradient_boosting_model_inference(
    method: str, test_df: pd.DataFrame, features: list
):
    x_test = test_df[features]
    if method == "lightgbm":
        test_pred = lightgbm_inference(x_test)
    return test_pred


def Predicting(input_df: pd.DataFrame, features: list):
    output_df = input_df.copy()
    output_df["pred"] = 0
    for method in CFG.METHOD_LIST:
        output_df[f"{method}_pred"] = gradient_boosting_model_inference(
            method, input_df, features
        )
        output_df["pred"] += CFG.model_weight_dict[method] * output_df[f"{method}_pred"]
    return output_df


logger.info("Start Preprocessing...")
train_df = Preprocessing(train_df)
test_df = Preprocessing(test_df)
logger.info("Done.")

logger.info("Start Learning...")
Learning(train_df, features)
logger.info("Done.")

logger.info("Start Predicting...")
test_df = Predicting(test_df, features)
logger.info("Done.")

submit_df[1] = test_df["pred"]
submit_df.to_csv(
    CFG.SUB_DATA_PATH / f"seed{CFG.seed}_ver{CFG.VER}_{CFG.AUTHOR}_submission.csv",
    header=False,
    index=False,
)
