import lightgbm as lgb

def train_lgbm(X_train, y_train, X_valid, y_valid, config):
    model = lgb.LGBMClassifier(
        num_leaves=config["num_leaves"],
        max_depth=config["max_depth"],
        learning_rate=config["learning_rate"],
        n_estimators=config["n_estimators"],
        class_weight=config["class_weight"],
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="auc",
        early_stopping_rounds=50,
        verbose=50,
    )

    return model
