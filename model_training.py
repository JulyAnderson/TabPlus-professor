from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils import class_weight
import numpy as np


def train_model(X_train, y_train):
    # Calculate class weights for models
    weights = class_weight.compute_class_weight('balanced', classes=np.array([0, 1]), y=y_train)
    class_weights = {0: weights[0], 1: weights[1]}

    # Train models
    models = {
        "Random Forest": RandomForestClassifier(random_state=42, class_weight=class_weights),
        "Logistic Regression": LogisticRegression(max_iter=200, class_weight='balanced'),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42)
    }

    # Fit models
    for name, model in models.items():
        model.fit(X_train, y_train)

    return models, class_weights
