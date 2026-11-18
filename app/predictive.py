"""Two actual scikit-learn estimators trained on explicitly synthetic history."""
from __future__ import annotations
import numpy as np
from functools import lru_cache
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score

FEATURES=['size_kg','product_complexity','equipment_health','ambient_humidity','previous_line_load','operator_skill']

def batch_features(batch):
    return [batch.kilograms, batch.complexity, batch.equipment_health,
            batch.humidity, batch.prior_load, batch.operator_skill]

@lru_cache(maxsize=1)
def train_models():
    rng=np.random.default_rng(20261118)
    n=2600
    size=rng.uniform(210,430,n); complexity=rng.uniform(.90,1.35,n)
    health=rng.uniform(.54,1.0,n); humidity=rng.uniform(38,75,n)
    load=rng.uniform(.55,1.60,n); skill=rng.uniform(.50,1.0,n)
    X=np.column_stack([size,complexity,health,humidity,load,skill])
    # Ground-truth generator is a deliberately artificial nonlinear plant relationship.
    factor=(.59 + .00058*size + .30*(complexity-.85)
            + .42*(1-health) + .0034*np.maximum(0,humidity-49)
            + .15*np.maximum(0,load-.85) - .14*(skill-.5)
            + .10*(health < .70)*(load > 1.13)
            + rng.normal(0,.035,n))
    y=np.maximum(.65,factor)
    probability=1/(1+np.exp(-(-3.6 + 7.5*(1-health)+2.3*np.maximum(0,load-1)
                        + .038*np.maximum(0,humidity-55) + 1.2*(complexity-1))))
    failure=rng.binomial(1,probability)
    idx_train,idx_test=train_test_split(np.arange(n),test_size=.22,random_state=71,stratify=failure)
    model=RandomForestRegressor(n_estimators=95,min_samples_leaf=3,n_jobs=1,random_state=17)
    classifier=RandomForestClassifier(n_estimators=110,min_samples_leaf=4,n_jobs=1,random_state=29)
    model.fit(X[idx_train],y[idx_train]);classifier.fit(X[idx_train],failure[idx_train])
    predictions=model.predict(X[idx_test]); risk=classifier.predict_proba(X[idx_test])[:,1]
    metrics={'training_records':int(len(idx_train)), 'test_records':int(len(idx_test)),
             'duration_mae':round(float(mean_absolute_error(y[idx_test],predictions)),4),
             'duration_r2':round(float(r2_score(y[idx_test],predictions)),3),
             'downtime_auc':round(float(roc_auc_score(failure[idx_test],risk)),3),
             'data_type':'SYNTHETIC','features':FEATURES,
             'feature_importance':{FEATURES[i]:round(float(model.feature_importances_[i]),3) for i in range(len(FEATURES))},
             'estimator':'RandomForestRegressor + RandomForestClassifier'}
    return model,classifier,metrics

def predict_batches(batches):
    model, classifier, metrics=train_models()
    X=np.array([batch_features(b) for b in batches],dtype=float)
    mult=np.clip(model.predict(X),.65,1.6)
    risk=classifier.predict_proba(X)[:,1]
    return {b.id:{'duration_multiplier':float(round(mult[i],3)),
                  'downtime_probability':float(round(risk[i],3)),
                  'risk_level':'high' if risk[i]>=.55 else ('medium' if risk[i]>=.30 else 'low')}
            for i,b in enumerate(batches)},metrics
