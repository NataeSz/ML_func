import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from scipy import stats

def back_propagation(X, y):
    lm = LogisticRegression()
    lm.fit(X,y)
    params = np.append(lm.intercept_,lm.coef_)
    predictions = lm.predict(X)

    X_with_intercept = np.append(np.ones((len(X),1)), X, axis=1)
    MSE = (sum((y-predictions)**2))/(len(X_with_intercept)-len(X_with_intercept[0]))

    var_b = MSE*(np.linalg.inv(np.dot(X_with_intercept.T,X_with_intercept)).diagonal())
    sd_b = np.sqrt(var_b) # Standard Errors
    ts_b = params/ sd_b # t values
    p_values =[2*(1-stats.t.cdf(np.abs(i),(len(X_with_intercept)-len(X_with_intercept[0])))) for i in ts_b]
    if any(x >= 0.1 for x in p_values[1:]):
        highest_p = pd.Series(p_values[1:], index=(X.columns.to_list())).sort_values(ascending = False).index[0]
        print('deleting', highest_p)
        return back_propagation(X.drop(highest_p, axis = 1), y) # recurrence
    else:
        return X.columns