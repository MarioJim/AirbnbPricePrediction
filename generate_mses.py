import pathlib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor

df_base = pd.read_csv('airbnb_clean.csv').set_index("id")
df_ammenities = pd.read_csv('airbnb_amenities_clean.csv').set_index("id")
df = pd.concat([df_base, df_ammenities], axis=1)

X = df.drop(columns=["log_price", "host_since", "host_has_profile_pic", "city",
                     "host_identity_verified", "property_type", "room_type"])
Y = df["log_price"]

X["cleaning_fee"] = X["cleaning_fee"].astype(int)
X["instant_bookable"] = X["instant_bookable"].astype(int)
X[df_ammenities.columns] = X[df_ammenities.columns].astype(int)

rf_mses = []
gb_mses = []
lr_mses = []
kn_mses = []
vr_mses = []

for i in range(30):
    print(f"Calculating scores for iteration {i+1}")
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, random_state=i)

    random_forest = RandomForestRegressor(n_estimators=100, n_jobs=-1)
    random_forest.fit(X_train, y_train)
    rf_pred = random_forest.predict(X_test)
    rf_mses.append(mean_squared_error(y_test, rf_pred))

    gradient_boosting = GradientBoostingRegressor(n_estimators=150)
    gradient_boosting.fit(X_train, y_train)
    gb_pred = gradient_boosting.predict(X_test)
    gb_mses.append(mean_squared_error(y_test, gb_pred))

    linear_reg = LinearRegression(fit_intercept=True)
    linear_reg.fit(X_train, y_train)
    lr_pred = linear_reg.predict(X_test)
    lr_mses.append(mean_squared_error(y_test, lr_pred))

    k_neighbors = KNeighborsRegressor(n_neighbors=30, n_jobs=-1)
    k_neighbors.fit(X_train, y_train)
    kn_pred = k_neighbors.predict(X_test)
    kn_mses.append(mean_squared_error(y_test, kn_pred))

    voting_reg = VotingRegressor(
        [("RF", random_forest), ("GB", gradient_boosting), ("KN", k_neighbors), ("LN", linear_reg)], n_jobs=-1)
    voting_reg.fit(X_train, y_train)
    vr_pred = voting_reg.predict(X_test)
    vr_mses.append(mean_squared_error(y_test, vr_pred))

    print(
        f"{rf_mses[i]:2.5f} {gb_pred[i]:2.5f} {lr_pred[i]:2.5f} {vr_pred[i]:2.5f}")


pathlib.Path('mses').mkdir(parents=True, exist_ok=True)
np.savetxt("mses/RandomForestRegressor.csv", rf_mses)
np.savetxt("mses/GradientBoostingRegressor.csv", gb_mses)
np.savetxt("mses/LinearRegression.csv", lr_mses)
np.savetxt("mses/KNeighborsRegressor.csv", kn_mses)
np.savetxt("mses/VotingRegressor.csv", vr_mses)
