import pathlib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

df_base = pd.read_csv('airbnb_clean.csv').set_index("id")
df_ammenities = pd.read_csv('airbnb_amenities_clean.csv').set_index("id")
df = pd.concat([df_base, df_ammenities], axis=1)

Y = df['log_price']
df.drop(columns=['log_price', 'Washer', 'latitude', 'host_since',
                 'host_has_profile_pic', 'Wireless_Internet',
                 'host_identity_verified'],
        inplace=True)
X = df

dict_column_dict = {}
column_len_unique_value = []
for column in df.columns:
    column_len_unique_value.append(len(df[column].unique()))
    column_dictionary = {df.columns[i]: column_len_unique_value[i]
                         for i in range(0, len(column_len_unique_value))}
dict_column_dict = (column_dictionary)


categorical_col = []
for key, value in dict_column_dict.items():
    if((value >= 2 and value < 700) and (key != 'bathrooms' and key != 'review_scores_rating')):
        categorical_col.append(key)

for i in range(len(categorical_col)):
    categorical_col[i] = df.columns.get_loc(categorical_col[i])

columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(
    handle_unknown="ignore"), categorical_col)], remainder='passthrough')

rf_mses = []
gb_mses = []
lr_mses = []
kn_mses = []
vr_mses = []


for i in range(30):
    print(f"Calculating scores for iteration {i+1}")
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, random_state=i)
    X_train = columnTransformer.fit_transform(X_train)
    X_test = columnTransformer.transform(X_test)

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

    voting_reg = VotingRegressor([("RF", random_forest), ("GB", gradient_boosting),
                                  ("KN", k_neighbors), ("LN", linear_reg)], n_jobs=-1)
    voting_reg.fit(X_train, y_train)
    vr_pred = voting_reg.predict(X_test)
    vr_mses.append(mean_squared_error(y_test, vr_pred))

    print(f"{rf_mses[i]:2.5f} {gb_mses[i]:2.5f}", end=" ")
    print(f"{lr_mses[i]:2.5f} {kn_mses[i]:2.5f} {vr_mses[i]:2.5f}")


pathlib.Path('mses').mkdir(parents=True, exist_ok=True)
np.savetxt("mses/RandomForestRegressor.csv", rf_mses)
np.savetxt("mses/GradientBoostingRegressor.csv", gb_mses)
np.savetxt("mses/LinearRegression.csv", lr_mses)
np.savetxt("mses/KNeighborsRegressor.csv", kn_mses)
np.savetxt("mses/VotingRegressor.csv", vr_mses)
