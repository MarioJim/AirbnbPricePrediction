import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

df_base = pd.read_csv('airbnb_clean.csv').set_index("id")
df_ammenities = pd.read_csv('airbnb_amenities_clean.csv').set_index("id")
df = pd.concat([df_base, df_ammenities], axis=1)

X = df.drop(columns=["log_price", "host_since", "host_has_profile_pic", "city",
                     "host_identity_verified", "property_type", "room_type"])
y = df['log_price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1)

random_forest = RandomForestRegressor(n_estimators=100, random_state=1)
random_forest.fit(X_train, y_train)
rf_pred = random_forest.predict(X_test)
rf_score = r2_score(y_test, rf_pred)
rf_mse = mean_squared_error(y_test, rf_pred)
print(f"Random forest     : R2 {rf_score:.5f}  MSE {rf_mse:2.5f}")

gradient_boosting = GradientBoostingRegressor(n_estimators=150, random_state=1)
gradient_boosting.fit(X_train, y_train)
gb_pred = gradient_boosting.predict(X_test)
gb_score = r2_score(y_test, gb_pred)
gb_mse = mean_squared_error(y_test, gb_pred)
print(f"Gradient boosting : R2 {gb_score:.5f}  MSE {gb_mse:2.5f}")

linear_reg = LinearRegression(fit_intercept=True)
linear_reg.fit(X_train, y_train)
lr_pred = linear_reg.predict(X_test)
lr_score = r2_score(y_test, lr_pred)
lr_mse = mean_squared_error(y_test, lr_pred)
print(f"Linear regression : R2 {lr_score:.5f}  MSE {lr_mse:2.5f}")

k_neighbors = KNeighborsRegressor(n_neighbors=30)
k_neighbors.fit(X_train, y_train)
kn_pred = k_neighbors.predict(X_test)
kn_score = r2_score(y_test, kn_pred)
kn_mse = mean_squared_error(y_test, kn_pred)
print(f"K Neighbors       : R2 {kn_score:.5f}  MSE {kn_mse:2.5f}")

voting_reg = VotingRegressor(
    [("RF", random_forest), ("GB", gradient_boosting), ("KN", k_neighbors), ("LN", linear_reg)])
voting_reg.fit(X_train, y_train)
vr_pred = voting_reg.predict(X_test)
vr_score = r2_score(y_test, vr_pred)
vr_mse = mean_squared_error(y_test, vr_pred)
print(f"Voting regressor  : R2 {vr_score:.5f}  MSE {vr_mse:2.5f}")

rf_pred_col = rf_pred.reshape(-1, 1)
gb_pred_col = gb_pred.reshape(-1, 1)
kn_pred_col = kn_pred.reshape(-1, 1)
lr_pred_col = lr_pred.reshape(-1, 1)
vr_pred_col = vr_pred.reshape(-1, 1)
y_test_col = y_test.values.reshape(-1, 1)

plt.figure()
plt.plot(y_test_col[0:50, :], "gd", label="Valores reales")
plt.plot(rf_pred_col[0:50, :], "xb", label="Random Forest Regressor")
plt.plot(gb_pred_col[0:50, :], "xr", label="Gradient Boosting Regressor")
plt.plot(lr_pred_col[0:50, :], "xy", label="Linear Regressor")
plt.plot(kn_pred_col[0:50, :], "xm", label="K-Neighbors Regressor")
plt.plot(vr_pred_col[0:50, :], "xc", label="Voting Regressor")

plt.tick_params(axis='x', which="both", bottom=True,
                top=False, labelbottom=True)
plt.ylabel("Logaritmo del precio")
plt.xlabel("Índice de la propiedad")
plt.legend(loc="best")
plt.title("Predicciones vs valores reales")

plt.show()
