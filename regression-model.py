from readfromurl import *
fetch_housing_data()
housing=load_housing_data()
housing.head()
housing.info()
housing.describe()


train_set, test_set = split_train_test(housing, 0.2)

housing_with_id = housing.reset_index()   # adds an `index` column
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")

housing_with_id["id"] = housing["longitude"] * 1000 + housing["latitude"]
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "id")

from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)


housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
housing["income_cat"].where(housing["income_cat"] < 5, 8.0, inplace=True)


from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]


housing = strat_train_set.copy()
housing_test=strat_test_set.copy()
y_test=housing_test.iloc[:, 8].values


y_train=housing.iloc[:, 8].values

housing=housing.drop("median_house_value", axis=1)
housing_test=housing_test.drop("median_house_value", axis=1)

X = housing.iloc[:, 0:10].values
X_test=housing_test.iloc[:, 0:10].values

housing_tr = pd.DataFrame(X, columns=housing.columns)


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:, 8] = labelencoder.fit_transform(X[:, 8])

labelencoder_test = LabelEncoder()
X_test[:, 8] = labelencoder.fit_transform(X_test[:, 8])

housing_tr2 = pd.DataFrame(X, columns=housing.columns)
K=X[:, 9]
housing_tr1= pd.DataFrame(K)


from sklearn.preprocessing import Imputer

imputer = Imputer(strategy="median")

imputer.fit(X)
X = imputer.transform(X)


imputer_test = Imputer(strategy="median")

imputer.fit(X_test)
X_test = imputer.transform(X_test)

onehotencoder = OneHotEncoder(categorical_features = [8])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

onehotencoder2 = OneHotEncoder(categorical_features = [8])
X_test= onehotencoder.fit_transform(X_test).toarray()
X_test = X_test[:, 1:]

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X)
housing_tr = pd.DataFrame(X_train)

sc_X_test = StandardScaler()
X_test = sc_X.fit_transform(X_test)


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)


y_pred = regressor.predict(X_test)


from sklearn.metrics import mean_squared_error

lin_mse = mean_squared_error(y_test, y_pred)
lin_rmse = np.sqrt(lin_mse)
lin_rmse


from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor()
tree_reg.fit(X_train, y_train)

dt_y_pred = tree_reg.predict(X_test)

lin_mse = mean_squared_error(y_test, dt_y_pred)
lin_rmse = np.sqrt(lin_mse)
lin_rmse


from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor()
forest_reg.fit(X_train, y_train)

rf_y_pred = forest_reg.predict(X_test)

lin_mse = mean_squared_error(y_test, rf_y_pred)
lin_rmse = np.sqrt(lin_mse)
lin_rmse



from sklearn.model_selection import GridSearchCV

param_grid = [
    {'n_estimators': [30, 50, 100], 'max_features': [4, 6, 8, 10]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
  ]

forest_reg = RandomForestRegressor()

grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error')

grid_search.fit(X_train, y_train)

grid_search.best_params_


forest_reg= RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
           max_features=10, max_leaf_nodes=None, min_impurity_decrease=0.0,
           min_impurity_split=None, min_samples_leaf=1,
           min_samples_split=2, min_weight_fraction_leaf=0.0,
           n_estimators=100, n_jobs=1, oob_score=False, random_state=None,
           verbose=0, warm_start=False)


forest_reg.fit(X_train, y_train)

rf_y_pred = forest_reg.predict(X_test)

lin_mse = mean_squared_error(y_test, rf_y_pred)
lin_rmse = np.sqrt(lin_mse)
lin_rmse



from sklearn.svm import SVR
svr_regressor = SVR(kernel = 'rbf', C=100, gamma=100)
svr_regressor.fit(X_train, y_train)

svr_y_pred = svr_regressor.predict(X_test)

lin_mse = mean_squared_error(y_test, svr_y_pred)
lin_rmse = np.sqrt(lin_mse)
lin_rmse






from sklearn.model_selection import RandomizedSearchCV

param_grid =  {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]}

  

forest_reg = RandomForestRegressor()

randomized_search = RandomizedSearchCV(forest_reg, param_grid, cv=5, n_jobs=1)

randomized_search.fit(X_train, y_train)

grid_search.best_params_




from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_distribs = {
        'n_estimators': randint(low=1, high=200),
        'max_features': randint(low=1, high=8),
    }

forest_reg = RandomForestRegressor(random_state=42)
rnd_search = RandomizedSearchCV(forest_reg, param_distributions=param_distribs,
                                n_iter=10, cv=5, scoring='neg_mean_squared_error', random_state=42)
rnd_search.fit(X_train, y_train)


k = 5

grid_search.best_estimator_.feature_importances_

from sklearn.pipeline import Pipeline

cat_attribs = ["ocean_proximity"]

cat_pipeline = Pipeline([
        ('selector', DataFrameSelector(cat_attribs)),
        ('cat_encoder', CategoricalEncoder(encoding="onehot-dense")),
    ])

cat_encoder = CategoricalEncoder()
housing_cat_reshaped = housing_cat.values.reshape(-1, 1)
housing_cat_1hot = cat_encoder.fit_transform(housing_cat_reshaped)
housing_cat_1hot

extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]
cat_encoder = cat_pipeline.named_steps["cat_encoder"]
cat_one_hot_attribs = list(cat_encoder.categories_[0])
attributes = num_attribs + extra_attribs + cat_one_hot_attribs
sorted(zip(feature_importances, attributes), reverse=True)


top_k_feature_indices = indices_of_top_k(feature_importances, k)
top_k_feature_indices