import pandas as pd
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor


def create_kn_classifier(k, scaler):
    file = "dataset/" + "spritzguss.csv"
    data = pd.read_csv((file))
    data = data.iloc[:, :-1]

    data = scaler.transform(data)
    data = pd.DataFrame(data)

    X = data.iloc[:, 0:6] # ändern
    Y = data.iloc[:, 6:13]
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    knn = KNeighborsRegressor(n_neighbors=k, algorithm='brute')
    
    knn.fit(X_train, Y_train)
    
    y_pred = knn.predict(X_test)
    mse = mean_squared_error(Y_test, y_pred)
    r2 = r2_score(Y_test, y_pred)

    print(f'Mean Squared Error: {mse}')
    print(f'R-squared: {r2}')
    return knn