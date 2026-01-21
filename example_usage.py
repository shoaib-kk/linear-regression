import linear_regression as lr
import numpy as np

def incorrect_usage():

    X, y = lr.sample_data(sample_size=50, random_state=42, std_dev=5.0)

    model = lr.LinearRegression(fit_intercept=True,normalise=True,)

    model.fit(X, y)

    X_train, X_test, y_train, y_test = lr.split_data(X, y, test_size=0.2)

    predictions = model.predict(X_test)

    lr.visualise_data(X_test, y_test, overlay_regression=True, model=model)

    r2 = lr.calculate_r2_score(y_test, predictions)
    print(f"Equation : y = {model.coefficients[0]} * x + {model.intercept}")
    print(f"R² score: {r2}\n")



def correct_usage():

    X, y = lr.sample_data(sample_size=50, random_state=42, std_dev=5.0)

    X_train, X_test, y_train, y_test = lr.split_data(X, y, test_size=0.2)

    model = lr.LinearRegression(fit_intercept=True,normalise=True,)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    lr.visualise_data(X_test, y_test, overlay_regression=True, model=model)

    r2 = lr.calculate_r2_score(y_test, predictions)
    
    print(f"Equation : y = {model.coefficients[0]} * x + {model.intercept}")
    print(f"R² score: {r2}\n")


def main():
    print("===================================================")
    print("Running incorrect usage example:")
    print("===================================================\n")
    incorrect_usage()
    
    print("===================================================")
    print("Running correct usage example:")
    print("===================================================\n")
    correct_usage()

    print("===================================================\n")
    print("Try spot the difference in the 2 use cases above")
    print("===================================================\n")
if __name__ == "__main__":
    main()