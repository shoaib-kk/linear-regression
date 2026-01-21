import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 

class LinearRegression:
    def __init__(self, alpha=0.0, fit_intercept=True, normalise=False, random_state=42):
        
        self.coefficients = None
        self.intercept = None
        
        self.alpha = alpha  # L2 regularization strength
        self.fit_intercept = fit_intercept
        self.normalise = normalise
        self.random_state = random_state
        
        # Stats for normalization - computed during training
        self.feature_mean_ = None
        self.feature_std_ = None
        
        self.loss_history_ = []  # track convergence

    def fit(self, X, y, learning_rate = 0.01, n_iterations = 1000):
        
        # numpy operations are faster, so convert pandas objects
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            X = X.values
        if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
            y = y.values
        
        n_samples, n_features = X.shape
        
        # prevents features with larger scales from dominating gradient
        if self.normalise:
            self.feature_mean_ = np.mean(X, axis=0)
            self.feature_std_ = np.std(X, axis=0)
            X = (X - self.feature_mean_) / self.feature_std_
        
        # include intercept in matrix multiplication instead of separate addition
        if self.fit_intercept:
            X = np.concatenate([np.ones((n_samples, 1)), X], axis=1)
            n_features += 1
                
        # random init breaks symmetry, helps gradient descent explore better
        rng = np.random.default_rng(self.random_state)
        self.coefficients = rng.normal(loc=0.0, scale=1.0, size=n_features)
        
        for iteration in range(n_iterations):

            # Compute how far off our predictions are
            predictions = X.dot(self.coefficients)
            errors = predictions - y
            
            # Compute gradient
            gradient = (2/n_samples) * X.T.dot(errors) + 2 * self.alpha * self.coefficients
            
            # Update coefficients using gradient
            self.coefficients -= learning_rate * gradient
            
            # useful for debugging convergence issues
            loss = (1/n_samples) * np.sum(errors ** 2) + self.alpha * np.sum(self.coefficients ** 2)
            self.loss_history_.append(loss)
        
        # separate storage makes predict() cleaner and matches sklearn API
        if self.fit_intercept:
            self.intercept = self.coefficients[0]
            self.coefficients = self.coefficients[1:]
        else:
            self.intercept = 0
    def predict(self, X):

        if self.coefficients is None:
            raise ValueError("Please fit model before making predictions")
        
        # need same data type as fit() to avoid broadcasting errors
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            X = X.values
        
        n_samples = X.shape[0]
        
        # must match the same scaling used during training
        if self.normalise:
            X = (X - self.feature_mean_) / self.feature_std_
        
        # reconstruct same matrix structure as training
        if self.fit_intercept:
            X = np.concatenate([np.ones((n_samples, 1)), X], axis=1)
            coefficients = np.concatenate([[self.intercept], self.coefficients])
        else:
            coefficients = self.coefficients
        
        return X.dot(coefficients)
    
def sample_data(sample_size = 1000, random_state = 42, std_dev = 1.0):

    rng = np.random.default_rng(random_state)

    X = rng.uniform(0, 10, size=(sample_size, 1))

    # noise makes it more realistic for testing gradient descent
    noise = rng.normal(0, std_dev, size=(sample_size, 1))

    y = 3 * X + 7 + noise

    return X, y.ravel()

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Splits data into training and testing sets.
    Make sure to always split the data before fitting the model
    parameters:
        -param X: Features
        -param y: Target variable
        -param test_size: Proportion of data to be used as test set
        -param random_state: Seed for random number generator for reproducibility
    
    return: X_train, X_test, y_train, y_test
    """
    rng = np.random.default_rng(random_state)
    no_samples = X.shape[0]
    indices = np.arange(no_samples)
    rng.shuffle(indices)

    split_index = int(no_samples * test_size)
    test_index = indices[:split_index]
    train_index = indices[split_index:]

    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    return X_train, X_test, y_train, y_test


def visualise_data(X, y, overlay_regression=False, model=None):
    figure, axes = plt.subplots()

    axes.scatter(X, y, color='red', label='Data points')
    axes.set_xlabel('X')
    axes.set_ylabel('y')
    axes.set_title('Scatter plot of data')
    axes.text(0, 1.05, "Press any key to close the figure", fontsize=8, verticalalignment='bottom')
    if overlay_regression:
        axes.plot(X, model.predict(X), color='black', label='Regression line')

    axes.legend()

    # Close figure when a key is pressed
    def on_key(event):
        plt.close(event.canvas.figure)

    figure.canvas.mpl_connect("key_press_event", on_key)

    plt.show()
    
#def visualise_loss()

def calculate_r2_score(y_true, y_pred):
    # R² = 1 - (SS_residual / SS_total)
    # Represents how much of the total variance in the target variable is explained by the model

    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    r2_score = 1 - (ss_residual / ss_total)
    return r2_score

 