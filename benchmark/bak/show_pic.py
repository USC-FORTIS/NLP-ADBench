from pyod.models.iforest import IForest
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt


def IForest_benchmark_pca(X_train, X_test, y_train, y_test, IForest_Hyerparameters):
    # Normalize the data
    scaler = StandardScaler()
    X_train_norm = scaler.fit_transform(X_train)
    X_test_norm = scaler.transform(X_test)

    # Reduce the data into two principal components for visualization
    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train_norm)
    X_test_pca = pca.transform(X_test_norm)

    # Train IForest detector
    clf_name = 'IForest'
    clf = IForest()  # Add hyperparameters if necessary
    clf.fit(X_train_pca)

    # Get the prediction on the training data
    y_train_pred = clf.predict(X_train_pca)
    y_train_scores = clf.decision_function(X_train_pca)

    # Get the prediction on the test data
    y_test_pred = clf.predict(X_test_pca)
    y_test_scores = clf.decision_function(X_test_pca)

    # Evaluate and print the results
    print("\nOn Training Data:")
    evaluate_print(clf_name, y_train, y_train_scores)
    print("\nOn Test Data:")
    evaluate_print(clf_name, y_test, y_test_scores)

    # Visualization
    plt.figure(figsize=(10, 7))

    # Generate grid for plotting
    xx, yy = np.meshgrid(np.linspace(X_train_pca[:, 0].min(), X_train_pca[:, 0].max(), 100), 
                         np.linspace(X_train_pca[:, 1].min(), X_train_pca[:, 1].max(), 100))

    # Decision function on the grid points
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot decision boundary and data points
    plt.contourf(xx, yy, Z, cmap=plt.cm.Blues_r)

    # True inliers
    plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c='white', s=20, edgecolor='k', label='true inliers')
    
    # True outliers
    plt.scatter(X_train_pca[y_train_pred == 1, 0], X_train_pca[y_train_pred == 1, 1], c='black', s=20, edgecolor='k', label='true outliers')
    
    errors = (y_test_pred != y_test).sum()
    plt.title(f'{clf_name} (errors: {errors})')

    plt.legend()
    plt.show()
    

# Assuming X_train, X_test, y_train, y_test are defined and properly preprocessed
IForest_Hyerparameters = None
IForest_benchmark_pca(X_train, X_test, y_train, y_test, IForest_Hyerparameters)
