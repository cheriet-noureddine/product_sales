import time
from joblib import Parallel, delayed
from svm import SVM

def evaluate_params(C, gamma, X_train, y_train, X_test, y_test):
    model = SVM(C=C, gamma=gamma)
    model.train(X_train, y_train)
    acc = model.accuracy(X_test, y_test)
    return (C, gamma, acc)

def grid_search_parallel(X_train, y_train, X_test, y_test, C_values, gamma_values, n_jobs=-1):
    start = time.time()

    results = Parallel(n_jobs=n_jobs)(
        delayed(evaluate_params)(C, gamma, X_train, y_train, X_test, y_test)
        for C in C_values for gamma in gamma_values
    )

    best = max(results, key=lambda x: x[2])
    end = time.time()
    duration = end - start

    with open("results/parallel_results.txt", "w") as f:
        f.write(f"Best Params: C={best[0]}, gamma={best[1]}\n")
        f.write(f"Accuracy: {best[2]:.4f}\n")
        f.write(f"Time: {duration:.2f} seconds\n")

    return best, duration
