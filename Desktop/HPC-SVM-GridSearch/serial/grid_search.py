import time
from svm import SVM

def grid_search(X_train, y_train, X_test, y_test, C_values, gamma_values):
    best_acc = 0
    best_params = None
    start = time.time()

    for C in C_values:
        for gamma in gamma_values:
            model = SVM(C=C, gamma=gamma)
            model.train(X_train, y_train)
            acc = model.accuracy(X_test, y_test)
            if acc > best_acc:
                best_acc = acc
                best_params = (C, gamma)

    end = time.time()
    duration = end - start

    with open("results/serial_results.txt", "w") as f:
        f.write(f"Best Params: C={best_params[0]}, gamma={best_params[1]}\n")
        f.write(f"Accuracy: {best_acc:.4f}\n")
        f.write(f"Time: {duration:.2f} seconds\n")

    return best_params, best_acc, duration
