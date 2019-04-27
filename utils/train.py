from predict import predict
from cost import cost

def fit_with_cost(x, y, theta, alpha, num_iters):
    m = len(y)
    cost_history= []
    for index in range(num_iters):
        som = 0
        for i in range(m):
            som += (predict(x[i], theta) - y[i])
        t0 = theta[0] - (alpha / m) * som
        som = 0
        for i in range(m):
            som += ((predict(x[i], theta) - y[i]) * x[i])
        t1 = theta[1] - (alpha / m) * som
        theta[0] = t0
        theta[1] = t1
        _cost = cost(x, y, theta)
        print("Cost after iteration ", index, ": ", _cost[0], end="\r")
        cost_history.extend(_cost)

    print("Cost after iteration ", num_iters, ": ", cost(x, y, theta)[0], end="\n\n")
    return theta, cost_history