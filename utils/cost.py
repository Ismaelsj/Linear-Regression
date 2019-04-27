from predict import predict

def cost(x, y, theta):
    m = len(y)
    som = 0
    for i in range(m):
        som += (predict(x[i], theta) - y[i])**2
    return (som / (2 * m))