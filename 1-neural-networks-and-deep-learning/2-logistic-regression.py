import numpy as np
import numpy.typing as nt


def propagate(
      w: nt.NDArray[np.float64], 
      b: float, 
      x: nt.NDArray[np.float64], 
      y: nt.NDArray[np.float64]
) -> tuple[
      nt.NDArray[np.float64], 
      float,
      float
]:
      m = y.shape[1]
      z = np.dot(w.T, x) + b
      a = 1/(1 + np.exp(-z))
      cost = -(1/m) * np.sum(y * np.log(a) + (1 - y) * np.log(1 - a))  # type: ignore
      dw = (1/m) * np.dot(x, (a - y).T)
      db = (1/m) * np.sum(a - y)  # type: ignore
      return dw, db, cost


def optimize(
      w: nt.NDArray[np.float64], 
      b: float, 
      x: nt.NDArray[np.float64], 
      y: nt.NDArray[np.float64], 
      alpha: float, 
      iterations: int
) -> tuple[
      nt.NDArray[np.float64], 
      float, 
      nt.NDArray[np.float64]
]:
      costs = np.zeros((iterations, ))
      for i in range(iterations):
            dw, db, cost = propagate(w, b, x, y)
            w -= alpha * dw
            b -= alpha * db
            costs[i] = cost
      return w, b, costs


def predict(
      w: nt.NDArray[np.float64], 
      b: float, 
      x: nt.NDArray[np.float64]
) -> nt.NDArray[np.float64]:
      z = np.dot(w.T, x) + b
      y_pred = 1/(1 + np.exp(-z))
      y_pred[y_pred>0.5] = 1
      y_pred[y_pred<=0.5] = 0
      return y_pred


if __name__ == '__main__':
      x = np.array([[1, 2, 3], [4, 5, 6]])
      y = np.array([[1, 0, 1]])
      n = x.shape[0]
      w = np.zeros((n, 1))
      b = 0
      alpha = 0.01
      iterations = 1000
      w, b, costs = optimize(w, b, x, y, alpha, iterations)
      log = "\n".join([f"Iteration {i}: cost = {cost}" for i, cost in enumerate(costs)])
      print(log)
      print(f"w: {w}; b: {b}")
      y_pred = predict(w, b, x)
      print(y_pred)
      