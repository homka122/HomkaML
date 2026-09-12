import numpy as np


class FullyConnectedLayer:
    def __init__(self, input_dim, output_dim, rng=None, weight_scale=None):
        rng = np.random.default_rng(rng)
        if weight_scale is None:
            weight_scale = np.sqrt(2.0 / input_dim)

        self.W = rng.normal(0.0, weight_scale, size=(input_dim, output_dim))
        self.b = np.zeros(output_dim)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def forward(self, X, training=True):
        self.X = X
        return X @ self.W + self.b

    def backward(self, grad_output):
        self.dW = self.X.T @ grad_output
        self.db = grad_output.sum(axis=0)
        return grad_output @ self.W.T

    def parameters(self):
        return [("W", self.W, self.dW), ("b", self.b, self.db)]


class ReluLayer:
    def forward(self, X, training=True):
        self.mask = X > 0
        return X * self.mask

    def backward(self, grad_output):
        return grad_output * self.mask

    def parameters(self):
        return []


class BatchNormLayer:
    def __init__(self, dim, eps=1e-5, momentum=0.9):
        self.eps = eps
        self.momentum = momentum
        self.gamma = np.ones(dim)
        self.beta = np.zeros(dim)
        self.dgamma = np.zeros_like(self.gamma)
        self.dbeta = np.zeros_like(self.beta)
        self.running_mean = np.zeros(dim)
        self.running_var = np.ones(dim)

    def forward(self, X, training=True, update_running=True):
        if training:
            mean = X.mean(axis=0)
            var = X.var(axis=0)
            centered = X - mean
            inv_std = 1.0 / np.sqrt(var + self.eps)
            X_hat = centered * inv_std

            if update_running:
                self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
                self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var

            self.cache = (centered, inv_std, X_hat)
        else:
            X_hat = (X - self.running_mean) / np.sqrt(self.running_var + self.eps)

        return self.gamma * X_hat + self.beta

    def backward(self, grad_output):
        centered, inv_std, X_hat = self.cache
        batch_size = grad_output.shape[0]

        self.dgamma = (grad_output * X_hat).sum(axis=0)
        self.dbeta = grad_output.sum(axis=0)

        grad_x_hat = grad_output * self.gamma
        grad_var = (grad_x_hat * centered * -0.5 * inv_std**3).sum(axis=0)
        grad_mean = (grad_x_hat * -inv_std).sum(axis=0)
        grad_mean += grad_var * (-2.0 * centered).mean(axis=0)

        return (
            grad_x_hat * inv_std + grad_var * 2.0 * centered / batch_size + grad_mean / batch_size
        )

    def parameters(self):
        return [("gamma", self.gamma, self.dgamma), ("beta", self.beta, self.dbeta)]


class CrossEntropyLoss:
    def forward(self, logits, y):
        y = y.astype(int)
        shifted_logits = logits - logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(shifted_logits)
        self.probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        self.y = y

        correct_probs = self.probs[np.arange(len(y)), y]
        return -np.log(correct_probs + 1e-12).mean()

    def backward(self):
        grad = self.probs.copy()
        grad[np.arange(len(self.y)), self.y] -= 1.0
        return grad / len(self.y)


class Sequential:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, X, training=True):
        output = X
        for layer in self.layers:
            output = layer.forward(output, training=training)
        return output

    def backward(self, grad_output):
        grad = grad_output
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def parameters(self):
        result = []
        for layer_id, layer in enumerate(self.layers):
            for name, param, grad in layer.parameters():
                result.append((f"layer{layer_id}.{name}", param, grad))
        return result

    def l2_loss(self, l2):
        loss = 0.0
        for layer in self.layers:
            if isinstance(layer, FullyConnectedLayer):
                loss += 0.5 * l2 * np.sum(layer.W**2)
        return loss

    def add_l2_gradients(self, l2):
        for layer in self.layers:
            if isinstance(layer, FullyConnectedLayer):
                layer.dW += l2 * layer.W

    def predict(self, X):
        return self.forward(X, training=False).argmax(axis=1)


class SGD:
    def __init__(self, learning_rate=1e-2):
        self.learning_rate = learning_rate

    def step(self, parameters):
        for _, param, grad in parameters:
            param -= self.learning_rate * grad


class Momentum:
    def __init__(self, learning_rate=1e-2, momentum=0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity = {}

    def step(self, parameters):
        for name, param, grad in parameters:
            velocity = self.velocity.get(name)
            if velocity is None:
                velocity = np.zeros_like(param)
            velocity = self.momentum * velocity - self.learning_rate * grad
            param += velocity
            self.velocity[name] = velocity


class Adam:
    def __init__(self, learning_rate=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.first_moment = {}
        self.second_moment = {}
        self.step_count = 0

    def step(self, parameters):
        self.step_count += 1
        for name, param, grad in parameters:
            m = self.first_moment.get(name, np.zeros_like(param))
            v = self.second_moment.get(name, np.zeros_like(param))

            m = self.beta1 * m + (1 - self.beta1) * grad
            v = self.beta2 * v + (1 - self.beta2) * grad**2

            m_hat = m / (1 - self.beta1**self.step_count)
            v_hat = v / (1 - self.beta2**self.step_count)
            param -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.eps)

            self.first_moment[name] = m
            self.second_moment[name] = v


def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


def minibatches(X, y, batch_size, rng, shuffle=True):
    indices = np.arange(len(X))
    if shuffle:
        rng.shuffle(indices)

    for start in range(0, len(indices), batch_size):
        batch_ids = indices[start : start + batch_size]
        yield X[batch_ids], y[batch_ids]


def train_epoch(model, loss_fn, optimizer, X, y, batch_size=128, l2=0.0, rng=None):
    rng = np.random.default_rng(rng)
    total_loss = 0.0
    total_correct = 0

    for X_batch, y_batch in minibatches(X, y, batch_size, rng, shuffle=True):
        logits = model.forward(X_batch, training=True)
        loss = loss_fn.forward(logits, y_batch) + model.l2_loss(l2)
        grad_logits = loss_fn.backward()
        model.backward(grad_logits)
        model.add_l2_gradients(l2)
        optimizer.step(model.parameters())

        total_loss += loss * len(X_batch)
        total_correct += (logits.argmax(axis=1) == y_batch).sum()

    return {
        "loss": total_loss / len(X),
        "accuracy": total_correct / len(X),
    }


def evaluate(model, loss_fn, X, y, batch_size=512, l2=0.0):
    total_loss = 0.0
    total_correct = 0
    rng = np.random.default_rng(0)

    for X_batch, y_batch in minibatches(X, y, batch_size, rng, shuffle=False):
        logits = model.forward(X_batch, training=False)
        loss = loss_fn.forward(logits, y_batch) + model.l2_loss(l2)
        total_loss += loss * len(X_batch)
        total_correct += (logits.argmax(axis=1) == y_batch).sum()

    return {
        "loss": total_loss / len(X),
        "accuracy": total_correct / len(X),
    }


def relative_error(analytic, numeric):
    denominator = np.maximum(1e-8, np.abs(analytic) + np.abs(numeric))
    return np.max(np.abs(analytic - numeric) / denominator)


def gradient_check_model(model, loss_fn, X, y, l2=0.0, eps=1e-5, num_checks=10, random_state=42):
    rng = np.random.default_rng(random_state)
    logits = model.forward(X, training=True)
    loss_fn.forward(logits, y)
    model.backward(loss_fn.backward())
    model.add_l2_gradients(l2)

    results = []
    for name, param, grad in model.parameters():
        for _ in range(num_checks):
            index = tuple(rng.integers(0, size) for size in param.shape)
            old_value = param[index]

            param[index] = old_value + eps
            plus = loss_fn.forward(model.forward(X, training=True), y) + model.l2_loss(l2)

            param[index] = old_value - eps
            minus = loss_fn.forward(model.forward(X, training=True), y) + model.l2_loss(l2)

            param[index] = old_value
            numeric_grad = (plus - minus) / (2 * eps)
            results.append(
                {
                    "parameter": name,
                    "index": index,
                    "analytic": grad[index],
                    "numeric": numeric_grad,
                    "relative_error": relative_error(grad[index], numeric_grad),
                }
            )

    return results


def gradient_check_layer(layer, X, grad_output, eps=1e-5, num_checks=10, random_state=42):
    rng = np.random.default_rng(random_state)

    def forward(input_value):
        if isinstance(layer, BatchNormLayer):
            return layer.forward(input_value, training=True, update_running=False)
        return layer.forward(input_value, training=True)

    forward(X)
    grad_input = layer.backward(grad_output)
    results = []

    for _ in range(num_checks):
        index = tuple(rng.integers(0, size) for size in X.shape)
        old_value = X[index]

        X[index] = old_value + eps
        plus = np.sum(forward(X) * grad_output)

        X[index] = old_value - eps
        minus = np.sum(forward(X) * grad_output)

        X[index] = old_value
        numeric_grad = (plus - minus) / (2 * eps)
        results.append(
            {
                "parameter": "input",
                "index": index,
                "analytic": grad_input[index],
                "numeric": numeric_grad,
                "relative_error": relative_error(grad_input[index], numeric_grad),
            }
        )

    for name, param, grad in layer.parameters():
        for _ in range(num_checks):
            index = tuple(rng.integers(0, size) for size in param.shape)
            old_value = param[index]

            param[index] = old_value + eps
            plus = np.sum(forward(X) * grad_output)

            param[index] = old_value - eps
            minus = np.sum(forward(X) * grad_output)

            param[index] = old_value
            numeric_grad = (plus - minus) / (2 * eps)
            results.append(
                {
                    "parameter": name,
                    "index": index,
                    "analytic": grad[index],
                    "numeric": numeric_grad,
                    "relative_error": relative_error(grad[index], numeric_grad),
                }
            )

    return results
