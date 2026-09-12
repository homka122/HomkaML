import pickle
import unittest
from unittest.mock import patch

import numpy as np

from simple_nn import (
    Adam,
    BatchNormLayer,
    CrossEntropyLoss,
    FullyConnectedLayer,
    Momentum,
    ReluLayer,
    SGD,
    Sequential,
    evaluate,
    fit,
    gradient_check_layer,
    gradient_check_model,
    train_epoch,
)


def make_model(batchnorm=False, seed=42):
    rng = np.random.default_rng(seed)
    layers = [FullyConnectedLayer(4, 6, rng=rng)]
    if batchnorm:
        layers.append(BatchNormLayer(6))
    layers.extend([ReluLayer(), FullyConnectedLayer(6, 3, rng=rng)])
    return Sequential(layers)


class NeuralNetworkTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(123)
        self.X = rng.normal(size=(17, 4))
        self.y = np.arange(17) % 3

    def test_cross_entropy_extreme_logits_and_gradients(self):
        logits = np.array([[0.0, -100.0], [1000.0, -1000.0], [-1000.0, -1000.0]])
        labels = np.array([1, 1, 0])
        loss = CrossEntropyLoss()
        self.assertAlmostEqual(loss.forward(logits, labels), (100 + 2000 + np.log(2)) / 3)
        analytic = loss.backward()
        for index in np.ndindex(logits.shape):
            plus, minus = logits.copy(), logits.copy()
            plus[index] += 1e-4
            minus[index] -= 1e-4
            numeric = (loss.forward(plus, labels) - loss.forward(minus, labels)) / 2e-4
            self.assertAlmostEqual(analytic[index], numeric, places=8)

    def test_layer_gradients(self):
        for seed in range(5):
            rng = np.random.default_rng(seed)
            bn = BatchNormLayer(4)
            bn.gamma[:] = rng.normal(size=4)
            bn.beta[:] = rng.normal(size=4)
            for layer, output_dim in [
                (FullyConnectedLayer(4, 3, rng), 3),
                (ReluLayer(), 4),
                (bn, 4),
            ]:
                with self.subTest(seed=seed, layer=type(layer).__name__):
                    checks = gradient_check_layer(
                        layer,
                        rng.normal(size=(6, 4)),
                        rng.normal(size=(6, output_dim)),
                        num_checks=20,
                        random_state=seed,
                    )
                    np.testing.assert_allclose(
                        [c["analytic"] for c in checks],
                        [c["numeric"] for c in checks],
                        rtol=1e-5,
                        atol=1e-8,
                    )

    def test_full_model_gradients_with_l2_and_batchnorm(self):
        for batchnorm in [False, True]:
            checks = gradient_check_model(
                make_model(batchnorm),
                CrossEntropyLoss(),
                self.X,
                self.y,
                l2=0.03,
                num_checks=30,
            )
            np.testing.assert_allclose(
                [c["analytic"] for c in checks],
                [c["numeric"] for c in checks],
                rtol=1e-4,
                atol=1e-8,
            )

    def test_gradient_check_preserves_parameters_statistics_and_caches(self):
        model, loss = make_model(True), CrossEntropyLoss()
        loss.forward(model.forward(self.X), self.y)
        model.backward(loss.backward())
        before = pickle.dumps((model, loss))
        gradient_check_model(model, loss, self.X[:8], self.y[:8], l2=0.01)
        self.assertEqual(pickle.dumps((model, loss)), before)

    def test_gradient_check_restores_state_after_exception(self):
        model, loss = make_model(True), CrossEntropyLoss()
        before = pickle.dumps((model, loss))
        original_forward = loss.forward
        calls = 0

        def fail_during_difference(*args):
            nonlocal calls
            calls += 1
            if calls == 3:
                raise RuntimeError("interrupted finite difference")
            return original_forward(*args)

        with patch.object(loss, "forward", side_effect=fail_during_difference):
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                gradient_check_model(model, loss, self.X, self.y)
        self.assertEqual(pickle.dumps((model, loss)), before)

    def test_every_update_is_checked_without_changing_training(self):
        for batchnorm in [False, True]:
            unchecked, checked = make_model(batchnorm), make_model(batchnorm)
            train_epoch(
                unchecked, CrossEntropyLoss(), SGD(0.01), self.X, self.y, batch_size=8, rng=7
            )
            result = train_epoch(
                checked,
                CrossEntropyLoss(),
                SGD(0.01),
                self.X,
                self.y,
                batch_size=8,
                rng=7,
                gradient_checks=2,
            )
            self.assertEqual(result["gradient_steps"], 3)
            self.assertEqual(result["gradient_checks"], 3 * 2 * len(checked.parameters()))
            self.assertLessEqual(result["gradient_max_error"], 1)
            for name, value in unchecked.state_dict().items():
                np.testing.assert_array_equal(value, checked.state_dict()[name])

    def test_incorrect_gradient_stops_before_optimizer_update(self):
        model = make_model()
        layer = model.layers[-1]
        backward = layer.backward

        def wrong_backward(grad):
            result = backward(grad)
            layer.db *= -1
            return result

        before = model.state_dict()
        with patch.object(layer, "backward", side_effect=wrong_backward):
            with self.assertRaisesRegex(AssertionError, "Gradient check failed"):
                train_epoch(
                    model,
                    CrossEntropyLoss(),
                    SGD(0.01),
                    self.X,
                    self.y,
                    gradient_checks=3,
                )
        for name, value in before.items():
            np.testing.assert_array_equal(value, model.state_dict()[name])

    def test_fit_restores_best_epoch_including_batchnorm_statistics(self):
        model = make_model(True)
        snapshots = []
        scores = [0.4, 0.8, 0.5]
        losses = [1.2, 0.7, 1.0]
        calls = 0

        def controlled_evaluation(model, *args, **kwargs):
            nonlocal calls
            epoch = calls // 2
            calls += 1
            if calls % 2 == 0:
                snapshots.append(model.state_dict())
            return {"accuracy": scores[epoch], "loss": losses[epoch]}

        with patch("simple_nn.evaluate", side_effect=controlled_evaluation):
            history = fit(
                model,
                Adam(0.01),
                self.X,
                self.y,
                self.X,
                self.y,
                epochs=3,
                batch_size=8,
            )
        self.assertEqual(history[1]["val_accuracy"], 0.8)
        self.assertFalse(np.array_equal(snapshots[1]["layer0.W"], snapshots[2]["layer0.W"]))
        for name, value in snapshots[1].items():
            np.testing.assert_array_equal(value, model.state_dict()[name])

    def test_optimizers_against_closed_form_updates(self):
        gradients = [np.array([0.2, -0.4]), np.array([-0.8, 0.1]), np.array([0.5, 0.6])]
        for optimizer in [SGD(0.03), Momentum(0.03), Adam(0.03)]:
            actual = np.array([1.0, -2.0])
            expected = actual.copy()
            for t, grad in enumerate(gradients, 1):
                optimizer.step([("p", actual, grad)])
                if isinstance(optimizer, SGD):
                    update = grad
                elif isinstance(optimizer, Momentum):
                    update = sum(0.9 ** (t - 1 - i) * gradients[i] for i in range(t))
                else:
                    m = sum(0.1 * 0.9 ** (t - 1 - i) * gradients[i] for i in range(t))
                    v = sum(0.001 * 0.999 ** (t - 1 - i) * gradients[i] ** 2 for i in range(t))
                    update = (m / (1 - 0.9**t)) / (np.sqrt(v / (1 - 0.999**t)) + 1e-8)
                expected -= 0.03 * update
                np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    def test_small_dataset_can_be_memorized(self):
        for batchnorm in [False, True]:
            model = make_model(batchnorm)
            optimizer = Adam(0.03)
            for epoch in range(300):
                train_epoch(model, CrossEntropyLoss(), optimizer, self.X, self.y, rng=epoch)
            metrics = evaluate(model, CrossEntropyLoss(), self.X, self.y)
            self.assertEqual(metrics["accuracy"], 1)


if __name__ == "__main__":
    unittest.main()
