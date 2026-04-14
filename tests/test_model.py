import numpy as np
import pytest

from phitodeep import loss as ls
from phitodeep import model as m
from phitodeep.layers import activation as a
from phitodeep.layers import base as b

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_model():
    """Minimal 3-layer model with no training needed."""
    return m.Sequential(
        b.Dense(4, 8),
        a.ReLu(),
        b.Dense(8, 2),
        epochs=2,
        batch_size=4,
    )


@pytest.fixture
def small_data():
    """16-sample, 4-feature classification dataset with fixed seed."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((16, 4)).astype(np.float32)
    y = rng.integers(0, 2, 16)
    return X, y


@pytest.fixture
def trainable_model():
    """Small model + CCE loss suitable for end-to-end train() tests."""
    return m.Sequential(
        b.Dense(4, 8),
        a.ReLu(),
        b.Dense(8, 2),
        a.Softmax(),
        loss_class=ls.CategoricalCrossEntropy(),
        optimizer="adam",
        epochs=3,
        batch_size=4,
    )


# ---------------------------------------------------------------------------
# TestSequential
# ---------------------------------------------------------------------------


class TestSequential:
    # --- __init__ defaults ---

    def test_default_optimizer_is_adam(self):
        model = m.Sequential()
        assert model.optimizer == "adam"

    def test_default_alpha(self):
        model = m.Sequential()
        assert model.alpha == 0.01

    def test_default_batch_size(self):
        model = m.Sequential()
        assert model.batch_size == 1

    def test_default_epochs(self):
        model = m.Sequential()
        assert model.epochs == 1000

    def test_default_loss_is_mse(self):
        model = m.Sequential()
        assert isinstance(model.loss_class, ls.MeanSquaredError)

    def test_empty_model_has_no_layers(self):
        model = m.Sequential()
        assert model.layers == []

    def test_layers_stored_as_list(self, simple_model):
        assert isinstance(simple_model.layers, list)

    def test_layer_count_matches_constructor_args(self):
        model = m.Sequential(b.Dense(4, 8), a.ReLu(), b.Dense(8, 2))
        assert len(model.layers) == 3

    def test_layers_preserve_order(self):
        d1 = b.Dense(4, 8)
        r = a.ReLu()
        d2 = b.Dense(8, 2)
        model = m.Sequential(d1, r, d2)
        assert model.layers[0] is d1
        assert model.layers[1] is r
        assert model.layers[2] is d2

    # --- add ---

    def test_add_increases_layer_count(self, simple_model):
        before = len(simple_model.layers)
        simple_model.add(a.Sigmoid())
        assert len(simple_model.layers) == before + 1

    def test_add_appends_to_end(self, simple_model):
        sig = a.Sigmoid()
        simple_model.add(sig)
        assert simple_model.layers[-1] is sig

    def test_add_multiple_layers(self, simple_model):
        before = len(simple_model.layers)
        simple_model.add(a.ReLu())
        simple_model.add(b.Dense(2, 1))
        assert len(simple_model.layers) == before + 2

    # --- setoptimizer / setbatchsize / setloss ---

    def test_setoptimizer_updates_attribute(self, simple_model):
        simple_model.setoptimizer("sgd")
        assert simple_model.optimizer == "sgd"

    def test_setbatchsize_updates_attribute(self, simple_model):
        simple_model.setbatchsize(64)
        assert simple_model.batch_size == 64

    def test_setloss_updates_attribute(self, simple_model):
        cce = ls.CategoricalCrossEntropy()
        simple_model.setloss(cce)
        assert simple_model.loss_class is cce

    # --- predict ---

    def test_predict_output_shape_batch(self, simple_model):
        X = np.random.randn(8, 4).astype(np.float32)
        out = simple_model.predict(X)
        assert out.shape == (8, 2)

    def test_predict_output_shape_single_sample(self, simple_model):
        X = np.random.randn(1, 4).astype(np.float32)
        out = simple_model.predict(X)
        assert out.shape == (1, 2)

    def test_predict_returns_numpy_array(self, simple_model):
        X = np.random.randn(4, 4).astype(np.float32)
        out = simple_model.predict(X)
        assert isinstance(out, np.ndarray)

    def test_predict_is_deterministic_without_training(self, simple_model):
        X = np.random.randn(4, 4).astype(np.float32)
        out1 = simple_model.predict(X)
        out2 = simple_model.predict(X)
        np.testing.assert_array_equal(out1, out2)

    # --- __call__ ---

    def test_call_matches_predict(self, simple_model):
        X = np.random.randn(8, 4).astype(np.float32)
        np.testing.assert_array_equal(simple_model(X), simple_model.predict(X))

    # --- backward ---

    def test_backward_runs_without_error(self, simple_model):
        X = np.random.randn(8, 4).astype(np.float32)
        simple_model.predict(X)
        grad = np.random.randn(8, 2).astype(np.float32)
        simple_model.backward(grad)  # should not raise

    # --- train ---

    def test_train_returns_list(self, trainable_model, small_data):
        X, y = small_data
        losses = trainable_model.train(X, y, X, y)
        assert isinstance(losses, list)

    def test_train_loss_count_equals_epochs(self, trainable_model, small_data):
        X, y = small_data
        losses = trainable_model.train(X, y, X, y)
        assert len(losses) == trainable_model.epochs

    def test_train_each_entry_is_tuple_of_two(self, trainable_model, small_data):
        X, y = small_data
        losses = trainable_model.train(X, y, X, y)
        assert all(len(entry) == 2 for entry in losses)

    def test_train_losses_are_finite(self, trainable_model, small_data):
        X, y = small_data
        losses = trainable_model.train(X, y, X, y)
        for train_loss, test_loss in losses:
            assert np.isfinite(train_loss)
            assert np.isfinite(test_loss)

    @pytest.mark.parametrize("optimizer", ["sgd", "adam"])
    def test_train_works_with_both_optimizers(self, small_data, optimizer):
        X, y = small_data
        model = m.Sequential(
            b.Dense(4, 8),
            a.ReLu(),
            b.Dense(8, 2),
            a.Softmax(),
            loss_class=ls.CategoricalCrossEntropy(),
            optimizer=optimizer,
            epochs=2,
            batch_size=4,
        )
        losses = model.train(X, y, X, y)
        assert len(losses) == 2

    def test_train_raises_on_invalid_optimizer(self, trainable_model, small_data):
        X, y = small_data
        trainable_model.setoptimizer("invalid_opt")
        with pytest.raises(ValueError, match="invalid_opt"):
            trainable_model.train(X, y, X, y)

    # --- copy ---

    def test_copy_returns_sequential_instance(self, simple_model):
        assert isinstance(simple_model.copy(), m.Sequential)

    def test_copy_has_same_layer_count(self, simple_model):
        copy = simple_model.copy()
        assert len(copy.layers) == len(simple_model.layers)

    def test_copy_layers_are_different_objects(self, simple_model):
        copy = simple_model.copy()
        for orig, copied in zip(simple_model.layers, copy.layers):
            assert orig is not copied

    def test_copy_dense_weights_are_equal(self, simple_model):
        copy = simple_model.copy()
        for orig, copied in zip(simple_model.layers, copy.layers):
            if isinstance(orig, b.Dense):
                np.testing.assert_array_equal(copied.W, orig.W)
                np.testing.assert_array_equal(copied.b, orig.b)

    def test_copy_dense_weights_do_not_share_memory(self, simple_model):
        copy = simple_model.copy()
        for orig, copied in zip(simple_model.layers, copy.layers):
            if isinstance(orig, b.Dense):
                copied.W[0, 0] += 99.0
                assert orig.W[0, 0] != copied.W[0, 0]

    def test_copy_preserves_alpha(self, simple_model):
        copy = simple_model.copy()
        assert copy.alpha == simple_model.alpha

    def test_copy_preserves_optimizer(self, simple_model):
        copy = simple_model.copy()
        assert copy.optimizer == simple_model.optimizer

    def test_copy_preserves_batch_size(self, simple_model):
        copy = simple_model.copy()
        assert copy.batch_size == simple_model.batch_size

    def test_copy_preserves_epochs(self, simple_model):
        copy = simple_model.copy()
        assert copy.epochs == simple_model.epochs


# ---------------------------------------------------------------------------
# TestSequentialBuilder
# ---------------------------------------------------------------------------


class TestSequentialBuilder:
    # --- build ---

    def test_build_returns_sequential(self):
        assert isinstance(m.SequentialBuilder().build(), m.Sequential)

    def test_build_empty_has_no_layers(self):
        model = m.SequentialBuilder().build()
        assert len(model.layers) == 0

    # --- layer builder methods ---

    def test_flatten_adds_flatten_layer(self):
        model = m.SequentialBuilder().flatten().build()
        assert isinstance(model.layers[0], b.Flatten)

    def test_dense_adds_dense_layer(self):
        model = m.SequentialBuilder().dense(4, 8).build()
        assert isinstance(model.layers[0], b.Dense)

    def test_dense_sets_input_size(self):
        model = m.SequentialBuilder().dense(4, 8).build()
        assert model.layers[0].input_size == 4

    def test_dense_sets_output_size(self):
        model = m.SequentialBuilder().dense(4, 8).build()
        assert model.layers[0].output_size == 8

    @pytest.mark.parametrize(
        "method, expected_type",
        [
            ("relu", a.ReLu),
            ("sigmoid", a.Sigmoid),
            ("tanh", a.Tanh),
            ("softmax", a.Softmax),
        ],
    )
    def test_activation_methods_add_correct_type(self, method, expected_type):
        model = getattr(m.SequentialBuilder(), method)().build()
        assert isinstance(model.layers[0], expected_type)

    def test_elu_adds_elu_layer(self):
        model = m.SequentialBuilder().elu().build()
        assert isinstance(model.layers[0], a.ELU)

    def test_elu_sets_alpha_activation(self):
        model = m.SequentialBuilder().elu(alpha_activation=0.5).build()
        assert model.layers[0].alpha_activation == 0.5

    # --- hyperparameter methods ---

    def test_optimizer_sets_correctly(self):
        model = m.SequentialBuilder().optimizer("sgd").build()
        assert model.optimizer == "sgd"

    def test_alpha_sets_correctly(self):
        model = m.SequentialBuilder().alpha(0.001).build()
        assert model.alpha == 0.001

    def test_batch_sets_correctly(self):
        model = m.SequentialBuilder().batch(64).build()
        assert model.batch_size == 64

    def test_epochs_sets_correctly(self):
        model = m.SequentialBuilder().epochs(50).build()
        assert model.epochs == 50

    def test_loss_sets_correctly(self):
        cce = ls.CategoricalCrossEntropy()
        model = m.SequentialBuilder().loss(cce).build()
        assert model.loss_class is cce

    # --- fluent API ---

    def test_all_builder_methods_return_self(self):
        builder = m.SequentialBuilder()
        assert builder.flatten() is builder
        assert builder.dense(4, 8) is builder
        assert builder.relu() is builder
        assert builder.sigmoid() is builder
        assert builder.tanh() is builder
        assert builder.softmax() is builder
        assert builder.elu() is builder
        assert builder.optimizer("adam") is builder
        assert builder.alpha(0.01) is builder
        assert builder.batch(32) is builder
        assert builder.epochs(100) is builder
        assert builder.loss(ls.MeanSquaredError()) is builder

    # --- full chain ---

    def test_full_chain_layer_count(self):
        model = (
            m.SequentialBuilder()
            .flatten()
            .dense(784, 128)
            .relu()
            .dense(128, 10)
            .softmax()
            .build()
        )
        assert len(model.layers) == 5

    def test_full_chain_layer_order(self):
        model = (
            m.SequentialBuilder()
            .flatten()
            .dense(784, 128)
            .relu()
            .dense(128, 10)
            .softmax()
            .build()
        )
        assert isinstance(model.layers[0], b.Flatten)
        assert isinstance(model.layers[1], b.Dense)
        assert isinstance(model.layers[2], a.ReLu)
        assert isinstance(model.layers[3], b.Dense)
        assert isinstance(model.layers[4], a.Softmax)

    def test_full_chain_hyperparams(self):
        cce = ls.CategoricalCrossEntropy()
        model = (
            m.SequentialBuilder()
            .dense(4, 2)
            .softmax()
            .optimizer("adam")
            .alpha(0.001)
            .batch(64)
            .epochs(10)
            .loss(cce)
            .build()
        )
        assert model.optimizer == "adam"
        assert model.alpha == 0.001
        assert model.batch_size == 64
        assert model.epochs == 10
        assert model.loss_class is cce

    def test_multiple_builds_are_independent(self):
        builder = m.SequentialBuilder().dense(4, 8).relu()
        model_a = builder.build()
        model_b = builder.build()
        assert model_a is not model_b
        assert model_a.layers[0] is not model_b.layers[0]
