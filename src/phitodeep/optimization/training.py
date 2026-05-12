import numpy as np
from .optimizers import Optimizer


def train_loop(
    model, X, y, X_test, y_test, loss_class, optimizer, epochs=1000, batch_size=None
):

    losses = []
    rng = np.random.default_rng()

    for epoch in range(epochs):
        if batch_size is not None:
            for _ in range(len(X) // batch_size):
                indices = rng.integers(0, len(X), batch_size)
                X_batch = X[indices]
                y_batch = y[indices]

                # Forward pass
                y_pred = model.predict(X_batch)
                loss = loss_class.loss_func(y_pred, y_batch)

                # Compute gradient of loss w.r.t. predictions (dy)
                dy = loss_class.loss_gradient(y_pred, y_batch)

                # Backward pass with gradient of loss
                model.backward(dy)
                optimizer.step(model.layers)
        else:
            indices = rng.integers(0, len(X), len(X))
            # Forward pass
            y_pred = model.predict(X[indices])
            loss = loss_class.loss_func(y_pred, y[indices])

            # Compute gradient of loss w.r.t. predictions (dy)
            dy = loss_class.loss_gradient(y_pred, y[indices])

            # Backward pass with gradient of loss
            model.backward(dy)
            optimizer.step(model.layers)

        y_pred = model.predict(X)
        loss = loss_class.loss_func(y_pred, y)

        y_pred_test = model.predict(X_test)
        loss_test = loss_class.loss_func(y_pred_test, y_test)

        losses.append((loss, loss_test))

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}, Test Loss: {loss_test:.4f}")

    return losses
