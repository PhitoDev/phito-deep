# Library Overview

Phito-Deep is a deep learning framework developed by [me](https://ralphdugue.com/). It provides a sequential API for architecting deep neural networks. It has many logical defaults that are also easily customizable. It&rsquo;s main purpose is to to beeducational, and will continuously be updated as I learn more about deep learning architectures and their components.

## Initialization

There are currently two ways to initialize a model:

-   The \`Sequential\` class. This is the main class for impeachmenting all architectures.
```python
    from phitodeep.model import Sequential
    from phitodeep.layers.base import Dense, Flatten
    from phitodeep.layers.activation import Softmax, ReLu
    from phitodeep.loss import CategoricalCrossEntropy
    
    model = Sequential(
        Flatten(),
        Dense(64, 128),
        ReLu(),
        Dense(128, 64),
        ReLu(),
        Dense(64, 10),
    )
    model.add(Softmax())
    model.setoptimizer("adam")
    model.setloss(CategoricalCrossEntropy())
    model.setbatchsize(4)
    
    model.summary()
``````
-   The \`SequentialBuilder\` class. This just a wrapper around the model class to allow for a more fluent building experience.
```python
    from phitodeep.model import SequentialBuilder
    from phitodeep.loss import CategoricalCrossEntropy
    
    model = (
        SequentialBuilder()
        .flatten()
        .dense(64, 128)
        .relu()
        .dense(128, 64)
        .relu()
        .dense(64, 10)
        .softmax()
        .optimizer("adam")
        .loss(CategoricalCrossEntropy())
        .alpha(0.05)
        .epochs(5)
        .batch(4)
        .build()
    )
    
    model.summary()
```
If you run both cell blocks, you will see they generate similar models. The builder will save you time on imports, but the \`Sequential\` class is a bit more dynamic.

## Training

Training a model pretty straight forward. You call the \`train\` function with your data, and it handles all forward and back propagation:
```python
    import numpy as np
    
    rng = np.random.default_rng(42)
    
    X = rng.standard_normal((64, 8, 8)).astype(np.float32)
    y = rng.integers(0, 10, 64)
    
    split = int(0.6 * X.shape[0])
    x_train, y_train = X[:split], y[:split]
    x_temp, y_temp = X[split:], y[split:]
    
    split = int(0.5 * x_temp.shape[0])
    x_test, y_test = x_temp[:split], y_temp[:split]
    x_val, y_val = x_temp[split:], y_temp[split:]
    
    
    losses = model.train(x_train, y_train, x_test, y_test)
``````
And of course predictions:
```python
    y_pred = model.predict(x_val)
    accuracy = np.mean(np.argmax(y_pred, axis=1) == y_test)
    print(f"Test Accuracy: {accuracy:.4f} %")
```
