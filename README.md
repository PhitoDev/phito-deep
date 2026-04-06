# phitodeep

Deep learning framework built from scratch with numpy!

## Installation

```bash
$ pip install phitodeep
```

## Usage
MNIST quickstart:
```python
import numpy as np
from datasets import load_dataset

import phitodeep.loss as loss
import phitodeep.model as m

train_dataset = load_dataset("ylecun/mnist", split="train")
test_dataset = load_dataset("ylecun/mnist", split="test")

X_train = train_dataset["image"]
y_train = train_dataset["label"]
X_test = test_dataset["image"]
y_test = test_dataset["label"]

X_train = np.array(X_train).astype(np.float32) / 255.0
y_train = np.array(y_train)
X_test = np.array(X_test).astype(np.float32) / 255.0
y_test = np.array(y_test)
print(X_train.shape, y_train.shape)

model = (
    m.SequentialBuilder()
    .flatten()
    .dense(784, 128)
    .relu()
    .dense(128, 10)
    .softmax()
    .optimizer("adam")
    .loss(loss.CategoricalCrossEntropy())
    .alpha(0.001)
    .epochs(300)
    .batch(32)
    .build()
)

model.summary()

model.train(X_train, y_train, X_test, y_test)

```

## Contributing

Interested in contributing? Check out the contributing guidelines. Please note that this project is released with a Code of Conduct. By contributing to this project, you agree to abide by its terms.

## License

`phitodeep` was created by Ralph Dugue. It is licensed under the terms of the Apache License 2.0 license.

## Credits

`phitodeep` was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).
