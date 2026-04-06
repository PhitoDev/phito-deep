import numpy as np
from datasets import load_dataset

import phitodeep.layers as layers
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
    .alpha(0.01)
    .epochs(100)
    .batch(32)
    .build()
)

model.summary()

model.train(X_train, y_train, X_test, y_test)
