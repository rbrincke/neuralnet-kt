# Neural net

Based on Gilbert Strang's _Introduction to Linear Algebra (6th)_ and _Linear Algebra and Learning From Data_ this
repository contains ongoing experiments with neural nets. This led me to find Andrej Karpathy's [_Neural Net: Zero to Hero_](https://karpathy.ai/zero-to-hero.html)
course, which is very helpful too.

The code in the repository is written in Kotlin.

## Scalar

The **scalar** imitates Karpathy's micrograd model, which is also PyTorch's model. But the code works differently: instead
of recreating nodes for every forward pass, the structure of the network (a directed graph) is fixed a priori. Both 
forward and backward passes are then executed against this graph. This avoids creating lots of nodes.

Here is a single neuron with input _x_, weight _w_ and bias _b_.

```kotlin
val x = Input(2.0)
val w = Parameter(-3.0)
val b = Parameter(6.881373587019543)

val loss = (w * x + b).tanh() // Not really a full loss function, but imagine it is.
```

Pass the loss function to the `Evaluator` to initiate the neural network's optimization:

```kotlin
val evaluator = Evaluator(loss)
```

The _Evaluator_ performs a topological sort once. A single (gradient descent) optimization loop consists of the 
following steps:

```kotlin
evaluator.recomputeValue() // Important!
evaluator.recomputeGradients()
evaluator.updateParameterValues(-0.1)
```
