package ch.brincke.ch.rbrincke.scalar

import kotlin.random.Random

class MultiLayerPerceptron(inputCount: Int, neuronCount: List<Int>, random: Random = Random.Default) {
    private val layers = (listOf(inputCount) + neuronCount)
        .windowed(2)
        .map { (inputCount, outputCount) ->
            Layer(inputCount, outputCount) { random.nextDouble(-1.0, 1.0) }
        }

    fun connect(inputs: List<Input>): List<Node> {
        var currentInputs: List<Node> = inputs

        layers.dropLast(1).forEach { layer ->
            currentInputs = layer.connect(currentInputs).map(Node::tanh)
        }

        return layers.last().connect(currentInputs)
    }
}

class Layer(inputCount: Int, neuronCount: Int, randomWeight: () -> Double) {
    private val neurons: List<Neuron> = List(neuronCount) { Neuron(inputCount, randomWeight) }

    fun connect(inputs: List<Node>): List<Node> {
        return this.neurons.map { it.connect(inputs) }
    }
}

class Neuron(inputCount: Int, randomWeight: () -> Double) {
    private val weights = List(inputCount) { Parameter(randomWeight()) }
    private val bias = Parameter(randomWeight())

    fun connect(inputs: List<Node>): Node {
        check(weights.size == inputs.size) {
            "Weight-input mismatch: ${weights.size} weights, but ${inputs.size} inputs."
        }

        return weights.zip(inputs)
            .map { (w, x) -> w * x }
            .reduce(Node::plus) + bias
    }
}
