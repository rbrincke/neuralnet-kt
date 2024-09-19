package ch.brincke

import ch.brincke.ch.rbrincke.scalar.*

fun main() {
    val outputs = listOf(1.0, -1.0, -1.0, 1.0)
    val inputs = listOf(
        listOf(2.0, 3.0, -1.0),
        listOf(3.0, -1.0, 0.5),
        listOf(0.5, 1.0, 1.0),
        listOf(1.0, 1.0, -1.0)
    )

    val neuralNet = MultiLayerPerceptron(3, listOf(4, 4, 1))

    val predicted = inputs
        .map { neuralNet.connect(it.map(::Input)).single().tanh() }

    val loss = outputs
        .map(::Expectation)
        .zip(predicted)
        .map { (predicted, actual) -> (predicted - actual).pow(2.0) }
        .reduce(Node::plus)

    val evaluator = Evaluator(loss)
    var currentLoss = evaluator.recomputeValue()

    println(currentLoss.value)
    println(predicted.map { e -> e.value })

    fun improve() {
        evaluator.recomputeGradients()
        evaluator.updateParameterValues(-0.1)
        currentLoss = evaluator.recomputeValue()
    }

    repeat(100) { improve() }

    println(currentLoss.value)
    println(predicted.map { e -> e.value })
}
