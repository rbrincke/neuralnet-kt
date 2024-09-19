package ch.brincke.ch.rbrincke.scalar

import kotlin.math.exp
import kotlin.math.max
import kotlin.math.pow

class Evaluator(private val loss: Node) {
    private val forward: List<Node>
    private val backward: List<Node>
    private val parameters: List<Parameter>

    init {
        this.forward = topologicalSort(loss)
        this.backward = this.forward.reversed()
        this.parameters = this.forward.filterIsInstance<Parameter>()
    }

    private fun topologicalSort(tail: Node): List<Node> {
        val visited = HashSet<Node>()
        val topological = mutableListOf<Node>()

        fun visit(v: Node) {
            if (v !in visited) {
                visited += v

                v.children().forEach(::visit)
                topological += v
            }
        }

        visit(tail)

        return topological.toList()
    }

    private fun resetGradients() {
        this.backward.forEach { e -> e.gradient = 0.0 }
        this.loss.gradient = 1.0
    }

    fun recomputeGradients() {
        this.resetGradients()
        this.backward.forEach(Node::backpropagateGradient)
    }

    fun recomputeValue(): Node {
        this.forward.forEach(Node::recomputeValue)
        return this.loss
    }

    fun updateParameterValues(learningRate: Double) {
        parameters.forEach { p ->
            p.value += learningRate * p.gradient
        }
    }
}

sealed class Node(var gradient: Double = 0.0, var value: Double = 0.0) {
    abstract fun backpropagateGradient()
    abstract fun children(): Collection<Node>
    abstract fun recomputeValue()

    operator fun plus(other: Node) = Addition(this, other)
    operator fun minus(other: Node) = Subtraction(this, other)
    operator fun times(other: Node) = Multiplication(this, other)
    operator fun unaryMinus() = Negative(this)
    fun pow(other: Double) = Power(this, other)
    fun relu() = RectifiedLinearUnit(this)
    fun tanh() = Tanh(this)
}

class Expectation(value: Double) : Node(value, value) {
    override fun recomputeValue() {}
    override fun backpropagateGradient() {}
    override fun children() = listOf<Node>()
}

class Input(value: Double) : Node(value, value) {
    override fun recomputeValue() {}
    override fun backpropagateGradient() {}
    override fun children() = listOf<Node>()
}

class Parameter(value: Double) : Node(value, value) {
    override fun recomputeValue() {}
    override fun backpropagateGradient() {}
    override fun children() = listOf<Node>()
}

class Addition(private val left: Node, private val right: Node) : Node() {
    override fun recomputeValue() {
        this.value = left.value + right.value
    }

    override fun backpropagateGradient() {
        left.gradient += this.gradient
        right.gradient += this.gradient
    }

    override fun children() = listOf(left, right)
}

class Subtraction(private val left: Node, private val right: Node) : Node() {
    override fun recomputeValue() {
        this.value = left.value - right.value
    }

    override fun backpropagateGradient() {
        left.gradient += -this.gradient
        right.gradient += -this.gradient
    }

    override fun children() = listOf(left, right)
}

class Power(private val previous: Node, private val exponent: Double): Node() {
    override fun recomputeValue() {
        this.value = previous.value.pow(exponent)
    }

    override fun backpropagateGradient() {
        previous.gradient += exponent * previous.value.pow(exponent - 1) * this.gradient
    }

    override fun children() = listOf(previous)
}

class Negative(private val previous: Node) : Node() {
    override fun recomputeValue() {
        this.value = -previous.value
    }

    override fun backpropagateGradient() {
        previous.gradient += -this.gradient
    }

    override fun children() = listOf(previous)
}

class Multiplication(private val left: Node, private val right: Node) : Node() {
    override fun recomputeValue() {
        this.value = left.value * right.value
    }

    override fun backpropagateGradient() {
        left.gradient += right.value * gradient
        right.gradient += left.value * gradient
    }

    override fun children() = listOf(left, right)
}

class RectifiedLinearUnit(private val previous: Node): Node() {
    override fun recomputeValue() {
        this.value = max(previous.value, 0.0)
    }

    override fun backpropagateGradient() {
        previous.gradient += value * gradient
    }

    override fun children() = listOf(previous)
}

class Tanh(private val previous: Node): Node() {
    override fun recomputeValue() {
        this.value = (exp(2 * previous.value) - 1) / (exp(2 * previous.value) + 1)
    }

    override fun backpropagateGradient() {
        previous.gradient += (1 - value.pow(2)) * gradient
    }

    override fun children() = listOf(previous)
}
