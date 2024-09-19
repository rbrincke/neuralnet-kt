package ch.rbrincke.scalar

import ch.brincke.ch.rbrincke.scalar.Evaluator
import ch.brincke.ch.rbrincke.scalar.Input
import ch.brincke.ch.rbrincke.scalar.Parameter
import kotlin.test.Test
import kotlin.test.assertEquals

class ScalarTest {
    @Test
    fun testForwardPass() {
        val a = Input(2.0)
        val b = Input(3.0)
        val c = Input(5.0)

        val structure = (a + b) * c
        val evaluator = Evaluator(structure)
        val loss = evaluator.recomputeValue()

        assertEquals(25.0, loss.value)
    }

    @Test
    fun testGradients() {
        val x1 = Input(2.0)
        val w1 = Parameter(-3.0)
        val x2 = Input(0.0)
        val w2 = Parameter(1.0)
        val b = Parameter(6.881373587019543)

        val loss = (w1 * x1 + w2 * x2 + b).tanh()
        val evaluator = Evaluator(loss)

        evaluator.recomputeValue()
        evaluator.recomputeGradients()

        assertEquals(w1.gradient, 1.0, 1e-10)
        assertEquals(w2.gradient, 0.0, 1e-10)
        assertEquals(x1.gradient, -1.5, 1e-10)
        assertEquals(x2.gradient, 0.5, 1e-10)
        assertEquals(b.gradient, 0.5, 1e-10)
    }
}
