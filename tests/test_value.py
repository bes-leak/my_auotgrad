import math
import pytest
from autograd import Value

def num_grad(f, x, eps=1e-3):
    return (f(x + eps) - f(x - eps)) / (2 * eps)


def test_add_froward():
    assert (Value(2.0) + Value(3.0)).data == 5.0


def test_mul_forward():
    assert (Value(2.0) * Value(3.0)).data == 6.0

def test_reverse_operators():
    assert (2 * Value(3.0)).data == 6.0     # __rmul__
    assert (2 + Value(3.0)).data == 5.0     # __radd__
    assert (-Value(3.0)).data == -3.0       # __neg__
    assert (Value(3.0) / 2).data == 1.5     # __truediv__

def test_grad_accumulation():
    # f = a*b + a -> df/da = b+1 = 5, df/db = a = 3 (a входит дважды!)
    a, b = Value(3.0), Value(4.0)
    f = a * b + a
    f.backward()
    assert a.grad == pytest.approx(5.0)
    assert b.grad == pytest.approx(3.0)

def test_pow_grad():
    # (x^3)' = 3x^2 = 12 при x=2
    x = Value(2.0)
    (x ** 3).backward()
    assert x.grad == pytest.approx(12.0)

def test_log_grad():
    # (ln x)' = 1/x = 0.2 при x=5
    x = Value(5.0)
    x.log().backward()
    assert x.grad == pytest.approx(0.2)

def test_relu_grad():
    xp = Value(2.0)
    xp.relu().backward()
    xn = Value(-1.0)
    xn.relu().backward()
    assert xp.grad == pytest.approx(1.0)
    assert xn.grad == pytest.approx(0.0)

@pytest.mark.parametrize("xv", [0.5, 1.0, 2.0, 3.0])
def test_backward_matches_numeric(xv):
    formula = lambda t: math.exp(t) * t ** 2     # f(x) = e^x · x²
    x = Value(xv)
    f = x.exp() * (x ** 2)
    f.backward()
    assert x.grad == pytest.approx(num_grad(formula, xv), rel=1e-4)


def test_gradient_descent_converges():
    w = Value(0.0)
    b = Value(0.0)
    delta = 0.1

    for _ in range(100):
        loss = (w - 1)**2 + (b + 2)**2 - 5
        loss.backward()
        w -= delta * w.grad
        b -= delta * b.grad
        w.grad = 0.0
        b.grad = 0.0

    assert w.data == pytest.approx(1.0)
    assert b.data == pytest.approx(-2.0)