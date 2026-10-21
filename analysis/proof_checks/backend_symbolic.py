"""Exact symbolic checks for covariance contractions and mean-field variance.

The moment check assumes independent centered loading, score and bias errors
with finite second moments. It keeps within-vector covariance terms. It does
not assume diagonal covariances or verify floating-point error/convergence.
"""

from __future__ import annotations

import json
from itertools import product

import sympy as sy


def check_identities() -> dict[str, bool]:
    p, n, q = 2, 3, 2
    a = sy.Matrix(p, q, sy.symbols("a:4"))
    s = sy.Matrix(q, n, sy.symbols("s:6"))
    av = [sy.Matrix(q, q, sy.symbols(f"av{i}_:4")) for i in range(p)]
    sv = [sy.Matrix(q, q, sy.symbols(f"sv{j}_:4")) for j in range(n)]

    def frobenius(x, y):
        return sum(
            x[row, column] * y[row, column]
            for row, column in product(range(q), repeat=2)
        )

    original = sum(
        (a.row(i) * sv[j] * a.row(i).T)[0]
        + (s.col(j).T * av[i] * s.col(j))[0]
        + frobenius(av[i], sv[j])
        for i, j in product(range(p), range(n))
    )
    zero = sy.zeros(q, q)
    collapsed = (
        frobenius(a.T * a, sum(sv, zero))
        + frobenius(s * s.T, sum(av, zero))
        + frobenius(sum(av, zero), sum(sv, zero))
    )
    if sy.expand(original - collapsed) != 0:
        raise ArithmeticError("complete covariance contraction differs")

    u0, u1, v0, v1, d = sy.symbols("u0 u1 v0 v1 d")
    a0, a1, s0, s1 = sy.symbols("a0 a1 s0 s1")
    aa, ab, bb, cc, cd, dd, muv = sy.symbols("aa ab bb cc cd dd muv")
    ca, cs = sy.Matrix([[aa, ab], [ab, bb]]), sy.Matrix([[cc, cd], [cd, dd]])

    def second_moment(powers, covariance):
        degree = sum(powers)
        if degree == 0:
            return sy.Integer(1)
        if degree == 1:
            return sy.Integer(0)
        if degree != 2:
            raise ValueError("only finite second moments are assumed")
        indices = [i for i, power in enumerate(powers) for _ in range(power)]
        return covariance[indices[0], indices[1]]

    def expectation(expr):
        value = sy.Integer(0)
        for powers, coefficient in sy.Poly(sy.expand(expr), u0, u1, v0, v1, d).terms():
            bias_moment = (1, 0, muv)[powers[4]]
            value += (
                coefficient
                * second_moment(powers[:2], ca)
                * second_moment(powers[2:4], cs)
                * bias_moment
            )
        return sy.expand(value)

    y = (a0 + u0) * (s0 + v0) + (a1 + u1) * (s1 + v1) + d
    mean = expectation(y)
    variance = expectation(y**2) - mean**2
    avec, svec = sy.Matrix([a0, a1]), sy.Matrix([s0, s1])
    formula = (
        (avec.T * cs * avec)[0] + (svec.T * ca * svec)[0] + frobenius(ca, cs) + muv
    )
    if sy.expand(variance - formula) != 0:
        raise ArithmeticError("mean-field marginal variance differs")
    return {"complete_covariance_contraction": True, "mean_field_variance": True}


if __name__ == "__main__":
    print(json.dumps(check_identities(), indent=2))
