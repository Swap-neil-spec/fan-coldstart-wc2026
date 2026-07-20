"""Validate the special functions against closed-form identities, so the core
does not silently depend on scipy being present."""
import math
import numpy as np

from fancoldstart.special import gammaln, betaln, hyp2f1, logsumexp2


def test_gammaln_matches_stdlib():
    xs = np.array([0.5, 1.0, 2.5, 7.0, 20.0])
    assert np.allclose(gammaln(xs), [math.lgamma(x) for x in xs])


def test_betaln_identity():
    a, b = 2.3, 4.1
    assert abs(float(betaln(a, b)) - (math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b))) < 1e-10


def test_hyp2f1_identity_power():
    # 2F1(a, b; b; z) = (1 - z) ** (-a)
    z = np.linspace(0.0, 0.9, 10)
    a = 1.7
    got = hyp2f1(a, 3.0, 3.0, z)
    assert np.allclose(got, (1 - z) ** (-a), atol=1e-8)


def test_hyp2f1_identity_log():
    # 2F1(1, 1; 2; z) = -ln(1 - z) / z
    z = np.linspace(0.05, 0.9, 10)
    got = hyp2f1(1.0, 1.0, 2.0, z)
    assert np.allclose(got, -np.log(1 - z) / z, atol=1e-8)


def test_logsumexp2_handles_neg_inf():
    a = np.array([0.0, 1.0, -np.inf])
    b = np.array([-np.inf, 2.0, -np.inf])
    out = logsumexp2(a, b)
    assert abs(out[0] - 0.0) < 1e-12
    assert abs(out[1] - math.log(math.e + math.e ** 2)) < 1e-10
    assert out[2] == -np.inf
