"""
Local robustness verification for the MNIST FC classifier using Marabou.

Given a correctly-classified input x with true label y, we verify:

    For all x' such that ||x' - x||_inf <= epsilon,
    argmax(f(x')) == y.

In SMT terms, Marabou tries to find a counterexample, i.e. an input x'
inside the L_inf ball whose top-1 class is NOT y. We encode this by
adding a disjunction of output constraints:
    OR over j != y:  logit_j  >=  logit_y
If Marabou returns SAT, such a counterexample exists -> NOT robust.
If Marabou returns UNSAT, no counterexample exists -> robust within epsilon.

This script is reused from test.py.
"""

import time
from pathlib import Path

import numpy as np
from maraboupy import Marabou, MarabouCore, MarabouUtils


ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "mnist_fc.onnx"


def load_network():
    """Load the ONNX model into Marabou and return (network, in_vars, out_vars)."""
    network = Marabou.read_onnx(str(MODEL_PATH))
    input_vars = network.inputVars[0].flatten()    # 784 variables
    output_vars = network.outputVars[0].flatten()  # 10 variables (logits)
    return network, input_vars, output_vars


def verify_robustness(image, true_label, epsilon, timeout=120, verbose=False):
    """Verify robustness of the network around `image` within an L_inf ball
    of radius `epsilon`.

    Args:
        image:      np.ndarray of shape (1, 28, 28) or (784,), values in [0, 1].
        true_label: int, the ground-truth class (0..9).
        epsilon:    float, perturbation radius.
        timeout:    seconds before Marabou gives up.
        verbose:    print Marabou's internal stats.

    Returns:
        dict with keys: result ('SAT' | 'UNSAT' | 'TIMEOUT'),
                        runtime (sec),
                        counterexample (np.ndarray or None),
                        adv_label (int or None).
    """
    network, in_vars, out_vars = load_network()
    flat = image.flatten()
    assert flat.size == in_vars.size, f"Input size mismatch: {flat.size} vs {in_vars.size}"

    # 1) Input constraints: an L_inf ball clipped to [0, 1]
    for i, v in enumerate(in_vars):
        lo = max(0.0, float(flat[i]) - epsilon)
        hi = min(1.0, float(flat[i]) + epsilon)
        network.setLowerBound(v, lo)
        network.setUpperBound(v, hi)

    # 2) Output constraints: we want SOME wrong class j to dominate the true class.
    #    Marabou expresses "OR" via addDisjunctionConstraint, where each
    #    disjunct is a list of linear equations.
    disjuncts = []
    for j in range(out_vars.size):
        if j == true_label:
            continue
        # Encode:  logit_j - logit_y >= 0
        eq = MarabouUtils.Equation(MarabouCore.Equation.GE)
        eq.addAddend(1.0, int(out_vars[j]))
        eq.addAddend(-1.0, int(out_vars[true_label]))
        eq.setScalar(0.0)
        disjuncts.append([eq])
    network.addDisjunctionConstraint(disjuncts)

    # 3) Solve
    options = Marabou.createOptions(timeoutInSeconds=timeout, verbosity=0)
    start = time.time()
    exit_code, vals, stats = network.solve(options=options, verbose=verbose)
    runtime = time.time() - start

    # Marabou returns a string like 'sat', 'unsat', or 'TIMEOUT'
    result = exit_code.upper() if isinstance(exit_code, str) else str(exit_code)

    counterexample = None
    adv_label = None
    if result == "SAT":
        # vals is a dict mapping variable index -> assigned value.
        # Reconstruct the adversarial input.
        adv = np.array([vals[int(v)] for v in in_vars]).reshape(image.shape)
        counterexample = adv
        # Compute the predicted label of the counterexample using the
        # output variables that Marabou also returns.
        logits = np.array([vals[int(v)] for v in out_vars])
        adv_label = int(np.argmax(logits))

    return {
        "result": result,
        "runtime": runtime,
        "counterexample": counterexample,
        "adv_label": adv_label,
    }


def run_epsilon_sweep(image, true_label, epsilons, timeout=120):
    """Run verification for a single image at multiple epsilon values."""
    rows = []
    for eps in epsilons:
        print(f"  -> verifying epsilon = {eps}")
        out = verify_robustness(image, true_label, eps, timeout=timeout)
        rows.append((eps, out["result"], out["runtime"], out["adv_label"]))
        print(f"     result = {out['result']:8s}  runtime = {out['runtime']:.2f}s"
              + (f"  adv -> {out['adv_label']}" if out["adv_label"] is not None else ""))
    return rows
