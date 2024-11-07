def converg_check(opts, lc, angleA, sd_iter=None):
    """
    Checks convergence criteria and generates a message for display.

    Parameters:
    - opts: Dictionary containing options for convergence checks (e.g., minangle, earlystop, rmsstop, cfstop).
    - lc: Dictionary with lists for convergence parameters, including 'prms', 'rms', and 'cost'.
    - angleA: Float representing the angle between subspaces.
    - sd_iter: Optional integer for checking if a slowing-down stop is required (defaults to None).

    Returns:
    - convmsg: String message indicating the convergence status.
    """

    convmsg = ""

    # Check angle convergence criterion
    if angleA < opts.get('minangle', float('inf')):
        convmsg = f"Convergence achieved (angle between subspaces smaller than {opts['minangle']:.2e})\n"

    # Check early stopping criterion
    elif opts.get('earlystop', False) and lc['prms'][-1] > lc['prms'][-2]:
        convmsg = "Early stopping.\n"

    # Check RMS stop criterion
    elif 'rmsstop' in opts and opts['rmsstop'] and len(lc['rms']) - 1 > opts['rmsstop'][0]:
        numiter = opts['rmsstop'][0]
        abs_tol = opts['rmsstop'][1]
        rel_tol = opts['rmsstop'][2] if len(opts['rmsstop']) > 2 else None

        rms1 = lc['rms'][-numiter - 1]
        rms2 = lc['rms'][-1]

        if abs(rms1 - rms2) < abs_tol or (rel_tol is not None and abs((rms1 - rms2) / rms2) < rel_tol):
            convmsg = f"Stop: RMS does not change much for {numiter} iterations.\n"

    # Check cost function stop criterion
    elif 'cfstop' in opts and opts['cfstop'] and len(lc['cost']) - 1 > opts['cfstop'][0]:
        numiter = opts['cfstop'][0]
        abs_tol = opts['cfstop'][1]
        rel_tol = opts['cfstop'][2] if len(opts['cfstop']) > 2 else None

        cost1 = lc['cost'][-numiter - 1]
        cost2 = lc['cost'][-1]

        if abs(cost1 - cost2) < abs_tol or (rel_tol is not None and abs((cost1 - cost2) / cost2) < rel_tol):
            convmsg = f"Stop: Cost does not change much for {numiter} iterations.\n"

    # Slowing-down stop criterion if sd_iter is provided
    elif sd_iter is not None and sd_iter == 40:
        convmsg = "Slowing-down stop. You may continue by changing the gradient type.\n"

    return convmsg
