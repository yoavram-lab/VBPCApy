# ARGSCHK - validate and process optional parameter/value pairs that are passed to a function.
# Merging User-Specified Options with Default Options
# Case-Insensitive Parameter Matching

def argschk(defopts, **kwargs):
    wrnmsg = ""
    opts = defopts.copy()
    opts_in = kwargs

    for key in list(opts_in.keys()):
        if(not key.islower()):
            opts_in[key.lower()] = opts_in.pop(key)

    unknown_params = []
    for key in opts_in:
        if(key not in opts):
            unknown_params.append(key)
        opts[key] = opts_in[key]

    if unknown_params:
        wrnmsg = f"Unknown parameter(s): {', '.join(unknown_params)}"

    return (opts, wrnmsg)