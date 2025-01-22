#Merges user-provided options (kwargs) with default options (defopts), normalizing keys to lowercase and warning about unknown parameters.
#Gets default options (defopts) as a dictionary and additional keyword arguments (kwargs) as user-provided options.
#Returns A merged options dictionary (opts) and a warning message (wrnmsg) for unknown parameters, if any.

def argschk(defopts, **kwargs):
    print("in argschk", flush=True)
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