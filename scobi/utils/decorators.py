import inspect

FUNCTIONS = dict()
PROPERTIES = dict()


# decorator to register properties and functions
def register(*args, **kwargs):

    def inner(func):
        sig = inspect.signature(func)
        ret_ano = sig.return_annotation
        sig_list = sig.parameters
        param_descs = kwargs["params"]
        ret_desc = kwargs["desc"]
        sig_dict = {"object": func, "expects": [], "returns": None}
        sig_dict["returns"] = (ret_ano, ret_desc)
        if len(sig_list) == len(param_descs):
            for k in sig_list.keys():
                desc = param_descs.pop(0)
                sig_dict["expects"].append((sig_list[k], desc))
        else:
            print("id error")
        name = kwargs["name"]
        if name in FUNCTIONS.keys():
            print("name already registered")
        else:
            if kwargs["type"] == "F": # function
                FUNCTIONS[kwargs["name"]] = sig_dict
            elif kwargs["type"] == "P": # property
                PROPERTIES[kwargs["name"]] = sig_dict
            else:
                print("unknown type")
    return inner