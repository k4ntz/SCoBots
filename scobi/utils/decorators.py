import inspect

FUNCTIONS = dict()
PROPERTIES = dict()
AGGREGATIONS = dict()


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

        concept_name = kwargs["name"]
        concept_type = kwargs["type"]

        if concept_type in ["F", "P"] and len(sig_list) == len(param_descs):
            for k in sig_list.keys():
                desc = param_descs.pop(0)
                sig_dict["expects"].append((sig_list[k], desc))
        elif concept_type == "A" and len(sig_list) == len(param_descs) + 1:
            for k in list(sig_list.keys())[1:]:
                desc = param_descs.pop(0)
                sig_dict["expects"].append((sig_list[k], desc))
        else:
            print("Index error.")

        if concept_name in FUNCTIONS.keys():
            print(f"Concept with name '{concept_name}' already registered.")
        else:
            if concept_type == "F":  # function
                FUNCTIONS[concept_name] = sig_dict
            elif concept_type == "P":  # property
                PROPERTIES[concept_name] = sig_dict
            elif concept_type == "A":   # aggregation
                AGGREGATIONS[concept_name] = sig_dict
            else:
                print(f"Unknown concept type {concept_type}.")

    return inner

    #     if len(sig_list) == len(param_descs):
    #         for k in sig_list.keys():
    #             desc = param_descs.pop(0)
    #             sig_dict["expects"].append((sig_list[k], desc))
    #     else:
    #         print("id error")
    #     name = kwargs["name"]
    #     if name in FUNCTIONS.keys():
    #         print("name already registered")
    #     else:
    #         if kwargs["type"] == "F": # function
    #             FUNCTIONS[kwargs["name"]] = sig_dict
    #         elif kwargs["type"] == "P": # property
    #             PROPERTIES[kwargs["name"]] = sig_dict
    #         elif concept_type == "A":   # aggregation
    #             AGGREGATIONS[concept_name] = sig_dict
    #         else:
    #             print("unknown type")
    # return inner