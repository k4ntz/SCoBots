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
            match concept_type:
                case "F":  # function
                    FUNCTIONS[concept_name] = sig_dict
                case "P":  # property
                    PROPERTIES[concept_name] = sig_dict
                case "A":   # aggregation
                    AGGREGATIONS[concept_name] = sig_dict
                case other:
                    print(f"Unknown concept type {other}.")

    return inner
