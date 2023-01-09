import yaml
import numpy as np
from pathlib import Path
from itertools import permutations
from scobi.concepts import init as concept_init
from scobi.utils.decorators import FUNCTIONS, PROPERTIES
from termcolor import colored


# TODO: restrict actions, define rewards, more verbose output when using focus files
def FocusFileParserError(msg):
    print(colored("scobi >", "light_red"), "Parser Error: "+msg)
    exit()

       


class Focus():
    def __init__(self, env_name, interactive, fodir, fofile, raw_features, actions):
        concept_init()
        self.PROPERTY_LIST = []
        self.FUNCTION_LIST = []
        self.OBJECTS = raw_features.values()
        self.OBJECT_NAMES = [x.name for x in self.OBJECTS]
        self.ACTIONS = actions
        self.ENV_NAME = env_name
        self.PARSED_OBJECTS = []
        self.PARSED_ACTIONS = []
        self.PARSED_PROPERTIES = []
        self.PARSED_FUNCTIONS = []
        self.FOCUSFILEPATH = None

        self.generate_property_set()
        self.generate_function_set()

        # rework the control flow here, keep interactive mode or not?
        if interactive == True:
            print(colored("scobi >", "blue"),  "Interactive Mode")
            if fofile:
                fpath = Path.cwd() / Path(fodir)  / Path(fofile)
                if fpath.exists():
                    print(colored("scobi >", "blue"), "Focus file %s found." % colored(fpath.name, "light_green"))
                    self.load_focus_file(fpath)
                    print(colored("scobi >", "blue"), "File is valid. Imported.")
                    self.FOCUSFILEPATH = fpath
                else:
                    print(colored("scobi >", "light_red"), "Specified focus file %s not found!" %  colored(fpath.name, "light_green"))
                    exit()
            else:
                fpath = Path.cwd() / Path(fodir) / Path("default_focus_" + env_name + ".yaml")
                if fpath.exists():
                    self.FOCUSFILEPATH = fpath
                    print(colored("scobi >", "light_red"), "No focus file specified, but found an auto-generated default. Edit %s and pass it to continue." % colored(fpath.name, "light_green"))
                    exit()
                else:
                    self.generate_fresh_yaml(fpath)
                    print(colored("scobi >", "light_red"), "No focus file specified! Auto-generated a default focus file. Edit %s and pass it to continue." % colored(fpath.name, "light_green"))
                    exit()
        else:
            print(colored("scobi >", "blue"),  "Non-Interactive Mode")
            if fofile:
                print(colored("scobi >", "yellow"), "Specified focus file ignored, because in non-interactive scobi mode. Using default.")
            fpath = Path.cwd() / Path(fodir)  / Path("default_focus_" + env_name + ".yaml")
            if not fpath.exists():
                self.generate_fresh_yaml(fpath)
                print(colored("scobi >", "yellow"), "No default focus file found. Auto-generated %s." % colored(fpath.name, "light_green"))
            print(colored("scobi >", "blue"), "Focus file %s found." % colored(fpath.name, "light_green"))
            self.load_focus_file(fpath)
            print(colored("scobi >", "blue"), "File is valid. Imported.")
            self.FOCUSFILEPATH = fpath



    def generate_property_set(self):
        for k, v in PROPERTIES.items():
            for o in self.OBJECTS:
                if type(o) == v["expects"][0][0].annotation: #assume only one input from property
                    e = [k, o.name]
                    self.PROPERTY_LIST.append(e)

    def generate_function_set(self):
        for k, v in FUNCTIONS.items():
            para_len = len(v["expects"])
            property_combis = permutations(self.PROPERTY_LIST, para_len)
            for combi in property_combis:
                combi_sig = [PROPERTIES[x[0]]["returns"][0] for x in combi]
                function_sig = [x[0].annotation for x in v["expects"]]
                if combi_sig == function_sig:
                    self.FUNCTION_LIST.append([k, list(combi)])

    def get_object_by_name(self, name, objs):
        if type(objs) == dict:
            for o in objs.values():
                if o.name == name:
                    return o
            return None
        else:
            for o in objs:
                if o.name == name:
                    return o
            return None


    def print_state(self):
        print("---OBJECTS---")
        for o in self.OBJECTS:
            print(o.name)
        print("---PROPERTIES---")
        for p in self.PROPERTY_LIST:
            print(p)
        print("---FUNCTIONS---")
        for f in self.FUNCTION_LIST:
            print(f)


    def avail_to_yaml_dict(self, k, v):
        out_dict = {
            k : {
                "in": [x[1] for x in v["expects"]],
                "description": v["returns"][1] 
            }
        }
        return out_dict

    def proplist_to_yaml_dict(self, x):
        out_dict = {
            x[0] : x[1]
        }
        return out_dict

    def funclist_to_yaml_dict(self, x):
        instance_arg_list = []

        for i in x[1]:
            instance_arg_list.append({i[0] : i[1]})

        out_dict = {
            x[0] : instance_arg_list
        }
        return out_dict


    def generate_fresh_yaml(self, fpath):
        yaml_dict = {
            "ENVIRONMENT" : "",
            "AVAILABLE_CONCEPTS" : {
                "objects" : [],
                "actions" : [],
                "properties" : [],
                "functions" : []
            },
            "SELECTION": {
                "objects" : [],
                "actions" : [],
                "properties" : [],
                "functions" : []
            }
        }

        #print(self.PROPERTY_LIST)
        yaml_dict["ENVIRONMENT"] = self.ENV_NAME
        avail = yaml_dict["AVAILABLE_CONCEPTS"]
        avail["objects"] = [x.name for x in self.OBJECTS]
        avail["actions"] = [x for x in self.ACTIONS]
        avail["properties"] = [self.avail_to_yaml_dict(k, v) for k, v in PROPERTIES.items()]
        avail["functions"] =  [self.avail_to_yaml_dict(k, v) for k, v in FUNCTIONS.items()]

        use = yaml_dict["SELECTION"]
        use["objects"] = [x.name for x in self.OBJECTS]
        use["actions"] = [x for x in self.ACTIONS]
        use["properties"] = [self.proplist_to_yaml_dict(x) for x in self.PROPERTY_LIST]
        use["functions"] = [self.funclist_to_yaml_dict(x) for x in self.FUNCTION_LIST]
        #print(yaml_dict)

        with open(fpath, "w") as f:
            yaml.dump(yaml_dict, f, sort_keys=False)


    def validate_objects(self, objs):
        if not objs:
            return False
        for o in objs:
            if o not in self.OBJECT_NAMES:
                return False
        return True


    def validate_actions(self, acts):
        if not acts:
            return False
        for o in acts:
            if o not in self.ACTIONS:
                return False
        return True


    def validate_properties(self, props):
        for p in props:
            if p not in PROPERTIES.keys():
                return False
        return True


    def validate_functions(self, funcs):
        for f in funcs:
            if f not in FUNCTIONS.keys():
                return False
        return True


    def validate_properties_signatures(self, propslist):
        for p in propslist:
            if p[1] not in self.OBJECT_NAMES:
               FocusFileParserError("Unknown object in properties selection: %s" % p[1])
            if p[0] not in PROPERTIES.keys():
                FocusFileParserError("Unknown object in properties selection: %s" % p[0])
            prop_definition = PROPERTIES[p[0]]
            o = self.get_object_by_name(p[1], self.OBJECTS)
            prop_sig = prop_definition["expects"][0][0].annotation
            if type(o) != prop_sig:
                print(colored("scobi >", "light_red"), "Parser Error")
                FocusFileParserError("Signature mismatch. Property '%s' expects '%s'" % (p[0], prop_sig))
        return True


    def validate_functions_signatures(self, funclist):
        for f in funclist:
            parsed_para_sig = []
            if f[0] not in FUNCTIONS.keys():
                FocusFileParserError("Unknown function in function selection: %s" % f[0])
            for para in f[1]:
                if para[0] not in PROPERTIES.keys():
                    FocusFileParserError("Unknown property in functions selection: %s" % para[0])
                if para[1] not in self.OBJECT_NAMES:
                    FocusFileParserError("Unknown object in functions selection: %s" % para[1])
                prop_definition = PROPERTIES[para[0]]
                o = self.get_object_by_name(para[1], self.OBJECTS)
                prop_sig = prop_definition["expects"][0][0].annotation
                parsed_para_sig.append(prop_definition["returns"][0])
                if type(o) != prop_sig:
                    FocusFileParserError("Signature mismatch in functions selection. Property '%s' expects '%s'" % (para[0], prop_sig))
            func_definition = FUNCTIONS[f[0]]
            function_sig = [x[0].annotation for x in func_definition["expects"]]
            sig_desc = [x[1] for x in func_definition["expects"]]
            if function_sig != parsed_para_sig:
                FocusFileParserError("Signature mismatch in functions selection. Function '%s' expects '%s'" % (f[0], sig_desc))
        return True


    def import_objects(self, objs):
        if self.validate_objects(objs):
            return objs
        else:
            FocusFileParserError("Invalid objects specified in objects selection!")

    def import_actions(self, acts):
        if self.validate_actions(acts):
            return acts
        else:
            FocusFileParserError("Invalid actions specified in actions selection!")

    def import_properties(self, props):
        out = []
        for p in props:
            out.append(list(p.items())[0])
        out_list =  list(map(list, out))
        if self.validate_properties_signatures(out_list):
            return out_list
        else:
            return None

    def import_functions(self ,funcs):
        out = []
        funcs_to_vali = []
        properties_to_vali = []
        objects_to_vali = []
        for p in funcs:
            item = list(p.items())[0]
            fname = item[0]
            fparas = item[1]
            funcs_to_vali.append(fname)
            para_list = [fname, []]
            for p in fparas:
                para_tuple = list(p.items())[0]
                properties_to_vali.append(para_tuple[0])
                objects_to_vali.append(para_tuple[1])
                para_list[1].append(list(para_tuple))
            out.append(para_list)
        if self.validate_functions_signatures(out):
            return out
        else:
            return None

    def load_focus_file(self, fpath):
        with open(fpath, "r") as f:
            in_dict = yaml.safe_load(f)
        parsed_env_name = in_dict["ENVIRONMENT"]
        if self.ENV_NAME != parsed_env_name:
            FocusFileParserError("Env and focus file env do not match: %s, %s" % (self.ENV_NAME, parsed_env_name))
        sdict = in_dict["SELECTION"]
        self.PARSED_OBJECTS = self.import_objects(sdict["objects"])
        self.PARSED_ACTIONS = self.import_actions(sdict["actions"])
        self.PARSED_PROPERTIES = self.import_properties(sdict["properties"])
        self.PARSED_FUNCTIONS = self.import_functions(sdict["functions"])


    def get_feature_vector(self, inc_objects):
        fv = []
        for p in self.PARSED_PROPERTIES:
            property = p[0]
            obj = p[1]
            prop = PROPERTIES[property]["object"]
            o = self.get_object_by_name(obj, inc_objects)
            feature = prop(o)
            fv.append(feature)
        for f in self.PARSED_FUNCTIONS:
            func = f[0]
            input_props = f[1]
            property_results = []
            for p in input_props:
                property = p[0]
                obj = p[1]
                prop = PROPERTIES[property]["object"]
                o = self.get_object_by_name(obj, inc_objects)
                res = prop(o)
                property_results.append(res)
            f = FUNCTIONS[func]["object"]
            feature = f(*property_results)
            fv.append(feature)
        return np.hstack(fv).tolist()