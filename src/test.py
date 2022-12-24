import inspect
from typing import Tuple
from xrl.agents.game_object import GameObject
from itertools import permutations
import yaml
import math
import numpy as np
import random
from xrl.agents.feature_processing.aari_feature_processer import FUNCTIONS
print(FUNCTIONS)

ENV_NAME = "Pong-v4"
FUNCTIONS = dict()
PROPERTIES = dict()
OBJECTS = [ GameObject("player", [1, 2, 3], [1, 1]),
            GameObject("enemy", [4, 5, 6], [2, 2]),
            GameObject("ball", [7, 8, 9], [3, 3])
]
OBJECT_NAMES = [x.name for x in OBJECTS]

ACTIONS = ["UP", "DOWN", "FIRE", "NOOP"]


class FocusFileParserError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


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



@register(type="P", name="POSITION", params= ["OBJECT"], desc="get the position for given object")
def get_position(obj: GameObject) -> Tuple[int, int]:
    return tuple(obj.get_coords()[0])

@register(type="P", name="POSITION_HISTORY", params= ["OBJECT"], desc="get the current and last position for given object")
def get_position_history(obj: GameObject) -> Tuple[int, int, int, int]:
    coords = obj.get_coords()
    return tuple(coords[0] + coords[1])

@register(type="P", name="RGB", params= ["OBJECT"], desc="get the rgb value for given object")
def get_rgb(obj: GameObject) -> Tuple[int, int, int]:
    return tuple(obj.rgb)



@register(type="F", name="LINEAR_TRAJECTORY", params=["POSITION", "POSITION_HISTORY"], desc="x, y distance to trajectory")
def calc_lin_traj(a_position: Tuple[int, int], b_history: Tuple[int, int, int, int]) -> Tuple[int, int]:
    return 1, 1


@register(type="F", name="DISTANCE", params=["POSITION", "POSITION"], desc="distance between two coordinates")
def calc_distance(a_position: Tuple[int, int], b_position: Tuple[int, int]) -> Tuple[int, int]:
    distx = b_position[0] - a_position[0]
    disty = b_position[1] - a_position[1]
    return distx, disty


@register(type="F", name="COLORNAME", params=["RGB"], desc="closest colorname of rgb value")
def get_colorname(rgb: Tuple[int, int, int]) -> str:
    return "colorname"


@register(type="F", name="VELOCITY", params=["POSITION_HISTORY"], desc="velocity of object")
def get_velocity(pos_history: Tuple[int, int, int, int]) -> float:
    obj_past = pos_history[2:4]
    obj = pos_history[0:2]
    vel = math.sqrt((obj_past[0] - obj[0])**2 + (obj_past[1] - obj[1])**2)
    return vel
  


PROPERTY_LIST = []
FUNCTION_LIST = []


for k, v in PROPERTIES.items():
    for o in OBJECTS:
        if type(o) == v["expects"][0][0].annotation: #assume only one input from property
            e = [k, o.name]
            PROPERTY_LIST.append(e)


for k, v in FUNCTIONS.items():
    para_len = len(v["expects"])
    property_combis = permutations(PROPERTY_LIST, para_len)
    for combi in property_combis:
        combi_sig = [PROPERTIES[x[0]]["returns"][0] for x in combi]
        function_sig = [x[0].annotation for x in v["expects"]]
        if combi_sig == function_sig:
            FUNCTION_LIST.append([k, list(combi)])


def get_object_by_name(name, obj_list):
    for o in obj_list:
        if o.name == name:
            return o
    return None

def print_state():
    print("---OBJECTS---")
    for o in OBJECTS:
        print(o.name)
    print("---PROPERTIES---")
    for p in PROPERTY_LIST:
        print(p)
    print("---FUNCTIONS---")
    for f in FUNCTION_LIST:
        print(f)



def avail_to_yaml_dict(k, v):
    out_dict = {
        k : {
            "in": [x[1] for x in v["expects"]],
            "description": v["returns"][1] 
        }
    }
    return out_dict

def proplist_to_yaml_dict(x):
    out_dict = {
        x[0] : x[1]
    }
    return out_dict

def funclist_to_yaml_dict(x):
    instance_arg_list = []

    for i in x[1]:
        instance_arg_list.append({i[0] : i[1]})

    out_dict = {
        x[0] : instance_arg_list
    }
    return out_dict



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

print(PROPERTY_LIST)
yaml_dict["ENVIRONMENT"] = ENV_NAME
avail = yaml_dict["AVAILABLE_CONCEPTS"]
avail["objects"] = [x.name for x in OBJECTS]
avail["actions"] = [x for x in ACTIONS]
avail["properties"] = [avail_to_yaml_dict(k, v) for k, v in PROPERTIES.items()]
avail["functions"] =  [avail_to_yaml_dict(k, v) for k, v in FUNCTIONS.items()]

use = yaml_dict["SELECTION"]
use["objects"] = [x.name for x in OBJECTS]
use["actions"] = [x for x in ACTIONS]
use["properties"] = [proplist_to_yaml_dict(x) for x in PROPERTY_LIST]
use["functions"] = [funclist_to_yaml_dict(x) for x in FUNCTION_LIST]
#print(yaml_dict)

with open("test_focus.yaml", "w") as f:
    yaml.dump(yaml_dict, f, sort_keys=False)

with open("test_focus.yaml", "r") as f:
    in_dict = yaml.safe_load(f)



def validate_objects(objs):
    if not objs:
        return False
    for o in objs:
        if o not in OBJECT_NAMES:
            return False
    return True

def validate_actions(acts):
    if not acts:
        return False
    for o in acts:
        if o not in ACTIONS:
            return False
    return True

def validate_properties(props):
    for p in props:
        if p not in PROPERTIES.keys():
            return False
    return True

def validate_functions(funcs):
    for f in funcs:
        if f not in FUNCTIONS.keys():
            return False
    return True

def validate_properties_signatures(propslist):
    for p in propslist:
        if p[1] not in OBJECT_NAMES:
            raise FocusFileParserError("Unknown object in properties selection: %s" % p[1])
        if p[0] not in PROPERTIES.keys():
            raise FocusFileParserError("Unknown object in properties selection: %s" % p[0])
        prop_definition = PROPERTIES[p[0]]
        o = get_object_by_name(p[1], OBJECTS)
        prop_sig = prop_definition["expects"][0][0].annotation
        if type(o) != prop_sig:
            raise FocusFileParserError("Signature mismatch. Property '%s' expects '%s'" % (p[0], prop_sig))
    return True

def validate_functions_signatures(funclist):
    for f in funclist:
        parsed_para_sig = []
        if f[0] not in FUNCTIONS.keys():
            raise FocusFileParserError("Unknown function in function selection: %s" % f[0])
        for para in f[1]:
            if para[0] not in PROPERTIES.keys():
                raise FocusFileParserError("Unknown property in functions selection: %s" % para[0])
            if para[1] not in OBJECT_NAMES:
                raise FocusFileParserError("Unknown object in functions selection: %s" % para[1])
            prop_definition = PROPERTIES[para[0]]
            o = get_object_by_name(para[1], OBJECTS)
            prop_sig = prop_definition["expects"][0][0].annotation
            parsed_para_sig.append(prop_definition["returns"][0])
            if type(o) != prop_sig:
                raise FocusFileParserError("Signature mismatch in functions selection. Property '%s' expects '%s'" % (para[0], prop_sig))
        func_definition = FUNCTIONS[f[0]]
        function_sig = [x[0].annotation for x in func_definition["expects"]]
        sig_desc = [x[1] for x in func_definition["expects"]]
        if function_sig != parsed_para_sig:
            raise FocusFileParserError("Signature mismatch in functions selection. Function '%s' expects '%s'" % (f[0], sig_desc))
    return True


def import_objects(objs):
    if validate_objects(objs):
        return objs
    else:
        raise FocusFileParserError("Invalid objects specified in objects selection!")

def import_actions(acts):
    if validate_actions(acts):
        return acts
    else:
        raise FocusFileParserError("Invalid actions specified in actions selection!")

def import_properties(props):
    out = []
    for p in props:
        out.append(list(p.items())[0])
    out_list =  list(map(list, out))
    if validate_properties_signatures(out_list):
        return out_list
    else:
        return None

def import_functions(funcs):
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
    if validate_functions_signatures(out):
        return out
    else:
        return None

parsed_env_name = in_dict["ENVIRONMENT"]
sdict = in_dict["SELECTION"]
PARSED_OBJECTS = import_objects(sdict["objects"])
PARSED_ACTIONS = import_actions(sdict["actions"])
PARSED_PROPERTIES = import_properties(sdict["properties"])
PARSED_FUNCTIONS = import_functions(sdict["functions"])


#print(PROPERTIES)
#print(FUNCTIONS)
#print("###")
#print(OBJECT_NAMES)
#print(ACTIONS)
#print(PROPERTY_LIST)
#print(FUNCTION_LIST)
#print("###")
#print(PARSED_OBJECTS)
#print(PARSED_ACTIONS)
#print(PARSED_PROPERTIES)
#print(PARSED_FUNCTIONS)
#print("###")


def get_feature_vector(inc_objects):
    fv = []
    for p in PARSED_PROPERTIES:
        property = p[0]
        obj = p[1]
        prop = PROPERTIES[property]["object"]
        o = get_object_by_name(obj, inc_objects)
        feature = prop(o)
        fv.append(feature)
    for f in PARSED_FUNCTIONS:
        func = f[0]
        input_props = f[1]
        property_results = []
        #print(input_props)
        for p in input_props:
            property = p[0]
            obj = p[1]
            prop = PROPERTIES[property]["object"]
            o = get_object_by_name(obj, inc_objects)
            res = prop(o)
            property_results.append(res)
        f = FUNCTIONS[func]["object"]
        feature = f(*property_results)
        fv.append(feature)
    #print(fv)
    return np.hstack(fv).tolist()


pl = GameObject("player", [1, 2, 3], [1, 1])
en = GameObject("enemy", [4, 5, 6], [2, 2])
ba = GameObject("ball", [7, 8, 9], [3, 3])
for i in range(10):
    testobjs = [pl, en, ba]
    for to in testobjs:
        a = random.randint(0,20)
        b = random.randint(0,20)
        c = random.randint(0,20)
        d = random.randint(0,20)
        e = random.randint(0,20)
        to.update_coords(a,b)
        to.rgb = (c, d, e)
    

    vec = get_feature_vector(testobjs)
    print(vec)