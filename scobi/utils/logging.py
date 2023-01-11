from termcolor import colored

SILENT = False

# TODO: replace with python logging, too lazy for now
# TODO: make silent mode ENV class var and parameter
def FocusFileParserError(msg):
    print(colored("scobi >", "light_red"), "Parser Error: "+msg)
    exit()

def GeneralInfo(msg):
    if SILENT:
        return
    print(colored("scobi >", "blue"), msg)

def GeneralError(msg):
    print(colored("scobi >", "light_red"), msg)
    exit()

def GeneralWarning(msg):
    if SILENT:
        return
    print(colored("scobi >", "yellow"), msg)