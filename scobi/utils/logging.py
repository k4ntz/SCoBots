from termcolor import colored

# TODO: replace with python logging, too lazy for now
# TODO: make silent mode ENV class var and parameter
class Logger():
    def __init__(self, silent=False):
        self.SILENT = silent

    def FocusFileParserError(self, msg):
        print(colored("scobi >", "light_red"), "Parser Error: "+msg)
        exit()

    def GeneralInfo(self, msg):
        if self.SILENT:
            return
        print(colored("scobi >", "blue"), msg)

    def GeneralError(self, msg):
        print(colored("scobi >", "light_red"), msg)
        exit()

    def GeneralWarning(self, msg):
        if self.SILENT:
            return
        print(colored("scobi >", "yellow"), msg)