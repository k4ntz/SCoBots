import os
import re
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, required=True, help="input python file")
    parser.add_argument("-o", "--output", type=str, required=False, help="output python file name")
    opts = parser.parse_args()
    #TODO: Replace state list with actual state variables like [ball.x-ball.y]

if __name__ == "__main__":
    main()