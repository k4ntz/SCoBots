import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--interactive", help="Use an interactive color extractor",
                    action="store_true")
parser.add_argument("-ari", help="Use ATARI ANNOTATED RAM module for gym",
                    action="store_true")
parser.add_argument("-r", "--render", help="Renders the agent playing",
                    action="store_true")
parser.add_argument("-g", "--game", help="Game to train on", required=True,
                    action="store", dest="game")
