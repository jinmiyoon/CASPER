### Main interface for generating output files.

import os
def span_window(char="-"):
    print(char * os.get_terminal_size().columns)
    return

def print_greeting():
    span_window("#")
    print("\t\tCASPER")
    print("Authors: Devin D. Whitten")
    print("Institute: University of Notre Dame")
    print("Please direct questions to: dwhitten@nd.edu")
    span_window("#")
