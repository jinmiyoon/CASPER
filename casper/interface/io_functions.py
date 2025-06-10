### Main interface for generating output files.

import os
def span_window(char="-"):
    print(char * os.get_terminal_size().columns)
    return

def print_greeting():
    span_window("#")
    print("\t\tCASPER")
    print("Authors: Devin D. Whitten and Jinmi Yoon")
    print("Please direct questions to: jinmi.yoon@gmail.com and devin.d.whitten@gmail.com")
    span_window("#")
