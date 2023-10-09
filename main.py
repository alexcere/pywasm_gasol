import sys
from pathlib import Path
from argparse import ArgumentParser, Namespace
import pywasm
# pywasm.on_debug()


def parse_args() -> Namespace:
    ap = ArgumentParser(description='Wasm Block Functional Specification tool')

    input_options = ap.add_argument_group('Input options')
    input_options.add_argument('input_file', help='Wasm file to analyze', action='store')
    input_options.add_argument('-bl', '--block', help='Parses the instructions from plain '
                                                      'representation instead of Wasm', action='store_true', dest='block')

    output_options = ap.add_argument_group('Output options')
    output_options.add_argument('-o', '--output', help="Folder to store blocks' specification",
                                dest="final_folder", action='store')
    output_options.add_argument('-d', '--debug', help='Debug mode enabled. Prints extra information',
                                action='store_true', dest='debug_mode')
    return ap.parse_args()


def initialize(arguments: Namespace) -> None:
    if arguments.final_folder is not None:
        pywasm.global_params.FINAL_FOLDER = Path(arguments.final_folder)
    pywasm.global_params.DEBUG_MODE = arguments.debug_mode


if __name__ == "__main__":
    parsed_args = parse_args()
    initialize(parsed_args)
    if parsed_args.block:
        pywasm.symbolic_exec_from_block(parsed_args.input_file)
    else:
        runtime = pywasm.load(parsed_args.input_file)
        # r = runtime.exec('main', [5,6])
        runtime.all_symbolic_exec()