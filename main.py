import sys
import os
from pathlib import Path
from argparse import ArgumentParser, Namespace
import pywasm
# pywasm.on_debug()
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/evmopt/evmx")


def options_wasm(ap: ArgumentParser) -> None:
    input_options = ap.add_argument_group('Input options')
    group_input = input_options.add_mutually_exclusive_group()
    input_options.add_argument('input_file', help='Wasm file to analyze', action='store')
    group_input.add_argument('-bl', '--block', help='Parses the instructions from plain '
                                                      'representation instead of Wasm', action='store_true', dest='block')
    group_input.add_argument('-sfs', '--sfs', help='Uses the SFS representation from a JSON file',
                             action='store_true', dest='sfs')
    group_input.add_argument('-info', action='store_true', dest='info')

    output_options = ap.add_argument_group('Output options')
    output_options.add_argument('-o', '--output', help="Folder to store blocks' specification",
                                dest="final_folder", action='store')
    output_options.add_argument('-d', '--debug', help='Debug mode enabled. Prints extra information',
                                action='store_true', dest='debug_mode')
    output_options.add_argument('-c', '--csv', help="Folder to store blocks' specification",
                                dest="csv_file", action='store')

    optimization_options = ap.add_argument_group('Optimization options')
    group_opt = input_options.add_mutually_exclusive_group()

    group_opt.add_argument('-g', '--greedy', help='Enable greedy alone', action='store_true',
                           dest='greedy')
    group_opt.add_argument('-ub', '--ub-greedy', help='Enable using the bound from the greedy algorithm in '
                                                      'the optimization process', action='store_true', dest='ub_greedy')
    group_opt.add_argument('-usfs', '--ub-sfs', help='Compute an upper bound using the greedy algorithm and'
                                                     'stores the corresponding JSON file', action='store_true', dest='ub_sfs')

    optimization_options.add_argument('-sp', '--split', help='Split large sequences to ensure they have at most SPLIT instructions',
                                      action='store', type=int, dest='split', default=-1)

    sat_options = ap.add_argument_group('SAT Options', 'Options for enabling flags in SAT')
    sat_options.add_argument('-sat-d', '--sat-dominance', action='store', dest='config_sat',
                             choices=['all', 'base', 'e', 'f', 'g', 'h', 'allbutf'], default='all')
    sat_options.add_argument('-ext', '--ext', action='store_true', dest='external')


def parse_args_wasm(ap: ArgumentParser) -> Namespace:
    return ap.parse_args()


def initialize(arguments: Namespace) -> None:

    if arguments.final_folder is not None:
        pywasm.global_params.FINAL_FOLDER = Path(arguments.final_folder)
        Path.mkdir(pywasm.global_params.FINAL_FOLDER, exist_ok=True, parents=True)

    if arguments.csv_file is not None:
        pywasm.global_params.CSV_FILE = arguments.csv_file
    else:
        pywasm.global_params.CSV_FILE = pywasm.global_params.FINAL_FOLDER.joinpath("statistics.csv")

    if arguments.greedy:
        pywasm.global_params.OPTIMIZER = "greedy"

    if arguments.ub_greedy:
        pywasm.global_params.GREEDY_BOUND = True

    pywasm.global_params.DEBUG_MODE = arguments.debug_mode
    pywasm.global_params.CONFIG_SAT = arguments.config_sat
    pywasm.global_params.EXTERNAL_SOLVER = arguments.external
    pywasm.global_params.UB_GREEDY = arguments.ub_greedy
    pywasm.global_params.SPLIT_BLOCK = arguments.split
    pywasm.global_params.UB_SFS = arguments.ub_sfs
    pywasm.global_params.INFO = arguments.info


def main(parsed_args: Namespace) -> None:
    if parsed_args.block:
        pywasm.symbolic_exec_from_block(parsed_args.input_file)
    elif parsed_args.sfs:
        pywasm.symbolic_exec_from_sfs(parsed_args.input_file)
    else:
        runtime = pywasm.load(parsed_args.input_file)
        # r = runtime.exec('main', [5,6])
        runtime.all_symbolic_exec()


if __name__ == "__main__":
    ap = ArgumentParser(description='Wasm Block Functional Specification tool')
    options_wasm(ap)
    parsed_args = parse_args_wasm(ap)
    initialize(parsed_args)
    main(parsed_args)