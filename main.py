import sys
import pywasm
# pywasm.on_debug()

if __name__ == "__main__":
    runtime = pywasm.load(sys.argv[1])
    # r = runtime.exec('main', [5,6])
    runtime.all_symbolic_exec()