import os

import jax

import config
import stacked_FC_parallel

if __name__ == "__main__":
    print("CWD:", os.getcwd())
    if config.DEBUG:
        with jax.disable_jit():
            stacked_FC_parallel.main()
    else:
        stacked_FC_parallel.main()
