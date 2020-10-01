import os

from jax.config import config as j_config

j_config.update("jax_enable_x64", True)
# j_config.update("jax_disable_jit", False)
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
