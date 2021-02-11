import os

from jax.config import config as j_config

j_config.update("jax_enable_x64", True)
# j_config.update("jax_disable_jit", False)
import jax

import config
import train

from jax.lib import xla_bridge

if __name__ == "__main__":
    print("CWD:", os.getcwd())
    if config.DEBUG:
        with jax.disable_jit():
            train.main()
    else:
        # if xla_bridge.get_backend().platform == "cpu":
        #     raise ValueError("Need CPU")
        train.main()
