"""Dead one-off script to split Warp kernels out of ``proxy_coupling.py``.

Not imported by the package or runtime API. Kept for reference only.
"""

import re

with open('proxy_coupling.py', 'r') as f:
    lines = f.readlines()

kernels_file = []
proxy_file = []

in_kernel = False

kernels_file.append('"""Warp kernels for M1 proxy coupling."""\n\n')
kernels_file.append('import warp as wp\n\n')

proxy_file.append(lines[0]) # """M1 proxy coupling...
proxy_file.append(lines[1])
proxy_file.append(lines[2])
proxy_file.append(lines[3])
proxy_file.append(lines[4])
proxy_file.append(lines[5])
proxy_file.append(lines[6])
proxy_file.append(lines[7])
proxy_file.append(lines[8])

proxy_file.append('from __future__ import annotations\n\n')
proxy_file.append('import dataclasses\n')
proxy_file.append('from collections.abc import Sequence\n')
proxy_file.append('from typing import Any\n\n')
proxy_file.append('import numpy as np\n')
proxy_file.append('import warp as wp\n\n')
proxy_file.append('from apple_pick_sim.coupled_fruiting.kernels import (\n')
proxy_file.append('    _align_body_q_prev_kernel,\n')
proxy_file.append('    _copy_body_state_kernel,\n')
proxy_file.append('    _limit_and_write_tcp_stem_wrench_kernel,\n')
proxy_file.append('    _zero_all_wrenches_kernel,\n')
proxy_file.append('    _zero_wrench_slots_kernel,\n')
proxy_file.append('    compute_proxy_reaction_wrench_kernel,\n')
proxy_file.append('    mirror_robot_tcp_to_proxy_and_apple_kernel,\n')
proxy_file.append('    mirror_robot_tcp_to_proxy_kernel,\n')
proxy_file.append('    mirror_robot_tcp_to_proxy_offset_kernel,\n')
proxy_file.append(')\n\n')

kernel_names = [
    '_corrected_proxy_twist_from_robot',
    'mirror_robot_tcp_to_proxy_kernel',
    'mirror_robot_tcp_to_proxy_offset_kernel',
    'compute_proxy_reaction_wrench_kernel',
    '_zero_wrench_slots_kernel',
    '_zero_all_wrenches_kernel',
    '_limit_and_write_tcp_stem_wrench_kernel',
    '_copy_body_state_kernel',
    '_align_body_q_prev_kernel',
    'mirror_robot_tcp_to_proxy_and_apple_kernel'
]

skip = False
for i in range(19, len(lines)): # start after imports
    line = lines[i]
    if line.startswith('@wp.func') or line.startswith('@wp.kernel'):
        in_kernel = True
        kernels_file.append(line)
        continue
    
    if in_kernel:
        kernels_file.append(line)
        # Check if the kernel block has ended by looking for next top-level def or empty lines ending the block.
        # Actually, python functions end when the next line is unindented (and not empty/comment).
        # Let's peek ahead or just check if line is empty.
        pass # we handle end of kernel differently
    else:
        proxy_file.append(line)

# Let's use a better parsing strategy: read blocks based on top-level definitions.
