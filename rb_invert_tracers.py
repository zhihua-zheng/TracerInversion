#!/usr/bin/env python3
from os import system

casenames = [# frontal zone
            # "f11_M003_Q000_W000_D000_St0",
            # "f11_M003_Q000_W000_D000_St0_Ri40",
            # "f11_M001_Q000_W000_D000_St0_Ri1",

             # double front
            # "d11_M003_Ri040_Q000_W000_D000_St0",
             "d11_M003_Ri040_Q001_W000_D000_St0",
             "d11_M003_Ri040_Q010_W000_D000_St0",
            ]

with open('Templates/run_invert_tracers.template', 'r') as f:
    pbs_script = f.read()

verbose = 1
pbs_filename = 'run_invert_tracers.sh'
for casename in casenames:
    pbs_case = pbs_script.format(casename)

    with open(pbs_filename, 'w+') as f:
        f.write(pbs_case)

    cmd_run = f'qsub {pbs_filename}'
    if verbose: print(cmd_run)
    system(cmd_run)
    print()
