#!/usr/bin/env python3
from os import system

casenames = [# frontal zone
            # "f11_M003_Q000_W000_D000_St0",
            # "f11_M003_Q000_W000_D000_St0_Ri40",
            # "f11_M001_Q000_W000_D000_St0_Ri1",
            # "f11_M010_Q000_W000_D000_St0_Ri1",

             # channel
              "c11_M010_Q000_W000_D000_St0_Ri10",
             #"c11_M001_Q000_W000_D000_St0_Ri1000",


             # wind + wave
            # "n11_Q000_W037_D000_St1",
            # "n11_Q000_W009_D000_St1",

             # wave
            # "n11_Q000_W000_D000_St1",
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
