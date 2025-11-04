#!/usr/bin/env python3
from os import system

casenames = [# double front
             # "d11_M003_Ri020_em01_Q000_W000_D000_St0",
             # "d11_M003_Ri020_em01_Q001_W000_D000_St0",
             # "d11_M003_Ri020_em01_Q004_W000_D000_St0",

              "d11_M006_Ri020_em01_Q000_W000_D000_St0",
              "d11_M006_Ri020_em01_Q001_W000_D000_St0",
              "d11_M006_Ri020_em01_Q004_W000_D000_St0",
            ]

with open('Templates/run_fske_budget.template', 'r') as f:
    pbs_script = f.read()

verbose = 1
pbs_filename = 'run_fske_budget.sh'
for casename in casenames:
    pbs_case = pbs_script.format(casename + '_init-tracer-fluxed')

    with open(pbs_filename, 'w+') as f:
        f.write(pbs_case)

    cmd_run = f'qsub {pbs_filename}'
    if verbose: print(cmd_run)
    system(cmd_run)
    print()
