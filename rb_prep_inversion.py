#!/usr/bin/env python3
from os import system

casenames = [# frontal zone

             # double front
            # "d11_M003_Ri020_em01_Q001_W000_D000_St0",

            # "d11_M003_Ri020_em01_Q010_W000_D000_St0",
            # "d11_M003_Ri020_em02_Q010_W000_D000_St0",
            # "d11_M003_Ri020_em03_Q010_W000_D000_St0",
            # "d11_M003_Ri020_em04_Q010_W000_D000_St0",
            # "d11_M003_Ri020_em05_Q010_W000_D000_St0",

             "d11_M003_Ri040_em01_Q000_W000_D000_St0",
            # "d11_M003_Ri040_em02_Q000_W000_D000_St0",
            # "d11_M003_Ri040_em03_Q000_W000_D000_St0",
            # "d11_M003_Ri040_em04_Q000_W000_D000_St0",
            # "d11_M003_Ri040_em05_Q000_W000_D000_St0",

            # "d11_M003_Ri040_em01_Q001_W000_D000_St0",
            # "d11_M003_Ri040_em02_Q001_W000_D000_St0",
            # "d11_M003_Ri040_em03_Q001_W000_D000_St0",
            # "d11_M003_Ri040_em04_Q001_W000_D000_St0",
            # "d11_M003_Ri040_em05_Q001_W000_D000_St0",

            # "d11_M003_Ri040_em01_Q010_W000_D000_St0",
            # "d11_M003_Ri040_em02_Q010_W000_D000_St0",
            # "d11_M003_Ri040_em03_Q010_W000_D000_St0",
            # "d11_M003_Ri040_em04_Q010_W000_D000_St0",
            # "d11_M003_Ri040_em05_Q010_W000_D000_St0",
            ]

with open('Templates/run_coarse_graining.template', 'r') as f:
    cflux_script = f.read()
with open('Templates/run_calc_gradients.template', 'r') as f:
    cgrad_script = f.read()

verbose = 1
cflux_filename = 'run_coarse_graining.sh'
cgrad_filename = 'run_calc_gradients.sh'

for casename in casenames:
    pbs_cflux = cflux_script.format(casename)
    pbs_cgrad = cgrad_script.format(casename)

    with open(cflux_filename, 'w+') as f:
        f.write(pbs_cflux)
    with open(cgrad_filename, 'w+') as f:
        f.write(pbs_cgrad)

    #cmd_run = f'qsub {pbs_filename}'
    cmd_run = f'JOBID1=$(qsub -h {cflux_filename}); qsub -W depend=afterok:$JOBID1 {cgrad_filename}; qrls $JOBID1'
    if verbose: print(cmd_run)
    system(cmd_run)
    print()
