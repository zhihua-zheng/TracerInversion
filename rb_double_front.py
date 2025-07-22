#!/usr/bin/env python3
import argparse
from os import system

parser = argparse.ArgumentParser()
parser.add_argument('--spinup', help='run simulations in spinup mode',
                    action='store_true')
args = parser.parse_args()

casenames = [# front

             # front + convection
             #"d11_M003_Ri020_em01_Q001_W000_D000_St0",

             #"d11_M003_Ri020_em01_Q010_W000_D000_St0",
             #"d11_M003_Ri020_em02_Q010_W000_D000_St0",
             #"d11_M003_Ri020_em03_Q010_W000_D000_St0",
             #"d11_M003_Ri020_em04_Q010_W000_D000_St0",
             #"d11_M003_Ri020_em05_Q010_W000_D000_St0",

             #"d11_M003_Ri040_em01_Q000_W000_D000_St0",
             #"d11_M003_Ri040_em02_Q000_W000_D000_St0",
             "d11_M003_Ri040_em03_Q000_W000_D000_St0",
             "d11_M003_Ri040_em04_Q000_W000_D000_St0",
             "d11_M003_Ri040_em05_Q000_W000_D000_St0",

             #"d11_M003_Ri040_em01_Q001_W000_D000_St0",
             #"d11_M003_Ri040_em02_Q001_W000_D000_St0",
             #"d11_M003_Ri040_em03_Q001_W000_D000_St0",
             #"d11_M003_Ri040_em04_Q001_W000_D000_St0",
             #"d11_M003_Ri040_em05_Q001_W000_D000_St0",

             #"d11_M003_Ri040_em01_Q010_W000_D000_St0",
             #"d11_M003_Ri040_em02_Q010_W000_D000_St0",
             #"d11_M003_Ri040_em03_Q010_W000_D000_St0",
             #"d11_M003_Ri040_em04_Q010_W000_D000_St0",
             #"d11_M003_Ri040_em05_Q010_W000_D000_St0",
            ]

with open('Templates/run_double_front.template', 'r') as f:
    main_script = f.read()
    if not args.spinup:
        main_script = main_script.replace('mem=32GB', 'cpu_type=milan:mem=80GB')
        main_script = main_script.replace('gpu_type=v100', 'gpu_type=a100')

verbose = 1
for casename in casenames:
    if args.spinup:
        main_filename = 'run_double_front_spinup.sh'
        pbs_main = main_script.format(casename, '--spinup')

        with open(main_filename, 'w+') as f:
            f.write(pbs_main)

        cmd_run = f'qsub {main_filename}'
        if verbose: print(cmd_run)
        system(cmd_run)
        print()
    else:
        pre_filename  = 'run_double_front_init_tracer.sh'
        main_filename = 'run_double_front.sh'
        pbs_pre  = main_script.format(casename, '--init_tracer')#--nday 1
        pbs_main = main_script.format(casename, '')

        with open(pre_filename, 'w+') as f:
            f.write(pbs_pre)
        with open(main_filename, 'w+') as f:
            f.write(pbs_main)

        #cmd_run = f'qsub {main_filename}'
        cmd_run = f'JOBID1=$(qsub -h {pre_filename}); qsub -W depend=afterok:$JOBID1 {main_filename}; qrls $JOBID1'
        if verbose: print(cmd_run)
        system(cmd_run)
        print()
