import os,sys
import argparse
import datetime
import time

sys.path.append('./scripts')
import train_rel2vec
import test_rel2vec
import train_mlp
import test_mlp

if __name__ == '__main__':
    parser = argparse.ArgumentParser("steps.py")
    parser.add_argument("--steps", type=str,
                        default = "1,2,3,4",
                        help="A combination of steps to run (default=1,2,3,4).")
    parser.add_argument("--model_dir", type=str,
                        default = "",
                        help="The model directory. If not given, a directory with current timestamp will be created.")
    
    args = parser.parse_args()
    if args.model_dir == '':
        model_dir = os.path.join('runs', datetime.datetime.now().strftime("%Y_%m_%d"))
    else:
        model_dir = args.model_dir
    
    steps = [False] * 4
    for n in args.steps.split(','):
        step_n = int(n)
        if not 1 <= step_n <= 4:
            raise ValueError('There are only 4 steps [1-4]')
        steps[step_n - 1] = True
    
    if steps[0]:
        print('\n')
        print('==========================================================')
        print('Step 1: Training GloRE model')
        print('(You can proceed to the next step at any moment by pressing CTRL+C)')
        print('==========================================================')
        print('\n')
        time.sleep(3)
        try:
            train_rel2vec.main(model_dir)
        except KeyboardInterrupt:
            print("WARNING: User interrupted program.")
            proceed = raw_input("Do you want to proceed to the next step? [y/n]")
            if proceed != 'n':
                pass
            else:
                sys.exit(-1)
    if steps[1]:
        print('\n')
        print('==========================================================')
        print('Step 2: Generating GloRE scores')
        print('(Do not interrupt until this step is finished)')
        print('==========================================================')
        print('\n')
        time.sleep(2)
        test_rel2vec.main(model_dir)
        
    if steps[2]:
        print('\n')
        print('==========================================================')
        print('Step 3: Training the merge model')
        print('(You can proceed to the next step at any moment by pressing CTRL+C)')
        print('==========================================================')
        print('\n')
        time.sleep(3)
        try:
            train_mlp.main(model_dir)
        except KeyboardInterrupt:
            print("WARNING: User interrupted program.")
            proceed = raw_input("Do you want to proceed to the next step? [y/n]")
            if proceed != 'n':
                pass
            else:
                sys.exit(-1)
    if steps[3]:
        print('\n')
        print('==========================================================')
        print('Step 4: Combining GloRE scores and external scores')
        print('(Do not interrupt until this step is finished)')
        print('==========================================================')
        print('\n')
        time.sleep(2)
        test_mlp.main(model_dir)

        