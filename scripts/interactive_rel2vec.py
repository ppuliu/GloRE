import subprocess
import os
import sys

def main(model_dir):
    model_dir = os.path.join(model_dir, 'rel2vec')
    data_dir = 'data'

    def load_config(config_file):
        args = {}
        with open(config_file, 'rb') as f:
            for line in f:
                if len(line) == 0:
                    continue
                param, value = line.strip('\n').split('=')
                param = '--' + param
                args[param] = value
        return args


    interpreter = ["python"]
    script = ["src/run_rel2vec.py"]
    config_file = os.path.join(model_dir, 'config')
    args_dict = load_config(config_file)

    args_dict['--mode'] = 'decode_interactive'

    args_dict['--fresh_start'] = 'False'
    args_dict['--decoder_vocab_size'] = '-1'

    args = []
    for key, value in args_dict.items():
        args = args + [key, value]
    command = interpreter + script + args
    print(args)
    subprocess.call(command)
    
if __name__ == '__main__':    
    main(sys.argv[1])
