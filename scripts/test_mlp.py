import subprocess
import os
import sys

def main(model_dir):
    output_dir = os.path.join(model_dir, 'output')
    model_dir = os.path.join(model_dir, 'merge')
    data_dir = 'data'
    
    def load_config(config_file):
        args = {}
        with open(config_file) as f:
            for line in f:
                if len(line) == 0:
                    continue
                param, value = line.strip('\n').split('=')
                param = '--' + param
                args[param] = value
        return args


    interpreter = ["python"]
    script = ["src/run_mlp.py"]
    log_dir = model_dir
    config_file = os.path.join(log_dir, 'config')
    args_dict = load_config(config_file)

    args_dict['--mode'] = 'test'
    args_dict['--create_input_files'] =  'True'
    args_dict['--add_meta_info'] = 'True'

    args_dict['--moana_file'] = os.path.join(data_dir, 'test_pcnn_att_scores.gz')
    args_dict['--kb_relation_file'] =  os.path.join(data_dir, 'kb_relation2id.txt')
    args_dict['--emb_score_file'] = os.path.join(output_dir, 'test_GloRE_scores.txt')
    args_dict['--features_file'] =  os.path.join(output_dir, 'test_features.txt')
    args_dict['--targets_file'] =  os.path.join(output_dir, 'test_targets.txt')
    args_dict['--our_pr_file'] =  os.path.join(output_dir, 'GloRE_pr.txt')
    args_dict['--their_pr_file'] =   os.path.join(output_dir, 'PCNN+ATT_pr.txt')
    args_dict['--output_n'] =  '2000'

    args = []
    for key, value in args_dict.items():
        args = args + [key, value]
    command = interpreter + script + args
    print(args)
    subprocess.call(command)
    
if __name__ == '__main__':
    main(sys.argv[1])
