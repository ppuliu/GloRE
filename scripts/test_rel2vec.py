import subprocess
import os
import sys

def main(model_dir):
    output_dir = os.path.join(model_dir, 'output')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model_dir = os.path.join(model_dir, 'rel2vec')
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
    script = ["src/run_rel2vec.py"]
    config_file = os.path.join(model_dir, 'config')
    args_dict = load_config(config_file)

    args_dict['--mode'] = 'gen_scores'
    args_dict['--batch_size'] = '1024'

    args_dict['--text_relation_file'] = \
        os.path.join(data_dir, 'train_textual_relation.gz')
    args_dict['--kb_relation_file'] = \
        os.path.join(data_dir, 'kb_relation2id.txt')
    args_dict['--scores_output_file'] = \
        os.path.join(output_dir, 'train_GloRE_scores.txt')
    args_dict['--fill_missing_scores'] = 'False'

    args_dict['--fresh_start'] = 'False'
    args_dict['--decoder_vocab_size'] = '-1'

    print('++++++++++++++++++++++++++++++++++++++++++++++')
    print('Generating GloRE scores for training data')
    print('++++++++++++++++++++++++++++++++++++++++++++++')
    args = []
    for key, value in args_dict.items():
        args = args + [key, value]
    command = interpreter + script + args
    print(args)
    subprocess.call(command)

    print('++++++++++++++++++++++++++++++++++++++++++++++')
    print('Generating GloRE scores for test data')
    print('++++++++++++++++++++++++++++++++++++++++++++++')
    args_dict['--text_relation_file'] = \
        os.path.join(data_dir, 'test_textual_relation.gz')
    args_dict['--scores_output_file'] = \
        os.path.join(output_dir, 'test_GloRE_scores.txt')
    args_dict['--fill_missing_scores'] = 'True'

    args = []
    for key, value in args_dict.items():
        args = args + [key, value]
    command = interpreter + script + args
    print(args)
    subprocess.call(command)
    
if __name__ == '__main__':    
    main(sys.argv[1])
