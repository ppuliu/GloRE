import subprocess
import os
import sys

def main(model_dir):
    output_dir = os.path.join(model_dir, 'output')
    model_dir = os.path.join(model_dir, 'merge')
    data_dir = 'data'
    
    interpreter = ["python"]
    script = ["src/run_mlp.py"]

    args = [
        "--mode", "train",
        "--create_input_files", "True",
        "--do_evaluation", "True",  
        "--use_cap", "True",
        "--use_gate", "False",
        "--normalize_data", "False",

        "--log_dir", model_dir,
        "--emb_score_file", os.path.join(output_dir, 'train_GloRE_scores.txt'),

        "--sizes", "",
        "--batch_size", "1024",
        "--optimizer_type", "adam",
        "--l2_penalty", "0.0",
        "--learning_rate", "1.0",
        "--dev_percentage", "0.1",
        "--eval_every", "1000",
        "--max_steps", "1000000",
        "--patience", "100",
        "--early_stop", "False",

        "--moana_file", os.path.join(data_dir, 'train_pcnn_att_scores.gz'),
        "--kb_relation_file", os.path.join(data_dir, 'kb_relation2id.txt'),
        "--features_file", os.path.join(output_dir, 'train_features.txt'),
        "--targets_file", os.path.join(output_dir, 'train_targets.txt'),
    ]
    command = interpreter + script + args
    print ' '.join(command)
    subprocess.call(command)
    
if __name__ == '__main__':
    main(sys.argv[1])
