import subprocess
import os
import sys

def main(model_dir):
    model_dir = os.path.join(model_dir, 'rel2vec')
    data_dir = 'data'
    
    interpreter = ["python"]
    script = ["src/run_rel2vec.py"]
    args = [
        # use mode
        "--mode", "train",

        # data configuration
        "--fresh_start", "True",
        "--data_dir", data_dir,
        "--model_dir", model_dir,
        "--encoder_vocab_size", "20000", 
        "--decoder_vocab_size", "-1",

        # learning configuration
        "--loss_choice", "GloRE",
        "--optimization_algorithm", "adam",
        "--learning_rate", "0.5",
        "--learning_rate_decay_factor", "0.95",
        "--max_gradient_norm", "5.0",
        "--batch_size", "128",
        "--steps_per_checkpoint", "300",
        "--steps_per_summary", "300",
        "--maximum_steps", "100000000",
        "--summarize_trainable_variables", "False",
        "--do_validation", "True",
        "--early_stop_tolerance", "100000",

        # model configuration
        "--num_layers", "1",
        "--size", "300",
        "--embedding_size", "300",
        "--use_lstm", "False",
        "--train_embedding", "True",
        "--use_attention", "False",
        "--word2vec_normalization", "None",
        "--use_word2vec", "True",  
        "--encoder_embedding_file", "left.20000.word2vec.vocab.npy",
        "--decoder_embedding_file", "None",
        "--input_keep_prob", "1.0",
        "--output_keep_prob", "1.0",
    ]
    command = interpreter + script + args
    subprocess.call(command)

if __name__ == '__main__':    
    main(sys.argv[1])
