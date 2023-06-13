'''runner script for training and testing DiscoFlanT5 model'''

import sys
import os

import torch
from transformers import set_seed
# append ../6_flant5/helper_fns to current directory and add to sys.path
sys.path.append(os.path.join(os.getcwd(), 'helper_fns'))
import argparse

from dataset import get_dataset, get_epochs, get_lr
from train import train_the_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    #arguments for organisers
    # dataset args
    parser.add_argument("--dataset_folder", type=str, default="/home/VD/kaveri/sharedtask2021/data/", 
                        help="path to dataset folder")
    parser.add_argument("--dataset_name", type=str, default="eng.pdtb.pdtb",
                        help="dataset name")
    parser.add_argument("--save_model_dir", type=str, default="./runs/",
                        help="model dir to save models")
    # model type args
    parser.add_argument("--refinement", type=str, default="True", required=True)


    # torch behaviour args
    parser.add_argument("--seed", type=int, default=22, required=False,
                        help="random seed")
    parser.add_argument("--cuda", type=int, default=2, 
                        help="cuda device number")
    
    #no need to change these arguments
    #model args
    parser.add_argument("--model_name", type=str, default="google/flan-t5-small", required=False,
                        help="model name")
    parser.add_argument("--max_seq_length", type=int, default=128, required=False,
                        help="max sequence length")
    parser.add_argument("--batch_size", type=int, default=16, required=False,
                        help="batch size")
    parser.add_argument("--num_epochs", type=int, default=2, required=False,
                        help="number of epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-3 , required=False,
                        help="learning rate")
    parser.add_argument("--early_stopping_patience", type=int, default=100, required=False,
                        help="early stopping patience")
    

    args = parser.parse_args()
    
    # extract args from parser
    SEED = args.seed
    device = args.cuda
    device = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")
    shared_folder = args.dataset_folder
    dataset_name = args.dataset_name
    SAVE_MODEL_DIR = os.path.join(args.save_model_dir, dataset_name)
    MODEL_NAME = args.model_name
    max_length = args.max_seq_length
    batch_size = args.batch_size
    early_stopping_patience = args.early_stopping_patience
    refinement = args.refinement
    learning_rate = get_lr(dataset_name, refinement)#args.learning_rate
    num_epochs = get_epochs(dataset_name, refinement)#args.num_epochs

    

    print('****************************************') 
    print('Training......', dataset_name)

    torch.manual_seed(SEED)
    set_seed(SEED)
    train_loader, val_loader, test_loader, num_labels, label_space, majority_class = get_dataset(MODEL_NAME, device, shared_folder, dataset_name, batch_size, max_length)
    train_the_model(num_epochs, train_loader, val_loader, test_loader, device, SAVE_MODEL_DIR, MODEL_NAME, 
                    num_labels, early_stopping_patience, learning_rate, label_space, majority_class, refinement,
                    dataset_name)