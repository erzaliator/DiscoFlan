'''train flant5 model on pdtb dataset'''

import sys
import os
import time
import pandas as pd

from transformers import T5Tokenizer, AutoTokenizer, T5ForConditionalGeneration, T5Config, Adafactor, AutoConfig, AutoModelForSeq2SeqLM
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from torch import Tensor, mean, nn
import torch
import numpy as np
from EarlyStopperUtil import MetricTracker



def predict(model, device, tokenizer, loader, label_max_length):
    """
    Function to evaluate model for predictions

    """
    model.eval()
    predictions = []
    target_outputs = []
    orig_inputs = []
    with torch.no_grad():
        for i, (input_ids, attention_masks, label_input_ids, label_attention_masks) in enumerate(loader):
            y = label_input_ids.to(device, dtype = torch.long)
            ids = input_ids.to(device, dtype = torch.long)
            mask = attention_masks.to(device, dtype = torch.long)

            generated_ids = model.generate(
            input_ids=ids,
            attention_mask=mask,
            max_length=16,
            num_beams=3,
            early_stopping=True
            )
            preds = [tokenizer.decode(
                g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in generated_ids]
            targs = [tokenizer.decode(
                t, skip_special_tokens=True, clean_up_tokenization_spaces=False) for t in y]
            inputs = [tokenizer.decode(
                inp, skip_special_tokens=True, clean_up_tokenization_spaces=False) for inp in ids]

            if i % 10==0:
                print(f'Completed {i}\n')
                assert len(preds) == len(targs) == len(inputs)
                print('target\tpredicted\tinput\n')
                for pred, target, inp in zip(preds, targs, inputs):
                    print(f'{target}\t{pred}\t{inp}\n')
            predictions.extend(preds)
            target_outputs.extend(targs)
            orig_inputs.extend(inputs)

    return predictions, target_outputs, orig_inputs, accuracy_score(target_outputs, predictions)

def train_with_api(model, device, tokenizer,
          train_loader, train_epochs,
          optimizer, output_dir, save_every_n_epochs):
    
    model.train()
    for epoch in range(train_epochs):
        for step, (input_ids, attention_masks, label_input_ids, label_attention_masks) in enumerate(train_loader):
            label_input_ids[label_input_ids == tokenizer.pad_token_id] = -100

            outputs = model(input_ids=input_ids, attention_mask=attention_masks, labels=label_input_ids)

            #get loss and logits
            loss = outputs[0]

            if step % 100 == 0:
                print(f"Epoch {epoch}, Step: {step}, Loss: {loss}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if save_every_n_epochs > 0:
            if epoch % save_every_n_epochs == 0:
                path = os.path.join(output_dir, f"model_files_ep{epoch}")
                log_path = os.path.join(output_dir, f"ep-{epoch}.log")
                model.save_pretrained(path)
                tokenizer.save_pretrained(path)
                print(f"""[Model] Model saved @ {path}\n""")

def T5Trainer(train_loader, val_loader, test_loader,
              output_dir,
              model,
              tokenizer,
              train_epochs,
              learning_rate,
              device,
              max_source_text_length,
              max_target_text_length,
              save_every=None):
    """
    T5 trainer

    """
    print(f'Device: {device}')
    model = model.to(device)

    print("[Data]: Reading data...\n")

    optimizer = Adafactor(params=model.parameters(), lr=learning_rate, relative_step=False)

    # Training loop
    print('[Initiating finetuning]...\n')

    train_with_api(model, device, tokenizer, train_loader, train_epochs, optimizer,
          output_dir, save_every_n_epochs=save_every)

    print(f'[Finished finetuning after {train_epochs} epochs.]')

    if train_epochs > 0:
        save_path = os.path.join(output_dir, "model_files")
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        print(f"""[Model] Model saved @ {os.path.join(output_dir, "model_files")}\n""")

    print("[Predicting with final checkpoint]...\n")

    loaders = {
        'train': train_loader,
        'dev': val_loader,
        'test': test_loader,
    }

    print(f"[Generating predictions on test...]\n")
    predictions, targets, orig_inputs, accuracy = predict(
        model, device, tokenizer, test_loader, max_target_text_length)
    final_df = pd.DataFrame({'target': targets, 'prediction': predictions,'input': orig_inputs})
    os.makedirs(output_dir, exist_ok=True)
    final_df.to_csv(os.path.join(output_dir, f'predictions_test.tsv'), sep='\t')
    print(f"""[Prediction accuracy] {accuracy}""")
    print(f"""Prediction data saved @ {os.path.join(output_dir)}\n""")

    print("[Prediction Completed.]\n")


def train_the_model(num_epochs, train_loader, val_loader, test_loader, device, SAVE_MODEL_DIR, MODEL_NAME, num_labels, early_stopping_patience, lr, 
                    label_space, majority_class, refinement, dataset_name):
    if SAVE_MODEL_DIR is not None:
        if not os.path.exists(SAVE_MODEL_DIR):
            os.makedirs(SAVE_MODEL_DIR)
        else:
            try:
                os.remove(os.path.join(SAVE_MODEL_DIR, 'predictions.csv'))
                os.remove(os.path.join(SAVE_MODEL_DIR, 'result.txt'))
            except:
                pass

    print('*****************Hyperparameters***********************')
    print('learning rate: ', lr)
    print('force_words_ids: ', label_space)

    # Load the T5 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if 'pdtb' in dataset_name:
        label_max_length = 25
        label_min_length = 3

    elif 'dep' in dataset_name:
        label_max_length = 3
        label_min_length = 2

    else:
        label_max_length = 3
        label_min_length = 2


    # Get model parameters
    config = AutoConfig.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_config(config)

    T5Trainer(train_loader, val_loader, test_loader,
              SAVE_MODEL_DIR,
              model,
              tokenizer,
              num_epochs,
              lr,
              device,
              1024,
              label_max_length,
              save_every=10)

    # # Define the T5 model architecture
    # class T5Classifier(nn.Module):
    #     def __init__(self):
    #         super().__init__()
    #         self.num_labels = num_labels
    #         self.label_space = tokenizer(label_space, add_special_tokens=False).input_ids
    #         self.bad_words_ids = tokenizer([' '], add_special_tokens=False).input_ids
    #         config = T5Config.from_pretrained(MODEL_NAME)
    #         self.t5 = T5ForConditionalGeneration.from_pretrained(MODEL_NAME, num_labels=self.num_labels)
    #         self.t5.config.update({"num_beams": 4})
    #         self.t5 = self.t5.to(device)

    #     def forward(self, input_ids, attention_masks, label_input_ids, label_attention_masks):
    #         outputs = self.t5(
    #                             input_ids=input_ids, 
    #                             attention_mask=attention_masks,
    #                             labels=label_input_ids,
    #                             decoder_attention_mask=label_attention_masks,
    #                             )
    #         loss = outputs.loss
    #         logits = outputs.logits
    #         return loss, logits
        
    #     def generator(self, input_ids, attention_mask):
    #         generated_ids = self.t5.generate(
    #                                         input_ids=input_ids,
    #                                         attention_mask=attention_mask,
    #                                         max_length = label_max_length,
    #                                         min_length = label_min_length,
    #                                         force_words_ids=self.label_space,
    #                                         )
    #         return generated_ids

    # # Create the model instance
    # model = T5Classifier()
    # optimizer = torch.optim.Adam(list(model.parameters()), lr=lr)
    # # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.1, verbose=True)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

    # def train(model, device, tokenizer,
    #       train_loader, train_epochs,
    #       optimizer, output_dir, save_every_n_epochs):

    # model.train()

    # #refine predctions. obtain the word that lies in label space and replace the label with that word.
    # def refine_predictions(preds, label_space, majority_class, mismatch, complete_mismatch):
    #     refined_preds = []
    #     for pred in preds:
    #         pred = pred.split()
    #         refined_pred = '-1'
    #         for word in pred:
    #             if word in label_space:
    #                 refined_pred = word
    #             elif word[:-1] in label_space:
    #                 refined_pred = word[:-1]
    #                 mismatch += 1
    #             elif word[:-2] in label_space:
    #                 refined_pred = word[:-2]
    #                 mismatch += 1
    #             elif word[:-3] in label_space:
    #                 refined_pred = word[:-3]
    #                 mismatch += 1
    #         if refined_pred == '-1':
    #             refined_pred = majority_class
    #             complete_mismatch += 1

    #         refined_preds.append(refined_pred)
    #     if refinement=='False':
    #         refined_preds = preds
    #     return refined_preds, mismatch, complete_mismatch


    # # Define the metrics for evaluation
    # def accuracy(logits, input_ids, label_input_ids, label_attention_masks, mismatch, complete_mismatch, report=False):

    #     generated_ids = model.generator(
    #             input_ids=input_ids,
    #             attention_mask=label_attention_masks,
    #             )

    #     raw_preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
    #     preds, mismatch, complete_mismatch = refine_predictions(raw_preds, label_space, majority_class, mismatch, complete_mismatch)
    #     target = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True) for t in label_input_ids]
    #     print('preds: ', preds, ' | target: ', target)
        
    #     if report: 
    #         with open(os.path.join(SAVE_MODEL_DIR, 'predictions.csv'), 'a') as f:
    #             for rp, p, t in zip(raw_preds, preds, target):
    #                 rp = str([rp])
    #                 p = str([p])
    #                 t = str([t])
    #                 f.write(rp+','+p+','+t+'\n')
    #     # print(mismatch, complete_mismatch)
    #     return accuracy_score(target, preds), [classification_report(target, preds, zero_division=0), mismatch, complete_mismatch, preds]
    

    # # define train function
    # def train(model, train_loader, val_loader, optimizer, scheduler):
    #     EarlyStopper = MetricTracker(patience=early_stopping_patience, metric_name='+accuracy')
    #     best_val_acc = 0
    #     mismatch_list = []
    #     complete_mismatch_list = []

    #     for epoch in range(num_epochs):
    #         start = time.time()
    #         model.train()
    #         total_train_loss = 0
    #         total_train_acc  = 0
            
    #         # logging for scheduler
    #         losses = []
    #         accuracies= []

    #         mismatch = 0
    #         complete_mismatch = 0

    #         for batch_idx, (input_ids, attention_masks, label_input_ids, label_attention_masks) in enumerate(train_loader):
    #             optimizer.zero_grad()
    #             loss, logits = model(input_ids, attention_masks, label_input_ids, label_attention_masks)
    #             acc, _ = accuracy(logits, input_ids, label_input_ids, label_attention_masks, mismatch, complete_mismatch, report=False)
    #             criterion = nn.CrossEntropyLoss()
    #             loss.backward()
    #             optimizer.step()
    #             total_train_loss += loss.item()
    #             total_train_acc  += acc.item()

    #             losses.append(loss)
    #             accuracies.append(acc)
    #             mismatch_list.append(mismatch)
    #             complete_mismatch_list.append(complete_mismatch)
            
    #         mean_loss = sum(losses)/len(losses)
    #         scheduler.step(mean_loss)

    #         train_acc  = total_train_acc/len(train_loader)
    #         train_loss = total_train_loss/len(train_loader)

    #         val_acc, val_loss, report = evaluate_accuracy(model, val_loader)
    #         mismatch += report[1]
    #         complete_mismatch += report[2]
    #         if val_acc>=best_val_acc:
    #             best_val_acc = val_acc
    #             torch.save(model.state_dict(), SAVE_MODEL_DIR+'/best.pt')
    #             print(f'Epoch {epoch+1}: Best val_acc: {best_val_acc:.4f}')

    #         end = time.time()
    #         hours, rem = divmod(end-start, 3600)
    #         minutes, seconds = divmod(rem, 60)

    #         print(f'Epoch {epoch+1}: train_loss: {train_loss:.4f} train_acc: {train_acc:.4f} | val_loss: {val_loss:.4f} val_acc: {val_acc:.4f}')
    #         print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))

    #         EarlyStopper.add_metric(val_acc)
    #         if EarlyStopper.should_stop_early(): break

    #     # print(mismatch_list)
    #     # print(complete_mismatch_list)

    # def write_preds(input_ids, preds):
    #     with open(os.path.join(SAVE_MODEL_DIR, 'task_preds.csv'), 'a') as f:
    #         for i, p in zip(input_ids, preds):
    #             i = tokenizer.decode(i, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    #             # i = str([i])
    #             # p = str([p])
    #             f.write(i+'\t'+p+'\n')

    # # Define the evaluation function
    # def evaluate_accuracy(model, val_loader, test=False):

    #     if test:
    #         with open(os.path.join(SAVE_MODEL_DIR, 'task_preds.csv'), 'w') as f:
    #             f.write('Input'+'\t'+'Predictions'+'\n')
                    
    #     model.eval()
    #     total_val_loss = 0
    #     total_val_acc  = 0
    #     losses = []
    #     accuracies= []
    #     mismatch = 0
    #     complete_mismatch = 0
    #     with torch.no_grad():
    #         for batch_idx, (input_ids, attention_masks, label_input_ids, label_attention_masks) in enumerate(val_loader):
    #             loss, logits = model(input_ids, attention_masks, label_input_ids, label_attention_masks)
    #             criterion = nn.CrossEntropyLoss()
    #             acc, report = accuracy(logits, input_ids, label_input_ids, label_attention_masks, mismatch, complete_mismatch, report=True)
    #             mismatch += report[1]
    #             complete_mismatch += report[2]
    #             preds = report[3]
    #             write_preds(input_ids, preds)
    #             total_val_loss += loss.item()
    #             total_val_acc  += acc.item()
            
    #             losses.append(loss)
    #             accuracies.append(acc)
    #         val_acc  = total_val_acc/len(val_loader)
    #         val_loss = total_val_loss/len(val_loader)
    #         if test==False:
    #             print(f'val_loss: {val_loss:.4f} val_acc: {val_acc:.4f} mismatch: {mismatch} complete_mismatch: {complete_mismatch}')
    #     return val_acc, val_loss, [report, mismatch, complete_mismatch]

    # # Train the model
    # train(model, train_loader, val_loader, optimizer, scheduler)

    # #Test the model
    # model.load_state_dict(torch.load(SAVE_MODEL_DIR+'/best.pt'))

    # #write header "pred", "target" to prediction.csv
    # with open(os.path.join(SAVE_MODEL_DIR, 'predictions.csv'), 'w') as f:
    #     f.write('rawpredictions, predictions, targets\n')

    # test_acc, test_loss, report = evaluate_accuracy(model, test_loader, test=True)
    # mismatch = report[1]
    # complete_mismatch = report[2]
    # report = report[0][0]
    # print(f'Test loss: {test_loss:.4f} Test accuracy: {test_acc:.4f} mismatch: {mismatch} complete_mismatch: {complete_mismatch}')

    # print([report])
    
    # with open(os.path.join(SAVE_MODEL_DIR, 'results.txt'), 'w') as f:
    #     f.write(report)