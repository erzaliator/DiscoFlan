'''train flant5 model on pdtb dataset'''

import sys
import os
import time

from transformers import T5Tokenizer, AutoTokenizer, T5ForConditionalGeneration, T5Config
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from torch import Tensor, mean, nn
import torch
from EarlyStopperUtil import MetricTracker


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

    


    # Define the T5 model architecture
    class T5Classifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.num_labels = num_labels
            config = T5Config.from_pretrained(MODEL_NAME)
            self.t5 = T5ForConditionalGeneration.from_pretrained(MODEL_NAME, num_labels=self.num_labels)#, torch_dtype=torch.float16)
            self.t5.config.update({"num_beams": 4})
            # self.t5.config.update({"max_length": label_max_length})
            # self.t5.config.update({"min_length": label_min_length})
            # self.t5.config.update({"penalty_alpha": 0.9})
            # self.t5.config.update({"do_sample": True})
            # self.t5.config.update({"temperature": 2.5})
            self.t5 = self.t5.to(device)

        def forward(self, input_ids, attention_masks, label_input_ids, label_attention_masks):
            outputs = self.t5(
                                input_ids=input_ids, 
                                attention_mask=attention_masks,
                                labels=label_input_ids,
                                decoder_attention_mask=label_attention_masks)
            loss = outputs.loss
            logits = outputs.logits
            return loss, logits
        
        def generator(self, input_ids, attention_mask):
            generated_ids = self.t5.generate(
                                            input_ids=input_ids,
                                            attention_mask=attention_mask,
                                            max_length = label_max_length,
                                            min_length = label_min_length
                                            # do_sample = True
                                            # length_penalty = 5,
                                            # early_stopping = True
                                            )
            return generated_ids

    # Create the model instance
    model = T5Classifier()
    optimizer = torch.optim.Adam(list(model.parameters()), lr=lr)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.1, verbose=True)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

    # Define the loss function
    def cross_entropy_loss(logits, labels):
        return nn.log_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    
    #refine predctions. obtain the word that lies in label space and replace the label with that word. 
    def refine_predictions(preds, label_space, majority_class, mismatch, complete_mismatch):
        refined_preds = []
        for pred in preds:
            pred = pred.split()
            refined_pred = '-1'
            for word in pred:
                if word in label_space:
                    refined_pred = word
                elif word[:-1] in label_space:
                    refined_pred = word[:-1]
                    mismatch += 1
                elif word[:-2] in label_space:
                    refined_pred = word[:-2]
                    mismatch += 1
                elif word[:-3] in label_space:
                    refined_pred = word[:-3]
                    mismatch += 1
            if refined_pred == '-1':
                refined_pred = majority_class
                complete_mismatch += 1

            refined_preds.append(refined_pred)
        if refinement=='False':
            refined_preds = preds
        return refined_preds, mismatch, complete_mismatch


    # Define the metrics for evaluation
    def accuracy(logits, input_ids, label_input_ids, label_attention_masks, mismatch, complete_mismatch, report=False):

        generated_ids = model.generator(
                input_ids=input_ids,
                attention_mask=label_attention_masks,
                )

        raw_preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
        preds, mismatch, complete_mismatch = refine_predictions(raw_preds, label_space, majority_class, mismatch, complete_mismatch)
        target = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True) for t in label_input_ids]
        # print('preds: ', preds, ' | target: ', target)
        
        if report: 
            with open(os.path.join(SAVE_MODEL_DIR, 'predictions.csv'), 'a') as f:
                for rp, p, t in zip(raw_preds, preds, target):
                    rp = str([rp])
                    p = str([p])
                    t = str([t])
                    f.write(rp+','+p+','+t+'\n')
        # print(mismatch, complete_mismatch)
        return accuracy_score(target, preds), [classification_report(target, preds, zero_division=0), mismatch, complete_mismatch, preds]
    

    # define train function
    def train(model, train_loader, val_loader, optimizer, scheduler):
        EarlyStopper = MetricTracker(patience=early_stopping_patience, metric_name='+accuracy')
        best_val_acc = 0
        mismatch_list = []
        complete_mismatch_list = []

        for epoch in range(num_epochs):
            start = time.time()
            model.train()
            total_train_loss = 0
            total_train_acc  = 0
            
            # logging for scheduler
            losses = []
            accuracies= []

            mismatch = 0
            complete_mismatch = 0

            for batch_idx, (input_ids, attention_masks, label_input_ids, label_attention_masks) in enumerate(train_loader):
                optimizer.zero_grad()
                loss, logits = model(input_ids, attention_masks, label_input_ids, label_attention_masks)
                acc, _ = accuracy(logits, input_ids, label_input_ids, label_attention_masks, mismatch, complete_mismatch, report=False)
                criterion = nn.CrossEntropyLoss()
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()
                total_train_acc  += acc.item()

                losses.append(loss)
                accuracies.append(acc)
                mismatch_list.append(mismatch)
                complete_mismatch_list.append(complete_mismatch)
            
            mean_loss = sum(losses)/len(losses)
            scheduler.step(mean_loss)

            train_acc  = total_train_acc/len(train_loader)
            train_loss = total_train_loss/len(train_loader)

            val_acc, val_loss, report = evaluate_accuracy(model, val_loader)
            mismatch += report[1]
            complete_mismatch += report[2]
            if val_acc>=best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), SAVE_MODEL_DIR+'/best.pt')
                print(f'Epoch {epoch+1}: Best val_acc: {best_val_acc:.4f}')

            end = time.time()
            hours, rem = divmod(end-start, 3600)
            minutes, seconds = divmod(rem, 60)

            print(f'Epoch {epoch+1}: train_loss: {train_loss:.4f} train_acc: {train_acc:.4f} | val_loss: {val_loss:.4f} val_acc: {val_acc:.4f}')
            print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))

            EarlyStopper.add_metric(val_acc)
            if EarlyStopper.should_stop_early(): break

        # print(mismatch_list)
        # print(complete_mismatch_list)

    def write_preds(input_ids, preds):
        with open(os.path.join(SAVE_MODEL_DIR, 'task_preds.csv'), 'a') as f:
            for i, p in zip(input_ids, preds):
                i = tokenizer.decode(i, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                # i = str([i])
                # p = str([p])
                f.write(i+'\t'+p+'\n')

    # Define the evaluation function
    def evaluate_accuracy(model, val_loader, test=False):

        if test:
            with open(os.path.join(SAVE_MODEL_DIR, 'task_preds.csv'), 'w') as f:
                f.write('Input'+'\t'+'Predictions'+'\n')
                    
        model.eval()
        total_val_loss = 0
        total_val_acc  = 0
        losses = []
        accuracies= []
        mismatch = 0
        complete_mismatch = 0
        with torch.no_grad():
            for batch_idx, (input_ids, attention_masks, label_input_ids, label_attention_masks) in enumerate(val_loader):
                loss, logits = model(input_ids, attention_masks, label_input_ids, label_attention_masks)
                criterion = nn.CrossEntropyLoss()
                acc, report = accuracy(logits, input_ids, label_input_ids, label_attention_masks, mismatch, complete_mismatch, report=True)
                mismatch += report[1]
                complete_mismatch += report[2]
                preds = report[3]
                write_preds(input_ids, preds)
                total_val_loss += loss.item()
                total_val_acc  += acc.item()
            
                losses.append(loss)
                accuracies.append(acc)
            val_acc  = total_val_acc/len(val_loader)
            val_loss = total_val_loss/len(val_loader)
            if test==False:
                print(f'val_loss: {val_loss:.4f} val_acc: {val_acc:.4f} mismatch: {mismatch} complete_mismatch: {complete_mismatch}')
        return val_acc, val_loss, [report, mismatch, complete_mismatch]

    # Train the model
    train(model, train_loader, val_loader, optimizer, scheduler)

    #Test the model
    model.load_state_dict(torch.load(SAVE_MODEL_DIR+'/best.pt'))

    #write header "pred", "target" to prediction.csv
    with open(os.path.join(SAVE_MODEL_DIR, 'predictions.csv'), 'w') as f:
        f.write('rawpredictions, predictions, targets\n')

    test_acc, test_loss, report = evaluate_accuracy(model, test_loader, test=True)
    mismatch = report[1]
    complete_mismatch = report[2]
    report = report[0][0]
    print(f'Test loss: {test_loss:.4f} Test accuracy: {test_acc:.4f} mismatch: {mismatch} complete_mismatch: {complete_mismatch}')

    print([report])
    
    with open(os.path.join(SAVE_MODEL_DIR, 'results.txt'), 'w') as f:
        f.write(report)