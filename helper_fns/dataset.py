'''reads pdtb data'''

import sys
import time

from transformers import T5Tokenizer, AutoTokenizer, T5ForConditionalGeneration 
from datasets import load_dataset
import pandas as pd
from preprocessing import read_df_custom
import torch

def get_dataset(MODEL_NAME, device, shared_folder, dataset_name, batch_size, max_length):

    # Load the PDTB text classification data from a CSV file

    #Load alternate train data for out-of-domain training
    df_train = get_alternate_train_df(dataset_name, shared_folder)
    # df_train = read_df_custom(shared_folder + dataset_name +'/' +dataset_name+ '_train'+ ".rels")
    df_dev = read_df_custom(shared_folder + dataset_name +'/' +dataset_name+ '_dev'+ ".rels")
    df_test = read_df_custom(shared_folder + dataset_name +'/' +dataset_name+ '_test'+ ".rels")

    

    # Load the T5 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Preprocess disrpt data
    def preprocess_disrpt(df):
        df['sentence_pair'] = 'sent1: '+df['unit1_txt']+' sent2: '+df['unit2_txt']
        sentences = list(df['sentence_pair'])
        labels = list(df['label'])
        num_labels = len(set(labels))

        # Prepend the prompt to the input sentences
        # should be for the following sentences: column_1_values = [
        # 'deu.rst.pcc', 'eng.dep.covdtb', 'eng.dep.scidtb', 'eng.pdtb.pdtb', 'eng.pdtb.tedm',
        # 'eng.rst.gum', 'eng.rst.rstdt', 'eng.sdrt.stac', 'eus.rst.ert', 'fas.rst.prstc',
        # 'fra.sdrt.annodis', 'ita.pdtb.luna', 'nld.rst.nldt', 'por.pdtb.crpc', 'por.pdtb.tedm',
        # 'por.rst.cstn', 'rus.rst.rrt', 'spa.rst.rststb', 'spa.rst.sctb', 'tha.pdtb.tdtb',
        # 'tur.pdtb.tdb', 'tur.pdtb.tedm', 'zho.dep.scidtb', 'zho.rst.gcdt', 'zho.pdtb.sctb',
        # 'zho.pdtb.cdtb']
        if dataset_name in ['zho.rst.sctb', 'zho.rst.gcdt', 'zho.pdtb.cdtb', 'zho.dep.scidtb']:
            prompt = "sent1 和 sent2 之间的话语关系是什么："
        # elif dataset_name in ['eng.sdrt.stac']:
        #     prompt = "What is the SDRT discourse relation between sent1 and sent2: "
        elif dataset_name in ['deu.rst.pcc']:
            prompt = "Welche Diskursrelation besteht zwischen sent1 und sent2: "
        elif dataset_name in ['eng.dep.covdtb', 'eng.dep.scidtb', 'eng.pdtb.pdtb', 'eng.pdtb.tedm', 'eng.rst.gum', 'eng.rst.rstdt', 'eng.sdrt.stac']:
            prompt = "what discourse relation holds between sent1 and sent2: " 
        elif dataset_name in ['eus.rst.ert']:
            prompt = "sent1 eta sent2 arteko harreman diskurtsiboa zer da: "
        elif dataset_name in ['fas.rst.prstc']:
            prompt = "رابطه گفتمانی بین sent1 و sent2 چیست: "
        elif dataset_name in ['fra.sdrt.annodis']:
            prompt = "Quelle est la relation discursive entre sent1 et sent2: "
        elif dataset_name in ['ita.pdtb.luna']:
            prompt = "Quale relazione discorsiva c'è tra sent1 e sent2: "
        elif dataset_name in ['nld.rst.nldt']:
            prompt = "Wat is de discourse relatie tussen sent1 en sent2: "
        elif dataset_name in ['por.pdtb.crpc', 'por.pdtb.tedm', 'por.rst.cstn']:
            prompt = "Qual é a relação discursiva entre sent1 e sent2: "
        elif dataset_name in ['rus.rst.rrt']:
            prompt = "Какое дискурсивное отношение между sent1 и sent2: "
        elif dataset_name in ['spa.rst.rststb', 'spa.rst.sctb']:
            prompt = "¿Cuál es la relación discursiva entre sent1 y sent2: "
        elif dataset_name in ['tha.pdtb.tdtb']:
            prompt = "ความสัมพันธ์ของข้อความระหว่าง sent1 และ sent2 คืออะไร: "
        elif dataset_name in ['tur.pdtb.tdb', 'tur.pdtb.tedm']:
            prompt = "sent1 ve sent2 arasındaki söylem ilişkisi nedir: "
        elif dataset_name in ['zho.dep.scidtb', 'zho.rst.gcdt']:
            prompt = "sent1 和 sent2 之间的话语关系是什么："
        #classify:   <- original prompt
        sentences = [prompt + sentence for sentence in sentences]

        # # Prepend the prompt to the labels
        # prompt = 'Discourse relation: '
        # labels = [prompt + label for label in labels]
        return sentences, labels, num_labels

    train_sentences, train_labels, num_labels = preprocess_disrpt(df_train)
    val_sentences, val_labels, _ = preprocess_disrpt(df_dev)
    test_sentences, test_labels, _ = preprocess_disrpt(df_test)

    # load fall back label space for out-of-domain data
    label_space = get_fall_back_label_space(dataset_name, val_labels, train_labels)
    majority_class = max(set(label_space), key=label_space.count)

    # # map train, test and val labels to numbers
    # label2id = {label: i for i, label in enumerate(set(train_labels) | set(val_labels) | set(test_labels))}
    # id2label = {i: label for label, i in label2id.items()}
    # train_labels = ["<"+str(label2id[label])+">" for label in train_labels]
    # val_labels = ["<"+str(label2id[label])+">" for label in val_labels]
    # test_labels = ["<"+str(label2id[label])+">" for label in test_labels]

    # # add special tokens to the tokenizer
    # train_label_set = set(train_labels)
    # val_label_set = set(val_labels)
    # test_label_set = set(test_labels)
    # special_tokens_dict = {'additional_special_tokens': list(train_label_set | val_label_set | test_label_set)}
    # tokenizer.add_special_tokens(special_tokens_dict)


    # snli logic commented out
    # train_d = load_dataset('snli', split='train')
    # val_d = load_dataset('snli', split='validation')
    # test_d = load_dataset('snli', split='test')

    # label_map = {0: 'entailment', 1: 'neutral', 2: 'contradiction', -1: 'unknown'}

    # train_sentences = [x['premise'] + ' ' + x['hypothesis'] for x in train_d]
    # train_labels = [label_map[x['label']] for x in train_d]
    # val_sentences = [x['premise'] + ' ' + x['hypothesis'] for x in val_d]
    # val_labels = [label_map[x['label']] for x in val_d]
    # test_sentences = [x['premise'] + ' ' + x['hypothesis'] for x in test_d]
    # test_labels = [label_map[x['label']] for x in test_d]


    # encode the sentences and labels
    train_encodings = tokenizer(train_sentences, truncation=True, padding="max_length", max_length=128, return_tensors='pt')
    train_labels = tokenizer(train_labels, truncation=True, padding="max_length", max_length=128, return_tensors='pt')
    val_encodings = tokenizer(val_sentences, truncation=True, padding="max_length", max_length=128, return_tensors='pt')
    val_labels = tokenizer(val_labels, truncation=True, padding="max_length", max_length=128, return_tensors='pt')
    test_encodings = tokenizer(test_sentences, truncation=True, padding="max_length", max_length=128, return_tensors='pt')
    test_labels = tokenizer(test_labels, truncation=True, padding="max_length", max_length=128, return_tensors='pt')

    # Create the huggingface datasets
    class T5Dataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.input_ids = encodings['input_ids'].to(device)
            self.attention_masks = encodings['attention_mask'].to(device)
            self.labels_input_ids = labels['input_ids'].to(device)
            self.labels_attention_masks = labels['attention_mask'].to(device)

        def __getitem__(self, idx):
            return self.input_ids[idx], self.attention_masks[idx], self.labels_input_ids[idx], self.labels_attention_masks[idx]
        
        def __len__(self):
            return self.input_ids.shape[0]# int(self.input_ids.shape[0]/1000)

    train_dataset = T5Dataset(train_encodings, train_labels)
    val_dataset = T5Dataset(val_encodings, val_labels)
    test_dataset = T5Dataset(test_encodings, test_labels)

    #convert huggingface dataset to pytorch dataloader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, num_labels, label_space, majority_class

def get_alternate_train_df(dataset_name, shared_folder):
    '''function to get alternate df_train for ENG.DEP.COVDTB, ENG.PDTB.TEDM, POR.PDTB.TEDM, TUR.PDTB.TEDM and or get original df_train for other datasets'''
    if dataset_name=='eng.dep.covdtb':
        altername_dataset_name = 'eng.dep.scidtb'
        df_train = read_df_custom(shared_folder + altername_dataset_name +'/' +altername_dataset_name+ '_train'+ ".rels")
    elif dataset_name=='eng.pdtb.tedm':
        altername_dataset_name = 'eng.pdtb.pdtb'
        # shared_folder = shared_folder.replace('2023', '2021')
        df_train = read_df_custom(shared_folder + altername_dataset_name +'/' +altername_dataset_name+ '_train'+ ".rels")
    elif dataset_name=='por.pdtb.tedm':
        altername_dataset_name = 'por.pdtb.crpc'
        df_train = read_df_custom(shared_folder + altername_dataset_name +'/' +altername_dataset_name+ '_train'+ ".rels")
    elif dataset_name=='tur.pdtb.tedm':
        altername_dataset_name = 'tur.pdtb.tdb'
        # shared_folder = shared_folder.replace('2023', '2021')
        df_train = read_df_custom(shared_folder + altername_dataset_name +'/' +altername_dataset_name+ '_train'+ ".rels")
    else:
        altername_dataset_name = None
        df_train = read_df_custom(shared_folder + dataset_name +'/' +dataset_name+ '_train'+ ".rels")
    return df_train

def get_fall_back_label_space(dataset_name, val_labels, train_labels):
    '''function to get alternate_label_space for ENG.DEP.COVDTB, ENG.PDTB.TEDM, POR.PDTB.TEDM, TUR.PDTB.TEDM and or get original label space for other datasets'''
    if dataset_name=='eng.dep.covdtb':
        alternate_dataset_name = 'eng.dep.scidtb'
        label_space = val_labels
    elif dataset_name=='eng.pdtb.tedm':
        alternate_dataset_name = 'eng.pdtb.pdtb'
        label_space = val_labels
    elif dataset_name=='por.pdtb.tedm':
        alternate_dataset_name = 'por.pdtb.crpc'
        label_space = val_labels
    elif dataset_name=='tur.pdtb.tedm':
        alternate_dataset_name = 'tur.pdtb.tdb'
        label_space = val_labels
    else:
        alternate_dataset_name = None
        label_space = train_labels
    label_space = list(set(label_space))
    return label_space

def get_epochs(lang, refinement):
    # data_dict = {
    # 'deu.rst.pcc': 19,
    # 'eng.dep.covdtb': 10,
    # 'eng.dep.scidtb': 5,
    # 'eng.pdtb.pdtb': 1,
    # 'eng.pdtb.tedm': 50,
    # 'eng.rst.gum': 2,
    # 'eng.rst.rstdt': 3,
    # 'eng.sdrt.stac': 4,
    # 'eus.rst.ert': 13,
    # 'fas.rst.prstc': 10,
    # 'fra.sdrt.annodis': 15,
    # 'ita.pdtb.luna': 32,
    # 'nld.rst.nldt': 22,
    # 'por.pdtb.crpc': 4,
    # 'por.pdtb.tedm': 50,
    # 'por.rst.cstn': 10,
    # 'rus.rst.rrt': 1,
    # 'spa.rst.rststb': 16,
    # 'spa.rst.sctb': 50,
    # 'tha.pdtb.tdtb': 5,
    # 'tur.pdtb.tdb': 15,
    # 'tur.pdtb.tedm': 50,
    # 'zho.dep.scidtb': 39,
    # 'zho.pdtb.cdtb': 9,
    # 'zho.rst.gcdt': 6,
    # 'zho.rst.sctb': 50
    # }

    # if refinement == 'True':

    #small datasets: lr=1e-3, epochs=50; large datasets: lr=1e-5, epochs=10
    #*h: empircal evidence. non standard hyperparameters - 
    #       eng.dep.scidtb lr1e-3: 10->43.48
    #       eng.sdrt.stac lr1e-3: 10->
    #       eng.dep.covdtb lr=1e-5: 5->5.95
    #*p: poor performance. 
    # eng.rst.rstdt lr1e-5: 10->58.19
    # eng.sdrt.stac lr1e-5: 10->28.60
    #*r: replication
    # nld.rst.nldt lr=1e-3: 50->60.00 (replicated with minor stddev)
    # data_dict = {#future note: eng.rst.rstdt not good, por.rst.csnt not good
    #     'deu.rst.pcc': 50, #[[50]]lr=1e-3: 10->83.46 50->83.84
    #     'eng.dep.covdtb': 5, #LDC [[10]] lr=1e-5: 10->5.91 5->5.95 | lr1e-3: 10->3.98
    #     'eng.dep.scidtb': 10, #[[10]] lr=1e-5: 10->42.85 50->43.48 | lr1e-3: 10->43.48
    #     'eng.pdtb.pdtb': 5, #[[5]] lre-5: 5->98.75% |lr1e-3: 5->63.09
    #     'eng.pdtb.tedm': 5, #LDC
    #     'eng.rst.gum': 10, #[[10]]lr1e-5: 10->70.15 | lr=1e-4: 10->66.25
    #     'eng.rst.rstdt': 10, #[[10]]lr1e-5: 10->58.19
    #     'eng.sdrt.stac': 10, #[[10]]lr1e-5: 10->28.60 | lr1e-3: 10->25.69 5->25.69    lr1e-5, lr1e-3 was too slow to converge
    #     'eus.rst.ert': 50, #[[50]]lr1e-3: 50->74.77
    #     'fas.rst.prstc': 50, #[[50]]lr1e-3: 50->71.45
    #     'fra.sdrt.annodis': 50, #[[50]]lr1e-3: 10->67.84 50->70.08
    #     'ita.pdtb.luna': 50, #[[50]] lr1e-3: 10->65.52 50->68.68
    #     'nld.rst.nldt': 50, #[[50]] lr=1e-3: 50->59.50 (replicated successfully) | lr=1e-5: 50->51.38
    #     'por.pdtb.crpc': 10, #[[10]] lr=1e-5: 10->100.0
    #     'por.pdtb.tedm': 10, #LDC
    #     'por.rst.cstn': 50, # [[50]] lr=1e-5: 10->45.95| lr=1e-3: 10->51.47 50->51.10   10, 1e-3 like de   (rerun lr=1e-3: 10->51.10 50->86.02)             
    #     'rus.rst.rrt': 10, #[[10]] lr=1e-5: 10->73.02
    #     'spa.rst.rststb': 50, #[[50]] lr=1e-3: 50->72.76
    #     'spa.rst.sctb': 50,#[[50]] lr=1e-5: 2->47 10->50 (training 2 61) 20->50(training 10 56) 50->46 | lr=1e-4: 2-> 10->50 | lr=1e-3: 10->53 50->53
    #     'tha.pdtb.tdtb': 10, #[[10]] lr=1e-5: 10->99.55
    #     'tur.pdtb.tdb': 50, #[[50]] lr=1e-5: 10->85.54
    #     'tur.pdtb.tedm': 50, #LDC [[50]] lr=1e-5: 50->
    #     'zho.dep.scidtb': 50, #[[50]] lr=1e-3: 50->72.09
    #     'zho.pdtb.cdtb': 50, #LDC
    #     'zho.rst.gcdt': 50, #[[50]] lr=1e-3: 50->00.10 | lr=1e-5: 10-> 50->03.56 debugging for lower lr
    #     'zho.rst.sctb': 50 #[[50]] lr=1e-3: 50->54.08 100->51.57
    #     }
    
    data_dict = {#future note: eng.rst.rstdt not good, por.rst.csnt not good
        'deu.rst.pcc': 50, #[[50]]lr=1e-3: 10->83.46 50->83.84
        'eng.dep.covdtb': 10, #LDC [[10]] lr=1e-5: 10->5.91 5->5.95 | lr1e-3: 10->3.98
        'eng.dep.scidtb': 10, #[[10]] lr=1e-5: 10->42.85 50->43.48 | lr1e-3: 10->43.48
        'eng.pdtb.pdtb': 5, #[[5]] lre-5: 5->98.75% |lr1e-3: 5->63.09
        'eng.pdtb.tedm': 5, #LDC
        'eng.rst.gum': 10, #[[10]]lr1e-5: 10->70.15 | lr=1e-4: 10->66.25
        'eng.rst.rstdt': 10, #[[10]]lr1e-5: 10->58.19
        'eng.sdrt.stac': 10, #[[10]]lr1e-5: 10->28.60 | lr1e-3: 10->25.69 5->25.69    lr1e-5, lr1e-3 was too slow to converge
        'eus.rst.ert': 50, #[[50]]lr1e-3: 50->74.77
        'fas.rst.prstc': 50, #[[50]]lr1e-3: 50->71.45
        'fra.sdrt.annodis': 50, #[[50]]lr1e-3: 10->67.84 50->70.08
        'ita.pdtb.luna': 50, #[[50]] lr1e-3: 10->65.52 50->68.68
        'nld.rst.nldt': 50, #[[50]] lr=1e-3: 50->59.50 (replicated successfully) | lr=1e-5: 50->51.38
        'por.pdtb.crpc': 10, #[[10]] lr=1e-5: 10->100.0
        'por.pdtb.tedm': 10, #LDC
        'por.rst.cstn': 50, # [[50]] lr=1e-5: 10->45.95| lr=1e-3: 10->51.47 50->51.10   10, 1e-3 like de   (rerun lr=1e-3: 10->51.10 50->86.02)             
        'rus.rst.rrt': 10, #[[10]] lr=1e-5: 10->73.02
        'spa.rst.rststb': 50, #[[50]] lr=1e-3: 50->72.76
        'spa.rst.sctb': 50,#[[50]] lr=1e-5: 2->47 10->50 (training 2 61) 20->50(training 10 56) 50->46 | lr=1e-4: 2-> 10->50 | lr=1e-3: 10->53 50->53
        'tha.pdtb.tdtb': 10, #[[10]] lr=1e-5: 10->99.55
        'tur.pdtb.tdb': 50, #[[50]] lr=1e-5: 10->85.54
        'tur.pdtb.tedm': 50, #LDC [[50]] lr=1e-5: 50->
        'zho.dep.scidtb': 50, #[[50]] lr=1e-3: 50->72.09
        'zho.pdtb.cdtb': 50, #LDC
        'zho.rst.gcdt': 50, #[[50]] lr=1e-3: 50->00.10 | lr=1e-5: 10-> 50->03.56 debugging for lower lr
        'zho.rst.sctb': 50 #[[50]] lr=1e-3: 50->54.08 100->51.57
        }

    return data_dict[lang]


def get_lr(lang, refinement):
    # data_dict = {
    #     'deu.rst.pcc': 1e-3,#[[1e-3]]
    #     'eng.dep.covdtb': 1e-5, #LDC[[1e-5]]]
    #     'eng.dep.scidtb': 1e-3, #[[1e-5]]*h
    #     'eng.pdtb.pdtb': 1e-5, #[[1e-5]]
    #     'eng.pdtb.tedm': 1e-5, #LDC
    #     'eng.rst.gum': 1e-5, #[[1e-3]]
    #     'eng.rst.rstdt': 1e-5, #[[1e-5]]*p
    #     'eng.sdrt.stac': 1e-5, #[[1e-5]]*p
    #     'eus.rst.ert': 1e-3, #[[1e-3]]
    #     'fas.rst.prstc': 1e-3, #[[1e-3]]
    #     'fra.sdrt.annodis': 1e-3, #[[1e-3]]
    #     'ita.pdtb.luna': 1e-3, #[[1e-3]]
    #     'nld.rst.nldt': 1e-3, #[[1e-3]]
    #     'por.pdtb.crpc': 1e-5, #[[1e-5]]
    #     'por.pdtb.tedm': 1e-5, #LDC
    #     'por.rst.cstn': 1e-3, #[[1e-3]]
    #     'rus.rst.rrt': 1e-5, #[[1e-5]]]
    #     'spa.rst.rststb': 1e-3, #[[1e-3]]
    #     'spa.rst.sctb': 1e-3,#[[1e-3]]
    #     'tha.pdtb.tdtb': 1e-5, #[[1e-5]]
    #     'tur.pdtb.tdb': 1e-5, #[[1e-5]]
    #     'tur.pdtb.tedm': 1e-5, #LDC
    #     'zho.dep.scidtb': 1e-3, #[[1e-3]]
    #     'zho.pdtb.cdtb': 1e-3, #LDC
    #     'zho.rst.gcdt': 1e-3, #[[1e-3]]
    #     'zho.rst.sctb': 1e-3#[[1e-3]]
    #     }
    
    data_dict = {
        'deu.rst.pcc': 1e-3,#1e-3,#[[1e-3]]
        'eng.dep.covdtb': 1e-5, #LDC[[1e-5]]]
        'eng.dep.scidtb': 1e-5, #[[1e-5]]*h
        'eng.pdtb.pdtb': 1e-5, #[[1e-5]]
        'eng.pdtb.tedm': 1e-5, #LDC
        'eng.rst.gum': 1e-5, #[[1e-3]]
        'eng.rst.rstdt': 1e-5, #[[1e-5]]*p
        'eng.sdrt.stac': 1e-5, #[[1e-5]]*p
        'eus.rst.ert': 1e-3, #[[1e-3]]
        'fas.rst.prstc': 1e-3, #[[1e-3]]
        'fra.sdrt.annodis': 1e-3, #[[1e-3]]
        'ita.pdtb.luna': 1e-3, #[[1e-3]]
        'nld.rst.nldt': 1e-3, #[[1e-3]]
        'por.pdtb.crpc': 1e-5, #[[1e-5]]
        'por.pdtb.tedm': 1e-5, #LDC
        'por.rst.cstn': 1e-3, #[[1e-3]]
        'rus.rst.rrt': 1e-5, #[[1e-5]]]
        'spa.rst.rststb': 1e-3, #[[1e-3]]
        'spa.rst.sctb': 1e-3,#1e-3,#[[1e-3]]
        'tha.pdtb.tdtb': 1e-5, #[[1e-5]]
        'tur.pdtb.tdb': 1e-3, #[[1e-5]]
        'tur.pdtb.tedm': 1e-3, #LDC
        'zho.dep.scidtb': 1e-3, #[[1e-3]]
        'zho.pdtb.cdtb': 1e-3, #LDC
        'zho.rst.gcdt': 1e-3, #[[1e-3]]
        'zho.rst.sctb': 1e-3#1e-3#[[1e-3]]
        }
    return data_dict[lang]

#TODO: fixes
# refined eng.pdtb.pdtb
# eng.rst.gum 1e-3 diya tha 1e-5 hona chahiye
# zho itna low iska paper analysis only

#urgent TODO: por.pdtb.crpc, tha.pdtb.tdtb 1e-5 verify the high number
# removed sdrt prompt. 6% gain which is significant