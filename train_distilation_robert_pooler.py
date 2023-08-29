import torch
import torch.nn.functional as F
import logging
import pandas as pd
from PIL import Image
from torch import nn,Tensor
import numpy as np
import os, json
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPTextModel, CLIPTokenizer, CLIPTextConfig
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup
from tqdm import tqdm
import time
from shutil import copyfile
import matplotlib.pyplot as plt
from transformers import AutoModel, AutoTokenizer

class EnKoDataset(Dataset):
    def __init__(self, df, tokenizer, tokenizer_teacher, max_length=77):
        self.text_e = tokenizer(list(df['english']), padding="max_length", max_length=max_length, truncation=True, return_tensors='pt').input_ids
        self.text_k = tokenizer(list(df['korea']), padding="max_length", max_length=max_length, truncation=True, return_tensors='pt').input_ids
        self.text_gt = tokenizer_teacher(list(df['english']), padding="max_length", max_length=max_length, truncation=True, return_tensors='pt').input_ids
        
    def __len__(self):
        return len(self.text_e)

    def __getitem__(self, idx):
        return self.text_e[idx], self.text_k[idx], self.text_gt[idx]

def dataset_split(dataset, ratio):
    train_size = int(ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    return train_dataset, val_dataset
    
def _validate(model, teacher_model, val_dataloader, device):
    
    loss_fn = nn.MSELoss()
    losslist_en, losslist_ko, losslist = [], [], []
    
    model.eval() 
    
    for step_idx, batch_data in tqdm(enumerate(val_dataloader), f"[Eval] ", total=len(val_dataloader)):
        with torch.no_grad():
            text_e, text_k, text_gt = batch_data
            batch_size = text_e.shape[0]
            
            texts = torch.cat((text_e, text_k), dim=0).to(device)
            embeds = model(texts)[1]
            
            embed_en, embed_ko = embeds[:batch_size], embeds[batch_size:]
            embed_gt = teacher_model(text_gt.to(device))[1]
            
            loss_en = loss_fn(embed_en, embed_gt.detach())
            loss_ko = loss_fn(embed_ko, embed_gt.detach())
            loss = loss_en + loss_ko
            
            losslist_en.append(loss_en.cpu().clone().detach().numpy())
            losslist_ko.append(loss_ko.cpu().clone().detach().numpy())
            losslist.append(loss.cpu().clone().detach().numpy())

    mean_loss, mean_loss_en, mean_loss_ko = np.mean(losslist), np.mean(losslist_en), np.mean(losslist_ko)
    model.train()

    return mean_loss, mean_loss_en, mean_loss_ko


def train(model, teacher_model, optimizer, scheduler, train_dataloader, val_dataloader, device, batch_size, logger, max_length, hidden_dim):
    
    model.to(device)
    teacher_model.to(device)  
    model.train() 
    
    start_time = time.time()
    
    loss_fn = nn.MSELoss()
    losslist_en, losslist_ko, losslist = [], [], []
    train_loss, valid_loss = [], [] 
    
     
    for epoch_id in range(epochs):             
        for step_idx, batch_data in tqdm(enumerate(train_dataloader), f"[TRAIN] E{epoch_id}", total=len(train_dataloader)):
            
            optimizer.zero_grad()
            
            global_step = len(train_dataloader) * epoch_id + step_idx + 1
            
            text_en, text_ko, text_gt = batch_data
            batch_size = text_en.shape[0]
            texts = torch.cat((text_en, text_ko), dim=0).to('cuda')
            embeds = model(texts)[1]
            embed_en, embed_ko = embeds[:batch_size], embeds[batch_size:]
            embed_gt = teacher_model(text_gt.to(device))[1]
            
            loss_en = loss_fn(embed_en, embed_gt.detach())
            loss_ko = loss_fn(embed_ko, embed_gt.detach())
            loss = loss_en + loss_ko
            
            losslist_en.append(loss_en.cpu().clone().detach().numpy())
            losslist_ko.append(loss_ko.cpu().clone().detach().numpy())
            losslist.append(loss.cpu().clone().detach().numpy())
                
            loss.backward()
            optimizer.step()
            scheduler.step()

        mean_loss, mean_loss_en, mean_loss_ko = np.mean(losslist), np.mean(losslist_en), np.mean(losslist_ko)
        val_loss, val_loss_en, val_loss_ko = _validate(model, teacher_model, val_dataloader, device)
        
        losslist.clear()
        losslist_en.clear()
        losslist_ko.clear()
        
        logger.info(f"\tTRAIN_EPOCH:{epoch_id}\t global_step:{global_step} loss:{mean_loss:.4f}")
        logger.info(f"\tVALIDATION\t global_step:{global_step} loss:{val_loss:.4f} | {(time.time() - start_time):.2f}secs passed")
        
        logger.info(f"\tTrain loss : \tloss_en:{mean_loss_en:.4f} loss_ko:{mean_loss_ko:.4f}")
        logger.info(f"\tValid loss : \tloss_en:{val_loss_en:.4f} loss_ko:{val_loss_ko:.4f}")

        if len(valid_loss) == 0 or val_loss >= max(valid_loss) :
            state_dict = model.state_dict()
            torch.save(state_dict, model_path + '/pytorch_model.bin')    
            
        train_loss.append(mean_loss) 
        valid_loss.append(val_loss)
        
    return model, train_loss, valid_loss


tokenizer_teacher = CLIPTokenizer.from_pretrained("/home/sangyeob/dev/d2d/5-K_stable_diffusion/tokenizer")
text_encoder_teacher = CLIPTextModel.from_pretrained("/home/sangyeob/dev/d2d/5-K_stable_diffusion/text_encoder")

tokenizer = AutoTokenizer.from_pretrained("klue/roberta-small")
text_encoder_student = AutoModel.from_pretrained("klue/roberta-small")

df_500000 = pd.read_csv('/home/sangyeob/dev/d2d/5-K_stable_diffusion/df_500000.csv')

# Config
epochs = 5
learning_rate = 5e-4
weight_decay = 1e-4
# save_interval = 1000 
batch_size = 32
dataset_df = df_500000.sample(250000)

logger = logging.getLogger()  
logger.setLevel(logging.INFO)  

# Dataset
logging.info("\tdataset loading..")
start_time = time.time()
dataset = EnKoDataset(dataset_df, tokenizer, tokenizer_teacher)
train_dataset, val_dataset = dataset_split(dataset, 0.8)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
logging.info(f"\tdataset loaded.. {(time.time() - start_time):.2f} passed")


# optimizer
param_optimizer = list(text_encoder_student.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(
        nd in n for nd in no_decay)], 'weight_decay': weight_decay},
    {'params': [p for n, p in param_optimizer if any(
        nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, correct_bias=False)

# scheduler
step_per_epoch = len(train_dataloader)
num_total_steps = step_per_epoch * epochs

# scheduler = torch.optim.lr_scheduler.OneCycleLR(
#     optimizer,
#     learning_rate,
#     total_steps=num_total_steps
# )

scheduler = get_cosine_schedule_with_warmup(
    optimizer, 
    num_warmup_steps = num_total_steps * 0.1, 
    num_training_steps = num_total_steps
)

step_per_epoch = len(train_dataloader)
step_per_epoch_val = len(val_dataloader)
total_train_dataset_len = len(train_dataset)
total_valid_dataset_len = len(val_dataset)
num_total_steps = step_per_epoch * epochs
model_path = f'/home/sangyeob/dev/d2d/5-K_stable_diffusion/text_encoder/pooler/{total_train_dataset_len + total_valid_dataset_len}_{learning_rate}_{batch_size}'
if not os.path.isdir(model_path) : os.mkdir(model_path)

logging.info(f'\tstep_per_epoch train/val : {step_per_epoch} / {step_per_epoch_val} steps')
logging.info(f'\tnum_total_steps : {num_total_steps} steps')
logging.info(f'\ttotal_dataset_len : {total_train_dataset_len + total_valid_dataset_len} ({total_train_dataset_len} + {total_valid_dataset_len})')
logging.info(f'\tlearning_rate : {learning_rate}')
logging.info(f'\tbatch_size : {batch_size}')
logging.info(f'\tmodel_path : {model_path}')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

with torch.no_grad() : 
    
    text_e = 'dog painted by pen'
    text_k = '펜으로 그린 강아지'
    token_gt = tokenizer_teacher(text_e, padding="max_length", max_length=77, truncation=True, return_tensors='pt').input_ids 
    token_e = tokenizer(text_e, padding="max_length", max_length=77, truncation=True, return_tensors='pt').input_ids
    token_k = tokenizer(text_k, padding="max_length", max_length=77, truncation=True, return_tensors='pt').input_ids

    embed_gt = text_encoder_teacher(input_ids = token_gt)[0].to('cuda')
    init_embed_e = text_encoder_student(input_ids = token_e)[0].to('cuda')
    init_embed_k = text_encoder_student(input_ids = token_k)[0].to('cuda')



text_encoder_student, train_loss, valid_loss = train(model=text_encoder_student,
                                                        teacher_model=text_encoder_teacher, 
                                                        optimizer=optimizer, 
                                                        scheduler=scheduler, 
                                                        train_dataloader=train_dataloader, 
                                                        val_dataloader=val_dataloader, 
                                                        device=device, 
                                                        batch_size=batch_size, 
                                                        logger=logger,
                                                        max_length=77, 
                                                        hidden_dim=768)

state_dict = text_encoder_student.state_dict()
torch.save(state_dict, model_path + '/pytorch_model.bin')
copyfile('/home/sangyeob/dev/d2d/5-K_stable_diffusion/text_encoder/config.json', model_path + '/config.json')

logger.info(
    f"saved at {model_path}"    
)

with torch.no_grad() : 
    
    mse_loss = nn.MSELoss()
    
    embed_e = text_encoder_student(input_ids = token_e.to('cuda'))[0]
    embed_k = text_encoder_student(input_ids = token_k.to('cuda'))[0]
    
    print(f'학습 전 gt, 영어 간 mse_loss : {mse_loss(embed_gt, init_embed_e)}')
    print(f'학습 전 gt, 한국어 간 mse_loss : {mse_loss(embed_gt, init_embed_k)}')
    print(f'학습 전 영어, 한국어 간 mse_loss : {mse_loss(init_embed_e, init_embed_k)}')
    print(f'학습 후 gt, 영어 간 mse_loss : {mse_loss(embed_gt, embed_e)}')
    print(f'학습 후 gt, 한국어 간 mse_loss : {mse_loss(embed_gt, embed_k)}')
    print(f'학습 후 영어, 한국어 간 mse_loss : {mse_loss(embed_e, embed_k)}')

    # init_embed_e = init_embed_e.reshape(1, -1)
    # init_embed_k = init_embed_k.reshape(1, -1)
    # embed_e = embed_e.reshape(1, -1)
    # embed_k = embed_k.reshape(1, -1)
    # target = torch.ones(1)
    
    # print(f'학습 전 영어, 한국어 간 cos_loss : {cos_loss(init_embed_e, init_embed_k, target)}')
    # print(f'학습 후 영어, 한국어 간 cos_loss : {cos_loss(embed_e, embed_k, target)}')
    # print(f'원래 임베딩과 영어 텍스트 간 cos_loss : {cos_loss(embed_e, init_embed_e, target)}')
    # print(f'원래 임베딩과 한국어 텍스트 간 cos_loss : {cos_loss(embed_k, init_embed_e, target)}')
    # print(f'전체 cos_loss : {cos_loss(embed_e, embed_k, target) + cos_loss(embed_e, init_embed_e, target) + cos_loss(embed_k, init_embed_e, target)}')
    
# 한국말끼리는 cos simmilarity가 높은 모양

plt.plot(train_loss)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training/Valid loss history')
plt.savefig(model_path + '/loss.jpg')
plt.show()

plt.plot(valid_loss)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training/Valid loss history')
plt.savefig(model_path + '/loss.jpg')