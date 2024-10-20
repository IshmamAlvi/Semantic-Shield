
import torch
from transformers import BertTokenizer, BertModel
import nltk
# nltk.download('punkt')
import os
import clip
from torchvision.datasets import CIFAR100, CIFAR10
from PIL import Image
import torch.nn as nn
from torch.optim import Adam
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np



# Load BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Example multi-sentence input
input_text = "Coat type: short or long, curly or straight, colors such as black, brown, white, or gray.\
    Ear shape: floppy or erect.\
    Tail shape: long and straight, short and curled.\
    Size: small or large."

# Tokenize input into sentences
sentences = nltk.sent_tokenize(input_text)

# Add special tokens to each sentence and tokenize using BERT's tokenizer
tokens = []
for sentence in sentences:
    # Prepend [CLS] token
    sentence = '[CLS] ' + sentence
    # Append [SEP] token
    sentence += ' [SEP]'
    # Tokenize using BERT's tokenizer
    sentence_tokens = tokenizer.tokenize(sentence)
    # Convert tokens to token IDs
    sentence_ids = tokenizer.convert_tokens_to_ids(sentence_tokens)
    # Add to list of token IDs for all sentences
    tokens.append(sentence_ids)

# Pad or truncate all sentences to a fixed length
max_length = 77 # change here from 128 to 77 to sinc with clip tokenize
tokens = [t[:max_length] + [0] * (max_length - len(t)) if len(t) < max_length else t[:max_length] for t in tokens]

# print (tokens)
# Convert tokens to PyTorch tensors
input_ids = torch.tensor(tokens)

# Feed input through BERT model to obtain embeddings for each sentence
with torch.no_grad():
    outputs = model(input_ids)
    # Extract embeddings for each sentence
    sentence_embeddings = [output[:len(sentence), :] for output, sentence in zip(outputs[0], sentences)]

    # print('sentence embeddings: ', len(sentence_embeddings), sentence_embeddings)



########################################## CLIP ##############################################

# Prepare the inputs


device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Download the dataset
cifar100 = CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False)

# Prepare the inputs
image, class_id = cifar100[3637]

image_input = preprocess(image).unsqueeze(0).to(device)
# text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in cifar100.classes]).to(device)
print(sentences)
text_inputs = torch.cat([clip.tokenize(c) for c in sentences]).to(device)

############################## train ######################
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
batch_size  = 128
trainset = CIFAR10(root='./data', train=True, download=True, transform =transform)
train_dataloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

testset = CIFAR10(root='./data', train=False, download=True, transform =transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=True, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

loss_img = nn.CrossEntropyLoss()
loss_txt = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=5e-5,betas=(0.9,0.98),eps=1e-6,weight_decay=0.2) #Params from paper

for batch in train_dataloader:
    optimizer.zero_grad()

    images,texts = batch 
    
    images= images.to(device)
    texts = texts.to(device)
    print(texts.shape)

    logits_per_image, logits_per_text = model(images, texts)

    ground_truth = torch.arange(batch_size).to(device)
    print("ground truth: ", ground_truth)
    print("logits_per_image: ", logits_per_image)
    print("logits_per_text: ", logits_per_text)
    loss_image = loss_img(logits_per_image, ground_truth) 
    loss_kg =  loss_txt(logits_per_text, ground_truth)
    total_loss = (loss_image + loss_kg) / 2
    total_loss.backward()

    # convert_models_to_fp32(model)
    optimizer.step()

# Calculate features
# with torch.no_grad():
#     image_features = model.encode_image(image_input)
#     text_features = model.encode_text(text_inputs)

# # Pick the top 5 most similar labels for the image
# image_features /= image_features.norm(dim=-1, keepdim=True)
# text_features /= text_features.norm(dim=-1, keepdim=True)
# similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
# print(image_features.shape, text_features.shape, similarity)
# values, indices = similarity[0].topk(3)

# Print the result
# print("\nTop predictions:\n")
# for value, index in zip(values, indices):
#     print(f"{cifar100.classes[index]:>16s}: {100 * value.item():.2f}%")





