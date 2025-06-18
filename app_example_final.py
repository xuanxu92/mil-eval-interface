import gradio as gr
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from tqdm import tqdm
import numpy as np
import torch
import torchvision
import os,glob
from os.path import join as j_
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas as pd



from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F

# Dummy MIL model definition
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix 
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.manifold import TSNE

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

class FeatureBagsDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]

def data_list(data_csv):
    feat_list = []
    label_list = data_csv['label'].tolist()
    embedding_csv = data_csv['embeddings'].tolist()

    for i in tqdm(range(len(data_csv))):
        feat = torch.load(embedding_csv[i])
        feat_list.append(feat)

    return feat_list, torch.tensor(label_list)

def save_checkpoint(state, filename="checkpoint.pth"):
    torch.save(state, filename)

def evaluate_model(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for features, labels in dataloader:
            features = features.squeeze(0).float().to(device)
            labels = labels.long().to(device)
            logits = model(features)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            _, predicted = torch.max(logits, dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    return avg_loss, accuracy

class DeepAttentionMIL(nn.Module):
    def __init__(self, instance_dim, hidden_dim, num_classes):
        super(DeepAttentionMIL, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(instance_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.attention_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=0)
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, bag):
        instance_features = self.feature_extractor(bag)
        attention_weights = self.attention_layer(instance_features)
        bag_representation = torch.sum(attention_weights * instance_features, dim=0)
        output = self.classifier(bag_representation.unsqueeze(0))
        return output

def evaluate_model_on_test(model, dataloader, criterion, device):
    model.eval()
    predictions = []
    true_labels = []
    total_loss = 0
    with torch.no_grad():
        for features, labels in tqdm(dataloader):
            features = features.squeeze(0).float().to(device)
            labels = labels.long().to(device)
            logits = model(features)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            _, predicted = torch.max(logits, dim=1)
            predictions.extend(predicted.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, average='macro')
    recall = recall_score(true_labels, predictions, average='macro')
    f1 = f1_score(true_labels, predictions, average='macro')
    return predictions, true_labels, accuracy, precision, recall, f1

pancancer = '/data05/shared/jkaczmarzyk/pretrained-mil/deep-features/tcga/pancancer/'
model_dict = {'UNI':'uni-256px_0.5mpp','Phikon':'phikon-256px_0.5mpp','CTransPath':'ctranspath-256px_0.5mpp','Remedis':'remedis-152x2-m-256px_0.5mpp','Retccl':'retccl-256px_0.5mpp'}
csv_path = '/data07/shared/xxu/downstream/tcgapan/dataframes/'
# cancer_dict = {'NSCLC':'nsclc_subtype_class_set.csv','BRCA':'brca_subytpe_class_set.csv','COADREAD':'coadread_subtype_class_set.csv'}
cancer_dict = {'NSCLC':'nsclc_subtype_class_set.csv','BRCA':'brca_subytpe_class_set.csv','COADREAD':'coadread_subtype_class_set.csv','RCC':'rcc_subtype_class_set.csv'}

def get_label_distribution(df):
    """Get label distribution as a string."""
    distribution_text = ""
    set_column = 'set'
    label_column = df.columns[1]
    for set_type, group_df in df.groupby(set_column):
        label_distribution = group_df[label_column].value_counts()
        distribution_text += f"\n{set_type.capitalize()} Set:\n"
        distribution_text += '\n'.join([f"Class {label}: {percentage} " for label, percentage in label_distribution.items()])
        distribution_text += '\n'
    return distribution_text

def create_tsne_plot(embeddings, labels, original_dict, model_name, cancer_type):
    """Create a t-SNE plot for the embeddings."""
    print('tsne plot')
    label_mapping =  {value: key for key, value in original_dict.items()}
    pooled_embeddings = torch.stack([torch.mean(emb, dim=0) for emb in embeddings])

    # Step 2: Convert to numpy array
    pooled_embeddings_np = pooled_embeddings.numpy()

    # Step 3: Apply t-SNE to reduce dimensions to 2D
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(pooled_embeddings_np)
    # labels_np = np.hstack(labels)

    plt.figure(figsize=(8, 6))
    colors = ['blue', 'orange']  # Define colors for LUAD and LUSC

    for label in np.unique(labels):
        indices = np.where(np.array(labels) == label)
        plt.scatter(embeddings_2d[indices, 0], embeddings_2d[indices, 1], 
                    c=colors[label], label=label_mapping[label], alpha=0.6)
    plt.title(f't-SNE Plot for {model_name} - {cancer_type}')
    plt.legend()
    tsne_plot_path = f'/data07/shared/xxu/downstream/tcgapan/plots/{model_name}_{cancer_type}_tsne_plot.png'
    plt.savefig(tsne_plot_path)
    plt.close()

    return tsne_plot_path

def train_mil_model(model_name, cancer_type, split_num, optimizer_name, lr, weight_decay=0.0, epochs=20, batch_size=1):
    embedding_path = pancancer + model_dict[model_name]
    df_path = csv_path + cancer_dict[cancer_type]
    df = pd.read_csv(df_path)
    embeddings_list = glob.glob(embedding_path + '/*.pt')
    embeddings_list.sort()
    df_embedding = pd.DataFrame({'embeddings': embeddings_list})
    df_embedding['slideID'] = df_embedding['embeddings'].apply(lambda x: x.split('/')[-1].split('.')[0])
    df_embedding[df.columns[0]] = df_embedding['embeddings'].apply(lambda x: x.split('/')[-1].split('.')[0][0:15])
    df_merge = pd.merge(df, df_embedding, on=df.columns[0], how='left')
    subtype_names = np.unique(df_merge[df_merge.columns[1]]).tolist()
    subtype_mapping = {name: idx for idx, name in enumerate(subtype_names)}
    df_merge['label'] = df_merge[df_merge.columns[1]].map(subtype_mapping)
    df_merge = df_merge.dropna()

    label_distribution = get_label_distribution(df_merge)
    test_df_cur = df_merge.loc[df_merge['set']=='test']
    test_df_cur = test_df_cur.reset_index(drop=True)
    train_val_df_cur = df_merge.loc[df_merge['set']=='train']
    train_val_df_cur = train_val_df_cur.reset_index(drop=True)
    

    # train_val_df_cur, test_df_cur = train_test_split(df_merge, test_size=0.2, stratify=df_merge['label'], random_state=42)

    X = train_val_df_cur.drop(columns=['label'])
    y = train_val_df_cur['label']

    skf = StratifiedKFold(n_splits=split_num, shuffle=True, random_state=42)
    folds = []
    for train_index, val_index in skf.split(X, y):
        train_df = train_val_df_cur.iloc[train_index]
        val_df = train_val_df_cur.iloc[val_index]
        folds.append((train_df, val_df))

    test_df_cur = test_df_cur.reset_index(drop=True)
    test_feats, test_labels = data_list(test_df_cur)
    test_dataset = FeatureBagsDataset(test_feats, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    for num in range(1):
        train_df, val_df = folds[num]
        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)

        train_feats, train_labels = data_list(train_df)
        val_feats, val_labels = data_list(val_df)
        train_dataset = FeatureBagsDataset(train_feats, train_labels)
        val_dataset = FeatureBagsDataset(val_feats, val_labels)

        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

        feature_dim = train_feats[0].shape[1]
        attention_dim = 128
        model = DeepAttentionMIL(instance_dim=feature_dim, hidden_dim=attention_dim, num_classes=len(np.unique(train_labels)))
        model = model.to(device)

        if optimizer_name == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=lr)
        elif optimizer_name == 'AdamW':
            optimizer = optim.AdamW(model.parameters(), lr=lr)
        elif optimizer_name == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)

        criterion = nn.CrossEntropyLoss()

        train_losses, val_losses = [], []
        for epoch in tqdm(range(epochs), desc="Training Epochs"):
            model.train()
            total_loss = 0
            for bag_features, label in train_loader:
                bag_features = bag_features.squeeze(0).float().to(device)
                label = label.long().to(device)

                logits = model(bag_features)
                loss = criterion(logits, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            train_loss = total_loss / len(train_loader)
            val_loss, val_accuracy = evaluate_model(model, val_loader, criterion)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            print(f'Epoch {epoch}, Train Loss: {train_loss}, Val Loss: {val_loss}, Val Accuracy: {val_accuracy}')

        print('evaluation on test')
        predictions, true_labels, accuracy, precision, recall, f1 = evaluate_model_on_test(model, test_loader, criterion, device)
        print('tsne plot plot')
        tsne_plot_path = create_tsne_plot(train_feats, train_labels, subtype_mapping, model_name, cancer_type)
        print('tsne plot end')

    # return train_losses, val_losses, accuracy, f1, precision, recall, label_distribution
    return train_losses, val_losses, accuracy, f1, precision, recall, label_distribution, tsne_plot_path


# Visualization and training function
results = []
def visualize_and_train(model_name, cancer_type, optimizer_name, lr, weight_decay, epochs):
    global results

    train_losses, val_losses, accuracy, f1, precision, recall, label_distribution,tsne_plot_path = train_mil_model(
        model_name=model_name, cancer_type=cancer_type, split_num=5,
        optimizer_name=optimizer_name, lr=lr, weight_decay=weight_decay, epochs=epochs, batch_size=1
    )

    # Plotting the loss
    print('plot loss')
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{model_name} - {cancer_type} Training Loss')
    plt.legend()
    plt.grid(True)
    loss_plot_path = f'/data07/shared/xxu/downstream/tcgapan/plots/{model_name}_{cancer_type}_loss_plot.png'
    plt.savefig(loss_plot_path)
    plt.close()
    print('loss plot close')

    result = {
        'model_name': model_name,
        'cancer_type': cancer_type,
        'optimizer_name': optimizer_name,
        'lr': lr,
        'epochs': epochs,
        'accuracy': accuracy,
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'loss_plot_path': loss_plot_path
    }
    results.append(result)

    comparison_text = "\n".join(
        [f"{res['model_name']} - {res['cancer_type']} (Opt: {res['optimizer_name']} LR: {res['lr']} Epochs: {res['epochs']}): "
         f"Acc = {res['accuracy']:.4f}, F1 = {res['f1_score']:.4f}, Precision = {res['precision']:.4f}, Recall = {res['recall']:.4f}"
         for res in results]
    )

    return (
        f'{model_name} - {cancer_type} Model Results:\nAccuracy: {accuracy:.4f}, F1 Score: {f1:.4f}',
        loss_plot_path,
        comparison_text,
        label_distribution,
        tsne_plot_path
    )
    # return (
    #     f'{model_name} - {cancer_type} Model Results:\nAccuracy: {accuracy:.4f}, F1 Score: {f1:.4f}',
    #     'training_loss_plot.png',
    #     'Comparison text placeholder',
    #     'Label distribution placeholder',
    #     tsne_plot_path
    # )

# # Gradio Interface
iface = gr.Interface(
    fn=visualize_and_train,
    inputs=[
        gr.Dropdown(choices=["Phikon", "UNI", "CTransPath", "Remedis", "Retccl"], label="Select Foundation Model"),
        gr.Dropdown(choices=["NSCLC", "BRCA", "COADREAD","RCC"], label="Select Cancer Type"),
        gr.Dropdown(choices=["Adam", "AdamW", "SGD"], label="Select Optimizer"),
        gr.Number(label="Learning Rate"),
        gr.Number(label="Weight Decay", value=0.0),
        gr.Slider(minimum=1, maximum=100, step=1, value=20, label="Number of Epochs")
    ],
    outputs=[
        gr.Textbox(label="Model results on test set"),
        gr.Image(type="pil", label="Training Loss Plot"),
        gr.Textbox(label="Previous Results Comparison on test set"),
        gr.Textbox(label="Label Distribution"),
        gr.Image(type="pil", label="t-SNE Plot")
    ],
    title="MIL Model Training Visualization"
)
# Define Gradio Interface layout
# with gr.Blocks() as iface:
#     with gr.Row():
#         with gr.Column(scale=1):
#             tsne_plot = gr.Image(type="pil", label="t-SNE Plot")
#         with gr.Column(scale=2):
#             model_results = gr.Textbox(label="Model Results")
#             training_loss_plot = gr.Image(type="pil", label="Training Loss Plot")
#             comparison_text = gr.Textbox(label="Previous Results Comparison")
#             label_distribution = gr.Textbox(label="Label Distribution")
    
#     inputs = [
#         gr.Dropdown(choices=["Phikon", "UNI", "CTransPath", "Remedis", "Retccl"], label="Select Foundation Model"),
#         gr.Dropdown(choices=["NSCLC", "BRCA", "COADREAD"], label="Select Cancer Type"),
#         gr.Dropdown(choices=["Adam", "AdamW", "SGD"], label="Select Optimizer"),
#         gr.Number(label="Learning Rate"),
#         gr.Number(label="Weight Decay", value=0.0),
#         gr.Slider(minimum=1, maximum=100, step=1, value=20, label="Number of Epochs")
#     ]
#     outputs = [
#         model_results,
#         training_loss_plot,
#         comparison_text,
#         label_distribution,
#         tsne_plot
#     ]
    
#     gr.Interface(
#         fn=visualize_and_train,
#         inputs=inputs,
#         outputs=outputs,
#         live=False
#     ).launch(share=True)

if __name__ == "__main__":
    iface.launch(share=True)
