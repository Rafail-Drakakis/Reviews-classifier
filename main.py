from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import os, time, torch, nltk, zipfile, shutil, gdown, pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from torch.utils.data import Dataset, DataLoader
from sklearn.naive_bayes import MultinomialNB
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
import torch.nn as nn
import seaborn as sns
import numpy as np

class ImdbDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.LongTensor(X)
        self.y = torch.FloatTensor(y)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=100, hidden_dim=128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        e, _ = self.lstm(self.embed(x))
        out = self.fc(e[:, -1, :])
        return torch.sigmoid(out).squeeze()

class CNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=100, num_filters=100, filter_sizes=[3, 4, 5]):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, fs)
            for fs in filter_sizes
        ])
        self.fc = nn.Linear(num_filters * len(filter_sizes), 1)
    def forward(self, x):
        e = self.embed(x).permute(0, 2, 1)
        cs = [torch.relu(conv(e)).max(dim=2)[0] for conv in self.convs]
        cat = torch.cat(cs, dim=1)
        return torch.sigmoid(self.fc(cat)).squeeze()

def ensure_nltk_data():
    for pkg, path in [('punkt', 'tokenizers/punkt'), ('punkt_tab', 'tokenizers/punkt_tab'), ('stopwords', 'corpora/stopwords')]:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(pkg)

def load_imdb_data(data_dir):
    texts, labels = [], []
    for label in ['pos', 'neg']:
        folder = os.path.join(data_dir, label)
        for fname in os.listdir(folder):
            if not fname.endswith('.txt'):
                continue
            with open(os.path.join(folder, fname), encoding='utf-8') as f:
                texts.append(f.read())
                labels.append(1 if label == 'pos' else 0)
    return texts, labels

def preprocess_texts(texts):
    sw = set(stopwords.words('english'))
    out = []
    for t in texts:
        toks = word_tokenize(t.lower())
        toks = [w for w in toks if w.isalpha() and w not in sw]
        out.append(' '.join(toks))
    return out

def save_confusion(cm, classes, title, filename):
    os.makedirs("figures", exist_ok=True)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(title)
    plt.tight_layout()
    filepath = os.path.join("figures", filename)
    plt.savefig(filepath)
    plt.close()
    print(f"Saved confusion matrix to {filepath}")

def train_evaluate_classical(train_texts, train_labels, test_texts, test_labels):
    tfidf = TfidfVectorizer(max_features=5000)
    X_train = tfidf.fit_transform(train_texts)
    X_test = tfidf.transform(test_texts)

    models = {
        'LogReg': LogisticRegression(max_iter=1000, random_state=42),
        'NaiveBayes': MultinomialNB(alpha=1.0),
        'LinearSVM': LinearSVC(C=1.0)
    }

    results = {}
    for name, m in models.items():
        print(f"\n=== Training {name} ===")
        m.fit(X_train, train_labels)
        t0 = time.time()
        preds = m.predict(X_test)
        t1 = time.time()
        acc = accuracy_score(test_labels, preds)
        print(f"{name} Accuracy: {acc:.4f}")
        print(classification_report(test_labels, preds, target_names=['neg', 'pos']))
        cm = confusion_matrix(test_labels, preds)
        save_confusion(cm, ['neg', 'pos'], f"{name} Confusion", f"{name}_confusion.png")
        results[name] = {
            'model': m,
            'vectorizer': tfidf,
            'accuracy': acc,
            'inference_time_per_sample_ms': (t1 - t0) / len(test_labels) * 1000
        }
    return results

def build_vocab(tokenized_texts, max_size=20000, min_freq=2):
    freq = {}
    for toks in tokenized_texts:
        for w in toks:
            freq[w] = freq.get(w, 0) + 1
    vocab = ['<PAD>', '<UNK>'] + [w for w, c in sorted(freq.items(), key=lambda x: -x[1]) if c >= min_freq][:max_size]
    return {w: i for i, w in enumerate(vocab)}

def texts_to_indices(tokenized, w2i):
    unk = w2i['<UNK>']
    return [[w2i.get(w, unk) for w in toks] for toks in tokenized]

def pad_sequences(seqs, max_len=200):
    padded = []
    for s in seqs:
        if len(s) > max_len:
            s = s[:max_len]
        padded.append(s + [0] * (max_len - len(s)))
    return np.array(padded)

def train_pytorch_model(model, train_loader, test_loader, device, epochs=5, lr=1e-3, save_path='model.pt'):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    best_acc = 0.0
    best_model_state = None

    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        print(f"Epoch {ep}/{epochs} train loss: {total_loss / len(train_loader):.4f}")

        model.eval()
        all_preds, all_y = [], []
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(device)
                preds = (model(xb) > 0.5).long().cpu().numpy()
                all_preds.extend(preds.tolist())
                all_y.extend(yb.numpy().astype(int).tolist())

        acc = accuracy_score(all_y, all_preds)
        print(f"Validation Accuracy: {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            best_model_state = model.state_dict()
            torch.save(best_model_state, save_path)
            print(f"Saved best model with acc = {acc:.4f}")

    print(f"Best test accuracy: {best_acc:.4f}")
    model.load_state_dict(torch.load(save_path))
    return model, best_acc

def train_deep_models(train_texts, train_labels, test_texts, test_labels):
    tok_train = [t.split() for t in train_texts]
    tok_test = [t.split() for t in test_texts]
    w2i = build_vocab(tok_train)

    with open('vocab.pkl', 'wb') as f:
        pickle.dump(w2i, f)

    seq_train = pad_sequences(texts_to_indices(tok_train, w2i))
    seq_test = pad_sequences(texts_to_indices(tok_test, w2i))

    bs = 64
    train_ds = ImdbDataset(seq_train, train_labels)
    test_ds = ImdbDataset(seq_test, test_labels)
    tl = DataLoader(train_ds, batch_size=bs, shuffle=True)
    vl = DataLoader(test_ds, batch_size=bs)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("\n--- Training LSTM ---")
    lstm = LSTMClassifier(len(w2i))
    train_pytorch_model(lstm, tl, vl, device)

    print("\n--- Training CNN ---")
    cnn = CNNClassifier(len(w2i))
    train_pytorch_model(cnn, tl, vl, device)

def download_gdrive_file(file_id, filename):
    if os.path.exists(filename):
        print(f"{filename} already exists in CWD.")
        return

    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    output = 'aclImdb.zip'
    gdown.download(url, output, quiet=False)

def prepare_dataset(zip_path):
    final_dir = './aclImdb'
    
    if os.path.exists(final_dir):
        print("Final dataset folder already exists. Skipping.")
        return
    else:
        download_gdrive_file("12XQAilUs1qEtKgg_t8OdBcPSgOdGIJvV", "aclImdb.zip")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall()

        for root, dirs, _ in os.walk('.', topdown=True):
            for dir_name in dirs:
                potential_path = os.path.join(root, dir_name)
                nested_acl_path = os.path.join(potential_path, 'aclImdb')
                if os.path.exists(nested_acl_path):
                    shutil.move(nested_acl_path, final_dir)
                    shutil.rmtree(potential_path)
                    print("Unzip and move completed.")
                    break

def main():
    prepare_dataset('aclImdb.zip')
    ensure_nltk_data()
    print("Loading data…")
    train_texts, train_labels = load_imdb_data('aclImdb/train')
    test_texts, test_labels = load_imdb_data('aclImdb/test')

    print("Preprocessing…")
    train_clean = preprocess_texts(train_texts)
    test_clean = preprocess_texts(test_texts)

    print("\n=== CLASSICAL ML ===")
    _ = train_evaluate_classical(train_clean, train_labels, test_clean, test_labels)

    print("\n=== DEEP LEARNING ===")
    train_deep_models(train_clean, train_labels, test_clean, test_labels)

if __name__ == '__main__':
    main()