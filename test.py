from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import torch, pickle, nltk
import torch.nn as nn

nltk.download('punkt')
nltk.download('stopwords')

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

def load_vocab(path='vocab.pkl'):
    with open(path, 'rb') as f:
        return pickle.load(f)

def load_model(w2i, path):
    model = CNNClassifier(len(w2i))
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

def preprocess(text):
    sw = set(stopwords.words('english'))
    toks = word_tokenize(text.lower())
    return [w for w in toks if w.isalpha() and w not in sw]

def to_indices(toks, w2i, max_len=200):
    unk = w2i.get('<UNK>', 1)
    idxs = [w2i.get(w, unk) for w in toks]
    return idxs[:max_len] + [0] * (max_len - len(idxs))

def predict_sentiment(text, model, w2i, max_len=200):
    tokens = preprocess(text)
    indices = to_indices(tokens, w2i, max_len)
    x = torch.LongTensor([indices])
    with torch.no_grad():
        prob = model(x).item()
        label = int(prob > 0.5)
    return label, prob

def main(input_text):
    w2i = load_vocab()
    model = load_model(w2i, 'model.pt')
    label, prob = predict_sentiment(input_text, model, w2i)
    
    print("\nReview Text:")
    print(f"\"{input_text}\"\n")
    percent = prob * 100
    if label == 1:
        print(f"Predicted Sentiment: Positive ({percent:.1f}% confidence)")
    else:
        print(f"Predicted Sentiment: Negative ({100 - percent:.1f}% confidence)")

if __name__ == "__main__":
    input_text = "Yet even by the depressing standards set by the Mortal Kombat movies, Uncharted and the first two miserable Sonic the Hedgehog outings, this third Sonic is staggeringly poor."
    main(input_text)

    input_text = "Worst movie ever."
    main(input_text)

    input_text = "The best movie ever."
    main(input_text)