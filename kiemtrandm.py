
import os, math, time
from pathlib import Path
from typing import List, Tuple
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import sentencepiece as spm
import sacrebleu

DATA_DIR = "./data"
EXPT_DIR = "./model"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(EXPT_DIR, exist_ok=True)

def file_exists_set(data_dir):
    req = ["train.de","train.en","valid.de","valid.en"]
    have = {f: Path(data_dir, f).exists() for f in req}
    return have, all(have.values())

have_map, all_ok = file_exists_set(DATA_DIR)

if not all_ok:
    demo_train = [
        ("hallo welt .", "hello world ."),
        ("ich liebe maschinelles lernen .", "i love machine learning ."),
        ("dies ist ein einfacher satz .", "this is a simple sentence ."),
        ("wie geht es dir ?", "how are you ?"),
        ("das ist ein buch .", "this is a book ."),
        ("das wetter ist heute schoen .", "the weather is nice today ."),
        ("ich habe hunger .", "i am hungry ."),
        ("ich trinke kaffee .", "i drink coffee ."),
        ("wo ist der bahnhof ?", "where is the train station ?"),
        ("ich lerne deutsch .", "i am learning german ."),
        ("ich spiele fussball gern .", "i like playing football ."),
        ("er geht zur schule .", "he goes to school ."),
        ("sie liest eine zeitung .", "she reads a newspaper ."),
        ("wir reisen morgen .", "we travel tomorrow ."),
        ("bitte sprechen sie langsam .", "please speak slowly ."),
        ("kannst du mir helfen ?", "can you help me ?"),
        ("ich verstehe nicht .", "i do not understand ."),
        ("wo ist das badezimmer ?", "where is the bathroom ?"),
        ("was kostet das ?", "how much is that ?"),
        ("danke schoen .", "thank you very much .")
    ]
    demo_valid = [
        ("guten morgen .", "good morning ."),
        ("gute nacht .", "good night ."),
        ("ich komme aus deutschland .", "i come from germany ."),
        ("er trinkt wasser .", "he drinks water ."),
        ("das ist lecker .", "that is tasty ."),
        ("bis spaeter !", "see you later !")
    ]
    with open(Path(DATA_DIR, "train.de"), "w", encoding="utf-8") as f1, open(Path(DATA_DIR, "train.en"), "w", encoding="utf-8") as f2:
        for de,en in demo_train:
            f1.write(de+"\n"); f2.write(en+"\n")
    with open(Path(DATA_DIR, "valid.de"), "w", encoding="utf-8") as f1, open(Path(DATA_DIR, "valid.en"), "w", encoding="utf-8") as f2:
        for de,en in demo_valid:
            f1.write(de+"\n"); f2.write(en+"\n")

def load_parallel(src_path, tgt_path):
    src_lines = Path(src_path).read_text(encoding="utf-8").splitlines()
    tgt_lines = Path(tgt_path).read_text(encoding="utf-8").splitlines()
    pairs = []
    for s, t in zip(src_lines, tgt_lines):
        s, t = s.strip(), t.strip()
        if s and t:
            pairs.append((s,t))
    return pairs

train_pairs = load_parallel(Path(DATA_DIR, "train.de"), Path(DATA_DIR, "train.en"))
valid_pairs = load_parallel(Path(DATA_DIR, "valid.de"), Path(DATA_DIR, "valid.en"))

SP_MODEL_PREFIX = Path(EXPT_DIR, "spm_de_en")
VOCAB_SIZE = 1000

joint_corpus = Path(EXPT_DIR, "joint_corpus.txt")
with open(joint_corpus, "w", encoding="utf-8") as f:
    for s,t in train_pairs:
        f.write(s+"\n"); f.write(t+"\n")
    for s,t in valid_pairs:
        f.write(s+"\n"); f.write(t+"\n")

if not Path(str(SP_MODEL_PREFIX)+".model").exists():
    spm.SentencePieceTrainer.Train(
        input=str(joint_corpus),
        model_prefix=str(SP_MODEL_PREFIX),
        vocab_size=VOCAB_SIZE,
        model_type='bpe',
        character_coverage=1.0,
        pad_id=0, pad_piece="<pad>",
        unk_id=1, unk_piece="<unk>",
        bos_id=2, bos_piece="<s>",
        eos_id=3, eos_piece="</s>"
    )

sp = spm.SentencePieceProcessor()
sp.load(str(SP_MODEL_PREFIX)+".model")
PAD_ID = sp.pad_id(); UNK_ID = sp.unk_id(); BOS_ID = sp.bos_id(); EOS_ID = sp.eos_id()

MAX_LEN = 64
BATCH_SIZE = 32

def encode_sentence(text: str):
    ids = sp.encode(text, out_type=int)
    ids = [BOS_ID] + ids + [EOS_ID]
    return ids[:MAX_LEN]

class ParallelDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs
    def __len__(self):
        return len(self.pairs)
    def __getitem__(self, idx):
        de, en = self.pairs[idx]
        return torch.tensor(encode_sentence(de)), torch.tensor(encode_sentence(en))

def collate_fn(batch):
    src, tgt = zip(*batch)
    max_s = max(len(x) for x in src); max_t = max(len(y) for y in tgt)
    def pad(list_tensors, L):
        out = torch.full((len(list_tensors), L), PAD_ID, dtype=torch.long)
        for i,t in enumerate(list_tensors):
            out[i,:len(t)] = t
        return out
    return pad(src, max_s), pad(tgt, max_t)

train_loader = DataLoader(ParallelDataset(train_pairs), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
valid_loader = DataLoader(ParallelDataset(valid_pairs), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))
        pe[:,0::2] = torch.sin(pos*div)
        pe[:,1::2] = torch.cos(pos*div)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x):
        return x + self.pe[:,:x.size(1)]

class TransformerSeq2Seq(nn.Module):
    def __init__(self, vocab, d_model=128, nhead=8, nlayers=2, ffn=512, dropout=0.1):
        super().__init__()
        self.emb_src = nn.Embedding(vocab, d_model, padding_idx=PAD_ID)
        self.emb_tgt = nn.Embedding(vocab, d_model, padding_idx=PAD_ID)
        self.pos = PositionalEncoding(d_model)
        self.tr = nn.Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=nlayers, num_decoder_layers=nlayers, dim_feedforward=ffn, dropout=dropout, batch_first=True)
        self.proj = nn.Linear(d_model, vocab)
    def forward(self, src, tgt_in):
        src_mask = (src==PAD_ID)
        tgt_mask = (tgt_in==PAD_ID)
        T = tgt_in.size(1)
        causal = torch.triu(torch.ones(T,T,dtype=torch.bool,device=tgt_in.device),1)
        src = self.pos(self.emb_src(src))
        tgt = self.pos(self.emb_tgt(tgt_in))
        mem = self.tr.encoder(src, src_key_padding_mask=src_mask)
        out = self.tr.decoder(tgt, mem, tgt_mask=causal, tgt_key_padding_mask=tgt_mask, memory_key_padding_mask=src_mask)
        return self.proj(out)

model = TransformerSeq2Seq(vocab=len(sp)).to(DEVICE)

torch.manual_seed(0)
LR = 1e-3
EPOCHS = 4
LABEL_SMOOTH = 0.1

criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID, label_smoothing=LABEL_SMOOTH)
optim = torch.optim.Adam(model.parameters(), betas=(0.9, 0.98), lr=LR)

use_amp = (DEVICE == "cuda")
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

def run_epoch(loader, train=True):
    model.train(train)
    tot, tok = 0.0, 0
    for src, tgt in loader:
        src, tgt = src.to(DEVICE, non_blocking=True), tgt.to(DEVICE, non_blocking=True)
        tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(src, tgt_in)
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
        if train:
            optim.zero_grad(set_to_none=True)
            if use_amp:
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optim)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
        ntok = (tgt_out != PAD_ID).sum().item()
        tot += loss.item() * ntok
        tok += ntok
    return tot / max(tok, 1)

best = float("inf")
ckpt = Path(EXPT_DIR, "best.pt")

for ep in range(1, EPOCHS + 1):
    t0 = time.time()
    tr = run_epoch(train_loader, True)
    va = run_epoch(valid_loader, False)
    if va < best:
        best = va
        torch.save({"model": model.state_dict(), "spm": str(Path(EXPT_DIR, "spm_de_en.model"))}, ckpt)
    print("Epoch %02d | trainCE %.4f | validCE %.4f | best %.4f | %.1fs" % (ep, tr, va, best, time.time()-t0))

@torch.no_grad()
def greedy_decode(src_text, max_len=64):
    model.eval()
    s = torch.tensor([[2] + sp.encode(src_text, out_type=int) + [3]], device=DEVICE)
    ys = torch.tensor([[2]], device=DEVICE)
    for _ in range(max_len):
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(s, ys)
        nxt = logits[0, -1].argmax(-1).item()
        ys = torch.cat([ys, torch.tensor([[nxt]], device=DEVICE)], dim=1)
        if nxt == 3:
            break
    out = ys[0, 1:].tolist()
    if 3 in out:
        out = out[:out.index(3)]
    return sp.decode(out)

state = torch.load(ckpt, map_location=DEVICE)
model.load_state_dict(state["model"])

n_show = min(5, len(valid_pairs))
print("Demo translations:")
for i in range(n_show):
    src_txt = valid_pairs[i][0]
    hyp = greedy_decode(src_txt)
    print("[%d] DE: %s" % (i, src_txt))
    print("    EN: %s" % hyp)

MAX_EVAL = min(500, len(valid_pairs))
refs = [[en for _, en in valid_pairs[:MAX_EVAL]]]
hyps = [greedy_decode(de) for de, _ in valid_pairs[:MAX_EVAL]]
print("BLEU on validation subset:", sacrebleu.corpus_bleu(hyps, refs).format())
print("Checkpoint saved to:", ckpt)
