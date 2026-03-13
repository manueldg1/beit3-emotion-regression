import torch
from modeling_finetune import beit3_large_patch16_480_valence_arousal

# 1. Carica il modello (usa la tua architettura specifica)
model = beit3_large_patch16_480_valence_arousal()

# 2. Carica l'ultimo checkpoint salvato
checkpoint = torch.load("./output/va_test_mse_ablation/checkpoint-best.pth", map_location="cpu")
model.load_state_dict(checkpoint['model'])
model.eval()

# 3. Crea un input "dummy" (un'immagine e un testo casuali)
img = torch.randn(1, 3, 480, 480)
text = torch.randint(0, 1000, (1, 64))
padding_mask = torch.zeros(1, 64).bool()

# 4. Estrai l'embedding (prima dello strato finale di regressione)
with torch.no_grad():
    # Nota: il nome del metodo dipende da come è strutturato il tuo beit3
    outputs = model.beit3.encoder(img, text, padding_mask)
    embedding = outputs[0][:, 0, :] # Prendi il CLS token

print(f"Somma assoluta embedding: {torch.sum(torch.abs(embedding)).item()}")
print(f"Primi 5 valori: {embedding[0, :5]}")
