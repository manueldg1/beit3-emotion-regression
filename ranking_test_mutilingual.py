import torch
import torch.nn.functional as F
import modeling_finetune
from dump_beit3 import load_beit3_model

def test_ranking_multilingual(model, tokenizer):
    model.eval()
    
    # Coppie di test: (Query Portoghese, Target Inglese)
    test_cases = [
        ("cachorro", "dog"),
        ("gato", "cat"),
        ("montanha", "mountain"),
        ("maçã", "apple"),
        ("sol", "sun"),
        ("computador", "computer"),
        ("carro", "car"),
        ("casa", "house"),
        ("oceano", "ocean"),
        ("pão", "bread")
    ]

    # Lista fissa di candidati inglesi per il ranking
    candidates_en = [pair[1] for pair in test_cases]

    print(f"\n" + "="*50)
    print(f"   RANKING TEST: PORTOGHESE -> INGLESE")
    print(f"="*50)

    for query_pt, target_en in test_cases:
        # 1. Embedding Query (Portoghese)
        pt_inputs = tokenizer(query_pt, return_tensors="pt", add_special_tokens=False)
        with torch.no_grad():
            pt_vec = model.get_input_embeddings().weight[pt_inputs["input_ids"][0]].mean(dim=0)
            pt_vec = F.normalize(pt_vec, p=2, dim=0)

        # 2. Calcolo similarità con tutti i candidati inglesi
        scores = []
        for word_en in candidates_en:
            en_inputs = tokenizer(word_en, return_tensors="pt", add_special_tokens=False)
            with torch.no_grad():
                en_vec = model.get_input_embeddings().weight[en_inputs["input_ids"][0]].mean(dim=0)
                en_vec = F.normalize(en_vec, p=2, dim=0)
            
            sim = F.cosine_similarity(pt_vec.unsqueeze(0), en_vec.unsqueeze(0)).item()
            scores.append((word_en, sim))

        # 3. Ordinamento
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Trova la posizione del target corretto
        rank = [s[0] for s in scores].index(target_en) + 1
        top_match = scores[0][0]
        top_sim = scores[0][1]

        status = "✅" if rank == 1 else "⚠️"
        print(f"PT: {query_pt:12} | Target: {target_en:10} | Rank: {rank}/10 {status}")
        if rank > 1:
            print(f"   -> Top Match: {top_match} (Sim: {top_sim:.4f})")

    print("="*50 + "\n")

if __name__ == "__main__":
    MODEL_NAME = "beit3_large_patch16_480_valence_arousal"
    CHECKPOINT_PATH = "beit3_large_indomain_patch16_224.pth"
    XLMR_EMB_PATH = "Multilingual_projection/xlmr_fixed.pt"
    TOKENIZER_NAME = "xlm-roberta-large"

    model, tokenizer = load_beit3_model(MODEL_NAME, TOKENIZER_NAME, XLMR_EMB_PATH, CHECKPOINT_PATH)
    test_ranking_multilingual(model, tokenizer)
