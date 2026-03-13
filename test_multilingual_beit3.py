import torch
import torch.nn.functional as F
from dump_beit3 import load_beit3_model

def verify_with_csls(model, tokenizer):
    """
    Verifies alignment using CSLS metric to penalize 'hub' words.
    Based on Artetxe et al. (2016) principles.
    """
    model.eval()
    
    # Test pairs (Portuguese, English)
    test_pairs = [
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
    
    # 1. Pre-calculate all candidate embeddings for the target language (English)
    targets_en = [p[1] for p in test_pairs]
    en_vectors = []
    for word in targets_en:
        inputs = tokenizer(word, return_tensors="pt", add_special_tokens=False)
        with torch.no_grad():
            vec = model.get_input_embeddings().weight[inputs["input_ids"][0]].mean(dim=0)
            en_vectors.append(F.normalize(vec, p=2, dim=0))
    
    en_vectors = torch.stack(en_vectors) # Shape: [num_words, 1024]

    print("\n" + "="*55)
    print("   ALIGNMENT TEST: COSINE VS CSLS (Artetxe Method)")
    print("="*55)
    
    for pt, en_target in test_pairs:
        # Get Portuguese vector
        pt_inputs = tokenizer(pt, return_tensors="pt", add_special_tokens=False)
        with torch.no_grad():
            vec_pt = model.get_input_embeddings().weight[pt_inputs["input_ids"][0]].mean(dim=0)
            vec_pt = F.normalize(vec_pt, p=2, dim=0)
        
        # Calculate standard Cosine Similarity with all candidates
        cos_sims = F.cosine_similarity(vec_pt.unsqueeze(0), en_vectors)
        
        # Calculate CSLS penalty
        # rT(y) is the average similarity of the target to its neighbors
        # In this small-scale test, we use the mean of the current candidate pool
        r_pt = cos_sims.mean() 
        
        # CSLS Formula: 2 * Cos(x,y) - r(x) - r(y)
        # Here we use a simplified version: Cos(x,y) penalized by the local density
        csls_scores = 2 * cos_sims - r_pt
        
        # Get score for the specific target
        target_idx = targets_en.index(en_target)
        final_cos = cos_sims[target_idx].item()
        final_csls = csls_scores[target_idx].item()
        
        status = "✅" if final_csls > final_cos else "⚓"
        print(f"PT: {pt:12} -> EN: {en_target:10}")
        print(f"   Cosine: {final_cos:.4f} | CSLS: {final_csls:.4f} {status}")

    print("="*55 + "\n")

if __name__ == "__main__":
    MODEL_NAME = "beit3_large_patch16_224"
    CHECKPOINT_PATH = "beit3_large_indomain_patch16_224.pth"
    XLMR_EMB_PATH = "xlmr_in_beit3_space.pt"
    TOKENIZER_NAME = "xlm-roberta-large"

    model, tokenizer = load_beit3_model(MODEL_NAME, TOKENIZER_NAME, XLMR_EMB_PATH, CHECKPOINT_PATH)
    verify_with_csls(model, tokenizer)