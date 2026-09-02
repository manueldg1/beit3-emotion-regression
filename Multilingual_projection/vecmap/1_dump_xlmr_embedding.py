"""
Export XLM-RoBERTa-Large embeddings to Word2Vec text format
(required by VecMap) into a file named "xlmr.txt".
"""

from transformers import XLMRobertaTokenizer, XLMRobertaModel

OUTPUT_TXT = "xlmr.txt"


def main():
    # 1) load the pretrained tokenizer and model
    tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-large")
    model = XLMRobertaModel.from_pretrained("xlm-roberta-large")

    # 2) get ordered vocab list
    #    (tokenizer.get_vocab() returns a dict token -> id)
    vocab = sorted(tokenizer.get_vocab().items(), key=lambda x: x[1])
    tokens, ids = zip(*vocab)

    # 3) extract embedding matrix (|V| x D)
    embs = model.embeddings.word_embeddings.weight.detach().cpu().numpy()

    # 4) write Word2Vec text file
    D = embs.shape[1]
    with open(OUTPUT_TXT, "w", encoding="utf-8") as fout:
        fout.write(f"{len(tokens)} {D}\n")
        for tok, vec in zip(tokens, embs):
            vec_str = " ".join(f"{x:.6f}" for x in vec)
            fout.write(f"{tok} {vec_str}\n")

    print(f"Wrote {OUTPUT_TXT} with {len(tokens)} tokens x {D} dimensions")


if __name__ == "__main__":
    main()
