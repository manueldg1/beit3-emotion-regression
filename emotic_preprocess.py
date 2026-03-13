import scipy.io
import pandas as pd
import os

def prepare_emotic_minimal(mat_path, output_csv):
    print("Caricamento Annotations.mat...")
    mat = scipy.io.loadmat(mat_path)
    
    all_data = []

    # EMOTIC divide in train, val, test. Li scorriamo tutti.
    for split in ['train', 'val', 'test']:
        if split not in mat:
            continue
            
        print(f"Elaborazione split: {split}...")
        subset = mat[split][0]
        
        for entry in subset:
            # folder (es. 'mscoco') e filename (es. 'img.jpg')
            folder = entry[0][0]
            filename = entry[1][0]
            
            # person_entries contiene i dati delle persone nella foto
            person_entries = entry[4][0]
            
            for person in person_entries:
                try:
                    # Estrazione VAD (Indice 3)
                    # Scala originale EMOTIC: 1-10
                    vad = person[3][0][0]
                    v_orig = float(vad[0])
                    a_orig = float(vad[1])
                    
                    # --- NORMALIZZAZIONE [-1, 1] ---
                    # Formula: 2 * ((valore - min) / (max - min)) - 1
                    # Con min=1 e max=10, diventa: 2 * ((valore - 1) / 9) - 1
                    valence = 2 * ((v_orig - 1) / 9) - 1
                    arousal = 2 * ((a_orig - 1) / 9) - 1
                    
                    # Costruzione del percorso completo
                    image_path = os.path.join(folder, filename)
                    
                    all_data.append({
                        'image_path': image_path,
                        'valence': round(valence, 4),
                        'arousal': round(arousal, 4)
                    })
                except (IndexError, TypeError, ValueError):
                    # Salta se mancano dati o se la struttura è corrotta
                    continue

    # Creazione DataFrame e salvataggio
    df = pd.DataFrame(all_data)
    df.to_csv(output_csv, index=False)
    print(f"\nOperazione completata!")
    print(f"Totale annotazioni estratte: {len(df)}")
    print(df.head())

# Esegui il comando (assicurati che il path sia corretto)
prepare_emotic_minimal("Annotations/Annotations.mat", "emotic_v_a_only.csv")