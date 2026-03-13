import pandas as pd
import os

def prepare_wikiart(root_dir, output_csv):
    """
    root_dir: la cartella che contiene le sottocartelle (es. 'cubism', 'impressionism')
    """
    all_data = []
    
    # Lista delle cartelle (stili/generi)
    styles = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    
    print(f"Trovati {len(styles)} stili/cartelle in WikiArt.")

    for style in styles:
        style_path = os.path.join(root_dir, style)
        images = [f for f in os.listdir(style_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for img in images:
            # Creiamo il percorso relativo: 'nome_stile/immagine.jpg'
            relative_path = os.path.join(style, img)
            
            # Qui dovresti mappare Valence/Arousal da un file esterno se disponibile.
            # Se non li hai, inseriamo dei segnaposto (placeholder)
            all_data.append({
                'image_path': relative_path,
                'valence': 0.0, # Placeholder: da sostituire con i valori reali
                'arousal': 0.0  # Placeholder: da sostituire con i valori reali
            })

    df = pd.DataFrame(all_data)
    df.to_csv(output_csv, index=False)
    print(f"Fatto! Salvate {len(df)} immagini di WikiArt in {output_csv}")

# ESECUZIONE
# Assicurati di puntare alla cartella che contiene le sottocartelle delle immagini
prepare_wikiart("/cfs/home/u036743/unilm/beit3/wikiart", "wikiart_v_a.csv")


# Carica i percorsi che abbiamo appena estratto
df_paths = pd.read_csv("wikiart_v_a.csv")

# Carica le annotazioni reali (esempio se avessi un file 'wikiart_emotions.csv')
# Questo file deve avere una colonna con il nome del file e i valori V/A
df_emotions = pd.read_csv("wikiart_emotions.csv") 

# Esempio di normalizzazione se i valori sono 1-7
# df_emotions['valence'] = 2 * ((df_emotions['valence'] - 1) / 6) - 1

# Unione finale
df_final = pd.merge(df_paths, df_emotions, on='image_path', how='inner')