import datasets
import numpy as np
from PIL import Image
from pyts.image import GramianAngularField
from typing import Dict, Union, Callable, List, Optional
import warnings

class GenomicImageGenerator:
    """
    Una classe per trasformare un dataset di sequenze genomiche di Hugging Face
    in un dataset di immagini, utilizzando varie tecniche di rappresentazione.

    Questa classe è progettata per preparare i dati per l'addestramento di modelli
    di computer vision (es. con timm e HF Trainer).
    """

    def __init__(
        self,
        image_size: int,
        sequence_col: str = "sequence",
        label_col: str = "label",
    ):
        """
        Inizializza il generatore.

        Args:
            image_size (int): La dimensione (altezza e larghezza) dell'immagine quadrata in output.
            sequence_col (str): Il nome della colonna nel dataset contenente le sequenze genomiche.
            label_col (str): Il nome della colonna nel dataset contenente le etichette.
        """
        if image_size <= 0:
            raise ValueError("image_size must be a positive integer.")
            
        self.image_size = image_size
        self.sequence_col = sequence_col
        self.label_col = label_col

        # Mappatura standard dei nucleotidi per le rappresentazioni numeriche
        self.dna_map = {'A': 0.25, 'C': 0.50, 'G': 0.75, 'T': 1.0, 'N': 0.0}
        self._cgr_map = {'A': (0, 0), 'C': (0, 1), 'G': (1, 1), 'T': (1, 0)}

    def _preprocess_sequence(self, sequence: str, target_len: int, strategy: str = "replication") -> str:
        """
        Normalizza la sequenza a una lunghezza fissa e gestisce caratteri non validi.

        Args:
            sequence (str): La sequenza di DNA di input.
            target_len (int): La lunghezza desiderata per la sequenza.
            strategy (str): Strategia per gestire la lunghezza ('replication' per le corte, 'truncate' per le lunghe).
        
        Returns:
            str: La sequenza processata.
        """
        # Filtra solo i caratteri validi
        valid_chars = "ACGT" # Rimuoviamo 'N' per non includerlo nel replication padding
        sequence = "".join([char.upper() for char in sequence if char.upper() in valid_chars])

        if not sequence: # Se la sequenza è vuota dopo il filtraggio
            return 'N' * target_len

        seq_len = len(sequence)

        if seq_len > target_len:
            # Il troncamento è ancora una soluzione sub-ottimale. 
            # Un approccio 'tiling' sarebbe migliore ma più complesso da implementare qui.
            return sequence[:target_len]
        
        if seq_len < target_len:
            if strategy == "replication":
                # REPLICATION PADDING: ripete la sequenza per preservare le proprietà statistiche.
                repeats = (target_len // seq_len) + 1
                return (sequence * repeats)[:target_len]
            else: # Default a padding con 'N' se specificato
                return sequence + 'N' * (target_len - seq_len)
                
        return sequence
    
    def _validate_and_warn(self, hf_dataset: datasets.Dataset):
        """
        Controlla la coerenza tra la dimensione dell'immagine e la lunghezza media
        delle sequenze su un campione del dataset, generando warnings se necessario.
        """
        sample_size = min(1000, len(hf_dataset)) # Campiona al massimo 1000 sequenze
        if sample_size == 0:
            return # Non fare nulla se il dataset è vuoto

        # Calcola la lunghezza media delle sequenze sul campione
        sample_sequences = hf_dataset.select(range(sample_size))[self.sequence_col]
        avg_seq_len = sum(len(s) for s in sample_sequences) / sample_size
        
        num_pixels = self.image_size ** 2

        # --- Warning per immagini potenzialmente troppo grandi (Sparsity) ---
        # Heuristica: se ci sono > 20 volte più pixel che basi nella sequenza media,
        # l'immagine sarà probabilmente molto vuota.
        sparsity_ratio = 20
        if num_pixels > sparsity_ratio * avg_seq_len:
           # Metodo consigliato: usare le parentesi per unire la stringa su più righe
            warnings.warn(
                (
                    f"La dimensione dell'immagine ({self.image_size}x{self.image_size} = {num_pixels} pixel) "
                    f"è significativamente maggiore della lunghezza media delle sequenze (~{avg_seq_len:.0f} basi)."
                    "\nLe immagini generate, in particolare la CGR, potrebbero risultare molto sparse."
                ),
                UserWarning
            )
            
    def _normalize_matrix(self, matrix: np.ndarray) -> np.ndarray:
        """Normalizza una matrice in un range [0, 1]."""
        min_val, max_val = matrix.min(), matrix.max()
        if max_val == min_val:
            return np.zeros_like(matrix)
        return (matrix - min_val) / (max_val - min_val)

    # ===================================================================
    # Tecniche di Generazione Immagini (Metodi Privati)
    # ===================================================================

    def _generate_cgr(self, sequence: str) -> np.ndarray:
        """Genera una matrice tramite Frequency Chaos Game Representation (CGR)."""
        cgr_matrix = np.zeros((self.image_size, self.image_size))
        
        # Prepara la sequenza per la CGR (non è necessario che abbia una lunghezza specifica)
        seq = self._preprocess_sequence(sequence, target_len=len(sequence))
        if not seq:
            return cgr_matrix

        # Punto di partenza al centro
        last_pos = np.array([0.5, 0.5])

        for base in seq:
            corner = self._cgr_map.get(base)
            if corner is None:
                continue
            
            # Calcola il punto medio
            last_pos = (last_pos + np.array(corner)) / 2.0
            
            # Mappa le coordinate [0,1] alla dimensione dell'immagine
            img_x = int(last_pos[0] * self.image_size)
            img_y = int(last_pos[1] * self.image_size)
            
            # Incrementa la frequenza in quel pixel
            if 0 <= img_x < self.image_size and 0 <= img_y < self.image_size:
                cgr_matrix[img_x, img_y] += 1
        
        return self._normalize_matrix(cgr_matrix)

    def _generate_hilbert(self, sequence: str) -> np.ndarray:
        """Genera una matrice mappando la sequenza su una curva di Hilbert."""
        # L'ordine p della curva genera 4^p punti. Troviamo p tale che 4^p >= image_size^2
        p = int(np.ceil(np.log(self.image_size**2) / np.log(4)))
        curve_len = 4**p
        
        # Prepara e mappa la sequenza a valori numerici
        seq = self._preprocess_sequence(sequence, target_len=curve_len)
        numeric_seq = np.array([self.dna_map.get(base, 0.0) for base in seq])

        # Matrice di output
        hilbert_matrix = np.zeros((2**p, 2**p))
        
        for i in range(curve_len):
            x, y = self._hilbert_integer_to_xy(i, p)
            if x < self.image_size and y < self.image_size:
                hilbert_matrix[x, y] = numeric_seq[i]
        
        # Ritaglia alla dimensione desiderata
        final_matrix = hilbert_matrix[:self.image_size, :self.image_size]
        return self._normalize_matrix(final_matrix) # Già normalizzato da dna_map, ma sicuro

    def _hilbert_integer_to_xy(self, i: int, p: int) -> tuple[int, int]:
        """Converte un intero i alle coordinate (x,y) sulla curva di Hilbert di ordine p."""
        states = [(0, 0), (0, 1), (1, 1), (1, 0)]
        x = y = 0
        for j in range(p - 1, -1, -1):
            s = 2**j
            quadrant = (i >> (2*j)) & 3
            dx, dy = states[quadrant]
            x += s * dx
            y += s * dy
            # Ruota e rifletti per il prossimo quadrante
            i, states = self._hilbert_rotate(i, quadrant, states)
        return x, y

    def _hilbert_rotate(self, i: int, quadrant: int, states: list) -> tuple[int, list]:
        """Helper per la rotazione/riflessione della curva di Hilbert."""
        if quadrant == 0:
            states = [states[1], states[0], states[3], states[2]]
            return i, [(y, x) for x, y in states]
        if quadrant == 3:
            states = [states[3], states[2], states[1], states[0]]
            return i, [(1-y, 1-x) for x, y in states]
        return i, states
        
    def _generate_gaf(self, sequence: str) -> np.ndarray:
        """Genera una matrice tramite Gramian Angular Fields (GAF)."""
        # GAF richiede una lunghezza minima
        seq_len = self.image_size * self.image_size
        seq = self._preprocess_sequence(sequence, target_len=seq_len)
        numeric_seq = np.array([self.dna_map.get(base, 0.0) for base in seq]).reshape(1, -1)
        
        # Usa GAF per trasformare la serie numerica in un'immagine
        gaf = GramianAngularField(image_size=self.image_size, method='summation')
        gaf_matrix = gaf.fit_transform(numeric_seq)[0]
        
        return self._normalize_matrix(gaf_matrix)

    # ===================================================================
    # Metodi Pubblici per la Generazione di Dataset
    # ===================================================================

    def _get_generation_function(self, method: str) -> Callable[[str], np.ndarray]:
        """Restituisce la funzione di generazione appropriata in base al metodo richiesto."""
        method_map = {
            "cgr": self._generate_cgr,
            "hilbert": self._generate_hilbert,
            "gaf": self._generate_gaf,
        }
        if method.lower() not in method_map:
            raise ValueError(f"Metodo '{method}' non supportato. Scegliere tra {list(method_map.keys())}.")
        return method_map[method.lower()]
        
    def generate_single_channel_dataset(
        self,
        hf_dataset: datasets.Dataset,
        method: str,
    ) -> datasets.Dataset:
        """
        Trasforma un dataset HF generando immagini a singolo canale con la tecnica specificata.
        """
        gen_func = self._get_generation_function(method)
        self._validate_and_warn(hf_dataset)

        def apply_transform(examples: Dict[str, List]) -> Dict[str, List]:
            images = []
            for seq in examples[self.sequence_col]:
                matrix = gen_func(seq)
                img_array = (matrix * 255).astype(np.uint8)
                images.append(Image.fromarray(img_array))
            return {"image": images}

        processed_ds = hf_dataset.map(
            apply_transform,
            batched=True,
            remove_columns=[self.sequence_col],
        )
        
        # Ora processed_ds contiene ['image', 'label'], quindi il cast funzionerà.
        # Il cast è ancora utile per assicurare che la colonna 'image' abbia il tipo corretto.
        features = datasets.Features({
            'image': datasets.Image(),
            'label': hf_dataset.features[self.label_col],
        })
        
        final_ds = processed_ds.cast(features)
        return final_ds

    def _generate_3_channel_image_from_sequence(self, sequence: str) -> Image.Image:
        """
        Crea una singola immagine a 3 canali da una sequenza.

        Canale R: CGR
        Canale G: Hilbert Curve
        Canale B: GAF
        """
        cgr_matrix = self._generate_cgr(sequence)
        hilbert_matrix = self._generate_hilbert(sequence)
        gaf_matrix = self._generate_gaf(sequence)
        
        # Stack delle matrici normalizzate per creare un'immagine a 3 canali
        # e conversione a 8-bit (0-255)
        rgb_array = (np.stack([cgr_matrix, hilbert_matrix, gaf_matrix], axis=-1) * 255).astype(np.uint8)
        
        return Image.fromarray(rgb_array, 'RGB')
        

    def generate_3_channel_dataset(self, hf_dataset: datasets.Dataset) -> datasets.Dataset:
        """
        Trasforma un dataset HF generando immagini a 3 canali (RGB).
        """
        def apply_transform(examples: Dict[str, List]) -> Dict[str, List]:
            images = [self._generate_3_channel_image_from_sequence(seq) for seq in examples[self.sequence_col]]
            return {"image": images}
        
        self._validate_and_warn(hf_dataset)

        processed_ds = hf_dataset.map(
            apply_transform,
            batched=True,
            remove_columns=[self.sequence_col],
        )
        
        features = datasets.Features({
            'image': datasets.Image(decode=True),
            'label': hf_dataset.features[self.label_col],
        })
        
        final_ds = processed_ds.cast(features)
        return final_ds
    

