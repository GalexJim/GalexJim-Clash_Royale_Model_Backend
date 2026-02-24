"""
SISTEMA RECOMENDADOR INTERACTIVO V3.0
Análisis estratégico avanzado con:
- Win conditions principal/secundaria
- Arquetipos
- Sinergias
- Balance aire/tierra
- Economía de elixir
- Análisis de roles
"""

import numpy as np
import pandas as pd
import os
import sys
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import warnings
import random
import re
from typing import List, Tuple, Dict, Set
from dataclasses import dataclass
from collections import Counter
import traceback

warnings.filterwarnings('ignore')

np.random.seed(42)
torch.manual_seed(42)
random.seed(42)


# ============================================
# METADATOS DE CARTAS
# ============================================

# ============================================
# METADATOS DE CARTAS (OBTENIDOS EXCLUSIVAMENTE DEL DATASET)
# ============================================

# Nota: El sistema exige que la metadata provenga del dataset. No se usan
# constantes embebidas como fallback; si faltan columnas requeridas, se
# lanzará un error para que el usuario corrija el archivo de entrada.

# ============================================
# DATASET DE COMBOS Y MAZOS (del prompt del usuario)
# ============================================
CLASSIC_COMBOS = [
    ["Sabueso de Lava", "Globo Bombástico"],
    ["Gigante", "Bruja", "Bebé Dragón"],
    ["Montapuercos", "Espíritu Hielo", "Descarga (Zap)"],
    ["Minero", "Veneno"],
    ["P.E.K.K.A", "Mago Eléctrico"],
    ["Barril de Duendes", "Princesa", "El Tronco"],
    ["Golem", "Leñador"],
]

FULL_DECK_TEMPLATES = {
    "LavaLoon Control": [
        "Sabueso de Lava", "Globo Bombástico", "Mega Caballero", "Bola de Fuego",
        "Flechas", "Lápida", "Dragón Infernal", "Murciélagos"
    ],
    "Hog Cycle Clásico": [
        "Montapuercos", "Mosquetera", "Espíritu Hielo", "Descarga (Zap)",
        "Cañón", "Bola de Fuego", "Esqueletos", "El Tronco"
    ],
    "P.E.K.K.A Bridge Control": [
        "P.E.K.K.A", "Bandida", "Mago Eléctrico", "Veneno",
        "Barril de Bárbaro", "Caballero", "Torre Infernal", "Murciélagos"
    ],
    "Gigante Cementerio": [
        "Gigante", "Cementerio", "Bebé Dragón", "Veneno",
        "Bola de Nieve", "Lápida", "Esbirros", "Caballero"
    ],
    "Log Bait Tradicional": [
        "Barril de Duendes", "Princesa", "Caballero", "Duendes",
        "Ejército de Esq.", "Torre Infernal", "El Tronco", "Cohete"
    ],
    "Golem Beatdown": [
        "Golem", "Bruja", "Bebé Dragón", "Leñador",
        "Bola de Fuego", "Flechas", "Mega Caballero", "Murciélagos"
    ],
}

@dataclass
class Config:
    embedding_dim: int = 32
    hidden_dim: int = 64
    num_heads: int = 4
    dropout: float = 0.3
    batch_size: int = 32
    learning_rate: float = 0.001
    num_epochs: int = 30
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    synthetic_num_decks: int = 500
    attack_damage_threshold: float = 30.0


config = Config()


class RealDatasetLoader:
    def __init__(self, file_path: str = None):
        self.file_path = file_path
        self.cards_df = None
        self.load_data()
        
    def build_card_metadata(self) -> Dict[str, Dict]:
        """Construye un diccionario de metadata por carta a partir del dataset.
        Exige que la información provenga del dataset: si faltan columnas críticas
        se lanzará una excepción para que el usuario corrija el archivo.
        """
        df = self.cards_df
        if df is None:
            raise ValueError('Dataset no cargado. Proporciona un archivo válido.')

        # Validar columna de nombres (requerida)
        if 'Nombre' not in df.columns and 'name' not in df.columns:
            raise ValueError("Columna 'Nombre' (o 'name') no encontrada en el dataset.")

        card_metadata = {}
        # Nombres de columnas posibles para cada campo (tolerancia)
        col_map = {
            'win_condition': ['win_condition', 'win condition', 'win_condition_principal', 'win_condition_principal', 'win'],
            'arquetipo': ['arquetipo', 'arquetipo_mazo', 'archetype'],
            'roles': ['roles', 'role', 'roles_list'],
            'tiene_ataque_aereo': ['tiene_ataque_aereo', 'ataque_aereo', 'air_attack', 'tiene_ataque_aereo?'],
            'contra_tanques': ['contra_tanques', 'anti_tank', 'contra_tanque'],
            'ciclo': ['ciclo', 'is_cycle', 'cycle'],
            'hechizo_pequeno': ['hechizo_pequeno', 'small_spell', 'hechizo_pequeño'],
            'hechizo_fuerte': ['hechizo_fuerte', 'big_spell', 'hechizo_fuerte?'],
            'antiaereo': ['antiaereo', 'anti_air', 'antiaéreo'],
            'control_hordas': ['control_hordas', 'horde_control']
        }

        # Normalize column names to lower for lookup
        lower_cols = {c.lower(): c for c in df.columns}

        def find_col(names):
            for n in names:
                if n.lower() in lower_cols:
                    return lower_cols[n.lower()]
            return None

        # Helper to parse boolean-like values
        def parse_bool(v):
            if pd.isna(v):
                return False
            if isinstance(v, (int, float)):
                return bool(v)
            s = str(v).strip().lower()
            return s in ('1', 'true', 'si', 'sí', 'yes', 'y', 't')

        roles_col = find_col(col_map['roles'])

        for _, row in df.iterrows():
            name = row.get('Nombre') if 'Nombre' in row else row.get('name') if 'name' in row else None
            if not name:
                continue
            meta = {}
            # Simple string fields: tomar valores desde el dataset o None
            for key in ('win_condition', 'arquetipo'):
                col = find_col(col_map[key])
                if col and col in df.columns:
                    val = row.get(col)
                    meta[key] = None if pd.isna(val) else str(val).strip()
                else:
                    meta[key] = None

            # Roles parsing: devolver lista vacía si no existe
            if roles_col and roles_col in df.columns:
                raw = row.get(roles_col)
                if pd.isna(raw):
                    meta['roles'] = []
                else:
                    if isinstance(raw, (list, tuple)):
                        meta['roles'] = list(raw)
                    else:
                        s = str(raw)
                        parts = [p.strip() for p in re.split(r'[;,\|/]', s) if p.strip()]
                        if len(parts) == 1 and parts[0].startswith('[') and parts[0].endswith(']'):
                            inner = parts[0][1:-1]
                            parts = [p.strip().strip("'\"") for p in inner.split(',') if p.strip()]
                        meta['roles'] = parts
            else:
                meta['roles'] = []

            # Boolean flags: tomar False por defecto si no existe la columna
            for flag in ('tiene_ataque_aereo', 'contra_tanques', 'ciclo', 'hechizo_pequeno', 'hechizo_fuerte', 'antiaereo', 'control_hordas'):
                col = find_col(col_map[flag])
                if col and col in df.columns:
                    meta[flag] = parse_bool(row.get(col))
                else:
                    meta[flag] = False

            card_metadata[str(name)] = meta

        return card_metadata
    
    def _find_first_dataset(self):
        # Buscar en el directorio de trabajo archivos comunes de dataset
        patterns = ['*.xlsx', '*.xls', '*.csv', '*.json']
        for pat in patterns:
            files = glob.glob(pat)
            if files:
                return files[0]
        return None

    def load_data(self):
        # Si no se dio ruta, intentar detectar automáticamente
        if not self.file_path:
            found = self._find_first_dataset()
            if found:
                print(f"[INFO] Usando dataset detectado: {found}")
                self.file_path = found
            else:
                raise FileNotFoundError('No se proporcionó dataset y no se encontró ninguno en el directorio actual.')

        _, ext = os.path.splitext(self.file_path)
        ext = ext.lower()
        try:
            if ext in ['.xlsx', '.xls']:
                self.cards_df = pd.read_excel(self.file_path, sheet_name=0)
            elif ext == '.csv':
                self.cards_df = pd.read_csv(self.file_path)
            elif ext == '.json':
                self.cards_df = pd.read_json(self.file_path)
            else:
                # Intentar leer como excel, luego csv
                try:
                    self.cards_df = pd.read_excel(self.file_path, sheet_name=0)
                except Exception:
                    self.cards_df = pd.read_csv(self.file_path)
        except Exception as e:
            print(f"Error cargando dataset '{self.file_path}': {e}")
            raise
    
    def get_cards_with_features(self) -> pd.DataFrame:
        df = self.cards_df.copy()
        df['Costo'] = pd.to_numeric(df['Costo'], errors='coerce').fillna(3)
        df['Daño_num'] = df['Daño (Unidad / Gen.)'].astype(str).str.extract(r'(\d+)')[0]
        df['Daño_num'] = pd.to_numeric(df['Daño_num'], errors='coerce').fillna(10)
        df['Vida_num'] = df['Vida (Unidad / Gen.)'].astype(str).str.extract(r'(\d+)')[0]
        df['Vida_num'] = pd.to_numeric(df['Vida_num'], errors='coerce').fillna(50)
        df['alcance_num'] = df['Alcance'].astype(str).str.extract(r'(\d+\.?\d*)')[0]
        df['alcance_num'] = pd.to_numeric(df['alcance_num'], errors='coerce').fillna(1)
        return df
    
    def get_card_names(self) -> List[str]:
        return self.cards_df['Nombre'].tolist()


class DeckAnalyzer:
    def __init__(self, cards_df: pd.DataFrame, card_names: List[str], card_metadata: Dict = None):
        self.cards_df = cards_df
        self.card_names = card_names
        self.card_metadata = card_metadata or {}
    
    def analyze(self, deck_indices: np.ndarray) -> Dict:
        analysis = {}
        deck_cards = [self.cards_df.iloc[idx] for idx in deck_indices]
        deck_names = [name for name, idx in zip(self.card_names, range(len(self.card_names))) if idx in deck_indices]
        
        main_wc = [name for name in deck_names if self.card_metadata.get(name, {}).get('win_condition') == 'main']
        secondary_wc = [name for name in deck_names if self.card_metadata.get(name, {}).get('win_condition') == 'secondary']
        analysis['win_condition_principal'] = main_wc[0] if main_wc else None
        analysis['win_condition_secundaria'] = secondary_wc[0] if secondary_wc else None
        
        analysis['costo_promedio'] = np.mean([float(card['Costo']) for card in deck_cards])
        analysis['cartas_ciclo'] = len([card for card in deck_cards if float(card['Costo']) <= 3])
        analysis['ciclo_bajo'] = len([name for name in deck_names if self.card_metadata.get(name, {}).get('ciclo', False)])

        analysis['tiene_hechizo_pequeno'] = any([self.card_metadata.get(name, {}).get('hechizo_pequeno', False) for name in deck_names])
        analysis['tiene_hechizo_fuerte'] = any([self.card_metadata.get(name, {}).get('hechizo_fuerte', False) for name in deck_names])
        analysis['tiene_antiaereo'] = any([self.card_metadata.get(name, {}).get('antiaereo', False) for name in deck_names])
        analysis['tiene_contra_tanques'] = any([self.card_metadata.get(name, {}).get('contra_tanques', False) for name in deck_names])
        
        aerial = len([name for name in deck_names if self.card_metadata.get(name, {}).get('tiene_ataque_aereo', False)])
        analysis['balance_aereo'] = aerial
        
        arquetipos = [self.card_metadata.get(name, {}).get('arquetipo') for name in deck_names]
        arquetipos_filtrados = [a for a in arquetipos if a]
        if arquetipos_filtrados:
            arquetipo_mas_comun = Counter(arquetipos_filtrados).most_common(1)[0][0]
        else:
            arquetipo_mas_comun = 'mixed'
        analysis['arquetipo'] = arquetipo_mas_comun
        
        return analysis


class DeckGenerator:
    def __init__(self, cards_list: List[str], num_decks: int = 500):
        self.cards_list = cards_list
        self.num_decks = num_decks
        self.num_cards = len(cards_list)
    
    def generate_decks(self, deck_size: int = 8) -> np.ndarray:
        decks = []
        for _ in range(self.num_decks):
            deck = np.random.choice(self.num_cards, size=deck_size, replace=False)
            decks.append(deck)
        return np.array(decks)
    
    def generate_incomplete_decks(self, decks: np.ndarray, deck_size: int = 7):
        incomplete_decks = decks[:, :deck_size]
        target_cards = decks[:, deck_size]
        return incomplete_decks, target_cards


class DataPreprocessor:
    def __init__(self, cards_df: pd.DataFrame, num_cards: int):
        self.cards_df = cards_df
        self.num_cards = num_cards
        self.scaler = StandardScaler()
        self.card_features = self._extract_features()
    
    def _extract_features(self) -> np.ndarray:
        # Construir características de forma robusta: costo, daño, vida, alcance, tipo_codificado
        features_list = []
        for i in range(min(len(self.cards_df), self.num_cards)):
            row = self.cards_df.iloc[i]
            costo = float(row.get('Costo', 3)) if 'Costo' in row else 3.0
            dano = float(row.get('Daño_num', 10)) if 'Daño_num' in row else 10.0
            vida = float(row.get('Vida_num', 50)) if 'Vida_num' in row else 50.0
            alcance = float(row.get('alcance_num', 1.0)) if 'alcance_num' in row else 1.0
            tipo_raw = row.get('Tipo', '') if 'Tipo' in row else ''
            # Codificar tipo de carta a valor numérico estable
            if isinstance(tipo_raw, str):
                tipo = tipo_raw.strip().lower()
            else:
                tipo = ''
            if tipo == 'hechizo':
                tipo_code = 1.0
            elif tipo == 'estructura':
                tipo_code = 0.5
            elif tipo == 'tropa':
                tipo_code = 0.8
            else:
                tipo_code = 0.6

            features_list.append([costo, dano, vida, alcance, tipo_code])

        # Si hay menos filas que num_cards, rellenar con ceros
        feature_dim = 5
        features = np.zeros((self.num_cards, feature_dim), dtype=float)
        for i, vec in enumerate(features_list):
            features[i, :len(vec)] = vec

        self.feature_dim = feature_dim
        return self.scaler.fit_transform(features)
    
    def encode_deck(self, deck: np.ndarray) -> torch.Tensor:
        deck_features = []
        for card_idx in deck:
            if card_idx < len(self.card_features):
                deck_features.append(self.card_features[card_idx])
        while len(deck_features) < 7:
            deck_features.append(np.zeros(4))
        return torch.tensor(np.array(deck_features[:7]), dtype=torch.float32)
    
    def get_card_features(self, card_idx: int) -> torch.Tensor:
        if card_idx < len(self.card_features):
            return torch.tensor(self.card_features[card_idx], dtype=torch.float32)
        return torch.zeros(4, dtype=torch.float32)


class ClashRoyaleDataset(Dataset):
    def __init__(self, incomplete_decks: np.ndarray, target_cards: np.ndarray, preprocessor: DataPreprocessor):
        self.incomplete_decks = incomplete_decks
        self.target_cards = target_cards
        self.preprocessor = preprocessor
    
    def __len__(self) -> int:
        return len(self.incomplete_decks)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        deck = self.incomplete_decks[idx]
        target = self.target_cards[idx]
        deck_encoded = self.preprocessor.encode_deck(deck)
        target_features = self.preprocessor.get_card_features(target)
        return deck_encoded, target_features


class AttentionLayer(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int = 4):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.queries = nn.Linear(embedding_dim, embedding_dim)
        self.keys = nn.Linear(embedding_dim, embedding_dim)
        self.values = nn.Linear(embedding_dim, embedding_dim)
        self.fc_out = nn.Linear(embedding_dim, embedding_dim)
    
    def forward(self, deck_embeddings: torch.Tensor) -> torch.Tensor:
        Q = self.queries(deck_embeddings)
        K = self.keys(deck_embeddings)
        V = self.values(deck_embeddings)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.embedding_dim)
        attention = torch.softmax(scores, dim=-1)
        context = torch.matmul(attention, V)
        out = self.fc_out(context)
        return out


class CardRecommendationModel(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int, num_heads: int, dropout: float, input_dim: int = 4, output_dim: int = 4):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.embedding = nn.Linear(input_dim, embedding_dim)
        self.attention = AttentionLayer(embedding_dim, num_heads)
        flattened_dim = embedding_dim * 7
        self.fc1 = nn.Linear(flattened_dim, hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(hidden_dim // 2, output_dim)
    
    def forward(self, deck: torch.Tensor) -> torch.Tensor:
        batch_size = deck.shape[0]
        embedded = self.embedding(deck)
        attended = self.attention(embedded)
        x = attended.reshape(batch_size, -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x


class CardRecommendationSystemV3:
    def __init__(self, model: CardRecommendationModel, preprocessor: DataPreprocessor, 
                 cards_df: pd.DataFrame, card_names: List[str], config: Config, card_metadata: Dict = None):
        self.model = model
        self.preprocessor = preprocessor
        self.cards_df = cards_df
        self.card_names = card_names
        self.config = config
        self.device = torch.device(config.device)
        self.model.eval()
        self.card_metadata = card_metadata or {}
        self.deck_analyzer = DeckAnalyzer(cards_df, card_names, self.card_metadata)
    
    def _analyze_deck_needs(self, incomplete_deck: np.ndarray) -> Dict:
        """Analiza qué necesita el mazo específicamente"""
        needs = {
            'tiene_tropa_ataque_aereo': False,
            'tiene_estructura': False,
            'tiene_hechizo': False,
            'ataque_aereo_fuerte_tropas': 0  # Contar tropas con ataque aéreo fuerte
        }
        
        for card_idx in incomplete_deck:
            if card_idx < len(self.card_names):
                card_name = self.card_names[int(card_idx)]
                card_type = self.cards_df.iloc[int(card_idx)]['Tipo']
                metadata = self.card_metadata.get(card_name, {})
                
                if card_type == 'Estructura':
                    needs['tiene_estructura'] = True
                elif card_type == 'Hechizo':
                    needs['tiene_hechizo'] = True
                elif card_type == 'Tropa':
                    if metadata.get('tiene_ataque_aereo'):
                        needs['tiene_tropa_ataque_aereo'] = True
                        # Contar si tiene daño fuerte (win_condition o damage_dealer)
                        if 'damage_dealer' in metadata.get('roles', []) or metadata.get('win_condition'):
                            needs['ataque_aereo_fuerte_tropas'] += 1
        
        return needs
    
    def _get_card_priority_score(self, card_idx: int, deck_needs: Dict, base_similarity: float) -> float:
        """Calcula prioridad de una carta basada en necesidades del mazo"""
        if card_idx >= len(self.card_names):
            return base_similarity
        
        card_name = self.card_names[card_idx]
        card_type = self.cards_df.iloc[card_idx]['Tipo']
        metadata = self.card_metadata.get(card_name, {})
        
        priority_boost = 1.0
        
        # PRIORIDAD MÁXIMA: Tropas con ataque aéreo fuerte si no hay suficientes
        if card_type == 'Tropa' and metadata.get('tiene_ataque_aereo'):
            if 'damage_dealer' in metadata.get('roles', []) or metadata.get('win_condition'):
                if deck_needs['ataque_aereo_fuerte_tropas'] < 2:
                    priority_boost *= 2.5  # Boost máximo
        
        # PRIORIDAD ALTA: Estructura defensiva si no hay
        elif card_type == 'Estructura' and not deck_needs['tiene_estructura']:
            priority_boost *= 2.0
        
        # PRIORIDAD MEDIA: Hechizo si no hay
        elif card_type == 'Hechizo' and not deck_needs['tiene_hechizo']:
            priority_boost *= 1.8
        
        # PRIORIDAD NORMAL: Solamente ajuste por tipo
        elif card_type == 'Hechizo':
            priority_boost *= 1.2
        elif card_type == 'Estructura':
            priority_boost *= 1.1
        
        return base_similarity * priority_boost

    def _compute_synergy_score(self, deck_card_names: List[str], recommended_card_name: str,
                               base_similarity: float, deck_analysis: Dict) -> float:
        """Calcula una puntuación de eficiencia realista (0..1) combinando la similitud base
        con bonificaciones por sinergia de combos y mazos completos.
        La salida se limita por debajo de 1.0 para evitar porcentajes irreales.
        """
        base = float(np.clip(base_similarity, 0.0, 1.0))

        total_bonus = 0.0

        # Bonificaciones por combos clásicos
        for combo in CLASSIC_COMBOS:
            if recommended_card_name in combo:
                others = [c for c in combo if c != recommended_card_name]
                present = sum(1 for c in others if c in deck_card_names)
                if present == len(others) and len(others) > 0:
                    total_bonus += 0.35
                elif present > 0:
                    total_bonus += 0.12 * present

        # Bonificaciones por similitud con mazos completos (plantillas)
        for tpl_name, tpl_list in FULL_DECK_TEMPLATES.items():
            if recommended_card_name in tpl_list:
                shared = sum(1 for c in tpl_list if c in deck_card_names)
                # escalar modestamente según cuantas cartas coinciden
                total_bonus += 0.03 * shared

        # Bonificación por arquetipo coincidente
        rec_meta = self.card_metadata.get(recommended_card_name, {})
        if rec_meta.get('arquetipo') and rec_meta.get('arquetipo') == deck_analysis.get('arquetipo'):
            total_bonus += 0.08

        # Bonificaciones por llenar necesidades del mazo
        if not deck_analysis.get('tiene_antiaereo') and rec_meta.get('antiaereo'):
            total_bonus += 0.12
        if not deck_analysis.get('tiene_hechizo_fuerte') and rec_meta.get('hechizo_fuerte'):
            total_bonus += 0.12

        # Límite de bonificación acumulada para evitar valores sobrestimados
        total_bonus = min(total_bonus, 0.45)

        # Combinar base y bonificaciones en un factor realista
        # base multiplica un rango entre 0.6 y 1.05 (con cap posterior)
        efficiency = base * (0.6 + total_bonus)
        efficiency = float(np.clip(efficiency, 0.0, 0.97))
        return efficiency
    
    def recommend_card(self, incomplete_deck: np.ndarray, top_k: int = 3, focus_type: str = None):
        with torch.no_grad():
            deck_encoded = self.preprocessor.encode_deck(incomplete_deck).unsqueeze(0)
            deck_encoded = deck_encoded.to(self.device)
            output = self.model(deck_encoded).squeeze(0)
            output = output.cpu().numpy()
        
        deck_analysis = self.deck_analyzer.analyze(incomplete_deck)
        deck_needs = self._analyze_deck_needs(incomplete_deck)
        
        similarities = []
        cards_in_deck = set(incomplete_deck)
        deck_card_names = [self.card_names[int(i)] for i in incomplete_deck if int(i) < len(self.card_names)]
        
        for card_idx in range(len(self.card_names)):
            if card_idx not in cards_in_deck:
                card_features = self.preprocessor.get_card_features(card_idx).numpy()
                similarity = np.dot(output, card_features) / (np.linalg.norm(output) * np.linalg.norm(card_features) + 1e-8)
                similarity_normalized = (similarity + 1) / 2
                
                if focus_type:
                    if card_idx < len(self.cards_df):
                        card_name = self.card_names[card_idx] if card_idx < len(self.card_names) else f"Carta {card_idx}"
                        card_type = self.cards_df.iloc[card_idx]['Tipo']
                        
                        # Obtener metadata de la carta
                        card_meta = self.card_metadata.get(card_name, {})
                        
                        if focus_type == 'ataque':
                            # Relajamos el filtro de ataque: aceptar
                            # - Tropas que sean win_condition o damage_dealer
                            # - Hechizos fuertes (damage) como opciones de ataque
                            roles = card_meta.get('roles', [])
                            is_tropa = (card_type == 'Tropa')
                            is_hechizo_fuerte = bool(card_meta.get('hechizo_fuerte'))
                            has_damage_role = any(r for r in roles if 'damage' in str(r).lower() or 'damage_dealer' in str(r).lower())

                            # Heurística basada en la columna numérica 'Daño_num'
                            try:
                                dano_num = float(self.cards_df.iloc[card_idx].get('Daño_num', 0) or 0)
                            except Exception:
                                dano_num = 0.0
                            damage_threshold = self.config.attack_damage_threshold if hasattr(self, 'config') else 30.0

                            if not (is_tropa or is_hechizo_fuerte):
                                # No es ni tropa ni hechizo útil para ataque
                                continue

                            # Si es tropa, requerir que tenga rol de daño o win_condition
                            if is_tropa:
                                # Aceptar si es win_condition o tiene rol de daño o si su daño numérico es alto
                                if card_meta.get('win_condition') is None and not has_damage_role and dano_num < damage_threshold:
                                    continue
                            # Si es hechizo fuerte, aceptarlo (ya cubierto por is_hechizo_fuerte)
                        elif focus_type == 'defensa':
                            # Cartas con roles defensivos específicos
                            roles = card_meta.get('roles', [])
                            
                            # Definir qué es defensivo
                            is_defensive = (
                                'estructura_defensiva' in roles or
                                'antiaereo' in roles or
                                'control_hordas' in roles or
                                'soporte' in roles
                            )
                            
                            # Para hechizos, solo incluir si tienen rol defensivo
                            if card_type == 'Hechizo':
                                if not is_defensive:
                                    continue
                            # Para tropas
                            elif card_type == 'Tropa':
                                if not is_defensive:
                                    continue
                            # Todas las estructuras son defensivas
                            # (card_type == 'Estructura')
                            
                            # Aplicar boost de prioridad según necesidades del mazo
                            similarity_normalized = self._get_card_priority_score(card_idx, deck_needs, similarity_normalized)
                
                # Calcular eficiencia realista basada en sinergias y la similitud base
                card_name = self.card_names[card_idx] if card_idx < len(self.card_names) else f"Carta {card_idx}"
                efficiency = self._compute_synergy_score(deck_card_names, card_name, similarity_normalized, deck_analysis)
                similarities.append((card_idx, efficiency))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Si es defensa, diversificar los tipos de cartas recomendadas
        if focus_type == 'defensa':
            diversified = []
            seen_types = set()
            seen_cards = set()
            
            # Primera pasada: obtener un card de cada tipo
            for card_idx, similarity in similarities:
                if card_idx < len(self.cards_df):
                    card_type = self.cards_df.iloc[card_idx]['Tipo']
                    
                    if card_type not in seen_types:
                        diversified.append((card_idx, similarity))
                        seen_types.add(card_type)
                        seen_cards.add(card_idx)
            
            # Segunda pasada: llenar el resto con las mejores recomendaciones
            if len(diversified) < top_k:
                for card_idx, similarity in similarities:
                    if card_idx not in seen_cards:
                        diversified.append((card_idx, similarity))
                        seen_cards.add(card_idx)
                        if len(diversified) >= top_k:
                            break
            
            similarities = diversified[:top_k]
        else:
            similarities = similarities[:top_k]
        
        recommendations = []
        
        # Calcular y adjuntar el costo promedio del mazo si se añade cada carta recomendada
        current_costs = [float(self.cards_df.iloc[int(i)]['Costo']) for i in incomplete_deck if int(i) < len(self.cards_df)]
        current_sum = sum(current_costs)

        for card_idx, similarity in similarities:
            card_name = self.card_names[card_idx] if card_idx < len(self.card_names) else f"Carta {card_idx}"
            explanation = self._generate_advanced_explanation(incomplete_deck, card_idx, deck_analysis, focus_type)

            # Obtener costo de la carta recomendada y calcular promedio sobre 8 cartas
            try:
                rec_cost = float(self.cards_df.iloc[card_idx]['Costo']) if card_idx < len(self.cards_df) else float(np.nan)
            except Exception:
                rec_cost = float(np.nan)

            avg_with_card = (current_sum + (rec_cost if not np.isnan(rec_cost) else 0.0)) / 8.0
            recommendations.append((card_idx, card_name, similarity, explanation, avg_with_card))
        
        return recommendations, deck_analysis
    
    def _generate_advanced_explanation(self, deck: np.ndarray, recommended_card_idx: int, 
                                       deck_analysis: Dict, focus_type: str = None) -> str:
        if recommended_card_idx >= len(self.cards_df):
            return "Completa tu mazo"
        
        card = self.cards_df.iloc[recommended_card_idx]
        card_name = self.card_names[recommended_card_idx]
        metadata = self.card_metadata.get(card_name, {})
        explanations = []
        
        if metadata.get('win_condition') == 'main':
            explanations.append("win condition principal")
        elif metadata.get('win_condition') == 'secondary':
            explanations.append("win condition secundaria")
        
        if not deck_analysis.get('tiene_hechizo_pequeno') and metadata.get('hechizo_pequeno'):
            explanations.append("añade hechizo pequeno")
        
        if not deck_analysis.get('tiene_hechizo_fuerte') and metadata.get('hechizo_fuerte'):
            explanations.append("añade hechizo fuerte")
        
        if not deck_analysis.get('tiene_antiaereo') and metadata.get('antiaereo'):
            explanations.append("mejora defensa aerea")
        
        if not deck_analysis.get('tiene_contra_tanques') and metadata.get('contra_tanques'):
            explanations.append("contra tanques enemigos")
        
        arquetipo = metadata.get('arquetipo')
        if arquetipo == deck_analysis.get('arquetipo'):
            explanations.append(f"sinergia {arquetipo}")
        
        if float(card['Costo']) <= deck_analysis['costo_promedio']:
            explanations.append("economica en elixir")
        
        return " + ".join(explanations) if explanations else "opcion flexible"


def main():
    print("\n" + "="*70)
    print("SISTEMA RECOMENDADOR INTERACTIVO V3.0 - ANALISIS AVANZADO")
    print("="*70)
    
    # Permitir especificar dataset por argumento de línea de comandos
    dataset_path = None
    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]

    # Si no se pasó argumento, usar detección automática sin pedir al usuario
    loader = RealDatasetLoader(dataset_path)
    cards_df = loader.get_cards_with_features()
    card_names = loader.get_card_names()
    # Construir metadata dinámicamente desde el dataset (estrictamente sin fallbacks)
    try:
        card_metadata = loader.build_card_metadata()
    except Exception as e:
        print(f"[ERROR] No se pudo construir metadata desde el dataset: {e}")
        sys.exit(1)
    num_cards = len(cards_df)
    
    print(f"\n[OK] {num_cards} cartas cargadas\n")
    
    deck_gen = DeckGenerator(list(range(num_cards)), config.synthetic_num_decks)
    full_decks = deck_gen.generate_decks(deck_size=8)
    incomplete_decks, target_cards = deck_gen.generate_incomplete_decks(full_decks, deck_size=7)
    
    preprocessor = DataPreprocessor(cards_df, num_cards)
    dataset = ClashRoyaleDataset(incomplete_decks, target_cards, preprocessor)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)
    
    input_dim = preprocessor.card_features.shape[1]
    model = CardRecommendationModel(config.embedding_dim, config.hidden_dim, 
                                   config.num_heads, config.dropout, input_dim=input_dim, output_dim=input_dim)
    device = torch.device(config.device)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()
    
    print("[ENTRENANDO MODELO...]")
    for epoch in range(config.num_epochs):
        model.train()
        total_loss = 0
        for deck_batch, target_batch in train_loader:
            deck_batch = deck_batch.to(device)
            target_batch = target_batch.to(device)
            optimizer.zero_grad()
            outputs = model(deck_batch)
            loss = criterion(outputs, target_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch + 1:2d}/30: Loss = {total_loss / len(train_loader):.4f}")
    
    recommendation_system = CardRecommendationSystemV3(
        model, preprocessor, cards_df, card_names, config, card_metadata
    )
    
    print("\n" + "="*70)
    print("[OK] MODELO LISTO - INTERFAZ INTERACTIVA")
    print("="*70 + "\n")
    
    interactive_recommender_v3(recommendation_system, card_names, cards_df)


def interactive_recommender_v3(recommendation_system, card_names, cards_df):
    """Interfaz interactiva mejorada con múltiples opciones"""
    
    while True:
        print("\n" + "="*70)
        print("RECOMENDADOR DE CARTAS V3.0")
        print("="*70)
        print("\nCartas disponibles:")
        for idx, name in enumerate(card_names, 1):
            costo = cards_df.iloc[idx-1]['Costo']
            tipo = cards_df.iloc[idx-1]['Tipo']
            print(f"  {idx:2d}. {name:25s} ({tipo:10s}) - Costo: {costo}")
        
        print("\n" + "="*70)
        print("INGRESA TU MAZO (7 CARTAS)")
        print("="*70)
        
        deck_indices = []
        used_cards = set()
        
        for i in range(7):
            while True:
                try:
                    user_input = input(f"Carta {i+1}: ").strip()
                    
                    if user_input.lower() == 'salir':
                        print("\nGracias por usar el recomendador!")
                        return
                    
                    card_num = int(user_input)
                    
                    if card_num < 1 or card_num > len(card_names):
                        print(f"Error: Numero debe estar entre 1 y {len(card_names)}")
                        continue
                    
                    if card_num in used_cards:
                        print("Error: Ya usaste esa carta")
                        continue
                    
                    used_cards.add(card_num)
                    deck_indices.append(card_num - 1)
                    print(f"  OK: {card_names[card_num - 1]}")
                    break
                except ValueError:
                    print("Error: Ingresa un numero valido")
        
        deck_array = np.array(deck_indices)
        
        print("\n" + "="*70)
        print("TU MAZO")
        print("="*70)
        for i, idx in enumerate(deck_indices, 1):
            card = cards_df.iloc[idx]
            print(f"  {i}. {card_names[idx]:30s} ({card['Tipo']}) - Costo: {card['Costo']}")
        
        print("\n" + "="*70)
        # Permitir cambiar umbral de daño para foco 'ataque'
        try:
            current_thresh = recommendation_system.config.attack_damage_threshold
        except Exception:
            current_thresh = None

        if current_thresh is not None:
            while True:
                resp = input(f"Valor umbral de daño para ATAQUE actual = {current_thresh}. Ingresa un nuevo valor o Enter para mantener: ").strip()
                if resp == "":
                    break
                try:
                    new_val = float(resp)
                    recommendation_system.config.attack_damage_threshold = new_val
                    print(f"Umbral de daño para ataque actualizado a {new_val}")
                    break
                except ValueError:
                    print("Entrada inválida. Ingresa un número (ej. 30) o Enter para mantener el valor actual.")

        print("TIPO DE MEJORA")
        print("="*70)
        while True:
            mejora = input("Quieres mejorar (0. ataque, 1. defensa): ").strip().lower()
            if mejora in ['0', 'ataque']:
                mejora_tipo = 'ataque'
                break
            elif mejora in ['1', 'defensa'] or 'defensa' in mejora or 'foco en defensa' in mejora:
                mejora_tipo = 'defensa'
                break
            else:
                print("Entrada no válida. Por favor, ingresa 0/ataque o 1/defensa.")
        
        print("\n[ANALIZANDO MAZO...]")
        try:
            all_recommendations, analysis = recommendation_system.recommend_card(deck_array, top_k=9, focus_type=mejora_tipo)
        except Exception as e:
            print(f"[ERROR] Al generar recomendaciones: {e}")
            print(traceback.format_exc())
            print("Vuelve a intentar o revisa el dataset/metadata.")
            continue
        
        print("\n" + "="*70)
        print("ANALISIS DEL MAZO")
        print("="*70)
        print(f"Win Condition Principal: {analysis['win_condition_principal'] or 'Sin definir'}")
        print(f"Win Condition Secundaria: {analysis['win_condition_secundaria'] or 'Sin definir'}")
        print(f"Costo promedio: {analysis['costo_promedio']:.1f} elixir")
        print(f"Cartas ciclo bajo: {analysis['ciclo_bajo']}/7")
        print(f"Arquetipo: {analysis['arquetipo'].upper()}")
        print(f"Balance aereo: {analysis['balance_aereo']}/7")
        print(f"Tiene hechizo pequeno: {'SI' if analysis['tiene_hechizo_pequeno'] else 'NO'}")
        print(f"Tiene hechizo fuerte: {'SI' if analysis['tiene_hechizo_fuerte'] else 'NO'}")
        print(f"Antiaereo: {'SI' if analysis['tiene_antiaereo'] else 'NO'}")
        
        # Mostrar recomendaciones en lotes de 3
        current_index = 0
        
        while True:
            # Mostrar siguiente lote de 3 recomendaciones
            if current_index < len(all_recommendations):
                next_batch = all_recommendations[current_index:current_index + 3]
                
                print("\n" + "="*70)
                print("RECOMENDACIONES")
                print("="*70 + "\n")
                
                for rank, (card_idx, card_name, score, explanation, avg_cost) in enumerate(next_batch, current_index + 1):
                    percentage = min(score * 100, 100.0)  # Capear a 100% máximo
                    print(f"{rank}. {card_name:35s} - Efectividad: {percentage:5.1f}%")
                    print(f"   {explanation}")
                    print(f"   Costo promedio si se añade: {avg_cost:.1f} elixir\n")
                
                current_index += 3
                
                # Preguntar qué hacer ahora
                print("="*70)
                if current_index < len(all_recommendations):
                    # Aún hay más opciones sin mostrar
                    while True:
                        opcion = input("\nDeseas (0. ver mas opciones, 1. probar otro mazo, 2. salir): ").strip()
                        if opcion in ['0', 'mas', 'más']:
                            # Preguntar nuevo tipo de mejora
                            print("\n" + "="*70)
                            print("TIPO DE MEJORA")
                            print("="*70)
                            while True:
                                nueva_mejora = input("Quieres mejorar (0. ataque, 1. defensa): ").strip().lower()
                                if nueva_mejora in ['0', 'ataque']:
                                    mejora_tipo = 'ataque'
                                    break
                                elif nueva_mejora in ['1', 'defensa'] or 'defensa' in nueva_mejora or 'foco en defensa' in nueva_mejora:
                                    mejora_tipo = 'defensa'
                                    break
                                else:
                                    print("Entrada no válida. Por favor, ingresa 0/ataque o 1/defensa.")
                            
                            # Regenerar recomendaciones con nuevo filtro
                            print("\n[ANALIZANDO MAZO...]")
                            try:
                                all_recommendations, _ = recommendation_system.recommend_card(deck_array, top_k=9, focus_type=mejora_tipo)
                            except Exception as e:
                                print(f"[ERROR] Al generar recomendaciones: {e}")
                                print(traceback.format_exc())
                                print("Volver al menu principal.")
                                break
                            current_index = 0
                            break
                        elif opcion in ['1', 'otro']:
                            print("\n" + "="*70)
                            return
                        elif opcion in ['2', 'salir']:
                            print("\nGracias por usar el recomendador V3.0!")
                            return
                        else:
                            print("Opcion no valida. Escribe 0, 1 o 2")
                else:
                    # No hay más opciones sin mostrar
                    while True:
                        opcion = input("\nNo hay mas opciones. Deseas (0. cambiar filtro, 1. probar otro mazo, 2. salir): ").strip()
                        if opcion in ['0', 'cambiar']:
                            # Preguntar nuevo tipo de mejora
                            print("\n" + "="*70)
                            print("TIPO DE MEJORA")
                            print("="*70)
                            while True:
                                nueva_mejora = input("Quieres mejorar (0. ataque, 1. defensa): ").strip().lower()
                                if nueva_mejora in ['0', 'ataque']:
                                    mejora_tipo = 'ataque'
                                    break
                                elif nueva_mejora in ['1', 'defensa'] or 'defensa' in nueva_mejora or 'foco en defensa' in nueva_mejora:
                                    mejora_tipo = 'defensa'
                                    break
                                else:
                                    print("Entrada no válida. Por favor, ingresa 0/ataque o 1/defensa.")
                            
                            # Regenerar recomendaciones con nuevo filtro
                            print("\n[ANALIZANDO MAZO...]")
                            try:
                                all_recommendations, _ = recommendation_system.recommend_card(deck_array, top_k=9, focus_type=mejora_tipo)
                            except Exception as e:
                                print(f"[ERROR] Al generar recomendaciones: {e}")
                                print(traceback.format_exc())
                                print("Volver al menu principal.")
                                break
                            current_index = 0
                            break
                        elif opcion in ['1', 'otro']:
                            print("\n" + "="*70)
                            return
                        elif opcion in ['2', 'salir']:
                            print("\nGracias por usar el recomendador V3.0!")
                            return
                        else:
                            print("Opcion no valida. Escribe 0, 1 o 2")
            else:
                # Todas las recomendaciones mostradas
                break


if __name__ == "__main__":
    main()
