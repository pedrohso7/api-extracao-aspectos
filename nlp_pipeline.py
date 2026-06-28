import re
import unicodedata
import pandas as pd
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from pysentimiento import create_analyzer

# Stopwords personalizadas de aspecto do projeto original
STOPWORDS_PT_ASPECTS = {
    "bom", "otimo", "excelente", "maravilhoso", "perfeito", "parabens", "sucesso", "incrivel",
    "gostei", "produtivo", "ruim", "lento", "caindo", "pessimo", "dificil", "erro", "falha",
    "amei", "adorando", "legal", "massa", "feliz", "obrigado", "obrigada", "parabenizar",
    "tudo", "nada", "muito", "dia", "hoje", "evento", "comentario", "comentarios",
    "ano", "vez", "certeza", "gente", "coisa", "alguem", "lindo", "linda", "top", "show",
    "nao", "sim", "ja", "estou", "quero", "wit", "csbc", "ainda"
}

def normalizar_texto(texto):
    texto = str(texto).lower()
    texto = ''.join(c for c in unicodedata.normalize('NFD', texto) if unicodedata.category(c) != 'Mn')
    return texto

def dividir_sentencas(texto):
    padrao = re.compile(r'[.!?]|\be\b|\bmas\b|\bporem\b|\bcontudo\b|\btodavia\b')
    partes = padrao.split(str(texto))
    
    partes_limpas = []
    for p in partes:
        p_clean = p.strip()
        if len(p_clean) > 4:
            partes_limpas.append(p_clean)
            
    return partes_limpas if partes_limpas else [str(texto).strip()]

class AspectExtractionPipeline:
    def __init__(self, distance_threshold=0.3):
        self.distance_threshold = distance_threshold
        
        # Carregamento do spaCy (POS Tagging)
        self.nlp = spacy.load("pt_core_news_sm")
        
        # Carregamento do SentenceTransformer (Embeddings Multilíngue)
        self.encoder = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2').to("cpu")
        
        # Carregamento do pysentimiento (Análise de Sentimentos com Transformers)
        self.sentiment_analyzer = create_analyzer(task="sentiment", lang="pt")
        
    def processar(self, comments_list):
        """
        Recebe uma lista de dicionários [{"id": "uuid1", "comentario": "texto"}, ...]
        Retorna uma lista estruturada de arestas (usuário -> aspecto) com scores de sentimento.
        """
        
        # 1. Divisão de Sentenças (Sentence Splitting)
        expanded_records = []
        for item in comments_list:
            usuario = item.get('id', 'anonimo')
            texto_original = item.get('comentario', '')
            
            oracoes = dividir_sentencas(texto_original)
            for oracao in oracoes:
                expanded_records.append({
                    "usuario": usuario,
                    "texto": oracao
                })
        
        df_split = pd.DataFrame(expanded_records)
        if df_split.empty:
            return []
            
        # 2. Extração Física de Candidatos a Aspecto (POS Tagging)
        frases_com_entidades = []
        entidades_unicas = set()
        
        for i, row in df_split.iterrows():
            comentario = row['texto']
            usuario = row['usuario']
            doc = self.nlp(str(comentario))
            
            entidades_frase = []
            for token in doc:
                if token.pos_ in ['NOUN', 'PROPN']:
                    entidade_limpa = normalizar_texto(token.text).strip()
                    if len(entidade_limpa) > 2 and entidade_limpa not in STOPWORDS_PT_ASPECTS:
                        entidades_frase.append(entidade_limpa)
                        entidades_unicas.add(entidade_limpa)
                        
            frases_com_entidades.append({
                "usuario": usuario, 
                "texto": comentario, 
                "entidades": entidades_frase
            })
            
        entidades_lista = list(entidades_unicas)
        
        if len(entidades_lista) == 0:
            return []
            
        # 3. Agrupamento Semântico (Clustering)
        embeddings = self.encoder.encode(entidades_lista, show_progress_bar=False, batch_size=32)
        
        clustering_model = AgglomerativeClustering(
            n_clusters=None, 
            distance_threshold=self.distance_threshold, 
            metric='cosine', 
            linkage='average'
        )
        cluster_labels = clustering_model.fit_predict(embeddings)
        
        freq_entidades = {}
        for item in frases_com_entidades:
            for ent in item['entidades']: 
                freq_entidades[ent] = freq_entidades.get(ent, 0) + 1
                
        cluster_canonical = {}
        for label in set(cluster_labels):
            entidades_no_cluster = [entidades_lista[i] for i, l in enumerate(cluster_labels) if l == label]
            cluster_canonical[label] = max(entidades_no_cluster, key=lambda e: freq_entidades.get(e, 0)).capitalize()
            
        entidade_para_aspecto = {ent: cluster_canonical[cluster_labels[i]] for i, ent in enumerate(entidades_lista)}
        
        # 4. Construção das arestas e Análise de Sentimento (pysentimiento)
        dados_finais = []
        for item in frases_com_entidades:
            if not item['entidades']: 
                continue
                
            aspectos_mencionados = set([entidade_para_aspecto[e] for e in item['entidades']])
            
            # Executa o pysentimiento para a frase inteira
            texto_frase = item['texto']
            resultado_sentimento = self.sentiment_analyzer.predict(texto_frase)
            
            # pysentimiento retorna probabilidades para POS, NEG, NEU.
            # Convertendo para um score contínuo entre [-1, 1] similar ao VADER original.
            probas = resultado_sentimento.probas
            score_continuo = probas['POS'] - probas['NEG']
            
            # Mapeamento descritivo da polaridade
            polaridade_descritiva = "Neutro"
            if score_continuo > 0.3:
                polaridade_descritiva = "Positivo"
            elif score_continuo < -0.3:
                polaridade_descritiva = "Negativo"
            
            for aspecto in aspectos_mencionados:
                dados_finais.append({
                    "usuario_id": item['usuario'],
                    "texto_analisado": texto_frase,
                    "aspecto": aspecto,
                    "sentimento_score": score_continuo,
                    "polaridade": polaridade_descritiva
                })
                
        return dados_finais
