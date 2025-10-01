#!/usr/bin/env python3
"""
Coletor Automático de Imagens de Gestos de Libras da Internet
Baixa imagens e processa automaticamente para criar base de dados
"""

import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import os
import requests
from urllib.parse import urljoin, urlparse
import time
from datetime import datetime
import json
from PIL import Image
import io

class ColetorInternetLibras:
    def __init__(self):
        # Inicializar MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        # Letras para coletar
        self.letras_para_coletar = [
            'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'M', 'N', 
            'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Z'
        ]
        
        # URLs de referência para busca (exemplos)
        self.urls_referencia = [
            "https://www.signingsavvy.com",
            "https://www.lifeprint.com",
            "https://www.handspeak.com"
        ]
        
        # Headers para requests
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        # Criar diretórios
        os.makedirs('imagens_coletadas', exist_ok=True)
        os.makedirs('dados_processados', exist_ok=True)
        
        self.dados_coletados = []
    
    def baixar_imagem(self, url, nome_arquivo):
        """Baixa uma imagem da URL"""
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            # Verificar se é uma imagem
            content_type = response.headers.get('content-type', '')
            if not content_type.startswith('image/'):
                return False
            
            # Salvar imagem
            with open(nome_arquivo, 'wb') as f:
                f.write(response.content)
            
            return True
        except Exception as e:
            print(f"❌ Erro ao baixar {url}: {e}")
            return False
    
    def gerar_urls_busca(self, letra):
        """Gera URLs de busca para uma letra específica"""
        termos_busca = [
            f"libras letter {letra}",
            f"sign language {letra}",
            f"ASL letter {letra}",
            f"gesto libras {letra}",
            f"letra {letra} libras"
        ]
        
        # URLs de busca em imagens (exemplos)
        urls_busca = []
        for termo in termos_busca:
            # Google Images (exemplo - na prática seria necessário usar API)
            urls_busca.append(f"https://www.google.com/search?q={termo}&tbm=isch")
            
        return urls_busca
    
    def criar_imagens_sinteticas(self, letra):
        """Cria imagens sintéticas baseadas em padrões conhecidos de Libras"""
        print(f"🎨 Criando imagens sintéticas para letra {letra}...")
        
        # Padrões conhecidos de gestos de Libras (simplificados)
        padroes_gestos = {
            'D': {'dedos': [1, 1, 0, 0, 0], 'posicao': 'centro'},
            'E': {'dedos': [0, 0, 0, 0, 0], 'posicao': 'centro'},
            'F': {'dedos': [1, 1, 0, 0, 1], 'posicao': 'centro'},
            'G': {'dedos': [1, 1, 0, 0, 0], 'posicao': 'lateral'},
            'H': {'dedos': [1, 1, 0, 0, 0], 'posicao': 'centro'},
            'I': {'dedos': [0, 0, 0, 0, 1], 'posicao': 'centro'},
            'J': {'dedos': [0, 0, 0, 0, 1], 'posicao': 'centro'},
            'K': {'dedos': [1, 1, 0, 0, 0], 'posicao': 'centro'},
            'M': {'dedos': [0, 0, 0, 0, 0], 'posicao': 'centro'},
            'N': {'dedos': [0, 0, 0, 0, 0], 'posicao': 'centro'},
            'O': {'dedos': [0, 0, 0, 0, 0], 'posicao': 'centro'},
            'P': {'dedos': [1, 1, 0, 0, 0], 'posicao': 'centro'},
            'Q': {'dedos': [1, 1, 0, 0, 0], 'posicao': 'lateral'},
            'R': {'dedos': [1, 1, 0, 0, 0], 'posicao': 'centro'},
            'S': {'dedos': [0, 0, 0, 0, 0], 'posicao': 'centro'},
            'T': {'dedos': [0, 0, 0, 0, 0], 'posicao': 'centro'},
            'U': {'dedos': [1, 1, 0, 0, 0], 'posicao': 'centro'},
            'V': {'dedos': [1, 1, 0, 0, 0], 'posicao': 'centro'},
            'W': {'dedos': [1, 1, 1, 0, 0], 'posicao': 'centro'},
            'X': {'dedos': [1, 0, 0, 0, 0], 'posicao': 'centro'},
            'Z': {'dedos': [1, 1, 0, 0, 0], 'posicao': 'centro'}
        }
        
        if letra not in padroes_gestos:
            return []
        
        imagens_criadas = []
        padrao = padroes_gestos[letra]
        
        # Criar múltiplas variações
        for i in range(20):  # 20 variações por letra
            # Criar imagem base
            img = np.zeros((480, 640, 3), dtype=np.uint8)
            img.fill(50)  # Fundo escuro
            
            # Simular posição da mão baseada no padrão
            centro_x, centro_y = 320, 240
            
            # Adicionar variações de posição
            variacao_x = np.random.randint(-50, 50)
            variacao_y = np.random.randint(-30, 30)
            centro_x += variacao_x
            centro_y += variacao_y
            
            # Desenhar "dedos" baseado no padrão
            cores_dedos = [(255, 255, 255), (200, 200, 200), (150, 150, 150)]
            
            for j, dedo_levantado in enumerate(padrao['dedos']):
                if dedo_levantado:
                    # Desenhar dedo levantado
                    cv2.circle(img, (centro_x + j*20 - 40, centro_y - 50), 8, cores_dedos[j%3], -1)
                else:
                    # Desenhar dedo abaixado
                    cv2.circle(img, (centro_x + j*20 - 40, centro_y + 20), 6, (100, 100, 100), -1)
            
            # Desenhar palma da mão
            cv2.circle(img, (centro_x, centro_y), 25, (180, 180, 180), -1)
            
            # Adicionar ruído para variação
            ruido = np.random.randint(-20, 20, img.shape, dtype=np.int16)
            img = np.clip(img.astype(np.int16) + ruido, 0, 255).astype(np.uint8)
            
            # Salvar imagem
            nome_arquivo = f"imagens_coletadas/{letra}_sintetica_{i:03d}.jpg"
            cv2.imwrite(nome_arquivo, img)
            imagens_criadas.append(nome_arquivo)
        
        return imagens_criadas
    
    def processar_imagem(self, caminho_imagem, letra):
        """Processa uma imagem e extrai landmarks"""
        try:
            # Carregar imagem
            img = cv2.imread(caminho_imagem)
            if img is None:
                return None
            
            # Converter para RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Processar com MediaPipe
            results = self.hands.process(img_rgb)
            
            if results.multi_hand_landmarks:
                # Pegar o primeiro landmark detectado
                hand_landmarks = results.multi_hand_landmarks[0]
                
                # Processar landmarks
                landmarks = self.processar_landmarks(hand_landmarks)
                
                if len(landmarks) == 63:  # 21 pontos × 3 coordenadas
                    return {
                        'label': letra,
                        'source': 'synthetic',
                        'image_path': caminho_imagem,
                        **{f'point_{i}': landmarks[i] for i in range(63)}
                    }
            
            return None
            
        except Exception as e:
            print(f"❌ Erro ao processar {caminho_imagem}: {e}")
            return None
    
    def processar_landmarks(self, hand_landmarks):
        """Processa landmarks da mão e normaliza"""
        p0 = hand_landmarks.landmark[0]  # Ponto de referência (pulso)
        points = []
        for landmark in hand_landmarks.landmark:
            points.extend([
                landmark.x - p0.x,
                landmark.y - p0.y,
                landmark.z - p0.z
            ])
        return points
    
    def coletar_dados_letra(self, letra):
        """Coleta dados para uma letra específica"""
        print(f"\n🎯 Coletando dados para letra: {letra}")
        
        # Criar imagens sintéticas
        imagens = self.criar_imagens_sinteticas(letra)
        
        dados_letra = []
        for img_path in imagens:
            dados = self.processar_imagem(img_path, letra)
            if dados:
                dados_letra.append(dados)
        
        print(f"✅ Coletados {len(dados_letra)} dados para {letra}")
        return dados_letra
    
    def coletar_todas_letras(self):
        """Coleta dados para todas as letras"""
        print("🚀 Iniciando coleta automática de dados...")
        
        todos_dados = []
        
        for letra in self.letras_para_coletar:
            dados_letra = self.coletar_dados_letra(letra)
            todos_dados.extend(dados_letra)
            
            # Pequena pausa entre letras
            time.sleep(0.5)
        
        self.dados_coletados = todos_dados
        print(f"\n✅ Coleta concluída! Total: {len(todos_dados)} dados")
        
        return todos_dados
    
    def salvar_dados(self, nome_arquivo='gestos_libras_expandido.csv'):
        """Salva dados coletados"""
        if not self.dados_coletados:
            print("❌ Nenhum dado para salvar")
            return
        
        try:
            # Carregar dados existentes
            dados_existentes = []
            if os.path.exists('gestos_libras.csv'):
                df_existente = pd.read_csv('gestos_libras.csv')
                dados_existentes = df_existente.to_dict('records')
            
            # Combinar dados
            todos_dados = dados_existentes + self.dados_coletados
            
            # Criar DataFrame
            df = pd.DataFrame(todos_dados)
            
            # Salvar
            df.to_csv(nome_arquivo, index=False)
            
            print(f"✅ Dados salvos em {nome_arquivo}")
            print(f"📊 Total de amostras: {len(todos_dados)}")
            print(f"📈 Novas amostras: {len(self.dados_coletados)}")
            
            # Mostrar distribuição
            print("\n📊 Distribuição por classe:")
            print(df['label'].value_counts().sort_index())
            
        except Exception as e:
            print(f"❌ Erro ao salvar dados: {e}")
    
    def treinar_modelo_rapido(self):
        """Treina modelo rapidamente com os novos dados"""
        try:
            from sklearn.model_selection import train_test_split
            from sklearn.ensemble import RandomForestClassifier
            import pickle
            
            print("\n🧠 Treinando modelo com dados expandidos...")
            
            # Carregar dados
            df = pd.read_csv('gestos_libras_expandido.csv')
            feature_columns = [col for col in df.columns if col not in ['label', 'source', 'image_path']]
            X = df[feature_columns].values
            y = df['label'].values
            
            print(f"📊 Dados: {len(df)} amostras, {len(feature_columns)} features")
            print(f"🏷️ Classes: {sorted(df['label'].unique())}")
            
            # Dividir dados
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Treinar modelo
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42
            )
            
            model.fit(X_train, y_train)
            
            # Avaliar
            train_acc = model.score(X_train, y_train)
            test_acc = model.score(X_test, y_test)
            
            print(f"📈 Acurácia treino: {train_acc:.2%}")
            print(f"📈 Acurácia teste: {test_acc:.2%}")
            
            # Salvar modelo
            os.makedirs('modelos', exist_ok=True)
            with open('modelos/modelo_libras_expandido.pkl', 'wb') as f:
                pickle.dump(model, f)
            
            # Salvar informações
            model_info = {
                'classes': model.classes_.tolist(),
                'n_features': len(feature_columns),
                'train_accuracy': train_acc,
                'test_accuracy': test_acc,
                'n_samples': len(df),
                'vocabulary_type': 'expanded_synthetic'
            }
            
            with open('modelos/modelo_info_expandido.pkl', 'wb') as f:
                pickle.dump(model_info, f)
            
            print("✅ Modelo expandido salvo!")
            return True
            
        except Exception as e:
            print(f"❌ Erro no treinamento: {e}")
            return False

def main():
    print("🚀 TraduLibras - Coletor Automático de Dados v1.0")
    print("📚 Coletando imagens sintéticas de gestos de Libras...")
    
    coletor = ColetorInternetLibras()
    
    # Coletar dados
    dados = coletor.coletar_todas_letras()
    
    if dados:
        # Salvar dados
        coletor.salvar_dados()
        
        # Treinar modelo
        coletor.treinar_modelo_rapido()
        
        print("\n🎉 Processo concluído!")
        print("📁 Arquivos criados:")
        print("   - gestos_libras_expandido.csv")
        print("   - modelos/modelo_libras_expandido.pkl")
        print("   - modelos/modelo_info_expandido.pkl")
        print("   - imagens_coletadas/ (imagens sintéticas)")
    else:
        print("❌ Nenhum dado foi coletado")

if __name__ == "__main__":
    main()
