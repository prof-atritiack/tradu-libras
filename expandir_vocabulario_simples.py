#!/usr/bin/env python3
"""
Script simples para expandir vocabulário do TraduLibras
Foco em processar imagens de gestos
"""

import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import os
import glob
from pathlib import Path
import json
from datetime import datetime

class ExpansorSimples:
    def __init__(self):
        # Inicializar MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        # Letras faltantes
        self.letras_faltantes = ['D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Z']
        
        # Criar pastas necessárias
        self.criar_pastas()
    
    def criar_pastas(self):
        """Cria pastas necessárias"""
        pastas = ['imagens/letras', 'imagens/numeros', 'backup']
        for pasta in pastas:
            Path(pasta).mkdir(parents=True, exist_ok=True)
            print(f"✅ Pasta criada: {pasta}")
    
    def processar_landmarks(self, hand_landmarks):
        """Processa landmarks da mão e normaliza"""
        p0 = hand_landmarks.landmark[0]
        points = []
        for landmark in hand_landmarks.landmark:
            points.extend([
                landmark.x - p0.x,
                landmark.y - p0.y,
                landmark.z - p0.z
            ])
        return points
    
    def processar_imagem(self, caminho_imagem):
        """Processa uma única imagem e extrai landmarks"""
        try:
            imagem = cv2.imread(caminho_imagem)
            if imagem is None:
                return None
            
            rgb_imagem = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_imagem)
            
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                landmarks = self.processar_landmarks(hand_landmarks)
                
                if len(landmarks) == 63:
                    return landmarks
                else:
                    print(f"⚠️ Landmarks inválidos em: {caminho_imagem}")
                    return None
            else:
                print(f"⚠️ Mão não detectada em: {caminho_imagem}")
                return None
                
        except Exception as e:
            print(f"❌ Erro ao processar {caminho_imagem}: {e}")
            return None
    
    def extrair_letra_nome_arquivo(self, nome_arquivo):
        """Extrai a letra do nome do arquivo"""
        import re
        nome_sem_ext = os.path.splitext(nome_arquivo)[0]
        
        # Padrões comuns
        padroes = [
            r'([A-Z])',  # Letra maiúscula
            r'letra_([A-Z])',  # letra_A.jpg
            r'gesto_([A-Z])',  # gesto_A.jpg
            r'([A-Z])_',  # A_001.jpg
            r'_([A-Z])_',  # img_A_001.jpg
        ]
        
        for padrao in padroes:
            match = re.search(padrao, nome_sem_ext)
            if match:
                return match.group(1)
        
        return None
    
    def processar_imagens_pasta(self, pasta_imagens):
        """Processa todas as imagens de uma pasta"""
        print(f"🖼️ Processando imagens da pasta: {pasta_imagens}")
        
        if not os.path.exists(pasta_imagens):
            print(f"❌ Pasta não encontrada: {pasta_imagens}")
            return []
        
        # Buscar imagens
        formatos = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
        imagens = []
        for formato in formatos:
            imagens.extend(glob.glob(os.path.join(pasta_imagens, f"*{formato}")))
            imagens.extend(glob.glob(os.path.join(pasta_imagens, f"*{formato.upper()}")))
        
        if not imagens:
            print(f"⚠️ Nenhuma imagem encontrada em: {pasta_imagens}")
            return []
        
        print(f"📊 Encontradas {len(imagens)} imagens")
        
        todos_dados = []
        sucessos = 0
        
        for imagem_path in imagens:
            nome_arquivo = os.path.basename(imagem_path)
            print(f"Processando: {nome_arquivo}")
            
            # Extrair letra do nome
            letra = self.extrair_letra_nome_arquivo(nome_arquivo)
            if not letra:
                print(f"⚠️ Não foi possível identificar a letra em: {nome_arquivo}")
                continue
            
            landmarks = self.processar_imagem(imagem_path)
            if landmarks is not None:
                todos_dados.append({
                    'label': letra,
                    **{f'point_{j}': landmarks[j] for j in range(63)},
                    'imagem_origem': imagem_path,
                    'timestamp': datetime.now().isoformat()
                })
                sucessos += 1
        
        print(f"✅ Processadas {sucessos}/{len(imagens)} imagens")
        return todos_dados
    
    def salvar_dados(self, novos_dados, arquivo_saida='gestos_libras.csv'):
        """Salva dados processados no arquivo CSV"""
        try:
            # Carregar dados existentes
            if os.path.exists(arquivo_saida):
                df_existente = pd.read_csv(arquivo_saida)
                dados_existentes = df_existente.to_dict('records')
                print(f"📁 Carregados {len(dados_existentes)} dados existentes")
            else:
                dados_existentes = []
                print("📁 Criando novo arquivo de dados")
            
            # Combinar dados
            todos_dados = dados_existentes + novos_dados
            df = pd.DataFrame(todos_dados)
            
            # Fazer backup
            if os.path.exists(arquivo_saida):
                backup_name = f"backup/backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{arquivo_saida}"
                os.rename(arquivo_saida, backup_name)
                print(f"📁 Backup criado: {backup_name}")
            
            # Salvar
            df.to_csv(arquivo_saida, index=False)
            
            print(f"✅ Dados salvos em {arquivo_saida}")
            print(f"📊 Total de amostras: {len(todos_dados)}")
            print(f"📈 Novas amostras: {len(novos_dados)}")
            
            # Mostrar distribuição
            print("\n📊 Distribuição por classe:")
            distribuicao = df['label'].value_counts().sort_index()
            for classe, count in distribuicao.items():
                print(f"  {classe}: {count} amostras")
            
            return True
            
        except Exception as e:
            print(f"❌ Erro ao salvar dados: {e}")
            return False
    
    def mostrar_estatisticas(self):
        """Mostra estatísticas dos dados"""
        try:
            if os.path.exists('gestos_libras.csv'):
                df = pd.read_csv('gestos_libras.csv')
                print(f"\n📊 ESTATÍSTICAS ATUAIS:")
                print(f"📈 Total de amostras: {len(df)}")
                print(f"🏷️ Classes únicas: {len(df['label'].unique())}")
                print(f"📋 Classes: {sorted(df['label'].unique())}")
                
                print(f"\n📊 DISTRIBUIÇÃO:")
                distribuicao = df['label'].value_counts().sort_index()
                for classe, count in distribuicao.items():
                    print(f"  {classe}: {count} amostras")
                
                # Calcular progresso
                letras_implementadas = [c for c in df['label'].unique() if c.isalpha()]
                letras_faltantes = [l for l in self.letras_faltantes if l not in letras_implementadas]
                
                print(f"\n📈 PROGRESSO:")
                print(f"✅ Letras implementadas: {len(letras_implementadas)}/26")
                print(f"⏳ Letras faltantes: {len(letras_faltantes)}")
                if letras_faltantes:
                    print(f"📝 Faltam: {', '.join(letras_faltantes)}")
                
            else:
                print("❌ Arquivo gestos_libras.csv não encontrado")
                
        except Exception as e:
            print(f"❌ Erro ao mostrar estatísticas: {e}")
    
    def treinar_modelo(self):
        """Treina modelo com dados coletados"""
        try:
            print("🧠 Treinando modelo...")
            
            # Carregar dados
            df = pd.read_csv('gestos_libras.csv')
            feature_columns = [col for col in df.columns if col not in ['label', 'imagem_origem', 'timestamp']]
            X = df[feature_columns].values
            y = df['label'].values
            
            print(f"📊 Dados: {len(df)} amostras, {len(feature_columns)} features")
            print(f"🏷️ Classes: {sorted(df['label'].unique())}")
            
            # Treinar modelo
            from sklearn.model_selection import train_test_split
            from sklearn.ensemble import RandomForestClassifier
            import pickle
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            model = RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=3,
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
            with open('modelos/modelo_libras.pkl', 'wb') as f:
                pickle.dump(model, f)
            
            # Salvar informações
            model_info = {
                'classes': model.classes_.tolist(),
                'n_features': len(feature_columns),
                'train_accuracy': train_acc,
                'test_accuracy': test_acc,
                'n_samples': len(df),
                'timestamp': datetime.now().isoformat()
            }
            
            with open('modelos/modelo_info.pkl', 'wb') as f:
                pickle.dump(model_info, f)
            
            print("✅ Modelo treinado e salvo!")
            return True
            
        except Exception as e:
            print(f"❌ Erro no treinamento: {e}")
            return False

def main():
    print("🚀 TraduLibras - Expansor de Vocabulário Simples")
    print("=" * 60)
    
    expansor = ExpansorSimples()
    
    while True:
        print("\n📋 OPÇÕES DISPONÍVEIS:")
        print("1. 🖼️ Processar imagens de uma pasta")
        print("2. 📊 Ver estatísticas atuais")
        print("3. 🧠 Treinar modelo")
        print("4. 🚪 Sair")
        print("=" * 60)
        
        opcao = input("Digite o número da opção (1-4): ").strip()
        
        if opcao == '1':
            pasta = input("Digite o caminho da pasta com imagens: ").strip()
            if not pasta:
                pasta = 'imagens/letras/'
            
            dados = expansor.processar_imagens_pasta(pasta)
            if dados:
                expansor.salvar_dados(dados)
        
        elif opcao == '2':
            expansor.mostrar_estatisticas()
        
        elif opcao == '3':
            expansor.treinar_modelo()
        
        elif opcao == '4':
            print("👋 Até logo!")
            break
        
        else:
            print("❌ Opção inválida! Digite 1, 2, 3 ou 4")

if __name__ == "__main__":
    main()

