# Setup e Configurações da EfficientNetB0 para Classificação de Plantas Daninhas

## 1. Introdução

A EfficientNetB0 foi selecionada como um dos modelos base para a tarefa de classificação binária de plantas daninhas em culturas de cana-de-açúcar. Este documento apresenta uma descrição formal da arquitetura, configurações de treinamento e fundamentação técnica do setup implementado.

---

## 2. Arquitetura da EfficientNetB0

### 2.1 Características Gerais

A EfficientNetB0 é uma rede neural convolucional pré-treinada no conjunto de dados ImageNet, desenvolvida por Tan e Le (2019) com foco em eficiência computacional. A arquitetura utiliza **composição balanceada** entre profundidade, largura e resolução da entrada, resultando em um modelo mais compacto e eficiente em comparação com arquiteturas convencionais.

### 2.2 Estrutura da Implementação

A implementação da EfficientNetB0 para esta pesquisa segue a seguinte estrutura:

```
Entrada (224 × 224 × 3)
    ↓
EfficientNetB0 Base (sem cabeçalho de classificação)
    ↓
Global Average Pooling 2D
    ↓
Dropout (taxa = 0,2)
    ↓
Dense (1 neurônio, ativação sigmoid)
    ↓
Saída Binária (0 ou 1)
```

### 2.3 Componentes Arquiteturais

#### 2.3.1 Rede Base (EfficientNetB0)

A rede base é carregada com os seguintes parâmetros:

| Parâmetro | Valor | Descrição |
|-----------|-------|-----------|
| `include_top` | `False` | Remove o cabeçalho de classificação padrão do ImageNet |
| `weights` | `'imagenet'` | Utiliza pesos pré-treinados no ImageNet para inicialização |
| `input_shape` | (224, 224, 3) | Dimensões de entrada: altura × largura × canais RGB |
| `trainable` | `True` | Todos os pesos podem ser ajustados durante o treinamento (fine-tuning completo) |

**Justificativa do Fine-tuning:** O ajuste completo dos pesos permite que o modelo se adapte especificamente ao domínio de classificação de plantas daninhas, uma vez que as características visuais desta tarefa diferem significativamente do conjunto de dados ImageNet.

#### 2.3.2 Global Average Pooling 2D

A camada de pooling global médio reduz o tensor de características espaciais a um vetor 1D, realizando a média de todos os valores em cada mapa de características. Esta abordagem:

- Reduz drasticamente o número de parâmetros
- Mantém informação global sobre as características detectadas
- Melhora a generalização do modelo

#### 2.3.3 Dropout

Uma camada de Dropout com taxa de 0,2 (20%) é aplicada após o pooling global com o objetivo de:

- Reduzir o overfitting durante o treinamento
- Promover aprendizagem de características mais robustas
- Melhorar a capacidade de generalização em dados não vistos

#### 2.3.4 Camada de Classificação Final

Uma camada densa totalmente conectada com um único neurônio e função de ativação sigmoid realiza a classificação binária:

$$\text{saída} = \sigma(z) = \frac{1}{1 + e^{-z}}$$

Esta configuração produz probabilidades no intervalo [0, 1], onde valores próximos a 0 indicam "não-daninha" e próximos a 1 indicam "daninha".

---

## 3. Configurações de Treinamento

### 3.1 Parâmetros de Entrada de Dados

#### 3.1.1 Dimensões e Formato

| Parâmetro | Valor | Descrição |
|-----------|-------|-----------|
| Tamanho da imagem | 224 × 224 pixels | Dimensão padrão de entrada para EfficientNetB0 |
| Modo de cor | RGB | Três canais de cores (vermelho, verde, azul) |
| Normalização | Integrada ao EfficientNetB0 | Pré-processamento automático pela arquitetura |
| Batch size | 4 | Número de amostras processadas por iteração |

#### 3.1.2 Divisão do Conjunto de Dados

O dataset foi dividido em três conjuntos com a seguinte proporção:

| Conjunto | Proporção | Utilização |
|----------|-----------|-----------|
| Treinamento | 70% | Otimização dos pesos do modelo |
| Validação | 15% | Ajuste de hiperparâmetros e early stopping |
| Teste | 15% | Avaliação final e não enviesada |

A divisão foi realizada de forma aleatória com seed=42 para garantir reprodutibilidade dos resultados.

### 3.2 Otimizador e Função de Perda

#### 3.2.1 Otimizador Adam

A otimização utiliza o algoritmo **Adam** (Adaptive Moment Estimation) com os seguintes parâmetros:

| Parâmetro | Valor | Descrição |
|-----------|-------|-----------|
| Learning rate (taxa de aprendizado) | 1 × 10⁻⁴ | Taxa reduzida para fine-tuning de modelos pré-treinados |
| β₁ (momento exponencial) | 0,9 | Padrão do Adam |
| β₂ (segundo momento) | 0,999 | Padrão do Adam |

**Justificativa da Taxa de Aprendizado Reduzida:** Como os pesos já foram inicializados com conhecimento do ImageNet, uma taxa de aprendizado menor (1e-4) evita que mudanças abruptas destruam as características pré-aprendidas, permitindo uma adaptação gradual ao novo domínio.

#### 3.2.2 Função de Perda

$$L(\hat{y}, y) = -\frac{1}{n}\sum_{i=1}^{n} \left[y_i \log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)\right]$$

Onde:
- $\hat{y}_i$ = probabilidade predita (saída do modelo)
- $y_i$ = rótulo verdadeiro (0 ou 1)
- $n$ = número de amostras no batch

A **entropia cruzada binária** (binary crossentropy) é apropriada para classificação binária e fornece gradientes adequados para o treinamento.

### 3.3 Métricas de Avaliação

Durante o treinamento, as seguintes métricas são monitoradas:

#### 3.3.1 Acurácia

$$\text{Acc} = \frac{\text{Verdadeiros Positivos} + \text{Verdadeiros Negativos}}{\text{Total de Amostras}}$$

Mede a proporção geral de predições corretas.

#### 3.3.2 Recall (Sensibilidade)

$$\text{Recall} = \frac{\text{Verdadeiros Positivos}}{\text{Verdadeiros Positivos} + \text{Falsos Negativos}}$$

Mede a proporção de plantas daninhas corretamente identificadas. Esta métrica é crítica pois falsos negativos (classificar uma daninha como não-daninha) podem prejudicar a aplicação agrícola.

#### 3.3.3 F1-Score

$$F_1 = 2 \times \frac{\text{Precisão} \times \text{Recall}}{\text{Precisão} + \text{Recall}}$$

Fornece uma média harmônica entre precisão e recall, sendo útil para datasets desbalanceados.

### 3.4 Parâmetros de Treinamento

| Parâmetro | Valor | Descrição |
|-----------|-------|-----------|
| Número de épocas | 10 | Número de passagens completas pelo conjunto de treinamento |
| Número de rodadas | 3 | Repetições do experimento para construção de boxplots comparativos |
| Validação | A cada época | Avaliação no conjunto de validação |
| Verbose | 1 | Exibição de progresso durante o treinamento |

### 3.5 Técnicas de Otimização Adicional

#### 3.5.1 Mixed Precision Training

A política de precisão mista **mixed_float16** foi aplicada para:

- Reduzir o uso de memória GPU
- Acelerar o treinamento mantendo acurácia
- Utilizar armazenamento float16 para ativações enquanto mantém float32 para gradientes críticos

#### 3.5.2 Data Pipeline Otimizado

O pipeline de dados foi otimizado com:

- **Prefetching**: O atributo `tf.data.AUTOTUNE` permite que TensorFlow otimize automaticamente o número de amostras carregadas antecipadamente
- **Shuffling**: Embaralhamento das amostras no início para garantir representatividade em cada batch
- **Batch Processing**: Processamento de múltiplas amostras simultaneamente para melhor utilização de hardware

---

## 4. Fluxo de Execução

### 4.1 Pipeline de Treinamento

```
1. Carregamento de dados
   ├─ Leitura do diretório de imagens
   ├─ Mapeamento automático de labels (nao_daninha → 0, daninha → 1)
   └─ Divisão em conjuntos de treino/validação/teste

2. Para cada rodada:
   ├─ Instanciação da EfficientNetB0
   ├─ Compilação do modelo
   ├─ Treinamento iterativo
   │  └─ Para cada época:
   │     ├─ Forward pass no conjunto de treinamento
   │     ├─ Cálculo da perda e retropropagação
   │     ├─ Atualização de pesos via Adam
   │     └─ Validação no conjunto de validação
   ├─ Avaliação no conjunto de teste
   └─ Armazenamento de métricas

3. Pós-processamento
   ├─ Geração de gráficos de histórico de treinamento
   ├─ Criação de boxplots comparativos
   └─ Salvatagem de resultados em CSV
```

### 4.2 Estrutura de Diretórios

```
federated_learning/
├── src/
│   ├── models/
│   │   └── model.py (contém get_efficientnet_model)
│   ├── utils/
│   │   └── training_utils.py (funções de visualização)
│   ├── train.py (script principal)
│   └── data/
│       └── datasets/
│           └── daninhas__25-12-03/
│               ├── nao_daninha/
│               └── daninha/
└── results/
    ├── final_metrics.csv
    ├── EfficientNetB0_history_plot.png
    └── boxplot_*.png
```

---

## 5. Considerações Técnicas e Justificativas

### 5.1 Escolha da EfficientNetB0

A EfficientNetB0 foi selecionada pelos seguintes motivos:

1. **Eficiência Computacional**: Reduz parâmetros e FLOPs em comparação com ResNet e VGG, permitindo treinamento mais rápido
2. **Performance**: Mantém acurácia competitiva mesmo com arquitetura mais compacta
3. **Adaptabilidade**: Fine-tuning em domínios específicos como agricultura é bem-sucedido
4. **Relevância para Federated Learning**: O tamanho reduzido é favorável para distribuição em dispositivos edge

### 5.2 Fine-tuning vs. Transfer Learning

Escolheu-se o fine-tuning completo (todos os pesos treináveis) em vez de congelamento da base porque:

- O domínio agrícola possui características visuais suficientemente diferentes do ImageNet
- Dados de plantas daninhas possuem padrões específicos (textura, forma de folhas) não bem representados em ImageNet
- Recursos computacionais suficientes estavam disponíveis para treinamento completo

### 5.3 Batch Size Reduzido

O batch size de 4 foi escolhido para:

- Reduzir consumo de memória GPU
- Melhorar capacidade de generalização (gradientes mais ruidosos podem atuar como regularização)
- Compatibilidade com environments de federated learning de menor capacidade

---

## 6. Resultados Esperados

Com este setup, espera-se que a EfficientNetB0 alcance:

- **Acurácia**: > 85% no conjunto de teste
- **Recall**: > 90% (priorização de detecção de daninhas)
- **F1-Score**: > 0,87 (balanço entre precisão e recall)

O treinamento de 10 épocas com batch size 4 geralmente requer cerca de 5-15 minutos por rodada em GPU moderna (NVIDIA A100/RTX30 series).

---

## 7. Conclusão

O setup da EfficientNetB0 apresentado neste documento combina arquitetura moderna e eficiente com configurações de treinamento apropriadas para a tarefa de classificação de plantas daninhas em contexto agrícola. A abordagem de fine-tuning, otimizador bem calibrado e métricas adequadas fornecem uma base sólida para avaliação de performance e comparação com outros modelos (ResNet50 e Vision Transformer).

---

## Referências

- Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. In *International Conference on Machine Learning* (pp. 6105-6114). PMLR.
- Kingma, D. P., & Ba, J. (2014). Adam: A method for stochastic optimization. *arXiv preprint arXiv:1412.6980*.
- TensorFlow Documentation: https://www.tensorflow.org/api_docs/python/tf/keras/applications/EfficientNetB0
