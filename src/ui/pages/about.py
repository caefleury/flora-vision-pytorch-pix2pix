import streamlit as st


def render():
    st.title("Sobre o FloraVision")
    
    st.markdown("""
    **Projeto FloraVision**: sistema de IA para detecção de anomalias em folhas por 
    reconstrução generativa e heatmaps, desenvolvido para a disciplina de 
    **Introdução à Inteligência Artificial da UnB**.
    """)
    
    st.divider()
    
    # Overview section
    st.header("🌿 Visão Geral")
    st.markdown("""
    O FloraVision utiliza uma abordagem inovadora para detectar doenças e anomalias 
    em folhas de plantas. Em vez de treinar um classificador tradicional que precisa 
    aprender padrões de cada doença específica, nosso sistema aprende como uma folha 
    **saudável** deve ser colorida — e usa essa expectativa para identificar anomalias.
    
    A ideia central é simples mas poderosa: se o modelo aprendeu perfeitamente como 
    colorir folhas saudáveis, ele terá dificuldade em reconstruir corretamente as 
    cores de uma folha doente. Essa **diferença de reconstrução** revela as anomalias.
    """)
    
    st.divider()
    
    # Pix2Pix section
    st.header("🧠 A Arquitetura Pix2Pix")
    
    st.subheader("O que é o Pix2Pix?")
    st.markdown("""
    O **Pix2Pix** é uma Rede Adversarial Generativa Condicional (cGAN) desenvolvida 
    por Isola et al. (2017). Ele é projetado para tarefas de **tradução de imagem 
    para imagem** — onde queremos converter uma imagem de um domínio para outro 
    preservando a estrutura espacial.
    
    Exemplos clássicos incluem:
    - Converter mapas em fotos de satélite
    - Colorir imagens em preto e branco
    - Converter esboços em fotos realistas
    - Transformar fotos diurnas em noturnas
    
    No nosso caso, usamos o Pix2Pix para **colorização**: convertemos a versão em 
    escala de cinza (canal L do espaço LAB) para a versão colorida (canais a e b).
    """)
    
    st.subheader("Componentes Principais")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎨 Gerador (U-Net)")
        st.markdown("""
        O gerador usa uma arquitetura **U-Net** com skip connections:
        
        - **Encoder**: Reduz progressivamente a resolução enquanto aumenta a 
          profundidade de features, capturando informações semânticas abstratas
        
        - **Decoder**: Reconstrói a imagem na resolução original, usando as 
          features aprendidas
        
        - **Skip Connections**: Conectam camadas correspondentes do encoder e 
          decoder, preservando detalhes espaciais finos que seriam perdidos 
          durante a compressão
        
        Entrada: Canal L (luminância/escala de cinza)
        Saída: Canais a e b (informação de cor)
        """)
    
    with col2:
        st.markdown("### 🔍 Discriminador (PatchGAN)")
        st.markdown("""
        O discriminador usa uma arquitetura **PatchGAN**:
        
        - Em vez de classificar a imagem inteira como real ou falsa, avalia 
          **patches (regiões)** de 70x70 pixels
        
        - Isso força o gerador a produzir texturas realistas em toda a imagem
        
        - Reduz o número de parâmetros comparado a um discriminador tradicional
        
        - Durante o treinamento, tenta distinguir entre pares (input, output) 
          reais e gerados
        
        Entrada: Concatenação do canal L + canais ab (real ou gerado)
        Saída: Mapa de probabilidades por patch
        """)
    
    st.subheader("Treinamento Adversarial")
    st.markdown("""
    O Pix2Pix é treinado com duas funções de perda combinadas:
    
    1. **Perda Adversarial (GAN Loss)**: O gerador tenta enganar o discriminador, 
       enquanto o discriminador tenta distinguir imagens reais das geradas. Esse 
       "jogo" leva a resultados mais realistas.
    
    2. **Perda L1**: Penaliza a diferença pixel a pixel entre a saída gerada e a 
       imagem real. Isso garante que o gerador produza cores próximas às originais, 
       não apenas "realistas".
    
    A combinação dessas perdas resulta em colorização que é tanto **visualmente 
    convincente** quanto **fiel à imagem original**.
    """)
    
    st.divider()
    
    # Our approach section
    st.header("🔬 Nossa Abordagem: Detecção por Reconstrução")
    
    st.subheader("Fase de Treinamento")
    st.markdown("""
    1. **Coleta de dados**: Reunimos imagens de folhas **apenas saudáveis**
    
    2. **Pré-processamento**: Cada imagem é convertida para o espaço de cor LAB:
       - **L** (Luminância): Escala de cinza, de 0 a 100
       - **a**: Verde (-) a Vermelho (+)
       - **b**: Azul (-) a Amarelo (+)
    
    3. **Treinamento do modelo**: O Pix2Pix aprende a mapear L → ab
       - Entrada: Canal L (estrutura da folha)
       - Saída: Canais ab (cores da folha)
       - Objetivo: Reconstruir perfeitamente as cores de folhas saudáveis
    
    4. **Resultado**: Um modelo especialista em colorir folhas saudáveis
    """)
    
    st.subheader("Fase de Detecção")
    st.markdown("""
    1. **Nova imagem**: Recebemos uma folha para análise (saudável ou doente)
    
    2. **Extração**: Separamos o canal L (escala de cinza) da imagem
    
    3. **Colorização**: O modelo gera os canais ab que ele "espera" ver
    
    4. **Comparação**: Calculamos a diferença entre:
       - **Cor real**: Os canais ab originais da imagem
       - **Cor esperada**: Os canais ab gerados pelo modelo
    
    5. **Métrica CIEDE2000**: Uma fórmula perceptual que quantifica a diferença 
       de cor de forma similar à visão humana
    
    6. **Heatmap de anomalias**: Visualização das regiões com maior diferença
    """)
    
    st.subheader("Por que funciona?")
    st.info("""
    **A intuição é elegante**: O modelo foi treinado exclusivamente com folhas 
    saudáveis, então ele "conhece" apenas padrões de coloração saudável.
    
    Quando recebe uma folha doente:
    - O modelo tenta colorir como se fosse saudável
    - As regiões doentes (manchas, necroses, descolorações) serão coloridas 
      "incorretamente"
    - A diferença entre a cor real (doente) e a cor esperada (saudável) 
      revela a anomalia
    
    Isso é chamado de **detecção por anomalia de reconstrução**.
    """)
    
    st.divider()
    
    # CIEDE2000 section
    st.header("📊 Métrica CIEDE2000")
    st.markdown("""
    O **CIEDE2000** (ΔE*₀₀) é o padrão internacional para medir diferença 
    perceptual de cores. Desenvolvido pela CIE (Commission Internationale de 
    l'Éclairage), ele modela como humanos percebem diferenças de cor.
    
    Características importantes:
    
    | Valor ΔE | Percepção |
    |----------|-----------|
    | < 1.0 | Imperceptível |
    | 1.0 - 2.0 | Perceptível apenas por olhos treinados |
    | 2.0 - 3.5 | Perceptível |
    | 3.5 - 5.0 | Diferença clara |
    | > 5.0 | Cores claramente diferentes |
    
    No contexto do FloraVision:
    - **Score baixo** (< 3.0): A colorização foi precisa → folha provavelmente saudável
    - **Score alto** (> 3.0): A colorização divergiu → possível anomalia
    """)
    
    st.divider()
    
    # Heatmaps section
    st.header("🗺️ Heatmaps e Visualização")
    st.markdown("""
    O sistema gera **mapas de calor (heatmaps)** que mostram pixel a pixel onde 
    as anomalias estão localizadas:
    
    - **Cores frias (azul/verde)**: Baixa diferença de cor → Região saudável
    - **Cores quentes (amarelo/vermelho)**: Alta diferença de cor → Possível anomalia
    
    Isso permite:
    - **Localização precisa** das regiões afetadas
    - **Quantificação** da severidade da anomalia
    - **Verificação visual** para validação humana
    """)
    
    st.subheader("Grad-CAM")
    st.markdown("""
    Além dos heatmaps de diferença de cor, oferecemos visualização **Grad-CAM** 
    (Gradient-weighted Class Activation Mapping):
    
    - Mostra quais regiões da imagem a rede neural considera mais importantes
    - Ajuda a entender o "raciocínio" do modelo
    - Útil para validar se o modelo está focando nas regiões corretas
    """)
    
    st.divider()
    
    # Technical details
    st.header("⚙️ Detalhes Técnicos")
    
    with st.expander("Arquitetura do Gerador (U-Net 256)"):
        st.code("""
Encoder (downsampling):
  Conv2d(1, 64) → LeakyReLU
  Conv2d(64, 128) → BatchNorm → LeakyReLU
  Conv2d(128, 256) → BatchNorm → LeakyReLU
  Conv2d(256, 512) → BatchNorm → LeakyReLU
  Conv2d(512, 512) → BatchNorm → LeakyReLU
  Conv2d(512, 512) → BatchNorm → LeakyReLU
  Conv2d(512, 512) → BatchNorm → LeakyReLU
  Conv2d(512, 512) → ReLU (bottleneck)

Decoder (upsampling com skip connections):
  ConvTranspose2d(512, 512) → BatchNorm → Dropout → ReLU + skip
  ConvTranspose2d(1024, 512) → BatchNorm → Dropout → ReLU + skip
  ConvTranspose2d(1024, 512) → BatchNorm → Dropout → ReLU + skip
  ConvTranspose2d(1024, 512) → BatchNorm → ReLU + skip
  ConvTranspose2d(1024, 256) → BatchNorm → ReLU + skip
  ConvTranspose2d(512, 128) → BatchNorm → ReLU + skip
  ConvTranspose2d(256, 64) → BatchNorm → ReLU + skip
  ConvTranspose2d(128, 2) → Tanh

Entrada: 1 x 256 x 256 (canal L normalizado)
Saída: 2 x 256 x 256 (canais ab normalizados)
        """, language="text")
    
    with st.expander("Espaço de Cor LAB"):
        st.markdown("""
        O espaço de cor **CIELAB** (LAB) separa luminosidade de informação de cor:
        
        - **L** (Luminosidade): 0 (preto) a 100 (branco)
        - **a**: -128 (verde) a +127 (vermelho)  
        - **b**: -128 (azul) a +127 (amarelo)
        
        Vantagens para colorização:
        1. **Separação de estrutura e cor**: O canal L contém toda a informação 
           de bordas, texturas e formas
        2. **Perceptualmente uniforme**: Distâncias no espaço LAB correspondem 
           melhor à percepção humana
        3. **Natural para a tarefa**: O modelo aprende apenas a adicionar cor, 
           não a modificar a estrutura
        """)
    
    with st.expander("Parâmetros de Treinamento"):
        st.markdown("""
        Configurações típicas para o treinamento:
        
        | Parâmetro | Valor | Descrição |
        |-----------|-------|-----------|
        | `netG` | unet_256 | Arquitetura do gerador |
        | `ngf` | 64 | Filtros base do gerador |
        | `n_epochs` | 100 | Épocas com LR constante |
        | `n_epochs_decay` | 100 | Épocas com LR decrescente |
        | `lr` | 0.0002 | Learning rate inicial |
        | `beta1` | 0.5 | Parâmetro Adam |
        | `lambda_L1` | 100 | Peso da perda L1 |
        | `batch_size` | 4-16 | Depende da GPU disponível |
        """)
    
    st.divider()
    
    # References
    st.header("📚 Referências")
    st.markdown("""
    1. **Isola, P., Zhu, J. Y., Zhou, T., & Efros, A. A.** (2017). 
       *Image-to-Image Translation with Conditional Adversarial Networks*. 
       CVPR 2017. [arXiv:1611.07004](https://arxiv.org/abs/1611.07004)
    
    2. **Zhu, J. Y., Park, T., Isola, P., & Efros, A. A.** (2017). 
       *Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks*. 
       ICCV 2017. [arXiv:1703.10593](https://arxiv.org/abs/1703.10593)

    3. **Ryoya Katafuchi, Terumasa Tokunaga (2020)**
        *Image-based Plant Disease Diagnosis with Unsupervised Anomaly Detection Based on Reconstructability of Colors*
        [arXiv:2011.14306](https://arxiv.org/abs/2011.14306)
    """)
    
