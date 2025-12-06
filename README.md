# Sistema de Detecção de Doenças em Folhas

Sistema de detecção de doenças em folhas utilizando modelos de colorização pix2pix e métricas Grad-CAM.

## Instalação

1. Clone o repositório
2. Crie um ambiente virtual:

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/Mac
   ```

3. Instale as dependências:

   ```bash
   pip install -r requirements.txt
   ```

## Uso

### Interface Web (Streamlit)

```bash
streamlit run app.py
```

### Scripts CLI

**Detectar anomalias:**

```bash
python scripts/detect_anomalies.py --results_dir pytorch-CycleGAN-and-pix2pix/results/model_name/test_latest
```

**Comparar treinamentos:**

```bash
python scripts/compare_trainings.py
```

**Visualizar heatmaps:**

```bash
python scripts/visualize_heatmaps.py --results_dir pytorch-CycleGAN-and-pix2pix/results/model_name/test_latest
```

## Licenças

### Código deste projeto

Este projeto está licenciado sob a licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.

### Licenças de Terceiros

Este projeto utiliza o [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix), licenciado sob a licença BSD:

``` txt
Copyright (c) 2017, Jun-Yan Zhu and Taesung Park
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.
```

A pasta `pytorch-CycleGAN-and-pix2pix/` contém o código original com sua licença completa em `pytorch-CycleGAN-and-pix2pix/LICENSE`.

## Projeto

Trabalho Final - Introdução à Inteligência Artificial - UnB - 2025/2
