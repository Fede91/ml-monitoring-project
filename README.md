---
title: Corporate Sentiment Monitoring
colorFrom: gray
colorTo: red
sdk: gradio
pinned: false
sdk_version: 4.44.1
app_file: app/app.py
---

# Step 1:

- Download dataset Twitter Sentiment Analysis https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis

- Faccio il primo giro di inferenza per raccogliere il primo dato di accuratezza sul dataset iniziale e lo salvo come "base_matrics"
  e genero il modello che poi sarà pubblicato su hugging face

- Le volte successive, eseguiremo lo script di addestramento per aggiornare il modello e pubblicarlo

In gradio visualizzare la distribuzione delle etichette ed individuare quella più probabile.
Questo rientra nel monitoraggio.
Inoltre dare la possibilità di inserire un testo, una etichetta e fare inferenza
-> registrare i log che verranno usati per il riaddestramento
