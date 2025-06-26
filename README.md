---
title: Corporate Sentiment Monitoring
colorFrom: gray
colorTo: red
sdk: gradio
pinned: false
sdk_version: 4.44.1
app_file: app/app.py
---

# Corporate Sentiment Monitoring

Un sistema completo di MLOps per l'analisi del sentiment di testi social media con pipeline automatizzate di training, deploy e monitoraggio continuo.

## Panoramica del Progetto

Questo progetto implementa un sistema end-to-end di Machine Learning Operations (MLOps) per l'analisi del sentiment, progettato per dimostrare le best practices nell'automazione del ciclo di vita dei modelli ML.

### Obiettivi Raggiunti

- **Modello di classificazione del sentiment** capace di distinguere tra sentiment positivo, neutro e negativo;
- **Pipeline CI/CD completamente automatizzate** per training, testing e deploy del modello e dell'applicazione web;
- **Sistema di monitoraggio continuo** che rileva il degrado delle performance;
- **Interfaccia web interattiva** per inferenza e raccolta di feedback dal mondo reale;
- **Retraining automatico** basato su soglie di performance predefinite.

## Architettura e Decisioni Progettuali

### Scelta del Modello Base

Il progetto utilizza il modello `cardiffnlp/twitter-roberta-base-sentiment-latest` come punto di partenza già integrato nell'ecosistema HuggingFace.

### Dataset e Strategia di Suddivisione

Il dataset [Twitter Sentiment Analysis](https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis) è stato scelto per la sua coerenza con il dominio del modello base.
Una decisione importante è stata quella di **suddividere il training set in chunks multipli** durante la fase di inizializzazione. Questa scelta permette di:

- **Simulare scenari reali** dove nuovi dati arrivano incrementalmente;
- **Testare il retraining automatico** senza dover aspettare dati reali di produzione;
- **Validare la pipeline** di monitoraggio e riaddestramento in ambiente controllato;

## Fase 1: Inizializzazione del Progetto

La fase di inizializzazione è stata strutturata in più fasi che trasformano un dataset grezzo in un modello completamente deployato e monitorato.

### Fase 1.1: Preparazione e Formattazione dei Dati

```bash
python ./init/0_generate_sets.py --dataset data/data.csv --output data
```

Lo script `0_generate_sets.py` rappresenta il primo step del progetto che prevede la conversione del dataset originale in un formato standardizzato compatibile con la pipeline,
estrando solamente le colonne `text`, `sentiment` dal CSV e la suddivisione del training set in porzioni più piccole per simulare l'arrivo incrementale di nuovi dati nel tempo.

### Fase 1.2: Creazione del Modello Baseline

```bash
python ./init/1_generate_model.py
```

Il secondo script esegue l'inferenza sul dataset di validazione partendo dal modello pre-addestrato e genera una copia del modello che rappresenterà il modello baseline.
Il processo di inferenza permette di calcolare l'accuratezza iniziale del modello, che servirà come riferimento per tutti i retraining futuri.

Le metriche calcolate in questa fase vengono salvate in `metrics/base_metrics.json`.

### Fase 1.3: Deploy Iniziale

```bash
python ./init/2_deploy_model.py --path ./model
```

L'ultimo script della fase di inizializzazione gestisce il deploy del modello appena creato, eseguendo l'upload su HuggingFace Hub.

## Fase 2: Pipeline Automatizzate

Il sistema implementa tre pipeline automatizzate che lavorano insieme per mantenere il modello sempre aggiornato e performante.

### Pipeline Modello: Training e Deploy Automatico (`model.yml`)

Questa pipeline è responsabile del retraining automatico e viene triggerata nelle seguenti condizioni:

- **Push su dati**: Ogni volta che vengono aggiunti nuovi dati di training (simulando l'arrivo di nuovi campioni etichettati);
- **Modifiche al codice**: Quando il codice di training viene aggiornato con miglioramenti o fix;
- **Scheduling temporale**: Esecuzione automatica per verificare se ci sono nuovi dati di log da processare.

È stata impostata la soglia del **2% di miglioramento** per evitare deploy di versioni marginalmente migliori.

**Step:**

1. **Training incrementale**: Utilizza il modello esistente come punto di partenza ed esecuzione del training con i nuovi dati;
2. **Valutazione su validation set**: Calcola le nuove metriche di performance (accuratezza);
3. **Confronto con baseline**: Confronta dell'accuratezza del modello esistente con il modello addestrato;
4. **Deploy condizionale**: Solo se il miglioramento è significativo (+2%)
5. **Aggiornamento baseline**: Le nuove metriche diventano il nuovo riferimento

### Pipeline App: Deploy dell'Applicazione (`app.yml`)

L'applicazione è stata implementata tramite la libreria Gradio che permette la realizzazione di interfacce web in modo agevole.
L'utente può eseguire l'inferenza sul modello addestrato fornendo il testo e la predizione attesa. Il modello restituirà, quindi,
il valore predetteo e la distribuzione di ogni classe.

Per la gesione del deploy automatico, è stata implementata questa pipeline gestisce il **ciclo di vita dell'interfaccia utente**.

Solamente se l'aggiornamento dell'applicazione supera tutti i test d'integrazione tra interfaccia e modello, questa verrà pubblicata
su HuggingFace Space.

**URL**: [Corporate Sentiment Monitoring](https://huggingface.co/spaces/0xfedev/corporate-sentiment-monitoring)

### Fase 3: Monitoraggio e Retraining (`monitor.yml`)

Questa pipeline permette di monitorare le performance del modello, facendo scattare l'addestramento incrementale non appena viene rilevato
un peggioramento delle performance durante l'utilizzo dell'applicazione web.

**Strategia di Monitoraggio:**

1. **Raccolta dati di produzione**: Scarica le inferenze loggiate dall'app Gradio, tramite il Dataset pubblico e aggiorna il train set;
2. **Valutazione performance**: Calcola l'accuracy sui dati reali con etichette fornite dagli utenti;
3. **Rilevamento degrado**: Confronta con le metriche baseline;
4. **Retrain automatico**: Se rileva degrado, aggiorna il training set e lancia il retraining.

Nell'app Gradio è stato implementato un sistema di logging che traccia:

- **Input utente**: Testo da analizzare;
- **Predizione modello**: Sentiment predetto con scores dell'accuracy;
- **Ground truth**: Etichetta corretta fornita dall'utente.

Tutti i dati raccolti vengono pubblicati nel Dataset [0xfedev/corporate-sentiment-logs](https://huggingface.co/datasets/0xfedev/corporate-sentiment-logs) che potranno,
così, essere usati dalla pipeline di training per il retrain del modello.

Nell'ottica di scale-up del progetto, quindi, sia avrà:

- Più utenti utilizzano l'app → Più dati vengono raccolti
- Più dati vengono raccolti → Migliore monitoraggio delle performance
- Migliore monitoraggio → Retraining più preciso e tempestivo
- Retraining migliore → App più accurata
