# Guide d'installation et de démarrage de Text Generation Inference (TGI) sur Linux

## Prérequis

Avant de commencer, assurez-vous que votre serveur dispose des éléments suivants :

1. **Docker** : Veillez à ce que Docker soit installé et en cours d'exécution.
2. **GPU NVIDIA (optionnel)** : Si vous utilisez un GPU, assurez-vous que les *drivers* NVIDIA sont installés et fonctionnels.
    - Consultez les instructions officielles de Docker pour l'utilisation des GPU : [Docker GPU Support Guide](https://docs.docker.com/desktop/features/gpu/).
3. **Modèle d'inférence souhaité** : Identifiez le modèle que vous souhaitez utiliser depuis Hugging Face. Exemple : `HuggingFaceTB/SmolLM2-135M-Instruct`.

---

## Étapes de mise en œuvre

### 1. Commande Docker pour démarrer TGI

Utilisez la commande suivante adaptée à vos besoins :

```bash
docker run -p 8080:80 \
  --shm-size 4g \
  --gpus all \
  -v ${PWD}/tgi-data:/data \
  ghcr.io/huggingface/text-generation-inference:3.1.0 \
  --model-id HuggingFaceTB/SmolLM2-135M-Instruct \
  --validation-workers 2
```

**Détails des paramètres :**

- `-p 8080:80` : Mappe le port 80 du conteneur au port 8080 de votre machine hôte.
- `--shm-size 4g` : Alloue 4G de mémoire partagée pour le conteneur.
- `--gpus all` : Permet au conteneur d'accéder à toutes les GPU disponibles (si applicable).
- `-v ${PWD}/tgi-data:/data` : Monte le répertoire local `tgi-data` à l'intérieur du conteneur sous `/data`.
- `--model-id` : ID du modèle requis disponible sur Hugging Face.
- `--validation-workers` : Définit le nombre d'instances de validation pour accélérer le déploiement.

Plus options disponible ici: https://huggingface.co/docs/text-generation-inference/en/reference/launcher

---

### 2. Exemple avec `curl`

Une fois que le conteneur TGI est en cours d'exécution, vous pouvez interagir avec le modèle à l'aide de l'API exposée.

Utilisez la commande suivante pour envoyer une instruction au modèle via une requête HTTP `POST` :

```bash
curl localhost:8080/v1/chat/completions \
    -X POST \
    -d '{
  "model": "tgi",
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant."
    },
    {
      "role": "user",
      "content": "What is deep learning?"
    }
  ],
  "stream": true,
  "max_tokens": 20
}' \
    -H 'Content-Type: application/json'
```

## Documentation supplémentaire

- Documentation complète de TGI : [Text Generation Inference - Documentation](https://huggingface.co/docs/text-generation-inference/en/index)
- Guide d'utilisation des GPU avec Docker : [Docker GPU Documentation](https://docs.docker.com/desktop/features/gpu/)

---