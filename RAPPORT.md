# Rapport d'Implémentation du RAG Chatbot

## Difficultés Rencontrées

### 1. Configuration d'Ollama
L'intégration d'Ollama a présenté quelques défis, notamment pour assurer la communication entre le backend Python et le service Ollama. La configuration initiale nécessitait de vérifier que le service Ollama était correctement lancé et accessible depuis notre application.

### 2. Gestion de la Mémoire
Le traitement de grands volumes de documentation a parfois entraîné des problèmes de mémoire, en particulier lors de l'indexation des documents dans ChromaDB. J'ai dû optimiser la taille des chunks et gérer efficacement les ressources.

### 3. Interface Web vs. CLI
Bien que l'examen demandait initialement une interface en ligne de commande, j'ai choisi d'implémenter une interface web plus conviviale. Cela a ajouté une complexité supplémentaire avec la mise en place de Flask, mais a grandement amélioré l'expérience utilisateur.

### 4. Qualité des Réponses
L'optimisation du prompt et des paramètres de recherche a été cruciale pour obtenir des réponses pertinentes. Trouver le bon équilibre entre nombre de documents récupérés, longueur du contexte et formulation du prompt a nécessité plusieurs itérations.

## Améliorations Possibles

### 1. Système de Feedback
Ajouter un mécanisme permettant aux utilisateurs d'évaluer la qualité des réponses et d'utiliser ce feedback pour améliorer le système.

### 2. Gestion de l'Historique de Conversation
Implémenter une mémoire de conversation pour que le chatbot puisse comprendre le contexte des questions précédentes et offrir des réponses plus cohérentes dans le temps.

### 3. Optimisation des Embeddings
Expérimenter avec différents modèles d'embedding pour améliorer la pertinence des documents récupérés, potentiellement en utilisant un modèle plus grand comme BERT ou MPNet.

### 4. Filtrage et Re-ranking
Ajouter une étape de re-ranking après la récupération initiale des documents pour affiner la sélection avant de les envoyer au modèle de génération.

### 5. Parallélisation
Optimiser les performances en parallélisant certaines opérations comme le traitement des documents et la création d'embeddings.

### 6. Recherche Hybride
Combiner la recherche sémantique (embeddings) avec une recherche par mots-clés pour améliorer la précision des documents récupérés.

### 7. Interface Utilisateur Améliorée
Enrichir l'interface web avec des fonctionnalités comme la mise en évidence des sources, la possibilité de voir les documents de référence, et des suggestions de questions.