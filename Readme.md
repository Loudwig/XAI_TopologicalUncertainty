Dans le dépôt associé à l’article (disponible sur le GitHub de Théo Lacombe), il y avait déjà une implémentation de la version sur des datasets et modèles de graphes, ainsi que sur un toy dataset.

Pour ne pas implémenter quelque chose d’existant, avec Kilian nous avons choisi de nous concentrer sur l’implémentation de la partie image des datasets présentés dans le papier. Puis, dans un second temps, d’étendre l’étude à un dataset plus complexe que CIFAR.

En faisant ça, on s’est rendu compte qu’il y avait certains points un peu discutables dans les résultats présentés pour la partie image, en particulier concernant la baseline. Les auteurs citent un papier de référence pour cette baseline, mais les résultats qu’ils reportent ne semblent pas cohérents avec ceux du papier cité.

C’est notamment pour cette raison que nous avons ajouté dans le rapport une partie plus critique sur le papier, en plus de la reproduction et de l’analyse expérimentale.
