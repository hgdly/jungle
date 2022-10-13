<!-- 
  git.md
  Git
  Hugo D.
  Created : 13 octobre 2022
  Updated : 13 octobre 2022
-->

# Git <!-- omit in toc -->

- [Historique de Git](#historique-de-git)
- [Architecture d'un dépôt](#architecture-dun-dépôt)
  - [3 espaces](#3-espaces)
  - [États des fichiers du *working directory*](#états-des-fichiers-du-working-directory)
  - [SHA1](#sha1)
- [Commandes](#commandes)
  - [`checkout`](#checkout)
  - [`commit`](#commit)
  - [`merge`](#merge)
  - [`pull`](#pull)
  - [`push`](#push)
  - [`status`](#status)

## Historique de Git

- Système de contrôle de version décentralisé  
- Logiciel libre sous licence GNU GPL v2  
- 7 avril 2005, première version de Git (1.01)

## Architecture d'un dépôt

### 3 espaces

- *working directory* : espace de travail
- *staging area* ou *index* : aire d'embarquement stockée dans le fichier `.git/index`
- historique : ensemble des *commits* stocké dans le dossier `.git`
  
### États des fichiers du *working directory*

- *untracked* : fichiers non révisionnés
- *unmodified* : fichiers "à jour"
- *modified* : fichiers avant subi des changements non sauvegardés
- *staged* :  fichiers avant subi des changements embarqués dans l'index

### SHA1

## Commandes

### `checkout`

### `commit`

### `merge`

### `pull`

### `push`

### `status`
