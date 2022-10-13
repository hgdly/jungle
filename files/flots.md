<!-- 
  flots.md
  Flots
  Hugo D.
  Created : 11 octobre 2022
  Updated : 13 octobre 2022
-->

# Flots <!-- omit in toc -->

- [Introduction](#introduction)
- [Flots dans les réseaux de transport](#flots-dans-les-réseaux-de-transport)

## Introduction

But : aximiser la satifscation globale => faire circuler le plus de biens possible de la production vers la consommation.

Exemples :

- Acheminement de l'eau
- Redirection d'automobilistes
- Acheminements de gaz

D'autres problèmes se ramènent à des problèmes de flots :

- Planification d'examens
- Qualification

## Flots dans les réseaux de transport

> Définition 1 : Réseau de transport (X, U), s, t, c.
>
> - Graphe orienté (X, U)
> - 1 source s, 1 puit t
> - capacité c : $U \rightarrow \mathbb{R}^+ \{+\infty\}$

> Définition 2 : Flot $\varphi$ sur le réseau de transport (X, U), s, t, c.
>
> - $\varphi : U \rightarrow \mathbb{R}^+$. $\varphi(x, y)$ est le flux qui circule dans l'arc (x, y)
> - compatible (avec c) : $\varphi(x, y)\leqslant c(x,y), \forall(x,y)\in U$  
> Si $\varphi(x,y) = c(x,y)$, alors l'arc est saturé
> - vérifie la loi de Kirchhoff (conservation du flux) :  
> $\forall x \in X - \{s,t\}: \sum_{y \in pred(x)}{\varphi(y,x)}=\sum_{y \in succ(x)}{\varphi(x,y)}$  
> en s, t : $\sum_{y \in pred(t)}{\varphi(y,t)}=\sum_{y \in succ(s)}{\varphi(s,y)}$
> - graphe avec arc de retour : $U \cup \{(t, s)\}$ avec c(t, s) = $+\infty$
> - valeur de $\varphi$  : $v(\varphi)=  \varphi(t,s) = \sum_{y \in pred(t)}{\varphi(y,t)}=\sum_{y \in succ(s)}{\varphi(s,y)}$

