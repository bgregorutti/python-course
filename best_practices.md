# Erreurs courantes et bonnes pratiques de programmation

Ce document donne les erreurs ou approximations fréquemment rencontrées chez les personnes débutantes en Python. A noter que ce qui est décrit ne génère pas d'erreurs de syntaxe ou d'exécution.

Je recommande l'utilisation de pylint et les conventions PEP8 pour écrire du code de qualité.

* Utilisation de ";" en fin de ligne : le caractère ";" n'a aucun effet sur l'exécution car l'interpréteur Python ne l'utilise pas pour déterminer les fin de lignes. Python utilise les indentations pour déterminer les blocs de code
* Dans une boucle for, il est préférable d'utiliser le caractère _ si les valeurs de l'objet itérable ne sont pas utilisées. Ex :

```python
for _ in range(10):
    print("hey!")
```

* Dans les structures conditionnelles

Pour les scalaires :

```python
# Mauvaise pratique
if n == 0:
    print("I am zero")

# Bonne pratique
if not n:
    print("I am zero")

```

Pour les objets itérables (list, dict, etc.) :

```python
# Mauvaise pratique
if iterable == []: # ou bien == {}
    print("I am empty")

# Bonne pratique
if not iterable:
    print("I am empty")
```

* Eviter le `range` si c'est possible et préférer

```python
for position, item in enumerate(some_list):
    print(position, item)
```


* "Si je suis vrai, alors je suis vrai"

```python
# Mauvaise pratique, cas 1
if obj == True:
    boolean = True
else:
    boolean = False

# Bonne pratique, cas 1
boolean = obj

# Mauvaise pratique, cas 2
if obj > 0:
    boolean = True
else:
    boolean = False

# Bonne pratique, cas 2
boolean = obj > 0
```

* Eviter de multiplier les `return` dans les `if`, par soucis de lisibilité et pour éviter des erreurs

```python
# Mauvaise pratique
if condition1:
    return something1
else: # le else qui ne sert à rien d'ailleurs
    return something2

# Bonne pratique
if condition1:
    some_value = something1
else:
    some_value = something2
return some_value
```

* Eviter d'utilise l'instruction `pass` qui ne sert à rien

```python
# Mauvaise pratique
if condition:
    return False
else:
    pass
return True

# Bonne pratique
boolean = True
if condition:
    boolean = False
return boolean
```



* Utiliser la fonction isinstance pour tester le type d'un objet

```python
# Mauvaise pratique
if type(obj) == int:
    print("I am an integer")

# Bonne pratique
if isinstance(obj, int):
    print("I am an integer")
```

* Ne pas sur-parenthéser les structures de contrôle (`if`, `for`, `while`) et les `return`. Dans ce dernier cas, une fonction qui retourne plusieurs objets retourne en réalité un `tuple`. Le seul cas où il est nécessaire de mettre des parenthèse dans un `if` est pour des conditions multiples avec l'utilisation des opérateurs `&` (et) et `|` (ou).

```python
# cas 1
if a == 1 and b == 2:
    print("OK")

# cas 2 équivalent
if (a == 1) & (b == 2):
    print("OK")
```

Le cas deux peut être nécessaire pour des conditions complexes.

* Aérer le code, mais pas trop. Utiliser un retour à la ligne pour délimiter des blocs de code logiques
* Choisir des noms de variables, fonctions, classes qui ont du sens et proscrire des noms en une seule lettre. Des exceptions peuvent être **éventuellement** acceptées pour certaines qui concernent des notations mathématiques (i, n, x, y).
* Utiliser des listes/dictionnaires de compréhension quand c'est possible
* Tester si une clé existe dans un dictionnaire

```python
# Mauvaise pratique
if "some_key" in some_dict.keys():
    value = some_dict["some_key"]

# Bonne pratique 1
if "some_key" in some_dict:
    value = some_dict["some_key"]

# Bonne pratique 2
value = some_dict.get("some_key")
if not value:
    print("Key not found")
```

* Objets mutables en argument d'une fonction : si la fonction modifie l'objet passé en argument, il sera modifié globalement et pas uniquement localement (dans la fonction). A éviter donc, ou bien faire une copie explicite en début de fonction
* Chaque objet créé doit être utilisé car ils prennent de l'espace mémoire. Idem pour les imports des packages
* Nommer les objets en lettres minuscules, à l'exception des noms de classes (première lettre en capitale).


