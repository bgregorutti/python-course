<!-- # Des modules et des librairies pour structurer un projet

Une question fondamentale lorsque l'on travaille sur un projet informatique, quelque soit le langage de programmation, est de savoir comment structurer les fichiers et programmes.
En Python, les modules et les librairies permettent d'organiser, de réutiliser et de structurer le code de manière efficace. Ils jouent un rôle essentiel dans le développement d'applications Python complexes et modulaires. 
Autrement dit, il ne s'agit pas d'écrire des milliers de lignes de codes dans un unique fichier mais de structurer le projet de façon cohérente, et ce en utilisant des modules et des librairies.

Un module est ni plus ni moins qu'un fichier Python contenant des variables, des fonctions et des classes. Tout ce que vous implémentez dans un module peut être importé et réutilisé dans d'autres fichiers composant de votre projet. Une librairie est un ensemble de modules répondant à une organisation bien précise.

Par exemple, supposons que vous ayez un fichier nommé "mymodule.py" contenant le code suivant :

```python
# mymodule.py

def hello():
    print("Hello world")

global_variable = 42
```

Vous pouvez maintenant importer ce module dans un autre fichier "main.py" et utiliser ses fonctionnalités, dès lors que les deux fichiers sont dans le même dossier.

Exemple 1 :

```python
# main.py

import mymodule

mymodule.hello()                # prints "Hello world"
print(mymodule.global_variable) # prints 42
```

Exemple 2 :

```python
# main.py

import mymodule as mod

mod.hello()                # prints "Hello world"
print(mod.global_variable) # prints 42
```

Ici, la variable `mod` est un alias pour le module et permet d'alléger le code.

Exemple 3 :

```python
# main.py

from mymodule import hello, global_variable

hello()                # prints "Hello world"
print(global_variable) # prints 42
```

Dans ce cas, uniquement la fonction et la variable sont importées via l'instruction `from`. Attention à ne pas oublier des éléments à importer pour ne pas que le programme ait une erreur.

Le module `mymodule` ci-dessus est dit local par opposition à un module intégré à une librairie externe, telle que NumPy. L'import d'un module local se fait de la même manière qu'un module externe.


Exemple 4 :

```python
# main.py

from mymodule import *

hello()                # prints "Hello world"
print(global_variable) # prints 42
```

En utilisant le caractère "*", vous importez l'ensemble des objets du module. Cette utilisation, certes très pratique, est **à proscrire**. En effet, elle est induit des conflits dans les variables.

Prenons l'exemple de la fonction `sqrt` qui calcule la racine carrée d'un nombre. Cette fonction est intégrée à la librairie `math` :

```python
# main.py

import math

math.sqrt(4) # returns 2
```

Or cette fonction est également dans la librairie NumPy pour calculer la racine carrée d'un nombre ou de chacun des éléments d'un tableau. Vous comprendrez facilement le problème avec le code suivant :

```python
# main.py

from numpy import *
from math import *

sqrt([1, 2, 3])
```

**Précision sur les librairies**

Une librairie (ou package) est un répertoire qui contient un ensemble de modules Python. Elles sont utilisés pour organiser le code et pour regrouper des fonctionnalités connexes.
Par exemple, l'organisation des fichiers suivantes est suffisante pour définir `mypackage` (le dossier parent) comme une librairie.

```
mypackage/
    __init__.py
    module1.py
    module2.py
```

La seule contrainte est d'avoir un fichier nommé `__init__.py` pour que l'interpréteur reconnaisse le dossier comme une librairie. C'est tout !

Ensuite, chaque fichier `module1.py` et `module2.py` est considéré comme un module par l'interpréteur et peut être importé de la manière suivante :

```python
# main.py

from mypackage import module1

module1.hello()
```









**Utilisation des Modules et des Packages :**

- Les modules et les packages facilitent la réutilisation du code. Vous pouvez organiser vos fonctions et vos classes en modules, puis les importer dans d'autres projets Python lorsque vous en avez besoin.

- Les packages permettent de créer une structure hiérarchique dans votre projet, ce qui est utile pour les projets de grande envergure. Vous pouvez avoir des sous-packages à l'intérieur de packages pour une organisation plus détaillée.

- Python possède une bibliothèque standard riche qui est essentiellement constituée de modules et de packages. Vous pouvez utiliser ces modules pour accomplir diverses tâches sans avoir à réinventer la roue.

- Les modules et les packages rendent le code plus lisible et maintenable, car ils permettent de diviser logiquement le code en morceaux gérables.

En résumé, les modules et les packages sont des outils puissants pour organiser et structurer votre code Python, améliorant ainsi la réutilisabilité, la lisibilité et la maintenance de vos projets. Ils font partie intégrante du développement Python moderne.
 -->





# Librairie SciPy

SciPy est l'une des bibliothèques open-source incontournable de l'écosystème Python. Complémentaire à la librairie NumPy, elle est spécialement conçue pour le calcul scientifique, mathématique et pour l'analyse de données.
<!-- 
![](figs/scipy.png)

## Installation

SciPy est intégrée à Anaconda et ne nécessite pas d'installation explicite.

Toutefois, l'installation se fait soit via `conda` (recommandé), soit `pip` :

```bash
conda install scipy
```

ou bien

```bash
pip install scipy
``` -->


## Fonctionnalités principales de SciPy

1. **Intégration numérique :** des outils avancés pour effectuer des intégrations numériques. Vous pouvez calculer des intégrales simples et multiples, résoudre des équations différentielles ordinaires (EDO), et bien plus encore.

2. **Optimisation :** des méthodes d'optimisation numérique pour résoudre des problèmes d'optimisation avec ou sans contraintes.

3. **Algèbre linéaire :** des fonctionnalités pour effectuer des opérations d'algèbre linéaire, telles que la résolution de systèmes linéaires, la décomposition en valeurs singulières (SVD) et la résolution de problèmes aux moindres carrés.

4. **Traitement du signal :** des fonctions pour traiter des signaux faire de l'analyse spectrale : transformée de Fourier rapides et discrètes (FFT et DFT), filtrage, convolution, etc.

5. **Statistiques :** une large gamme de fonctions statistiques, notamment des tests d'hypothèses, des distributions statistiques, des fonctions de densité de probabilité, etc.

6. **Interpolation :** des fonctions d'interpolation de données : interpolation polynomiale, spline, etc.

7. **Traitement d'Images :** La bibliothèque prend en charge le traitement d'images, y compris la lecture, l'écriture et la manipulation d'images, ainsi que des opérations telles que la convolution et la transformation.
<!-- 
## Applications courantes de SciPy

**Analyse de données :** en complément des bibliothèques NumPy et Pandas, SciPy offre un large éventail de fonctionnalités pour analyser, manipuler et visualiser des ensembles de données. Par exemple, vous pouvez calculer des statistiques descriptives, effectuer des tests d'hypothèses, ajuster des modèles statistiques, etc. Par exemple :

```python
from scipy import stats

# Générer des données aléatoires
data = stats.norm.rvs(size=1000)

# Test de normalité
p_value = stats.normaltest(data).pvalue
```

**Modélisation et simulation :** SciPy permet de modéliser et simuler des systèmes dynamiques. Les capacités de résolution d'équations différentielles ordinaires (EDO) de SciPy sont essentielles pour simuler des phénomènes qui évoluent dans le temps. Par exemple, vous pouvez modéliser un oscillateur harmonique et visualiser son comportement au fil du temps :

```python
from scipy.integrate import solve_ivp

# Définition de l'EDO pour un oscillateur harmonique
def oscillateur(t, y):
    return [y[1], -y[0]]

# Conditions initiales
y0 = [1, 0]  # Position initiale et vitesse

# Intervalle de temps
t_span = (0, 10)

# Résolution de l'EDO
sol = solve_ivp(oscillateur, t_span, y0, t_eval=range(10))
```

**Traitement du signal :** les ingénieurs et les chercheurs en traitement du signal utilisent SciPy pour effectuer diverses opérations de traitement du signal, telles que la convolution, la détection de pics et la transformation de Fourier. Par exemple, vous pouvez appliquer une convolution à un signal pour lisser ou filtrer les données :

```python
from math import sin, pi
from scipy import signal

# Générer un signal
nr_points = 1000
time_list = [item / nr_points for item in range(nr_points)]
signal_in = [sin(2 * pi * 5 * t) for t in time_list] # Signal sinusoïdal

# Appliquer une convolution
kernel = signal.gaussian(50, std=10)
signal_out = signal.convolve(signal_in, kernel, mode="same") / sum(kernel)
```


**Optimisation :** SciPy offre diverses méthodes numériques pour résoudre des problèmes d'optimisation, avec ou sans contraintes. Voici un exemple de minimisation d'une fonction :

```python
from scipy.optimize import minimize

# Définition de la fonction à minimiser
def objectif(x):
    return x**2 + 2*x + 1

# Minimisation de la fonction
resultat = minimize(objectif, x0=0)

print(f"Minimum trouvé à x = {resultat.x}")
``` -->


# Librairie NumPy

La librairie NumPy est une référence pour le calcul numérique et l'analyse de données. Pour certains problèmes mathématiques et/ou numériques, elle complète parfaitement SciPy.

NumPy c'est :

1. Une structure de données représentant des tableaux multidimensionnels
2. Un ensemble très complet de routines permettant des opérations optimisées sur les tableaux : opérations mathématiques et logiques, tris, algèbre linéaire, opérations statistiques et simulations aléatoires, etc.

<!-- <img width="30%" height="30%" src="https://numpy.org/images/logo.svg"> -->


<!-- ## Installation et importation

Comme pour SciPy, NumPy est intégrée à Anaconda et ne nécessite pas d'installation explicite.

Toutefois, l'installation se fait soit via `conda` (recommandé), soit `pip` :

```bash
conda install numpy
```

ou bien

```bash
pip install numpy
```

L'import se fait usuellement via un alias et c'est ce que vous verrez dans la plupart des forums spécialisés :

```python
import numpy as np
``` -->

## Structure de données `ndarray`

L'intérêt majeur de NumPy est la définition d'une nouvelle structure de données mutable représentant des tableaux multidimensionnels, les `ndarray`.
Cette nouvelle structure est utilisée pour stocker et manipuler des données numériques de manière optimisée, en particulier lorsqu'elles sont de très grande taille.

### Création de tableaux NumPy

Vous pouvez créer un tableau NumPy en utilisant la fonction `np.array()`, en général à partir d'un objet itérable tel qu'une liste ou un tuple

```python
import numpy as np

arr = np.array([1, 2, 3, 4])  # 1D
print(arr)

arr = np.array([[1, 2], [3, 4]]) # 2D
print(arr)

arr = np.array([[[1, 2], [3, 4]], [[10, 20], [30, 40]]]) # 3D
print(arr)
```

ou bien en utilisant des routines spécifiques

```python
empty_array = np.empty(shape=(2, 3))  # Crée un tableau vide 2D avec 2 lignes et 3 colonnes
print(empty_array)

ones = np.ones(shape=10)  # Crée un tableau 1D avec des 1
print(ones)

zeros = np.zeros(shape=(2, 3, 4))  # Crée un tableau 3D avec des 0 : 2 tableaux 2D de taille (3, 4)
print(zeros)

identity = np.eye(10) # Crée une matrice identité de taille 10 x 10
print(identity)

numbers = np.linspace(0, 10, 20) # Génère une série de 20 nombres régulièrement espacés entre 0 et 10
print(numbers)

numbers = np.arange(0, 10, .1) # Génère une série de nombres régulièrement espacés entre 0 et 10 avec un pas de 0.1
print(numbers)
```

Par abus de langage, on parle de façon indifférenciée de
* **tableau** pour désigner un objet de classe `ndarray`
* **vecteur** pour désigner un tableau 1D
* **matrice** pour désigner un tableau 2D. A noter qu'il existe un type `matrix` dans NumPy pour des données 2D mais ce type sera définitivement supprimé dans les versions futures.
* **tensor** pour désigner un tableau > 2D. Ce cas est particulièrement adapté pour des images et dans les réseaux de neurones.


### Quelques propriétés des tableaux NumPy

- **Dimensions, attribut `shape` :** l'attribut `shape` d'un tableau est un tuple indiquant le nombre d'éléments dans chaque dimension. Par exemple :

```python
arr = np.array([1, 2, 3, 4])  # 1D
arr.shape   # renvoie (4,)

zeros = np.zeros(shape=(2, 3, 2))
zeros.shape # renvoie (2, 3, 2)
```
- **Nombre de dimensions, attribut `ndim` :** l'attribut ndim renvoie le nombre de dimensions du tableau, comme par exemple :

```python
zeros = np.zeros(shape=(2, 3, 2))
zeros.ndim # renvoie 3
```
- **Type des données stockées, attribut `dtype` :** les tableaux NumPy sont **homogènes**, ce qui signifie qu'ils contiennent des éléments de même type de données. Généralement, le type est un type numérique (`int` ou `float`) mais il est possible d'y stocker d'autres types. Le type des données stockées dans un tableau est inféré automatiquement mais peut être spécifié explicitement.

```python
arr = np.array([1, 2, 3, 4])
arr.dtype # renvoie dtype('int64')

arr = arr.astype("float")  # change le type en float et écrase la variable
arr.dtype # renvoie dtype('float64')
```

Notons au passage que NumPy implémente une extension des types numériques. Il s'agit de
* `np.int64` et `np.int32` : des entiers codés respectivement sur 64 bits et 32 bits
* `np.float64` et `np.float32` : des flottants codés respectivement sur 64 bits et 32 bits
* `np.complex` : pour les nombres complexes
* `np.bool` : pour les booléens


### Accès aux éléments d'un tableau NumPy et indexation

**Indexation simple :**

L'accès aux éléments d'un tableau NumPy se fait en spécifiant les positions pour chaque dimension. 

```python
my_list = [[1, 2], [3, 4]]

arr = np.array(my_list)
print(arr[0, 1])  # Accède à l'élément à la première ligne (indice 0) et deuxième colonne (indice 1)

print(my_list[0][1]) # équivalent pour la liste
```

**Indexation avancée :**

NumPy prend en charge l'indexation avancée, ce qui vous permet d'extraire des sous-ensembles de données en fonction de certaines conditions. Pour rappel, accéder à un sous-ensemble d'une liste s'effectue de la manière suivante :

```python
my_list[start:stop:step]
```

dans le cas des tableaux NumPy, c'est pareil mais en spécifiant les indices `start`, `stop` et `step` dans chacune des dimensions en les séparant par des virgules :

```python
arr = np.array([[1, 2], [3, 4], [5, 6]])

print(arr[1:3]) # renvoie les lignes 2 et 3 et toutes les colonnes
print(arr[1:3, :]) # idem en utilisant ":" pour la seconde dimension
print(arr[1:3, 0]) # renvoie les lignes 2 et 3 et la première colonne
print(arr[0:3:2, :]) # renvoie les lignes 1 et 3 et toutes les colonnes
```

**Boolean indexing :**

L'indéxation par booléen, ou boolean indexing en anglais, désigne une façon de récupérer un sous-ensemble d'un tableau de données selon des valeurs booléennes. Ce cette manière, vous pouvez spécifier non pas les indices mais un vecteur de booléen correspondant aux lignes que vous souhaitez extraire ou non.

Dans l'exemple suivant 

```python
booleans = np.array([True, False, True])
arr = np.array([[1.1, 1.2], [3, 4], [1.5, 1.6]])
extract = arr[booleans]
```

on extrait la première et la troisième ligne du tableau, selon la position des `True` dans le vecteur `booleans`.

*Remarque : on parle parfois de masque, au sens où l'on souhaite masquer une partie des données.*

Il est commun d'utiliser ce type d'indexation pour la sélection des éléments du tableau selon d'une **condition** :

```python
arr = np.array([[1.1, 1.2], [3, 4], [1.5, 1.6]])
extract = arr[arr<2]
```

Dans ce cas, on filtre les valeurs du tableau pour ne conserver que celles qui satisfont la condition. Le résultat est un tableau 1D, malgré le fait que l'expression `arr<2` retourne un tableau 2D.
Il est possible de transformer le tableau résultant en un tableau 2D en utilisant la méthode `reshape` : 

```python
extract.reshape((2, 2))
```

### `np.array` ou `list` ?

La façon de définir les tableaux et l'indexation change assez radicalement la façon de travailler sur des données, par rapport aux listes. En effet, toutes les opérations possibles sur les tableau NumPy sont optimisées pour agir sur tous les éléments en même temps sans avoir besoin d'utiliser des boucles.

Par exemple, il est très simple d'accéder à une colonne d'un tableau 2D alors que dans le cas d'une liste, il aurait fallu utiliser une boucle for, plus couteuse :

```python
def extract_column(data_list_2d, idx):
    return [item[idx] for item in data_list_2d]
```

Mieux, les opérations mathématiques (addition, soustraction, multiplication et division) sont prises en charge de même que les opérations matricielles plus complexes.

```python
# Exemple d'opération sur les tableaux
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
addition = a + b
multiplication = a * b
print(addition)        # Renvoie [5 7 9]
print(multiplication)  # Renvoie [4 10 18]
```

Exemples de fonctions d'algèbre linéaire (dans le module `np.linalg`)

```python
a = [[1, 0], [0, 1]]
b = [[4, 1], [2, 2]]

# Produit matricel
np.dot(a, b)

# Inverse
np.linalg.inv(b)

# Norme matricielle
np.linalg.norm(b)
```

## Fonctions mathématiques

NumPy offre un grand nombre de fonctions pour effectuer des opérations mathématiques et statistiques sur les tableaux. Par exemple, `np.sum()`, `np.mean()`, `np.max()`, `np.min()`, entre autres, étendent les opérations de la librairie `math`.


Dans le cas ci-dessous, les opérations s'effectuent sur un tableau 2D mais renvoie une unique valeur. De façon équivalente, les résultats sont les mêmes sur le tableau **vectorisé** via la fonction `np.ravel()`.

```python
# opérations mathématique sur un tableau 2D
data = np.array([[1, 2], [3, 4], [5, 6]])
print(np.sum(data))
print(np.mean(data))
print(np.max(data))
print(np.min(data))

# De façon équivalente
flat_data = np.ravel(data)
print(np.sum(flat_data))
print(np.mean(flat_data))
print(np.max(flat_data))
print(np.min(flat_data))
```

**Argument `axis`**

La puissance de NumPy est aussi sa flexibilité. En effet, si l'on peut faire des opérations sur l'ensemble des éléments du tableau comme ci-dessus, il est également possible de faire les mêmes opérations sur les lignes ou les colonnes en utilisant l'argument `axis`.

```python
data = np.array([[1, 2], [3, 4], [5, 6]])
print(np.sum(data, axis=0)) # somme de toutes les lignes, pour chaque colonne
print(np.sum(data, axis=1)) # somme de toutes les colonnes, pour chaque ligne
```

Dans ce cas, le résultat n'est plus un scalaire mais un tableau de dimension égale au nombre de lignes ou au nombre de colonnes.

*Remarque : vous retrouverez l'argument axis dans plein d'autres fonctions NumPy mais également dans la librairie Pandas*


## Module `np.random`

Le module `np.random` permet de générer des nombres aléatoires et effectuer des opérations liées au hasard. Il offre un contrôle sur le processus de génération des nombres aléatoires, ce qui est utile applications scientifiques, statistiques et de simulation de processus stochastiques.

Les concepts de base abordés dans cette section :
1. Génération de nombres aléatoires
2. Distribution de probabilité : distribution uniforme, Gaussienne, etc.
3. Échantillonnage aléatoire
4. Permutation et mélange

Attention à ne pas confondre `np.random` avec `random` présente dans la librairie standard de Python. Tout comme les opérations mathématiques, NumPy étend la génération de nombres aléatoires à des tableaux. Dans les exemples ci-dessous, nous comparons les deux librairies.


**Génération de nombres aléatoires**

NumPy propose diverses fonctions pour générer des nombres aléatoires, tels que `rand`, `randint`, `randn`. En particulier, pour générer aléatoirement un entier entre 0 et 10, on utilise la fonction `randint`.

```python
import random
import numpy as np

# numpy
print(np.random.randint(0, 10))

# random
print(random.randint(0, 10))
```

Ces fonctions permettent de générer un unique nombre (pseudo-)aléatoire. Bien entendu, l'intérêt de NumPy n'est pas limité à ce cas précis. En effet, NumPy permet de générer des tableaux, objets ndarray, dont les éléments sont générés aléatoirement, en général via un argument `size`. En utilisant `random`, on est obligé de passer par une boucle (liste de compréhension ici).

```python
import random
import numpy as np

size = 5

# numpy
print(np.random.randint(low=0, high=10, size=size))

# random
print([random.randint(a=0, b=10) for _ in range(size)])
```

Le résultat de `np.random.randint` est un tableau 1D car l'argument size est un entier. Comment générer un tableau de dimension supérieure à 1 ?

```python
import random
import numpy as np

size = (3, 5)

# numpy
print(np.random.randint(low=0, high=10, size=size))

# random
nrows, ncols = size
print([[random.randint(a=0, b=10) for _ in range(ncols)] for _ in range(nrows)])
```

(ça devient compliqué en utilisant `random`...)

*Remarque : le caractère "_" dans une boucle `for` permet de ne pas stocker la valeur récupérée depuis l'objet itérable, si bien sûr elle n'est pas nécessaire.*


**Distribution de probabilité : uniforme, Gaussienne, etc.**

La plupart des loi de probabilité sont intégrées à NumPy. Dans les exemples ci-dessous, nous générons des nombres aléatoires selon un distributions uniforme et Gaussienne. Vous trouverez aisément comment générer selon d'autres lois de probabilité.

Distribution uniforme :

```python
import random
import numpy as np

size = 5

# numpy / 1D
print(np.random.rand(size))
print(np.random.uniform(low=0, high=1, size=size))

# random
print([random.uniform(a=0, b=1) for _ in range(size)])

# numpy > 1D
shape = (3, 5, 2)
print(np.random.uniform(low=0, high=1, size=shape))
```

Distribution Gaussienne :

```python
import random
import numpy as np

size = 5

# numpy / 1D
print(np.random.normal(loc=0, scale=1, size=size))

# random
print([random.gauss(mu=0, sigma=1) for _ in range(size)])

# numpy > 1D
shape = (3, 5, 2)
print(np.random.normal(loc=0, scale=1, size=shape))
```


**Échantillonnage aléatoire**

NumPy permet d'effectuer un échantillonnage aléatoire, avec ou sans remplacement, à partir d'un tableau de données. Ici encore, l'utilisateur peut renseigner une valeur à l'argument `size` pour la dimension du tableau de données généré. L'argument peut être un tuple si l'on souhaite un tableau de dimension supérieure à 1.

```python
import random
import numpy as np

size = 5
data = [1, 2, 3, 4, 5]
replace = True

# numpy
print(np.random.choice(data, size=size, replace=replace))

# random with replacement
print([random.choice(data) for _ in range(size)]) 
print(random.choices(data, k=size))

# random without replacement
sample = []
while data:
    random_number = random.choice(data)
    if random_number in data:
        sample.append(random_number)
        data.pop(data.index(random_number))
```


**Permutation et mélange**

Vous pouvez utiliser NumPy pour permuter ou mélanger des éléments dans un tableau. Par exemple, pour mélanger un tableau :

```python
# Mélanger un tableau d'éléments
tableau = np.array([1, 2, 3, 4, 5])
np.random.shuffle(tableau)
print(tableau)
```





## Importer et stocker des données

Lorsque l'on travaille sur des données, il est important de savoir les importer et les stocker. La façon se faire dépend du format du fichier que vous aurez à traiter. Le format CSV, pour comma-separated values, est très courant car très facile à ouvrir à la fois dans des outils de bureautique mais aussi dans python. On parle d'ailleurs de **données structurées** car elles sont organisées en lignes et colonnes.

Le CSV n'est pas le seul format rencontré en pratique :
   * `.txt` : format similaire au CSV mais avec un séparateur parfois différent (espace, tabulation, etc.)
   * `.npy` et `.npz` : format spécifique à NumPy pour stocker des arrays
   * `.json` et `.xml` : formats de **données non structurées** suivant la logique clé/valeur

Dans NumPy, les fichiers CSV ou TXT peuvent être importés via les fonctions `np.loadtxt` et `np.genfromtxt`. Exemple :

```python
np.loadtxt("myfile.txt", delimiter=" ")
np.genfromtxt("myfile.csv", delimiter=",")
```

L'argument `delimiter` est une chaine de caractères indiquant le délimiteur défini dans les données. Les plus courants sont l'espace, la virgule ou le point-virgule.

A l'inverse, des tableaux NumPy peuvent être stockés dans les mêmes formats avec la fonction `np.savetxt`, en indiquant le délimiteur :

```python
arr = np.ones((5, 5))
np.savetxt("ones.csv", arr, demiliter=";")
```

Il est à noter que des fichiers TXT ou CSV compressé (avec ZIP, GZ ou autres) également être importés sans décompression préalable :

```python
np.loadtxt("myfile.zip", delimiter=" ")
```


Comme dit plus haut, NumPy possède deux formats spécifiques aux arrays, `.npy` et `.npz`. Il s'agit de stocker les tableaux dans des fichiers binaires, lisibles uniquement avec Python. Le format `.npy` permet le stockage d'un unique tableau tandis que le format `np.npz` permet le stockage de plusieurs tableaux dans un seul fichier.

Stocker des tableaux :

```python
ones = np.ones((5, 5))
zeros = np.zeros((2, 5, 5))

# Format .npy
np.save("ones", ones)

# Format .npz, specifying the two arrays
np.savez("multiple_arrays", ones, zeros)
```

Lire des tableaux :

```python
# Format .npy
ones = np.load("ones.npy")

# Format .npz
obj = np.load("multiple_arrays.npz")
zeros = obj["arr_0"]
ones = obj["arr_1"]
```

L'objet que retourne la fonction `np.load` dans le cas d'un fichier .npz n'est pas lisible directement. Il s'apparente à un dictionnaire, dans lequel les tableaux sont accessibles par leur clé. Par défaut, les clés sont nommées arr_0, arr_1, etc. mais il est possible de modifier les clés pour des termes plus précis lors du stockage.



Références :

* [NumPy: the absolute basics for beginners](https://numpy.org/doc/stable/user/absolute_beginners.html)
* [Random sampling](https://numpy.org/doc/1.16/reference/routines.random.html)







# Librairie Pandas pour la manipulation de données

Pandas est une bibliothèque très largement utilisée dans le domaine de l'analyse des données. Elle propose un grand nombre d'outils destinés à manipuler et analyser des **données tabulaires** et repose sur des structures de données adaptées.

<!-- <img height="300" src="https://pandas.pydata.org/static/img/pandas_mark.svg"> -->


<!-- ## Installation et importation

Pandas est intégrée à Anaconda et ne nécessite pas d'installation explicite dans la plupart des cas. On peut cependant l'installer via `pip` ou `conda` 

```
pip install pandas
conda install pandas
```

La convention d'importation est la suivante :

```python
import pandas as pd
``` -->

## Structures de données de Pandas

Pandas repose sur deux structures de données principales : les séries pour stocker un tableau unidimensionnel et les DataFrames pour les tableaux bidimensionnels.

Concrètement, les données tabulaires sont un ensemble de données structurées en lignes et en colonnes et il est important de se munir d'une structure de données adaptée pour les analyser en Python. Par exemple, en statistique, on parle d'observations ou d'individus pour désigner les lignes et de variables, de covariables ou de features pour désigner les colonnes. Une fois ce formalisme défini, l'utilisateur peux conduire des analyses statistiques sur les individus ou les variables.

A la différence de NumPy, Pandas ne propose pas de structure de données de dimension supérieure à 2. De plus le type des données stockées via Pandas est mixte au sens où chaque colonne à un type de données fixé mais différents d'un colonne à une autre.
On retrouvera cependant quelques concepts rendant Pandas et NumPy deux librairies complémentaires.

### Séries, tableau unidimensionnel

Une série est un tableau unidimensionnel capable de contenir n'importe quel type de données. Elle est similaire à une colonne ou une ligne d'un tableau. Pour créer une série, vous pouvez utiliser la fonction `pd.Series()` en y passant un objet itérable, une liste ici :

```python
import pandas as pd

data = [1, "hello", True, 3.14, -1]
serie = pd.Series(data)
print(serie)
```

L'affichage de la série est différent de ce que peu donner NumPy. Vous observerez en effet trois choses :

* les données affichées en colonnes
* les valeurs numérotées de 0 à 4. On parle d'indices de la série
* le type des données, ici `object` car le types des données est mixte (entiers, flottant, booléen et chaîne de caractères)

La structure de données `pd.series` a quatre arguments principaux : `data`, `index`, `dtype` et `name`.

L'argument `data` est naturellement pour renseigner les données à stocker dans la série (une liste par exemple).

L'argument optionnel `index` permet de définir l'indices des éléments et d'y accéder facilement.
Par défaut, les indices sont des incréments partants de 0.

L'argument optionnel `dtype` permet de définir le type des données. S'il n'est pas renseigné, Pandas infère le type des données

L'argument optionnel `name` permet de donner un nom à la série.

```python
data = [1, "hello", True, 3.14, -1]
serie = pd.Series(data=data, index=("item_1", "item_2", "item_3", "item_4", "item_5"), name="example")
print(serie)
```

**Accès aux éléments :**

L'accès aux éléments d'une série se fait de façon similaire à un tableau NumPy en spécifiant l'index de l'élément souhaité :

```python
serie["item_1"]
```

Il est également possible de faire du *slicing* pour accéder aux éléments de façon plus avancée :

```python
serie["item_1":"item_3":2]
```

Dans l'exemple, on extrait les éléments correspondant aux positions allant de `item_1` à `item_3` par pas de 2.

De façon alternative, et on reverra cette manière de faire un peu plus bas, on peut utiliser l'attribut `.loc` :

```python
serie.loc["item_1"]
serie.loc["item_1":"item_3":2]
```

**Opérateurs mathématiques :**

Comme pour NumPy, les opérateurs mathématiques de base (addition, produit, division, différence, etc.) sont intégrés à Pandas. Les opérations se font éléments par éléments et renvoie une série.

```python
data = [1, 2, 3, 4, 5]
serie = pd.Series(data=data)

print(serie + serie)
print(serie * 2)
print(serie ** 2)
print(serie / 2)
print(serie // 2)
```

Si le type des données n'est pas numérique, ces opérations peuvent quand même être possible (voir le chapitre sur le typage). C'est le cas notamment pour les chaînes de caractères pour lesquelles les opérateurs `+` et `*` sont définis :

```python
serie = pd.Series(data=["a string", "another string"])
print(serie + serie)
print(serie * 3)
```



### DataFrame, tableau bidimensionnel

Une DataFrame est une structure de données bidimensionnelle, similaire à une table ou à une feuille de calcul et utilisée largement pour des analyses statistiques. Il est composé de lignes (les observations) et de colonnes (les variables) identifiées respectivement par des *index* ou des *labels* permettant d'identifier la position de chaque élément du tableau. De fait, chaque ligne (resp. colonne) de la dataframe est un ensemble de séries partageant **le même index** (resp. **label**). 

Pour créer une dataframe, vous pouvez utiliser la fonction `pd.DataFrame()` avec un objet itérable contenant les éléments à stocker.

Exemple 1 : `list -> DataFrame`

```python
import pandas as pd

data = [('Alice', 25), ('Bob', 30), ('Charlie', 35)]
df = pd.DataFrame(data=data, columns=("name", "age"), index=("item_1", "item_2", "item_3"))
print(df)
```

Exemple 2 : `dict -> DataFrame`


```python
import pandas as pd

data = {'name': ['Alice', 'Bob', 'Charlie'], 'age': [25, 30, 35]}
df = pd.DataFrame(data=data, index=("item_1", "item_2", "item_3"))
print(df)
```

L'avantage de l'utilisation d'un dictionnaire pour définir une dataframe est de ne pas avoir besoin de spécifier les noms de colonnes dans la définition car ils sont explicites.

Pour afficher les caractéristiques du tableau :

```python
print(df.index) # affiche les labels des lignes
print(df.columns) # affiche les noms de colonnes
print(df.shape) # affiche les dimensions du tableau
print(df.info())
print(df.dtypes)
```


**Accès aux éléments :**

L'accès à un ou plusieurs éléments d'une dataframe est similaire aux séries mais il faut distinguer plusieurs cas.

Cas 1 (sélection d'une colonne) : on sélectionne une colonne en spécifiant le **label** correspondant.

```python
df["name"]
df.names # alternative
```

Cas 2 (sélection d'une ligne) : on sélectionne une colonne en spécifiant l'**index** correspondant et avec l'attribut `.loc`.

```python
df.loc["item_1", :]
```

Cas 3 (sélection d'un élément) : on sélectionne un élément en spécifiant l'**index** et la **colonne** correspondant et avec l'attribut `.loc`.

```python
df.loc["item_1", "name"]
```

Bien entendu, il est possible d'extraire un sous-ensemble en utilisant la méthode du slicing comme pour les séries.

*Remarques :*
* *sélectionner une ligne ou une colonne entière renvoie un objet de type `pd.series`.*
* *Pandas permet également de définir des indices/labels multi-niveaux (ou hiérarchiques), voir [ici](https://pandas.pydata.org/docs/user_guide/advanced.html)*

## Opérations sur les dataframes

### Opérateurs mathématiques

Les opérations mathématiques de base sont possibles pour des Dataframe de la même manière que pour les séries, si bien sûr elles sont définies pour les types concernés. Les opérations s'effectuent élément par élément :

```python
data = {'name': ['Alice', 'Bob', 'Charlie'], 'age': [30, 25, 35]}
df = pd.DataFrame(data=data)

print(df + df)
print(df * 3)
```

Dans ce cas, les opérations mathématiques sont limitées car les données sont de type différents (`int` et `str`). Python n'essaiera pas d'effectuer le calcul si l'opération n'est pas possible sur l'une des colonnes.

### Fonctions statistiques

Comme dit plus haut, l'usage classique de Pandas est l'analyse de données. Il est donc naturel d'avoir recours à des statistiques de base telles que le minimum, le maximum, la moyenne, etc.
Ces statistiques sont calculées à partir d'une dataframe préalablement construite, via les méthodes suivantes :
* `df.min()` et `df.max()` : minimum et maximum
* `df.mean()`, `df.median()` et `df.std()` : moyenne, médiane
* `df.std()` et `df.var()` : écart-type et variance
* `df.cov` et `df.corr` : covariance et corrélation

Cette liste est non exhaustive et ce sont les méthodes principales.

Il est intéressant de noter que ces méthodes ont exactement des arguments en communs :
* `axis` : faire le calcul sur les lignes ou les colonnes
* `skipna` : doit-on exclure les valeurs manquantes ?
* `level` : faire le calcul sur un niveau particulier dans le cas d'une dataframe multi-niveau (voir [ici](https://pandas.pydata.org/docs/user_guide/advanced.html))
* `numeric_only` : inclure des colonnes de type non numérique

***A vous de jouer !***

*Tester le code ci-dessous et commenter le résultat :*

```python
data = {'name': ['Alice', 'Bob', 'Charlie'], 'age': [30, 25, 35]}
df = pd.DataFrame(data=data)
df.min()
```

En complément de ces méthodes, on peut mentionner la méthode `df.describe()` qui calcul des statistiques de base pour l'ensemble des colonnes de la dataframe.
Or, si la dataframe contient à la fois des types numériques et non-numériques, le calcul ne se fera que sur les valeurs numérique...

Pour inclure les variables non-numériques, des chaînes de caractères par exemple, utiliser `df.describe(include=object)`.


## Chargement et manipulation de données

Pandas offre une grande variété de méthodes pour charger des données à partir de différentes sources. Pour des fichiers CSV, la méthode la plus courante est `pd.read_csv()`

```python
import pandas as pd
data = pd.read_csv("donnees.csv", sep=",")
```

Le gros avantage de cette méthodes (et de Pandas) est que les types de données lus et les noms de colonnes sont inférés lors de l'importation et non renseignés pas l'utilisateur.

A part le format CSV, tous les formats standards sont importable de la même manière pour avoir un objet de type DataFrame :
* `read_excel`
* `read_html`
* `read_sql`
* `read_xml`
* `read_json`
* etc.

Le premier argument de ces méthodes est le chemin vers le fichier à importer ou son URL si le fichier en question est sur un serveur distant.
Encore plus fort, Pandas peut importer un fichier zippé sans décompression préalable :

```python
pd.read_csv("foo.zip")
```

L'opération inverse, sauvegarder une dataframe dans un fichier CSV se fait de la manière suivante :

```python
data.to_csv("donnees_modifiees.csv", index=False)
```

L'argument booléen `index` permet de sauvegarder (ou pas) les indices de la dataframe. Si les indices de la dataframe sont des incréments, il n'est pas utile de les sauvegarder, d'où la valeur `False` dans l'exemple.


### Exploration des données

Comme dit plus, Pandas offre de nombreuses fonctions pour explorer rapidement vos données. Vous pouvez utiliser `head()`, `tail()`, `info()`, `describe()` et d'autres méthodes pour en obtenir un aperçu :

```python
import pandas as pd

data = pd.read_csv("donnees.csv")
print(data.head())      # Affiche les premières lignes
print(data.tail())      # Affiche les dernières lignes
print(data.describe())  # Résume les statistiques des données
```



### Manipulation des données

Pandas offre des fonctionnalités pour manipuler les données. 

**Trier les valeurs :**

```python
import pandas as pd

df = pd.DataFrame(data={"name": ["Alice", "Bob", "Charlie"], "age": [30, 25, 35]})
df_sorted = data.sort_values(by="Age")
print(df_sorted)
```

**Grouper et agréger :**

La méthode groupby, similaire au SQL, permet de grouper les observations de la dataframe en fonction des valeurs d'une colonne particulière :

```python
df = pd.DataFrame({"Animal": ["Falcon", "Falcon", "Parrot", "Parrot"], "Max Speed": [380., 370., 24., 26.]})
gb = df.groupby(["Animal"]) # groupby
gb.mean() # agregation
```

Le résultat du groupby est ensuite agrégé par une statistique qui agit sur chacun des groupes, ici la moyenne.

Les autres statistiques incluent les functions : `min`, `max`, `count`, `sum`, etc.

Parfois, il est nécessaire de définir la fonction d'agrégation explicitement et d'utiliser la méthode `apply()` sur l'objet `gb` :

```python
gb = df.groupby(["Animal"]) # groupby
gb.apply(lambda x: x.median()) # compute the median on the results
```

Voir aussi les fonctions :
* `pivot_table()`
* `crosstab()`


**Concaténer plusieurs dataframes :**

La fonction `pd.concat` permet de rassembler plusieurs dataframes ou séries en une seule dataframe selon un axe donné. Le fonctionnement est le suivant

```python
df1 = pd.DataFrame({"col1": ["a", "b", "c", "d"], "col2": [1, 2, 3, 4]})
df2 = pd.DataFrame({"col1": ["e", "e", "f", "f"], "col2": [10, 20, 30, 40]})
list_of_dfs = [df1, df2]
pd.concat(list_of_dfs)
```

où `liste_of_dfs` est une liste d'objets de type `DataFrame`.

Par défaut, les tableaux sont concaténés *en lignes*, c'est-à-dire que de nouvelles lignes correspondant aux différentes dataframe sont ajoutées. Cela correspond à passer une valeur à l'argument `axis` comme suit :

```python
pd.concat(list_of_dfs, axis=0)
```

*Remarque : notez que les indices ne sont plus uniques. Ils correspondent aux indices des dataframes initiales. Pour ignorer les indices et en créer de nouveaux, faire :*

```python
pd.concat(list_of_dfs, axis=0, ignore_index=True)
```

Question : Que se passe-t-il si les dataframes n'ont pas les mêmes noms de colonnes ?

A l'inverse, si l'utilisateur passe la valeur 1 à l'argument `axis`, de nouvelles colonnes sont ajoutées dans la dataframe finale. Exemple :

```python
pd.concat(list_of_dfs, axis=1)
```

Dans ce cas, les dataframes sont assemblées selon les indices communs, s'il y en a et il est possible que les noms de colonnes soient les mêmes comme c'est le cas dans l'exemple.

Question : Que se passe-t-il si les dataframes n'ont pas les mêmes indices ?



**Réorganiser une dataframe**

En pratique, il existe deux manières d'organiser des données tabulaires.

Le format large (*wide* en anglais) désigne une organisation où les lignes correspondent à des individus uniques et les colonnes à des caractéristiques distinctes. On parle également de table de pivot ou de tableau croisé. Dans l'exemple ci-dessous, la variable *Person* correspond aux individus distincts et les variables *Age*, *Weight* et *Height* à leurs caractéristiques :

| Person  | Age  | Weight  | Height  |
|:-------:|:----:|:-------:|:-------:|
| Bob     | 32   | 168     | 180     |
| Alice   | 24   | 150     | 175     |
| Steve   | 64   | 144     | 165     |

A l'inverse, le format long (*stacked* en anglais) contient trois colonnes distinctes, peu importe le nombre des caractéristiques collectées. La première correspond à l'identifiant des individus (ex. *Person*), la deuxièmes correspond à l'identifiant des caractéristiques (ex. *Variable*) et la troisième correspond aux valeurs des caractéristiques (ex. *Value*).

| Person  | Variable  | Value  |
|:-------:|:---------:|:------:|
| Bob     | Age       | 32     |
| Bob     | Weight    | 168    |
| Bob     | Height    | 180    |
| Alice   | Age       | 24     |
| Alice   | Weight    | 150    |
| Alice   | Height    | 175    |
| Steve   | Age       | 64     |
| Steve   | Weight    | 144    |
| Steve   | Height    | 165    |

Dans Pandas, le format large est obtenu via la méthode `pivot` :

```python
df = pd.DataFrame({"Person": ["Bob", "Bob", "Bob", "Alice", "Alice", "Alice", "Steve", "Steve", "Steve"],
                   "Variable": ["Age", "Weight", "Height", "Age", "Weight", "Height", "Age", "Weight", "Height"],
                   "Value": [32, 168, 180, 24, 150, 175, 64, 144, 165]})
df_long = df.pivot(index="Person", columns="Variable", values="Value")
```

L'opération inverse qui consiste à passer du format large au format long se fait via la fonction `pd.melt` :

```python
df_wide = pd.melt(df_long)
```

Or dans ce cas, l'information des personnes, c'est-à-dire les indexes de la dataframe `df_long`, est perdue. Pour la conserver, procéder comme suit :

```python
pd.melt(df_long.reset_index(), id_vars=["Person"])
```

### Visualisation des données

Bien que Pandas ne soit pas principalement une bibliothèque de visualisation, il s'intègre bien avec d'autres bibliothèques de visualisation telles que Matplotlib et Seaborn pour créer des graphiques et des tracés. On le verra plus en détail dans le prochain chapitre.

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.DataFrame(data={"name": ["Alice", "Bob", "Charlie"], "age": [30, 25, 35]})
df['Age'].plot(kind='hist', bins=10)
plt.show()
```


Références :
* [Pandas cheat sheet](https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf)
