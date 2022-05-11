import numpy as np

OPTIMAL = 0
INFINITE = 1

# Função auxiliar que adiciona variáveis de folga
def add_slack_vars(A, b, n):
  A = np.append(A, np.identity(n), axis=1)
  return np.append(A, b.reshape((n, 1)), axis=1)

# Função auxiliar que gera um tableau a partir de uma matriz A e uma matriz c
def get_tableau(A, c, n, m):
  c = c * (-1)
  c = np.append(c, np.zeros(n + 1))
  # Parte direita do tableau
  A = np.append(c.reshape((1, m + n + 1)), A, axis=0)

  # Parte esquerda do tableau
  tableau = np.append(np.zeros((1, n)), np.identity(n), axis=0)
  
  return np.append(tableau, A, axis=1)

def find_solution(tableau, n, m):
  x = []
  # Lista com as colunas da base e a linha onde fica o valor 1
  basisColumns = []
  
  # Itera pelas colunas de A
  for i in range(n, n + m):
    if (tableau[0, i] == 0):
      pivotIndex = np.where(tableau[:, i] != 0)[0]
      if pivotIndex.size > 1 or tableau[:, i][pivotIndex[0]] != 1:
        # Coluna tem 0 na linha de c, mas não está na base
        x.append(0.0)
      else:
        # Insere o valor em b na solução para a coluna da base
        x.append(tableau[pivotIndex[0], -1]) 
        basisColumns.append((i - n, pivotIndex[0] - 1))
    else:
      x.append(0.0)

  return x, basisColumns

# Função que aplica o Simplex para um determinado tableau
def simplex(tableau, n, m):
  while True:
    # A parte de C que não está na base
    cNIndex = np.where(tableau[0, n:-1] < 0)[0]
    if cNIndex.size == 0:
      # Possui valor ótimo
      optimalVal = tableau[0, -1]

      x, _ = find_solution(tableau, n, m)

      certificate = tableau[0, :n]
      
      return OPTIMAL, (optimalVal, x, certificate)

    # Pega a coluna de menor índice em cN. É preciso somar n para corrigir o índice
    k = cNIndex[0] + n
    Ak = tableau[1:, k]

    # Testa se todos os valores da coluna Ak são negativos
    if (Ak <= 0).all():
      # É ilimitada
      x, basisColumns = find_solution(tableau, n, m)
      # Fixa a coluna k como 1 no certificado e define os outros valores como 0
      certificate = [0.0] * m
      certificate[k - n] = 1.0

      # Corrige os valores no certificado para as colunas da base, de modo que A.dot(certificate) = 0
      for i, j in basisColumns:
        certificate[i] = Ak[j] * (-1)

      return INFINITE, (x, certificate)

    # Pega apenas os valores não nulos de Ak
    gt0columns = np.where(Ak > 0)[0]
    tableau[1:, -1][gt0columns] / Ak[gt0columns]

    minDiv = np.argmin(tableau[1:, -1][gt0columns] / Ak[gt0columns])
    # É preciso somar 1 para corrigir o índice
    r = gt0columns[minDiv] + 1

    tableau[r] = tableau[r] * (1 / tableau[r, k])

    # Eliminação Gaussiana
    # O objetivo é colocar o tableau na forma canônica
    for i in range(0, n + 1):
      if i != r and tableau[i, k] != 0:
        tableau[i] -= tableau[i, k] * tableau[r]

# Função que gera o tableau PL auxiliar a partir de A e b
def get_aux_lp(A, b, n, m):
  aux = np.append(A, np.identity(n), axis=1)
  aux = np.append(aux, b.reshape((n, 1)), axis=1)
  aux = np.append(np.zeros((1, m + n + 1)), aux, axis=0)

  aux[0, :m] = np.sum(A, axis=0) * (-1)
  aux[0, -1] = np.sum(aux[:, -1], axis=0) * (-1)

  vero = np.append(np.zeros((1, n)), np.identity(n), axis=0)
  aux = np.append(vero, aux, axis=1)

  aux[0, :n] = np.ones((1, n)) * (-1)

  return aux

def main():
  [n, m] = input().split()
  n, m = int(n), int(m)

  c = input().split()
  c = np.asarray(c, dtype=np.float64)

  A = np.zeros((n, m), dtype=np.float64)
  b = []
  for i in range(n):
    a = input().split()
    A[i] = a[:-1]
    b.append(a[-1])

  b = np.asarray(b, dtype=np.float64)
  
  # Fazendo simplex na PL auxiliar
  aux = get_aux_lp(A, b, n, m)
  result, (optimalVal, x, certificate) = simplex(aux, n, m)
  if optimalVal < 0:
    print('inviavel')
    print(*certificate)
    return

  # Adicionando variáveis de folga para ficar em FPI
  A = add_slack_vars(A, b, n)

  # Monta o tableau
  tableau = get_tableau(A, c, n, m)
  
  result, values = simplex(tableau, n, m)

  if result == OPTIMAL:
    optimalVal, x, certificate = values
    print('otima')
    print(optimalVal)
    print(*x)
    print(*certificate)
  elif result == INFINITE:
    x, certificate = values
    print('ilimitada')
    print(*x)
    print(*certificate)

if __name__ == '__main__':
  main()