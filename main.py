import numpy as np
np.set_printoptions(linewidth=1000)


OPTIMAL = 0
INFINITE = 1

# Verifica se alguma linha da matriz b tem valores negativos. Se tiver, multiplica por -1
def verify_neg_b(tableau):
  negB = np.where(tableau[:, -1] < 0)[0]

  tableau[negB] *= (-1)
  
  return tableau

# Função auxiliar que adiciona variáveis de folga
def add_slack_vars(A, b, n):
  A = np.append(A, np.identity(n), axis=1)
  return np.append(A, b.reshape((n, 1)), axis=1)

# Adiciona a matriz de registro de operações
def add_vero(A, n):
  return np.append(np.identity(n), A, axis=1)

# Função auxiliar que gera um tableau a partir de uma matriz A e uma matriz c
def get_tableau(A, b, c, n, m):
  A_tableau = add_slack_vars(A, b, n)
  A_tableau = add_vero(A_tableau, n)
  # A_tableau = verify_neg_b(A_tableau)

  c_tableau = c.reshape((1, m))
  c_tableau = c_tableau * (-1)
  c_tableau = np.append(c_tableau, np.zeros(n + 1))
  c_tableau = np.append(np.zeros(n), c_tableau)

  return np.append(c_tableau.reshape((1, n + m + n + 1)), A_tableau, axis=0)

# Função que gera o tableau PL auxiliar a partir de A e b
def get_aux_lp(A, b, n):
  A_aux = add_slack_vars(A, b, n)
  A_aux = add_vero(A_aux, n)
  A_aux = verify_neg_b(A_aux)
  aux = np.append(np.zeros((1, A_aux.shape[1])), A_aux, axis=0)

  # Somando as linhas de A na parte de cima do tableau para ficar em forma canônica
  aux[0] = np.sum(A_aux, axis=0) * (-1)

  return aux

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
      
      return OPTIMAL, (np.round(optimalVal, 7), np.round(x, 7), np.round(certificate, 7))

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

      return INFINITE, (np.round(x, 7), np.round(certificate, 7))

    # Pega apenas os valores positivos de Ak
    gt0values = np.where(Ak > 0)[0]

    # Escolhe o valor que minimiza a razão com o valor em b
    minDiv = np.argmin(tableau[1:, -1][gt0values] / Ak[gt0values])
    # É preciso somar 1 para corrigir o índice
    r = gt0values[minDiv] + 1

    tableau[r] = tableau[r] * (1 / tableau[r, k])

    print(tableau)
    print()
    # Eliminação Gaussiana
    # O objetivo é colocar o tableau na forma canônica
    for i in range(0, n + 1):
      if i != r and tableau[i, k] != 0:
        tableau[i] -= tableau[i, k] * tableau[r]
    print(tableau)
    print('----------')

def dual_simplex(tableau, n, m):
  while((tableau[1:, -1] < 0).any()):
    # Linhas com valor negativo em b
    negBIndex = np.where(tableau[1:, -1] < 0)[0]
    # Menor índice da linha negativa em b
    k = negBIndex[0] + 1
    Ak = tableau[k, :]

    # Pega apenas os valores negativos de Ak cujo c associado é positivo
    lt0AkValues = np.where(Ak < 0)[0]
    lt0cValues = np.where(tableau[0, lt0AkValues] > 0)

    validValues = lt0AkValues[lt0cValues]

    minDiv = np.argmin(tableau[0, validValues] / (-1 * Ak[validValues]))
    pivotIndex = validValues[minDiv]

    tableau[k] = tableau[k] * (1 / tableau[k, pivotIndex])

    # Eliminação Gaussiana
    # O objetivo é colocar o tableau na forma canônica
    for i in range(0, n + 1):
      if i != k and tableau[i, pivotIndex] != 0:
        tableau[i] -= tableau[i, pivotIndex] * tableau[k]

  return tableau

def main():
  # n restrições e m variáveis
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

  aux = get_aux_lp(A, b, n)
  print(aux)
  print('---------------SIMPLEX AUXILIAR---------------')
  result, (optimalVal, x, certificate) = simplex(aux, n, m)
  if optimalVal < 0:
    print('inviavel')
    print(*certificate)
    print(certificate.dot(A))
    print(certificate.dot(b))
    return

  # Monta o tableau
  tableau = get_tableau(A, b, c, n, m)
  print(tableau)

  # Caso em que é necessário usar o simplex dual: c todo negativo e algum valor negativo em b
  if (c < 0).all() and (b < 0).any():
    tableau = dual_simplex(tableau, n, m)
  
  print('-------------------SIMPLEX-------------------')
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
    print(A.dot(x))
    print(A.dot(certificate))
    print(c.dot(certificate))

if __name__ == '__main__':
  main()