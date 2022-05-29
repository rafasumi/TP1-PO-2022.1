import numpy as np

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

# Função auxiliar que gera um tableau a partir do tableau auxiliar
def get_tableau(tableau_aux, c, n, m, basisColumns):
  # Copiando parte de baixo do tableau da auxiliar, exceto as variáveis de folga da auxiliar
  tableau = np.append(tableau_aux[:, :-n-1], tableau_aux[:, -1].reshape((n, 1)), axis=1)

  # Gerando o vetor c para o tableau
  c_tableau = c.reshape((1, m))
  c_tableau = c_tableau * (-1)
  c_tableau = np.append(c_tableau, np.zeros(n + 1))
  c_tableau = np.append(np.zeros(n), c_tableau)

  # Gerando o tableau
  tableau = np.append(c_tableau.reshape((1, n + m + n + 1)), tableau, axis=0)
  
  # Colocando o tableau em forma canônica, caso não esteja após unir o c com o tableau da auxiliar
  notCanonColumns = np.where(np.round(tableau[0, [i[0] + n for i in basisColumns]], 7) != 0)[0]
  if notCanonColumns.size > 0:
    for column in notCanonColumns:
      col, row = basisColumns[column]
      tableau[0] -= tableau[0, col + n] * tableau[row + 1]

  return tableau

# Função que gera o tableau PL auxiliar a partir de A e b
def get_aux_lp(A, b, n):
  A_aux = add_slack_vars(A, b, n)
  A_aux = add_vero(A_aux, n)
  A_aux = verify_neg_b(A_aux)
  aux = np.append(np.zeros((1, A_aux.shape[1])), A_aux, axis=0)

  # Somando as linhas de A na parte de cima do tableau para ficar em forma canônica
  aux[0] = np.sum(A_aux, axis=0) * (-1)

  # Insere a parte direita do tableau auxiliar
  aux_right = np.append(np.zeros((1, n)), np.identity(n), axis=0)
  aux = np.append(np.append(aux[:, :-1], aux_right, axis=1), aux[:, -1].reshape((n + 1, 1)), axis=1)

  return aux

# Encontra uma solução viável a partir de um tableau em estado de ótimo/ilimitada
# Também retorna o índice das colunas da base
def find_solution(tableau, n, m):
  x = []
  # Lista com as colunas da base e a linha onde fica o valor 1
  basisColumns = []

  # Itera pelas colunas de A
  for i in range(n, n + m):
    if tableau[0, i] < 1e-7:
      pivotIndex = np.where(np.round(tableau[:, i], 7) != 0)[0]

      if pivotIndex.size > 1 or np.round(tableau[:, i][pivotIndex[0]], 7) != 1:
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

      x, basisColumns = find_solution(tableau, n, m)

      certificate = tableau[0, :n]
      
      return OPTIMAL, (np.round(optimalVal, 7), np.round(x, 7), np.round(certificate, 7)), tableau, basisColumns

    # Pega a coluna de menor índice em cN. É preciso somar n para corrigir o índice
    k = cNIndex[0] + n
    Ak = tableau[1:, k]

    # Testa se todos os valores da coluna Ak são negativos
    if (Ak <= 1e-7).all():
      # É ilimitada
      x, basisColumns = find_solution(tableau, n, m)
      # Fixa a coluna k como 1 no certificado e define os outros valores como 0
      certificate = [0.0] * (m + n)
      certificate[k - n] = 1.0

      # Corrige os valores no certificado para as colunas da base, de modo que A.dot(certificate) = 0
      for i, j in basisColumns:
        certificate[i] = Ak[j] * (-1)

      return INFINITE, (np.round(x, 7), np.round(certificate[:m], 7)), tableau, basisColumns

    # Pega apenas os valores positivos de Ak
    gt0values = np.where(Ak > 0)[0]

    # Escolhe o valor que minimiza a razão com o valor em b
    minDiv = np.argmin(tableau[1:, -1][gt0values] / Ak[gt0values])
    # É preciso somar 1 para corrigir o índice
    r = gt0values[minDiv] + 1

    tableau[r] /=  tableau[r, k]

    # Eliminação Gaussiana
    # O objetivo é colocar o tableau na forma canônica
    for i in range(0, n + 1):
      if i != r and tableau[i, k] != 0:
        tableau[i] -= tableau[i, k] * tableau[r]

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
  
  # Gera a PL auxiliar e faz o Simplex para verificar se a PL original é viável ou inviável
  aux = get_aux_lp(A, b, n)

  # Aplica o Simplex na PL auxiliar
  result, values, tableau_aux, basisColumns = simplex(aux, n, m)
  if result == OPTIMAL:
    optimalVal, _, certificate = values
    if optimalVal < 0:
      print('inviavel')
      print(*certificate)
      return
  elif result == INFINITE:
    _, certificate = values
    if np.round(tableau_aux[0, -1], 7) < 0:
      print('inviavel')
      print(*certificate)
      return
  
  # Monta o tableau
  tableau = get_tableau(tableau_aux[1:, :], c, n, m, basisColumns)

  # Aplica o Simplex
  result, values, _, _ = simplex(tableau, n, m)

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